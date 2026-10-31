"""Queue logic (Sections 9-10, fleet plan): jobs, dispatch order, planning, coordinator state."""

import json

import pytest

from qpar import cloud
from qpar.cloud import (Coordinator, CostModel, Job, config_paths, dispatch_key, instance_hours, job_from, jobs_for,
                        phase, planning_dependencies, simulate)

PARTS = ["cal", "depol0.001", "depol0.005", "depol0.01", "depol0.02"]


def main_jobs():
    return jobs_for(config_paths(include_sweep=False), [0, 1, 2, 3, 4], PARTS)


def fake_cost_model():
    probe = CostModel({"rows": []}, {"costs": []}, 15)
    rows = []
    for cid in {probe._smoke_id(j) for j in main_jobs() if j.task == "train"}:
        rows.append({"config_id": cid, "t_train_per_sample": 0.002 if "amplitude" in cid else 0.001,
                     "t_eval_per_sample": 0.0005})
    for n in (2, 6, 8):
        for p in ("pre", "middle", "post"):
            for e in ("angle", "amplitude"):
                rows.append({"config_id": f"mnist_{p}_{e}_fullyconnected_n{n}_tf0.2",
                             "t_train_per_sample": 0.001 * n, "t_eval_per_sample": 0.0005})
    costs = [{"n": n, "encoding": e, "t_cal_per_image": 0.01 * n, "t_depol_per_image": 0.005 * n, "t_transpile": 0.1}
             for n in (2, 4, 6, 8) for e in ("angle", "amplitude")]
    return CostModel({"rows": rows}, {"costs": costs}, 15)


def test_job_counts():
    jobs = main_jobs()
    assert sum(j.task == "train" for j in jobs) == 270 + 105 + 270 + 105
    assert sum(j.task != "train" for j in jobs) == 540 * len(PARTS)
    assert len({j.key for j in jobs}) == len(jobs)


def test_dispatch_order_phases():
    jobs = main_jobs()
    order = sorted(jobs, key=lambda j: dispatch_key(j, 1.0))
    phases = [phase(j) for j in order]
    assert phases == sorted(phases) and phases[0] == 0 and phases[-1] == 3
    first = [j for j in order if phase(j) == 0]
    assert len(first) == 180 and all(j.train_frac == 1.0 for j in first)
    assert [j.block for j in first] == ["M1"] * 90 + ["M2"] * 90


def test_longest_first_within_phase():
    make = lambda n: Job(task="train", block="S", config_path="x", seed=0, run_id=f"r{n}", dataset="mnist",
                         train_frac=1.0, kind="hybrid", placement="pre", encoding="angle", topology="linear", n_qubits=n)
    jobs = sorted([make(2), make(8), make(6)], key=lambda j: dispatch_key(j, 10.0 * j.n_qubits))
    assert [j.n_qubits for j in jobs] == [8, 6, 2]


def test_simulate_respects_dependencies():
    a = Job("train", "M1", "x", 0, "a", "mnist", 1.0, "hybrid", "pre", "angle", "linear", 4)
    b = Job("cal", "N", "x", 0, "a", "mnist", 1.0, "hybrid", "pre", "angle", "linear", 4)
    c = Job("train", "B1", "x", 0, "c", "mnist", 0.2, "base_cnn", "", "", "", 0)
    cost = {a.key: 10.0, b.key: 5.0, c.key: 3.0}
    deps = {a.key: set(), b.key: {a.key}, c.key: set()}
    assert simulate([a, b, c], cost, deps, workers=2)[0] == 15.0
    assert simulate([a, b, c], cost, deps, workers=1)[0] == 18.0


def test_planning_dependencies_for_sweep():
    jobs = main_jobs() + cloud.hypothetical_sweep_jobs("angle", [0], PARTS)
    deps = planning_dependencies(jobs, speculative=set())
    s = next(j for j in jobs if j.block == "S" and j.task == "train" and j.dataset == "fashionmnist")
    assert len(deps[s.key]) == 90 and all(k.startswith("fashionmnist_") for k in deps[s.key])


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "FLEET", tmp_path)
    monkeypatch.setattr(cloud, "FLEET_STATE", tmp_path / "fleet_state.json")
    monkeypatch.setattr(cloud, "EVENTS", tmp_path / "events.log")
    monkeypatch.setattr(cloud, "LOGS", tmp_path)
    for name in ("rsync", "push_files", "start_runner", "terminate"):
        monkeypatch.setattr(cloud, name, lambda *a, **k: None)
    monkeypatch.setattr(cloud, "require_credentials", lambda: None)
    monkeypatch.setattr(cloud, "job_complete", lambda job, epochs: False)
    c = Coordinator({"training": {"epochs": 15}, "seeds": [0, 1, 2, 3, 4],
                     "noise": {"depolarizing_p": [0.001, 0.005, 0.01, 0.02]}}, fake_cost_model())
    c.state = {"created": 0.0, "instances": {}, "assign": {}, "transit": {}, "migrating": {}, "cancel": {},
               "discarded": {}, "selection": {}, "launch_failures": 0, "raised": [], "plan": None,
               "last_rebalance": 0.0, "prior_spend": 0.0, "delivered": {}, "migration_planned": False}
    for name, workers in (("qpar-1", 15), ("qpar-2", 15)):
        c.state["instances"][name] = {"status": "running", "workers": workers, "ip": "x", "id": name,
                                      "shard_version": 0, "runner_started": 1.0, "launched_at": 0.0,
                                      "usd_per_hour": 1.0}
    return c


def write_remote(tmp_path, name, **state):
    (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / name / "queue_state.json").write_text(json.dumps(state))


def test_assign_bundles_keeps_runs_together_and_balances(coordinator):
    coordinator.assign_bundles(list(coordinator.jobs), ["qpar-1", "qpar-2"])
    owner_of_run = {}
    for key, owner in coordinator.state["assign"].items():
        owner_of_run.setdefault(key.split(":")[0], set()).add(owner)
    assert all(len(o) == 1 for o in owner_of_run.values())
    load = {n: sum(coordinator.cost[k] for k in coordinator.owned(n)) for n in ("qpar-1", "qpar-2")}
    assert abs(load["qpar-1"] - load["qpar-2"]) / max(load.values()) < 0.02


def test_handoff_started_job_stays_with_source(coordinator, tmp_path):
    keys = [k for k in coordinator.jobs if k.endswith(":train")][:2]
    for key in keys:
        coordinator.state["assign"][key] = None
        coordinator.state["transit"][key] = {"from": "qpar-1", "to": "qpar-2", "version": 3}
    write_remote(tmp_path, "qpar-1", shard_version=2, started=[keys[0]])
    coordinator.handoffs()
    assert set(coordinator.state["transit"]) == set(keys)  # source has not applied version 3 yet
    write_remote(tmp_path, "qpar-1", shard_version=3, started=[keys[0]])
    coordinator.handoffs()
    assert coordinator.state["assign"][keys[0]] == "qpar-1" and coordinator.state["assign"][keys[1]] == "qpar-2"
    assert not coordinator.state["transit"]


def test_no_termination_before_sweep_selection(coordinator, monkeypatch):
    called = []
    monkeypatch.setattr(coordinator, "verify_and_terminate", lambda name: called.append(name))
    coordinator.terminate_finished()  # owns nothing, but the sweep jobs do not exist yet
    assert called == []
    coordinator.state["selection"] = {"mnist": {}, "fashionmnist": {}}
    coordinator.terminate_finished()
    assert sorted(called) == ["qpar-1", "qpar-2"]


def test_rebalance_moves_unstarted_bundles(coordinator, tmp_path):
    train = [k for k, j in coordinator.jobs.items() if j.task == "train" and j.kind == "hybrid"][:40]
    for key in train:
        rid = coordinator.jobs[key].run_id
        for k in coordinator.jobs:
            if k.startswith(rid + ":"):
                coordinator.state["assign"][k] = "qpar-1"
    write_remote(tmp_path, "qpar-1", shard_version=1, started=[train[0]], running={})
    write_remote(tmp_path, "qpar-2", shard_version=1, started=[], running={})
    coordinator.rebalance(now=10.0)
    moved = set(coordinator.state["transit"])
    assert moved and train[0] not in moved
    for key in moved:  # a moved training run takes all its noise parts along
        if coordinator.jobs[key].task == "train":
            rid = coordinator.jobs[key].run_id
            assert {k for k in coordinator.jobs if k.startswith(rid + ":")} <= moved


def test_handoff_part_stays_with_its_training_run(coordinator, tmp_path):
    train = next(k for k, j in coordinator.jobs.items() if j.task == "train" and j.kind == "hybrid")
    part = train.replace(":train", ":cal")
    for key in (train, part):
        coordinator.state["assign"][key] = None
        coordinator.state["transit"][key] = {"from": "qpar-1", "to": "qpar-2", "version": 2}
    write_remote(tmp_path, "qpar-1", shard_version=2, started=[train])  # training started just in time
    coordinator.handoffs()
    assert coordinator.state["assign"][train] == "qpar-1" and coordinator.state["assign"][part] == "qpar-1"


def candidate(dataset, placement, topology, seed):
    cfg = {"kind": "hybrid", "block": "S", "dataset": dataset, "placement": placement, "encoding": "amplitude",
           "topology": topology, "n_qubits": 8, "train_frac": 1.0}
    return job_from(cfg, f"configs/sweep_candidates/{dataset}_{placement}_{topology}.yaml", seed, "train")


def test_simulate_cancels_at_trigger():
    t = Job("train", "M1", "x", 0, "t", "mnist", 1.0, "hybrid", "pre", "angle", "linear", 4)
    keep, drop = candidate("mnist", "pre", "linear", 0), candidate("mnist", "pre", "random", 0)
    cost = {t.key: 2.0, keep.key: 10.0, drop.key: 10.0}
    deps = {k: set() for k in cost}
    makespan, profile = simulate([t, keep, drop], cost, deps, 3, cancel={"mnist": {drop.key}},
                                 trigger={"mnist": {t.key}})
    assert makespan == 10.0 and profile[0] == (0.0, 3) and (2.0, 1) in profile
    assert instance_hours([(0.0, 3), (3600.0, 1), (7200.0, 0)], cores=2) == 3.0


def test_n8_amplitude_sweep_runs_dispatch_first():
    jobs = main_jobs()[:50] + [candidate("mnist", "post", "linear", 1)]
    assert phase(sorted(jobs, key=lambda j: dispatch_key(j, 1.0))[0]) == -1


def add_candidates(coordinator):
    for ds in ("mnist", "fashionmnist"):
        for pl in ("pre", "middle", "post"):
            for topo, owner in (("linear", "qpar-1"), ("fullyconnected", "qpar-2"), ("random", "qpar-1")):
                for seed in range(5):
                    job = candidate(ds, pl, topo, seed)
                    coordinator.jobs[job.key], coordinator.cost[job.key] = job, 80000.0
                    coordinator.state["assign"][job.key] = owner


def test_selection_discards_unselected_candidates(coordinator, monkeypatch):
    add_candidates(coordinator)
    sel = {"pre": {"encoding": "amplitude", "topology": "fullyconnected"},
           "middle": {"encoding": "angle", "topology": "linear"},
           "post": {"encoding": "amplitude", "topology": "random"}}
    monkeypatch.setattr(cloud, "select_sweep", lambda dataset, seeds: sel)
    for key, job in coordinator.jobs.items():
        if job.block == "M1" and job.task == "train" and job.train_frac == 1.0:
            coordinator.done.add(key)
    coordinator.selections()
    kept = {(j.placement, j.topology) for j in coordinator.jobs.values() if phase(j) == -1 and j.dataset == "mnist"}
    assert kept == {("pre", "fullyconnected"), ("post", "random")}
    assert len([d for d in coordinator.state["discarded"]]) == 5 * 7  # 9 (placement, topology) - 2 kept
    cancelled = coordinator.state["cancel"]["qpar-1"] + coordinator.state["cancel"]["qpar-2"]
    assert len(cancelled) == 35 and all(k.startswith("mnist_") for k in cancelled)
    assert "fashionmnist" not in coordinator.state["selection"]  # its M2 runs are not done


def test_migration_consolidates_survivors(coordinator, tmp_path, monkeypatch):
    add_candidates(coordinator)
    survivors = [k for k, j in coordinator.jobs.items() if phase(j) == -1 and j.placement == "pre"
                 and j.topology in ("linear", "fullyconnected") and j.dataset == "mnist"]
    for key in [k for k in coordinator.jobs if phase(coordinator.jobs[k]) == -1 and k not in survivors]:
        del coordinator.jobs[key]
        coordinator.state["assign"].pop(key)
    coordinator.state["selection"] = {"mnist": {}, "fashionmnist": {}}
    touched = []
    monkeypatch.setattr(cloud, "ssh", lambda ip, cmd, **k: touched.append(cmd))
    coordinator.state["assign"].update({k: "qpar-1" for k in survivors[:7]})
    coordinator.state["assign"].update({k: "qpar-2" for k in survivors[7:]})
    coordinator.plan_migration()
    moving = coordinator.state["migrating"]
    assert len(moving) == 3 and all(m["from"] == "qpar-2" and m["to"] == "qpar-1" for m in moving.values())
    assert touched and "results/yield/" in touched[0]
    first = next(iter(moving))
    write_remote(tmp_path, "qpar-2", shard_version=99, started=list(moving), yielded=[first],
                 running={k: 0.0 for k in moving if k != first})  # the other two are still running there
    monkeypatch.setattr(cloud, "rsync", lambda *a, **k: None)
    coordinator.complete_migrations()
    assert coordinator.state["assign"][first] == "qpar-1" and len(coordinator.state["migrating"]) == 2


def test_unassigned_sweep_parts_do_not_block_termination(coordinator, monkeypatch):
    add_candidates(coordinator)
    for key in [k for k in coordinator.jobs if phase(coordinator.jobs[k]) == -1]:
        coordinator.state["assign"][key] = "qpar-1"
    part = candidate("mnist", "pre", "linear", 0)
    part = Job(**{**part.__dict__, "task": "cal", "block": "N"})
    coordinator.jobs[part.key], coordinator.cost[part.key] = part, 100.0  # unassigned until its run is done
    for key in list(coordinator.state["assign"]):
        if coordinator.state["assign"][key] != "qpar-1":
            coordinator.state["assign"][key] = "qpar-1"
    coordinator.state["selection"] = {"mnist": {}, "fashionmnist": {}}
    called = []
    monkeypatch.setattr(coordinator, "verify_and_terminate", lambda name: called.append(name))
    coordinator.terminate_finished()
    assert called == ["qpar-2"]


def test_failed_job_is_retried_once_elsewhere(coordinator, tmp_path):
    key = next(k for k, j in coordinator.jobs.items() if j.task == "train")
    coordinator.state["assign"][key] = "qpar-1"
    write_remote(tmp_path, "qpar-1", failed=[key])
    write_remote(tmp_path, "qpar-2", failed=[])
    coordinator.retry_failed(now=0.0)
    assert coordinator.state["assign"][key] == "qpar-2" and coordinator.state["retried"][key]["from"] == "qpar-1"
    coordinator.retry_failed(now=0.0)  # not retried twice
    assert coordinator.state["assign"][key] == "qpar-2"
    write_remote(tmp_path, "qpar-2", failed=[key])
    coordinator.retry_failed(now=0.0)
    assert coordinator.state["retried"][key]["final"] is True


def test_rebalance_ignores_candidate_hours_and_full_instances(coordinator, tmp_path):
    coordinator.state["instances"]["qpar-3"] = {**coordinator.state["instances"]["qpar-2"], "id": "qpar-3"}
    cands = [candidate("mnist", "pre", t, s) for t in ("linear", "random", "fullyconnected") for s in range(5)]
    for job in cands:  # qpar-1: every core held by a running candidate
        coordinator.jobs[job.key], coordinator.cost[job.key] = job, 80000.0
        coordinator.state["assign"][job.key] = "qpar-1"
    main = [k for k, j in coordinator.jobs.items() if j.task == "train" and j.block == "M1"][:60]
    for key in main:
        coordinator.state["assign"][key] = "qpar-2"
    write_remote(tmp_path, "qpar-1", shard_version=1, started=[j.key for j in cands],
                 running={j.key: 5.0 for j in cands})
    write_remote(tmp_path, "qpar-2", shard_version=1, started=[], running={})
    write_remote(tmp_path, "qpar-3", shard_version=1, started=[], running={})
    assert coordinator.free_cores("qpar-1") == 0 and coordinator.main_finish("qpar-1", 10.0) == 0.0
    coordinator.rebalance(now=10.0)
    targets = {m["to"] for m in coordinator.state["transit"].values()}
    assert targets == {"qpar-3"} and all(m["from"] == "qpar-2" for m in coordinator.state["transit"].values())


def test_rebalance_sizes_moves_by_receiver_capacity(coordinator, tmp_path):
    cands = [candidate("mnist", "pre", t, s) for t in ("linear", "random", "fullyconnected") for s in range(5)][:14]
    for job in cands:  # qpar-1: 14 of 15 cores held by candidates -> one free core
        coordinator.jobs[job.key], coordinator.cost[job.key] = job, 80000.0
        coordinator.state["assign"][job.key] = "qpar-1"
    main = [k for k, j in coordinator.jobs.items() if j.task == "train" and j.block == "M1"][:90]
    for key in main:
        coordinator.state["assign"][key] = "qpar-2"
    write_remote(tmp_path, "qpar-1", shard_version=1, started=[j.key for j in cands], running={j.key: 5.0 for j in cands})
    write_remote(tmp_path, "qpar-2", shard_version=1, started=[], running={})
    gap = coordinator.main_finish("qpar-2", 10.0)
    coordinator.rebalance(now=10.0, max_moves=1)
    moved = sum(coordinator.cost[k] for k in coordinator.state["transit"])
    assert 0 < moved <= gap / (1 / 15 + 1 / 1) + 1e-6  # about one core's worth, not half of qpar-2's work


def test_migrating_survivors_reserve_cores(coordinator):
    job = candidate("mnist", "post", "random", 0)
    coordinator.jobs[job.key], coordinator.cost[job.key] = job, 80000.0
    coordinator.state["migrating"][job.key] = {"from": "qpar-2", "to": "qpar-1", "version": 3}
    assert coordinator.free_cores("qpar-1") == 14 and coordinator.free_cores("qpar-2") == 15


def test_consolidation_gathers_long_runs_once_nothing_is_startable(coordinator, tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "ssh", lambda *a, **k: None)
    coordinator.state["instances"]["qpar-3"] = {**coordinator.state["instances"]["qpar-2"], "id": "qpar-3"}
    coordinator.state.update(migration_planned=True, selection={"mnist": {}, "fashionmnist": {}})
    runs = [k for k, j in coordinator.jobs.items() if j.task == "train" and j.block == "M1" and j.train_frac == 1.0][:12]
    placement = {k: ("qpar-1" if i < 7 else "qpar-2" if i < 10 else "qpar-3") for i, k in enumerate(runs)}
    for key, owner in placement.items():
        coordinator.state["assign"][key] = owner
        coordinator.cost[key] = 6 * 3600.0
    for name in ("qpar-1", "qpar-2", "qpar-3"):
        mine = [k for k, o in placement.items() if o == name]
        write_remote(tmp_path, name, shard_version=1, started=mine, running={k: 0.0 for k in mine})
    moved_run = runs[8]
    part = moved_run.replace(":train", ":cal")
    coordinator.state["assign"][part] = "qpar-2"
    coordinator.plan_consolidation(now=600.0)
    moving = coordinator.state["migrating"]
    assert len(moving) == 5 and {m["to"] for m in moving.values()} == {"qpar-1"}
    assert coordinator.state["assign"][part] == "qpar-1"  # the noise part follows its run
    assert coordinator.state["consolidated"] is True


def test_migration_completes_for_a_run_that_crashed_after_withdrawal(coordinator, tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "rsync", lambda *a, **k: None)
    monkeypatch.setattr(cloud, "ssh", lambda *a, **k: None)
    job = candidate("fashionmnist", "post", "fullyconnected", 0)
    coordinator.jobs[job.key], coordinator.cost[job.key] = job, 80000.0
    coordinator.state["migrating"][job.key] = {"from": "qpar-1", "to": "qpar-2", "version": 9}
    write_remote(tmp_path, "qpar-1", shard_version=10, started=[job.key], running={}, yielded=[], failed=[])
    coordinator.complete_migrations()
    assert coordinator.state["assign"][job.key] == "qpar-2" and not coordinator.state["migrating"]


def test_endgame_waits_for_the_first_survivor_and_sizes_the_fleet(coordinator, tmp_path, monkeypatch):
    import torch
    launched = []
    monkeypatch.setattr(coordinator, "launch", lambda name: launched.append(name))
    monkeypatch.setattr(cloud, "RESULTS", tmp_path)
    coordinator.state.update(consolidated=True, plan={"cores_per_instance": 15,
                                                      "instance_type": {"name": "gpu_1x_a10", "usd_per_hour": 1.0}})
    job = candidate("mnist", "post", "random", 0)
    coordinator.jobs[job.key], coordinator.cost[job.key] = job, 80000.0
    for part in PARTS:
        pj = Job(**{**job.__dict__, "task": part, "block": "N"})
        coordinator.jobs[pj.key], coordinator.cost[pj.key] = pj, 3000.0 * 15  # 15 core-hours each
    (tmp_path / "partial").mkdir()
    torch.save({"log": [{"seconds": 3600.0}] * 10}, tmp_path / "partial" / f"{job.run_id}.pt")
    coordinator.maybe_endgame(now=0.0)  # 5 epochs x 1 h left: not yet
    assert not launched and not coordinator.state.get("endgame_checked")
    torch.save({"log": [{"seconds": 3600.0}] * 14}, tmp_path / "partial" / f"{job.run_id}.pt")
    coordinator.maybe_endgame(now=0.0)  # 1 h left: 5 parts x 12.5 core-h over 15-core instances in 1 h -> 5 total
    assert launched == ["fleet-3", "fleet-4", "fleet-5"] and coordinator.state["endgame_checked"]


def test_evacuate_moves_training_runs_off_a_slow_instance(coordinator, tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(cloud, "ssh", lambda ip, command, **k: commands.append(command))
    runs = [k for k, j in coordinator.jobs.items() if j.task == "train" and j.block == "M1" and j.train_frac == 1.0][:4]
    for i, key in enumerate(runs):
        coordinator.state["assign"][key] = "qpar-2"
        coordinator.cost[key] = (i + 1) * 3600.0
    part = runs[3].replace(":train", ":cal")
    coordinator.state["assign"][part] = "qpar-2"
    write_remote(tmp_path, "qpar-2", shard_version=1, started=runs, running={k: 0.0 for k in runs})
    write_remote(tmp_path, "qpar-1", shard_version=1, running={f"busy{i}": 0.0 for i in range(12)})
    coordinator.state["evacuate"] = {"from": "qpar-2", "to": "qpar-1"}
    coordinator.evacuate(now=100.0)
    moving = coordinator.state["migrating"]
    assert set(moving) == set(runs[1:])  # 3 free workers on qpar-1: the three longest runs move
    assert all(m["from"] == "qpar-2" and m["to"] == "qpar-1" for m in moving.values())
    assert coordinator.state["assign"][part] == "qpar-1"  # the noise part follows its run
    assert coordinator.state["assign"][runs[0]] == "qpar-2"
    assert coordinator.state["evacuate"]["done"] == 100.0 and "results/yield/" in commands[-1]
    coordinator.evacuate(now=200.0)  # once only
    assert len(coordinator.state["migrating"]) == 3 and coordinator.slow_instances() == {"qpar-2"}


def test_lost_waits_when_the_mac_is_offline(coordinator, monkeypatch):
    """An instance unreachable while the Lambda API is unreachable too means the Mac is offline:
    nothing is declared dead and nothing is relaunched."""
    def offline(*a, **k):
        raise OSError(9, "Bad file descriptor")
    launched = []
    monkeypatch.setattr(cloud, "api", offline)
    monkeypatch.setattr(coordinator, "launch", lambda name: launched.append(name))
    key = next(k for k in coordinator.jobs if k.endswith(":train"))
    coordinator.state["assign"][key] = "qpar-1"
    coordinator.inst("qpar-1")["last_seen"] = 0.0
    coordinator.lost("qpar-1", "unreachable for 30 min")
    assert coordinator.inst("qpar-1")["status"] == "running" and not launched
    assert coordinator.state["assign"][key] == "qpar-1" and len(coordinator.state["instances"]) == 2
    assert coordinator.inst("qpar-1")["last_seen"] > 0  # its 30 min start again once the Mac is back
    monkeypatch.setattr(cloud, "api", lambda *a, **k: {"status": "terminated"})
    coordinator.lost("qpar-1", "unreachable for 30 min")  # the API answers: a real loss
    assert coordinator.inst("qpar-1")["status"] == "dead" and launched == ["fleet-3"]
    assert coordinator.state["assign"][key] == "fleet-3"


def test_launch_planned_launches_instances_added_by_hand(coordinator, monkeypatch):
    launched = []
    monkeypatch.setattr(coordinator, "launch", lambda name: launched.append(name))
    coordinator.state["instances"]["fleet-12"] = {"status": "planned", "workers": 30, "shard_version": 0}
    coordinator.launch_planned()
    assert launched == ["fleet-12"]
