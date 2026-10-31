"""Cloud execution (protocol Section 10; fleet plan of 2026-09-26 — results/DECISIONS.md §0, §9).

Instance side: `run_instance` executes the jobs listed in results/shard.json, in order, one
single-threaded worker per physical core; a job starts only when its inputs exist locally.
Mac side: `Coordinator` launches and provisions instances, assigns run bundles (a training run
and its noisy-evaluation parts) so instances finish together, hands unstarted jobs from lagging
instances to others (two-phase: never run twice), selects the qubit-sweep configurations,
syncs continuously, relaunches dead instances, watches cost and STOP conditions, writes
results/STATUS, and terminates an instance only after its outputs are verified synced (SHA-256).
Credentials come from LAMBDA_API_KEY, LAMBDA_SSH_KEY_NAME, LAMBDA_SSH_PRIVATE_KEY_PATH; never
written or printed.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
import yaml

from qpar import ROOT
from qpar.train import (RESULTS, config_id, load_config, noise_path, run_id, run_paths,
                        valid_noise_record, valid_training_record, write_json)

CONFIGS = ROOT / "configs"
CANDIDATES = CONFIGS / "sweep_candidates"  # speculative n = 8 amplitude sweep configs (decision §12)
BLOCK_ORDER = ("M1", "B1", "M2", "B2", "S", "N")  # Section 9 blocks; N = noisy evaluation
MAIN_BLOCK = {"mnist": "M1", "fashionmnist": "M2"}
SWEEP_N = (2, 6, 8)  # n = 4 reuses the M1/M2 runs
LOGS = RESULTS / "logs"
FLEET = RESULTS / "fleet"
SHARD = RESULTS / "shard.json"               # instance side: the jobs assigned to this instance
INSTANCE_STATE = RESULTS / "queue_state.json"  # instance side: what this instance has done
FLEET_STATE = FLEET / "fleet_state.json"      # Mac side
EVENTS = LOGS / "fleet_events.log"
API = "https://cloud.lambda.ai/api/v1"
CREDENTIALS = ("LAMBDA_API_KEY", "LAMBDA_SSH_KEY_NAME", "LAMBDA_SSH_PRIVATE_KEY_PATH")
REMOTE = "qpar"
UV = "~/.local/bin/uv"
KNOWN_HOSTS = Path.home() / ".cache" / "qpar" / "known_hosts"  # no spaces: ssh splits this option on whitespace
RESULT_DIRS = ("runs", "checkpoints", "noise", "partial", "logs")  # logs are pulled per instance
SETUP_HOURS = 0.3


# ============================================================================= jobs

@dataclass(frozen=True)
class Job:
    task: str          # "train", or a noisy-evaluation part: "cal", "depol0.001", ...
    block: str         # M1, B1, M2, B2, S; "N" for noisy-evaluation parts
    config_path: str   # relative to code/
    seed: int
    run_id: str
    dataset: str
    train_frac: float
    kind: str
    placement: str
    encoding: str
    topology: str
    n_qubits: int

    @property
    def key(self):
        return f"{self.run_id}:{self.task}"


def job_from(cfg, rel_path, seed, task):
    return Job(task=task, block=cfg["block"] if task == "train" else "N", config_path=rel_path, seed=seed,
               run_id=run_id(cfg, seed), dataset=cfg["dataset"], train_frac=cfg["train_frac"], kind=cfg["kind"],
               placement=cfg.get("placement", ""), encoding=cfg.get("encoding", ""),
               topology=cfg.get("topology", ""), n_qubits=cfg.get("n_qubits", 0))


def jobs_for(config_paths, seeds, parts):
    """A training job per (config, seed), plus every noisy-evaluation part for hybrid runs."""
    jobs = []
    for path in config_paths:
        cfg = load_config(path)
        rel = str(path.relative_to(ROOT))
        for seed in seeds:
            jobs.append(job_from(cfg, rel, seed, "train"))
            if cfg["kind"] == "hybrid":
                jobs += [job_from(cfg, rel, seed, part) for part in parts]
    return jobs


def config_paths(include_sweep=True):
    dirs = ("matrix", "baselines", "sweep") if include_sweep else ("matrix", "baselines")
    return [p for d in dirs for p in sorted((CONFIGS / d).glob("*.yaml"))]


def train_key(job):
    return f"{job.run_id}:train"


def outputs(job):
    """Result files a job produces, relative to results/."""
    if job.task == "train":
        return [f"runs/{job.run_id}.json", f"checkpoints/{job.run_id}.pt"]
    return [f"noise/{job.run_id}.{job.task}.json"]


def job_complete(job, epochs):
    if job.task == "train":
        return valid_training_record(run_paths(job.run_id)[0], epochs)
    return valid_noise_record(noise_path(job.run_id, job.task), job.task)


def phase(job):
    """-1: n = 8 amplitude sweep training (the floor; started speculatively); 0: M tf=1.0 training
    (the sweep selection needs it); 1: other sweep training; 2: other training; 3: noise."""
    if job.task != "train":
        return 3
    if job.block == "S" and job.encoding == "amplitude" and job.n_qubits == 8:
        return -1
    if job.block in ("M1", "M2") and job.train_frac == 1.0:
        return 0
    return 1 if job.block == "S" else 2


def dispatch_key(job, seconds):
    """Dispatch order: phase, then (phase 0) M1 before M2, then longest first."""
    return phase(job), BLOCK_ORDER.index(job.block) if phase(job) == 0 else 0, -seconds, job.key


# ============================================================================= cost model and planning

class CostModel:
    """Projected seconds of a job on one worker, from the smoke rates measured on the cloud
    (results/smoke/archs.json and noise.json). MNIST rates serve both datasets; sweep n and
    noise costs use FullyConnected (the most CNOTs) for every topology."""

    def __init__(self, archs, noise, epochs, smoke_frac=0.2):
        self.rows = {r["config_id"]: r for r in archs["rows"]}
        self.noise = {(c["n"], c["encoding"]): c for c in noise["costs"]}
        self.epochs, self.tf = epochs, smoke_frac

    def _smoke_id(self, job):
        if job.kind == "base_cnn":
            return f"mnist_base_cnn_tf{self.tf}"
        if job.kind == "matched":
            return f"mnist_matched_{job.placement}_{job.encoding}_tf{self.tf}"
        topology = job.topology if job.n_qubits == 4 else "fullyconnected"
        return f"mnist_{job.placement}_{job.encoding}_{topology}_n{job.n_qubits}_tf{self.tf}"

    def __call__(self, job):
        if job.task == "train":
            row = self.rows[self._smoke_id(job)]
            t_train, t_eval = row["t_train_per_sample"], row["t_eval_per_sample"]
            return self.epochs * (round(job.train_frac * 55000) * t_train + 5000 * t_eval) + 10000 * t_eval + 15.0
        c = self.noise[(job.n_qubits, job.encoding)]
        if job.task == "cal":
            return 2000 * c["t_cal_per_image"] + c["t_transpile"] + 10.0
        return 2000 * c["t_depol_per_image"] + 10.0


def candidate_configs():
    """Speculative n = 8 amplitude sweep candidates: every (dataset, placement, topology)."""
    return [{"kind": "hybrid", "block": "S", "dataset": dataset, "placement": placement, "encoding": "amplitude",
             "topology": topology, "n_qubits": 8, "train_frac": 1.0}
            for dataset in MAIN_BLOCK for placement in ("pre", "middle", "post")
            for topology in ("linear", "fullyconnected", "random")]


def write_candidate_configs():
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    paths = []
    for cfg in candidate_configs():
        path = CANDIDATES / f"{config_id(cfg)}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
        paths.append(path)
    return paths


def hypothetical_sweep_jobs(encoding, seeds, parts):
    """The sweep's jobs if every placement's winner used `encoding` with FullyConnected (the
    costliest). With amplitude the n = 8 training runs are the surviving speculative candidates,
    so only their noise parts are added here."""
    jobs = []
    for dataset in MAIN_BLOCK:
        for placement in ("pre", "middle", "post"):
            for n in SWEEP_N:
                cfg = {"kind": "hybrid", "block": "S", "dataset": dataset, "placement": placement,
                       "encoding": encoding, "topology": "fullyconnected", "n_qubits": n, "train_frac": 1.0}
                rel = f"configs/sweep/{config_id(cfg)}.yaml"
                tasks = parts if (encoding == "amplitude" and n == 8) else ("train", *parts)
                for seed in seeds:
                    jobs += [job_from(cfg, rel, seed, task) for task in tasks]
    return jobs


def planning_dependencies(jobs, speculative):
    """Noise part -> its training run; sweep training (not speculative) -> the dataset's M tf=1.0 runs."""
    main_tf1 = {ds: {j.key for j in jobs if j.task == "train" and j.block == blk and j.train_frac == 1.0}
                for ds, blk in MAIN_BLOCK.items()}
    deps = {}
    for j in jobs:
        if j.task != "train":
            deps[j.key] = {train_key(j)}
        elif j.block == "S" and j.key not in speculative:
            deps[j.key] = main_tf1[j.dataset]
        else:
            deps[j.key] = set()
    return deps


def simulate(jobs, cost, deps, workers, cancel=None, trigger=None):
    """List-schedule `jobs` in dispatch order on `workers` identical workers. When every key in
    trigger[d] has finished, the jobs in cancel[d] are cancelled (dropped, or their worker freed).
    Returns (makespan in s, busy profile [(t, busy workers)])."""
    cancel, trigger = cancel or {}, trigger or {}
    pending = sorted(jobs, key=lambda j: dispatch_key(j, cost[j.key]))
    finished, running, now, profile, fired = set(), {}, 0.0, [], set()
    while pending or running:
        i = 0
        while len(running) < workers and i < len(pending):
            job = pending[i]
            if deps[job.key] <= finished:
                running[job.key] = now + cost[job.key]
                pending.pop(i)
            else:
                i += 1
        profile.append((now, len(running)))
        if not running:
            raise RuntimeError("unsatisfiable dependencies")
        key = min(running, key=running.get)
        now = running.pop(key)
        finished.add(key)
        for d, keys in trigger.items():
            if d not in fired and keys <= finished:
                fired.add(d)
                dropped = cancel.get(d, set())
                for k in dropped & set(running):
                    del running[k]
                pending = [j for j in pending if j.key not in dropped]
                finished |= dropped  # nothing waits on a cancelled candidate
    profile.append((now, 0))
    return now, profile


def instance_hours(profile, cores):
    """Instance-hours if instances are consolidated as busy workers drop (idle ones terminate)."""
    return sum((t1 - t0) / 3600 * -(-busy // cores) for (t0, busy), (t1, _) in zip(profile, profile[1:]))


def plan_fleet(cost_model, base, instance_type, cores, max_instances=8, target_ratio=1.5):
    """Cheapest fleet (homogeneous, the step-1 type) whose makespan is within target_ratio x the
    floor (unlimited workers) with the speculative n = 8 amplitude candidates, in the worst case
    (every sweep winner amplitude); the angle case is reported too."""
    parts = noise_parts_of(base)
    seeds = base["seeds"]
    main = jobs_for(config_paths(include_sweep=False), seeds, parts)
    candidates = [job_from(cfg, f"configs/sweep_candidates/{config_id(cfg)}.yaml", seed, "train")
                  for cfg in candidate_configs() for seed in seeds]
    speculative = {j.key for j in candidates}
    trigger = {ds: {j.key for j in main if j.task == "train" and j.block == blk and j.train_frac == 1.0}
               for ds, blk in MAIN_BLOCK.items()}
    spec_instances = -(-len(candidates) // cores)
    scenarios = {}
    for encoding in ("amplitude", "angle"):
        survive = "fullyconnected" if encoding == "amplitude" else None
        cancel = {ds: {j.key for j in candidates if j.dataset == ds and j.topology != survive} for ds in MAIN_BLOCK}
        jobs = main + candidates + hypothetical_sweep_jobs(encoding, seeds, parts)
        cost = {j.key: cost_model(j) for j in jobs}
        deps = planning_dependencies(jobs, speculative)
        floor, _ = simulate(jobs, cost, deps, len(jobs), cancel, trigger)
        fleets = []
        for k in range(spec_instances, max_instances + 1):
            makespan, profile = simulate(jobs, cost, deps, k * cores, cancel, trigger)
            hours = instance_hours(profile, cores) + k * SETUP_HOURS
            fleets.append({"instances": k, "workers": k * cores, "wall_hours": makespan / 3600 + SETUP_HOURS,
                           "instance_hours": hours, "usd": hours * instance_type["usd_per_hour"],
                           "ratio_to_floor": makespan / floor})
        scenarios[encoding] = {"floor_hours": floor / 3600, "fleets": fleets}
    worst = scenarios["amplitude"]["fleets"]
    within = [f for f in worst if f["ratio_to_floor"] <= target_ratio]
    chosen = min(within, key=lambda f: f["usd"]) if within else worst[-1]
    return {"instance_type": instance_type, "cores_per_instance": cores, "target_ratio": target_ratio,
            "speculative_instances": spec_instances, "chosen": chosen, "scenarios": scenarios}


def noise_parts_of(base):
    return ["cal"] + [f"depol{p}" for p in base["noise"]["depolarizing_p"]]


# ============================================================================= sweep selection (9, S)

def select_sweep(dataset, seeds):
    """Decision 6a: per placement, the (encoding, topology) with the highest mean best-validation
    accuracy over seeds at tf = 1.0 in the dataset's main block. Writes configs/sweep/*.yaml."""
    scores = {}
    for path in sorted((CONFIGS / "matrix").glob(f"{dataset}_*_tf1.0.yaml")):
        cfg = load_config(path)
        values = []
        for seed in seeds:
            with open(run_paths(run_id(cfg, seed))[0]) as f:
                values.append(json.load(f)["best_val_acc"])
        scores[(cfg["placement"], cfg["encoding"], cfg["topology"])] = mean(values)
    selection = {}
    for placement in ("pre", "middle", "post"):
        candidates = {k: v for k, v in scores.items() if k[0] == placement}
        best = max(candidates, key=lambda k: candidates[k])  # ties: first in sorted file order
        selection[placement] = {"encoding": best[1], "topology": best[2], "mean_best_val_acc": candidates[best]}
        for n in SWEEP_N:
            cfg = {"kind": "hybrid", "block": "S", "dataset": dataset, "placement": placement,
                   "encoding": best[1], "topology": best[2], "n_qubits": n, "train_frac": 1.0}
            path = CONFIGS / "sweep" / f"{config_id(cfg)}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    write_json(RESULTS / f"sweep_selection_{dataset}.json",
               {"criterion": "mean best-validation accuracy over seeds, tf = 1.0 (decision 6a)",
                "scores": {"/".join(k): v for k, v in scores.items()}, "selection": selection})
    return selection


# ============================================================================= instance runner

YIELD_EXIT = 75  # scripts/run.py: stopped at an epoch boundary for migration


def run_instance(workers, base, poll=5.0):
    """Instance side: run the jobs in results/shard.json (in listed order) on `workers`
    single-threaded subprocesses. A noise part waits for its training run's outputs. Jobs
    withdrawn from the shard are never started; jobs listed under "cancel" are killed and their
    partial state deleted; a run that yields (exit 75) is left for the coordinator to move.
    Complete jobs are skipped (resume); failed jobs are retried once. Never exits."""
    epochs, threads = base["training"]["epochs"], base["training"]["threads"]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    LOGS.mkdir(parents=True, exist_ok=True)
    mtime, version, jobs, cancel = None, -1, [], set()
    done, running, attempts, failed, started, inputs_ready = set(), {}, {}, set(), set(), set()
    yielded, cancelled = set(), set()
    while True:
        if SHARD.exists() and SHARD.stat().st_mtime != mtime:
            mtime = SHARD.stat().st_mtime
            data = json.loads(SHARD.read_text())
            version, jobs, cancel = data["version"], [Job(**j) for j in data["jobs"]], set(data.get("cancel", []))
            yielded -= {j.key for j in jobs}  # reassigned here deliberately
            done |= {j.key for j in jobs if j.key not in done and job_complete(j, epochs)}
        for key in cancel - cancelled:
            if key in running:
                proc = running.pop(key)[0]
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            (RESULTS / "partial" / f"{key.split(':')[0]}.pt").unlink(missing_ok=True)
            cancelled.add(key)
        for key, (proc, job, _) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[key]
            if proc.returncode == YIELD_EXIT:
                yielded.add(key)
            elif proc.returncode == 0 and job_complete(job, epochs):
                done.add(key)
            elif attempts[key] >= 2:
                failed.add(key)
        for job in jobs:
            if len(running) >= workers:
                break
            if job.key in done or job.key in running or job.key in failed or job.key in cancelled \
                    or job.key in yielded:
                continue
            if job.task != "train" and job.run_id not in inputs_ready:
                train = Job(**{**asdict(job), "task": "train", "block": "?"})
                if train_key(job) in done or job_complete(train, epochs):
                    inputs_ready.add(job.run_id)
                else:
                    continue
            attempts[job.key] = attempts.get(job.key, 0) + 1
            started.add(job.key)
            log = open(LOGS / f"{job.run_id}.{job.task}.log", "a")
            cmd = [sys.executable, "scripts/run.py", job.config_path, "--seed", str(job.seed), "--task", job.task]
            running[job.key] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT),
                                job, time.time())
        write_json(INSTANCE_STATE, {"time": time.time(), "shard_version": version, "workers": workers,
                                    "done": sorted(done), "started": sorted(started), "failed": sorted(failed),
                                    "yielded": sorted(yielded), "cancelled": sorted(cancelled),
                                    "running": {k: t for k, (_, _, t) in running.items()}})
        time.sleep(poll)


# ============================================================================= Lambda API (Mac)

def require_credentials():
    missing = [v for v in CREDENTIALS if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"STOP 15.6: environment variables not set: {', '.join(missing)}")


def api(method, path, body=None):
    request = urllib.request.Request(
        API + path, method=method, data=None if body is None else json.dumps(body).encode(),
        # Cloudflare in front of the API rejects urllib's default User-Agent (error 1010)
        headers={"Authorization": f"Bearer {os.environ['LAMBDA_API_KEY']}", "Content-Type": "application/json",
                 "User-Agent": "qpar-experiment/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["data"]


def allowed_type(name):
    """GPU instance types up to and including H100 (B200 excluded)."""
    return name.startswith("gpu_") and "b200" not in name


def available_types():
    """[(name, region, vcpus, usd_per_hour)] with capacity now, best vCPU/$ first."""
    out = []
    for name, entry in api("GET", "/instance-types").items():
        spec, regions = entry["instance_type"], entry["regions_with_capacity_available"]
        if regions and allowed_type(name):
            out.append((name, regions[0]["name"], spec["specs"]["vcpus"], spec["price_cents_per_hour"] / 100))
    return sorted(out, key=lambda t: -t[2] / t[3])


# ============================================================================= remote execution (Mac)

def _ssh_options():
    KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    return ["-i", os.environ["LAMBDA_SSH_PRIVATE_KEY_PATH"], "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20"]


def ssh(ip, command, check=True, timeout=None):
    return subprocess.run(["ssh", *_ssh_options(), f"ubuntu@{ip}", command], check=check,
                          capture_output=True, text=True, timeout=timeout)


def rsync(src, dst, *extra, timeout=1800):
    """rsync over the fleet ssh options. Exit 24 (source files vanished mid-transfer — a partial
    checkpoint deleted by a run that just finished) is not an error."""
    shell = "ssh " + " ".join(f"'{o}'" if " " in o else o for o in _ssh_options())
    result = subprocess.run(["rsync", "-az", "--partial", "-e", shell, *extra, src, dst],
                            capture_output=True, timeout=timeout)
    if result.returncode not in (0, 24):
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def push_code(ip):
    """Code, configs, lockfile and the downloaded datasets; never local results or .venv."""
    rsync(f"{ROOT}/", f"ubuntu@{ip}:{REMOTE}/", "--exclude", ".venv", "--exclude", "/results/",
          "--exclude", "__pycache__", "--exclude", ".pytest_cache", "--exclude", ".fleet")


def push_files(ip, rel_paths):
    """Copy results files (relative to results/) from the Mac to the instance."""
    if not rel_paths:
        return
    listing = FLEET / "push_list.txt"
    listing.write_text("\n".join(rel_paths) + "\n")
    rsync(f"{RESULTS}/", f"ubuntu@{ip}:{REMOTE}/results/", "--files-from", str(listing))


def pull_results(name, ip):
    """Results from one instance into the Mac's results/ (its queue state under fleet/<name>/,
    fetched first so the heartbeat never depends on the directory syncs)."""
    (FLEET / name).mkdir(parents=True, exist_ok=True)
    rsync(f"ubuntu@{ip}:{REMOTE}/results/queue_state.json", str(FLEET / name / "queue_state.json"))
    for sub in RESULT_DIRS:
        local = RESULTS / "logs" / "instances" / name if sub == "logs" else RESULTS / sub
        local.mkdir(parents=True, exist_ok=True)
        rsync(f"ubuntu@{ip}:{REMOTE}/results/{sub}/", f"{local}/", "--exclude", "*.tmp")


def instance_fully_synced(ip):
    """Every run JSON, checkpoint and noise result on the instance exists locally, byte-identical
    (SHA-256): the instance holds nothing the Mac does not have."""
    out = ssh(ip, f"cd {REMOTE}/results && find runs checkpoints noise -type f "
                  "\\( -name '*.json' -o -name '*.pt' \\) -print0 2>/dev/null | xargs -0 -r sha256sum",
              timeout=900).stdout
    for line in filter(None, out.splitlines()):
        digest, rel = line.split(maxsplit=1)
        local = RESULTS / rel
        if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            return False
    return True


def launch_instance(type_name, region, name):
    """POST launch; returns the instance id (raises urllib.error.HTTPError on failure)."""
    body = {"region_name": region, "instance_type_name": type_name,
            "ssh_key_names": [os.environ["LAMBDA_SSH_KEY_NAME"]], "name": name}
    return api("POST", "/instance-operations/launch", body)["instance_ids"][0]


def wait_active(instance_id, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        info = api("GET", f"/instances/{instance_id}")
        if info["status"] == "active" and info.get("ip"):
            return info["ip"]
        if info["status"] in ("terminated", "terminating", "unhealthy", "preempted"):
            raise RuntimeError(f"instance {instance_id} is {info['status']}")
        time.sleep(15)
    raise TimeoutError(f"instance {instance_id} not active after {timeout} s")


def provision(ip):
    """Code + data, uv, the locked environment, the unit tests; returns physical core count."""
    for _ in range(60):  # sshd can lag the "active" status
        if ssh(ip, "true", check=False, timeout=60).returncode == 0:
            break
        time.sleep(10)
    push_code(ip)
    ssh(ip, "command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh", timeout=600)
    ssh(ip, f"cd {REMOTE} && {UV} sync --frozen", timeout=1800)
    ssh(ip, f"cd {REMOTE} && OMP_NUM_THREADS=1 {UV} run pytest -q -x", timeout=1800)
    out = ssh(ip, "lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l", timeout=60).stdout
    return int(out.strip())


def start_runner(ip, workers):
    """(Re)start the instance runner. Stopping old processes is a separate ssh call: a command line
    that also contains the launch text would match its own pkill pattern and die with it."""
    ssh(ip, "pkill -f '[s]cripts/run.py' ; pkill -f '[q]ueue.py run' ; true", timeout=120)
    ssh(ip, f"cd {REMOTE} && mkdir -p results/logs && (nohup setsid {UV} run python scripts/queue.py run "
            f"--workers {workers} > results/logs/runner.log 2>&1 < /dev/null &)", timeout=120)


def terminate(instance_id):
    api("POST", "/instance-operations/terminate", {"instance_ids": [instance_id]})


# ============================================================================= coordinator (Mac)

def event(kind, message):
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {kind} {message}"
    with open(EVENTS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


class Coordinator:
    """Mac-side fleet control. All state lives in results/fleet/fleet_state.json, so a restarted
    coordinator resumes where it stopped. Instance status: planned -> provisioning -> running ->
    terminated; `failed` (launch or provisioning failed, never ran a job) -> abandoned; `dead`
    (lost mid-run; its unfinished jobs go to a replacement).

    assign: job key -> instance (None while in transit). Jobs absent from `assign` are unassigned:
    the surviving candidates' noise parts wait there until their training run is done."""

    def __init__(self, base, cost_model, cap_usd=200.0, stop_usd=150.0, max_hours=48.0, interval=120,
                 rebalance_every=900):
        require_credentials()
        self.base, self.cost_model = base, cost_model
        self.epochs, self.seeds, self.parts = base["training"]["epochs"], base["seeds"], noise_parts_of(base)
        self.cap_usd, self.stop_usd, self.max_hours = cap_usd, stop_usd, max_hours
        self.interval, self.rebalance_every = interval, rebalance_every
        self.lock = threading.Lock()
        self.state = json.loads(FLEET_STATE.read_text()) if FLEET_STATE.exists() else None
        self.jobs, self.cost, self.done = {}, {}, set()
        self.refresh_jobs()

    # ------------------------------------------------------------------ bookkeeping
    def save(self):
        with self.lock:
            write_json(FLEET_STATE, self.state)

    def refresh_jobs(self):
        discarded = set(self.state["discarded"]) if self.state else set()
        candidates = jobs_for(sorted(CANDIDATES.glob("*.yaml")), self.seeds, []) if CANDIDATES.exists() else []
        for job in jobs_for(config_paths(), self.seeds, self.parts) + candidates:
            if job.key not in self.jobs and job.key not in discarded:
                self.jobs[job.key] = job
                self.cost[job.key] = self.cost_model(job)

    def update_done(self):
        for key, job in self.jobs.items():
            if key not in self.done and job_complete(job, self.epochs):
                self.done.add(key)

    def inst(self, name):
        return self.state["instances"][name]

    def names(self, *statuses):
        return [n for n, i in self.state["instances"].items() if i["status"] in statuses]

    def workers(self, name):
        return max(1, self.inst(name).get("workers") or 1)

    def owned(self, name):
        return [k for k, owner in self.state["assign"].items() if owner == name]

    def shard_of(self, name):
        return sorted((self.jobs[k] for k in self.owned(name)), key=lambda j: dispatch_key(j, self.cost[j.key]))

    def spent(self, now=None):
        now = now or time.time()
        return sum(((i.get("terminated_at") or now) - i["launched_at"]) / 3600 * i["usd_per_hour"]
                   for i in self.state["instances"].values() if i.get("launched_at")) + self.state["prior_spend"]

    def raise_once(self, tag, message):
        if tag not in self.state["raised"]:
            self.state["raised"].append(tag)
            event("STOP", message)

    def remote_state(self, name):
        path = FLEET / name / "queue_state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def remaining_seconds(self, name, now):
        """Projected time until `name` has finished every job it owns or is about to receive."""
        running = self.remote_state(name).get("running", {})
        incoming = [k for k, m in {**self.state["transit"], **self.state["migrating"]}.items() if m["to"] == name]
        work = longest = 0.0
        for key in self.owned(name) + incoming:
            if key in self.done or key not in self.jobs:
                continue
            left = max(0.0, self.cost[key] - (now - running[key])) if key in running else self.cost[key]
            work += left
            if key in running:
                longest = max(longest, left)
        return max(longest, work / self.workers(name))

    def free_cores(self, name):
        """Workers not held by this instance's unfinished speculative candidates, counting the
        survivors migrating to it (their cores are reserved before they arrive)."""
        held = sum(1 for k in self.owned(name) if k not in self.done and phase(self.jobs[k]) == -1)
        incoming = sum(1 for m in self.state["migrating"].values() if m["to"] == name)
        return self.workers(name) - held - incoming

    def main_finish(self, name, now):
        """Projected time until `name` finishes its non-candidate work on the cores its candidates
        leave free (inf if work is waiting but no core is free)."""
        running = self.remote_state(name).get("running", {})
        incoming = [k for k, m in {**self.state["transit"], **self.state["migrating"]}.items() if m["to"] == name]
        work = longest = 0.0
        for key in self.owned(name) + incoming:
            if key in self.done or key not in self.jobs or phase(self.jobs[key]) == -1:
                continue
            left = max(0.0, self.cost[key] - (now - running[key])) if key in running else self.cost[key]
            work += left
            if key in running:
                longest = max(longest, left)
        if work == 0:
            return 0.0
        free = self.free_cores(name)
        return float("inf") if free <= 0 else max(longest, work / free)

    def assign_bundles(self, keys, targets):
        """Assign run bundles (a training run and its noise parts) longest first, each to the
        target with the least projected load per worker."""
        now = time.time()
        load = {n: self.main_finish(n, now) for n in targets}
        bundles = {}
        for key in keys:
            bundles.setdefault(self.jobs[key].run_id, []).append(key)
        for members in sorted(bundles.values(), key=lambda m: -sum(self.cost[k] for k in m)):
            target = min(targets, key=lambda n: (self.free_cores(n) <= 0, load[n]))
            for key in members:
                self.state["assign"][key] = target
            load[target] += sum(self.cost[k] for k in members) / max(1, self.free_cores(target))

    # ------------------------------------------------------------------ fleet setup
    def initialize(self, plan, prior_spend):
        """Fresh fleet: the speculative candidates grouped by topology on the first instances
        (so the survivors of a common topology share an instance), every M/B job assigned to the
        others (or to all, if there are none), all instances launched."""
        write_candidate_configs()
        self.state = {"created": time.time(), "instances": {}, "assign": {}, "transit": {}, "migrating": {},
                      "cancel": {}, "discarded": {}, "selection": {}, "launch_failures": 0, "raised": [],
                      "plan": plan, "last_rebalance": time.time(), "prior_spend": prior_spend,
                      "delivered": {}, "migration_planned": False}
        self.refresh_jobs()
        k, cores = plan["chosen"]["instances"], plan["cores_per_instance"]
        names = [f"fleet-{i + 1}" for i in range(k)]
        for name in names:
            self.state["instances"][name] = {"status": "planned", "type": plan["instance_type"]["name"],
                                             "workers": cores, "shard_version": 0}
        candidates = sorted((j for j in self.jobs.values() if j.block == "S"),
                            key=lambda j: (j.topology, j.dataset, j.placement, j.seed))
        spec = names[:plan["speculative_instances"]]
        for i, job in enumerate(candidates):
            self.state["assign"][job.key] = spec[min(i // cores, len(spec) - 1)]
        main = [k for k, j in self.jobs.items() if j.block != "S"]
        self.assign_bundles(main, names[len(spec):] or names)
        self.save()
        event("PHASE", f"fleet of {k} x {plan['instance_type']['name']}: {len(candidates)} speculative n=8 amplitude "
                       f"candidates on {spec}, {len(main)} M/B jobs on the rest")
        for name in names:
            self.launch(name)

    def launch(self, name):
        """Launch `name` (its planned type if available, else the best available) and provision
        it in a thread. A second consecutive creation failure raises STOP 15.6."""
        record = self.inst(name)
        try:
            types = sorted(available_types(), key=lambda t: t[0] != record.get("type"))
            if not types:
                raise RuntimeError("no allowed instance type has capacity")
            type_name, region, vcpus, usd = types[0]
            instance_id = launch_instance(type_name, region, name)
        except Exception as exc:  # recorded and surfaced; handled in handle_failed
            self.state["launch_failures"] += 1
            record.update(status="failed", failure="launch")
            event("ERROR", f"launch {name} failed: {exc!r}")
            if self.state["launch_failures"] >= 2:
                self.raise_once("launch", "15.6: instance creation failed twice in a row")
            self.save()
            return False
        self.state["launch_failures"] = 0
        record.update(id=instance_id, type=type_name, region=region, vcpus=vcpus, usd_per_hour=usd,
                      launched_at=time.time(), status="provisioning", ip=None)
        self.save()
        event("LAUNCH", f"{name} {type_name} {region} ${usd}/h")
        threading.Thread(target=self._provision, args=(name,), daemon=True).start()
        return True

    def _provision(self, name):
        record = self.inst(name)
        try:
            record["ip"] = wait_active(record["id"])
            workers = provision(record["ip"])
            with self.lock:
                record.update(workers=workers, status="running", last_seen=time.time())
            event("READY", f"{name} {workers} workers")
        except Exception as exc:
            event("ERROR", f"provision {name} failed: {exc!r}")
            with self.lock:
                record.update(status="failed", failure="provision")
        self.save()

    def start(self, name):
        """Push inputs (partials of unfinished runs, training outputs of pending noise parts), the
        candidate and sweep configs, and the shard; then start the runner."""
        record = self.inst(name)
        ssh(record["ip"], "mkdir -p " + " ".join(f"{REMOTE}/results/{d}" for d in RESULT_DIRS), timeout=120)
        needed = set()
        for job in self.shard_of(name):
            if job.key in self.done:
                continue
            if job.task == "train" and (RESULTS / "partial" / f"{job.run_id}.pt").exists():
                needed.add(f"partial/{job.run_id}.pt")
            if job.task != "train" and train_key(job) in self.done:
                needed.update(outputs(self.jobs[train_key(job)]))
        push_files(record["ip"], sorted(needed))
        self.push_configs(name)
        self.push_shard(name)
        start_runner(record["ip"], record["workers"])
        record["runner_started"] = time.time()
        self.state["last_rebalance"] = 0.0  # rebalance on the next step (a new instance may be empty)
        self.save()
        event("START", f"{name}: {len(self.shard_of(name))} jobs on {record['workers']} workers")

    def push_configs(self, name):
        ip = self.inst(name)["ip"]
        for sub in ("sweep", "sweep_candidates"):
            (CONFIGS / sub).mkdir(exist_ok=True)
            rsync(f"{CONFIGS / sub}/", f"ubuntu@{ip}:{REMOTE}/configs/{sub}/")

    def push_shard(self, name):
        record = self.inst(name)
        record["shard_version"] = record.get("shard_version", 0) + 1
        payload = {"version": record["shard_version"], "jobs": [asdict(j) for j in self.shard_of(name)],
                   "cancel": self.state["cancel"].get(name, [])}
        local = FLEET / name / "shard.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        write_json(local, payload)
        if record["status"] == "running":
            rsync(str(local), f"ubuntu@{record['ip']}:{REMOTE}/results/shard.json")

    # ------------------------------------------------------------------ failures
    def handle_failed(self):
        """A failed instance never ran a job. Terminate it (nothing to sync), then: after a launch
        failure, retry once with a replacement (a second consecutive failure is STOP 15.6); after
        a provisioning failure (a software problem, surfaced as ERROR) or a second launch failure,
        hand its jobs to the instances that are up."""
        for name in self.names("failed"):
            record = self.inst(name)
            if record.get("id") and not record.get("terminated_at"):
                try:
                    terminate(record["id"])
                except Exception as exc:
                    event("ERROR", f"terminate failed {name}: {exc!r}")
                record["terminated_at"] = time.time()
            record["status"] = "abandoned"
            orphans = [k for k in self.owned(name) if k not in self.done]
            if record.get("failure") == "launch" and self.state["launch_failures"] < 2:
                replacement = f"fleet-{len(self.state['instances']) + 1}"
                self.state["instances"][replacement] = {"status": "planned", "type": record.get("type"),
                                                        "workers": record.get("workers"), "shard_version": 0}
                for key in orphans:
                    self.state["assign"][key] = replacement
                self.launch(replacement)
            elif orphans:
                targets = self.names("running", "provisioning")
                if targets:
                    self.assign_bundles(orphans, targets)
                    for target in targets:
                        if self.inst(target).get("runner_started"):
                            self.push_shard(target)
                    event("REASSIGN", f"{len(orphans)} jobs of {name} -> {targets}")

    def lost(self, name, why):
        """Declare `name` dead; its unfinished jobs go to a replacement (resuming synced partials).
        If the Lambda API is unreachable as well, the Mac is offline (asleep, no network), not the
        instance: nothing is declared (DECISIONS.md §13, 2026-09-27 12:14-17:50)."""
        record = self.inst(name)
        try:
            status = api("GET", f"/instances/{record['id']}")["status"]
        except Exception as exc:
            if time.time() - record.get("offline_logged", 0) > 600:
                record["offline_logged"] = time.time()
                event("ERROR", f"{name} and the Lambda API unreachable ({exc!r}): the Mac seems offline; waiting")
            # an offline Mac is no evidence against the instance: its 30 min start again (else the
            # first failed sync after the network returns would declare it dead, as on 09-28 05:00)
            record["last_seen"] = time.time()
            return
        record.update(status="dead", dead_reason=f"{why}; api status {status}")
        if status in ("terminated", "terminating", "preempted"):
            record["terminated_at"] = time.time()
        event("RELAUNCH", f"{name} dead: {record['dead_reason']}")
        replacement = f"fleet-{len(self.state['instances']) + 1}"
        self.state["instances"][replacement] = {"status": "planned", "type": record.get("type"),
                                                "workers": record.get("workers"), "shard_version": 0}
        for key in self.owned(name):
            if key not in self.done:
                self.state["assign"][key] = replacement
        for table in ("transit", "migrating"):
            for key, move in list(self.state[table].items()):
                if move["from"] == name:
                    self.state["assign"][key] = replacement
                    del self.state[table][key]
        self.state["cancel"].pop(name, None)
        self.save()
        self.launch(replacement)

    # ------------------------------------------------------------------ periodic steps
    def sync(self, now):
        for name in self.names("running"):
            record = self.inst(name)
            try:
                pull_results(name, record["ip"])
                record["last_seen"] = now
            except (subprocess.SubprocessError, OSError) as exc:
                if now - record.get("sync_error_logged", 0) > 600:
                    record["sync_error_logged"] = now
                    event("ERROR", f"sync {name} failed: {exc!r}"[:600])
                if now - record.get("last_seen", now) > 1800:
                    self.lost(name, f"unreachable for 30 min ({exc!r})")

    def check_runner(self, name, now):
        record = self.inst(name)
        synced_recently = now - record.get("last_seen", 0) < 300  # never judge a heartbeat we could not fetch
        if synced_recently and self.remote_state(name).get("time", 0) < now - 600 \
                and now - record.get("runner_started", 0) > 600:
            event("ERROR", f"{name} runner heartbeat stale; restarting it")
            start_runner(record["ip"], record["workers"])
            record["runner_started"] = now

    def handoffs(self):
        """Finish two-phase moves once the source runs a shard without the moved jobs. A job the
        source had already started stays with it; a noise part moves only if its training run is
        done or moved too — otherwise it stays with the run."""
        ready = []
        for key, move in list(self.state["transit"].items()):
            if self.inst(move["to"])["status"] not in ("running", "provisioning", "planned"):
                self.state["assign"][key] = move["from"]
                del self.state["transit"][key]
            elif self.remote_state(move["from"]).get("shard_version", -1) >= move["version"]:
                ready.append(key)
        touched = set()
        for key in sorted(ready, key=lambda k: self.jobs[k].task != "train"):  # training runs first
            move = self.state["transit"].pop(key)
            source, target = move["from"], move["to"]
            job = self.jobs[key]
            if key in self.remote_state(source).get("started", []):
                owner = source
            elif job.task != "train" and train_key(job) not in self.done \
                    and self.state["assign"].get(train_key(job)) != target:
                owner = source
            else:
                owner = target
            self.state["assign"][key] = owner
            touched.add(owner)
        for name in touched:
            if self.inst(name).get("runner_started"):
                self.push_shard(name)

    def deliver_inputs(self):
        """Push training outputs to instances owning noise parts of runs trained elsewhere."""
        for name in self.names("running"):
            if not self.inst(name).get("runner_started"):
                continue
            delivered = set(self.state["delivered"].setdefault(name, []))
            files, runs = [], set()
            for job in self.shard_of(name):
                if job.task == "train" or job.key in self.done or job.run_id in delivered:
                    continue
                if train_key(job) in self.done and self.state["assign"].get(train_key(job)) != name:
                    files += outputs(self.jobs[train_key(job)])
                    runs.add(job.run_id)
            if files:
                push_files(self.inst(name)["ip"], sorted(set(files)))
                self.state["delivered"][name] = sorted(delivered | runs)

    def assign_ready_parts(self):
        """Unassigned noise parts (of the surviving candidates) whose training run is done: assign
        them now, spread over the running instances (inputs are delivered next step)."""
        ready = [k for k, j in self.jobs.items() if k not in self.state["assign"] and k not in self.state["transit"]
                 and j.task != "train" and train_key(j) in self.done]
        targets = [n for n in self.names("running") if self.inst(n).get("runner_started")]
        if ready and targets:
            self.assign_bundles(ready, targets)
            for name in targets:
                self.push_shard(name)
            event("ASSIGN", f"{len(ready)} noise parts of finished sweep runs -> {targets}")

    def rebalance(self, now, max_moves=8):
        """Move unstarted jobs from the instance projected to finish last to the one finishing
        first while the gap exceeds max(30 min, 15 %); up to `max_moves` moves per call. A
        training run moves with all its pending noise parts; a noise part alone moves only if its
        training run is done. Speculative candidates never move (migration consolidates them)."""
        ready = [n for n in self.names("running") if self.inst(n).get("runner_started")]
        if len(ready) < 2:
            return
        for _ in range(max_moves):
            finish = {n: self.main_finish(n, now) for n in ready}
            receivers = [n for n in ready if self.free_cores(n) > 0 and n not in self.slow_instances()]
            if not receivers:
                return
            late, early = max(finish, key=finish.get), min(receivers, key=finish.get)
            gap = finish[late] - finish[early]
            if late == early or gap < max(1800.0, 0.15 * finish[late]):
                return
            started = set(self.remote_state(late).get("started", []))
            pending = [j for j in self.shard_of(late) if j.key not in self.done and j.key not in started
                       and phase(j) != -1]
            parts_of = {}
            for j in pending:
                if j.task != "train":
                    parts_of.setdefault(j.run_id, set()).add(j.key)
            # the work that equalizes both finish times: gap / (1/F_late + 1/F_early) core-seconds
            # (an infinite gap — work waiting with no free core — moves everything movable)
            f_late, f_early = max(1, self.free_cores(late)), self.free_cores(early)
            budget = gap / (1 / f_late + 1 / f_early)
            moved = set()
            for job in sorted(pending, key=lambda j: -self.cost[j.key]):
                if job.key in moved:
                    continue
                if job.task == "train":
                    bundle = {job.key} | parts_of.get(job.run_id, set())
                elif train_key(job) in self.done:
                    bundle = {job.key}
                else:
                    continue  # waits for its training run, which is not moving
                size = sum(self.cost[k] for k in bundle)
                if size <= budget:
                    moved |= bundle
                    budget -= size
            if not moved:
                return
            for key in moved:
                self.state["assign"][key] = None
            self.push_shard(late)
            for key in moved:
                self.state["transit"][key] = {"from": late, "to": early, "version": self.inst(late)["shard_version"]}
            event("REBALANCE", f"{len(moved)} jobs {late} -> {early} (projected gap {gap / 3600:.1f} h)")

    def selections(self):
        """Decision 6a selection per dataset once its M tf=1.0 runs are done; then discard the
        speculative candidates that do not match (cancel + delete partials) and add the sweep's
        other jobs. Noise parts of surviving candidates stay unassigned until their run is done."""
        for dataset, main in MAIN_BLOCK.items():
            if dataset in self.state["selection"]:
                continue
            needed = [k for k, j in self.jobs.items() if j.task == "train" and j.block == main and j.train_frac == 1.0]
            if not needed or not all(k in self.done for k in needed):
                continue
            selection = select_sweep(dataset, self.seeds)
            self.state["selection"][dataset] = selection
            keep = {(p, s["topology"]) for p, s in selection.items() if s["encoding"] == "amplitude"}
            touched = set()
            for key, job in list(self.jobs.items()):
                if job.block == "S" and job.dataset == dataset and job.config_path.startswith("configs/sweep_candidates/") \
                        and (job.placement, job.topology) not in keep:
                    owner = self.state["assign"].pop(key, None)
                    partial = RESULTS / "partial" / f"{job.run_id}.pt"
                    epochs_done = len(torch.load(partial)["log"]) if partial.exists() else 0
                    partial.unlink(missing_ok=True)
                    self.state["discarded"][key] = {"at": time.time(), "owner": owner, "epochs_done": epochs_done}
                    if owner:
                        self.state["cancel"].setdefault(owner, []).append(key)
                        touched.add(owner)
                    del self.jobs[key], self.cost[key]
            before = set(self.jobs)
            self.refresh_jobs()
            new = [k for k in self.jobs if k not in before]
            survivors = {j.run_id for j in self.jobs.values() if phase(j) == -1 and j.dataset == dataset}
            now_assign = [k for k in new if self.jobs[k].run_id not in survivors]
            targets = self.names("running", "provisioning")
            self.assign_bundles(now_assign, targets)
            for name in self.names("running"):
                if self.inst(name).get("runner_started"):
                    self.push_configs(name)
                    self.push_shard(name)
            n_discard = sum(1 for d in self.state["discarded"].values() if d["at"] > time.time() - 60)
            event("SELECT", f"{dataset} " + " ".join(f"{p}={s['encoding']}/{s['topology']}" for p, s in selection.items())
                  + f"; {len(survivors)} candidates kept, {n_discard} discarded, {len(new)} new sweep jobs")

    def plan_migration(self):
        """After both selections: consolidate the surviving candidates onto as few instances as
        possible (the ones already holding most of them) so the others can terminate. Survivors
        elsewhere are withdrawn and asked to yield at their next epoch boundary."""
        if self.state["migration_planned"] or len(self.state["selection"]) < len(MAIN_BLOCK):
            return
        self.state["migration_planned"] = True
        survivors = [k for k, j in self.jobs.items() if phase(j) == -1 and k not in self.done]
        by_owner = {}
        for key in survivors:
            by_owner.setdefault(self.state["assign"].get(key), []).append(key)
        keepers, capacity = [], 0
        for owner in sorted(by_owner, key=lambda o: -len(by_owner[o])):
            if capacity >= len(survivors):
                break
            if owner and self.inst(owner)["status"] == "running":
                keepers.append(owner)
                capacity += self.workers(owner)
        room = {k: self.workers(k) - len(by_owner.get(k, [])) for k in keepers}
        moves = 0
        for owner, keys in by_owner.items():
            if owner in keepers or owner is None:
                continue
            for key in keys:
                target = max(room, key=room.get)
                if room[target] <= 0:
                    break
                room[target] -= 1
                self.state["assign"][key] = None
                self.state["migrating"][key] = {"from": owner, "to": target, "requested": time.time()}
                moves += 1
            self.push_shard(owner)
            for key in keys:
                if key in self.state["migrating"]:
                    self.state["migrating"][key]["version"] = self.inst(owner)["shard_version"]
            markers = " ".join(f"results/yield/{self.jobs[k].run_id}" for k in keys if k in self.state["migrating"])
            if markers:
                ssh(self.inst(owner)["ip"], f"cd {REMOTE} && mkdir -p results/yield && touch {markers}", timeout=120)
        event("MIGRATE", f"{len(survivors)} surviving candidates; keepers {keepers}; {moves} to move at their "
                         f"next epoch boundary")

    def plan_consolidation(self, now, long_hours=1.0):
        """Once no instance has startable work queued (the main phase is over), gather every
        unfinished training run with more than `long_hours` left onto the fewest instances —
        those already holding most of them — so the others can terminate. Runs elsewhere yield
        at their next epoch boundary and resume on a keeper (as in plan_migration)."""
        if not self.state["migration_planned"] or self.state.get("consolidated"):
            return
        for name in self.names("running"):
            started = set(self.remote_state(name).get("started", []))
            if any(k not in self.done and k not in started and (self.jobs[k].task == "train" or train_key(self.jobs[k]) in self.done)
                   for k in self.owned(name)):
                return  # startable work is still queued somewhere
        self.state["consolidated"] = True
        where = {}  # long run -> the instance it will run on (after pending migrations)
        for name in self.names("running"):
            running = self.remote_state(name).get("running", {})
            for key, t in running.items():
                job = self.jobs.get(key)
                if job and job.task == "train" and key not in self.done and self.state["assign"].get(key) == name \
                        and self.cost[key] - (now - t) > long_hours * 3600:
                    where[key] = name
        for key, move in self.state["migrating"].items():
            where[key] = move["to"]
        by_owner = {}
        for key, owner in where.items():
            by_owner.setdefault(owner, []).append(key)
        keepers, capacity = [], 0
        for owner in sorted(by_owner, key=lambda o: -len(by_owner[o])):
            if capacity >= len(where):
                break
            keepers.append(owner)
            capacity += self.workers(owner)
        room = {k: self.workers(k) - len(by_owner.get(k, [])) for k in keepers}
        moves = 0
        for owner, keys in by_owner.items():
            if owner in keepers:
                continue
            leaving = [k for k in keys if k not in self.state["migrating"]]
            targets = set()
            for key in leaving:
                target = max(room, key=room.get)
                if room[target] <= 0:
                    break
                room[target] -= 1
                self.state["assign"][key] = None
                self.state["migrating"][key] = {"from": owner, "to": target, "requested": now}
                moves += 1
                run = self.jobs[key].run_id
                for part in [k for k, o in self.state["assign"].items() if o == owner and k.startswith(run + ":")
                             and self.jobs[k].task != "train" and k not in self.done]:
                    self.state["assign"][part] = target  # cannot start before its run is done: safe to move now
                targets.add(target)
            for target in targets:
                self.push_shard(target)
            if leaving:
                self.push_shard(owner)
                for key in leaving:
                    if key in self.state["migrating"]:
                        self.state["migrating"][key]["version"] = self.inst(owner)["shard_version"]
                markers = " ".join(f"results/yield/{self.jobs[k].run_id}" for k in leaving if k in self.state["migrating"])
                if markers:
                    ssh(self.inst(owner)["ip"], f"cd {REMOTE} && mkdir -p results/yield && touch {markers}", timeout=120)
        event("MIGRATE", f"consolidation: {len(where)} long runs (> {long_hours:g} h left); keepers {keepers}; "
                         f"{moves} to move at their next epoch boundary")

    def slow_instances(self):
        request = self.state.get("evacuate")
        return {request["from"]} if request else set()

    def evacuate(self, now):
        """Move the unfinished training runs off an instance that proved slow (DECISIONS.md §13:
        fleet-3 ran identical work at half the speed of every other instance) to a running
        instance with free workers; requested through state["evacuate"] = {"from": slow, "to":
        target}. As in plan_consolidation, each run yields at its next epoch boundary and resumes
        on the target, and its queued noise parts go along; runs beyond the free workers stay."""
        request = self.state.get("evacuate")
        if not request or request.get("done"):
            return
        source, target = request["from"], request["to"]
        if self.inst(target)["status"] != "running" or not self.inst(target).get("runner_started"):
            return
        room = self.free_cores(target) - len(self.remote_state(target).get("running", {}))
        leaving = sorted((k for k in self.owned(source) if k not in self.done and self.jobs[k].task == "train"
                          and k not in self.state["migrating"]), key=lambda k: -self.cost[k])[:max(0, room)]
        for key in leaving:
            self.state["assign"][key] = None
            self.state["migrating"][key] = {"from": source, "to": target, "requested": now}
            run = self.jobs[key].run_id
            for part in [k for k, o in self.state["assign"].items() if o == source and k.startswith(run + ":")
                         and self.jobs[k].task != "train" and k not in self.done]:
                self.state["assign"][part] = target
        request["done"] = now
        if leaving:
            self.push_shard(target)
            self.push_shard(source)
            for key in leaving:
                self.state["migrating"][key]["version"] = self.inst(source)["shard_version"]
            markers = " ".join(f"results/yield/{self.jobs[k].run_id}" for k in leaving)
            ssh(self.inst(source)["ip"], f"cd {REMOTE} && mkdir -p results/yield && touch {markers}", timeout=120)
        event("MIGRATE", f"evacuating {source} (slow host): {len(leaving)} training runs to {target} at their "
                         f"next epoch boundary")

    def complete_migrations(self):
        """Move each survivor once its source has let go of it: yielded at an epoch boundary (fetch
        its latest partial, push it to the keeper, delete the stale copy on the source) or never
        started there. A target that went away is replaced by the source or another instance."""
        touched = set()
        for key, move in list(self.state["migrating"].items()):
            source, target = move["from"], move["to"]
            job = self.jobs[key]
            rstate = self.remote_state(source)
            if self.inst(target)["status"] != "running":
                alive = [n for n in self.names("running") if self.inst(n).get("runner_started")]
                target = source if source in alive else (alive[0] if alive else None)
                if target is None:
                    continue
                move["to"] = target
            if key in self.done:
                owner = source
            elif target == source:
                owner = source
            elif key in rstate.get("yielded", []) or (rstate.get("shard_version", -1) >= move.get("version", 1e9)
                                                       and key not in rstate.get("running", {})):
                # stopped on the source: yielded, never started, or crashed after its withdrawal (its
                # runner will not restart it) — continue on the target from the latest partial state
                partial = f"partial/{job.run_id}.pt"
                if key in rstate.get("started", []):
                    rsync(f"ubuntu@{self.inst(source)['ip']}:{REMOTE}/results/{partial}", str(RESULTS / partial))
                if (RESULTS / partial).exists():
                    push_files(self.inst(target)["ip"], [partial])
                    ssh(self.inst(source)["ip"], f"rm -f {REMOTE}/results/{partial}", timeout=120)
                owner = target
                event("MIGRATE", f"{job.run_id}: {source} -> {target}"
                      + (f" after epoch {len(torch.load(RESULTS / partial)['log'])}" if (RESULTS / partial).exists() else ""))
            else:
                continue
            self.state["assign"][key] = owner
            touched.add(owner)
            del self.state["migrating"][key]
        for name in touched:
            self.push_shard(name)

    def retry_failed(self, now):
        """A job that failed twice on its instance (whose runner will not retry it) is handed once
        to another instance, with its partial state; a failure there is final (ERROR event)."""
        retried = self.state.setdefault("retried", {})
        for name in self.names("running"):
            for key in self.remote_state(name).get("failed", []):
                if key in self.done or key not in self.jobs:
                    continue
                if key in retried:
                    if retried[key]["to"] == name and not retried[key].get("final"):
                        retried[key]["final"] = True
                        event("ERROR", f"{key} failed again on {name} after its retry; left failed")
                    continue
                if self.state["assign"].get(key) != name:
                    continue
                targets = [n for n in self.names("running") if n != name and self.inst(n).get("runner_started")]
                if not targets:
                    continue
                target = min(targets, key=lambda n: self.remaining_seconds(n, now))
                job = self.jobs[key]
                files = []
                if job.task == "train" and (RESULTS / "partial" / f"{job.run_id}.pt").exists():
                    files.append(f"partial/{job.run_id}.pt")
                if job.task != "train" and train_key(job) in self.done:
                    files += outputs(self.jobs[train_key(job)])
                push_files(self.inst(target)["ip"], files)
                self.state["assign"][key] = target
                retried[key] = {"from": name, "to": target, "at": now}
                self.push_shard(name)
                self.push_shard(target)
                event("RETRY", f"{key} failed twice on {name}; retrying once on {target}")

    def stop_checks(self, now):
        by_config = {}
        for key, job in self.jobs.items():
            if job.task == "train" and job.train_frac == 1.0 and not (job.block == "S" and job.n_qubits == 2):
                by_config.setdefault(job.config_path, []).append(key)  # decision (a): n = 2 sweep points exempt
        for cfg_path, keys in by_config.items():
            if len(keys) == len(self.seeds) and all(k in self.done for k in keys):
                a = mean(json.loads(run_paths(self.jobs[k].run_id)[0].read_text())["A"] for k in keys)
                if a < 0.5:
                    self.raise_once(f"lowA:{cfg_path}", f"15.3: {cfg_path} at tf=1.0 ended with mean A = {a:.3f} < 0.5")
        for name in self.names("dead"):
            record = self.inst(name)
            if not record.get("terminated_at") and now - record.get("last_seen", now) > 7200:
                self.raise_once(f"lost:{name}", f"{name} unreachable for 2 h and possibly still billed; "
                                                "its jobs were relaunched elsewhere")

    def projection(self, now):
        """(projected total cost, projected total wall-clock hours). Until both sweep selections
        exist, the plan's projection (which models the speculative candidates), or the spend
        so far; afterwards a simulation of the remaining work on the live fleet, with idle
        instances consolidated and terminated as in the plan. Recomputed at most every 10 min."""
        cached = self.state.get("projection")
        if cached and now - cached["at"] < 600:
            return cached["usd"], cached["hours"]
        plan = self.state["plan"]
        cores, price = plan["cores_per_instance"], plan["instance_type"]["usd_per_hour"]
        if len(self.state["selection"]) < len(MAIN_BLOCK):
            usd = max(self.state["prior_spend"] + plan["chosen"]["usd"], self.spent(now))
            hours = max(plan["chosen"]["wall_hours"], (now - self.state["created"]) / 3600)
        else:
            running = {}
            for name in self.names("running"):
                running.update(self.remote_state(name).get("running", {}))
            remaining = [j for k, j in self.jobs.items() if k not in self.done]
            cost = {j.key: max(60.0, self.cost[j.key] - (now - running[j.key])) if j.key in running
                    else self.cost[j.key] for j in remaining}
            deps = {j.key: ({train_key(j)} if j.task != "train" and train_key(j) not in self.done else set())
                    for j in remaining}
            workers = sum(self.workers(n) for n in self.names("running", "provisioning", "planned")) or cores
            makespan, profile = simulate(remaining, cost, deps, workers) if remaining else (0.0, [(0.0, 0)])
            usd = self.spent(now) + instance_hours(profile, cores) * price
            hours = (now - self.state["created"] + makespan) / 3600
        self.state["projection"] = {"at": now, "usd": usd, "hours": hours}
        return usd, hours

    def terminate_finished(self):
        """Terminate an instance once it owns no unfinished job, nothing is in transit or migrating
        to it, both selections exist, and no other instance holds work it could take."""
        if len(self.state["selection"]) < len(MAIN_BLOCK):
            return
        for name in self.names("running"):
            if not self.inst(name).get("runner_started"):
                continue
            if self.inst(name).get("endgame") and not self.remote_state(name).get("done"):
                continue  # launched for the survivors' noise parts: keep it until it has done some
            if any(k not in self.done for k in self.owned(name)):
                continue
            moves = list(self.state["transit"].values()) + list(self.state["migrating"].values())
            if any(m["to"] == name or m["from"] == name for m in moves):
                continue
            if name not in self.slow_instances() and self._work_elsewhere(name):
                continue
            self.verify_and_terminate(name)

    def _work_elsewhere(self, name):
        for other in self.names("running"):
            if other == name:
                continue
            started = set(self.remote_state(other).get("started", []))
            if any(j.key not in self.done and j.key not in started and phase(j) != -1
                   and (j.task == "train" or train_key(j) in self.done) for j in self.shard_of(other)):
                return True
        return False

    def verify_and_terminate(self, name):
        record = self.inst(name)
        try:
            pull_results(name, record["ip"])
            synced = instance_fully_synced(record["ip"])
        except (subprocess.SubprocessError, OSError) as exc:
            event("ERROR", f"{name} final sync failed: {exc!r}")
            return False
        if not synced:
            event("ERROR", f"{name}: checksum mismatch after final sync; not terminating")
            return False
        terminate(record["id"])
        record.update(status="terminated", terminated_at=time.time())
        self.save()
        cost = (record["terminated_at"] - record["launched_at"]) / 3600 * record["usd_per_hour"]
        event("TERMINATE", f"{name} verified synced; instance cost ${cost:.2f}")
        return True

    def status_line(self, now):
        train = [k for k, j in self.jobs.items() if j.task == "train"]
        noise = [k for k, j in self.jobs.items() if j.task != "train"]
        usd, hours = self.projection(now)
        running = self.names("running")
        failed = len({k for n in running for k in self.remote_state(n).get("failed", [])
                      if k not in self.done and (k not in self.state.get("retried", {})
                                                 or self.state["retried"][k].get("final"))})
        sel = " ".join(f"{d}{'+' if d in self.state['selection'] else '-'}" for d in MAIN_BLOCK)
        line = (f"train {sum(k in self.done for k in train)}/{len(train)} noise {sum(k in self.done for k in noise)}/"
                f"{len(noise)} | sweep {sel} | {len(running)} inst {sum(self.workers(n) for n in running)} cores | "
                f"{(now - self.state['created']) / 3600:.1f} h of ~{hours:.0f} h | ${self.spent(now):.2f} of ~${usd:.0f}"
                f" | failed {failed} | {time.strftime('%m-%d %H:%M')}")
        (RESULTS / "STATUS").write_text(line + "\n")
        return line

    # ------------------------------------------------------------------ main loop
    def launch_planned(self):
        """Launch instances added to the state as planned but not launched (by hand)."""
        for name in self.names("planned"):
            self.launch(name)

    def step(self):
        now = time.time()
        self.launch_planned()
        self.sync(now)
        self.update_done()
        self.handle_failed()
        for name in self.names("running"):
            if not self.inst(name).get("runner_started"):
                self.start(name)
            else:
                self.check_runner(name, now)
        self.handoffs()
        self.selections()
        self.plan_migration()
        self.plan_consolidation(now)
        self.evacuate(now)
        self.complete_migrations()
        self.assign_ready_parts()
        self.retry_failed(now)
        self.deliver_inputs()
        self.maybe_scale_up(now)
        self.maybe_endgame(now)
        idle = any(self.inst(n).get("runner_started") and self.main_finish(n, now) == 0
                   and len(self.remote_state(n).get("running", {})) < self.free_cores(n)
                   for n in self.names("running"))  # idle cores and nothing queued there
        if idle or now - self.state["last_rebalance"] > self.rebalance_every:
            self.rebalance(now)
            self.state["last_rebalance"] = now
        self.stop_checks(now)
        usd, hours = self.projection(now)
        if usd > self.stop_usd:
            self.raise_once("cost", f"15.2: projected cost ${usd:.0f} > ${self.stop_usd:.0f} (hard cap ${self.cap_usd:.0f})")
        if hours > self.max_hours:
            self.raise_once("hours", f"15.2: projected wall-clock {hours:.1f} h > {self.max_hours:.0f} h")
        if self.spent(now) >= self.cap_usd - 5:
            self.raise_once("cap", f"hard cap: ${self.spent(now):.2f} spent; terminating after verified sync")
            for name in self.names("running"):
                self.verify_and_terminate(name)
        self.terminate_finished()
        self.save()
        return self.status_line(now)

    def maybe_scale_up(self, now, max_instances=8, target_ratio=1.5):
        """Once both sweep selections exist: if the remaining work's projected makespan on the
        current fleet exceeds target_ratio x its floor, add instances of the planned type (up to
        max_instances live) when the projected total stays within the stop threshold."""
        if self.state.get("scale_checked") or len(self.state["selection"]) < len(MAIN_BLOCK):
            return
        self.state["scale_checked"] = True
        live = self.names("running", "provisioning", "planned")
        remaining = [j for k, j in self.jobs.items() if k not in self.done]
        deps = {j.key: ({train_key(j)} if j.task != "train" and train_key(j) not in self.done else set())
                for j in remaining}
        cost = {j.key: self.cost[j.key] for j in remaining}
        floor, _ = simulate(remaining, cost, deps, len(remaining) or 1)
        cores = self.state["plan"]["cores_per_instance"]
        price = self.state["plan"]["instance_type"]["usd_per_hour"]
        workers = sum(self.workers(n) for n in live)
        for extra in range(0, max_instances - len(live) + 1):
            makespan, profile = simulate(remaining, cost, deps, workers + extra * cores)
            if makespan <= target_ratio * floor:
                break
        usd = self.spent(now) + instance_hours(profile, cores) * price
        event("PHASE", f"after selection: remaining floor {floor / 3600:.1f} h; {extra} more instance(s) -> "
                       f"{makespan / 3600:.1f} h, projected total ${usd:.0f}")
        if extra and usd <= self.stop_usd:
            for _ in range(extra):
                name = f"fleet-{len(self.state['instances']) + 1}"
                self.state["instances"][name] = {"status": "planned", "type": self.state["plan"]["instance_type"]["name"],
                                                 "workers": cores, "shard_version": 0}
                self.launch(name)

    def maybe_endgame(self, now, lead_hours=1.0, finish_within_hours=1.0, max_instances=8):
        """When the first surviving candidate is within `lead_hours` of finishing (estimated from
        its own epoch times), launch enough instances that the survivors' noise parts (ready only
        as each finishes; ~0.8 h each at n = 8) can run in about `finish_within_hours` instead of
        queueing on the keeper — if the projected total stays within the stop threshold."""
        if self.state.get("endgame_checked") or not self.state.get("consolidated"):
            return
        survivors = {j.run_id for k, j in self.jobs.items() if phase(j) == -1 and k not in self.done}
        remaining = []
        for rid in survivors:
            path = RESULTS / "partial" / f"{rid}.pt"
            log = torch.load(path)["log"] if path.exists() else []
            if log:  # the latest epoch: the speed of the instance the run is on now
                remaining.append((self.epochs - len(log)) * log[-1]["seconds"])
        if not survivors or not remaining or min(remaining) > lead_hours * 3600:
            return
        self.state["endgame_checked"] = True
        parts = [k for k, j in self.jobs.items() if j.task != "train" and j.run_id in survivors and k not in self.done]
        cores, price = self.state["plan"]["cores_per_instance"], self.state["plan"]["instance_type"]["usd_per_hour"]
        live = self.names("running", "provisioning", "planned")
        needed = -(-sum(self.cost[k] for k in parts) // (finish_within_hours * 3600 * cores))
        extra = int(max(0, min(max_instances - len(live), needed - len(live))))
        usd = self.spent(now) + (len(live) + extra) * price * (max(remaining) / 3600 + finish_within_hours)
        event("PHASE", f"end game: first survivor done in {min(remaining) / 3600:.1f} h, last in {max(remaining) / 3600:.1f} h; "
                       f"{len(parts)} noise parts; {extra} extra instance(s), projected total ${usd:.0f}")
        if extra and usd <= self.stop_usd:
            for _ in range(extra):
                name = f"fleet-{len(self.state['instances']) + 1}"
                self.state["instances"][name] = {"status": "planned", "type": self.state["plan"]["instance_type"]["name"],
                                                 "workers": cores, "shard_version": 0, "endgame": True}
                self.launch(name)

    def finished(self):
        return (len(self.state["selection"]) == len(MAIN_BLOCK) and all(k in self.done for k in self.jobs)
                and not self.names("running", "provisioning", "planned"))

    def run(self):
        event("PHASE", "coordinator loop started")
        for name in self.names("provisioning"):  # a restarted coordinator resumes provisioning
            threading.Thread(target=self._provision, args=(name,), daemon=True).start()
        while True:
            try:
                print(self.step(), flush=True)
            except Exception:
                event("ERROR", "coordinator step failed: " + traceback.format_exc().replace("\n", " | "))
            if self.finished():
                merge_noise_results(self.jobs.values(), self.base)
                event("DONE", f"all {len(self.jobs)} jobs complete, all instances terminated; total cost "
                              f"${self.spent():.2f}; wall-clock {(time.time() - self.state['created']) / 3600:.1f} h")
                return
            time.sleep(self.interval)


# ============================================================================= consolidation

def merge_noise_results(jobs, base):
    """Fold results/noise/<run>.<part>.json into results/runs/<run>.json (the 12.1 layout).
    Classical baselines get null noise fields (4.6: noise conditions do not apply)."""
    epochs = base["training"]["epochs"]
    for job in jobs:
        if job.task != "train" or not job_complete(job, epochs):
            continue
        path = run_paths(job.run_id)[0]
        record = json.loads(path.read_text())
        if job.kind != "hybrid":
            record.update(acc_noise_calibrated=None, delta_cal=None, acc_depol=None, twoq_transpiled=None,
                          depth_transpiled=None)
        else:
            cal = json.loads(noise_path(job.run_id, "cal").read_text())
            record.update(acc_noise_calibrated=cal["acc_noise_calibrated"], twoq_transpiled=cal["twoq_transpiled"],
                          depth_transpiled=cal["depth_transpiled"], noise_backend=cal["noise_backend"],
                          shots=cal["shots"], delta_cal=record["A_subset_noiseless"] - cal["acc_noise_calibrated"],
                          acc_depol={str(p): json.loads(noise_path(job.run_id, f"depol{p}").read_text())["acc_depol"]
                                     for p in base["noise"]["depolarizing_p"]})
        write_json(path, record)
