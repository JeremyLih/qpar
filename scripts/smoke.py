"""Smoke tests (protocol 11.2, decision 2a) and the timing run.

    uv run python scripts/smoke.py archs --workers K   # all 18 M1 architectures; every (sweep n, placement,
                                                       # encoding) with FullyConnected; every baseline.
                                                       # 1 epoch at tf = 0.2, 500 test images: accuracy > 0.20,
                                                       # no NaN. K configs in parallel, longest first.
    uv run python scripts/smoke.py noise               # 6.2 with noise_model=None, shots=1024 (±1.0 pt) and
                                                       # 6.3 at p=0 (exact) on three smoke models; per-image cost
    uv run python scripts/smoke.py timing              # one full run (pre/amplitude/fullyconnected, n=4, tf=1.0)

Reports: results/smoke/{archs,noise,timing}.json; per-config rows in results/smoke/archs/.
The first smoke run (Mac, 2 % of the pool) and the Mac timing run are in results/smoke/mac/.
"""

import os
import sys
from pathlib import Path

sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]  # see queue.py
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from qiskit_aer.noise import NoiseModel  # noqa: E402

from qpar.data import SUBSET_SEED, DatasetBundle, stratified_sample  # noqa: E402
from qpar.evaluate import (AerCore, CircuitTemplate, DepolarizingCore, calibrated_backend,  # noqa: E402
                           quantum_core, replaced_core)
from qpar.models import PLACEMENTS, build_model  # noqa: E402
from qpar.quantum import QuantumCore, encoding_width, feature_map  # noqa: E402
from qpar.train import (RESULTS, config_id, environment, fraction_equal, load_base, predict,  # noqa: E402
                        train_run, write_json)

SMOKE = RESULTS / "smoke"
CKPT = SMOKE / "checkpoints"
ENCODINGS = ("angle", "amplitude")
TOPOLOGIES = ("linear", "fullyconnected", "random")
SMOKE_FRAC, SMOKE_TEST = 0.2, 500
TIMING_CFG = {"kind": "hybrid", "dataset": "mnist", "placement": "pre", "encoding": "amplitude",
              "topology": "fullyconnected", "n_qubits": 4, "train_frac": 1.0}
NOISE_VALIDATION = ["mnist_pre_amplitude_fullyconnected_n4_tf0.2", "mnist_middle_angle_random_n4_tf0.2",
                    "mnist_post_amplitude_linear_n4_tf0.2"]


def hybrid(placement, encoding, topology, n, train_frac=SMOKE_FRAC, dataset="mnist"):
    return {"kind": "hybrid", "dataset": dataset, "placement": placement, "encoding": encoding,
            "topology": topology, "n_qubits": n, "train_frac": train_frac}


def smoke_configs():
    cfgs = [hybrid(p, e, t, 4) for p in PLACEMENTS for e in ENCODINGS for t in TOPOLOGIES]
    cfgs += [hybrid(p, e, "fullyconnected", n) for n in (2, 6, 8) for p in PLACEMENTS for e in ENCODINGS]
    cfgs.append({"kind": "base_cnn", "dataset": "mnist", "train_frac": SMOKE_FRAC})
    cfgs += [{"kind": "matched", "dataset": "mnist", "placement": p, "encoding": e, "n_qubits": 4,
              "train_frac": SMOKE_FRAC} for p in PLACEMENTS for e in ENCODINGS]
    return cfgs


# ----------------------------------------------------------------------------- archs

def cmd_archs_one(base, index):
    cfg = smoke_configs()[index]
    cid = config_id(cfg)
    bundle = DatasetBundle("mnist")
    idx = torch.from_numpy(stratified_sample(bundle.y_test.numpy(), SMOKE_TEST, np.random.default_rng(SUBSET_SEED)))
    try:
        record, state = train_run(cfg, 0, bundle, base, epochs=1,
                                  test_x=bundle.x_test[idx], test_y=bundle.y_test[idx])
    except Exception as exc:  # recorded as a failure; the report must list every config
        write_json(SMOKE / "archs" / f"{cid}.json", {"config_id": cid, "config": cfg, "pass": False, "error": repr(exc)})
        raise
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(state, CKPT / f"{cid}.pt")
    ep = record["epochs"][0]
    row = {"config_id": cid, "config": cfg, "A": record["A"], "pass": record["A"] > 0.20,
           "train_loss": ep["train_loss"], "train_size": record["train_size"],
           "t_train_per_sample": ep["train_seconds"] / record["train_size"],
           "t_eval_per_sample": (ep["seconds"] - ep["train_seconds"]) / 5000,
           **{k: record.get(k) for k in ("params_total", "params_quantum", "G", "G_variational",
                                         "matched_map", "params_matched_map", "simulator")},
           "environment": record["environment"]}
    write_json(SMOKE / "archs" / f"{cid}.json", row)


def cmd_archs(base, workers):
    cfgs = smoke_configs()
    # longest first: larger n, amplitude before angle, hybrids before classical baselines
    order = sorted(range(len(cfgs)), key=lambda i: (-cfgs[i].get("n_qubits", 0) * (cfgs[i]["kind"] == "hybrid"),
                                                    cfgs[i].get("encoding") != "amplitude", i))
    env = {**os.environ, "OMP_NUM_THREADS": str(base["training"]["threads"])}
    (SMOKE / "logs").mkdir(parents=True, exist_ok=True)
    running, t0 = {}, time.perf_counter()
    while order or running:
        while order and len(running) < workers:
            i = order.pop(0)
            log = open(SMOKE / "logs" / f"{config_id(cfgs[i])}.log", "w")
            running[i] = subprocess.Popen([sys.executable, __file__, "archs-one", str(i)], env=env,
                                          stdout=log, stderr=subprocess.STDOUT)
        for i in [i for i, proc in running.items() if proc.poll() is not None]:
            del running[i]
        time.sleep(1)
    rows = []
    for cfg in cfgs:
        path = SMOKE / "archs" / f"{config_id(cfg)}.json"
        row = json.loads(path.read_text()) if path.exists() else {"config_id": config_id(cfg), "config": cfg,
                                                                   "pass": False, "error": "no result"}
        rows.append(row)
        print(f"{'ok  ' if row['pass'] else 'FAIL'} {row['config_id']:52s} A={row.get('A', float('nan')):.3f} "
              f"train {1e3 * row.get('t_train_per_sample', float('nan')):6.2f} ms/sample", flush=True)
    write_json(SMOKE / "archs.json", {"rows": rows, "all_pass": all(r["pass"] for r in rows), "workers": workers,
                                      "wall_seconds": time.perf_counter() - t0, "environment": environment()})
    print(f"{sum(r['pass'] for r in rows)}/{len(rows)} passed")


# ----------------------------------------------------------------------------- timing

def cmd_timing(base):
    t0 = time.perf_counter()
    record, state = train_run(TIMING_CFG, 0, DatasetBundle("mnist"), base)
    record["wall_seconds_total"] = time.perf_counter() - t0
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(state, CKPT / f"{config_id(TIMING_CFG)}.pt")
    write_json(SMOKE / "timing.json", record)
    print(f"{record['run_id']}: {record['wall_seconds_total'] / 60:.1f} min total, A={record['A']:.4f}")


# ----------------------------------------------------------------------------- noise

def cmd_noise(base):
    nz, threads = base["noise"], base["training"]["threads"]
    x, y = DatasetBundle("mnist").noisy_subset()
    backend, info = calibrated_backend(nz["backend"])
    rows = {r["config_id"]: r for r in json.loads((SMOKE / "archs.json").read_text())["rows"]}
    gates = []
    for cid in NOISE_VALIDATION:
        model = build_model(rows[cid]["config"])
        model.load_state_dict(torch.load(CKPT / f"{cid}.pt"))
        qcore = quantum_core(model)
        analytic = predict(model, x)
        tpl = CircuitTemplate(qcore.n, qcore.encoding, qcore.topology, backend,
                              nz["optimization_level"], nz["seed_transpiler"])
        with replaced_core(model, AerCore(qcore, tpl, None, nz["shots"], nz["seed_simulator"], threads)):
            shots = predict(model, x)
        with replaced_core(model, DepolarizingCore(qcore, 0.0, nz["two_qubit_factor"], threads)):
            p0 = predict(model, x)
        acc = {k: fraction_equal(v, y) for k, v in [("analytic", analytic), ("shots", shots), ("p0", p0)]}
        row = {"config_id": cid, "acc_analytic": acc["analytic"], "acc_shots_no_noise": acc["shots"],
               "diff_pp": 100 * (acc["shots"] - acc["analytic"]), "acc_depol_p0": acc["p0"],
               "shots_gate_pass": abs(acc["shots"] - acc["analytic"]) <= 0.010 + 1e-12,
               "p0_exact": acc["p0"] == acc["analytic"],
               "agreement_shots": fraction_equal(shots, analytic), "agreement_p0": fraction_equal(p0, analytic),
               "twoq_transpiled": tpl.twoq, "depth_transpiled": tpl.depth}
        gates.append(row)
        print(json.dumps(row), flush=True)

    noise_model = NoiseModel.from_backend(backend)
    costs = []
    for n in (2, 4, 6, 8):
        for encoding in ENCODINGS:
            torch.manual_seed(0)
            core = QuantumCore(n, encoding, "fullyconnected")
            batch = 40 if n == 8 else 100
            f = feature_map(encoding, torch.randn(batch, encoding_width(encoding, n)))
            t0 = time.perf_counter()
            tpl = CircuitTemplate(n, encoding, "fullyconnected", backend, nz["optimization_level"], nz["seed_transpiler"])
            t_transpile = time.perf_counter() - t0
            t0 = time.perf_counter()
            AerCore(core, tpl, noise_model, nz["shots"], 0, threads)(f)
            t_cal = (time.perf_counter() - t0) / batch
            t0 = time.perf_counter()
            DepolarizingCore(core, 0.01, nz["two_qubit_factor"], threads)(f)
            t_depol = (time.perf_counter() - t0) / batch
            costs.append({"n": n, "encoding": encoding, "twoq_transpiled": tpl.twoq, "depth_transpiled": tpl.depth,
                          "t_transpile": t_transpile, "t_cal_per_image": t_cal, "t_depol_per_image": t_depol})
            print(json.dumps(costs[-1]), flush=True)
    write_json(SMOKE / "noise.json", {"backend": info, "gates": gates, "costs": costs, "environment": environment(),
                                      "all_pass": all(r["shots_gate_pass"] and r["p0_exact"] for r in gates)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("archs", "archs-one", "noise", "timing"))
    parser.add_argument("index", type=int, nargs="?", help="archs-one: config index")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    base = load_base()
    torch.set_num_threads(base["training"]["threads"])
    if args.cmd == "archs":
        cmd_archs(base, args.workers)
    elif args.cmd == "archs-one":
        cmd_archs_one(base, args.index)
    else:
        {"noise": cmd_noise, "timing": cmd_timing}[args.cmd](base)


if __name__ == "__main__":
    main()
