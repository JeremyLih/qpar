"""Inference latency D (protocol 7), measured on Jeremy's Mac — never in the cloud.

Seed-0 checkpoint of every configuration (M1, M2, S, B1, B2); batch size 1; lightning.qubit
analytic; 20 warm-up samples, then the median over 200 samples, in milliseconds. The samples are
the first 220 images of the fixed 2,000-image noisy-evaluation subset. All configurations are
measured in one process, in one uninterrupted session, with the training thread settings.

    OMP_NUM_THREADS=1 uv run python -m qpar.latency     # writes results/latency.csv + latency_env.json
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qpar.data import DatasetBundle  # noqa: E402
from qpar.models import build_model  # noqa: E402
from qpar.train import RESULTS, config_id, environment, load_base, load_config, run_id, run_paths, write_json  # noqa: E402

CONFIG_DIRS = ("matrix", "baselines", "sweep")


@torch.no_grad()
def median_latency_ms(model, x, warmup, samples):
    model.eval()
    for i in range(warmup):
        model(x[i:i + 1])
    times = []
    for i in range(warmup, warmup + samples):
        t0 = time.perf_counter()
        model(x[i:i + 1])
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def main():
    base = load_base()
    torch.set_num_threads(base["training"]["threads"])
    warmup, samples = base["latency"]["warmup"], base["latency"]["samples"]
    cfgs = [load_config(p) for d in CONFIG_DIRS for p in sorted((RESULTS.parent / "configs" / d).glob("*.yaml"))]
    missing = [run_id(c, 0) for c in cfgs if not run_paths(run_id(c, 0))[1].exists()]
    if missing:
        raise SystemExit(f"{len(missing)} seed-0 checkpoints missing, e.g. {missing[:3]}; sync first")
    bundles = {}
    rows = []
    load_before, started = os.getloadavg(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for cfg in cfgs:
        if cfg["dataset"] not in bundles:
            bundles[cfg["dataset"]] = DatasetBundle(cfg["dataset"]).noisy_subset()[0][:warmup + samples]
        model = build_model(cfg)
        model.load_state_dict(torch.load(run_paths(run_id(cfg, 0))[1]))
        d = median_latency_ms(model, bundles[cfg["dataset"]], warmup, samples)
        rows.append({"config_id": config_id(cfg), "D_ms": d})
        print(f"{config_id(cfg):55s} D = {d:8.3f} ms", flush=True)
    with open(RESULTS / "latency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config_id", "D_ms"])
        writer.writeheader()
        writer.writerows(rows)
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    write_json(RESULTS / "latency_env.json", {**environment(), "cpu": cpu, "warmup": warmup, "samples": samples,
                                              "started_at": started,
                                              "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                              "load_average_before": load_before, "load_average_after": os.getloadavg()})


if __name__ == "__main__":
    main()
