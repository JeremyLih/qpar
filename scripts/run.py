"""Run one config + seed.

    uv run python scripts/run.py configs/matrix/<id>.yaml --seed 0                  # train + test (6.1)
    uv run python scripts/run.py configs/matrix/<id>.yaml --seed 0 --task cal       # 6.2, inference only
    uv run python scripts/run.py configs/matrix/<id>.yaml --seed 0 --task depol0.01 # 6.3 at p = 0.01

Training resumes from results/partial/<run_id>.pt if present; results/yield/<run_id> makes it stop
(exit 75) after the current epoch's state is saved, so it can resume on another instance. A job whose output exists and
validates is skipped (use --force to redo it).
"""

import os
import sys
from pathlib import Path

sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]  # see queue.py
os.environ.setdefault("OMP_NUM_THREADS", "1")  # before torch/lightning load OpenMP; base.yaml: threads

import argparse  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from qpar.data import DatasetBundle  # noqa: E402
from qpar.evaluate import noise_part, noise_parts  # noqa: E402
from qpar.models import build_model  # noqa: E402
from qpar.train import (Yielded, environment, load_base, load_config, noise_path, partial_path,  # noqa: E402
                        run_id, run_paths, save_run, train_run, valid_noise_record, valid_training_record,
                        write_json, yield_marker)

YIELD_EXIT = 75  # EX_TEMPFAIL: stopped at an epoch boundary for migration (not a failure)


def train(cfg, seed, base, force):
    rid = run_id(cfg, seed)
    json_path, _ = run_paths(rid)
    if not force and valid_training_record(json_path, base["training"]["epochs"]):
        print(f"skip {rid}: complete")
        return
    partial = partial_path(rid)
    try:
        record, state = train_run(cfg, seed, DatasetBundle(cfg["dataset"]), base, partial_path=partial,
                                  should_yield=yield_marker(rid).exists)
    except Yielded as done:
        yield_marker(rid).unlink(missing_ok=True)
        print(f"yielded {rid} after epoch {done.args[0]}; resumes elsewhere from {partial}")
        sys.exit(YIELD_EXIT)
    save_run(record, state)
    partial.unlink(missing_ok=True)
    print(f"done {rid}: A={record['A']:.4f} best_epoch={record['best_epoch']}")


def noise(cfg, seed, base, part, force):
    rid = run_id(cfg, seed)
    json_path, ckpt_path = run_paths(rid)
    out = noise_path(rid, part)
    if not force and valid_noise_record(out, part):
        print(f"skip {rid} {part}: complete")
        return
    if not valid_training_record(json_path, base["training"]["epochs"]):
        raise SystemExit(f"{rid}: no complete training record; train it first")
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt_path))
    x, y = DatasetBundle(cfg["dataset"]).noisy_subset()
    t0 = time.perf_counter()
    result = noise_part(model, x, y, base["noise"], base["training"]["threads"], part)
    write_json(out, {"run_id": rid, **result, "seconds": time.perf_counter() - t0, "environment": environment()})
    print(f"done {rid} {part}: {result}")


def main(argv=None):
    base = load_base()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", choices=["train", *noise_parts(base["noise"])], default="train")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(base["training"]["threads"])
    cfg = load_config(args.config)
    if args.task == "train":
        train(cfg, args.seed, base, args.force)
    elif cfg["kind"] != "hybrid":
        raise SystemExit("noise conditions apply to hybrid models only (6.2/6.3)")
    else:
        noise(cfg, args.seed, base, args.task, args.force)


if __name__ == "__main__":
    main()
