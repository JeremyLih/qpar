"""Training loop (protocol Section 5), configs, run ids and the per-run JSON."""

import copy
import json
import math
import os
import platform
import random
import time
from importlib.metadata import version

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from qpar import ROOT
from qpar.models import build_model, param_counts
from qpar.quantum import QuantumCore, gate_counts

BASE_CONFIG = ROOT / "configs" / "base.yaml"
RESULTS = ROOT / "results"
RUNS_DIR = RESULTS / "runs"
CHECKPOINT_DIR = RESULTS / "checkpoints"
PARTIAL_DIR = RESULTS / "partial"    # per-epoch state of unfinished runs (resume)
NOISE_DIR = RESULTS / "noise"        # one JSON per (run, noisy-evaluation part)
EVAL_BATCH = 500


# ----------------------------------------------------------------------------- configs and ids

def load_base():
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def config_id(cfg):
    """Appendix A naming, without the seed."""
    tf = f"tf{cfg['train_frac']}"
    if cfg["kind"] == "base_cnn":
        return f"{cfg['dataset']}_base_cnn_{tf}"
    if cfg["kind"] == "matched":
        return f"{cfg['dataset']}_matched_{cfg['placement']}_{cfg['encoding']}_{tf}"
    return (f"{cfg['dataset']}_{cfg['placement']}_{cfg['encoding']}_{cfg['topology']}"
            f"_n{cfg['n_qubits']}_{tf}")


def run_id(cfg, seed):
    return f"{config_id(cfg)}_s{seed}"


def set_seed(seed):
    """Seeds random, numpy and torch (torch.manual_seed covers CPU and every accelerator)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment():
    pkgs = ["torch", "pennylane", "pennylane-lightning", "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            **{p: version(p) for p in pkgs}}


# ----------------------------------------------------------------------------- training

@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([model(x[i:i + EVAL_BATCH]).argmax(1) for i in range(0, len(x), EVAL_BATCH)])


def fraction_equal(a, b):
    """Exact k/N as a Python float (a float32 mean would make 100/500 compare > 0.2)."""
    return (a == b).sum().item() / len(a)


def make_training(cfg, seed, bundle, base, train_frac=None):
    """Seeded model, optimizer and loader for one run (shared by train_run and the seeding test)."""
    set_seed(seed)
    tr = base["training"]
    x_train, y_train = bundle.train(cfg["train_frac"] if train_frac is None else train_frac, seed)
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"], betas=tuple(tr["betas"]),
                                 weight_decay=tr["weight_decay"])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=tr["batch_size"],
        shuffle=True, generator=torch.Generator().manual_seed(seed))
    return model, optimizer, loader


class Yielded(Exception):
    """Training stopped at an epoch boundary on request; the partial state is saved (migration)."""


def train_run(cfg, seed, bundle, base, *, epochs=None, train_frac=None, test_x=None, test_y=None,
              partial_path=None, should_yield=None):
    """Train one config + seed; evaluate the best-validation checkpoint on the test set once.

    Returns (record, best_state_dict). `epochs`, `train_frac` and `test_x/test_y` exist only for
    the smoke tests; production runs use the protocol values from the config and base.yaml.
    With `partial_path`, the full training state is saved after every epoch and an existing
    file is resumed from — bit-identically, including the shuffling generator. If `should_yield()`
    is true after an epoch's state is saved, Yielded is raised (the run continues elsewhere).
    """
    tr = base["training"]
    epochs = tr["epochs"] if epochs is None else epochs
    threshold = tr["threshold"][cfg["dataset"]]
    model, optimizer, loader = make_training(cfg, seed, bundle, base, train_frac)
    x_val, y_val = bundle.val()

    log, best_val, best_epoch, best_state, to_threshold, elapsed, resumes = [], -1.0, None, None, None, 0.0, []
    if partial_path is not None and partial_path.exists():
        saved = torch.load(partial_path)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        loader.generator.set_state(saved["generator"])
        log, best_val, best_epoch, best_state, to_threshold, elapsed = (
            saved[k] for k in ("log", "best_val", "best_epoch", "best_state", "to_threshold", "elapsed"))
        resumes = saved["resumes"] + [len(log)]
    t_start = time.perf_counter()
    for epoch in range(len(log) + 1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum, correct, seen, skipped = 0.0, 0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss in epoch {epoch}")
            loss.backward()
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                optimizer.step()
            else:  # an ill-defined gradient (DECISIONS.md §13): skip this one update
                skipped += 1
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        train_seconds = time.perf_counter() - t0
        val_acc = fraction_equal(predict(model, x_val), y_val)
        log.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                    "val_acc": val_acc, "seconds": time.perf_counter() - t0,
                    "train_seconds": train_seconds, "skipped_steps": skipped})
        if val_acc > best_val:  # strict: ties keep the earlier epoch
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if to_threshold is None and val_acc >= threshold:
            to_threshold = epoch
        if partial_path is not None:
            save_atomic(partial_path, {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "generator": loader.generator.get_state(), "log": log, "best_val": best_val,
                "best_epoch": best_epoch, "best_state": best_state, "to_threshold": to_threshold,
                "elapsed": elapsed + time.perf_counter() - t_start, "resumes": resumes})
            if should_yield is not None and epoch < epochs and should_yield():
                raise Yielded(epoch)
    total_train_seconds = elapsed + time.perf_counter() - t_start

    model.load_state_dict(best_state)
    t0 = time.perf_counter()
    if test_x is None:
        test_x, test_y = bundle.x_test, bundle.y_test
        pred = predict(model, test_x)
        A = fraction_equal(pred, test_y)
        noisy = torch.from_numpy(bundle.noisy_idx)
        A_subset = fraction_equal(pred[noisy], test_y[noisy])
    else:
        A = fraction_equal(predict(model, test_x), test_y)
        A_subset = None
    test_seconds = time.perf_counter() - t0

    record = {
        "run_id": run_id(cfg, seed), "config_id": config_id(cfg), "config": cfg, "seed": seed,
        "train_size": len(loader.dataset), "epochs": log, "best_epoch": best_epoch,
        "best_val_acc": best_val, "threshold": threshold, "epochs_to_threshold": to_threshold,
        "A": A, "A_subset_noiseless": A_subset,
        **param_counts(model),
        "timings": {"train_seconds": total_train_seconds, "test_seconds": test_seconds},
        "resumed_after_epochs": resumes,
        "skipped_steps_total": sum(e.get("skipped_steps", 0) for e in log),
        "environment": environment(),
    }
    if cfg["kind"] == "hybrid":
        record.update(gate_counts(cfg["n_qubits"], cfg["encoding"], cfg["topology"]))
        record["simulator"] = next(m for m in model.modules() if isinstance(m, QuantumCore)).device_name
    return record, best_state


# ----------------------------------------------------------------------------- run files

def run_paths(rid):
    return RUNS_DIR / f"{rid}.json", CHECKPOINT_DIR / f"{rid}.pt"


def partial_path(rid):
    return PARTIAL_DIR / f"{rid}.pt"


def yield_marker(rid):
    return RESULTS / "yield" / rid


def noise_path(rid, part):
    return NOISE_DIR / f"{rid}.{part}.json"


def valid_noise_record(path, part):
    try:
        with open(path) as f:
            return json.load(f).get("part") == part
    except (OSError, ValueError):
        return False


def save_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path, obj):
    """Atomic write: a JSON that exists is always complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def save_run(record, state):
    json_path, ckpt_path = run_paths(record["run_id"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    write_json(json_path, record)


def valid_training_record(path, epochs):
    """True if the run JSON exists, is complete, and its checkpoint exists."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return False
    return (len(rec.get("epochs", [])) == epochs and isinstance(rec.get("A"), float)
            and 0.0 <= rec["A"] <= 1.0 and not math.isnan(rec["A"])
            and run_paths(rec["run_id"])[1].exists())
