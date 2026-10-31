"""11.1: seeding — two runs of the same config and seed give identical losses for 20 batches."""

import pytest
import torch
import torch.nn.functional as F

from qpar.data import DatasetBundle
from qpar.train import fraction_equal, load_base, make_training, train_run


def test_accuracy_is_an_exact_ratio():
    pred = torch.tensor([1] * 100 + [0] * 400)
    acc = fraction_equal(pred, torch.ones(500, dtype=torch.long))
    assert acc == 0.2 and not acc > 0.20  # a float32 mean gives 0.20000000298 > 0.2


@pytest.fixture(scope="module")
def bundle():
    return DatasetBundle("mnist")


def first_losses(cfg, seed, bundle, n_batches=20):
    model, optimizer, loader = make_training(cfg, seed, bundle, load_base())
    model.train()
    losses = []
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if len(losses) == n_batches:
            return losses


def test_resume_is_bit_identical(bundle, tmp_path):
    torch.set_num_threads(1)
    cfg = {"kind": "hybrid", "dataset": "mnist", "placement": "middle", "encoding": "angle",
           "topology": "random", "n_qubits": 2, "train_frac": 0.02}
    base, few = load_base(), dict(test_x=bundle.x_test[:300], test_y=bundle.y_test[:300])
    straight, straight_state = train_run(cfg, 3, bundle, base, epochs=2, **few)
    partial = tmp_path / "run.pt"
    train_run(cfg, 3, bundle, base, epochs=1, partial_path=partial, **few)  # "dies" after epoch 1
    resumed, resumed_state = train_run(cfg, 3, bundle, base, epochs=2, partial_path=partial, **few)
    strip = lambda log: [{k: v for k, v in e.items() if "seconds" not in k} for e in log]
    assert strip(resumed["epochs"]) == strip(straight["epochs"])
    assert (resumed["A"], resumed["best_epoch"]) == (straight["A"], straight["best_epoch"])
    assert resumed["resumed_after_epochs"] == [1] and straight["resumed_after_epochs"] == []
    assert all(torch.equal(straight_state[k], resumed_state[k]) for k in straight_state)


@pytest.mark.parametrize("cfg", [
    {"kind": "base_cnn", "dataset": "mnist", "train_frac": 0.2},
    {"kind": "hybrid", "dataset": "mnist", "placement": "middle", "encoding": "amplitude",
     "topology": "fullyconnected", "n_qubits": 4, "train_frac": 0.2},
    {"kind": "hybrid", "dataset": "mnist", "placement": "pre", "encoding": "angle",
     "topology": "random", "n_qubits": 4, "train_frac": 0.5},
])
def test_same_seed_identical_losses(cfg, bundle):
    torch.set_num_threads(2)
    a = first_losses(cfg, 1, bundle)
    b = first_losses(cfg, 1, bundle)
    assert len(a) == 20 and a == b
    assert first_losses(cfg, 2, bundle) != a


def test_nonfinite_gradient_skips_the_update(bundle, monkeypatch):
    import qpar.train as train_module
    real_build = train_module.build_model
    calls = {"n": 0}

    def poisoned_build(cfg):
        model = real_build(cfg)
        def hook(grad):
            calls["n"] += 1
            return torch.full_like(grad, float("nan")) if calls["n"] == 3 else grad
        model.block.core.weights.register_hook(hook)
        return model

    monkeypatch.setattr(train_module, "build_model", poisoned_build)
    cfg = {"kind": "hybrid", "dataset": "mnist", "placement": "post", "encoding": "angle",
           "topology": "linear", "n_qubits": 2, "train_frac": 0.02}
    record, state = train_run(cfg, 0, bundle, load_base(), epochs=1, test_x=bundle.x_test[:200],
                              test_y=bundle.y_test[:200])
    assert record["epochs"][0]["skipped_steps"] == 1 and record["skipped_steps_total"] == 1
    assert all(torch.isfinite(t).all() for t in state.values())
