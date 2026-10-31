"""11.1: stratified splits, disjointness, determinism (both datasets)."""

import numpy as np
import pytest

from qpar.data import DatasetBundle, fixed_splits, train_subset

TOL = 0.005  # ±0.5 percentage points


@pytest.fixture(scope="module", params=["mnist", "fashionmnist"])
def bundle(request):
    return DatasetBundle(request.param)


def proportions(labels):
    return np.bincount(labels, minlength=10) / len(labels)


def test_sizes(bundle):
    assert (len(bundle.val_idx), len(bundle.pool_idx), len(bundle.noisy_idx)) == (5000, 55000, 2000)
    y = bundle.y_train_all.numpy()
    for frac, size in [(0.2, 11000), (0.5, 27500), (1.0, 55000)]:
        assert len(train_subset(bundle.pool_idx, y, frac, seed=3)) == size


def test_stratified_class_proportions(bundle):
    # Decision 5a: within ±0.5 pp of the *source* distribution (MNIST is not class-balanced).
    y, yt = bundle.y_train_all.numpy(), bundle.y_test.numpy()
    source, source_test = proportions(y), proportions(yt)
    assert np.abs(proportions(y[bundle.val_idx]) - source).max() < TOL
    assert np.abs(proportions(yt[bundle.noisy_idx]) - source_test).max() < TOL
    for frac in (0.2, 0.5):
        for seed in range(5):
            sub = train_subset(bundle.pool_idx, y, frac, seed)
            assert np.abs(proportions(y[sub]) - source).max() < TOL


def test_validation_and_test_disjoint_from_training(bundle):
    y = bundle.y_train_all.numpy()
    assert not np.intersect1d(bundle.val_idx, bundle.pool_idx).size
    for frac in (0.2, 0.5, 1.0):
        for seed in range(5):
            sub = train_subset(bundle.pool_idx, y, frac, seed)
            assert not np.intersect1d(sub, bundle.val_idx).size
            assert np.isin(sub, bundle.pool_idx).all() and len(np.unique(sub)) == len(sub)
    # the test set is the separate official file; the noisy subset indexes only it
    assert bundle.noisy_idx.max() < len(bundle.y_test) and len(np.unique(bundle.noisy_idx)) == 2000


def test_splits_byte_identical_for_same_seed(bundle):
    y, yt = bundle.y_train_all.numpy(), bundle.y_test.numpy()
    first, second = fixed_splits(y, yt), fixed_splits(y, yt)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(first, second))
    a = train_subset(bundle.pool_idx, y, 0.5, seed=2)
    assert a.tobytes() == train_subset(bundle.pool_idx, y, 0.5, seed=2).tobytes()
    assert a.tobytes() != train_subset(bundle.pool_idx, y, 0.5, seed=3).tobytes()
