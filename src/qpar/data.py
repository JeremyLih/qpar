"""Datasets, splits and subsets (protocol Section 3)."""

import numpy as np
import torch
from torchvision import datasets

from qpar import ROOT

DATA_DIR = ROOT / "data"
DATASETS = {"mnist": datasets.MNIST, "fashionmnist": datasets.FashionMNIST}
# Conventional mean/std of the 60,000 official training images, pixels scaled to [0, 1].
STATS = {"mnist": (0.1307, 0.3081), "fashionmnist": (0.2860, 0.3530)}

VAL_SIZE = 5000
SPLIT_SEED = 0
NOISY_SUBSET_SIZE = 2000
SUBSET_SEED = 0


def load_raw(name):
    """Official train and test sets as uint8 images [N, 28, 28] and int64 labels."""
    cls = DATASETS[name]
    train = cls(root=DATA_DIR, train=True, download=True)
    test = cls(root=DATA_DIR, train=False, download=True)
    return train.data, train.targets, test.data, test.targets


def normalize(images, name):
    """Scale uint8 images to [0, 1], then standardize: float32 [N, 1, 28, 28]."""
    mean, std = STATS[name]
    return ((images.float() / 255.0 - mean) / std).unsqueeze(1)


def stratified_sample(labels, size, rng):
    """Indices of `size` elements drawn without replacement, preserving class proportions.

    Per-class quotas use the largest-remainder method so they sum to `size` exactly; remainder
    ties go to the lower class index. Returned indices are sorted.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts * size / counts.sum()
    quota = np.floor(exact).astype(int)
    order = np.argsort(-(exact - quota), kind="stable")
    quota[order[: size - quota.sum()]] += 1
    chosen = [rng.choice(np.flatnonzero(labels == c), size=q, replace=False)
              for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(chosen))


def fixed_splits(train_labels, test_labels):
    """(val_idx, pool_idx, noisy_idx): fixed for every run of a dataset.

    val_idx and pool_idx index the official training set (5,000 / 55,000); noisy_idx indexes
    the official test set (2,000).
    """
    val_idx = stratified_sample(train_labels, VAL_SIZE, np.random.default_rng(SPLIT_SEED))
    pool_idx = np.setdiff1d(np.arange(len(train_labels)), val_idx)
    noisy_idx = stratified_sample(test_labels, NOISY_SUBSET_SIZE, np.random.default_rng(SUBSET_SEED))
    return val_idx, pool_idx, noisy_idx


def train_subset(pool_idx, train_labels, train_frac, seed):
    """Stratified `train_frac` of the pool, drawn with the run seed (indices into the train set)."""
    size = round(train_frac * len(pool_idx))
    if size == len(pool_idx):
        return pool_idx
    pos = stratified_sample(np.asarray(train_labels)[pool_idx], size, np.random.default_rng(seed))
    return pool_idx[pos]


class DatasetBundle:
    """Normalized tensors plus the fixed index sets for one dataset."""

    def __init__(self, name):
        x_train, y_train, x_test, y_test = load_raw(name)
        self.name = name
        self.y_train_all = y_train
        self.x_train_all = normalize(x_train, name)
        self.x_test = normalize(x_test, name)
        self.y_test = y_test
        self.val_idx, self.pool_idx, self.noisy_idx = fixed_splits(y_train.numpy(), y_test.numpy())

    def train(self, train_frac, seed):
        idx = torch.from_numpy(train_subset(self.pool_idx, self.y_train_all.numpy(), train_frac, seed))
        return self.x_train_all[idx], self.y_train_all[idx]

    def val(self):
        idx = torch.from_numpy(self.val_idx)
        return self.x_train_all[idx], self.y_train_all[idx]

    def noisy_subset(self):
        idx = torch.from_numpy(self.noisy_idx)
        return self.x_test[idx], self.y_test[idx]
