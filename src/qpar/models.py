"""Placements and classical baselines, assembled from cnn.py and quantum.py (protocol 4.3, 4.6).

Placement wiring: the canonical defaults of 4.3, no dropout (decision 1A; the archive's extracted
wiring and why it was not used are in results/DECISIONS.md §3a).
"""

import torch.nn as nn
import torch.nn.functional as F

from qpar.cnn import CNNBody, ConvFeatures
from qpar.quantum import LAYERS, QuantumBlock, QuantumCore, encoding_width

PLACEMENTS = ("pre", "middle", "post")
D_IN = {"pre": 784, "middle": 256, "post": 32}


class PreModel(nn.Module):
    """QB(784 -> n) on the flattened image; Linear(n, 784) expands its output, reshaped to
    1x28x28 and added to the image (residual); the unchanged CNN body then runs."""

    def __init__(self, block, n):
        super().__init__()
        self.block = block
        self.expand = nn.Linear(n, 784)
        self.cnn = CNNBody()

    def forward(self, x):
        residual = self.expand(self.block(x.flatten(1))).view(-1, 1, 28, 28)
        return self.cnn(x + residual)


class MiddleModel(nn.Module):
    """QB(256 -> n) replaces fc1; followed by Linear(n, 32) -> ReLU -> fc2."""

    def __init__(self, block, n):
        super().__init__()
        self.features = ConvFeatures()
        self.block = block
        self.fc_after = nn.Linear(n, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc_after(self.block(self.features(x)))))


class PostModel(nn.Module):
    """QB(32 -> n) replaces fc2; followed by Linear(n, 10)."""

    def __init__(self, block, n):
        super().__init__()
        self.features = ConvFeatures()
        self.fc1 = nn.Linear(256, 32)
        self.block = block
        self.head = nn.Linear(n, 10)

    def forward(self, x):
        return self.head(self.block(F.relu(self.fc1(self.features(x)))))


PLACEMENT_MODELS = {"pre": PreModel, "middle": MiddleModel, "post": PostModel}


def matched_candidates(k, n):
    """Parameter counts of the allowed classical maps k -> n (4.6), in the protocol's order."""
    counts = {"linear_nobias": k * n, "linear": k * n + n}
    for h in (1, 2, 3):
        counts[f"mlp_h{h}"] = (k * h + h) + (h * n + n)
    return counts


class MatchedCore(nn.Module):
    """cnn_matched: replaces encoding + variational circuit + readout by the classical map whose
    parameter count is closest to the quantum part's 3·n·L (ties: first in the protocol's order).
    It receives exactly the tensor the circuit would have consumed (feature-map output)."""

    def __init__(self, k, n, target):
        super().__init__()
        counts = matched_candidates(k, n)
        self.kind = min(counts, key=lambda c: abs(counts[c] - target))
        self.target = target
        if self.kind == "linear_nobias":
            self.map = nn.Linear(k, n, bias=False)
        elif self.kind == "linear":
            self.map = nn.Linear(k, n)
        else:
            h = int(self.kind[-1])
            self.map = nn.Sequential(nn.Linear(k, h), nn.Tanh(), nn.Linear(h, n))

    def forward(self, features):
        return self.map(features.float())


def build_model(cfg):
    kind = cfg["kind"]
    if kind == "base_cnn":
        return CNNBody()
    n, encoding, placement = cfg["n_qubits"], cfg["encoding"], cfg["placement"]
    if kind == "hybrid":
        core = QuantumCore(n, encoding, cfg["topology"])
    elif kind == "matched":
        core = MatchedCore(encoding_width(encoding, n), n, target=3 * n * LAYERS)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return PLACEMENT_MODELS[placement](QuantumBlock(D_IN[placement], n, encoding, core), n)


def param_counts(model):
    """Exact trainable parameter counts; for cnn_matched also the replacement map's size."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum = sum(p.numel() for m in model.modules() if isinstance(m, QuantumCore)
                  for p in m.parameters())
    counts = {"params_total": total, "params_quantum": quantum, "params_classical": total - quantum}
    for m in model.modules():
        if isinstance(m, MatchedCore):
            counts.update(matched_map=m.kind, params_matched_map=sum(p.numel() for p in m.parameters()),
                          params_matched_target=m.target)
    return counts
