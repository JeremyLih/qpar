"""Quantum block, encodings, topologies and gate counting (protocol 4.2, 4.4, 4.5, 7)."""

import math
from functools import cache, partial

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import yaml

from qpar import ROOT

LAYERS = 3
TOPOLOGY_SEED = 2026
TOPOLOGIES_FILE = ROOT / "configs" / "topologies.yaml"
# Elementary gate set for G and for the noisy circuits. GlobalPhase (emitted by the Möttönen
# decomposition) stays in the tape but is not a physical gate: excluded from G, never executed.
ELEMENTARY = {"RX", "RY", "RZ", "CNOT", "GlobalPhase"}
# Under adjoint differentiation PennyLane decomposes AmplitudeEmbedding into these gates itself;
# decomposing before the batch is split does it once per batch instead of once per sample.
_TRAINING_GATES = {"RY", "RZ", "CNOT", "Rot", "GlobalPhase"}


# ----------------------------------------------------------------------------- topologies (4.5)

def draw_random_edges(n, seed=TOPOLOGY_SEED):
    """n-1 distinct pairs drawn uniformly without replacement from all pairs; sorted.

    Draws repeat from the same generator until the edge set differs from Linear's (decision 4b:
    at n = 4 the first seed-2026 draw is the Linear chain). n = 2 has a single pair, so there
    Random, Linear and FullyConnected necessarily coincide.
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng(seed)
    while True:
        edge_set = sorted(pairs[k] for k in rng.choice(len(pairs), size=n - 1, replace=False))
        if n == 2 or edge_set != [(i, i + 1) for i in range(n - 1)]:
            return edge_set


@cache
def _stored_random_edges():
    with open(TOPOLOGIES_FILE) as f:
        return {int(n): [tuple(e) for e in es] for n, es in yaml.safe_load(f)["random"].items()}


def edges(topology, n):
    """Entangling CNOT pairs (control, target) applied in order after each Rot layer."""
    if topology == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if topology == "fullyconnected":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if topology == "random":
        return _stored_random_edges()[n]
    raise ValueError(f"unknown topology {topology!r}")


# ----------------------------------------------------------------------------- encodings (4.4)

def encoding_width(encoding, n):
    """k: the input projection width, n for angle and 2^n for amplitude encoding."""
    return n if encoding == "angle" else 2 ** n


def feature_map(encoding, u):
    """Map the projection output u to what the circuit consumes, in float64.

    angle: z = pi * tanh(u), in (-pi, pi).
    amplitude: v = u + 1e-8 e_0 (avoids the zero vector), L2-normalized; AmplitudeEmbedding
    re-normalizes (a no-op) as the protocol's `normalize=True` specifies.
    """
    u = u.double()
    if encoding == "angle":
        return math.pi * torch.tanh(u)
    v = torch.cat([u[..., :1] + 1e-8, u[..., 1:]], dim=-1)
    return v / v.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------------- circuit (4.2)

def encode(features, n, encoding):
    if encoding == "angle":
        for i in range(n):
            qml.RY(features[..., i], wires=i)
    else:
        qml.AmplitudeEmbedding(features, wires=range(n), normalize=True)


def variational(weights, n, edge_list):
    """L layers: Rot(phi, theta, omega) on every qubit, then the topology's CNOTs."""
    for layer in weights:
        for i in range(n):
            qml.Rot(layer[i, 0], layer[i, 1], layer[i, 2], wires=i)
        for a, b in edge_list:
            qml.CNOT(wires=[a, b])


def circuit(features, weights, n, encoding, edge_list):
    encode(features, n, encoding)
    variational(weights, n, edge_list)
    return [qml.expval(qml.PauliZ(i)) for i in range(n)]


def simulator(n):
    """lightning.qubit; default.qubit only if lightning is unavailable (recorded per run)."""
    try:
        return qml.device("lightning.qubit", wires=n)
    except qml.DeviceError:
        return qml.device("default.qubit", wires=n)


class QuantumCore(nn.Module):
    """Encoding + variational circuit + readout: k features -> n values <Z_i>.

    The 3·n·L Rot angles are the only trainable parameters, initialised uniform in [0, 2π)
    from the (seeded) torch generator. Analytic expectation values, adjoint gradients.
    """

    def __init__(self, n, encoding, topology, layers=LAYERS):
        super().__init__()
        self.n, self.encoding, self.topology = n, encoding, topology
        self.edges = edges(topology, n)
        self.weights = nn.Parameter(2 * math.pi * torch.rand(layers, n, 3))
        device = simulator(n)
        self.device_name = device.name
        body = partial(circuit, n=n, encoding=encoding, edge_list=self.edges)
        self._qnode = qml.QNode(body, device, interface="torch", diff_method="adjoint")
        self._training_qnode = (qml.transforms.decompose(self._qnode, gate_set=_TRAINING_GATES)
                                if encoding == "amplitude" else self._qnode)

    def forward(self, features):
        if torch.is_grad_enabled():
            out = self._training_qnode(features, self.weights.double())
        else:  # detached inputs: no Jacobian is computed during evaluation
            out = self._qnode(features.detach(), self.weights.detach().double())
        return torch.stack(out, dim=-1).float()


class QuantumBlock(nn.Module):
    """QB(d_in -> n): input projection Linear(d_in, k), feature map, then a core k -> n.

    For hybrid models the core is a QuantumCore; cnn_matched swaps in a classical map (4.6).
    """

    def __init__(self, d_in, n, encoding, core):
        super().__init__()
        self.encoding = encoding
        self.proj = nn.Linear(d_in, encoding_width(encoding, n))
        self.core = core

    def forward(self, x):
        return self.core(feature_map(self.encoding, self.proj(x)))


# ----------------------------------------------------------------------------- gate counting (7)

def decomposed_ops(features, weights, n, encoding, topology):
    """The circuit's gate sequence over ELEMENTARY for a batch of features (GlobalPhase dropped).

    The features are marked as requiring gradients so PennyLane emits the complete Möttönen
    structure — the structure G counts and the one executed during training — for every batch.
    Encoding rotations carry batched parameters of shape (B,); Rot-derived rotations are scalars.
    """
    f = features.detach().double().clone().requires_grad_(True)
    w = torch.as_tensor(weights).detach().double()
    with qml.queuing.AnnotatedQueue() as queue:
        circuit(f, w, n, encoding, edges(topology, n))
    tape = qml.tape.QuantumScript.from_queue(queue)
    (tape,), _ = qml.transforms.decompose(tape, gate_set=ELEMENTARY)
    return [op for op in tape.operations if op.name != "GlobalPhase"]


def _generic_inputs(n, encoding, layers):
    rng = np.random.default_rng(0)
    k = encoding_width(encoding, n)
    features = torch.tensor(rng.uniform(0.1, 1.0, k) * rng.choice([-1.0, 1.0], k))
    weights = torch.tensor(rng.uniform(0.1, 2 * np.pi, (layers, n, 3)))
    return features, weights


def gate_counts(n, encoding, topology, layers=LAYERS):
    """G (full decomposition, encoding included) and its split, counted with qml.specs."""
    features, weights = _generic_inputs(n, encoding, layers)
    edge_list = edges(topology, n)
    device = qml.device("default.qubit", wires=n)

    def count(fn):
        qnode = qml.transforms.decompose(qml.QNode(fn, device, interface="torch"), gate_set=ELEMENTARY)
        gates = dict(qml.specs(qnode, level="user")(features.clone().requires_grad_(True), weights)
                     .resources.gate_types)
        gates.pop("GlobalPhase", None)
        return gates

    def variational_only(f, w):
        variational(w, n, edge_list)
        return [qml.expval(qml.PauliZ(0))]

    full = count(partial(circuit, n=n, encoding=encoding, edge_list=edge_list))
    var = count(variational_only)
    G, G_var = sum(full.values()), sum(var.values())
    return {"G": G, "G_variational": G_var, "G_encoding": G - G_var,
            "gate_types": dict(sorted(full.items()))}
