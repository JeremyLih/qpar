"""11.1: topologies, encodings, gate counting; plus agreement of the training and evaluation paths."""

import math

import pytest
import torch

from qpar.quantum import (LAYERS, QuantumCore, TOPOLOGY_SEED, _stored_random_edges, draw_random_edges,
                          edges, feature_map, gate_counts)

NS = [2, 4, 6, 8]


@pytest.mark.parametrize("n", NS)
def test_linear_and_fullyconnected_edges_exact(n):
    assert edges("linear", n) == [(i, i + 1) for i in range(n - 1)]
    assert edges("fullyconnected", n) == [(i, j) for i in range(n) for j in range(i + 1, n)]
    assert len(edges("fullyconnected", n)) == n * (n - 1) // 2


@pytest.mark.parametrize("n", NS)
def test_random_edges_reproducible_from_seed_and_stored(n):
    drawn = draw_random_edges(n, TOPOLOGY_SEED)
    assert drawn == draw_random_edges(n, TOPOLOGY_SEED)
    assert edges("random", n) == drawn == _stored_random_edges()[n]
    assert len(drawn) == n - 1 and len(set(drawn)) == n - 1
    assert all(0 <= i < j < n for i, j in drawn)
    if n > 2:  # decision 4b: Random differs in structure from Linear
        assert drawn != edges("linear", n)


def test_random_edges_at_n4_is_the_first_non_linear_draw():
    assert draw_random_edges(4) == [(0, 1), (0, 2), (0, 3)]


def test_amplitude_features_normalized_including_zero_projection():
    u = torch.cat([torch.randn(64, 16) * 3, torch.zeros(1, 16)])
    v = feature_map("amplitude", u)
    assert v.dtype == torch.float64
    assert torch.allclose(v.norm(dim=1), torch.ones(65, dtype=torch.float64), atol=1e-12)


def test_angle_features_in_open_interval():
    # float64 tanh rounds to exactly ±1 only for |u| > ~18.4 (DECISIONS.md §2); below that the
    # open interval holds, and no input ever maps beyond ±π.
    u = torch.linspace(-18, 18, 10001, dtype=torch.float64).reshape(-1, 1)
    z = feature_map("angle", u)
    assert z.dtype == torch.float64
    assert (z.abs() < math.pi).all()
    extreme = feature_map("angle", torch.tensor([[-1e6, -30.0, 30.0, 1e6]]))
    assert (extreme.abs() <= math.pi).all()


def test_gate_count_hand_built_two_qubit_one_layer_angle_linear():
    # 2 x RY (encoding) + 2 x Rot = 2 x (RZ, RY, RZ) + 1 CNOT
    counts = gate_counts(2, "angle", "linear", layers=1)
    assert counts["G"] == 9
    assert counts["gate_types"] == {"CNOT": 1, "RY": 4, "RZ": 4}
    assert (counts["G_encoding"], counts["G_variational"]) == (2, 7)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("topology", ["linear", "fullyconnected", "random"])
@pytest.mark.parametrize("encoding", ["angle", "amplitude"])
def test_gate_count_structure(n, topology, encoding):
    counts = gate_counts(n, encoding, topology)
    assert counts["G_variational"] == LAYERS * (3 * n + len(edges(topology, n)))
    # angle: n RY; amplitude: full Möttönen, (2^n-1) RY + (2^n-1) RZ + 2(2^n-2) CNOT
    expected_enc = n if encoding == "angle" else 2 * (2 ** n - 1) + 2 * (2 ** n - 2)
    assert counts["G_encoding"] == expected_enc


@pytest.mark.parametrize("encoding,k", [("angle", 4), ("amplitude", 16)])
def test_training_path_equals_evaluation_path(encoding, k):
    torch.manual_seed(0)
    core = QuantumCore(4, encoding, "fullyconnected")
    f = feature_map(encoding, torch.randn(8, k))
    train_out = core(f)
    with torch.no_grad():
        eval_out = core(f)
    assert torch.allclose(train_out, eval_out, atol=1e-6)
