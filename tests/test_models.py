"""11.1: parameter counts (CNN body, quantum part, cnn_matched) and model shapes."""

import pytest
import torch

from qpar.cnn import CNNBody
from qpar.models import MatchedCore, build_model, matched_candidates, param_counts
from qpar.quantum import LAYERS, encoding_width


def hybrid(placement, encoding, n=4, topology="linear"):
    return {"kind": "hybrid", "dataset": "mnist", "placement": placement, "encoding": encoding,
            "topology": topology, "n_qubits": n, "train_frac": 1.0}


def test_cnn_body_has_11978_parameters():
    assert sum(p.numel() for p in CNNBody().parameters()) == 11_978


@pytest.mark.parametrize("n", [2, 4, 6, 8])
@pytest.mark.parametrize("encoding", ["angle", "amplitude"])
@pytest.mark.parametrize("placement", ["pre", "middle", "post"])
def test_quantum_part_is_3nL(placement, encoding, n):
    counts = param_counts(build_model(hybrid(placement, encoding, n)))
    assert counts["params_quantum"] == 3 * n * LAYERS


def test_matched_candidate_counts_match_instantiated_maps():
    for k, n in [(4, 4), (16, 4)]:
        counts = matched_candidates(k, n)
        for kind, expected in counts.items():
            core = MatchedCore(k, n, target=expected)  # a target equal to a count selects that map
            assert core.kind == kind
            assert sum(p.numel() for p in core.parameters()) == expected


@pytest.mark.parametrize("encoding,kind,size", [("angle", "mlp_h3", 31), ("amplitude", "mlp_h2", 46)])
def test_matched_map_is_closest_to_36(encoding, kind, size):
    core = MatchedCore(encoding_width(encoding, 4), 4, target=36)
    assert (core.kind, sum(p.numel() for p in core.parameters())) == (kind, size)


@pytest.mark.parametrize("encoding", ["angle", "amplitude"])
@pytest.mark.parametrize("placement", ["pre", "middle", "post"])
def test_matched_total_within_half_percent_of_hybrid(placement, encoding):
    h = param_counts(build_model(hybrid(placement, encoding)))
    m = param_counts(build_model({**hybrid(placement, encoding), "kind": "matched"}))
    assert m["params_quantum"] == 0
    assert h["params_total"] - h["params_quantum"] == m["params_total"] - m["params_matched_map"]
    assert abs(m["params_total"] - h["params_total"]) / h["params_total"] < 0.005


@pytest.mark.parametrize("cfg", [
    {"kind": "base_cnn"},
    hybrid("pre", "amplitude"), hybrid("middle", "angle"), hybrid("post", "amplitude", topology="random"),
    {**hybrid("pre", "angle"), "kind": "matched"},
])
def test_forward_shape(cfg):
    model = build_model(cfg)
    out = model(torch.randn(3, 1, 28, 28))
    assert out.shape == (3, 10) and out.dtype == torch.float32 and torch.isfinite(out).all()
