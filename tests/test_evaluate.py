"""Noisy-pipeline circuits are the lightning circuit: exact checks behind the 11.2 gates."""

import numpy as np
import pennylane as qml
import pytest
import torch
from qiskit.quantum_info import SparsePauliOp, Statevector

from qpar.evaluate import (CircuitTemplate, DepolarizingCore, calibrated_backend, noise_parts,
                           z_expectations)
from qpar.quantum import QuantumCore, decomposed_ops, encoding_width, feature_map, gate_counts


@pytest.fixture(scope="module")
def backend():
    return calibrated_backend("FakeAachen")[0]


def lightning(core, f):
    with torch.no_grad():
        return core(f).double()


def default_mixed(core, f, p, factor):
    """The protocol's 6.3 reference: PennyLane default.mixed with DepolarizingChannel(p) after
    every single-qubit gate and DepolarizingChannel(factor·p) on both qubits after every CNOT."""
    ops = decomposed_ops(f, core.weights, core.n, core.encoding, core.topology)
    noisy = []
    for op in ops:
        if op.name == "CNOT":
            noisy.append(qml.CNOT(wires=op.wires))
            strength = factor * p
        else:
            noisy.append(getattr(qml, op.name)(op.data[0].detach().numpy(), wires=op.wires))
            strength = p
        noisy += [qml.DepolarizingChannel(strength, wires=w) for w in op.wires]
    tape = qml.tape.QuantumScript(noisy, [qml.expval(qml.PauliZ(i)) for i in range(core.n)])
    res = qml.execute([tape], qml.device("default.mixed", wires=core.n))[0]
    return np.stack([np.broadcast_to(r, (len(f),)) for r in res], axis=-1)


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("encoding", ["angle", "amplitude"])
@pytest.mark.parametrize("topology", ["linear", "fullyconnected", "random"])
def test_qiskit_template_matches_lightning_and_G(n, encoding, topology, backend):
    torch.manual_seed(1)
    core = QuantumCore(n, encoding, topology)
    f = feature_map(encoding, torch.randn(3, encoding_width(encoding, n)))
    tpl = CircuitTemplate(n, encoding, topology, backend, optimization_level=1, seed_transpiler=0)
    assert sum(tpl.logical.count_ops().values()) - n == gate_counts(n, encoding, topology)["G"]
    values = tpl.values(decomposed_ops(f, core.weights, n, encoding, topology), len(f))
    expected = lightning(core, f).numpy()
    for b in range(len(f)):
        sv = Statevector(tpl.circuit.assign_parameters(dict(zip(tpl.theta, values[b]))))
        z = [sv.expectation_value(SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], n)).real for i in range(n)]
        np.testing.assert_allclose(z, expected[b], atol=1e-6)


@pytest.mark.parametrize("encoding", ["angle", "amplitude"])
def test_depolarizing_at_p0_matches_lightning(encoding):
    torch.manual_seed(2)
    core = QuantumCore(4, encoding, "fullyconnected")
    f = feature_map(encoding, torch.randn(5, encoding_width(encoding, 4)))
    aer = DepolarizingCore(core, p=0.0, factor=10, threads=1)(f).double()
    assert torch.allclose(aer, lightning(core, f), atol=1e-6)


@pytest.mark.parametrize("n,encoding,topology", [(2, "amplitude", "linear"), (4, "angle", "random"),
                                                 (4, "amplitude", "fullyconnected")])
@pytest.mark.parametrize("p", [0.005, 0.02])
def test_aer_depolarizing_matches_default_mixed_reference(n, encoding, topology, p):
    torch.manual_seed(3)
    core = QuantumCore(n, encoding, topology)
    f = feature_map(encoding, torch.randn(4, encoding_width(encoding, n)))
    aer = DepolarizingCore(core, p=p, factor=10, threads=1)(f).double().numpy()
    np.testing.assert_allclose(aer, default_mixed(core, f, p, 10), atol=1e-6)


def test_depolarizing_reduces_signal():
    torch.manual_seed(4)
    core = QuantumCore(4, "angle", "linear")
    f = feature_map("angle", torch.randn(5, 4))
    noisy = DepolarizingCore(core, p=0.02, factor=10, threads=1)(f).double().abs().sum()
    assert noisy < lightning(core, f).abs().sum()


def test_noise_parts():
    assert noise_parts({"depolarizing_p": [0.001, 0.02]}) == ["cal", "depol0.001", "depol0.02"]


def test_z_expectations_bit_order():
    # clbit 0 is the rightmost character: '01' means qubit 0 -> 1, qubit 1 -> 0
    assert list(z_expectations({"01": 1000}, 2, 1000)) == [-1.0, 1.0]
    assert list(z_expectations({"00": 500, "11": 500}, 2, 1000)) == [0.0, 0.0]
