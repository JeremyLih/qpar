"""Evaluation conditions (protocol Section 6): noiseless, calibrated device noise, depolarizing.

Only the quantum block changes between conditions: the model's QuantumCore is swapped for an
executor with the same interface (feature batch -> n values <Z_i>); every classical layer runs
unchanged in PyTorch. Both noisy executors run the gate sequence from quantum.decomposed_ops —
the circuit lightning executes and G counts.

6.3 runs on Aer's density-matrix method with exact expectation values (decision 3c); it matches
PennyLane's default.mixed, the protocol's reference, to <= 1e-9 (tests/test_evaluate.py).
"""

import contextlib

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import fake_provider

from qpar.quantum import LAYERS, QuantumCore, decomposed_ops, encoding_width
from qpar.train import fraction_equal, predict

AER_CHUNK = 250


def quantum_core(model):
    return next(m for m in model.modules() if isinstance(m, QuantumCore))


@contextlib.contextmanager
def replaced_core(model, core):
    original = model.block.core
    model.block.core = core
    try:
        yield model
    finally:
        model.block.core = original


def accuracy(model, x, y):
    return fraction_equal(predict(model, x), y)


def _structure(ops):
    return tuple((op.name, tuple(op.wires)) for op in ops)


def _rotation_values(ops, batch):
    """(batch, n_rotations) angles in circuit order; scalar (weight) angles are broadcast."""
    cols = [np.broadcast_to(op.data[0].detach().numpy(), (batch,)) for op in ops if op.name != "CNOT"]
    return np.stack(cols, axis=1)


class LogicalCircuit:
    """One architecture's circuit as a parameterized Qiskit circuit (no measurement).

    Built gate-for-gate from quantum.decomposed_ops (PennyLane wire i = Qiskit qubit i); every
    rotation angle is a circuit parameter, bound per sample.
    """

    def __init__(self, n, encoding, topology):
        self.n, self.encoding, self.topology = n, encoding, topology
        rng = np.random.default_rng(0)
        ops = decomposed_ops(torch.tensor(rng.uniform(-1, 1, (1, encoding_width(encoding, n)))),
                             torch.tensor(rng.uniform(0, 2 * np.pi, (LAYERS, n, 3))), n, encoding, topology)
        self.structure = _structure(ops)
        self.theta = ParameterVector("theta", sum(op.name != "CNOT" for op in ops))
        self.circuit = QuantumCircuit(n, n)
        j = 0
        for op in ops:
            if op.name == "CNOT":
                self.circuit.cx(*op.wires)
            else:
                getattr(self.circuit, op.name.lower())(self.theta[j], op.wires[0])
                j += 1

    def values(self, ops, batch):
        if _structure(ops) != self.structure:
            raise RuntimeError("circuit structure differs from the template")
        return _rotation_values(ops, batch)


def _run_bound(simulator, circuit, params, values, options):
    """Run `circuit` once per row of `values` (parameters bound in chunks of AER_CHUNK) and yield
    (result, index); `options(start)` gives the run options for the chunk starting at `start`."""
    present = set(circuit.parameters)  # optimization may merge rotations away
    bindable = [(j, p) for j, p in enumerate(params) if p in present]
    for start in range(0, len(values), AER_CHUNK):
        chunk = values[start:start + AER_CHUNK]
        binds = [{p: chunk[:, j].tolist() for j, p in bindable}]
        result = simulator.run(circuit, parameter_binds=binds, **options(start)).result()
        for i in range(len(chunk)):
            yield result, i


# ----------------------------------------------------------------------------- 6.2 calibrated noise

def calibrated_backend(name):
    backend = getattr(fake_provider, name)()
    info = {"name": backend.name, "num_qubits": backend.num_qubits,
            "basis_gates": sorted(backend.operation_names),
            "calibration_date": str(backend.properties().last_update_date)}
    return backend, info


class CircuitTemplate(LogicalCircuit):
    """The logical circuit measured into clbits 0..n-1 and transpiled once to the backend."""

    def __init__(self, n, encoding, topology, backend, optimization_level, seed_transpiler):
        super().__init__(n, encoding, topology)
        measured = self.circuit.copy()
        measured.measure(range(n), range(n))
        self.logical = measured
        self.transpiled = transpile(measured, backend, optimization_level=optimization_level,
                                    seed_transpiler=seed_transpiler)
        self.depth = self.transpiled.depth()
        self.twoq = sum(1 for inst in self.transpiled.data if inst.operation.num_qubits == 2)


def z_expectations(counts, n, shots):
    """<Z_i> from counts; Qiskit bitstrings are little-endian (clbit i is character -1-i)."""
    ev = np.zeros(n)
    for bits, c in counts.items():
        for i in range(n):
            ev[i] += c if bits[-1 - i] == "0" else -c
    return ev / shots


class AerCore(nn.Module):
    """Executes the trained circuit on Aer with `shots` shots (noise_model=None: ideal sampling)."""

    def __init__(self, qcore, template, noise_model, shots, seed_simulator, threads):
        super().__init__()
        self.n, self.encoding, self.topology = qcore.n, qcore.encoding, qcore.topology
        self.weights = qcore.weights.detach().clone()
        self.template, self.shots, self.seed = template, shots, seed_simulator
        self.simulator = AerSimulator(noise_model=noise_model, max_parallel_threads=threads)

    def forward(self, features):
        ops = decomposed_ops(features, self.weights, self.n, self.encoding, self.topology)
        values = self.template.values(ops, len(features))
        out = np.empty((len(values), self.n))
        runs = _run_bound(self.simulator, self.template.transpiled, self.template.theta, values,
                          lambda start: {"shots": self.shots, "seed_simulator": self.seed + start})
        for k, (result, i) in enumerate(runs):
            out[k] = z_expectations(result.get_counts(i), self.n, self.shots)
        return torch.from_numpy(out).float()


# ----------------------------------------------------------------------------- 6.3 depolarizing

def depolarizing_noise_model(p, factor):
    """PennyLane DepolarizingChannel(q) = (1-q)ρ + q/3 (XρX+YρY+ZρZ) = Qiskit depolarizing_error(4q/3):
    q = p after every single-qubit gate, q = factor·p on each qubit after every CNOT."""
    model = NoiseModel(basis_gates=["ry", "rz", "cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(4 * p / 3, 1), ["ry", "rz"])
    two = depolarizing_error(4 * factor * p / 3, 1)
    model.add_all_qubit_quantum_error(two.tensor(two), ["cx"])
    return model


class DepolarizingCore(nn.Module):
    """Density-matrix simulation (Aer), exact <Z_i>: DepolarizingChannel(p) after every
    single-qubit gate and DepolarizingChannel(factor·p) on both qubits after every CNOT,
    encoding gates included."""

    def __init__(self, qcore, p, factor, threads):
        super().__init__()
        self.n, self.encoding, self.topology = qcore.n, qcore.encoding, qcore.topology
        self.weights = qcore.weights.detach().clone()
        self.logical = LogicalCircuit(self.n, self.encoding, self.topology)
        self.circuit = self.logical.circuit.copy()
        for i in range(self.n):
            self.circuit.save_expectation_value(Pauli("Z"), [i], label=f"z{i}")
        self.simulator = AerSimulator(method="density_matrix", noise_model=depolarizing_noise_model(p, factor),
                                      max_parallel_threads=threads, fusion_enable=False)

    def forward(self, features):
        ops = decomposed_ops(features, self.weights, self.n, self.encoding, self.topology)
        values = self.logical.values(ops, len(features))
        out = np.empty((len(values), self.n))
        runs = _run_bound(self.simulator, self.circuit, self.logical.theta, values, lambda start: {"shots": 1})
        for k, (result, i) in enumerate(runs):
            data = result.data(i)
            out[k] = [data[f"z{q}"] for q in range(self.n)]
        return torch.from_numpy(out).float()


# ----------------------------------------------------------------------------- driver

def noise_parts(noise_cfg):
    """Independent pieces of one run's noisy evaluation: 'cal' (6.2) and 'depol<p>' (6.3)."""
    return ["cal"] + [f"depol{p}" for p in noise_cfg["depolarizing_p"]]


def noise_part(model, x, y, noise_cfg, threads, part):
    """One part of Section 6.2 / 6.3 for a trained hybrid model on (x, y)."""
    qcore = quantum_core(model)
    if part == "cal":
        backend, info = calibrated_backend(noise_cfg["backend"])
        template = CircuitTemplate(qcore.n, qcore.encoding, qcore.topology, backend,
                                   noise_cfg["optimization_level"], noise_cfg["seed_transpiler"])
        core = AerCore(qcore, template, NoiseModel.from_backend(backend), noise_cfg["shots"],
                       noise_cfg["seed_simulator"], threads)
        with replaced_core(model, core):
            acc = accuracy(model, x, y)
        return {"part": part, "acc_noise_calibrated": acc, "twoq_transpiled": template.twoq,
                "depth_transpiled": template.depth, "noise_backend": info, "shots": noise_cfg["shots"]}
    p = float(part.removeprefix("depol"))
    with replaced_core(model, DepolarizingCore(qcore, p, noise_cfg["two_qubit_factor"], threads)):
        return {"part": part, "p": p, "acc_depol": accuracy(model, x, y)}
