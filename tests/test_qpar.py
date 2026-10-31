"""11.1 / 8.1: Q-Par identities and the legacy arithmetic check."""

import numpy as np
import pytest

from qpar.qpar import qpar, qpar_legacy, sensitivity_settings

# One row of the v2 CSV (archive/v1_2025-09/results/qml_experiment_matrix_with_train_size.csv):
# Middle / Amplitude / Linear / 100%, the configuration the v2 manuscript reports as Q-Par = 0.4422.
# D_max and G_max are the maxima over that CSV's 54 rows. Arithmetic check only (8.1).
V2_ROW = dict(A=0.9936, sigma_a=1.35973995551052e-16, Q=4, D=0.000129060114755703, G=48)
V2_D_MAX, V2_G_MAX = 0.000190139354250277, 58
V2_QPAR = 0.442199569450629


@pytest.mark.parametrize("A", [0.1, 0.5, 0.987])
def test_qpar_reduces_to_A_at_the_maxima(A):
    assert qpar(A, 0.0, 8, 3.2, 1100, 8, 3.2, 1100) == pytest.approx(A, abs=1e-15)


def test_qpar_vectorized_and_monotone():
    A = np.array([0.9, 0.9, 0.9])
    q = qpar(A, 0.01, np.array([2, 4, 8]), 1.0, 100, 8, 2.0, 200)
    assert q.shape == (3,) and q[0] > q[1] > q[2]


def test_qpar_legacy_reproduces_v2_row():
    assert qpar_legacy(**V2_ROW, D_max=V2_D_MAX, G_max=V2_G_MAX) == pytest.approx(V2_QPAR, abs=1e-4)


def test_sensitivity_settings():
    s = sensitivity_settings()
    one_at_a_time = [k for k in s if "=" in k]
    assert len(one_at_a_time) + 1 == 11  # plus equal weights
    assert s["resource-first"] == {"alpha": 1.0, "beta": 1.0, "gamma": 2.0, "delta": 1.0, "epsilon": 2.0}
    assert len(s) == 14
