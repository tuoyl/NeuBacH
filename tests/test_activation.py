import numpy as np
import pytest

from neubach.physics.activation import activation_states, log_spaced_taus


def test_log_spaced_taus_cover_minutes_to_months():
    taus = log_spaced_taus()
    assert taus[0] == pytest.approx(300.0)
    assert taus[-1] == pytest.approx(100 * 86400.0)
    assert np.all(np.diff(taus) > 0)


def test_impulse_decays_exponentially():
    taus = np.array([600.0, 6000.0])
    time = np.arange(0.0, 3000.0, 1.0)
    dose = np.zeros_like(time)
    dose[1] = 5.0
    a = activation_states(time, dose, taus)
    # One second of dose rate 5 leaves ~5 units, then pure decay.
    assert a[1] == pytest.approx(5.0 * taus * (1 - np.exp(-1 / taus)))
    assert a[601, 0] / a[1, 0] == pytest.approx(np.exp(-1.0))
    assert a[601, 1] / a[1, 1] == pytest.approx(np.exp(-0.1))


def test_constant_dose_saturates_at_dose_times_tau():
    taus = np.array([50.0, 200.0])
    time = np.arange(0.0, 5000.0, 1.0)
    a = activation_states(time, np.full_like(time, 2.0), taus)
    assert a[-1] == pytest.approx(2.0 * taus, rel=1e-6)


def test_gap_is_treated_as_free_decay():
    taus = np.array([1000.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 2003.0])
    dose = np.array([0.0, 4.0, 4.0, 4.0, 4.0])
    a = activation_states(time, dose, taus, max_dt=2.0)
    before_gap = a[3, 0]
    expected = before_gap * np.exp(-2.0) + 4.0 * 1000.0 * (1 - np.exp(-2.0 / 1000.0))
    assert a[4, 0] == pytest.approx(expected)


def test_irregular_sampling_matches_fine_sampling():
    taus = np.array([300.0])
    fine_t = np.arange(0.0, 2000.0, 1.0)
    fine = activation_states(fine_t, np.full_like(fine_t, 1.5), taus)
    coarse_t = fine_t[::7]
    coarse = activation_states(coarse_t, np.full_like(coarse_t, 1.5), taus, max_dt=10.0)
    assert coarse[-1, 0] == pytest.approx(fine[coarse_t[-1].astype(int), 0], rel=1e-9)
