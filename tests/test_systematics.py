import numpy as np
import pytest

from neubach.eval.systematics import (
    background_systematics,
    intrinsic_dispersion,
    regroup_by_exposure,
)


def test_recovers_injected_bias_and_dispersion():
    rng = np.random.default_rng(0)
    n = 4000
    stat = rng.uniform(0.5, 1.5, n)
    residual = 0.3 + rng.normal(0.0, 2.0, n) + rng.normal(0.0, stat)
    res = intrinsic_dispersion(residual, stat)
    assert res.n == n
    assert res.bias == pytest.approx(0.3, abs=4 * res.bias_err)
    assert res.sigma_sys == pytest.approx(2.0, rel=0.05)
    assert res.sigma_sys_lo < res.sigma_sys < res.sigma_sys_hi


def test_pure_statistical_scatter_gives_zero_dispersion():
    rng = np.random.default_rng(1)
    stat = np.full(500, 1.0)
    # Scatter clearly below the quoted errors: chi2 < N - 1 already at D = 0.
    res = intrinsic_dispersion(rng.normal(0.0, 0.5, 500), stat)
    assert res.sigma_sys == 0.0


def test_solution_satisfies_defining_equation():
    rng = np.random.default_rng(2)
    stat = rng.uniform(0.2, 0.4, 300)
    residual = rng.normal(0.0, 1.0, 300)
    res = intrinsic_dispersion(residual, stat)
    var = stat**2 + res.sigma_sys**2
    mean = np.sum(residual / var) / np.sum(1.0 / var)
    assert np.sum((residual - mean) ** 2 / var) == pytest.approx(299.0, rel=1e-5)


def test_regroup_sums_and_drops_incomplete_tail():
    exposure = np.ones(25)
    counts = np.arange(25.0)
    spec = np.tile(np.arange(3.0), (25, 1))
    exp_g, counts_g, spec_g = regroup_by_exposure(exposure, 10.0, counts, spec)
    assert exp_g.tolist() == [10.0, 10.0]
    assert counts_g.tolist() == [45.0, 145.0]
    assert spec_g.shape == (2, 3)
    assert spec_g[0].tolist() == [0.0, 10.0, 20.0]


def test_background_systematics_on_simulated_blank_sky():
    """A model that is off by a random 3% per kilosecond should read back ~3%."""
    rng = np.random.default_rng(3)
    texp, n_groups, rate = 1000, 600, 300.0
    scale = np.repeat(1.0 + rng.normal(0.0, 0.03, n_groups), texp)
    exposure = np.ones(texp * n_groups)
    pred = np.full(exposure.size, rate)
    obs = rng.poisson(rate * scale)
    res = background_systematics(obs, pred, exposure, float(texp))
    assert res.n == n_groups
    assert res.sigma_sys == pytest.approx(0.03, rel=0.1)
    assert abs(res.bias) < 5 * res.bias_err

    perfect = background_systematics(rng.poisson(pred), pred, exposure, float(texp))
    assert perfect.sigma_sys < 0.002
