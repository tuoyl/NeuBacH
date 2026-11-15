"""Background-reproducibility metrics in the convention of the Insight-HXMT papers.

The official background models quote a *bias* and a *systematic error* obtained
from blank-sky residuals after removing the statistical scatter of the test
data.  For residuals ``R_i = S_obs,i - S_bkg,i`` with statistical errors
``sigma_i`` they solve (Liao et al. 2020, arXiv:2005.01661, Eq. 7-9; the same
method is used for LE in arXiv:2004.01432)::

    sum_i (R_i - R_m)^2 / (D_in^2 + sigma_i^2) = N - 1
    R_m = sum_i w_i R_i,   w_i ∝ 1 / (D_in^2 + sigma_i^2)

``R_m`` is the bias and ``D_in`` (the intrinsic dispersion) the systematic
error.  The test data are first merged and re-grouped into sub-observations of a
fixed exposure (1, 2, 4, ... ks), so the quoted numbers are exposure dependent.

Everything NeuBacH reports must go through these functions, so that results are
directly comparable with the published tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DispersionResult",
    "intrinsic_dispersion",
    "regroup_by_exposure",
    "background_systematics",
]


@dataclass(frozen=True)
class DispersionResult:
    """Bias and intrinsic dispersion of a set of residuals.

    ``sigma_sys_lo`` / ``sigma_sys_hi`` bound the intrinsic dispersion: they are
    the solutions for a chi-square of ``(N-1) ± sqrt(2 (N-1))``.
    """

    bias: float
    bias_err: float
    sigma_sys: float
    sigma_sys_lo: float
    sigma_sys_hi: float
    n: int


def _weighted_mean(residual: np.ndarray, var: np.ndarray) -> tuple[float, float]:
    w = 1.0 / var
    wsum = w.sum()
    return float((w * residual).sum() / wsum), float(np.sqrt(1.0 / wsum))


def _chi2(residual: np.ndarray, stat_var: np.ndarray, d_in: float) -> float:
    var = stat_var + d_in * d_in
    mean, _ = _weighted_mean(residual, var)
    return float(((residual - mean) ** 2 / var).sum())


def _solve_dispersion(
    residual: np.ndarray, stat_var: np.ndarray, target: float, tol: float
) -> float:
    """Return D >= 0 with chi2(D) = target; 0 if the scatter is already below it."""
    if target <= 0.0:
        return float("inf")
    if _chi2(residual, stat_var, 0.0) <= target:
        return 0.0
    # chi2(D) decreases monotonically to 0, so a bracket always exists.
    lo, hi = 0.0, float(np.std(residual)) or 1.0
    while _chi2(residual, stat_var, hi) > target:
        hi *= 2.0
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if _chi2(residual, stat_var, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def intrinsic_dispersion(
    residual: np.ndarray, stat_err: np.ndarray, *, tol: float = 1e-8
) -> DispersionResult:
    """Solve Eq. 7-9 of Liao et al. (2020) for the bias and systematic error.

    Parameters
    ----------
    residual
        ``R_i``: observed minus predicted background for N sub-observations.
    stat_err
        ``sigma_i``: 1-sigma statistical error of each residual (must be > 0).
    """
    residual = np.asarray(residual, dtype=np.float64).ravel()
    stat_err = np.asarray(stat_err, dtype=np.float64).ravel()
    if residual.shape != stat_err.shape:
        raise ValueError("residual and stat_err must have the same shape")
    n = residual.size
    if n < 2:
        raise ValueError("at least two sub-observations are required")
    if not np.all(stat_err > 0):
        raise ValueError("stat_err must be strictly positive")

    stat_var = stat_err**2
    dof = n - 1.0
    spread = np.sqrt(2.0 * dof)
    d_in = _solve_dispersion(residual, stat_var, dof, tol)
    # A larger chi-square target corresponds to a smaller dispersion.
    d_lo = _solve_dispersion(residual, stat_var, dof + spread, tol)
    d_hi = _solve_dispersion(residual, stat_var, dof - spread, tol)
    bias, bias_err = _weighted_mean(residual, stat_var + d_in * d_in)
    return DispersionResult(bias, bias_err, d_in, d_lo, d_hi, n)


def regroup_by_exposure(
    exposure: np.ndarray, texp: float, *arrays: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Merge time-ordered samples into sub-observations of ``texp`` seconds.

    Follows the papers: all test data are concatenated and cut into consecutive
    pieces of equal exposure; a trailing incomplete piece is dropped.

    Parameters
    ----------
    exposure
        Live time of each sample in seconds, shape ``(T,)``, time ordered.
    texp
        Target exposure of each sub-observation in seconds.
    *arrays
        Arrays with leading dimension ``T`` (e.g. observed and predicted counts
        of shape ``(T,)`` or ``(T, C)``) that are summed within each group.

    Returns
    -------
    ``(group_exposure, *summed_arrays)`` with leading dimension ``G``.
    """
    exposure = np.asarray(exposure, dtype=np.float64)
    if exposure.ndim != 1:
        raise ValueError("exposure must be one-dimensional")
    if texp <= 0:
        raise ValueError("texp must be positive")
    elapsed_before = np.cumsum(exposure) - exposure
    group = np.floor(elapsed_before / texp).astype(np.int64)
    n_groups = int(group[-1]) + 1 if group.size else 0
    starts = np.searchsorted(group, np.arange(n_groups), side="left")

    group_exposure = np.add.reduceat(exposure, starts) if n_groups else exposure[:0]
    # The last piece is usually short; keep it only if it is essentially full.
    keep = n_groups
    if n_groups and group_exposure[-1] < 0.999 * texp:
        keep -= 1

    out = [group_exposure[:keep]]
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.shape[0] != exposure.shape[0]:
            raise ValueError("arrays must share the leading dimension of exposure")
        summed = np.add.reduceat(arr, starts, axis=0) if n_groups else arr[:0]
        out.append(summed[:keep])
    return tuple(out)


def background_systematics(
    counts_obs: np.ndarray,
    counts_pred: np.ndarray,
    exposure: np.ndarray,
    texp: float,
    *,
    relative: bool = True,
) -> DispersionResult:
    """Bias and systematic error of a background prediction at exposure ``texp``.

    ``counts_obs`` / ``counts_pred`` are per-sample counts in one energy band
    (sum the channels of the band first), time ordered, shape ``(T,)``.  With
    ``relative=True`` the result is a fraction of the mean observed background
    rate, i.e. multiply by 100 to compare with the percentages in the papers.

    The statistical error of each residual is the Poisson error of the test
    data, taken from the predicted counts (the expected variance), which avoids
    the bias of ``sqrt(N_obs)`` at low counts.
    """
    exp_g, obs_g, pred_g = regroup_by_exposure(
        exposure,
        texp,
        np.asarray(counts_obs, dtype=np.float64),
        np.asarray(counts_pred, dtype=np.float64),
    )
    if obs_g.ndim != 1:
        raise ValueError("counts must be one-dimensional (one energy band)")
    residual = (obs_g - pred_g) / exp_g
    stat_err = np.sqrt(np.maximum(pred_g, 1.0)) / exp_g
    if relative:
        mean_rate = obs_g.sum() / exp_g.sum()
        residual = residual / mean_rate
        stat_err = stat_err / mean_rate
    return intrinsic_dispersion(residual, stat_err)
