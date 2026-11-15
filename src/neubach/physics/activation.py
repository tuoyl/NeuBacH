"""Activation-state features: irradiation history folded with decay kernels.

The HE background is dominated by delayed emission from nuclei activated by
trapped protons in the SAA and by cosmic rays (Xie et al. 2015,
arXiv:1511.02997; Zhang et al. 2020, arXiv:2009.14489).  After each SAA passage
it decays on time scales from minutes to days, and long-lived isotopes build up
over hundreds of days.  The official model only sees this history through the
ascending/descending flag and a per-channel long-term trend.

Here the history is made explicit.  For a dose-rate proxy ``D(t)`` (particle
monitor counts, or an orbit-propagated SAA map) and a set of time constants
``tau_k``, the state

    da_k/dt = -a_k / tau_k + D(t)

is the amount of not-yet-decayed dose on time scale ``tau_k`` -- the same idea as
the "PINUD build-up" parameters of the Suzaku/HXD model (Fukazawa et al. 2009).
A background component then scales as ``a_k(t) / tau_k``.  This NumPy version
uses fixed time constants; the trainable model re-implements the same recursion
with learnable ones.
"""

from __future__ import annotations

import numpy as np

__all__ = ["log_spaced_taus", "activation_states"]


def log_spaced_taus(
    tau_min: float = 300.0, tau_max: float = 100.0 * 86400.0, n: int = 12
) -> np.ndarray:
    """Logarithmically spaced decay constants in seconds (default 5 min - 100 d)."""
    if not 0 < tau_min < tau_max:
        raise ValueError("require 0 < tau_min < tau_max")
    return np.geomspace(tau_min, tau_max, n)


def activation_states(
    time: np.ndarray,
    dose_rate: np.ndarray,
    taus: np.ndarray,
    *,
    max_dt: float | None = None,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate ``da_k/dt = -a_k/tau_k + D(t)`` along a (possibly gappy) timeline.

    Parameters
    ----------
    time
        Sample times in seconds, strictly increasing, shape ``(T,)``.
    dose_rate
        Dose-rate proxy ``D`` for the interval ending at each sample, shape
        ``(T,)``.  The first sample only sets the starting point.
    taus
        Decay constants in seconds, shape ``(K,)``.
    max_dt
        Longest interval over which ``dose_rate`` is assumed to hold.  For a
        longer step (a telemetry gap) the dose is applied for the last
        ``max_dt`` seconds only and the state decays freely before that.
        Defaults to twice the median sampling interval.
    initial
        State at ``time[0]``, shape ``(K,)``; zero if omitted.

    Returns
    -------
    ``a`` with shape ``(T, K)``, in units of dose-rate x seconds.  A constant
    dose rate ``D`` saturates at ``D * tau_k``.
    """
    time = np.asarray(time, dtype=np.float64)
    dose_rate = np.asarray(dose_rate, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if time.ndim != 1 or dose_rate.shape != time.shape:
        raise ValueError("time and dose_rate must be 1-D arrays of equal length")
    if taus.ndim != 1 or not np.all(taus > 0):
        raise ValueError("taus must be a 1-D array of positive values")

    states = np.zeros((time.size, taus.size))
    if time.size == 0:
        return states
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("time must be strictly increasing")
    if max_dt is None:
        max_dt = 2.0 * float(np.median(dt)) if dt.size else 0.0

    state = np.zeros(taus.size) if initial is None else np.asarray(initial, float).copy()
    states[0] = state
    for j in range(1, time.size):
        step = dt[j - 1]
        irradiated = min(step, max_dt)
        # Free decay over the unobserved part of a gap, then exact integration
        # of a constant dose rate over the irradiated part.
        state = state * np.exp(-step / taus)
        state = state + dose_rate[j] * taus * (-np.expm1(-irradiated / taus))
        states[j] = state
    return states
