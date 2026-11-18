"""
nuisance.py -- linear background ("nuisance") models and exact Fisher marginalisation.

Data model per time bin i (live time tau_i, expected background rate B_i):

    rate_i = sum_k theta_k X_ik  +  S0 * T_i ,      Var(rate_i) = B_i / tau_i   (Poisson, B >> S)

Fisher information on S0 after marginalising all theta (Frisch-Waugh-Lovell):

    I = T'W T - T'W X (X'W X)^+ X'W T = || T_perp ||_W^2 ,   W_i = tau_i / B_i ,

where T_perp is the part of the template that is W-orthogonal to every nuisance
column.  X is split as [M, Z]:
  M : one-hot membership of each bin in a "cell" (e.g. geographic map cell,
      possibly x year, or pointing x visibility state) -> M'W M is diagonal and is
      eliminated analytically (weighted de-meaning inside each cell);
  Z : remaining sparse columns (one-hot bins of time-since-SAA, local solar time,
      beta angle, ..., linear-spline trend), handled through the dense Schur
      complement  G = Z'WZ - (M'WZ)' D^-1 (M'WZ),  pseudo-inverted by eigen-decomposition.

`Projector.residual(t)` returns the explicit vector T_perp, which is also what a
weighted least-squares fit would use as the optimal linear estimator:
    S0_hat = sum_i W_i T_perp,i y_i / sum_i W_i T_perp,i^2 .
"""
import numpy as np
import scipy.sparse as sp

# ----------------------------------------------------------------------------
# index builders (all return (index array, number of bins))
# ----------------------------------------------------------------------------
def map_cells(lon_deg, lat_rad, asc, cell_deg=5.0, latmax=45.0, split_asc=True):
    nlon = int(round(360.0 / cell_deg))
    nlat = int(round(2 * latmax / cell_deg))
    ix = np.minimum((lon_deg % 360.0 / cell_deg).astype(np.int64), nlon - 1)
    iy = np.clip(((np.degrees(lat_rad) + latmax) / cell_deg).astype(np.int64), 0, nlat - 1)
    idx = iy * nlon + ix
    n = nlon * nlat
    if split_asc:
        idx = idx + n * asc.astype(np.int64)
        n *= 2
    return idx, n


TSAA_EDGES_MIN = np.concatenate([np.arange(0, 30, 2), np.arange(30, 60, 5), np.arange(60, 120, 10),
                                 np.arange(120, 360, 30), np.arange(360, 1440, 120), [1440.0, 1e12]])


def tsaa_bins(t_since_saa_s):
    idx = np.searchsorted(TSAA_EDGES_MIN, t_since_saa_s / 60.0, side="right") - 1
    return np.clip(idx, 0, len(TSAA_EDGES_MIN) - 2), len(TSAA_EDGES_MIN) - 1


def lt_bins(lt_hours, nbin=24):
    return np.minimum((lt_hours / 24.0 * nbin).astype(np.int64), nbin - 1), nbin


def beta_bins(beta_deg, width=5.0, bmax=70.0):
    n = int(round(2 * bmax / width))
    return np.clip(((beta_deg + bmax) / width).astype(np.int64), 0, n - 1), n


def sunphase_beta_bins(sun_phase_deg, beta_deg, nphase=36, width=5.0, bmax=70.0):
    ib, nb = beta_bins(beta_deg, width, bmax)
    ip = np.minimum((sun_phase_deg / 360.0 * nphase).astype(np.int64), nphase - 1)
    return ib * nphase + ip, nb * nphase


def period_index(t_s, period_days):
    return (t_s / (period_days * 86400.0)).astype(np.int64)


def pointing_dir_bins(zen_deg, az_deg, nzen=12, naz=12):
    iz = np.minimum((zen_deg / 180.0 * nzen).astype(np.int64), nzen - 1)
    ia = np.minimum((az_deg / 360.0 * naz).astype(np.int64), naz - 1)
    return iz * naz + ia, nzen * naz


def earth_angle_bins(zen_deg, nbin=6):
    return np.minimum((zen_deg / 180.0 * nbin).astype(np.int64), nbin - 1), nbin


# ----------------------------------------------------------------------------
# sparse design blocks
# ----------------------------------------------------------------------------
def onehot(idx, n):
    N = len(idx)
    return sp.csr_matrix((np.ones(N), (np.arange(N), idx)), shape=(N, n))


def linear_spline(t_s, knot_days=30.0):
    """Hat-function (linear B-spline) basis in time with knots every knot_days."""
    x = t_s / (knot_days * 86400.0)
    i0 = np.floor(x).astype(np.int64)
    fr = x - i0
    n = int(i0.max()) + 2
    N = len(t_s)
    rows = np.concatenate([np.arange(N), np.arange(N)])
    cols = np.concatenate([i0, i0 + 1])
    vals = np.concatenate([1.0 - fr, fr])
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, n))


def compress(idx):
    """Re-label an index array to 0..K-1 using only the occupied values."""
    uniq, inv = np.unique(idx, return_inverse=True)
    return inv, len(uniq)


# ----------------------------------------------------------------------------
class Projector:
    """W-orthogonal projection onto the complement of span{M (cells), Z (sparse)}."""

    def __init__(self, W, cell=None, Z=None, rcond=1e-9):
        self.W = np.asarray(W, dtype=float)
        N = len(self.W)
        if cell is None:
            cell = np.zeros(N, dtype=np.int64)
        self.cell, self.ncell = compress(np.asarray(cell))
        self.sw = np.bincount(self.cell, weights=self.W, minlength=self.ncell)
        self.dinv = 1.0 / self.sw
        self.Z = None
        self.n_par = self.ncell
        if Z is not None and Z.shape[1] > 0:
            Z = Z.tocsr()
            keep = np.asarray(Z.getnnz(axis=0)).ravel() > 0     # drop empty columns
            Z = Z[:, np.where(keep)[0]]
            self.Z = Z
            self.ZtW = Z.T.multiply(self.W[None, :]).tocsr()    # (nz, N)
            M = onehot(self.cell, self.ncell)
            C = (self.ZtW @ M)                                  # (nz, ncell) = (M'WZ)'
            nz = Z.shape[1]
            if nz * self.ncell <= 8e7:
                Cd = C.toarray()
                G = (self.ZtW @ Z).toarray() - (Cd * self.dinv[None, :]) @ Cd.T
                self.C = Cd
            else:
                Cs = C.tocsr()
                G = (self.ZtW @ Z).toarray() - (Cs.multiply(self.dinv[None, :]) @ Cs.T).toarray()
                self.C = Cs
            G = 0.5 * (G + G.T)
            ev, V = np.linalg.eigh(G)
            good = ev > rcond * ev.max()
            self.rank_Z = int(good.sum())
            self.Gpinv = (V[:, good] / ev[good]) @ V[:, good].T
            self.n_par += self.rank_Z

    def residual(self, t):
        t = np.asarray(t, dtype=float)
        m = np.bincount(self.cell, weights=self.W * t, minlength=self.ncell) * self.dinv
        if self.Z is None:
            return t - m[self.cell]
        b = self.ZtW @ t - self.C @ m
        theta = self.Gpinv @ b
        mhat = m - self.dinv * (self.C.T @ theta)
        return t - mhat[self.cell] - self.Z @ theta

    def info(self, t):
        r = self.residual(t)
        return float((self.W * r * r).sum()), r
