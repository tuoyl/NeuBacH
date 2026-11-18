"""
geomag.py -- IGRF-14 field, McIlwain L and an approximate vertical cutoff
rigidity on a geographic grid at the satellite's geocentric radius.

* Gauss coefficients are read with the pip package ``ppigrf`` (IGRF14.shc,
  valid 1900-2030) and evaluated with a small vectorised spherical-harmonic
  routine written here (validated against ppigrf.igrf_gc in the self-test).
* McIlwain L: for each grid point the field line is traced (RK4) from the
  point (taken as the mirror point, B_m = |B|) to the conjugate mirror point,
  I = int sqrt(1 - B/B_m) ds, and L is obtained from Hilton's (1971)
  approximation of McIlwain's function with M = 0.311653 G R_E^3.
* Vertical cutoff rigidity:  COR = 14.5 GV / L^2  (Stoermer vertical cutoff
  written in terms of L; penumbra and the East-West effect are ignored).
  This is an approximation used only to build toy backgrounds and the
  "COR > 8 GV" good-time cut -- it is NOT the COR column of the HXMT EHK files.

Run as a script to (re)build the grid cache and print self-tests:
    python3 geomag.py
"""
import math
import os

import numpy as np

RE_IGRF = 6371.2            # km, IGRF reference radius
M_MCILWAIN = 31165.3        # nT R_E^3  (0.311653 G)
COR_K = 14.5                # GV
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "outputs", "geomag_grid.npz")


# ----------------------------------------------------------------------------
def load_coeffs(decimal_year, nmax=13):
    """Gauss coefficients g[n,m], h[n,m] (nT) linearly interpolated to decimal_year."""
    from ppigrf import ppigrf as pp
    g_df, h_df = pp.read_shc(pp.shc_fn)
    years = np.array([t.year + (t.dayofyear - 1) / 365.25 for t in g_df.index])
    g = np.zeros((nmax + 1, nmax + 1))
    h = np.zeros((nmax + 1, nmax + 1))
    for (n, m) in g_df.columns:
        if n <= nmax:
            g[n, m] = np.interp(decimal_year, years, g_df[(n, m)].values)
            h[n, m] = np.interp(decimal_year, years, h_df[(n, m)].values)
    return g, h


def field_gc(r_km, theta, phi, g, h, nmax=None):
    """
    IGRF field in geocentric spherical components (nT).
    r_km: radius, theta: colatitude (rad), phi: east longitude (rad); arrays of equal shape.
    Returns Br (outward), Btheta (southward), Bphi (eastward).
    """
    if nmax is None:
        nmax = g.shape[0] - 1
    ct, st = np.cos(theta), np.sin(theta)
    st_safe = np.where(np.abs(st) < 1e-10, 1e-10, st)
    ar = RE_IGRF / r_km
    Br = np.zeros_like(ct)
    Bt = np.zeros_like(ct)
    Bp = np.zeros_like(ct)
    cosm = [np.ones_like(phi)] + [np.cos(m * phi) for m in range(1, nmax + 1)]
    sinm = [np.zeros_like(phi)] + [np.sin(m * phi) for m in range(1, nmax + 1)]
    P = {(0, 0): np.ones_like(ct)}
    dP = {(0, 0): np.zeros_like(ct)}
    arn = ar * ar                      # (a/r)^(n+2) for n = 0
    for n in range(1, nmax + 1):
        arn = arn * ar
        if n == 1:
            P[(1, 1)] = st
            dP[(1, 1)] = ct
        else:
            f = math.sqrt((2 * n - 1) / (2.0 * n))
            P[(n, n)] = f * st * P[(n - 1, n - 1)]
            dP[(n, n)] = f * (st * dP[(n - 1, n - 1)] + ct * P[(n - 1, n - 1)])
        for m in range(0, n):
            den = math.sqrt(n * n - m * m)
            a = (2 * n - 1) / den
            Pn = a * ct * P[(n - 1, m)]
            dPn = a * (ct * dP[(n - 1, m)] - st * P[(n - 1, m)])
            if n - 2 >= m:
                b = math.sqrt((n - 1) ** 2 - m * m) / den
                Pn = Pn - b * P[(n - 2, m)]
                dPn = dPn - b * dP[(n - 2, m)]
            P[(n, m)] = Pn
            dP[(n, m)] = dPn
        for m in range(0, n + 1):
            gc = g[n, m] * cosm[m] + h[n, m] * sinm[m]
            Br += (n + 1) * arn * gc * P[(n, m)]
            Bt -= arn * gc * dP[(n, m)]
            if m > 0:
                Bp += arn * m * (g[n, m] * sinm[m] - h[n, m] * cosm[m]) * P[(n, m)] / st_safe
        # free memory of degree n-2
        for m in range(0, n - 1):
            P.pop((n - 2, m), None)
            dP.pop((n - 2, m), None)
    return Br, Bt, Bp


def field_xyz(x, y, z, g, h, nmax=None):
    """Field vector in geocentric Cartesian components (nT) at Cartesian positions (km)."""
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    Br, Bt, Bp = field_gc(r, theta, phi, g, h, nmax)
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    Bx = Br * st * cp + Bt * ct * cp - Bp * sp
    By = Br * st * sp + Bt * ct * sp + Bp * cp
    Bz = Br * ct - Bt * st
    return Bx, By, Bz


def hilton_L(I_re, Bm_nT, M=M_MCILWAIN):
    """McIlwain L from the integral invariant I (in R_E) and mirror field Bm (Hilton 1971)."""
    X = I_re ** 3 * Bm_nT / M
    a1, a2, a3 = 1.35047, 0.465376, 0.0475455
    Y = 1.0 + a1 * np.cbrt(X) + a2 * np.cbrt(X) ** 2 + a3 * X
    return np.cbrt(Y * M / Bm_nT)


def mcilwain_L(r_km, theta, phi, g, h, ds_km=40.0, nmax_trace=10, max_steps=4000, M=M_MCILWAIN):
    """
    McIlwain L for particles mirroring at the given points (vectorised field-line tracing).
    Returns (L, Bm_nT, I_re).
    """
    shape = np.shape(theta)
    r_km = np.broadcast_to(r_km, shape).astype(float).ravel()
    th = np.asarray(theta, dtype=float).ravel()
    ph = np.asarray(phi, dtype=float).ravel()
    x = r_km * np.sin(th) * np.cos(ph)
    y = r_km * np.sin(th) * np.sin(ph)
    z = r_km * np.cos(th)
    Bx, By, Bz = field_xyz(x, y, z, g, h)
    Bm = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    def unit(xx, yy, zz):
        bx, by, bz = field_xyz(xx, yy, zz, g, h, nmax_trace)
        b = np.sqrt(bx * bx + by * by + bz * bz)
        return bx / b, by / b, bz / b, b

    # direction along the field line in which |B| decreases (towards the magnetic equator)
    ux, uy, uz, _ = unit(x, y, z)
    eps = 5.0
    _, _, _, bplus = unit(x + eps * ux, y + eps * uy, z + eps * uz)
    _, _, _, bminus = unit(x - eps * ux, y - eps * uy, z - eps * uz)
    sgn = np.where(bplus < bminus, 1.0, -1.0)
    at_min = (bplus >= Bm) & (bminus >= Bm)        # already at the field minimum: I = 0

    n = len(x)
    I = np.zeros(n)
    active = np.where(~at_min)[0]
    px, py, pz = x.copy(), y.copy(), z.copy()
    f_prev = np.zeros(n)                            # integrand sqrt(1 - B/Bm) at the previous node (0 at mirror)
    b_prev = Bm.copy()
    # Bm evaluated with the truncated expansion, for a consistent B/Bm along the trace
    _, _, _, Bm_tr = unit(x, y, z)
    b_prev = Bm_tr.copy()
    for _ in range(max_steps):
        if active.size == 0:
            break
        a = active
        s = sgn[a] * ds_km
        k1x, k1y, k1z, _ = unit(px[a], py[a], pz[a])
        k2x, k2y, k2z, _ = unit(px[a] + 0.5 * s * k1x, py[a] + 0.5 * s * k1y, pz[a] + 0.5 * s * k1z)
        k3x, k3y, k3z, _ = unit(px[a] + 0.5 * s * k2x, py[a] + 0.5 * s * k2y, pz[a] + 0.5 * s * k2z)
        k4x, k4y, k4z, _ = unit(px[a] + s * k3x, py[a] + s * k3y, pz[a] + s * k3z)
        nx = px[a] + s * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        ny = py[a] + s * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        nz = pz[a] + s * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
        _, _, _, bn = unit(nx, ny, nz)
        ratio = bn / Bm_tr[a]
        fn = np.sqrt(np.clip(1.0 - ratio, 0.0, None))
        done = ratio >= 1.0
        # full steps for those not done
        I[a[~done]] += 0.5 * (f_prev[a[~done]] + fn[~done]) * ds_km
        # partial last step: linear interpolation of B to the crossing B = Bm
        if done.any():
            bp = b_prev[a[done]]
            frac = np.clip((Bm_tr[a[done]] - bp) / np.where(bn[done] - bp == 0, 1.0, bn[done] - bp), 0.0, 1.0)
            I[a[done]] += 0.5 * f_prev[a[done]] * frac * ds_km
        px[a], py[a], pz[a] = nx, ny, nz
        f_prev[a] = fn
        b_prev[a] = bn
        active = a[~done]
    I_re = I / RE_IGRF
    L = hilton_L(I_re, Bm, M)
    return L.reshape(shape), Bm.reshape(shape), I_re.reshape(shape)


# ----------------------------------------------------------------------------
def build_grid(a_km, epochs=(2017.5, 2020.5, 2023.5, 2026.5), dlon=3.0, dlat=1.5, latmax=45.0, verbose=True):
    """|B| (full degree 13), L and COR on a (epoch, lat, lon) grid at geocentric radius a_km."""
    import time
    lon = np.arange(0.0, 360.0 + 0.5 * dlon, dlon)              # includes 360 for wrap interpolation
    lat = np.arange(-latmax, latmax + 0.5 * dlat, dlat)         # geocentric latitude
    LON, LAT = np.meshgrid(np.radians(lon), np.radians(lat))
    Bmag = np.empty((len(epochs),) + LON.shape)
    Lsh = np.empty_like(Bmag)
    for k, ep in enumerate(epochs):
        t0 = time.time()
        g, h = load_coeffs(ep)
        L, Bm, _ = mcilwain_L(a_km, np.pi / 2 - LAT, LON, g, h)
        Bmag[k], Lsh[k] = Bm, L
        if verbose:
            print(f"  geomag grid epoch {ep}: {time.time() - t0:.1f} s, Bmin = {Bm.min():.0f} nT, "
                  f"L range {L.min():.3f}-{L.max():.3f}")
    cor = COR_K / Lsh ** 2
    return dict(epochs=np.array(epochs), lon=lon, lat=lat, B=Bmag, L=Lsh, COR=cor, a_km=a_km)


def get_grid(a_km, rebuild=False):
    """Load the cached grid (or build it)."""
    if os.path.exists(CACHE) and not rebuild:
        d = dict(np.load(CACHE))
        if abs(float(d["a_km"]) - a_km) < 1e-6:
            return d
    d = build_grid(a_km)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **d)
    return d


class GridInterp:
    """Trilinear interpolation (decimal year, geocentric lat, lon) of the cached grids."""

    def __init__(self, grid):
        self.g = grid
        self.ep, self.lon, self.lat = grid["epochs"], grid["lon"], grid["lat"]

    def __call__(self, name, dec_year, lat_deg, lon_deg):
        v = self.g[name]
        ep, lon, lat = self.ep, self.lon, self.lat
        fe = np.clip((np.asarray(dec_year) - ep[0]) / (ep[1] - ep[0]), 0.0, len(ep) - 1.000001)
        fy = np.clip((np.asarray(lat_deg) - lat[0]) / (lat[1] - lat[0]), 0.0, len(lat) - 1.000001)
        fx = np.clip((np.asarray(lon_deg) % 360.0) / (lon[1] - lon[0]), 0.0, len(lon) - 1.000001)
        ie, iy, ix = fe.astype(np.int64), fy.astype(np.int64), fx.astype(np.int64)
        te, ty, tx = fe - ie, fy - iy, fx - ix
        out = 0.0
        for de, we in ((0, 1 - te), (1, te)):
            for dy, wy in ((0, 1 - ty), (1, ty)):
                for dx, wx in ((0, 1 - tx), (1, tx)):
                    out = out + we * wy * wx * v[ie + de, iy + dy, ix + dx]
        return out


# ----------------------------------------------------------------------------
def self_test():
    import time
    from datetime import datetime
    import ppigrf
    rng = np.random.default_rng(3)
    n = 2000
    r = rng.uniform(6500, 9000, n)
    th = np.degrees(np.arccos(rng.uniform(-1, 1, n)))
    ph = rng.uniform(0, 360, n)
    g, h = load_coeffs(2022.0)
    Br, Bt, Bp = field_gc(r, np.radians(th), np.radians(ph), g, h)
    br, bt, bp = ppigrf.igrf_gc(r, th, ph, datetime(2022, 1, 1))
    print("max |dB| vs ppigrf (nT):", float(np.abs(Br - br[0]).max()), float(np.abs(Bt - bt[0]).max()),
          float(np.abs(Bp - bp[0]).max()))
    # dipole test of the L-shell tracer: pure axial dipole with B0 = M_MCILWAIN -> L = r/cos^2(lat)
    gd = np.zeros((14, 14))
    hd = np.zeros((14, 14))
    gd[1, 0] = -M_MCILWAIN
    lat = np.radians(np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]))
    rr = np.full_like(lat, 6921.0)
    t0 = time.time()
    L, Bm, I = mcilwain_L(rr, np.pi / 2 - lat, np.zeros_like(lat), gd, hd)
    Lexact = (rr / RE_IGRF) / np.cos(lat) ** 2
    print("dipole test L (traced):", np.round(L, 4), " exact:", np.round(Lexact, 4),
          " max rel err:", float(np.abs(L / Lexact - 1).max()), f"({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    import common as C
    self_test()
    grid = get_grid(C.A_ORBIT, rebuild=True)
    print("grid cached at", CACHE)
    k = 1
    j = np.unravel_index(grid["B"][k].argmin(), grid["B"][k].shape)
    print("epoch", grid["epochs"][k], ": min |B| =", grid["B"][k].min(), "nT at lon", grid["lon"][j[1]],
          "lat", grid["lat"][j[0]], "; COR range", grid["COR"][k].min(), grid["COR"][k].max())
