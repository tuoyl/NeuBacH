"""
common.py -- shared physics / geometry for the forecast
"Earth-shadowed dark-matter-wind orbital modulation with Insight-HXMT".

Everything here is a FORECAST TOOL (simulation), not an analysis of HXMT data.

Contents
--------
1. constants and adjustable parameters
2. Standard Halo Model in the observer frame (flux tables by quadrature)
3. Earth shadow: opaque Earth and partially transparent Earth (PREM + atmosphere)
4. Sun / Earth-velocity / frame conversions
5. mean circular orbit with J2 secular rates
6. small numerical helpers (bilinear table lookup, weighted statistics)

Units: km, s, km/s, degrees only at the I/O boundary (radians internally),
masses in GeV, cross sections in cm^2, column densities in g/cm^2.
"""
import math
import numpy as np

# ----------------------------------------------------------------------------
# 1. constants and adjustable parameters
# ----------------------------------------------------------------------------
GEV_PER_G = 5.609588603e23      # 1 g = 5.6096e23 GeV/c^2
KM_TO_CM = 1.0e5
R_E = 6371.0                    # km, mean Earth radius (shadow geometry, PREM)
R_EQ = 6378.137                 # km, equatorial radius (J2 secular rates)
MU_E = 398600.4418              # km^3 s^-2
J2 = 1.0826267e-3
V_ESC_EARTH = 11.2              # km/s, floor for "stopped" particles

# Halo (Baxter et al. 2021, arXiv:2105.00599, Table 1 / Eq. 14)
V0_DEFAULT = 238.0              # km/s, local standard of rest speed (= sqrt(2) sigma)
VESC_DEFAULT = 544.0            # km/s
RHO_DM = 0.3                    # GeV cm^-3
V_PEC = np.array([11.1, 12.24, 7.25])   # solar peculiar velocity (U,V,W) km/s
V_EARTH_ORB = 29.8              # km/s, mean orbital speed of the Earth

# HXMT nominal orbit (mission papers: 550 km, 43 deg) -- mean circular orbit
A_ORBIT = 6921.0                # km  (R_E + 550 km)
INC_DEG = 43.0
T_START = "2017-06-15T00:00:00"  # launch day (UTC)
T_END = "2026-11-01T00:00:00"

# Anchor of the mean orbit: the CelesTrak TLE retrieved 2026-11-18 (see README)
TLE_EPOCH_JD = 2461362.5 + 0.59089513   # 2026-11-18 14:10:53 UTC
TLE_RAAN_DEG = 42.5245
TLE_ARGLAT_DEG = 0.085                  # argp + true anomaly at epoch (ascending node)

# rotation matrix equatorial (ICRS) -> Galactic (Hipparcos definition)
A_G = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
                [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])


def gal_to_eq(vec_gal):
    """Galactic Cartesian (x->GC, y->l=90, z->NGP) to equatorial Cartesian."""
    return A_G.T @ np.asarray(vec_gal, dtype=float)


def eq_to_gal(vec_eq):
    return A_G @ np.asarray(vec_eq, dtype=float)


def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def vec_to_radec(v):
    v = np.asarray(v, dtype=float)
    n = np.sqrt((v ** 2).sum(axis=0))
    ra = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    dec = np.degrees(np.arcsin(v[2] / n))
    return ra, dec


def vsun_eq(v0=V0_DEFAULT, vpec=V_PEC):
    """Velocity of the Sun w.r.t. the Galactic rest frame, equatorial Cartesian, km/s."""
    vg = np.array([vpec[0], v0 + vpec[1], vpec[2]])
    return gal_to_eq(vg)


# ----------------------------------------------------------------------------
# 2. Standard Halo Model in the observer frame
# ----------------------------------------------------------------------------
def shm_norm(v0, vesc):
    z = vesc / v0
    return math.erf(z) - 2.0 * z * math.exp(-z * z) / math.sqrt(math.pi)


def f_gal(vg2, v0, vesc):
    """Truncated Maxwellian f(v_gal), normalised to 1 (units (km/s)^-3); vg2 = |v_gal|^2."""
    out = np.exp(-vg2 / v0 ** 2) / ((math.pi * v0 ** 2) ** 1.5 * shm_norm(v0, vesc))
    return np.where(vg2 <= vesc ** 2, out, 0.0)


class FluxTable:
    """
    Angular/speed structure of the DM flux for an observer moving with speed
    v_obs through the SHM.

    mu  = cosine of the angle between the ARRIVAL direction n (= -w/|w|) and the
          apex a (= direction of the observer's motion).  mu=+1: from the apex.
    w   = speed in the observer frame.

    G[iw, imu] = int_{w_i}^{w_max} w^3 f(w,mu) dw   (flux per steradian above w_i, km/s)
    H[iw, imu] = int_{w_i}^{w_max} w^2 f(w,mu) dw   (number density per steradian above w_i)

    so that 2*pi*int G[0,:] dmu = <w> (mean speed = total flux / n)
    and     2*pi*int H[0,:] dmu = 1.
    """

    def __init__(self, vobs, v0=V0_DEFAULT, vesc=VESC_DEFAULT, nw=1701, nmu=1441):
        self.vobs, self.v0, self.vesc = float(vobs), float(v0), float(vesc)
        self.wmax = vesc + vobs
        self.w = np.linspace(0.0, self.wmax, nw)
        self.mu = np.linspace(-1.0, 1.0, nmu)
        self.dmu = self.mu[1] - self.mu[0]
        w = self.w[:, None]
        vg2 = w ** 2 + vobs ** 2 - 2.0 * w * vobs * self.mu[None, :]
        f = f_gal(vg2, v0, vesc)
        dw = self.w[1] - self.w[0]
        self.G = self._cum_from_top(w ** 3 * f, dw)
        self.H = self._cum_from_top(w ** 2 * f, dw)
        self.mean_speed = 2.0 * np.pi * np.trapezoid(self.G[0], self.mu)
        self.norm_check = 2.0 * np.pi * np.trapezoid(self.H[0], self.mu)
        # first moment of the number-density angular distribution, c1 = <n.a>
        self.c1 = 2.0 * np.pi * np.trapezoid(self.H[0] * self.mu, self.mu)

    @staticmethod
    def _cum_from_top(y, dw):
        """trapezoidal cumulative integral from the top of the w grid down to each w_i."""
        seg = 0.5 * (y[1:] + y[:-1]) * dw
        out = np.zeros_like(y)
        out[:-1] = np.cumsum(seg[::-1], axis=0)[::-1]
        return out

    def at_w(self, table, wmin):
        """table (G or H) interpolated in w at wmin (scalar or array) -> (..., nmu)."""
        wmin = np.clip(np.atleast_1d(np.asarray(wmin, dtype=float)), 0.0, self.wmax)
        x = wmin / (self.w[1] - self.w[0])
        i0 = np.minimum(x.astype(int), len(self.w) - 2)
        fr = (x - i0)[:, None]
        return table[i0] * (1.0 - fr) + table[i0 + 1] * fr

    def interp_mu(self, rows, mu):
        """rows: (nrow, nmu) tabulated on self.mu; mu: (nrow, ...) -> values, linear interp."""
        x = (np.clip(mu, -1.0, 1.0) + 1.0) / self.dmu
        i0 = np.minimum(x.astype(int), len(self.mu) - 2)
        fr = x - i0
        idx = np.arange(rows.shape[0]).reshape((-1,) + (1,) * (mu.ndim - 1))
        return rows[idx, i0] * (1.0 - fr) + rows[idx, i0 + 1] * fr

    def flux_above(self, wthr):
        """Total (4pi) flux/n above speed threshold wthr, km/s."""
        g = self.at_w(self.G, wthr)[0]
        return 2.0 * np.pi * np.trapezoid(g, self.mu)

    def density_above(self, wthr):
        h = self.at_w(self.H, wthr)[0]
        return 2.0 * np.pi * np.trapezoid(h, self.mu)


# ----------------------------------------------------------------------------
# 3. Earth shadow
# ----------------------------------------------------------------------------
# PREM (Dziewonski & Anderson 1981) density polynomials, x = r / 6371 km, g/cm^3
_PREM = [
    (0.0, 1221.5, (13.0885, 0.0, -8.8381, 0.0)),
    (1221.5, 3480.0, (12.5815, -1.2638, -3.6426, -5.5281)),
    (3480.0, 5701.0, (7.9565, -6.4761, 5.5283, -3.0807)),
    (5701.0, 5771.0, (5.3197, -1.4836, 0.0, 0.0)),
    (5771.0, 5971.0, (11.2494, -8.0298, 0.0, 0.0)),
    (5971.0, 6151.0, (7.1089, -3.8045, 0.0, 0.0)),
    (6151.0, 6346.6, (2.6910, 0.6924, 0.0, 0.0)),
    (6346.6, 6356.0, (2.900, 0.0, 0.0, 0.0)),
    (6356.0, 6368.0, (2.600, 0.0, 0.0, 0.0)),
    (6368.0, 6371.0, (1.020, 0.0, 0.0, 0.0)),
]
ATM_RHO0 = 1.225e-3     # g/cm^3 sea-level air density
ATM_H = 7.0             # km, isothermal scale height (crude above ~100 km, where it no longer matters)
ATM_TOP = 200.0         # km, where the toy atmosphere is cut


def prem_density(r_km):
    """PREM density (g/cm^3) for r <= 6371 km; exponential toy atmosphere above."""
    r = np.asarray(r_km, dtype=float)
    rho = np.zeros_like(r)
    x = r / 6371.0
    for r0, r1, c in _PREM:
        m = (r >= r0) & (r < r1)
        rho = np.where(m, c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3, rho)
    atm = (r >= R_E) & (r < R_E + ATM_TOP)
    rho = np.where(atm, ATM_RHO0 * np.exp(-np.clip(r - R_E, 0.0, ATM_TOP) / ATM_H), rho)
    return rho


def chord_column(b_km, n=4000, include_atm=True):
    """
    Column density (g/cm^2) along a straight line with impact parameter b (km)
    through the Earth (+ toy atmosphere), X(b) = 2 int_0^smax rho(sqrt(b^2+s^2)) ds.
    """
    b = np.atleast_1d(np.asarray(b_km, dtype=float))
    rtop = R_E + (ATM_TOP if include_atm else 0.0)
    smax = np.sqrt(np.clip(rtop ** 2 - b ** 2, 0.0, None))
    # Gauss-Legendre in s on [0, smax] is poor across density jumps; use a fine midpoint rule
    t = (np.arange(n) + 0.5) / n
    s = smax[:, None] * t[None, :]
    r = np.sqrt(b[:, None] ** 2 + s ** 2)
    rho = prem_density(r)
    if not include_atm:
        rho = np.where(r < R_E, rho, 0.0)
    return 2.0 * (rho.sum(axis=1) * smax / n) * KM_TO_CM


def earth_half_angle(a_km=A_ORBIT, r_block=R_E):
    """Half-angle of the (opaque) Earth disc seen from geocentric distance a_km (radians)."""
    return math.asin(r_block / a_km)


class Shadow:
    """
    Quadrature over the cone of arrival directions that intersect the Earth
    (and, optionally, the atmosphere annulus), for a satellite at geocentric
    distance a_km.

    alpha = angle of the arrival direction from the nadir; beta = azimuth about
    the nadir, measured from the plane containing the apex.  For a satellite
    whose zenith makes the angle psi with the apex:
        mu = n.a = -cos(psi) cos(alpha) + sin(psi) sin(alpha) cos(beta)
    """

    def __init__(self, a_km=A_ORBIT, n_alpha=96, n_beta=128, n_alpha_atm=24, with_atm=False):
        self.a_km = a_km
        self.theta_E = earth_half_angle(a_km)
        segs = [(0.0, self.theta_E, n_alpha)]
        if with_atm:
            th_top = math.asin((R_E + ATM_TOP) / a_km)
            segs.append((self.theta_E, th_top, n_alpha_atm))
        al, wt = [], []
        for a0, a1, n in segs:
            x, w = np.polynomial.legendre.leggauss(n)       # nodes in cos(alpha)
            c0, c1 = math.cos(a1), math.cos(a0)
            ca = 0.5 * (c1 - c0) * x + 0.5 * (c1 + c0)
            al.append(np.arccos(ca))
            wt.append(0.5 * (c1 - c0) * w)
        self.alpha = np.concatenate(al)
        self.w_alpha = np.concatenate(wt)                    # weights for d(cos alpha)
        self.beta = (np.arange(n_beta) + 0.5) * np.pi / n_beta   # [0, pi], doubled by symmetry
        self.w_beta = 2.0 * np.pi / n_beta
        self.b_km = a_km * np.sin(self.alpha)                # impact parameters
        self.column = chord_column(self.b_km)                # g/cm^2 (Earth + atmosphere)
        self.column_solid = chord_column(self.b_km, include_atm=False)

    def _mu(self, psi):
        psi = np.atleast_1d(psi)[:, None, None]
        al = self.alpha[None, :, None]
        be = self.beta[None, None, :]
        return -np.cos(psi) * np.cos(al) + np.sin(psi) * np.sin(al) * np.cos(be)

    def blocked(self, tab, psi, wthr=0.0, sigma_over_m=None, table="G"):
        """
        Flux (table='G', km/s) or number density (table='H') arriving from the
        Earth cone that does NOT reach the detector above the speed threshold.

        sigma_over_m = None  -> opaque solid Earth (everything inside theta_E blocked)
        sigma_over_m in cm^2/GeV -> straight-line continuous slowing down,
            w_final = w_initial * exp(-X * sigma/m) with X in GeV/cm^2 (heavy DM,
            contact cross section equal for all nuclei; Cappiello+ 2021 Eqs. 1-2);
            a particle is 'blocked' if w_final < max(wthr, V_ESC_EARTH).
        Returns array over psi.
        """
        T = tab.G if table == "G" else tab.H
        w_lo = max(float(wthr), 0.0)
        base = tab.at_w(T, w_lo)[0]                           # (nmu,)
        if sigma_over_m is None:
            solid = self.alpha <= self.theta_E + 1e-12
            rows = np.where(solid[:, None], base[None, :], 0.0)
        else:
            tau = self.column * GEV_PER_G * sigma_over_m
            w_need = np.minimum(max(w_lo, V_ESC_EARTH) * np.exp(np.minimum(tau, 50.0)), tab.wmax)
            w_need = np.maximum(w_need, w_lo)
            rows = base[None, :] - tab.at_w(T, w_need)        # (nalpha, nmu)
        mu = self._mu(psi)                                    # (npsi, nalpha, nbeta)
        out = np.empty(mu.shape[0])
        for i in range(mu.shape[0]):
            val = tab.interp_mu(rows, mu[i])                  # (nalpha, nbeta)
            out[i] = (val.sum(axis=1) * self.w_beta * self.w_alpha).sum()
        return out

    def transmitted_fraction(self, tab, psi, wthr=0.0, sigma_over_m=None):
        """Transmitted flux above wthr divided by the unshadowed flux above wthr."""
        tot = tab.flux_above(wthr)
        return 1.0 - self.blocked(tab, psi, wthr, sigma_over_m) / tot

    def ram_kernel(self, tab, psi, sigma_over_m=None):
        """
        First-order correction for the satellite's own velocity v_sat (|v_sat|<<w):
            Rate/n = Phi_E(psi) + (v_sat . a) * K(psi),
            K(psi) = c1 - (1/sin psi) int_cone h(mu) sin(alpha) cos(beta) dOmega
        with h the number-density angular distribution in the Earth frame
        (see the notes for the derivation).  Opaque Earth only unless sigma_over_m given.
        """
        psi = np.atleast_1d(psi).astype(float)
        base = tab.H[0]
        if sigma_over_m is None:
            solid = self.alpha <= self.theta_E + 1e-12
            rows = np.where(solid[:, None], base[None, :], 0.0)
        else:
            tau = self.column * GEV_PER_G * sigma_over_m
            w_need = np.minimum(V_ESC_EARTH * np.exp(np.minimum(tau, 50.0)), tab.wmax)
            rows = base[None, :] - tab.at_w(tab.H, w_need)
        mu = self._mu(psi)
        out = np.empty(len(psi))
        sa = np.sin(self.alpha)[:, None]
        cb = np.cos(self.beta)[None, :]
        for i in range(len(psi)):
            val = tab.interp_mu(rows, mu[i]) * sa * cb
            integ = (val.sum(axis=1) * self.w_beta * self.w_alpha).sum()
            s = math.sin(psi[i])
            out[i] = tab.c1 - (integ / s if s > 1e-6 else 0.0)
        # limits psi -> 0, pi: integral ~ sin(psi); fill by neighbour extrapolation
        bad = np.sin(psi) <= 1e-6
        if bad.any() and (~bad).any():
            out[bad] = np.interp(psi[bad], psi[~bad], out[~bad])
        return out


# ----------------------------------------------------------------------------
# 4. Sun, Earth velocity, time
# ----------------------------------------------------------------------------
def jd_from_iso(iso):
    """UTC ISO string -> Julian date (no leap-second care; fine for this forecast)."""
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0 + 2440587.5


def gmst_deg(jd):
    """Greenwich mean sidereal time (deg), IAU 1982 linear term (UT1 ~ UTC assumed)."""
    d = np.asarray(jd, dtype=float) - 2451545.0
    return (280.46061837 + 360.98564736629 * d) % 360.0


def sun_state(jd):
    """
    Low-precision Sun (Astronomical Almanac): returns dict with ecliptic
    longitude lam (rad), mean longitude Lmean (rad), unit vector to the Sun
    (equatorial), RA/Dec (rad), and the Earth's heliocentric velocity vector
    (equatorial, km/s) including first-order eccentricity.
    """
    d = np.asarray(jd, dtype=float) - 2451545.0
    Lm = np.radians((280.460 + 0.9856474 * d) % 360.0)
    g = np.radians((357.528 + 0.9856003 * d) % 360.0)
    lam = Lm + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4.0e-7 * d)
    sx = np.cos(lam)
    sy = np.cos(eps) * np.sin(lam)
    sz = np.sin(eps) * np.sin(lam)
    ra = np.arctan2(sy, sx) % (2 * np.pi)
    dec = np.arcsin(sz)
    # Earth heliocentric true longitude l = lam + pi; velocity of a Kepler orbit:
    # v = v_mean/sqrt(1-e^2) * (-sin l - e sin w, cos l + e cos w, 0) (ecliptic),
    # w = longitude of perihelion of the Earth's orbit.
    e = 0.016709
    wp = np.radians(102.94)
    l = lam + np.pi
    k = V_EARTH_ORB / math.sqrt(1.0 - e * e)
    vx_ecl = k * (-np.sin(l) - e * math.sin(wp))
    vy_ecl = k * (np.cos(l) + e * math.cos(wp))
    vx = vx_ecl
    vy = np.cos(eps) * vy_ecl
    vz = np.sin(eps) * vy_ecl
    return dict(lam=lam, Lmean=Lm, sun=np.array([sx, sy, sz]), ra=ra, dec=dec,
                v_earth=np.array([vx, vy, vz]))


# ----------------------------------------------------------------------------
# 5. mean circular orbit with J2 secular rates
# ----------------------------------------------------------------------------
def orbit_rates(a_km=A_ORBIT, inc_deg=INC_DEG):
    """
    Secular rates of a circular orbit under J2 (first order):
      n0       = sqrt(mu/a^3)
      dOmega/dt = -(3/2) J2 (R_eq/a)^2 n0 cos i
      du/dt     = n0 [1 + (3/2) J2 (R_eq/a)^2 (3 - 4 sin^2 i)]   (argument of latitude)
    Returns (n0, raan_dot, u_dot) in rad/s.
    """
    inc = math.radians(inc_deg)
    n0 = math.sqrt(MU_E / a_km ** 3)
    fac = 1.5 * J2 * (R_EQ / a_km) ** 2
    raan_dot = -fac * n0 * math.cos(inc)
    u_dot = n0 * (1.0 + fac * (3.0 - 4.0 * math.sin(inc) ** 2))
    return n0, raan_dot, u_dot


def orbit_angles(jd, a_km=A_ORBIT, inc_deg=INC_DEG):
    """Argument of latitude u and node Omega (rad) of the mean orbit anchored at the TLE epoch."""
    _, raan_dot, u_dot = orbit_rates(a_km, inc_deg)
    dt = (np.asarray(jd, dtype=float) - TLE_EPOCH_JD) * 86400.0
    Om = (math.radians(TLE_RAAN_DEG) + raan_dot * dt) % (2 * np.pi)
    u = (math.radians(TLE_ARGLAT_DEG) + u_dot * dt) % (2 * np.pi)
    return u, Om


def sat_vectors(u, Om, inc_deg=INC_DEG):
    """Unit position rhat and unit velocity vhat (inertial equatorial) of the circular orbit."""
    inc = math.radians(inc_deg)
    cu, su, cO, sO = np.cos(u), np.sin(u), np.cos(Om), np.sin(Om)
    ci, si = math.cos(inc), math.sin(inc)
    rhat = np.array([cu * cO - su * ci * sO, cu * sO + su * ci * cO, su * si])
    vhat = np.array([-su * cO - cu * ci * sO, -su * sO + cu * ci * cO, cu * si])
    return rhat, vhat


def orbit_normal(Om, inc_deg=INC_DEG):
    inc = math.radians(inc_deg)
    return np.array([math.sin(inc) * np.sin(Om), -math.sin(inc) * np.cos(Om),
                     math.cos(inc) * np.ones_like(Om)])


# ----------------------------------------------------------------------------
# 6. helpers
# ----------------------------------------------------------------------------
class Table2D:
    """Bilinear lookup on a regular (x, y) grid; values[ix, iy]."""

    def __init__(self, x, y, values):
        self.x0, self.dx, self.nx = x[0], x[1] - x[0], len(x)
        self.y0, self.dy, self.ny = y[0], y[1] - y[0], len(y)
        self.v = np.asarray(values, dtype=float)

    def __call__(self, x, y):
        fx = np.clip((np.asarray(x) - self.x0) / self.dx, 0.0, self.nx - 1.000001)
        fy = np.clip((np.asarray(y) - self.y0) / self.dy, 0.0, self.ny - 1.000001)
        ix = fx.astype(np.int64)
        iy = fy.astype(np.int64)
        tx = fx - ix
        ty = fy - iy
        v = self.v
        return ((1 - tx) * (1 - ty) * v[ix, iy] + tx * (1 - ty) * v[ix + 1, iy]
                + (1 - tx) * ty * v[ix, iy + 1] + tx * ty * v[ix + 1, iy + 1])


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def wrms(x, w):
    return float(np.sqrt((x * x * w).sum() / w.sum()))
