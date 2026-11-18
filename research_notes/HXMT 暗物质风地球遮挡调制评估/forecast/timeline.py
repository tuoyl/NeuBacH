"""
timeline.py -- simulated mission timeline 2017-06-15 ... 2026-11-01 in fixed time bins.

For every bin (default 60 s) this module provides
  * the mean-circular-orbit state (u, Omega, unit position/velocity, lat, lon),
  * Sun-related quantities (local solar time, Sun elevation at the sub-satellite
    point, beta angle, orbital phase relative to the Sun, eclipse flag),
  * geomagnetic quantities from the IGRF-14 grid (|B|, McIlwain L, COR = 14.5/L^2),
  * a simple SAA flag (|B| below a threshold), time since SAA exit, time to next
    SAA entry, and exponentially filtered "dose" tracers used by the toy
    activation background,
  * a SIMULATED pointing schedule (random targets with a 70 deg Sun-avoidance
    cone) -- this is NOT the real HXMT observation history.

Nothing here is HXMT data.
"""
import math

import numpy as np

import common as C
import geomag

SAA_B_THRESHOLD_NT = 20500.0     # |B| at 550 km below which the satellite is "inside the SAA";
#                                  chosen so that the simulation gives ~8.7 passes/day of ~14 min
#                                  (HXMT papers: 8-9 passes/day, ~15 min each)
DT_DEFAULT = 60.0


def build(dt=DT_DEFAULT, a_km=C.A_ORBIT, inc_deg=C.INC_DEG, verbose=True):
    jd0 = C.jd_from_iso(C.T_START)
    jd1 = C.jd_from_iso(C.T_END)
    n = int(round((jd1 - jd0) * 86400.0 / dt))
    t = (np.arange(n) + 0.5) * dt                         # s since T_START (bin centres)
    jd = jd0 + t / 86400.0
    tl = dict(dt=dt, n=n, t=t, jd=jd, jd0=jd0, a_km=a_km, inc_deg=inc_deg)
    tl["dec_year"] = 2000.0 + (jd - 2451544.5) / 365.25

    # ---- orbit ----------------------------------------------------------
    u, Om = C.orbit_angles(jd, a_km, inc_deg)
    rhat, vhat = C.sat_vectors(u, Om, inc_deg)
    tl.update(u=u, Om=Om, rhat=rhat, vhat=vhat)
    tl["vsat_kms"] = math.sqrt(C.MU_E / a_km)
    lat = np.arcsin(np.clip(rhat[2], -1, 1))               # geocentric latitude (rad)
    ra_sat = np.arctan2(rhat[1], rhat[0])
    lon = (np.degrees(ra_sat) - C.gmst_deg(jd)) % 360.0     # east longitude (deg)
    tl.update(lat=lat, ra_sat=ra_sat % (2 * np.pi), lon=lon, asc=np.cos(u) > 0)

    # ---- Sun ------------------------------------------------------------
    sun = C.sun_state(jd)
    s = sun["sun"]
    tl["sun"] = s
    tl["sun_lam"] = sun["lam"]
    tl["sun_Lmean"] = sun["Lmean"]
    tl["v_earth"] = sun["v_earth"]
    tl["lt_hours"] = ((ra_sat - sun["ra"]) * 12.0 / np.pi + 12.0) % 24.0
    cs = (rhat * s).sum(axis=0)
    tl["sun_elev"] = np.degrees(np.arcsin(np.clip(cs, -1, 1)))
    theta_E = C.earth_half_angle(a_km)
    tl["theta_E"] = theta_E
    tl["eclipse"] = cs < -math.cos(theta_E)
    hn = C.orbit_normal(Om, inc_deg)
    tl["beta"] = np.degrees(np.arcsin(np.clip((hn * s).sum(axis=0), -1, 1)))
    # orbital phase of the satellite relative to the projection of the Sun on the orbit plane
    inc = math.radians(inc_deg)
    ex = np.array([np.cos(Om), np.sin(Om), np.zeros_like(Om)])
    ey = np.array([-math.cos(inc) * np.sin(Om), math.cos(inc) * np.cos(Om), math.sin(inc) * np.ones_like(Om)])
    u_sun = np.arctan2((ey * s).sum(axis=0), (ex * s).sum(axis=0))
    tl["sun_phase"] = np.degrees((u - u_sun) % (2 * np.pi))
    tl["doy_phase"] = (sun["Lmean"] % (2 * np.pi))          # annual phase (mean solar longitude)

    # ---- geomagnetic ----------------------------------------------------
    grid = geomag.get_grid(a_km)
    gi = geomag.GridInterp(grid)
    latd = np.degrees(lat)
    tl["Bmag"] = gi("B", tl["dec_year"], latd, lon).astype(np.float32)
    tl["L"] = gi("L", tl["dec_year"], latd, lon).astype(np.float32)
    tl["COR"] = (geomag.COR_K / tl["L"].astype(float) ** 2).astype(np.float32)

    # ---- SAA ------------------------------------------------------------
    saa = tl["Bmag"] < SAA_B_THRESHOLD_NT
    tl["saa"] = saa
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(saa, idx, -10 ** 9))
    tl["t_since_saa"] = np.where(last < 0, 1e9, (idx - last) * dt)
    nxt = np.minimum.accumulate(np.where(saa, idx, 10 ** 9)[::-1])[::-1]
    tl["t_to_saa"] = np.where(nxt >= 10 ** 9, 1e9, (nxt - idx) * dt)
    # toy "dose rate" inside the SAA and exponentially filtered tracers (activation)
    from scipy.signal import lfilter
    dose = np.clip((SAA_B_THRESHOLD_NT - tl["Bmag"].astype(float)) / SAA_B_THRESHOLD_NT, 0.0, None) ** 1.5
    tl["dose"] = dose
    for name, tau in (("act_6min", 360.0), ("act_36min", 2160.0), ("act_13h", 46800.0)):
        k = math.exp(-dt / tau)
        tl[name] = lfilter([1.0], [1.0, -k], dose)
    if verbose:
        ent = np.count_nonzero(saa[1:] & ~saa[:-1])
        days = n * dt / 86400.0
        print(f"timeline: {n} bins of {dt:.0f} s ({days:.0f} d); SAA fraction {saa.mean():.4f}, "
              f"{ent / days:.2f} SAA passes/day, mean pass {saa.sum() * dt / max(ent, 1) / 60:.1f} min")
    return tl


# ----------------------------------------------------------------------------
def toy_background(tl):
    """
    Relative background rate b(t) (dimensionless, to be normalised by the caller):
      60 % prompt cosmic-ray/albedo part  ~ ((COR + 1 GV)/10 GV)^-1.1
       5 % activation with tau =  6 min
      15 % activation with tau = 36 min   (e.g. 128-I, T1/2 = 25 min)
      20 % activation with tau = 13 h
    The fractions refer to the mean over non-SAA time.  This is a toy model whose
    only purpose is to give realistic Poisson weights and realistic nuisance
    structure; it is not a fit to HXMT background data.
    """
    ok = ~tl["saa"]
    g = ((tl["COR"].astype(float) + 1.0) / 10.0) ** (-1.1)
    b = 0.60 * g / g[ok].mean()
    for name, frac in (("act_6min", 0.05), ("act_36min", 0.15), ("act_13h", 0.20)):
        b = b + frac * tl[name] / tl[name][ok].mean()
    return b


# ----------------------------------------------------------------------------
def pointing_schedule(tl, seed=11, plane_fraction=0.6, median_days=1.0, sigma_ln=0.9, sun_avoid_deg=70.0):
    """
    SIMULATED sequence of inertially fixed pointings (NOT the real HXMT history).
    Durations: log-normal (median 1 d), clipped to [0.07, 20] d.
    Targets: 60 % near the Galactic plane (uniform l, b ~ N(0, 6 deg)), 40 % isotropic,
    always at least 70 deg from the Sun at mid-observation.
    Adds per-bin arrays: obs index, target elevation above the Earth limb (ELV),
    angle between the pointing axis and the local zenith, and its azimuth in the
    local (north, east) frame.
    """
    rng = np.random.default_rng(seed)
    total = tl["n"] * tl["dt"] / 86400.0
    starts, vecs = [], []
    tcur = 0.0
    while tcur < total:
        dur = float(np.clip(rng.lognormal(math.log(median_days), sigma_ln), 0.07, 20.0))
        jd_mid = tl["jd0"] + tcur + 0.5 * dur
        s = C.sun_state(np.array([jd_mid]))["sun"][:, 0]
        for _ in range(200):
            if rng.random() < plane_fraction:
                l = rng.uniform(0, 2 * np.pi)
                b = math.radians(rng.normal(0.0, 6.0))
                vg = np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)])
                v = C.gal_to_eq(vg)
            else:
                z = rng.uniform(-1, 1)
                ph = rng.uniform(0, 2 * np.pi)
                v = np.array([math.sqrt(1 - z * z) * math.cos(ph), math.sqrt(1 - z * z) * math.sin(ph), z])
            if math.degrees(math.acos(np.clip(v @ s, -1, 1))) >= sun_avoid_deg:
                break
        starts.append(tcur)
        vecs.append(v)
        tcur += dur
    starts = np.array(starts)
    vecs = np.array(vecs).T                                    # (3, n_obs)
    obs = np.searchsorted(starts * 86400.0, tl["t"], side="right") - 1
    p = vecs[:, obs]                                           # (3, n)
    r = tl["rhat"]
    cz = np.clip((p * r).sum(axis=0), -1, 1)                   # cos(angle pointing - zenith)
    zen = np.degrees(np.arccos(cz))
    elv = (180.0 - zen) - math.degrees(tl["theta_E"])          # target elevation above the Earth limb
    # local east/north
    east = np.array([-r[1], r[0], np.zeros_like(r[0])])
    east /= np.sqrt((east ** 2).sum(axis=0))
    north = np.array([r[1] * east[2] - r[2] * east[1], r[2] * east[0] - r[0] * east[2], r[0] * east[1] - r[1] * east[0]])
    az = np.degrees(np.arctan2((p * east).sum(axis=0), (p * north).sum(axis=0))) % 360.0
    tl.update(obs=obs.astype(np.int32), n_obs=len(starts), obs_start_days=starts, obs_vec=vecs,
              elv=elv.astype(np.float32), point_zen=zen.astype(np.float32), point_az=az.astype(np.float32))
    return tl
