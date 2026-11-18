#!/usr/bin/env python3
"""
s2_orbit_template.py -- Tasks 2 and 3 of the forecast.

  * checks the present orbit of Insight-HXMT (NORAD 42758) from the single
    current TLE (downloads/hxmt_tle_42758.txt) with SGP4, and compares with the
    mean circular J2 orbit used everywhere else in this forecast;
  * verifies numerically the coordinator's point P1 (decomposition of r.a and the
    nodal precession period);
  * builds the opaque-Earth DM-rate template T(u, Omega), splits it into the
    Omega-independent (latitude-degenerate) and Omega-dependent parts, and
    tabulates the orbital modulation depth versus node and season, the annual
    modulation, and the ram effect of the satellite velocity.

Outputs
  outputs/s2_summary.json, outputs/s2_depth_vs_node.csv, outputs/s2_annual.csv,
  outputs/s2_template_uOmega.npz
  figures/fig2_template_u_Omega.png, figures/fig3_modulation_depth_year.png

Run:  python3 s2_orbit_template.py     (about 30 s)
Simulation (forecast) only -- no HXMT data are used.
"""
import json
import math
import os

import numpy as np

import common as C
import plotstyle as P
from template import WindTemplate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
TLE_FILE = os.path.join(HERE, "downloads", "hxmt_tle_42758.txt")


# ----------------------------------------------------------------------------
def tle_check():
    """Decode the current TLE with SGP4 (data only: the file is parsed, never executed)."""
    res = {"tle_file_present": os.path.exists(TLE_FILE)}
    if not res["tle_file_present"]:
        return res
    try:
        from sgp4.api import Satrec
    except Exception as exc:                         # pragma: no cover
        res["error"] = f"sgp4 not available: {exc}"
        return res
    with open(TLE_FILE, "r", encoding="ascii", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    l1 = [ln for ln in lines if ln.startswith("1 ")][0]
    l2 = [ln for ln in lines if ln.startswith("2 ")][0]

    def cks(line):
        s = sum(int(ch) for ch in line[:68] if ch.isdigit()) + line[:68].count("-")
        return s % 10 == int(line[68])

    res["checksum_ok"] = bool(cks(l1) and cks(l2))
    sat = Satrec.twoline2rv(l1, l2)
    rk = sat.radiusearthkm
    res.update(
        epoch_jd=sat.jdsatepoch + sat.jdsatepochF,
        inclination_deg=math.degrees(sat.inclo), raan_deg=math.degrees(sat.nodeo),
        eccentricity=sat.ecco, mean_motion_rev_day=sat.no_kozai * 1440 / (2 * math.pi),
        a_km=sat.a * rk, apogee_alt_km=sat.alta * rk, perigee_alt_km=sat.altp * rk,
        mean_alt_above_equatorial_radius_km=sat.a * rk - C.R_EQ,
        mean_alt_above_mean_radius_km=sat.a * rk - C.R_E,
        sgp4_nodedot_deg_day=math.degrees(sat.nodedot) * 1440,
        sgp4_node_period_days=360.0 / abs(math.degrees(sat.nodedot) * 1440),
        sgp4_arglat_period_min=2 * math.pi / (sat.mdot + sat.argpdot),
    )
    # compare SGP4 positions over 2 days with a mean circular J2 orbit of the same a, i
    a_now = sat.a * rk
    inc_now = math.degrees(sat.inclo)
    n0, rd, ud = C.orbit_rates(a_now, inc_now)
    res["j2_formula_nodedot_deg_day_at_tle_a"] = math.degrees(rd) * 86400
    tmin = np.arange(0.0, 2880.0, 1.0)
    e, r, v = sat.sgp4_array(np.full(tmin.shape, sat.jdsatepoch), sat.jdsatepochF + tmin / 1440.0)
    r = np.asarray(r)
    rr = np.sqrt((r ** 2).sum(axis=1))
    res["sgp4_radius_min_max_km"] = [float(rr.min()), float(rr.max())]
    u0 = math.radians(sat.argpo * 180 / math.pi) + sat.mo      # crude start value
    # fit the phase offset of the circular model to SGP4 (one free constant)
    best = None
    for du in np.linspace(-0.02, 0.02, 401):
        u = u0 + du + ud * tmin * 60.0
        Om = sat.nodeo + rd * tmin * 60.0
        rh, _ = C.sat_vectors(u, Om, inc_now)
        sep = np.degrees(np.arccos(np.clip((rh.T * (r / rr[:, None])).sum(axis=1), -1, 1)))
        if best is None or sep.max() < best[0]:
            best = (float(sep.max()), float(np.sqrt((sep ** 2).mean())))
    res["circular_model_vs_sgp4_max_sep_deg_2days"] = best[0]
    res["circular_model_vs_sgp4_rms_sep_deg_2days"] = best[1]
    res["earth_half_angle_deg_at_tle_alt"] = math.degrees(math.asin(C.R_E / a_now))
    return res


# ----------------------------------------------------------------------------
def p1_check(alpha, delta, inc):
    """Numerical identity check of the r.a decomposition (P1)."""
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 2 * np.pi, 100000)
    Om = rng.uniform(0, 2 * np.pi, 100000)
    rh, _ = C.sat_vectors(u, Om, math.degrees(inc))
    a = np.array([math.cos(delta) * math.cos(alpha), math.cos(delta) * math.sin(alpha), math.sin(delta)])
    lhs = (rh * a[:, None]).sum(axis=0)
    t_node = np.cos(u) * math.cos(delta) * np.cos(Om - alpha) + np.sin(u) * math.cos(inc) * math.cos(delta) * np.sin(alpha - Om)
    t_lat = np.sin(u) * math.sin(inc) * math.sin(delta)
    lat = np.arcsin(rh[2])
    return dict(max_abs_residual=float(np.abs(lhs - (t_node + t_lat)).max()),
                max_abs_diff_latterm_vs_sinlat_sindelta=float(np.abs(t_lat - np.sin(lat) * math.sin(delta)).max()),
                rms_node_term=float(np.sqrt((t_node ** 2).mean())), rms_lat_term=float(np.sqrt((t_lat ** 2).mean())),
                corr_node_lat=float(np.corrcoef(t_node, t_lat)[0, 1]))


# ----------------------------------------------------------------------------
def main():
    summary = {}
    summary["tle"] = tle_check()

    # ---- nominal mean orbit -------------------------------------------------
    n0, rd, ud = C.orbit_rates()
    summary["nominal_orbit"] = dict(
        a_km=C.A_ORBIT, inc_deg=C.INC_DEG, kepler_period_min=2 * math.pi / n0 / 60,
        nodal_period_min=2 * math.pi / ud / 60, raan_rate_deg_day=math.degrees(rd) * 86400,
        node_precession_period_days=360.0 / abs(math.degrees(rd) * 86400),
        node_period_relative_to_sun_days=360.0 / (abs(math.degrees(rd) * 86400) + 360.0 / 365.2422),
        orbital_speed_kms=math.sqrt(C.MU_E / C.A_ORBIT), earth_half_angle_deg=math.degrees(C.earth_half_angle()),
        orbits_per_node_cycle=(360.0 / abs(math.degrees(rd) * 86400)) * 86400 / (2 * math.pi / ud))

    wt = WindTemplate()
    vs = wt.vsun
    alpha_d, delta_d = C.vec_to_radec(vs)
    alpha, delta, inc = math.radians(alpha_d), math.radians(delta_d), math.radians(C.INC_DEG)
    summary["apex_sun_only"] = dict(ra_deg=float(alpha_d), dec_deg=float(delta_d), speed=wt.vsun_speed,
                                    mean_speed_ref=wt.ref_mean_speed)
    summary["P1_check"] = p1_check(alpha, delta, inc)

    # ---- static template on the (u, Omega) plane ----------------------------
    nu, nO = 360, 360
    u = (np.arange(nu) + 0.5) * 2 * np.pi / nu
    dOm = (np.arange(nO) + 0.5) * 2 * np.pi / nO          # Omega - alpha_apex
    U, DO = np.meshgrid(u, dOm, indexing="ij")
    rh, vh = C.sat_vectors(U.ravel(), (DO.ravel() + alpha), C.INC_DEG)
    vsat = vh * math.sqrt(C.MU_E / C.A_ORBIT)
    Tfull, Tshadow, Tram, psi = wt.evaluate(rh, vsat, return_parts=True)
    Tfull, Tshadow, Tram, psi = (x.reshape(nu, nO) for x in (Tfull, Tshadow, Tram, psi))
    T0_u = Tshadow.mean(axis=1)                             # Omega-independent part (function of u)
    T1 = Tshadow - T0_u[:, None]                            # Omega-dependent part
    lat_u = np.arcsin(np.sin(u) * math.sin(inc))
    latg, T0lat = wt.lat_profile()
    T0_from_lat = np.interp(lat_u, latg, T0lat)
    mean_all = Tshadow.mean()
    dip = 0.0
    # dipole-only version of the same decomposition (uses Legendre a0, a1 of F)
    x = np.cos(wt.psi)[::-1]
    Fp = wt.F(wt.psi)[::-1]
    xx = np.linspace(-1, 1, 20001)
    yy = np.interp(xx, x, Fp)
    a0 = 0.5 * np.trapezoid(yy, xx)
    a1 = 1.5 * np.trapezoid(yy * xx, xx)
    Tdip = a0 + a1 * np.cos(psi)
    summary["template_static"] = dict(
        mean=float(mean_all), min=float(Tshadow.min()), max=float(Tshadow.max()),
        rms_total=float(Tshadow.std()), rms_Omega_independent=float(T0_u.std()),
        rms_Omega_dependent=float(T1.std()),
        variance_fraction_Omega_dependent=float(T1.var() / Tshadow.var()),
        max_abs_T0u_minus_T0lat=float(np.abs(T0_u - T0_from_lat).max()),
        legendre_a0=float(a0), legendre_a1=float(a1),
        rms_of_template_minus_dipole_approx=float((Tshadow - Tdip).std()),
        dipole_rms_node_part=float(a1 * math.cos(delta) * math.sqrt(0.5 * (1 + math.cos(inc) ** 2) / 2.0)),
        psi_range_deg=[float(np.degrees(psi.min())), float(np.degrees(psi.max()))],
    )
    # ram term
    summary["ram_effect"] = dict(
        rms=float(Tram.std()), max_abs=float(np.abs(Tram).max()),
        rms_relative_to_Omega_dependent=float(Tram.std() / T1.std()),
        corr_with_shadow_template=float(np.corrcoef(Tram.ravel(), Tshadow.ravel())[0, 1]),
        open_sky_ram_amplitude=float(C.FluxTable(wt.vsun_speed).c1 * math.sqrt(C.MU_E / C.A_ORBIT) / wt.ref_mean_speed),
    )
    # harmonic content in orbital phase at fixed node
    ft = np.fft.rfft(Tfull, axis=0) / nu
    amp = 2 * np.abs(ft[1:5])                                # amplitudes of harmonics 1..4, per node
    summary["orbital_harmonics"] = dict(
        mean_amp_h1=float(amp[0].mean()), mean_amp_h2=float(amp[1].mean()),
        mean_amp_h3=float(amp[2].mean()), mean_amp_h4=float(amp[3].mean()),
        max_amp_h1=float(amp[0].max()), max_amp_h2=float(amp[1].max()))

    # ---- depth vs node ------------------------------------------------------
    hn = C.orbit_normal(dOm + alpha)
    elev = np.degrees(np.arcsin(np.clip((hn * (vs / wt.vsun_speed)[:, None]).sum(axis=0), -1, 1)))  # apex elevation above orbit plane
    p2p_full = Tfull.max(axis=0) - Tfull.min(axis=0)
    p2p_shadow = Tshadow.max(axis=0) - Tshadow.min(axis=0)
    p2p_node = T1.max(axis=0) - T1.min(axis=0)
    rel = p2p_full / (Tfull.max(axis=0) + Tfull.min(axis=0))
    rows = np.column_stack([np.degrees(dOm), elev, Tfull.mean(axis=0), Tfull.min(axis=0), Tfull.max(axis=0),
                            p2p_full, rel, p2p_node, amp[0], amp[1]])
    np.savetxt(os.path.join(OUT, "s2_depth_vs_node.csv"), rows, delimiter=",", fmt="%.5f", comments="",
               header="Omega_minus_alpha_deg,apex_elev_above_orbit_plane_deg,orbit_mean_T,T_min,T_max,"
                      "peak_to_peak,relative_depth,peak_to_peak_Omega_dependent_part,amp_harm1,amp_harm2")
    summary["depth_vs_node"] = dict(
        p2p_max=float(p2p_full.max()), p2p_min=float(p2p_full.min()), p2p_mean=float(p2p_full.mean()),
        node_offset_of_max_deg=float(np.degrees(dOm[p2p_full.argmax()])),
        node_offset_of_min_deg=float(np.degrees(dOm[p2p_full.argmin()])),
        orbit_mean_T_min=float(Tfull.mean(axis=0).min()), orbit_mean_T_max=float(Tfull.mean(axis=0).max()))

    # ---- one year, day by day (2022): annual modulation + precession ---------
    jd0 = C.jd_from_iso("2022-01-01T00:00:00")
    days = np.arange(0.0, 365.0, 0.25)
    jd = jd0 + days
    sun = C.sun_state(jd)
    vE = vs[:, None] + sun["v_earth"]                        # Earth-frame velocity through the halo
    spd = np.sqrt((vE ** 2).sum(axis=0))
    ra_a, dec_a = C.vec_to_radec(vE)
    shift = np.degrees(np.arccos(np.clip((vE / spd * (vs / wt.vsun_speed)[:, None]).sum(axis=0), -1, 1)))
    _, Om_t = C.orbit_angles(jd)
    uu = (np.arange(180) + 0.5) * 2 * np.pi / 180
    depth_full = np.empty(len(jd))
    depth_static = np.empty(len(jd))
    mean_full = np.empty(len(jd))
    vc = math.sqrt(C.MU_E / C.A_ORBIT)
    for k in range(len(jd)):
        rh_k, vh_k = C.sat_vectors(uu, np.full_like(uu, Om_t[k]), C.INC_DEG)
        Tk = wt.evaluate(rh_k, vh_k * vc, vE[:, k])
        Ts = wt.evaluate(rh_k, vh_k * vc)
        depth_full[k] = Tk.max() - Tk.min()
        depth_static[k] = Ts.max() - Ts.min()
        mean_full[k] = Tk.mean()
    open_flux = wt.open_flux(spd)
    np.savetxt(os.path.join(OUT, "s2_annual.csv"),
               np.column_stack([days, spd, ra_a, dec_a, shift, open_flux, np.degrees(Om_t), depth_full, depth_static, mean_full]),
               delimiter=",", fmt="%.5f", comments="",
               header="day_of_2022,v_earthframe_kms,apex_ra_deg,apex_dec_deg,apex_shift_deg,open_flux_rel,"
                      "Omega_deg,p2p_depth_full,p2p_depth_static_apex,orbit_mean_T")
    imax, imin = int(spd.argmax()), int(spd.argmin())
    summary["annual"] = dict(
        v_max=float(spd.max()), day_of_v_max=float(days[imax]), v_min=float(spd.min()), day_of_v_min=float(days[imin]),
        open_flux_rel_max=float(open_flux.max()), open_flux_rel_min=float(open_flux.min()),
        open_flux_semi_amplitude=float(0.5 * (open_flux.max() - open_flux.min())),
        apex_shift_max_deg=float(shift.max()), apex_ra_range=[float(ra_a.min()), float(ra_a.max())],
        apex_dec_range=[float(dec_a.min()), float(dec_a.max())],
        depth_full_min=float(depth_full.min()), depth_full_max=float(depth_full.max()),
        depth_static_min=float(depth_static.min()), depth_static_max=float(depth_static.max()),
        max_abs_depth_change_from_annual=float(np.abs(depth_full - depth_static).max()))
    # monthly table of the apex
    monthly = []
    for m in range(12):
        j = C.jd_from_iso(f"2022-{m + 1:02d}-15T00:00:00")
        s = C.sun_state(np.array([j]))
        v = vs + s["v_earth"][:, 0]
        r_, d_ = C.vec_to_radec(v)
        sp = float(np.linalg.norm(v))
        monthly.append(dict(month=m + 1, v=sp, ra=float(r_), dec=float(d_), open_flux_rel=float(wt.open_flux(sp))))
    summary["annual_monthly"] = monthly

    np.savez_compressed(os.path.join(OUT, "s2_template_uOmega.npz"), u=u, dOmega=dOm, T_full=Tfull,
                        T_shadow=Tshadow, T_ram=Tram, T0_u=T0_u, T1=T1)
    with open(os.path.join(OUT, "s2_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    # ---- figure 2: template in the (u, Omega) plane -------------------------
    P.apply()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    ext = [0, 360, 0, 360]
    im0 = axs[0].imshow(Tshadow.T, origin="lower", extent=ext, aspect="auto", cmap=P.SEQ_BLUE, vmin=0, vmax=1)
    axs[0].set_title("(a) Template T(u, Omega)")
    im1 = axs[1].imshow(np.repeat(T0_u[:, None], nO, axis=1).T, origin="lower", extent=ext, aspect="auto",
                        cmap=P.SEQ_BLUE, vmin=0, vmax=1)
    axs[1].set_title("(b) Omega-independent part")
    vm = float(np.abs(T1).max())
    im2 = axs[2].imshow(T1.T, origin="lower", extent=ext, aspect="auto", cmap=P.DIV_BLUE_RED, vmin=-vm, vmax=vm)
    axs[2].set_title("(c) Omega-dependent part")
    for ax in axs:
        ax.set_xlabel("u: argument of latitude (deg)")
        ax.set_xticks(np.arange(0, 361, 90))
        ax.set_yticks(np.arange(0, 361, 90))
        ax.grid(False)
    axs[0].set_ylabel("Omega - RA(apex)  (deg)")
    for im, ax, lab in ((im0, axs[0], "T: transmitted DM flux (unshadowed = 1)"), (im1, axs[1], "T0(u): Omega-average, a function of latitude only"),
                        (im2, axs[2], "T - T0: the sidereal (node-dependent) signal")):
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.17, fraction=0.05, aspect=40)
        cb.set_label(lab, color=P.INK2, fontsize=8.5)
        cb.outline.set_edgecolor(P.AXIS)
    P.note(fig, "Simulation (forecast). Opaque Earth, isotropically sensitive detector, a = 6921 km, i = 43 deg, "
                f"static apex RA {alpha_d:.1f}, Dec {delta_d:+.1f} deg (Sun's velocity only); ram term not included in these panels.")
    fig.savefig(os.path.join(FIG, "fig2_template_u_Omega.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- figure 3: modulation depth over one year ---------------------------
    fig, axs = plt.subplots(3, 1, figsize=(9.6, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    ax = axs[0]
    ax.plot(days, depth_full, color=P.BLUE, label="with annual aberration + ram term")
    ax.plot(days, depth_static, color=P.ORANGE, lw=1.3, label="static apex (Sun's velocity only)")
    ax.set_ylabel("peak-to-peak depth per orbit\n(units of unshadowed rate S0)")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a) Orbital-period modulation depth of the DM template through 2022")
    ax.legend(loc="upper right", ncol=2)
    ax = axs[1]
    ax.plot(days, 100 * (open_flux - 1), color=P.BLUE)
    ax.set_ylabel("unshadowed flux\nchange (%)")
    ax.set_title("(b) Annual modulation of the unshadowed flux (Earth's orbital velocity)")
    ax = axs[2]
    ax.plot(days, shift, color=P.BLUE)
    ax.set_ylabel("apex shift (deg)")
    ax.set_title("(c) Angular offset of the apex from the Sun-only (annual-mean) apex")
    ax.set_xlabel("day of year 2022")
    ax.set_xlim(0, 365)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    P.note(fig, "Simulation (forecast). Mean circular J2 orbit anchored to the 2026-11-18 TLE node; the depth envelope repeats "
                "with the nodal precession period (65.7 d for a = 6921 km).")
    fig.savefig(os.path.join(FIG, "fig3_modulation_depth_year.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
