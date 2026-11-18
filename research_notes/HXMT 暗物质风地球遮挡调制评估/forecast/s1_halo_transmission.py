#!/usr/bin/env python3
"""
s1_halo_transmission.py -- Task 1 of the forecast.

Standard Halo Model flux in the frame of an observer moving through the halo,
and the fraction of that flux that reaches a satellite at 550 km altitude when
the Earth blocks arrival directions inside the Earth disc.

Outputs
  outputs/s1_transmission_table.csv   transmitted fraction vs psi (several thresholds / halo variants)
  outputs/s1_transparent_table.csv    partially transparent Earth: modulation depth vs sigma/m
  outputs/s1_summary.json             key numbers
  figures/fig1_transmission_vs_psi.png

Run:  python3 s1_halo_transmission.py        (about 20 s)

All numbers are simulation results under the stated halo/geometry assumptions.
"""
import json
import math
import os

import numpy as np

import common as C
import plotstyle as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

THRESHOLDS = [0.0, 300.0, 400.0, 500.0]          # km/s, speed in the detector frame
PSI_DEG = np.arange(0.0, 180.1, 2.0)
PSI = np.radians(PSI_DEG)


def legendre_coeffs(psi, F, lmax=6):
    """a_l such that F(psi) = sum_l a_l P_l(cos psi)."""
    x = np.cos(psi)[::-1]
    y = F[::-1]
    xx = np.linspace(-1, 1, 20001)
    yy = np.interp(xx, x, y)
    out = []
    for l in range(lmax + 1):
        Pl = np.polynomial.legendre.Legendre.basis(l)(xx)
        out.append((2 * l + 1) / 2.0 * np.trapezoid(yy * Pl, xx))
    return np.array(out)


def monte_carlo_check(vobs, v0, vesc, theta_E, psi_list, wthr_list, n=4_000_000, seed=1):
    """Brute-force check of the quadrature: sample the truncated Maxwellian."""
    rng = np.random.default_rng(seed)
    vg = rng.normal(0.0, v0 / math.sqrt(2.0), size=(int(n * 1.02), 3))
    vg = vg[(vg ** 2).sum(axis=1) <= vesc ** 2][:n]
    w = vg - np.array([0.0, 0.0, vobs])           # apex = +z
    sp = np.sqrt((w ** 2).sum(axis=1))
    what = w / sp[:, None]
    res = {}
    for wt in wthr_list:
        sel = sp >= wt
        tot = sp[sel].sum()
        row = []
        for ps in psi_list:
            rhat = np.array([math.sin(ps), 0.0, math.cos(ps)])
            blocked = (what[sel] @ rhat) > math.cos(theta_E)   # came up through the Earth
            row.append(1.0 - sp[sel][blocked].sum() / tot)
        res[wt] = np.array(row)
    mean_speed = sp.mean()
    return res, mean_speed, len(sp)


def ram_mc_check(vobs, v0, vesc, theta_E, psi, vsat=7.59, n=4_000_000, seed=2):
    """MC of Rate = <|v_e - v_sat| Theta(unblocked)> for v_sat along e_perp (in the (a, r) plane)."""
    rng = np.random.default_rng(seed)
    vg = rng.normal(0.0, v0 / math.sqrt(2.0), size=(int(n * 1.02), 3))
    vg = vg[(vg ** 2).sum(axis=1) <= vesc ** 2][:n]
    ve = vg - np.array([0.0, 0.0, vobs])
    rhat = np.array([math.sin(psi), 0.0, math.cos(psi)])
    eperp = (np.array([0.0, 0.0, 1.0]) - math.cos(psi) * rhat) / math.sin(psi)
    sp = np.sqrt((ve ** 2).sum(axis=1))
    unblocked = (ve @ rhat) / sp <= math.cos(theta_E)
    out = {}
    for sgn in (+1, 0, -1):
        vs = sgn * vsat * eperp
        rel = np.sqrt(((ve - vs) ** 2).sum(axis=1))
        out[sgn] = (rel * unblocked).mean()
    return out, vsat * math.sin(psi)   # rates and (v_sat . a) for sgn=+1


def main():
    summary = {}
    vs_eq = C.vsun_eq()
    vobs = float(np.linalg.norm(vs_eq))
    ra, dec = C.vec_to_radec(vs_eq)
    g = C.eq_to_gal(vs_eq)
    summary["apex_sun_only"] = dict(
        ra_deg=float(ra), dec_deg=float(dec),
        l_deg=math.degrees(math.atan2(g[1], g[0])), b_deg=math.degrees(math.asin(g[2] / vobs)),
        speed_kms=vobs)
    sh = C.Shadow()
    summary["earth_half_angle_deg"] = math.degrees(sh.theta_E)
    summary["earth_sky_fraction"] = (1 - math.cos(sh.theta_E)) / 2

    # ---------------- opaque Earth: baseline + variations ----------------
    variants = {
        "v0=238 (baseline)": dict(v0=238.0, vesc=544.0),
        "v0=220": dict(v0=220.0, vesc=544.0),
    }
    table_cols = {"psi_deg": PSI_DEG}
    halo_rows = []
    for name, par in variants.items():
        vsun = float(np.linalg.norm(C.vsun_eq(par["v0"])))
        tab = C.FluxTable(vsun, par["v0"], par["vesc"])
        row = dict(variant=name, v_obs=vsun, mean_speed=tab.mean_speed, c1=tab.c1,
                   norm_check=tab.norm_check)
        for wt in THRESHOLDS:
            F = sh.transmitted_fraction(tab, PSI, wthr=wt)
            key = f"F_{name.split()[0]}_w>{int(wt)}"
            table_cols[key] = F
            row[f"flux_frac_above_{int(wt)}"] = tab.flux_above(wt) / tab.mean_speed
            row[f"Fmax_{int(wt)}"] = float(F.max())
            row[f"Fmin_{int(wt)}"] = float(F.min())
            # sphere average (uniform in cos psi) -- must equal 1 - sky fraction for wt=0
            row[f"Fsphere_{int(wt)}"] = float(np.trapezoid(F[::-1], np.cos(PSI)[::-1]) / 2.0)
            row[f"legendre_{int(wt)}"] = legendre_coeffs(PSI, F).tolist()
        halo_rows.append(row)
    summary["halo_variants"] = halo_rows

    # annual extremes of the Earth-frame speed (computed properly in s2; here +-15 km/s bracket)
    base = variants["v0=238 (baseline)"]
    seas = []
    for dv in (-15.0, 0.0, +15.0, -7.6, +7.6):
        tab = C.FluxTable(vobs + dv, base["v0"], base["vesc"])
        F = sh.transmitted_fraction(tab, np.radians([0.0, 90.0, 180.0]))
        seas.append(dict(dv=dv, v_obs=vobs + dv, mean_speed=tab.mean_speed,
                         F0=float(F[0]), F90=float(F[1]), F180=float(F[2])))
    summary["speed_sensitivity"] = seas

    # ---------------- Monte Carlo validation ----------------
    tab0 = C.FluxTable(vobs, base["v0"], base["vesc"])
    psi_mc = np.radians([0.0, 45.0, 90.0, 135.0, 180.0])
    mc, mc_speed, nmc = monte_carlo_check(vobs, base["v0"], base["vesc"], sh.theta_E, psi_mc, THRESHOLDS)
    val = []
    for wt in THRESHOLDS:
        q = sh.transmitted_fraction(tab0, psi_mc, wthr=wt)
        val.append(dict(wthr=wt, quadrature=q.tolist(), monte_carlo=mc[wt].tolist(),
                        max_abs_diff=float(np.abs(q - mc[wt]).max())))
    summary["mc_validation"] = dict(n_samples=nmc, mean_speed_mc=float(mc_speed),
                                    mean_speed_quad=tab0.mean_speed, rows=val)
    # ram kernel check at psi = 60 and 120 deg
    ram = []
    for pdeg in (60.0, 120.0):
        ps = math.radians(pdeg)
        r, va = ram_mc_check(vobs, base["v0"], base["vesc"], sh.theta_E, ps)
        K = float(sh.ram_kernel(tab0, np.array([ps]))[0])
        ram.append(dict(psi_deg=pdeg, K_quadrature=K, K_mc=float((r[1] - r[-1]) / (2 * va)),
                        rate0_mc=float(r[0]),
                        rate0_quad=float(tab0.mean_speed - sh.blocked(tab0, np.array([ps]))[0])))
    summary["ram_kernel_check"] = ram
    Kpsi = sh.ram_kernel(tab0, PSI)
    table_cols["ram_kernel_K"] = Kpsi

    # ---------------- partially transparent Earth ----------------
    sha = C.Shadow(with_atm=True)
    som_grid = np.logspace(-36.0, -30.0, 49)                 # cm^2/GeV
    wthr_tr = [C.V_ESC_EARTH, 100.0, 200.0, 300.0, 400.0]
    tr_rows = []
    depth = {}
    for wt in wthr_tr:
        F_op = sh.transmitted_fraction(tab0, np.radians([0.0, 180.0]), wthr=wt)
        d_op = F_op[0] - F_op[1]
        d = []
        for som in som_grid:
            F = sha.transmitted_fraction(tab0, np.radians([0.0, 180.0]), wthr=wt, sigma_over_m=som)
            d.append(F[0] - F[1])
        d = np.array(d)
        depth[wt] = d / d_op
        # sigma/m at which the modulation depth reaches 10/50/90 % of the opaque value
        lv = {}
        for frac in (0.1, 0.5, 0.9):
            lv[frac] = float(10 ** np.interp(frac, d / d_op, np.log10(som_grid)))
        tr_rows.append(dict(wthr=wt, depth_opaque=float(d_op), som_10=lv[0.1], som_50=lv[0.5], som_90=lv[0.9]))
    summary["transparent_earth"] = dict(
        density_model="PREM (Dziewonski & Anderson 1981) + exponential atmosphere (H=7 km)",
        column_centre_g_cm2=float(C.chord_column([0.0])[0]),
        column_median_over_disc_g_cm2=float(np.median(sh.column_solid)),
        rows=tr_rows)
    # effect of the atmosphere at very large sigma/m (effective opaque radius grows)
    atm = []
    for som in (1e-28, 1e-26, 3e-25):
        F = sha.transmitted_fraction(tab0, np.radians([0.0, 90.0, 180.0]), sigma_over_m=som)
        atm.append(dict(sigma_over_m=som, F0=float(F[0]), F90=float(F[1]), F180=float(F[2])))
    Fo = sh.transmitted_fraction(tab0, np.radians([0.0, 90.0, 180.0]))
    atm.append(dict(sigma_over_m="solid opaque Earth, no atmosphere", F0=float(Fo[0]), F90=float(Fo[1]), F180=float(Fo[2])))
    summary["atmosphere_effect"] = atm

    # ---------------- write tables ----------------
    names = list(table_cols.keys())
    arr = np.column_stack([table_cols[k] for k in names])
    np.savetxt(os.path.join(OUT, "s1_transmission_table.csv"), arr, delimiter=",",
               header=",".join(names), comments="", fmt="%.6f")
    arr2 = np.column_stack([som_grid] + [depth[w] for w in wthr_tr])
    np.savetxt(os.path.join(OUT, "s1_transparent_table.csv"), arr2, delimiter=",",
               header="sigma_over_m_cm2_per_GeV," + ",".join(f"rel_depth_wthr{int(w)}" for w in wthr_tr),
               comments="", fmt="%.6e")
    with open(os.path.join(OUT, "s1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---------------- figure 1 ----------------
    P.apply()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3))
    ax = axs[0]
    labels = {0.0: "all speeds", 300.0: "w > 300 km/s", 400.0: "w > 400 km/s", 500.0: "w > 500 km/s"}
    for k, wt in enumerate(THRESHOLDS):
        ax.plot(PSI_DEG, table_cols[f"F_v0=238_w>{int(wt)}"], color=P.CAT[k], label=labels[wt])
    ax.plot(PSI_DEG, table_cols["F_v0=220_w>0"], color=P.MUTED, lw=1.3, ls=(0, (4, 2)),
            label="all speeds, v0 = 220 km/s")
    ax.axvline(math.degrees(math.pi - sh.theta_E), color=P.AXIS, lw=0.8)
    ax.text(math.degrees(math.pi - sh.theta_E) - 2, 0.03, "apex direction\nsets at 113 deg", fontsize=7.5,
            color=P.INK2, ha="right", va="bottom")
    ax.set_xlabel("psi: angle between satellite zenith and wind apex (deg)")
    ax.set_ylabel("transmitted fraction of the DM flux")
    ax.set_title("(a) Opaque Earth, h = 550 km")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.18))

    ax = axs[1]
    soms = [1e-35, 1e-34, 3e-34, 1e-33, 1e-32]
    cols = P.SEQ_BLUE(np.linspace(0.3, 0.95, len(soms)))
    wt_show = 300.0
    for som, col in zip(soms, cols):
        F = sha.transmitted_fraction(tab0, PSI, wthr=wt_show, sigma_over_m=som)
        ax.plot(PSI_DEG, F, color=col, label=f"sigma/m = {som:.0e} cm$^2$/GeV")
    ax.plot(PSI_DEG, table_cols["F_v0=238_w>300"], color=P.INK, lw=1.2, ls=(0, (4, 2)), label="opaque limit")
    ax.set_xlabel("psi (deg)")
    ax.set_ylabel("transmitted fraction (w > 300 km/s at detector)")
    ax.set_title("(b) Partially transparent Earth (PREM)")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.legend(loc="lower left")

    ax = axs[2]
    lab = {C.V_ESC_EARTH: "stopped only (w < 11 km/s)", 100.0: "threshold 100 km/s",
           200.0: "threshold 200 km/s", 300.0: "threshold 300 km/s", 400.0: "threshold 400 km/s"}
    for k, wt in enumerate(wthr_tr):
        ax.plot(som_grid, depth[wt], color=P.CAT[k], label=lab[wt])
    ax.set_xscale("log")
    ax.set_xlabel("sigma / m  (cm$^2$ / GeV)")
    ax.set_ylabel("shadow depth F(0) - F(180), relative to opaque Earth")
    ax.set_title("(c) Where the Earth becomes opaque")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    P.note(fig, "Simulation (forecast), not HXMT data. SHM: v0 = 238 km/s, v_esc = 544 km/s, Sun-only observer speed "
                f"{vobs:.1f} km/s; straight-line trajectories, no gravitational focusing; heavy-DM continuous slowing down in (b),(c).")
    fig.savefig(os.path.join(FIG, "fig1_transmission_vs_psi.png"))
    plt.close(fig)

    # ---------------- console report ----------------
    print(json.dumps({k: summary[k] for k in ("apex_sun_only", "earth_half_angle_deg", "earth_sky_fraction")}, indent=1))
    for r in halo_rows:
        print(f"\n{r['variant']}: v_obs={r['v_obs']:.2f}  <w>={r['mean_speed']:.2f} km/s  c1={r['c1']:.4f}")
        for wt in THRESHOLDS:
            i = int(wt)
            print(f"  w>{i:3d}: flux frac {r[f'flux_frac_above_{i}']:.4f}  Fmax {r[f'Fmax_{i}']:.4f}  "
                  f"Fmin {r[f'Fmin_{i}']:.4f}  sphere-avg {r[f'Fsphere_{i}']:.4f}  "
                  f"a_l(0..4) {np.round(r[f'legendre_{i}'][:5], 4)}")
    print("\nspeed sensitivity:", json.dumps(seas))
    print("\nMC validation:", json.dumps(summary["mc_validation"], indent=0)[:1500])
    print("\nram kernel check:", json.dumps(ram))
    print("\ntransparent Earth:", json.dumps(summary["transparent_earth"], indent=0))
    print("\natmosphere:", json.dumps(atm))
    print("\npsi  F(all)  F(>300) F(>400) F(>500) F(v0=220)  K_ram")
    for i in range(0, len(PSI_DEG), 5):
        print(f"{PSI_DEG[i]:5.0f} {table_cols['F_v0=238_w>0'][i]:.4f} {table_cols['F_v0=238_w>300'][i]:.4f} "
              f"{table_cols['F_v0=238_w>400'][i]:.4f} {table_cols['F_v0=238_w>500'][i]:.4f} "
              f"{table_cols['F_v0=220_w>0'][i]:.4f}  {Kpsi[i]:.4f}")


if __name__ == "__main__":
    main()
