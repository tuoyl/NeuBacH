#!/usr/bin/env python3
"""
s5_controls.py -- Task 6 of the forecast: control templates and null tests.

  (1) "Wind map": inject the full DM signal (true apex, annual aberration, ram term)
      and fit static templates centred on a grid of other sky directions, after
      marginalising nuisance model D (free 5x5 deg map x asc/desc + f(time since SAA)
      + f(local time) + f(beta) + monthly trend).  The map of recovered amplitude /
      significance is the point-spread function of the method.
  (2) Anti-sidereal construction: the same scan with the satellite's right
      ascension replaced by RA - 2 L_sun(t) (a frame that rotates the wrong way by
      one turn per year).  A real sidereal signal leaves no trace there; a
      seasonally modulated solar-diurnal background shows up in both frames.
  (3) Overlap of the DM template with Earth-occultation step templates of point
      sources (Cyg X-1, Cyg X-3, Crab, Galactic centre).

Outputs
  outputs/s5_windmap.npz, outputs/s5_summary.json, outputs/s5_occultation_overlap.csv
  figures/fig4_windmap_psf.png

Run:  python3 s5_controls.py [--dt 120] [--step 10]      (about 4-6 min)
Simulation (forecast) only.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

import common as C
import nuisance as NU
import plotstyle as P
import timeline as TL
from template import WindTemplate
from s4_nuisance_fisher import Blocks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")

SOURCES = {  # name: (RA, Dec) J2000 deg
    "Cyg X-1": (299.59, 35.20),
    "Cyg X-3": (308.11, 40.96),
    "Crab": (83.63, 22.01),
    "Galactic centre": (266.42, -29.01),
}
R_XRAY_BLOCK = C.R_E + 80.0      # km: the atmosphere is opaque to 20-250 keV photons below ~80 km (assumption)


def rotate_z(vec, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--step", type=float, default=10.0, help="grid step of the all-sky scan (deg)")
    ap.add_argument("--model", default="D")
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    tl = TL.build(args.dt)
    TL.pointing_schedule(tl)
    dt = tl["dt"]
    ok = ~tl["saa"]
    idx = np.where(ok)[0]
    b = TL.toy_background(tl)
    bb = b[idx] / b[idx].mean()
    W = dt / bb
    tlive = len(idx) * dt
    wt = WindTemplate()
    rhat = tl["rhat"][:, idx]
    vsat = tl["vhat"][:, idx] * tl["vsat_kms"]
    vE = wt.vsun[:, None] + tl["v_earth"][:, idx]
    T_full = wt.evaluate(rhat, vsat, vE)
    blk = Blocks(tl, idx)
    cell, Z = blk.model(args.model)
    proj = NU.Projector(W, cell, Z)
    I_true, r_true = proj.info(T_full)
    ra0, dec0 = C.vec_to_radec(wt.vsun)
    summary = dict(model=args.model, dt=dt, live_time_s=tlive, r_eff_true=math.sqrt(I_true / tlive),
                   apex_ra=float(ra0), apex_dec=float(dec0))

    # anti-sidereal pseudo-positions: RA' = RA - 2 L_sun (L_sun = mean solar longitude)
    Ls = tl["sun_Lmean"][idx]
    rhat_anti = rotate_z(rhat, -2.0 * Ls)
    rhat_ext = rotate_z(rhat, +Ls)          # "extended-sidereal" frame (2 extra turns/yr w.r.t. solar)

    def scan(rh, ra_list, dec_list):
        amp = np.empty((len(dec_list), len(ra_list)))
        snr = np.empty_like(amp)
        for i, d in enumerate(dec_list):
            for j, r in enumerate(ra_list):
                n = C.radec_to_vec(r, d)
                Tk = wt.evaluate(rh, None, wt.vsun_speed * n)
                Ik, rk = proj.info(Tk)
                cross = float((W * rk * T_full).sum())
                amp[i, j] = cross / Ik if Ik > 1e-9 * I_true else np.nan
                snr[i, j] = cross / math.sqrt(Ik * I_true) if Ik > 1e-9 * I_true else 0.0
        return amp, snr

    step = args.step
    ra_g = np.arange(0.0, 360.0, step) + step / 2
    dec_g = np.arange(-90.0 + step / 2, 90.0, step)
    t0 = time.time()
    amp_sid, snr_sid = scan(rhat, ra_g, dec_g)
    print(f"sidereal scan: {time.time() - t0:.0f} s; max snr fraction {np.nanmax(snr_sid):.4f}")
    t0 = time.time()
    amp_anti, snr_anti = scan(rhat_anti, ra_g, dec_g)
    print(f"anti-sidereal scan: {time.time() - t0:.0f} s; max |snr fraction| {np.abs(snr_anti).max():.4f}")
    amp_ext, snr_ext = scan(rhat_ext, ra_g[::2], dec_g[::2])
    print(f"extended-sidereal scan: max |snr fraction| {np.abs(snr_ext).max():.4f}")

    # fine cuts through the apex
    dra = np.arange(-90.0, 90.1, 2.0)
    cut_ra_amp, cut_ra = scan(rhat, (ra0 + dra) % 360.0, [dec0])
    ddec = np.arange(-40.0, 90.1, 2.0)
    cut_dec_amp, cut_dec = scan(rhat, [ra0], ddec)
    cut_ra, cut_dec = cut_ra[0], cut_dec[:, 0]
    best = max(cut_ra.max(), cut_dec.max(), np.nanmax(snr_sid))

    def halfwidth(x, y, level):
        """smallest |x - x_peak| on each side where y drops below level (linear interpolation)."""
        ip = int(np.argmax(y))
        out = []
        for sl in (slice(ip, None), slice(ip, None, -1)):
            xs, ys = x[sl], y[sl]
            below = np.where(ys < level)[0]
            if len(below) == 0:
                out.append(float("nan"))
                continue
            k = below[0]
            xx = xs[k - 1] + (level - ys[k - 1]) * (xs[k] - xs[k - 1]) / (ys[k] - ys[k - 1])
            out.append(abs(float(xx - x[ip])))
        return out, float(x[ip])

    loc = {}
    for nsig in (5.0, 10.0, 20.0):
        # 68% CL (2 parameters): delta chi2 = nsig^2 (1 - rho^2) = 2.30
        lvl = best * math.sqrt(max(0.0, 1.0 - 2.30 / nsig ** 2))
        (hr1, hr2), pr = halfwidth(dra, cut_ra, lvl)
        (hd1, hd2), pd = halfwidth(ddec, cut_dec, lvl)
        loc[f"{int(nsig)}sigma"] = dict(level=lvl / best, ra_halfwidth_deg=[hr1, hr2],
                                        ra_halfwidth_on_sky_deg=[hr1 * math.cos(math.radians(dec0)), hr2 * math.cos(math.radians(dec0))],
                                        dec_halfwidth_deg=[hd1, hd2])
    (fw1, fw2), _ = halfwidth(dra, cut_ra, 0.5 * best)
    (fd1, fd2), _ = halfwidth(ddec, cut_dec, 0.5 * best)
    summary["windmap"] = dict(
        best_snr_fraction=float(best), peak_ra_offset_deg=float(dra[np.argmax(cut_ra)]),
        peak_dec_deg=float(ddec[np.argmax(cut_dec)]),
        amplitude_at_true_apex=float(cut_ra_amp[0][np.argmin(np.abs(dra))]),
        fwhm_ra_deg=fw1 + fw2, fwhm_ra_on_sky_deg=(fw1 + fw2) * math.cos(math.radians(dec0)),
        fwhm_dec_deg=fd1 + fd2, localisation_68=loc,
        snr_fraction_at_antiapex=float(snr_sid[np.argmin(np.abs(dec_g + dec0)), np.argmin(np.abs(ra_g - (ra0 + 180) % 360))]),
        anti_sidereal_max_abs_snr_fraction=float(np.abs(snr_anti).max()),
        anti_sidereal_rms_snr_fraction=float(np.sqrt((snr_anti ** 2).mean())),
        extended_sidereal_max_abs_snr_fraction=float(np.abs(snr_ext).max()))
    for name, (r, d) in SOURCES.items():
        i, j = np.argmin(np.abs(dec_g - d)), np.argmin(np.abs(ra_g - r))
        sep = math.degrees(math.acos(np.clip(C.radec_to_vec(r, d) @ (wt.vsun / wt.vsun_speed), -1, 1)))
        summary["windmap"][f"snr_fraction_at_{name}"] = float(snr_sid[i, j])
        summary["windmap"][f"separation_from_apex_{name}_deg"] = sep

    # ---- seasonal x diurnal toy terms in both frames ---------------------
    lt = tl["lt_hours"][idx] * (2 * np.pi / 24.0)
    ann = tl["doy_phase"][idx]
    toys = {"cos(LT)cos(yr)": np.cos(lt) * np.cos(ann), "cos(LT)sin(yr)": np.cos(lt) * np.sin(ann),
            "sin(LT)cos(yr)": np.sin(lt) * np.cos(ann), "sin(LT)sin(yr)": np.sin(lt) * np.sin(ann),
            "cos(LT) [static]": np.cos(lt), "cos(LT)(1+0.033cos(yr-perihelion))": np.cos(lt) * (1 + 0.033 * np.cos(ann - math.radians(282.9 - 180.0)))}
    # dipole-like scan directions in each frame (coarse grid is enough for the maximum)
    ra_c = np.arange(0.0, 360.0, 20.0)
    dec_c = np.array([-60.0, -30.0, 0.0, 30.0, 47.5, 60.0])
    resid = {}
    for frame, rh in (("sidereal", rhat), ("anti-sidereal", rhat_anti)):
        lst = []
        for d in dec_c:
            for r in ra_c:
                Tk = wt.evaluate(rh, None, wt.vsun_speed * C.radec_to_vec(r, d))
                Ik, rk = proj.info(Tk)
                lst.append((Ik, rk))
        resid[frame] = lst
    toy_rows = []
    for gname, gv in toys.items():
        gv = gv - gv.mean()
        gv = gv / np.sqrt((gv * gv).mean())
        row = dict(term=gname)
        # equivalent S0 (per unit rms of g) = z / sqrt(I_true)
        row["true_apex_template_bias_S0_per_unit_rms"] = float((W * r_true * gv).sum() / I_true)
        for frame in ("sidereal", "anti-sidereal"):
            z = np.array([(W * rk * gv).sum() / math.sqrt(Ik) for Ik, rk in resid[frame]])
            row[f"max_abs_equiv_S0_{frame}"] = float(np.abs(z).max() / math.sqrt(I_true))
        toy_rows.append(row)
        print(row)
    summary["seasonal_diurnal_frames"] = toy_rows
    del resid

    # ---- occultation step templates of point sources ---------------------
    cos_lim = -math.cos(math.asin(R_XRAY_BLOCK / tl["a_km"]))
    T_static = wt.evaluate(rhat)
    occ_rows = []
    res_occ = {}
    projc = NU.Projector(W, None, None)                       # constant only
    for mname, pj in (("const", projc), (args.model, proj)):
        I_T, r_T = pj.info(T_static)
        Os, cs = [], []
        for name, (r, d) in SOURCES.items():
            s = C.radec_to_vec(r, d)
            vis = ((rhat * s[:, None]).sum(axis=0) > cos_lim).astype(float)
            I_O, r_O = pj.info(vis)
            cross = float((W * r_O * r_T).sum())
            rho = cross / math.sqrt(I_O * I_T)
            occ_rows.append(dict(model=mname, source=name, visible_fraction=float(vis.mean()),
                                 correlation=rho, degradation_if_fitted=1.0 / math.sqrt(1 - rho ** 2),
                                 bias_S0_per_unit_step_cts=cross / I_T,
                                 separation_from_apex_deg=summary["windmap"][f"separation_from_apex_{name}_deg"]))
            Os.append(r_O)
            cs.append(cross)
        G = np.array([[float((W * a * b_).sum()) for b_ in Os] for a in Os])
        cvec = np.array(cs)
        I_joint = I_T - cvec @ np.linalg.solve(G, cvec)
        res_occ[mname] = dict(joint_degradation_all_four=float(math.sqrt(I_T / I_joint)),
                              joint_degradation_cyg_pair=float(math.sqrt(I_T / (I_T - cvec[:2] @ np.linalg.solve(G[:2, :2], cvec[:2])))))
    summary["occultation"] = res_occ
    # idealised overlap: uniform sampling of the satellite direction over the sphere
    zz = np.linspace(-1, 1, 20001)
    Fz = wt.F(np.arccos(zz))
    step_ = (zz > cos_lim).astype(float)
    summary["occultation"]["ideal_same_direction_correlation_uniform_sphere"] = float(np.corrcoef(Fz, step_)[0, 1])

    with open(os.path.join(OUT, "s5_occultation_overlap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(occ_rows[0].keys()))
        w.writeheader()
        for r_ in occ_rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r_.items()})
    np.savez_compressed(os.path.join(OUT, "s5_windmap.npz"), ra=ra_g, dec=dec_g, amp_sid=amp_sid, snr_sid=snr_sid,
                        amp_anti=amp_anti, snr_anti=snr_anti, dra=dra, cut_ra=cut_ra, ddec=ddec, cut_dec=cut_dec,
                        apex=np.array([ra0, dec0]), best=best)
    summary["runtime_s"] = time.time() - t_start
    with open(os.path.join(OUT, "s5_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    for r_ in occ_rows:
        print(r_)
    make_figure()
    print(f"total runtime {time.time() - t_start:.0f} s")


def make_figure():
    d = np.load(os.path.join(OUT, "s5_windmap.npz"))
    summ = json.load(open(os.path.join(OUT, "s5_summary.json")))
    P.apply()
    import matplotlib.pyplot as plt
    ra, dec = d["ra"], d["dec"]
    step = ra[1] - ra[0]
    ra_e = np.append(ra - step / 2, 360.0)
    dec_e = np.append(dec - step / 2, 90.0)
    X, Y = np.meshgrid(np.radians(180.0 - ra_e), np.radians(dec_e))       # RA increases to the left, centre RA = 180

    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.08)
    best = float(d["best"])
    ims = []
    for k, (key, title) in enumerate((("snr_sid", "(a) Sidereal frame: fraction of the maximum significance recovered"),
                                      ("snr_anti", "(b) Anti-sidereal frame (null test): same scale"))):
        ax = fig.add_subplot(gs[0, k], projection="mollweide")
        im = ax.pcolormesh(X, Y, d[key] / best, cmap=P.DIV_BLUE_RED, vmin=-1, vmax=1, shading="flat", rasterized=True)
        ims.append(im)
        ax.set_title(title, fontsize=9.6, pad=12)
        ax.grid(True, color=P.GRID, lw=0.5)
        ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
        ax.set_xticklabels(["300", "240", "180", "120", "60"], fontsize=7.5, color=P.INK2)
        ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
        ax.tick_params(axis="y", labelsize=7.5)
        if k == 0:
            a_ra, a_dec = d["apex"]
            ax.plot(np.radians(180 - a_ra), np.radians(a_dec), marker="*", ms=13, color=P.INK, mec=P.SURFACE, mew=0.8, ls="none")
            ax.annotate("wind apex", (np.radians(180 - a_ra), np.radians(a_dec)), xytext=(18, 14), textcoords="offset points",
                        fontsize=8, color=P.INK, arrowprops=dict(arrowstyle="-", color=P.INK2, lw=0.6))
            offs = {"Cyg X-1": (-58, -22), "Cyg X-3": (-62, 4), "Crab": (10, 10), "Galactic centre": (12, -14)}
            for name, (r, de) in SOURCES.items():
                ax.plot(np.radians(180 - r), np.radians(de), marker="o", ms=5.5, color=P.SURFACE, mec=P.INK, mew=1.1, ls="none")
                ax.annotate(name, (np.radians(180 - r), np.radians(de)), xytext=offs[name], textcoords="offset points",
                            fontsize=8, color=P.INK, arrowprops=dict(arrowstyle="-", color=P.INK2, lw=0.6))
    cax = fig.add_axes([0.33, 0.50, 0.34, 0.014])
    cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cb.set_label("recovered significance / maximum (trial template centred on each sky direction)", fontsize=8.5, color=P.INK2)
    cb.outline.set_edgecolor(P.AXIS)

    ax = fig.add_subplot(gs[1, 0])
    cd = math.cos(math.radians(float(d["apex"][1])))
    ax.plot(d["dra"] * cd, d["cut_ra"] / best, color=P.BLUE, label="cut along right ascension (offset x cos Dec)")
    ax.plot(d["ddec"] - float(d["apex"][1]), d["cut_dec"] / best, color=P.ORANGE, label="cut along declination")
    for nsig, ls in ((5, (0, (4, 2))), (10, (0, (1, 1.5)))):
        lv = summ["windmap"]["localisation_68"][f"{nsig}sigma"]["level"]
        ax.axhline(lv, color=P.MUTED, lw=1.0, ls=ls)
        ax.text(-88, lv - 0.012, f"68% region for a {nsig} sigma detection", fontsize=7.5, color=P.INK2, va="top")
    ax.set_xlim(-90, 60)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("offset of the trial direction from the true apex (deg on the sky)")
    ax.set_ylabel("recovered significance / maximum")
    ax.set_title("(c) Cuts through the apex: the 'beam' of the wind map", fontsize=9.6)
    ax.legend(loc="lower center", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    w = summ["windmap"]
    l5, l10 = w["localisation_68"]["5sigma"], w["localisation_68"]["10sigma"]
    txt = (
        f"Nuisance model {summ['model']}, all non-SAA data, {summ['dt']:.0f}-s bins\n\n"
        f"FWHM of the beam:  {w['fwhm_ra_on_sky_deg']:.0f} deg along RA (on the sky),  {w['fwhm_dec_deg']:.0f} deg along Dec\n"
        f"68% localisation, 5 sigma detection:   RA -{l5['ra_halfwidth_on_sky_deg'][1]:.0f}/+{l5['ra_halfwidth_on_sky_deg'][0]:.0f} deg (sky),"
        f"  Dec -{l5['dec_halfwidth_deg'][1]:.0f}/+{l5['dec_halfwidth_deg'][0]:.0f} deg\n"
        f"68% localisation, 10 sigma detection:  RA -{l10['ra_halfwidth_on_sky_deg'][1]:.0f}/+{l10['ra_halfwidth_on_sky_deg'][0]:.0f} deg (sky),"
        f"  Dec -{l10['dec_halfwidth_deg'][1]:.0f}/+{l10['dec_halfwidth_deg'][0]:.0f} deg\n\n"
        f"Significance fraction recovered by a template centred on\n"
        f"   Cyg X-3: {w['snr_fraction_at_Cyg X-3']:.2f}    Cyg X-1: {w['snr_fraction_at_Cyg X-1']:.2f}\n"
        f"   Galactic centre: {w['snr_fraction_at_Galactic centre']:.2f}    Crab: {w['snr_fraction_at_Crab']:.2f}\n\n"
        f"Anti-sidereal frame: max |fraction| = {w['anti_sidereal_max_abs_snr_fraction']:.3f}"
        f" (rms {w['anti_sidereal_rms_snr_fraction']:.3f})")
    ax.text(0.0, 0.98, txt, fontsize=8.8, color=P.INK2, va="top", ha="left", linespacing=1.5, transform=ax.transAxes)
    P.note(fig, "Simulation (forecast). Injected: full opaque-Earth DM template; trial templates: static apex at each direction, same speed. "
                "Mean circular orbit, toy background; grid step of the maps as in the run options.")
    fig.savefig(os.path.join(FIG, "fig4_windmap_psf.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
