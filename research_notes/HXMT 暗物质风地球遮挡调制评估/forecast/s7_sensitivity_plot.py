#!/usr/bin/env python3
"""
s7_sensitivity_plot.py -- Task 8: projected HXMT reach in the
(DM mass, per-nucleus contact cross-section) plane.

ONE COMMAND re-draws the figure:

    python3 s7_sensitivity_plot.py

It overlays existing constraints if the file

    ../existing_constraints.csv
    (= research_notes/HXMT 暗物质风地球遮挡调制评估/existing_constraints.csv)

exists when the script runs (columns: label,convention,f,m_GeV,sigma_min_cm2,
sigma_max_cm2,source_url,quality; only rows with convention == "contact" and
f == 1 are used; each row says "at this mass sigma_min..sigma_max is excluded").
If the file is absent or has no usable rows the figure is drawn without them
and says so.  The CSV is treated as data only (parsed with the csv module).

The HXMT regions are FORECASTS for the regime "Earth opaque, every DM particle
that crosses a crystal is recorded", bounded
  below  by sigma_min (detectability; adjustable in sensitivity_lib.py),
  above  by satellite self-shielding sigma/m (adjustable),
  right  by the mass reach m = rho <w> A_eff / S0_limit, which shrinks where the
         Earth becomes transparent (shadow depth from s1, PREM density model).

Options:  --constraints PATH  --B_NaI 500 --B_CsI 8000 --tlive 2.5e8 --model R --out PATH
"""
import argparse
import csv
import math
import os

import numpy as np

import plotstyle as P
import sensitivity_lib as S

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(HERE, "..", "existing_constraints.csv"))
FIG = os.path.join(HERE, "figures")
quality = {}      # label -> quality flag of the CSV rows (table / figure_readoff / estimate)

M_RANGE = (1e2, 1e20)          # GeV
SIG_RANGE = (1e-26, 1e-7)      # cm^2


def read_constraints(path):
    """Return ({label: [(m, smin, smax), ...]}, message)."""
    if not os.path.exists(path):
        return {}, "existing limits not overlaid (existing_constraints.csv not found)"
    groups, n_bad, n_other = {}, 0, 0
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                try:
                    if row.get("convention", "").lower() != "contact":
                        n_other += 1
                        continue
                    if abs(float(row.get("f", "nan")) - 1.0) > 1e-9:
                        n_other += 1
                        continue
                    m, lo, hi = float(row["m_GeV"]), float(row["sigma_min_cm2"]), float(row["sigma_max_cm2"])
                    if not (m > 0 and lo > 0 and hi > lo and all(map(math.isfinite, (m, lo, hi)))):
                        raise ValueError
                except (KeyError, ValueError):
                    n_bad += 1
                    continue
                lab_ = row.get("label", "unlabelled") or "unlabelled"
                groups.setdefault(lab_, []).append((m, lo, hi))
                quality[lab_] = row.get("quality", "table")
    except Exception as exc:                        # unreadable file: draw without it
        return {}, f"existing limits not overlaid (could not read CSV: {type(exc).__name__})"
    for k in groups:
        groups[k].sort()
    if not groups:
        return {}, "existing limits not overlaid (no usable rows with convention='contact', f=1)"
    msg = f"existing limits: {len(groups)} set(s) from existing_constraints.csv"
    if n_bad or n_other:
        msg += f" ({n_other} rows of other conventions / f, {n_bad} unusable rows skipped)"
    return groups, msg


def region_mask(M, SG, sigma_min, m_reach, som_grid, depth, need_shadow=True):
    """Boolean mask of the projected region on the (M, SG) mesh."""
    ratio = SG / M
    ok = (SG >= sigma_min) & (ratio <= S.SELF_SHIELD_SIGMA_OVER_M)
    if need_shadow:
        d = np.interp(np.log10(ratio), np.log10(som_grid), depth, left=0.0, right=1.0)
        ok &= M <= m_reach * d
    else:
        ok &= M <= m_reach
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", default=DEFAULT_CSV)
    ap.add_argument("--B_NaI", type=float, default=500.0)
    ap.add_argument("--B_CsI", type=float, default=8000.0)
    ap.add_argument("--tlive", type=float, default=2.5e8)
    ap.add_argument("--model", default="R", help="nuisance model whose r_eff/r_plain are used (R, E, K, const)")
    ap.add_argument("--out", default=os.path.join(FIG, "fig6_sensitivity_mass_sigma.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fis = S.load_fisher()
    sel = S.LIVE_TIMES.get(args.tlive, "all non-SAA")
    r = fis.get((sel, args.model)) or fis[("all non-SAA", "R")]
    r_eff, r_plain = r["r_eff_node"], r["r_plain_node"]
    som_grid, depth, lv = S.load_earth_opacity()
    groups, msg = read_constraints(args.constraints)
    print(msg)
    # merge different published versions of the same experiment: short label = text before " ("
    short_groups = {}
    for label, pts in groups.items():
        short = label.split(" (")[0].strip()
        short_groups.setdefault(short, []).append((pts, quality.get(label, "table")))

    P.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    lm = np.linspace(math.log10(M_RANGE[0]), math.log10(M_RANGE[1]), 1081)
    ls = np.linspace(math.log10(SIG_RANGE[0]), math.log10(SIG_RANGE[1]), 1141)
    M, SG = np.meshgrid(10 ** lm, 10 ** ls)
    eps_list = [0.0, 1e-5, 1e-4, 1e-3]
    shades = [P.SEQ_BLUE(x) for x in (0.18, 0.40, 0.62, 0.86)]

    fig = plt.figure(figsize=(17.2, 8.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.52], wspace=0.06)
    axs = [fig.add_subplot(gs[0, 0])]
    axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))
    axk = fig.add_subplot(gs[0, 2])
    axk.axis("off")
    plt.setp(axs[1].get_yticklabels(), visible=False)
    for ax, det, B in ((axs[0], "NaI", args.B_NaI), (axs[1], "CsI", args.B_CsI)):
        a_eff = S.A_EFF_CM2[det]
        smin = S.SIGMA_MIN[det]
        # ---- existing constraints: light grey overlays, numbered per experiment group ----
        LS = {"table": "-", "figure_readoff": (0, (5, 2)), "estimate": (0, (1, 1.5))}
        for gi, (short, sets) in enumerate(short_groups.items(), start=1):
            for pts, qual in sets:
                m = np.array([p_[0] for p_ in pts])
                lo = np.array([p_[1] for p_ in pts])
                hi = np.array([p_[2] for p_ in pts])
                if len(m) == 1:
                    ax.plot([m[0], m[0]], [lo[0], hi[0]], color=P.MUTED, lw=1.5, zorder=4.5)
                else:
                    ax.fill(np.concatenate([m, m[::-1]]), np.concatenate([hi, lo[::-1]]), facecolor=P.MUTED,
                            edgecolor="none", alpha=0.10, zorder=4.2)
                    ax.plot(np.concatenate([m, m[::-1], m[:1]]), np.concatenate([hi, lo[::-1], hi[:1]]),
                            color=P.INK2, lw=0.7, ls=LS.get(qual, "-"), zorder=4.5)
            pts = sets[0][0]
            jm = len(pts) // 2
            xm = min(max(pts[jm][0], M_RANGE[0] * 3), M_RANGE[1] / 3)
            ym = min(max(pts[jm][1] * 1.6, SIG_RANGE[0] * 2), SIG_RANGE[1] / 3)
            ax.text(xm, ym, str(gi), fontsize=6.8, color=P.INK, ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="circle,pad=0.18", fc=P.SURFACE, ec=P.INK2, lw=0.6))
        # ---- HXMT projected regions (nested, ordered by systematic floor) --
        handles = []
        for eps, col in zip(eps_list, shades):
            tot, stat, sys_ = S.sigma_S0(B, args.tlive, r_eff, r_plain, eps)
            m_reach = S.mass_reach(S.Z90 * tot, a_eff)
            mask = region_mask(M, SG, smin, m_reach, som_grid, depth)
            ax.contourf(M, SG, mask.astype(float), levels=[0.5, 1.5], colors=[col], alpha=0.92, zorder=2)
            ax.contour(M, SG, mask.astype(float), levels=[0.5], colors=[P.SURFACE], linewidths=1.0, zorder=3)
            lab = ("statistics only" if eps == 0 else f"systematic floor {eps:.0e} B")
            handles.append(Patch(fc=col, ec=P.SURFACE, label=f"{lab}:  m up to {m_reach:.1e} GeV"))
            print(f"{det}: eps={eps:g}  S0(90% UL)={S.Z90 * tot:.3e} c/s  m_reach(f=1)={m_reach:.2e} GeV")
        # ---- background-free flux limit and raw-rate bound ------------------
        m_bf = S.one_event_mass(a_eff, args.tlive, 2.303)
        mask_bf = region_mask(M, SG, smin, m_bf, som_grid, depth, need_shadow=False)
        ax.contour(M, SG, mask_bf.astype(float), levels=[0.5], colors=[P.ORANGE], linewidths=1.6, zorder=4)
        handles.append(Line2D([0], [0], color=P.ORANGE, lw=1.6,
                              label=f"flux limit, background-free signature\n(2.3 crossings): m up to {m_bf:.1e} GeV"))
        m_raw = S.raw_rate_mass(B, a_eff)
        ax.plot([m_raw, m_raw], [smin, S.SELF_SHIELD_SIGMA_OVER_M * m_raw], color=P.INK, lw=1.3, ls=(0, (2, 1.5)), zorder=5)
        handles.append(Line2D([0], [0], color=P.INK, lw=1.3, ls=(0, (2, 1.5)),
                              label=f"DM rate = whole observed rate ({B:g} c/s): m = {m_raw:.1e} GeV"))
        # ---- analytic lines ---------------------------------------------------
        mm = np.array(M_RANGE)
        ax.plot(mm, S.SELF_SHIELD_SIGMA_OVER_M * mm, color=P.INK2, lw=1.2, zorder=5)
        ax.plot(mm, S.ATM_CEILING_SIGMA_OVER_M * mm, color=P.RED, lw=1.2, zorder=5)
        ax.fill_between(mm, lv["10%"] * mm, lv["90%"] * mm, color=P.AQUA, alpha=0.25, lw=0, zorder=1.5)
        ax.plot(mm, lv["50%"] * mm, color=P.AQUA, lw=1.2, zorder=5)
        ax.axhline(smin, color=P.INK2, lw=0.8, ls=(0, (1, 2)), zorder=5)

        def lab_line(k, m_at, text, dy=1.0, **kw):
            # on-screen angle of a slope-1 line in these log-log axes
            ndx = math.log10(M_RANGE[1] / M_RANGE[0])
            ndy = math.log10(SIG_RANGE[1] / SIG_RANGE[0])
            ang = math.degrees(math.atan((ax.bbox.height / ndy) / (ax.bbox.width / ndx)))
            ax.text(m_at, k * m_at * dy, text, rotation=ang, rotation_mode="anchor", fontsize=7.6, color=P.INK2,
                    ha="left", va="bottom", zorder=7, **kw)

        fig.canvas.draw()
        lab_line(S.SELF_SHIELD_SIGMA_OVER_M, 3e2, f"satellite self-shielding, sigma/m = {S.SELF_SHIELD_SIGMA_OVER_M:.0e} cm$^2$/GeV", 2.2)
        lab_line(S.ATM_CEILING_SIGMA_OVER_M, 3e7, f"sea-level ceiling (1033 g/cm$^2$ of air), sigma/m = {S.ATM_CEILING_SIGMA_OVER_M:.1e}", 2.2)
        lab_line(lv["50%"], 2e13, f"Earth opaque above this line (PREM), sigma/m = {lv['50%']:.1e}", 4.0)
        ax.text(1.5e2, smin * 1.5, f"detectability floor sigma_min = {smin:.0e} cm$^2$ (assumed)", fontsize=7.6, color=P.INK2,
                va="bottom", zorder=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*M_RANGE)
        ax.set_ylim(*SIG_RANGE)
        ax.set_xlabel("DM mass m (GeV)")
        ax.set_title(f"{det}: A_eff = {a_eff:.0f} cm$^2$, B = {B:g} counts/s, live time {args.tlive:.1e} s", fontsize=10)
        leg = ax.legend(handles=handles, loc="lower right", fontsize=7.4, frameon=True, framealpha=0.92, facecolor=P.SURFACE,
                        edgecolor=P.GRID, title="HXMT projection (90% CL, f = 1, model " + args.model + ")", title_fontsize=7.8)
        leg.set_zorder(20)
    axs[0].set_ylabel("contact cross-section per nucleus, sigma (cm$^2$)")
    # key of the numbered existing constraints
    if short_groups:
        lines = ["Existing constraints (f = 1, contact convention)", "outline: solid = tabulated, dashed = read off a figure,",
                 "dotted = estimate/extrapolation", ""]
        for gi, (short, sets) in enumerate(short_groups.items(), start=1):
            nm = short if len(short) <= 46 else short[:44] + ".."
            lines.append(f"{gi:2d}  {nm}" + (f"  [{len(sets)} versions]" if len(sets) > 1 else ""))
        axk.text(0.0, 1.0, "\n".join(lines), fontsize=7.4, color=P.INK2, va="top", ha="left", linespacing=1.45,
                 transform=axk.transAxes)
    else:
        axk.text(0.0, 1.0, "existing limits not overlaid", fontsize=11, color=P.RED, va="top", ha="left",
                 transform=axk.transAxes, weight="bold")
        axk.text(0.0, 0.94, msg, fontsize=7.4, color=P.INK2, va="top", ha="left", transform=axk.transAxes, wrap=True)
    fig.suptitle("Projected reach of an Earth-shadow DM-wind modulation search with Insight-HXMT/HE",
                 fontsize=11, color=P.INK, x=0.01, ha="left", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.115)
    P.note(fig, msg, y=0.028)
    P.note(fig, "FORECAST, not a result from HXMT data. Assumes: Earth opaque; every DM particle crossing a crystal is recorded; SHM (0.3 GeV/cm3, v0 = 238 km/s); "
                "sigma_min and self-shielding are adjustable inputs (sensitivity_lib.py); right edges scale as 1/(S0 limit).")
    fig.savefig(args.out)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
