#!/usr/bin/env python3
"""
s8_subcomponent.py -- realistic HXMT sensitivity to a RARE strongly-interacting
dark-matter sub-component (halo fraction f << 1), contact cross-section per nucleus.

ONE COMMAND:

    python3 s8_subcomponent.py          (~20 s)

Combines three existing sets of notes (nothing is re-derived here):
  * rate limits  S0_limit(B, t_live, model R, eps)        <- sensitivity_lib.py / s4 output
  * recording probability per crossing versus sigma       <- ../detector_response.md Tables 5.2, 5.3
    (two branches: A = pure contact, B = contact x Helm form factor)
  * existing f-dependent limits                           <- ../existing_constraints.csv (contact rows)

    f_min(m, sigma) = S0_limit * m / ( rho<w> * A_geo * P_rec(sigma) * D(sigma/m) )

with D = relative Earth-shadow depth (s1) and a hard self-shielding ceiling in sigma/m.

Writes
  figures/fig7_fmin_vs_mass.png
  figures/fig8_subcomponent_m_sigma.png
  outputs/s8_tables.md, outputs/s8_summary.json

EVERYTHING IS A FORECAST UNDER STATED ASSUMPTIONS -- not an analysis of HXMT data.
The CSV is parsed as data only.
"""
import csv
import json
import math
import os

import numpy as np

import plotstyle as P
import sensitivity_lib as S

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(HERE, "..", "existing_constraints.csv"))
FIG = os.path.join(HERE, "figures")
OUT = S.OUT

# ----------------------------------------------------------------------------
# INPUTS TAKEN FROM THE NOTES (adjustable)
# ----------------------------------------------------------------------------
T_LIVE = 2.5e8                      # s, "all non-SAA" (forecast_simulation.md sec. 7)
SELECTION, MODEL = "all non-SAA", "R"
EPS_LIST = [0.0, 1e-4, 1e-3]
EPS_NAME = {0.0: "statistics only", 1e-4: "floor 1e-4 B", 1e-3: "floor 1e-3 B"}

# geometric (orientation-averaged) areas, cm^2
A_PHOSWICH = 3720.0                 # 18 phoswich modules, S/4 (detector_response.md Table 5.2 / 6.1)
A_VETO = S.A_EFF_CM2["veto"]        # 12750 (forecast; = 6750 top + 6000 side in Table 6.1)

# background rate of each channel, counts/s : (central, low, high)
B_CH = {
    "NaI": (500.0, 400.0, 540.0),        # published: 17 detectors, 30-300 keV, after anticoincidence
    "CsI": (8000.0, 5000.0, 12000.0),    # published total of 18 CsI
    "veto": (6000.0, 2000.0, 20000.0),   # ASSUMPTION (unpublished): 18 panels x ~1e2-1e3 c/s (speculation in
}                                        # systematics_and_analysis_design.md sec. 1)

# recording probability per phoswich crossing, detector_response.md Table 5.2 (central QF, normal gain)
SIG_HE = np.array([1e-24, 3.2e-24, 1e-23, 3.2e-23, 1e-22, 3.2e-22, 1e-21, 3.2e-21, 1e-20, 3.2e-20, 1e-19, 1e-18, 1e-17])
P_HE = {
    ("A", "NaI"): [2.8e-3, 7.3e-3, 0.014, 0.015, 0.017, 0.021, 0.019, 0.011, 4.7e-3, 1.8e-3, 5.9e-4, 8e-5, 5e-6],
    ("A", "mix"): [6.1e-4, 4.6e-3, 0.032, 0.124, 0.197, 0.115, 0.034, 0.006, 8e-4, 1e-4, 1e-5, 0, 0],
    ("A", "CsI"): [6.0e-3, 0.026, 0.123, 0.383, 0.615, 0.487, 0.150, 0.037, 0.011, 3.4e-3, 1.1e-3, 9e-5, 7e-6],
    ("B", "NaI"): [5.2e-4, 1.8e-3, 6.3e-3, 0.021, 0.076, 0.229, 0.392, 0.337, 0.037, 0.005, 0.002, 2e-4, 2e-5],
    ("B", "mix"): [0, 0, 0, 1e-4, 1.7e-3, 0.029, 0.199, 0.245, 0.156, 0.002, 0, 0, 0],
    ("B", "CsI"): [0, 0, 0, 0, 0, 0.002, 0.030, 0.100, 0.196, 0.215, 0.138, 0.016, 1.7e-3],
}
# veto panels: P(E_ee >= 100 keVee) per panel crossing, Table 5.3 (central); model B differs by < 2%
SIG_HVT = np.array([1e-22, 3.2e-22, 1e-21, 3.2e-21, 1e-20, 3.2e-20, 1e-19])
P_HVT = [4e-4, 0.005, 0.050, 0.410, 0.872, 0.984, 0.998]
P_FLOOR = 1e-12                     # "0" in the tables

# self-shielding ceilings sigma/m (cm^2/GeV), hard cut (detector_response.md Table 6.2 and its summary)
CEIL = {
    "optimistic": {"HE": 5e-25, "veto": 3e-24},     # HE "most directions (1-5)e-25"; top panels 1/e at 3e-24
    "pessimistic": {"HE": 1e-25, "veto": 4.5e-25},  # lower end for HE; side panels (1/e)
    "worst": {"HE": 2e-26, "veto": 2e-26},          # everything must cross a neighbouring module / the platform
}
CEIL_ENVELOPE = (2e-26, 3e-24)      # range quoted in the notes
M_MIN_HXMT = 1e3                    # GeV: below this the kinetic energy cannot pass the HE thresholds (notes)

# window W1 of constraints_and_theory.md sec. 4
W1 = dict(m=(1e3, 6e7), s_lo=3e-22, atm=3e-27, s_hi=1.7e-19, shield=3e-24)

BENCH_M = [1e4, 1e5, 1e6, 1e7]
BENCH_S = [1e-22, 1e-21, 1e-20]
F_FIXED = [1e-3, 1e-6]
IMP_TAG = "IMP 7/8"                 # the doubtful unpublished recast (flagged in constraints_and_theory.md)

# analysis grid (decades): area of "new" parameter space is counted inside this box
LM = np.arange(3.0, 10.0 + 1e-9, 0.01)
LS = np.arange(-24.0, -16.0 + 1e-9, 0.01)
CELL = 0.01 * 0.01


# ----------------------------------------------------------------------------
# detector response
# ----------------------------------------------------------------------------
def _loginterp(x, xp, fp, low_slope, high_slope):
    """log-log interpolation; power-law extrapolation with the given slopes."""
    lx, lxp = np.log10(x), np.log10(xp)
    lfp = np.log10(np.maximum(fp, P_FLOOR))
    y = np.interp(lx, lxp, lfp)
    y = np.where(lx < lxp[0], lfp[0] + low_slope * (lx - lxp[0]), y)
    y = np.where(lx > lxp[-1], lfp[-1] + high_slope * (lx - lxp[-1]), y)
    return np.clip(10 ** y, 0.0, 1.0)


def p_record(channel, branch, sigma):
    """probability that one crossing is recorded in this channel (central values of the notes)."""
    sigma = np.asarray(sigma, float)
    if channel == "veto":
        return _loginterp(sigma, SIG_HVT, np.array(P_HVT), 3.0, 0.0)      # steep below 1e-22, 1 above 1e-19
    return _loginterp(sigma, SIG_HE, np.array(P_HE[(branch, channel)]), 1.0, -1.0)   # ~sigma below, ~1/sigma above


A_GEO = {"NaI": A_PHOSWICH, "CsI": A_PHOSWICH, "veto": A_VETO}
GROUP = {"NaI": "HE", "CsI": "HE", "veto": "veto"}
CHANNELS = ["NaI", "CsI", "veto"]


def s0_limit(B, eps, r):
    tot, stat, sys_ = S.sigma_S0(B, T_LIVE, r["r_eff_node"], r["r_plain_node"], eps)
    return S.Z90 * tot


def fmin_channel(channel, branch, eps, M, SG, r, depth_fn, ceiling=None, B=None):
    """90% CL minimum halo fraction for one channel on arrays M (GeV), SG (cm^2)."""
    B = B_CH[channel][0] if B is None else B
    flux = S.RHO_DM * S.MEAN_SPEED_KMS * 1e5                # GeV cm^-2 s^-1 for f = 1, times 1/m
    eff = p_record(channel, branch, SG) * depth_fn(SG / M)
    with np.errstate(divide="ignore", over="ignore"):
        f = s0_limit(B, eps, r) * M / (flux * A_GEO[channel] * eff)
    f = np.where(M < M_MIN_HXMT, np.inf, f)
    if ceiling is not None:
        f = np.where(SG / M > CEIL[ceiling][GROUP[channel]], np.inf, f)
    return f


def fmin_best(branch, eps, M, SG, r, depth_fn, ceiling):
    fs = np.array([fmin_channel(c, branch, eps, M, SG, r, depth_fn, ceiling) for c in CHANNELS])
    return fs.min(axis=0), fs.argmin(axis=0)


# ----------------------------------------------------------------------------
# existing constraints (contact convention), morphing in log f
# ----------------------------------------------------------------------------
def read_csv(path=CSV_PATH):
    """label -> {f: (log m, log sigma_min, log sigma_max) sorted by mass}."""
    lab = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                if row["convention"].strip().lower() != "contact":
                    continue
                f, m = float(row["f"]), float(row["m_GeV"])
                lo, hi = float(row["sigma_min_cm2"]), float(row["sigma_max_cm2"])
                if not (m > 0 and 0 < lo < hi):
                    continue
            except (KeyError, ValueError):
                continue
            lab.setdefault(row["label"].strip(), {}).setdefault(f, []).append((math.log10(m), math.log10(lo), math.log10(hi)))
    out = {}
    for k, d in lab.items():
        out[k] = {}
        for f, rows in d.items():
            a = np.array(sorted(rows))
            if len(a) >= 2:                      # single-mass entries have zero measure in the plane
                out[k][f] = a
        if not out[k]:
            del out[k]
    return out


def _mask_rows(a, lm, ls):
    lo = np.interp(lm, a[:, 0], a[:, 1])
    hi = np.interp(lm, a[:, 0], a[:, 2])
    return (lm >= a[0, 0]) & (lm <= a[-1, 0]) & (ls >= lo) & (ls <= hi)


def label_mask(d, f, lm, ls):
    """Is (m, sigma) excluded by this limit for halo fraction f?

    Rows exist only at f = 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8 (not every label at every f).
      f on a tabulated value            -> the rows themselves
      f above the largest tabulated f   -> rows of the largest f (an exclusion can only grow with f)
      f below the smallest tabulated f  -> not excluded (limit treated as vanished)
      f between two tabulated values    -> the two regions are morphed linearly in log f: the mass range
                                           [m_lo, m_hi] is interpolated in log m, and at the same fractional
                                           position inside the range log sigma_min / log sigma_max are interpolated.
    """
    fs = sorted(d)
    lf = math.log10(f)
    for g in fs:
        if abs(math.log10(g) - lf) < 1e-9:
            return _mask_rows(d[g], lm, ls)
    if f > fs[-1]:
        return _mask_rows(d[fs[-1]], lm, ls)
    if f < fs[0]:
        return np.zeros(np.broadcast(lm, ls).shape, bool)
    j = int(np.searchsorted(fs, f))
    fa, fb = fs[j - 1], fs[j]
    a, b = d[fa], d[fb]
    t = (lf - math.log10(fa)) / (math.log10(fb) - math.log10(fa))
    r0 = (1 - t) * a[0, 0] + t * b[0, 0]
    r1 = (1 - t) * a[-1, 0] + t * b[-1, 0]
    u = np.clip((lm - r0) / max(r1 - r0, 1e-12), 0.0, 1.0)
    ma = a[0, 0] + u * (a[-1, 0] - a[0, 0])
    mb = b[0, 0] + u * (b[-1, 0] - b[0, 0])
    lo = (1 - t) * np.interp(ma, a[:, 0], a[:, 1]) + t * np.interp(mb, b[:, 0], b[:, 1])
    hi = (1 - t) * np.interp(ma, a[:, 0], a[:, 2]) + t * np.interp(mb, b[:, 0], b[:, 2])
    return (lm >= r0) & (lm <= r1) & (ls >= lo) & (ls <= hi)


def excluded(cons, f, lm, ls, skip_imp=False):
    m = np.zeros(np.broadcast(lm, ls).shape, bool)
    for k, d in cons.items():
        if skip_imp and IMP_TAG in k:
            continue
        m |= label_mask(d, f, lm, ls)
    return m


LF_SCAN = np.arange(0.0, -8.0 - 1e-9, -0.05)       # log10 f from 1 down to 1e-8


def f_existing(cons, lm, ls, skip_imp=False):
    """smallest f (on a 0.05-dex scan, 1 ... 1e-8) at which the union of CSV limits excludes the point.
    Returns (f, label); f = inf if the point is allowed even for f = 1."""
    lm, ls = np.asarray(lm, float), np.asarray(ls, float)
    best = np.full(np.broadcast(lm, ls).shape, np.inf)
    who = np.full(best.shape, -1)
    atfloor = np.zeros(best.shape, bool)       # True: the limit sits on the lowest f tabulated for that label
    keys = list(cons)
    for i, k in enumerate(keys):
        if skip_imp and IMP_TAG in k:
            continue
        cur = np.full(best.shape, np.inf)
        for lf in LF_SCAN:
            cur = np.where(label_mask(cons[k], 10 ** lf, lm, ls), 10 ** lf, cur)
        fl = cur <= min(cons[k]) * 1.001
        atfloor = np.where(cur < best, fl, atfloor)
        who = np.where(cur < best, i, who)
        best = np.minimum(best, cur)
    return best, who, keys, atfloor


# ----------------------------------------------------------------------------
def e(x, nd=1):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}e}"


def span(mask, axis_vals, axis):
    """(min, max) of the coordinate over the True cells."""
    idx = np.where(mask.any(axis=axis))[0]
    if len(idx) == 0:
        return None
    return float(10 ** axis_vals[idx[0]]), float(10 ** axis_vals[idx[-1]])


def w1_mask(LMg, LSg):
    m, s = 10 ** LMg, 10 ** LSg
    lo = np.maximum(W1["s_lo"], W1["atm"] * m)
    hi = np.minimum(W1["s_hi"], W1["shield"] * m)
    return (m >= W1["m"][0]) & (m <= W1["m"][1]) & (s >= lo) & (s <= hi)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    fis = S.load_fisher()
    r = fis[(SELECTION, MODEL)]
    som, dep, lv = S.load_earth_opacity()
    depth_fn = lambda ratio: np.interp(np.log10(ratio), np.log10(som), dep, left=0.0, right=1.0)
    cons = read_csv()
    flux = S.RHO_DM * S.MEAN_SPEED_KMS * 1e5
    md, summ = [], {}
    md.append("<!-- generated by s8_subcomponent.py : FORECAST under stated assumptions, not HXMT data -->\n")

    # ---------------- Table A: rate limits and peak f_min/m per channel -------------
    md.append("**Table A. 90% CL limit on the unshadowed DM rate S0 (counts/s) and best-case f_min/m per channel** "
              f"(t_live = {T_LIVE:.1e} s, model {MODEL}: r_eff = {r['r_eff_node']:.4f}, r_plain = {r['r_plain_node']:.4f}; "
              f"rho<w> = {flux:.3e} GeV cm^-2 s^-1)\n")
    md.append("| channel | branch | B (c/s) | peak P_rec (at sigma, cm²) | A_geo×P (cm²) | S0 limit ε=0 | ε=1e-4 | ε=1e-3 | f_min/m_GeV ε=0 | ε=1e-4 | ε=1e-3 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    sg_scan = 10 ** np.arange(-24, -16.001, 0.01)
    summ["channels"] = []
    for ch in CHANNELS:
        for br in ("A", "B"):
            if ch == "veto" and br == "B":
                continue
            p = p_record(ch, br, sg_scan)
            jp = int(np.argmax(p))
            for B in B_CH[ch]:
                lim = [s0_limit(B, ep, r) for ep in EPS_LIST]
                fm = [x / (flux * A_GEO[ch] * p[jp]) for x in lim]
                tag = "" if B == B_CH[ch][0] else " (range)"
                md.append(f"| {ch}{tag} | {'A/B' if ch == 'veto' else br} | {B:g} | {p[jp]:.3f} ({sg_scan[jp]:.1e}) | {A_GEO[ch] * p[jp]:.0f} | "
                          + " | ".join(f"{x:.3g}" for x in lim) + " | " + " | ".join(e(x, 2) for x in fm) + " |")
                summ["channels"].append(dict(channel=ch, branch=br, B=B, peak_P=float(p[jp]), sigma_peak=float(sg_scan[jp]),
                                             S0_limit=lim, fmin_over_m=fm))

    # ---------------- sanity checks ---------------------------------------------------
    chk = {}
    for det, B in (("NaI", 500.0), ("CsI", 8000.0)):
        chk[f"mass_reach_f1_{det}_P1"] = {str(ep): S.mass_reach(s0_limit(B, ep, r), S.A_EFF_CM2[det]) for ep in [0.0, 1e-5, 1e-4, 1e-3]}
    chk["fmin_over_m_NaI_P1_stat_90UL"] = s0_limit(500.0, 0.0, r) / S.flux_mass_product(S.A_EFF_CM2["NaI"])
    chk["fmin_over_m_NaI_P1_stat_5sigma"] = chk["fmin_over_m_NaI_P1_stat_90UL"] * S.Z5 / S.Z90
    # the constraint note's ideal number, re-evaluated with ITS inputs
    N_cross_coeff = 0.3 * 3e7 / 4 * 5103 * 1.8e8                  # f/m_GeV
    S_need = (2e6, 5e6)
    chk["constraint_note_ideal_fmin_over_m"] = [x / N_cross_coeff for x in S_need]
    # with response folded in: best over channels and sigma, statistics only
    for br in ("A", "B"):
        best = min(fmin_channel(c, br, 0.0, 1e4, sg_scan, r, lambda x: 1.0).min() for c in CHANNELS) / 1e4
        chk[f"fmin_over_m_best_response_stat_90UL_{br}"] = float(best)
        chk[f"mass_reach_f1_best_response_stat_{br}"] = 1.0 / float(best)
        for ep in (1e-4, 1e-3):
            best = min(fmin_channel(c, br, ep, 1e4, sg_scan, r, lambda x: 1.0).min() for c in CHANNELS) / 1e4
            chk[f"fmin_over_m_best_response_eps{ep:g}_{br}"] = float(best)
    summ["sanity"] = chk

    # ---------------- grids ---------------------------------------------------------------
    LMg, LSg = np.meshgrid(LM, LS)
    M, SG = 10 ** LMg, 10 ** LSg
    W = w1_mask(LMg, LSg)
    w1_area = float(W.sum() * CELL)
    FMIN = {}
    for br in ("A", "B"):
        for ep in EPS_LIST:
            for ce in CEIL:
                FMIN[(br, ep, ce)] = fmin_best(br, ep, M, SG, r, depth_fn, ce)[0]
    EXC = {(f, s): excluded(cons, f, LMg, LSg, skip_imp=s) for f in F_FIXED for s in (False, True)}

    # ---------------- Table B: surviving new area at fixed f ---------------------------------
    md.append("\n**Table B. HXMT-accessible and not yet excluded region at fixed f** (area in decades² inside "
              f"m = 1e{LM[0]:.0f}–1e{LM[-1]:.0f} GeV, sigma = 1e{LS[0]:.0f}–1e{LS[-1]:.0f} cm²; in brackets: IMP 7/8 estimate not used). "
              f"W1 area = {w1_area:.2f} decades².\n")
    md.append("| f | ε | branch | ceiling | accessible (dec²) | accessible & not excluded, whole box (dec²) | NEW = same, above the atmosphere line sigma/m ≥ 3e-27 (dec²) | mass range of NEW (GeV) | sigma range of NEW (cm²) | NEW with m ≥ 1e4 GeV (dec²) | NEW inside W1 (dec²) | share of W1 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    HEAVY = M >= 1e4
    ATM = SG / M >= W1["atm"]
    summ["fixed_f"] = []
    for f in F_FIXED:
        for ep in EPS_LIST:
            for br in ("A", "B"):
                for ce in CEIL:
                    acc = FMIN[(br, ep, ce)] <= f
                    new = acc & ~EXC[(f, False)]
                    new2 = acc & ~EXC[(f, True)]
                    a_acc, a_new, a_new2 = acc.sum() * CELL, new.sum() * CELL, new2.sum() * CELL
                    na, na2 = new & ATM, new2 & ATM
                    a_na, a_na2 = na.sum() * CELL, na2.sum() * CELL
                    ms, ss = span(na, LM, 0), span(na, LS, 1)
                    ms2, ss2 = span(na2, LM, 0), span(na2, LS, 1)
                    inw, inw2 = (new & W).sum() * CELL, (new2 & W).sum() * CELL
                    a_h, a_h2 = (na & HEAVY).sum() * CELL, (na2 & HEAVY).sum() * CELL
                    rng = lambda x: "none" if x is None else f"{x[0]:.1e}–{x[1]:.1e}"
                    md.append(f"| {f:g} | {ep:g} | {br} | {ce} | {a_acc:.2f} | {a_new:.2f} ({a_new2:.2f}) | {a_na:.2f} ({a_na2:.2f}) | {rng(ms)} ({rng(ms2)}) | "
                              f"{rng(ss)} ({rng(ss2)}) | {a_h:.2f} ({a_h2:.2f}) | {inw:.2f} ({inw2:.2f}) | {100 * inw / w1_area:.0f}% ({100 * inw2 / w1_area:.0f}%) |")
                    summ["fixed_f"].append(dict(f=f, eps=ep, branch=br, ceiling=ce, accessible=float(a_acc), new_box=float(a_new),
                                                new_box_noIMP=float(a_new2), new=float(a_na), new_noIMP=float(a_na2), new_m_ge_1e4=float(a_h), new_m_ge_1e4_noIMP=float(a_h2), m_range=ms, sigma_range=ss, m_range_noIMP=ms2,
                                                sigma_range_noIMP=ss2, new_in_W1=float(inw), new_in_W1_noIMP=float(inw2)))

    # ---------------- Table C: W1 survival for any f (HXMT f_min below the existing limit) -----
    sub = (slice(None, None, 4), slice(None, None, 4))          # 0.04-dex grid for the f scan
    lmc, lsc = LMg[sub], LSg[sub]
    Wc = W[sub]
    FE = {s: f_existing(cons, lmc, lsc, skip_imp=s)[0] for s in (False, True)}
    cellc = CELL * 16
    md.append("\n**Table C. How much of W1 survives: share of W1 (area "
              f"{Wc.sum() * cellc:.2f} decades²) where HXMT f_min is below the strongest existing limit on f** "
              "(existing limit capped at f = 1 where the CSV has none; in brackets: IMP 7/8 estimate not used)\n")
    md.append("| ε | branch | ceiling | share of W1 with any gain | with gain ≥ 1 decade | with gain ≥ 2 decades | median gain where >0 (decades) | max gain (decades) |")
    md.append("|---|---|---|---|---|---|---|---|")
    summ["W1"] = dict(area=float(Wc.sum() * cellc), rows=[])
    for ep in EPS_LIST:
        for br in ("A", "B"):
            for ce in CEIL:
                cells = []
                rec = dict(eps=ep, branch=br, ceiling=ce)
                for s in (False, True):
                    fe = np.minimum(FE[s], 1.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        gain = np.log10(fe / FMIN[(br, ep, ce)][sub])
                    g = gain[Wc]
                    n = max(len(g), 1)
                    pos = g[g > 0]
                    cells.append((100 * (g > 0).sum() / n, 100 * (g >= 1).sum() / n, 100 * (g >= 2).sum() / n,
                                  float(np.median(pos)) if len(pos) else float("nan"), float(g.max()) if len(g) else float("nan")))
                    rec["noIMP" if s else "all"] = dict(share_any=cells[-1][0], share_1dec=cells[-1][1], share_2dec=cells[-1][2],
                                                        median_gain=cells[-1][3], max_gain=cells[-1][4])
                a, b = cells
                md.append(f"| {ep:g} | {br} | {ce} | {a[0]:.0f}% ({b[0]:.0f}%) | {a[1]:.0f}% ({b[1]:.0f}%) | {a[2]:.0f}% ({b[2]:.0f}%) | "
                          f"{a[3]:.1f} ({b[3]:.1f}) | {a[4]:.1f} ({b[4]:.1f}) |")
                summ["W1"]["rows"].append(rec)

    # ---------------- Table D: benchmark points ----------------------------------------------------
    bm = np.array([[m, s] for m in BENCH_M for s in BENCH_S])
    blm, bls = np.log10(bm[:, 0]), np.log10(bm[:, 1])
    fe_all, who_all, keys, fl_all = f_existing(cons, blm, bls, False)
    fe_no, who_no, _, fl_no = f_existing(cons, blm, bls, True)
    summ["benchmarks"] = []
    for br in ("A", "B"):
        md.append(f"\n**Table D-{br}. Benchmark points, branch {br}: HXMT f_min (90% CL; best channel, optimistic ceiling; "
                  "† = lost with the pessimistic ceiling, ‡ = lost only with the worst-case 2e-26 ceiling) next to the strongest existing limit on f "
                  "(0.05-dex scan of the CSV, log-f morphing between tabulated f; '≤' = the point is still excluded at the lowest f tabulated "
                  "for that limit, so the true limit is stronger and the gain is an upper bound) and the gain log10(f_exist / f_min); "
                  "'none' = not excluded even for f = 1, gain then counted from f = 1**\n")
        md.append("| m (GeV) | sigma (cm²) | sigma/m | best channel | f_min ε=0 | ε=1e-4 | ε=1e-3 | existing limit on f (source) | same without IMP 7/8 | gain ε=0 | gain ε=1e-4 | gain ε=1e-3 | gain without IMP (ε=0 / 1e-4 / 1e-3) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (m, s) in enumerate(bm):
            fm, flag, chn = [], "", ""
            for ep in EPS_LIST:
                v, arg = fmin_best(br, ep, np.array([m]), np.array([s]), r, depth_fn, "optimistic")
                fm.append(float(v[0]))
                chn = CHANNELS[int(arg[0])]
            vp = fmin_best(br, 0.0, np.array([m]), np.array([s]), r, depth_fn, "pessimistic")[0][0]
            vw = fmin_best(br, 0.0, np.array([m]), np.array([s]), r, depth_fn, "worst")[0][0]
            if not np.isfinite(fm[0]):
                flag = " (above ceiling)"
            elif not np.isfinite(vp):
                flag = " †"
            elif not np.isfinite(vw):
                flag = " ‡"
            def lim(fe, who, fl):
                if not np.isfinite(fe):
                    return "none (>1)", 1.0, ""
                src = keys[int(who)].split(" (")[0].split(":")[0]
                return f"{'≤' if fl else ''}{fe:.1e} ({src})", fe, ("≤" if fl else "")
            t1, f1, q1 = lim(fe_all[i], who_all[i], fl_all[i])
            t2, f2, q2 = lim(fe_no[i], who_no[i], fl_no[i])
            g1 = [math.log10(f1 / x) if np.isfinite(x) else float("nan") for x in fm]
            g2 = [math.log10(f2 / x) if np.isfinite(x) else float("nan") for x in fm]
            gs = lambda g, q="": "—" if not np.isfinite(g) else f"{q}{g:+.1f}"
            md.append(f"| {m:.0e} | {s:.0e} | {s / m:.0e} | {chn if np.isfinite(fm[0]) else '—'}{flag} | " + " | ".join(e(x) for x in fm)
                      + f" | {t1} | {t2} | " + " | ".join(gs(g, q1) for g in g1) + " | " + " / ".join(gs(g, q2) for g in g2) + " |")
            summ["benchmarks"].append(dict(branch=br, m=m, sigma=s, channel=chn, flag=flag.strip(), fmin=fm, f_exist=f1, f_exist_noIMP=f2,
                                           src=t1, src_noIMP=t2, gain=g1, gain_noIMP=g2))

    # ---------------- Table E: veto-panel background range at the benchmarks -----------------------
    md.append("\n**Table E. Effect of the ASSUMED veto-panel rate (unpublished): veto-channel f_min at m = 1e6 GeV, sigma = 1e-20 cm²**\n")
    md.append("| B_veto (c/s) | ε=0 | ε=1e-4 | ε=1e-3 |")
    md.append("|---|---|---|---|")
    for B in sorted(B_CH["veto"]):
        md.append(f"| {B:g} | " + " | ".join(e(float(fmin_channel('veto', 'A', ep, np.array([1e6]), np.array([1e-20]), r, depth_fn, None, B)[0]), 2)
                                              for ep in EPS_LIST) + " |")

    with open(os.path.join(OUT, "s8_tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "s8_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else float(o))
    print("\n".join(md))
    print(json.dumps(chk, indent=1))

    # =========================== FIGURES =============================================================
    P.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    EC = {0.0: P.BLUE, 1e-4: P.ORANGE, 1e-3: P.RED}

    # ---- fig 7: f_min versus mass at three cross sections -------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.4), sharey=True)
    lm = np.arange(3.0, 9.0 + 1e-9, 0.02)
    mm = 10 ** lm
    for ax, s in zip(axs, BENCH_S):
        sv = np.full_like(mm, s)
        # existing limits on f at this sigma
        fe = f_existing(cons, lm, np.log10(sv), False)[0]
        fn = f_existing(cons, lm, np.log10(sv), True)[0]
        fe1, fn1 = np.minimum(fe, 1.0), np.minimum(fn, 1.0)
        ax.fill_between(mm, fn1, 1.0, step=None, color=P.MUTED, alpha=0.22, lw=0, zorder=1)
        ax.fill_between(mm, fe1, fn1, color=P.MUTED, alpha=0.10, lw=0, hatch="////", edgecolor=P.MUTED, zorder=1)
        ax.plot(mm, fn1, color=P.INK2, lw=1.0, zorder=2)
        ax.plot(mm, fe1, color=P.INK2, lw=1.0, ls=(0, (1, 1.5)), zorder=2)
        # ideal-statistics band of the constraint note
        ax.fill_between(mm, 1e-12 * mm, 2.4e-12 * mm, color=P.AQUA, alpha=0.30, lw=0, zorder=1.5)
        for br, ls_ in (("A", "-"), ("B", (0, (5, 2)))):
            for ep in EPS_LIST:
                fo = fmin_best(br, ep, mm, sv, r, depth_fn, "optimistic")[0]
                fp = fmin_best(br, ep, mm, sv, r, depth_fn, "pessimistic")[0]
                fo = np.where(fo > 3.0, np.nan, fo)
                fp = np.where(fp > 3.0, np.nan, fp)
                ax.plot(mm, fo, color=EC[ep], lw=1.1, ls=ls_, alpha=0.45, zorder=4)      # only with the optimistic ceiling
                ax.plot(mm, fp, color=EC[ep], lw=2.0, ls=ls_, zorder=5)                  # survives the pessimistic ceiling
        for ce, c in (("optimistic", 0.45), ("pessimistic", 1.0)):
            for grp, yy in (("HE", 2.5e-12), ("veto", 6e-12)):
                x = s / CEIL[ce][grp]
                if 1e3 < x < 1e9:
                    ax.plot([x], [yy], marker="^", ms=5, color=P.INK2, alpha=c, zorder=6, clip_on=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e3, 1e9)
        ax.set_ylim(1e-12, 3.0)
        ax.set_xlabel("sub-component mass m (GeV)")
        ax.set_title(f"contact cross-section per nucleus = {s:.0e} cm$^2$")
    axs[0].set_ylabel("halo fraction f  (HXMT: minimum detectable, 90% CL)")
    h = [Line2D([0], [0], color=EC[ep], lw=2, label=f"HXMT, {EPS_NAME[ep]}") for ep in EPS_LIST]
    h += [Line2D([0], [0], color=P.INK, lw=2, label="branch A: no nuclear form factor"),
          Line2D([0], [0], color=P.INK, lw=2, ls=(0, (5, 2)), label="branch B: with Helm form factor"),
          Line2D([0], [0], color=P.INK, lw=1.1, alpha=0.45, label="thin: reachable only with optimistic self-shielding"),
          Patch(fc=P.MUTED, alpha=0.30, label="excluded by existing limits (CSV, log-f interpolation)"),
          Patch(fc=P.MUTED, alpha=0.12, hatch="////", ec=P.MUTED, label="excluded only by the IMP 7/8 estimate"),
          Patch(fc=P.AQUA, alpha=0.30, label="earlier ideal estimate f = (1-2.4)e-12 m/GeV"),
          Line2D([0], [0], color=P.INK2, marker="^", ms=5, lw=0, label="self-shielding sets in below this mass (lower row HE, upper row veto; faint = optimistic)")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.05))
    fig.suptitle("Minimum detectable halo fraction of a strongly-interacting sub-component: Insight-HXMT Earth-shadow modulation search "
                 "(best of HE NaI-class, HE CsI-class, veto-panel rates)", fontsize=11, color=P.INK, x=0.01, ha="left", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.26, wspace=0.05)
    P.note(fig, "FORECAST under stated assumptions, not a result from HXMT data. 90% CL, live time 2.5e8 s, nuisance model R; B = 500 (NaI-class), 8000 (CsI-class), "
                "6000 counts/s (veto panels: ASSUMED, unpublished);\nrecording probability vs cross-section from detector_response.md Tables 5.2/5.3; "
                "self-shielding is applied as a hard cut in sigma/m (markers on the bottom axis); systematic floor = fraction of each channel's count rate. Flat steps of the grey edge at 1e-6 / 1e-8 = lowest f tabulated for that limit (true limit is stronger).", y=0.006)
    fig.savefig(os.path.join(FIG, "fig7_fmin_vs_mass.png"))
    plt.close(fig)

    # ---- fig 8: (m, sigma) plane at fixed f -------------------------------------------------------------
    shades = {0.0: P.SEQ_BLUE(0.22), 1e-4: P.SEQ_BLUE(0.55), 1e-3: P.SEQ_BLUE(0.92)}
    fig, axs = plt.subplots(2, 2, figsize=(15.5, 12.4), sharex=True, sharey=True)
    for i, f in enumerate(F_FIXED):
        for j, br in enumerate(("A", "B")):
            ax = axs[i, j]
            for ep in EPS_LIST:
                acc = (FMIN[(br, ep, "optimistic")] <= f).astype(float)
                if acc.any():
                    ax.contourf(M, SG, acc, levels=[0.5, 1.5], colors=[shades[ep]], zorder=2)
                    ax.contour(M, SG, acc, levels=[0.5], colors=[P.SURFACE], linewidths=0.8, zorder=2.5)
                accp = (FMIN[(br, ep, "pessimistic")] <= f).astype(float)
                if accp.any() and ep == 0.0:
                    ax.contour(M, SG, accp, levels=[0.5], colors=[P.INK], linewidths=1.0, linestyles=[(0, (4, 2))], zorder=6)
            ex_all, ex_no = EXC[(f, False)], EXC[(f, True)]
            ax.contourf(M, SG, ex_no.astype(float), levels=[0.5, 1.5], colors=[P.MUTED], alpha=0.62, zorder=3)
            imp_only = (ex_all & ~ex_no).astype(float)
            if imp_only.any():
                ax.contourf(M, SG, imp_only, levels=[0.5, 1.5], colors=[P.MUTED], alpha=0.25, hatches=["////"], zorder=3)
            ax.contour(M, SG, ex_all.astype(float), levels=[0.5], colors=[P.INK2], linewidths=0.7, zorder=3.5)
            ax.contour(M, SG, W.astype(float), levels=[0.5], colors=[P.ORANGE], linewidths=1.6, zorder=7)
            xx = np.array([1e3, 1e10])
            for k, lab_ in zip(CEIL_ENVELOPE, ("self-shielding 2e-26 cm$^2$/GeV (worst case)", "self-shielding 3e-24 (top veto panels)")):
                ax.plot(xx, k * xx, color=P.INK2, lw=0.9, ls=(0, (1, 2)), zorder=5)
            ax.plot(xx, S.ATM_CEILING_SIGMA_OVER_M * xx, color=P.RED, lw=1.0, zorder=5)
            txt = []
            for ep in EPS_LIST:
                rec = {c: next(x for x in summ["fixed_f"] if x["f"] == f and x["eps"] == ep and x["branch"] == br and x["ceiling"] == c)
                       for c in ("optimistic", "pessimistic")}
                txt.append(f"{EPS_NAME[ep]}: new {rec['optimistic']['new']:.2f} / {rec['pessimistic']['new']:.2f} dec$^2$"
                           f"  (no IMP: {rec['optimistic']['new_noIMP']:.2f} / {rec['pessimistic']['new_noIMP']:.2f})")
            ax.text(0.985, 0.03, "new area above the atmosphere line, optimistic / pessimistic self-shielding\n" + "\n".join(txt), transform=ax.transAxes, fontsize=7.6,
                    color=P.INK, ha="right", va="bottom", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.4", fc=P.SURFACE, ec=P.GRID, alpha=0.95))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e3, 1e10)
            ax.set_ylim(1e-24, 1e-16)
            ax.set_title(f"f = {f:g},  branch {br}: " + ("no nuclear form factor" if br == "A" else "with Helm form factor"))
            if i == 1:
                ax.set_xlabel("sub-component mass m (GeV)")
            if j == 0:
                ax.set_ylabel("contact cross-section per nucleus, sigma (cm$^2$)")
    h = [Patch(fc=shades[ep], ec=P.SURFACE, label=f"HXMT reach, {EPS_NAME[ep]}") for ep in EPS_LIST]
    h += [Line2D([0], [0], color=P.INK, lw=1.0, ls=(0, (4, 2)), label="statistics-only reach with pessimistic self-shielding"),
          Patch(fc=P.MUTED, alpha=0.62, label="excluded at this f (existing_constraints.csv)"),
          Patch(fc=P.MUTED, alpha=0.25, hatch="////", label="excluded only by the IMP 7/8 estimate"),
          Line2D([0], [0], color=P.ORANGE, lw=1.6, label="window W1 of the constraint notes"),
          Line2D([0], [0], color=P.RED, lw=1.0, label="sea-level atmosphere ceiling, 1.7e-27 cm$^2$/GeV"),
          Line2D([0], [0], color=P.INK2, lw=0.9, ls=(0, (1, 2)), label="self-shielding sigma/m = 2e-26 and 3e-24 cm$^2$/GeV")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8.2, bbox_to_anchor=(0.5, 0.032))
    fig.suptitle("Insight-HXMT reach for a rare strongly-interacting sub-component at fixed halo fraction f; blue left visible = new parameter space",
                 fontsize=11, color=P.INK, x=0.01, ha="left", y=0.99)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.945, bottom=0.135, wspace=0.05, hspace=0.10)
    P.note(fig, "FORECAST under stated assumptions, not a result from HXMT data. 90% CL, 2.5e8 s, model R; best of HE NaI-class (B = 500), CsI-class (8000), veto panels "
                "(6000 counts/s, ASSUMED). Blue shades: optimistic self-shielding (HE 5e-25, veto 3e-24 cm2/GeV);\npessimistic: HE 1e-25, veto 4.5e-25. "
                "Limits without a CSV row at this f are morphed in log f between neighbouring rows. m < 1e3 GeV not evaluated.", y=0.004)
    fig.savefig(os.path.join(FIG, "fig8_subcomponent_m_sigma.png"))
    plt.close(fig)
    print("wrote fig7_fmin_vs_mass.png, fig8_subcomponent_m_sigma.png")


if __name__ == "__main__":
    main()
