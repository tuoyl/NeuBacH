"""
sensitivity_lib.py -- numbers and formulas shared by the sensitivity tables (s6)
and the (mass, cross-section) plot (s7).

ALL RESULTS ARE FORECASTS UNDER THE ASSUMPTIONS LISTED HERE -- not HXMT data.

Signal model (opaque Earth, every DM particle that crosses a crystal is recorded):
    S(t) = S0 * T(t),   S0 = f * rho * <w> * A_eff / m          [counts/s]
    (S0 = unshadowed rate; the orbit-averaged rate is <T> * S0 with <T> ~ 0.70)

Statistical error of the fitted S0 (Poisson, background-dominated):
    sigma_stat = sqrt(Bbar / t_live) / r_eff
Systematic floor: a background-model residual that is correlated with the
(projected) template and has rms eps * Bbar biases S0 by
    delta_sys = eps * Bbar / r_plain
Total:  sigma_tot = sqrt(sigma_stat^2 + delta_sys^2)
    90% CL upper limit (one-sided, null result) = 1.2816 * sigma_tot
    5 sigma discovery threshold                 = 5 * sigma_tot

Adjustable parameters are collected at the top of this file.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

# ----------------------------------------------------------------------------
# ADJUSTABLE PARAMETERS
# ----------------------------------------------------------------------------
RHO_DM = 0.3                    # GeV/cm^3 (Baxter et al. 2021)
MEAN_SPEED_KMS = 354.1          # <w> for v0 = 238 km/s, v_esc = 544 km/s (s1); 329.9 for v0 = 220
MEAN_T = 0.704                  # orbit/node-averaged transmitted fraction (s2)
GEV_PER_G = 5.609588603e23

# detectability floor in cross section (adjustable; from the earlier survey notes, not derived here)
SIGMA_MIN = {"NaI": 1.0e-22, "CsI": 4.0e-23}             # cm^2
# satellite self-shielding ceiling (adjustable): sigma/m above which DM is slowed by the spacecraft itself
SELF_SHIELD_SIGMA_OVER_M = 3.0e-25                        # cm^2/GeV  (~6 g/cm^2)
# sea-level detectors: vertical atmosphere column
ATM_COLUMN_G_CM2 = 1033.0
ATM_CEILING_SIGMA_OVER_M = 1.0 / (ATM_COLUMN_G_CM2 * GEV_PER_G)   # speed e-fold: 1.7e-27 cm^2/GeV
# Earth opacity (defaults from s1 for a 300 km/s detector-frame threshold; overwritten from s1 output if present)
EARTH_OPACITY_SIGMA_OVER_M = {"10%": 3.4e-35, "50%": 1.24e-34, "90%": 3.3e-34}
EARTH_THRESHOLD_COLUMN = "rel_depth_wthr300"

# orientation-averaged effective areas (s3): S/4 of each convex body, summed
A_EFF_CM2 = {"NaI": 2646.0,     # 18 discs, 19 cm diameter x 3.5 mm
             "CsI": 3626.0,     # 18 cylinders, 19 cm diameter x 4 cm
             "veto": 12750.0}   # 2.55 m^2 of thin panels, panel crossings (area/2)

# template information (s4); defaults are overwritten from outputs/s4_fisher_table.csv if present
#   key: (data selection, nuisance model) -> (r_eff, r_plain) for the node ("sidereal") template
R_DEFAULT = {
    ("all non-SAA", "const"): (0.2215, 0.1970),
    ("all non-SAA", "E"): (0.2198, 0.1948),
    ("all non-SAA", "R"): (0.1638, 0.1487),
    ("all non-SAA", "K"): (0.1384, 0.1276),
    ("good time ~50% (ELV>10, SAA margins)", "const"): (0.2132, 0.1912),
    ("good time ~50% (ELV>10, SAA margins)", "E"): (0.2039, 0.1830),
    ("good time ~50% (ELV>10, SAA margins)", "R"): (0.1645, 0.1517),
}
R_FULL_CONST_DEFAULT = {"all non-SAA": (0.2740, 0.2555), "good time ~50% (ELV>10, SAA margins)": (0.2605, 0.2453)}
R_FULL_KNOWN_DEFAULT = {"all non-SAA": (0.8550, 0.7650), "good time ~50% (ELV>10, SAA margins)": (0.8534, 0.7780)}

Z90 = 1.2816
Z5 = 5.0
LIVE_TIMES = {2.5e8: "all non-SAA", 1.5e8: "good time ~50% (ELV>10, SAA margins)"}
BACKGROUNDS = [1.0, 10.0, 100.0, 500.0, 2000.0, 8000.0]
BKG_MEANING = {
    1.0: "special event class / one particle-monitor-like channel (~1-2 c/s)",
    10.0: "narrow energy window in one module",
    100.0: "sub-band or ~3 NaI modules",
    500.0: "NaI, 17 modules, 30-300 keV after anticoincidence (538.6 c/s measured)",
    2000.0: "CsI sub-band or several CsI modules",
    8000.0: "CsI, all 18 crystals (5000-12000 c/s quoted)",
}
EPS_LIST = [0.0, 1e-5, 1e-4, 1e-3]


# ----------------------------------------------------------------------------
def load_fisher():
    """(selection, model) -> dict of r values, from s4 output when available."""
    res = {}
    path = os.path.join(OUT, "s4_fisher_table.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    res[(row["scenario"], row["model"])] = dict(
                        r_eff_node=float(row["r_eff_node"]), r_plain_node=float(row["r_plain_node"]),
                        r_eff_full=float(row["r_eff_full"]), r_plain_full=float(row["r_plain_full"]))
                except (KeyError, ValueError):
                    continue
    for key, (re_, rp) in R_DEFAULT.items():
        res.setdefault(key, {}).setdefault("r_eff_node", re_)
        res[key].setdefault("r_plain_node", rp)
    for sel, (re_, rp) in R_FULL_CONST_DEFAULT.items():
        res.setdefault((sel, "const"), {}).setdefault("r_eff_full", re_)
        res[(sel, "const")].setdefault("r_plain_full", rp)
    for sel, (re_, rp) in R_FULL_KNOWN_DEFAULT.items():
        res.setdefault((sel, "none"), {}).setdefault("r_eff_full", re_)
        res[(sel, "none")].setdefault("r_plain_full", rp)
    return res


def load_earth_opacity():
    """sigma/m grid and relative shadow depth from s1 (or a smooth default)."""
    import numpy as np
    path = os.path.join(OUT, "s1_transparent_table.csv")
    if os.path.exists(path):
        try:
            with open(path) as f:
                header = f.readline().strip().split(",")
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
            j = header.index(EARTH_THRESHOLD_COLUMN)
            som, dep = arr[:, 0], np.clip(arr[:, j], 0.0, 1.0)
            lv = {k: float(10 ** np.interp(v, dep, np.log10(som))) for k, v in (("10%", 0.1), ("50%", 0.5), ("90%", 0.9))}
            return som, dep, lv
        except Exception:
            pass
    som = np.logspace(-36, -30, 49)
    x = np.log10(som / EARTH_OPACITY_SIGMA_OVER_M["50%"])
    dep = 1.0 / (1.0 + np.exp(-x / 0.2))
    return som, dep, dict(EARTH_OPACITY_SIGMA_OVER_M)


def flux_mass_product(a_eff_cm2):
    """rho * <w> * A_eff  in GeV/s : S0 = f * this / m."""
    return RHO_DM * MEAN_SPEED_KMS * 1.0e5 * a_eff_cm2


def sigma_S0(B, t_live, r_eff, r_plain, eps=0.0):
    stat = math.sqrt(B / t_live) / r_eff
    sys_ = eps * B / r_plain
    return math.sqrt(stat ** 2 + sys_ ** 2), stat, sys_


def mass_reach(S0_limit, a_eff_cm2, f=1.0):
    return f * flux_mass_product(a_eff_cm2) / S0_limit


def f_min(S0_limit, m_gev, a_eff_cm2):
    return S0_limit * m_gev / flux_mass_product(a_eff_cm2)


def one_event_mass(a_eff_cm2, t_live, n_events=1.0, mean_T=MEAN_T):
    """Mass at which n_events DM particles cross the detector in t_live (f = 1)."""
    return flux_mass_product(a_eff_cm2) * mean_T * t_live / n_events


def raw_rate_mass(B, a_eff_cm2, mean_T=MEAN_T):
    """Mass below which the DM rate alone (f = 1) would exceed the whole observed rate B."""
    return flux_mass_product(a_eff_cm2) * mean_T / B
