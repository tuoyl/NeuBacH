#!/usr/bin/env python3
"""
s4_nuisance_fisher.py -- Task 5 of the forecast: how much of the DM-template
information survives marginalisation over background nuisance models, and how
strongly un-modelled background terms leak into the template amplitude.

Method (see nuisance.py): exact linear-algebra Fisher marginalisation on 60-s
time bins over the whole simulated mission (2017-06-15 ... 2026-11-01, ~4.9e6
bins), Poisson weights from a toy background that varies by a factor ~2.5-3
between low and high cutoff rigidity and carries SAA-activation tails.

Templates
  full   : complete template (annual aberration + ram term included)
  static : static apex (Sun's velocity only), no ram term
  node   : static template minus its Omega-average T0(lat)  -> the purely
           node-dependent ("sidereal") part; this is the recommended search template.

Key output numbers per (data selection, nuisance model, template):
  r_eff  : sqrt(<T_perp^2 / b>)  -> sigma(S0) = sqrt(Bbar / t_live) / r_eff
  r_plain: sqrt(<T_perp^2>)      -> bias(S0) = eps * Bbar / r_plain for a
           template-correlated residual of rms eps*Bbar
  recov  : expected fitted amplitude when the data contain the FULL signal
  degradation factors relative to "free constant only" and to "background known".
Leakage table: for un-modelled toy terms g(t) of unit rms,
  beta  = bias on S0 per unit rms amplitude of g,   kappa = beta * r_plain
  (kappa = fraction of the term's rms that appears as a template-correlated residual).

Outputs
  outputs/s4_fisher_table.csv, outputs/s4_leakage.csv, outputs/s4_summary.json
  figures/fig5_degradation_factors.png

Run:  python3 s4_nuisance_fisher.py [--dt 60] [--quick]     (about 3-5 min; --quick ~1 min)
Simulation (forecast) only; the pointing history, background and SAA model are toys.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp

import nuisance as NU
import plotstyle as P
import timeline as TL
from template import WindTemplate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

MODEL_LABELS = {
    "none": "background known (ideal)",
    "const": "free constant",
    "A": "A: free 5x5 deg map x asc/desc",
    "A_fine": "A': 2.5x2.5 deg map x asc/desc",
    "B": "B: A + f(time since SAA)",
    "C": "C: B + f(local time) + f(beta)",
    "C2": "C2: B + f(Sun phase, beta) 2-D",
    "D": "D: C + monthly spline trend",
    "E": "E: D, map free per year",
    "E180": "E180: D, map free per 180 d",
    "E90": "E90: D, map free per 90 d",
    "E66": "E66: D, map free per 66 d",
    "E30": "E30: D, map free per 30 d",
    "E10": "E10: D, map free per 10 d",
    "H": "H: E + f(pointing dir. in local frame)",
    "F": "F: C + per-pointing visible/occulted offsets",
    "G": "G: C + per-pointing f(Earth angle)",
    "R": "R: E + per-pointing vis/occ + f(Sun phase,beta) + lat x season + zen x poly(COR)",
    "K": "K: G + f(Sun phase,beta) + lat x season + zen x poly(COR)",
}
PERIODS = {"E": 365.25, "E180": 180.0, "E90": 90.0, "E66": 65.75, "E30": 30.0, "E10": 10.0}
LEAK_MODELS = ("const", "A", "A_fine", "B", "C", "C2", "D", "E", "E66", "H", "F", "G", "R", "K")


def lat_season_block(lat_rad, ann_phase, nlat=18, latmax=45.0):
    """Columns (lat bin) x (cos, sin of the annual phase): seasonal change of the N-S profile."""
    il = np.clip(((np.degrees(lat_rad) + latmax) / (2 * latmax) * nlat).astype(np.int64), 0, nlat - 1)
    N = len(il)
    rows = np.concatenate([np.arange(N), np.arange(N)])
    cols = np.concatenate([2 * il, 2 * il + 1])
    vals = np.concatenate([np.cos(ann_phase), np.sin(ann_phase)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, 2 * nlat))


def zen_poly_block(zen_deg, cor, nzen=12, deg=2):
    """(pointing-zenith-angle bin) x (1, COR, COR^2): Earth-angle profile whose amplitude varies smoothly with COR."""
    iz = np.minimum((zen_deg / 180.0 * nzen).astype(np.int64), nzen - 1)
    c = np.asarray(cor, dtype=float) / 10.0
    N = len(iz)
    rows = np.concatenate([np.arange(N)] * (deg + 1))
    cols = np.concatenate([iz * (deg + 1) + k for k in range(deg + 1)])
    vals = np.concatenate([c ** k for k in range(deg + 1)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, nzen * (deg + 1)))


class Blocks:
    """Lazily built design blocks for one data selection (index array idx)."""

    def __init__(self, tl, idx):
        self.tl, self.idx = tl, idx
        self._c = {}

    def get(self, name):
        if name in self._c:
            return self._c[name]
        tl, idx = self.tl, self.idx
        if name == "map":
            v = NU.map_cells(tl["lon"][idx], tl["lat"][idx], tl["asc"][idx], 5.0)
        elif name == "map_fine":
            v = NU.map_cells(tl["lon"][idx], tl["lat"][idx], tl["asc"][idx], 2.5)
        elif name == "Zmap":
            i, n = self.get("map")
            v = NU.onehot(i, n)
        elif name == "tsaa":
            v = NU.onehot(*NU.tsaa_bins(tl["t_since_saa"][idx]))
        elif name == "lt":
            v = NU.onehot(*NU.lt_bins(tl["lt_hours"][idx]))
        elif name == "beta":
            v = NU.onehot(*NU.beta_bins(tl["beta"][idx]))
        elif name == "sun2d":
            v = NU.onehot(*NU.sunphase_beta_bins(tl["sun_phase"][idx], tl["beta"][idx]))
        elif name == "trend":
            v = NU.linear_spline(tl["t"][idx], 30.0)
        elif name == "pdir":
            v = NU.onehot(*NU.pointing_dir_bins(tl["point_zen"][idx], tl["point_az"][idx]))
        elif name == "latseason":
            v = lat_season_block(tl["lat"][idx], tl["doy_phase"][idx])
        elif name == "zenpoly":
            v = zen_poly_block(tl["point_zen"][idx], tl["COR"][idx])
        elif name == "Zvis":
            i, n = NU.compress(tl["obs"][idx].astype(np.int64) * 2 + (tl["elv"][idx] > 0))
            v = NU.onehot(i, n)
        else:
            raise KeyError(name)
        self._c[name] = v
        return v

    def stack(self, *names):
        return sp.hstack([self.get(n) for n in names]).tocsr()

    def model(self, name):
        """Return (cell index or None, sparse Z or None)."""
        tl, idx = self.tl, self.idx
        if name in ("none", "const"):
            return None, None
        if name == "A":
            return self.get("map")[0], None
        if name == "A_fine":
            return self.get("map_fine")[0], None
        if name == "B":
            return self.get("map")[0], self.get("tsaa")
        if name == "C":
            return self.get("map")[0], self.stack("tsaa", "lt", "beta")
        if name == "C2":
            return self.get("map")[0], self.stack("tsaa", "sun2d")
        if name == "D":
            return self.get("map")[0], self.stack("tsaa", "lt", "beta", "trend")
        if name in PERIODS:
            m, nmap = self.get("map")
            return m + nmap * NU.period_index(tl["t"][idx], PERIODS[name]), self.stack("tsaa", "lt", "beta", "trend")
        if name == "H":
            m, nmap = self.get("map")
            return (m + nmap * NU.period_index(tl["t"][idx], 365.25),
                    self.stack("tsaa", "lt", "beta", "trend", "pdir"))
        obs = tl["obs"][idx].astype(np.int64)
        if name == "F":
            return obs * 2 + (tl["elv"][idx] > 0), self.stack("Zmap", "tsaa", "lt", "beta")
        if name == "G":
            ib, nb = NU.earth_angle_bins(tl["point_zen"][idx])
            return obs * nb + ib, self.stack("Zmap", "tsaa", "lt", "beta")
        if name == "R":
            m, nmap = self.get("map")
            return (m + nmap * NU.period_index(tl["t"][idx], 365.25),
                    self.stack("Zvis", "tsaa", "lt", "sun2d", "latseason", "zenpoly"))
        if name == "K":
            ib, nb = NU.earth_angle_bins(tl["point_zen"][idx])
            return obs * nb + ib, self.stack("Zmap", "tsaa", "lt", "sun2d", "latseason", "zenpoly")
        raise KeyError(name)


def toy_terms(tl, idx, b_parts, seed=5):
    """Un-modelled toy background terms g(t) (dimensionless shapes) for the leakage study."""
    rng = np.random.default_rng(seed)
    t = tl["t"][idx]
    yr = t / (365.25 * 86400.0)
    lt = tl["lt_hours"][idx] * (2 * np.pi / 24.0)
    ann = tl["doy_phase"][idx]
    lat = tl["lat"][idx]
    cor = tl["COR"][idx].astype(float)
    hcor = ((cor + 1.0) / 10.0) ** (-1.1)
    hcor = hcor - hcor.mean()
    g = {}
    g["diurnal cos(LT)"] = np.cos(lt)
    g["diurnal sin(LT)"] = np.sin(lt)
    g["diurnal x annual: cos(LT)cos(yr)"] = np.cos(lt) * np.cos(ann)
    g["diurnal x annual: cos(LT)sin(yr)"] = np.cos(lt) * np.sin(ann)
    g["diurnal x annual: sin(LT)cos(yr)"] = np.sin(lt) * np.cos(ann)
    g["diurnal x annual: sin(LT)sin(yr)"] = np.sin(lt) * np.sin(ann)
    g["sunlit sub-satellite point: max(0, sin elev)"] = np.clip(np.sin(np.radians(tl["sun_elev"][idx])), 0, None)
    g["eclipse flag"] = tl["eclipse"][idx].astype(float)
    g["sin(beta)"] = np.sin(np.radians(tl["beta"][idx]))
    g["N-S x annual: sin(lat)cos(yr)"] = np.sin(lat) * np.cos(ann)
    g["N-S x annual: sin(lat)sin(yr)"] = np.sin(lat) * np.sin(ann)
    # solar modulation of the rigidity-dependent prompt component
    g["h(COR) x 11-yr solar cycle"] = hcor * np.sin(2 * np.pi * (yr - 2.5) / 11.0)
    ph = np.cumsum(rng.normal(0, 0.25, size=int(yr.max() * 365.25 / 27.0) + 3))      # random phase walk per rotation
    rot = t / (27.0 * 86400.0)
    g["h(COR) x 27-d solar rotation (phase-wandering)"] = hcor * np.sin(2 * np.pi * rot + ph[rot.astype(int)])
    fd = np.zeros_like(t)
    for t0 in rng.uniform(0, t.max(), size=int(6 * yr.max())):
        amp = rng.exponential(1.0)
        dtt = (t - t0) / 86400.0
        fd += np.where(dtt > 0, -amp * np.exp(-np.clip(dtt, 0, 200) / 5.0), 0.0)
    g["h(COR) x Forbush decreases (6/yr, 5-d recovery)"] = hcor * fd
    st = np.zeros_like(t)
    for t0 in rng.uniform(0, t.max(), size=int(10 * yr.max())):
        dur = rng.uniform(1.0, 3.0) * 86400.0
        st += np.where((t > t0) & (t < t0 + dur), rng.lognormal(0, 0.7), 0.0)
    g["storm episodes x (L > 2)"] = st * (tl["L"][idx] > 2.0)
    # components of the toy background itself (what the nuisance model cannot absorb of them)
    for k, v in b_parts.items():
        g[f"toy bkg component: {k}"] = v[idx]
    # pointing related (simulated pointing history)
    amp_obs = rng.lognormal(0.0, 1.0, size=tl["n_obs"])
    g["per-pointing source flux x visibility step"] = amp_obs[tl["obs"][idx]] * (tl["elv"][idx] > 0)
    g["Earth in FOV: max(0,-cos zen) x h(COR)"] = np.clip(-np.cos(np.radians(tl["point_zen"][idx])), 0, None) * (hcor + 1.0)
    out = {}
    for name, gv in g.items():
        gv = gv - gv.mean()
        rms = float(np.sqrt((gv * gv).mean()))
        if rms > 0:
            out[name] = gv / rms                                    # unit plain rms over the live time
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=60.0, help="time-bin length in seconds")
    ap.add_argument("--quick", action="store_true", help="skip the heavy models (E10, H, F, G, R, K) and the MC")
    ap.add_argument("--figure-only", action="store_true", help="redraw fig5 from outputs/s4_fisher_table.csv")
    args = ap.parse_args()
    if args.figure_only:
        with open(os.path.join(OUT, "s4_fisher_table.csv")) as f:
            rws = []
            for r_ in csv.DictReader(f):
                rws.append({k: (float(v) if k not in ("scenario", "model", "label") else v) for k, v in r_.items()})
        make_figure(rws)
        return
    t_start = time.time()

    tl = TL.build(args.dt)
    TL.pointing_schedule(tl)
    dt = tl["dt"]
    ok = ~tl["saa"]
    # toy background and its components (weights; and leakage of its own un-modelled part)
    gcor = ((tl["COR"].astype(float) + 1.0) / 10.0) ** (-1.1)
    b_parts = {"prompt ~ f(COR)": 0.60 * gcor / gcor[ok].mean()}
    for name, frac in (("act_6min", 0.05), ("act_36min", 0.15), ("act_13h", 0.20)):
        b_parts[f"activation {name[4:]}"] = frac * tl[name] / tl[name][ok].mean()
    b_all = sum(b_parts.values())

    # ---- templates over all bins ----------------------------------------
    wt = WindTemplate()
    vsat = tl["vhat"] * tl["vsat_kms"]
    vE = wt.vsun[:, None] + tl["v_earth"]
    T_full = wt.evaluate(tl["rhat"], vsat, vE)
    T_static = wt.evaluate(tl["rhat"])
    latg, T0lat = wt.lat_profile()
    T_node = T_static - np.interp(tl["lat"], latg, T0lat)
    templates = {"full": T_full, "static": T_static, "node": T_node}

    gti50 = ok & (tl["elv"] > 10) & (tl["t_since_saa"] > 300) & (tl["t_to_saa"] > 300)
    scenarios = {
        "all non-SAA": ok,
        "good time ~50% (ELV>10, SAA margins)": gti50,
        "good time strict (+ COR>8)": gti50 & (tl["COR"] > 8.0),
        "target occulted (ELV<0)": ok & (tl["elv"] < 0) & (tl["t_since_saa"] > 300),
    }
    full_ladder = ["none", "const", "A", "A_fine", "B", "C", "C2", "D", "E", "E180", "E90", "E66", "E30", "E10",
                   "H", "F", "G", "R", "K"]
    short_ladder = ["none", "const", "A", "B", "C", "D", "E", "E66", "F", "R"]
    heavy = ("E10", "H", "F", "G", "R", "K")
    if args.quick:
        full_ladder = [m for m in full_ladder if m not in heavy]
        short_ladder = [m for m in short_ladder if m not in heavy]
    first, second = "all non-SAA", "good time ~50% (ELV>10, SAA margins)"

    rows, leak_rows = [], []
    summary = dict(dt=dt, n_bins=int(tl["n"]), total_days=tl["n"] * dt / 86400.0, n_pointings=int(tl["n_obs"]),
                   saa_fraction=float(tl["saa"].mean()), scenarios={})
    resid_store = {}
    for sc, mask in scenarios.items():
        idx = np.where(mask)[0]
        frac = len(idx) / tl["n"]
        bb = b_all[idx] / b_all[idx].mean()
        W = dt / bb
        tlive = len(idx) * dt
        summary["scenarios"][sc] = dict(
            live_fraction=frac, live_time_s=tlive, mean_T_full=float(T_full[idx].mean()),
            bkg_p05_p95=[float(np.percentile(bb, 5)), float(np.percentile(bb, 95))],
            mean_inv_b=float((1 / bb).mean()))
        blk = Blocks(tl, idx)
        base, terms = {}, None
        print(f"\n=== {sc}: live fraction {frac:.3f} ({tlive:.3e} s) ===")
        for m in (full_ladder if sc == first else short_ladder):
            t0 = time.time()
            if m == "none":
                res = {k: (float((W * v[idx] ** 2).sum()), v[idx]) for k, v in templates.items()}
                npar = 0
            else:
                cell, Z = blk.model(m)
                proj = NU.Projector(W, cell, Z)
                res = {k: proj.info(v[idx]) for k, v in templates.items()}
                npar = proj.n_par
                del proj
            out = dict(scenario=sc, model=m, label=MODEL_LABELS[m], n_par=npar)
            for k, (I, r) in res.items():
                out[f"r_eff_{k}"] = math.sqrt(I / tlive)
                out[f"r_plain_{k}"] = float(np.sqrt((r * r).mean()))
                out[f"recov_{k}"] = float((W * r * T_full[idx]).sum() / I)
                # deterministic bias from the part of the toy background the model cannot absorb
                out[f"toybkg_bias_over_B_{k}"] = float((W * r * bb).sum() / I)
                out[f"toybkg_eps_{k}"] = out[f"toybkg_bias_over_B_{k}"] * out[f"r_plain_{k}"]
            if m in ("none", "const"):
                base[m] = out
            for k in templates:
                out[f"deg_vs_const_{k}"] = base["const"][f"r_eff_{k}"] / out[f"r_eff_{k}"] if "const" in base else float("nan")
                out[f"deg_vs_known_{k}"] = base["none"][f"r_eff_{k}"] / out[f"r_eff_{k}"]
            rows.append(out)
            print(f"{m:7s} npar={npar:7d} r_eff full/node={out['r_eff_full']:.4f}/{out['r_eff_node']:.4f} "
                  f"r_plain full/node={out['r_plain_full']:.4f}/{out['r_plain_node']:.4f} "
                  f"deg(const) full={out['deg_vs_const_full']:.3f} deg(known) node={out['deg_vs_known_node']:.3f} "
                  f"recov(node)={out['recov_node']:.3f} toy-bkg eps full/node={out['toybkg_eps_full']:+.1e}/"
                  f"{out['toybkg_eps_node']:+.1e} [{time.time() - t0:.1f} s]")
            # ---- leakage of un-modelled toy terms (first two scenarios) ----
            if sc in (first, second) and m in LEAK_MODELS:
                if terms is None:
                    terms = toy_terms(tl, idx, b_parts)
                for k in ("full", "node"):
                    I, r = res[k]
                    rpl = out[f"r_plain_{k}"]
                    for gname, gv in terms.items():
                        beta = float((W * r * gv).sum() / I)
                        leak_rows.append(dict(scenario=sc, model=m, template=k, term=gname, beta=beta, kappa=beta * rpl))
                if sc == first and m in ("D", "E", "R"):
                    resid_store[m] = {k: (res[k][1], res[k][0]) for k in ("full", "node")}
                    resid_store[m]["idx"], resid_store[m]["W"] = idx, W

    # ---- Monte-Carlo validation of the Fisher error (all non-SAA) ----------
    # The same Poisson realisations are analysed with every stored estimator.
    mc = {}
    if not args.quick and resid_store:
        rng = np.random.default_rng(123)
        Bbar, S_inj, nmc = 500.0, 0.2, 200
        any_m = next(iter(resid_store))
        idx, W = resid_store[any_m]["idx"], resid_store[any_m]["W"]
        bb = b_all[idx] / b_all[idx].mean()
        mu = (Bbar * bb + S_inj * T_full[idx]) * dt
        wr = {(m, k): W * st[k][0] / st[k][1] for m, st in resid_store.items() for k in ("full", "node")}
        ests = {key: np.empty(nmc) for key in wr}
        for j in range(nmc):
            y = rng.poisson(mu) / dt
            for key in wr:
                ests[key][j] = (wr[key] * y).sum()
        for (m, k), e in ests.items():
            I = resid_store[m][k][1]
            sig_fisher = math.sqrt(Bbar / I)
            bias_bkg = Bbar * float((wr[(m, k)] * bb).sum())
            recov = S_inj * float((wr[(m, k)] * T_full[idx]).sum())
            mc.setdefault(m, {})[k] = dict(
                Bbar=Bbar, S_injected=S_inj, n_mc=nmc, mean_estimate=float(e.mean()),
                std_estimate=float(e.std(ddof=1)), fisher_sigma=sig_fisher, expected_signal_part=recov,
                predicted_bias_from_toy_bkg=bias_bkg, bias_over_Bbar=bias_bkg / Bbar)
            print(f"MC model {m}, template {k}: mean {e.mean():+.4f} (pred {recov + bias_bkg:+.4f} = signal "
                  f"{recov:.4f} + toy-bkg leak {bias_bkg:+.4f}), std {e.std(ddof=1):.4f} (Fisher {sig_fisher:.4f})")
    summary["mc_validation"] = mc

    # ---- write tables ---------------------------------------------------
    with open(os.path.join(OUT, "s4_fisher_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r_.items()})
    with open(os.path.join(OUT, "s4_leakage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "model", "template", "term", "beta", "kappa"])
        w.writeheader()
        for r_ in leak_rows:
            w.writerow({k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in r_.items()})
    summary["runtime_s"] = time.time() - t_start
    with open(os.path.join(OUT, "s4_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    make_figure(rows)
    print(f"\ntotal runtime {time.time() - t_start:.0f} s")


def make_figure(rows):
    P.apply()
    import matplotlib.pyplot as plt
    scs = ["all non-SAA", "good time ~50% (ELV>10, SAA margins)", "good time strict (+ COR>8)"]
    cols = [P.BLUE, P.ORANGE, P.AQUA]
    marks = ["o", "s", "D"]
    dodge = [0.2, 0.0, -0.2]
    order = ["const", "A", "A_fine", "B", "C", "C2", "D", "E", "H", "F", "G", "R", "K"]
    get = lambda sc, m, key: next((r[key] for r in rows if r["scenario"] == sc and r["model"] == m), None)
    present = [m for m in order if get(scs[0], m, "r_eff_full") is not None]
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1.0]))
    handles = []
    for ax, key, title in ((axs[0], "deg_vs_const_full", "(a) Full template:\nsigma(S0) relative to 'free constant only'"),
                           (axs[1], "deg_vs_known_node", "(b) Omega-dependent part:\nsigma relative to 'background known'")):
        y = np.arange(len(present))[::-1].astype(float)
        handles = []
        for sc, col, mk, dy in zip(scs, cols, marks, dodge):
            xs, ys = [], []
            for m, yy in zip(present, y):
                v = get(sc, m, key)
                if v is not None:
                    xs.append(v)
                    ys.append(yy + dy)
            h, = ax.plot(xs, ys, mk, color=col, ms=6.5, mec=P.SURFACE, mew=1.0, label=sc, ls="none")
            handles.append(h)
        xmax = max(get(scs[0], m, key) for m in present)
        for m, yy in zip(present, y):
            v = get(scs[0], m, key)
            ax.text(v + 0.012 * xmax, yy + dodge[0], f"{v:.2f}", fontsize=7.5, color=P.INK2, ha="left", va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABELS[m] for m in present], fontsize=7.8)
        ax.set_ylim(-0.7, len(present) - 0.3)
        ax.set_xlabel("degradation factor of the amplitude uncertainty")
        ax.set_title(title, fontsize=9.4)
        ax.set_xlim(0.92, xmax * 1.09)
        ax.grid(axis="y", visible=False)
    axs[1].set_yticklabels([])
    fig.legend(handles, scs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), fontsize=8.5,
               title="data selection (value labels: all non-SAA)", title_fontsize=8.5)
    ax = axs[2]
    per = [(365.25, "E"), (180.0, "E180"), (90.0, "E90"), (65.75, "E66"), (30.0, "E30"), (10.0, "E10")]
    pts = [(p, get(scs[0], m, "deg_vs_const_full")) for p, m in per if get(scs[0], m, "deg_vs_const_full") is not None]
    xs, ys = [p for p, _ in pts], [v for _, v in pts]
    ax.plot(xs, ys, "-o", color=P.BLUE, ms=6.5, mec=P.SURFACE, mew=1.0)
    for xx, yy in zip(xs, ys):
        if xx in (10.0, 30.0, 65.75, 365.25):
            ax.text(xx, yy + 0.13, f"{yy:.2f}", fontsize=7.5, color=P.INK2, ha="center", va="bottom")
    ax.axvline(65.75, color=P.AXIS, lw=0.8)
    ax.set_ylim(0.9, max(ys) * 1.12)
    ax.text(72, max(ys) * 1.08, "nodal precession\nperiod 65.7 d", fontsize=7.5, color=P.INK2, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("map re-fit interval (days)")
    ax.set_ylabel("degradation factor (full template, all non-SAA)")
    ax.set_title("(c) Model D with the map\nre-fitted per interval", fontsize=9.4)
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    P.note(fig, "Simulation (forecast): exact Fisher marginalisation on 60-s bins, 2017-06-15 to 2026-11-01, toy background and "
                "SIMULATED pointing history (not the real HXMT observation log).")
    fig.savefig(os.path.join(FIG, "fig5_degradation_factors.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
