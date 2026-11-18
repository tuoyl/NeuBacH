# Forecast simulation: Earth-shadowed DM-wind orbital modulation with Insight-HXMT

Everything here is a **forecast under stated assumptions** (simulation). No HXMT data are read or analysed.
Notes with all assumptions, formulas and result tables: `../forecast_simulation.md` (Chinese).

## Requirements

Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `ppigrf` (IGRF-14 coefficients), `sgp4` (TLE check only).

```
pip install numpy scipy matplotlib ppigrf sgp4
```

## Reproduce (run inside this folder)

```
python3 s1_halo_transmission.py     # Task 1: halo flux, Earth-shadow transmission, fig1            (~10 s)
python3 s2_orbit_template.py        # Tasks 2-3: TLE check, P1 check, template T(u,Omega), fig2, fig3 (~10 s)
python3 s3_orientation.py           # Task 4: pointing dependence for discs/cylinders/panels, figS1   (~15 s)
python3 geomag.py                   # (optional) rebuild IGRF / McIlwain-L grid cache + self-tests     (~1 min)
python3 s4_nuisance_fisher.py       # Task 5: Fisher marginalisation, leakage, MC check, fig5         (~3 min)
python3 s5_controls.py              # Task 6: wind-map PSF, anti-sidereal null test, occultation, fig4 (~8 min)
python3 s6_sensitivity.py           # Task 7: sensitivity tables (reads s4 output if present)          (<1 s)
python3 s7_sensitivity_plot.py      # Task 8: (mass, cross-section) plot, fig6                         (~10 s)
```

**One command to redraw the sensitivity plot** (e.g. after `../existing_constraints.csv` has been updated):

```
python3 s7_sensitivity_plot.py
```

The script overlays the rows with `convention == contact` and `f == 1` if the CSV exists, and otherwise draws only
the HXMT regions and the analytic lines and writes "existing limits not overlaid" on the figure.
Options: `--constraints PATH --B_NaI 500 --B_CsI 8000 --tlive 2.5e8 --model R --out PATH`.
Adjustable physics inputs (sigma_min for NaI/CsI, self-shielding sigma/m, effective areas, halo numbers) are at the
top of `sensitivity_lib.py`.

Other options: `s4_nuisance_fisher.py --quick` (skips heavy models and the MC), `--figure-only` (redraw fig5 from
the saved table), `--dt 120`; `s5_controls.py --step 15 --model E`.

## Files

| file | content |
|---|---|
| `common.py` | constants, SHM flux tables (quadrature), PREM + toy atmosphere, Earth-shadow quadrature, Sun/Earth velocity, mean J2 orbit |
| `template.py` | DM-rate template T(t) (shadow + annual aberration + ram term) |
| `geomag.py` | IGRF-14 evaluation, McIlwain L by field-line tracing, COR = 14.5/L^2 grid |
| `timeline.py` | simulated 60-s timeline 2017-06-15..2026-11-01, SAA model, toy background, **simulated** pointing schedule |
| `nuisance.py` | sparse design blocks and exact Fisher marginalisation (Frisch-Waugh-Lovell / Schur complement) |
| `sensitivity_lib.py` | adjustable parameters and sensitivity formulas |
| `plotstyle.py` | figure style |
| `s1..s7_*.py` | task scripts (see above) |
| `outputs/` | result tables (CSV/JSON/NPZ) written by the scripts |
| `figures/` | PNG figures (200 dpi) |
| `downloads/hxmt_tle_42758.txt` | the single current TLE (CelesTrak, retrieved 2026-11-18; data only, never executed) |

## Figures

1. `fig1_transmission_vs_psi.png` — transmitted flux fraction vs psi (opaque Earth, speed thresholds, partially transparent Earth).
2. `fig2_template_u_Omega.png` — template in the (u, Omega) plane and its Omega-independent / Omega-dependent parts.
3. `fig3_modulation_depth_year.png` — orbital modulation depth through one year, annual flux modulation, apex shift.
4. `fig4_windmap_psf.png` — all-sky map of recovered significance for an injected signal (PSF), anti-sidereal null map, cuts.
5. `fig5_degradation_factors.png` — degradation factors per nuisance model and data selection; cost of re-fitting the map.
6. `fig6_sensitivity_mass_sigma.png` — projected reach in (mass, contact cross-section) with analytic lines and existing limits.
S1. `figS1_orientation.png` — pointing dependence of crossing rates (supplementary).

## Main simplifications (see the notes for the full list)

Mean circular J2 orbit (a = 6921 km, i = 43 deg) anchored to the 2026-11-18 TLE node — not the historical ephemeris;
toy background and SAA model; simulated pointing history; no detector response (every crossing counted);
straight-line DM trajectories, no gravitational focusing.
