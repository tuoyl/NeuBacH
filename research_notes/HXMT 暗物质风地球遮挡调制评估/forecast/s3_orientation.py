#!/usr/bin/env python3
"""
s3_orientation.py -- Task 4 of the forecast: orientation (pointing) dependence of
the DM crossing rate for non-spherical detectors.

HXMT/HE crystals are thin discs whose normal is the (inertially fixed) pointing
axis; the anti-coincidence panels face along (+top) and perpendicular (side) to
it.  For a flat face of area A with unit normal p, the crossing rate is

    R = n A * Int  g(n.a) |n.p| Theta(direction not blocked by the Earth) dOmega ,

g = flux per steradian of arrival direction n (SHM, apex a).  For an isotropic
flux R_iso = n <w> A / 2, and all numbers below are quoted relative to R_iso
(or to n <w> S/4 for a closed convex body of surface S).

Outputs
  outputs/s3_orientation_unshadowed.csv, outputs/s3_summary.json
  figures/figS1_orientation.png  (supplementary)

Run:  python3 s3_orientation.py      (about 1 min)
Simulation (forecast) only; real detector response is not included.
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

# geometry (cm) -- HE phoswich crystals and veto panels (instrument papers)
D_CRYSTAL = 19.0
T_NAI = 0.35
T_CSI = 4.0
A_FACE = 283.5                     # cm^2 (quoted geometric area of one crystal face)
TOP_PANELS = 6 * 2250.0            # cm^2, normal along the pointing axis
SIDE_PANELS_PER_AXIS = 6 * 1000.0  # cm^2 on the two walls perpendicular to x' (and the same for y')


class SphereQuad:
    """Gauss-Legendre (cos theta) x uniform (phi) quadrature, z axis = apex."""

    def __init__(self, nth=200, nph=400):
        x, w = np.polynomial.legendre.leggauss(nth)
        ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
        self.mu = np.repeat(x, nph)
        st = np.sqrt(1 - self.mu ** 2)
        phh = np.tile(ph, nth)
        self.n = np.array([st * np.cos(phh), st * np.sin(phh), self.mu])
        self.w = np.repeat(w, nph) * (2 * np.pi / nph)


def flat_rate(tab, quad, chi, wthr=0.0):
    """Unshadowed crossing rate of a two-sided flat face whose normal is at angle chi from the apex, / (n A)."""
    g = np.interp(quad.mu, tab.mu, tab.at_w(tab.G, wthr)[0])
    out = []
    for c in np.atleast_1d(chi):
        p = np.array([math.sin(c), 0.0, math.cos(c)])
        out.append(float((g * np.abs(p @ quad.n) * quad.w).sum()))
    return np.array(out)


def cylinder_rate(tab, quad, chi, diam, height, wthr=0.0):
    """Unshadowed rate through a cylinder (axis at angle chi from the apex) / n ; cm^2 km/s."""
    g = np.interp(quad.mu, tab.mu, tab.at_w(tab.G, wthr)[0])
    out = []
    for c in np.atleast_1d(chi):
        p = np.array([math.sin(c), 0.0, math.cos(c)])
        ca = np.abs(p @ quad.n)
        proj = math.pi * diam ** 2 / 4 * ca + diam * height * np.sqrt(np.clip(1 - ca ** 2, 0, None))
        out.append(float((g * proj * quad.w).sum()))
    return np.array(out)


def shadowed_flat(tab, sh, rhat, ahat, phat, nbeta=96, chunk=400):
    """
    Blocked part of the flat-face rate, Int_cone g(n.a) |n.p| dOmega, for arrays of satellite
    directions rhat (3, N); opaque Earth.  Returns array (N,) in km/s (per unit area and density).
    """
    N = rhat.shape[1]
    base = tab.G[0]
    al = sh.alpha[sh.alpha <= sh.theta_E + 1e-12]
    wa = sh.w_alpha[: len(al)]
    be = (np.arange(nbeta) + 0.5) * 2 * np.pi / nbeta
    wb = 2 * np.pi / nbeta
    ca, sa = np.cos(al)[None, :, None], np.sin(al)[None, :, None]
    cb, sb = np.cos(be)[None, None, :], np.sin(be)[None, None, :]
    out = np.empty(N)
    for i0 in range(0, N, chunk):
        r = rhat[:, i0:i0 + chunk]
        cpsi = np.clip(ahat @ r, -1, 1)
        spsi = np.sqrt(np.clip(1 - cpsi ** 2, 1e-12, None))
        e1 = (ahat[:, None] - cpsi[None, :] * r) / spsi[None, :]
        e2 = np.cross(r.T, e1.T).T
        mu = -ca * cpsi[:, None, None] + sa * spsi[:, None, None] * cb
        npdot = (-ca * (phat @ r)[:, None, None]
                 + sa * (cb * (phat @ e1)[:, None, None] + sb * (phat @ e2)[:, None, None]))
        g = np.interp(mu, tab.mu, base)
        out[i0:i0 + chunk] = ((g * np.abs(npdot)).sum(axis=2) * wb * wa[None, :]).sum(axis=1)
    return out


def main():
    wt = WindTemplate()
    vobs = wt.vsun_speed
    tab = C.FluxTable(vobs)
    quad = SphereQuad()
    sh = C.Shadow()
    summary = {}

    # ---------------- unshadowed orientation dependence ----------------
    chi_deg = np.arange(0.0, 90.1, 5.0)
    chi = np.radians(chi_deg)
    cols = {"chi_deg": chi_deg}
    for wthr in (0.0, 300.0, 500.0):
        iso = tab.flux_above(wthr) / 2.0
        cols[f"disc_rel_w>{int(wthr)}"] = flat_rate(tab, quad, chi, wthr) / iso
    S_csi = 2 * math.pi * D_CRYSTAL ** 2 / 4 + math.pi * D_CRYSTAL * T_CSI
    S_nai = 2 * math.pi * D_CRYSTAL ** 2 / 4 + math.pi * D_CRYSTAL * T_NAI
    cols["csi_cylinder_rel"] = cylinder_rate(tab, quad, chi, D_CRYSTAL, T_CSI) / (tab.mean_speed * S_csi / 4)
    cols["nai_disc_with_edge_rel"] = cylinder_rate(tab, quad, chi, D_CRYSTAL, T_NAI) / (tab.mean_speed * S_nai / 4)
    # veto: top panels (normal = pointing) and side panels (normals perpendicular to the pointing axis, roll omega)
    d_of = lambda c: np.interp(np.abs(c), np.cos(chi)[::-1], (flat_rate(tab, quad, chi) / (tab.mean_speed / 2))[::-1])
    top = TOP_PANELS * d_of(np.cos(chi))
    rolls = np.radians(np.arange(0.0, 90.1, 5.0))
    side = np.array([[SIDE_PANELS_PER_AXIS * (d_of(math.sin(c) * math.cos(om)) + d_of(math.sin(c) * math.sin(om)))
                      for om in rolls] for c in chi])
    cols["veto_top_rel_area_cm2"] = top                      # effective area in units where isotropic = A
    cols["veto_side_rel_area_cm2_rollmean"] = side.mean(axis=1)
    cols["veto_top_over_side_rollmean"] = top / side.mean(axis=1)
    cols["veto_top_over_side_rollmin"] = (top[:, None] / side).min(axis=1)
    cols["veto_top_over_side_rollmax"] = (top[:, None] / side).max(axis=1)
    names = list(cols.keys())
    np.savetxt(os.path.join(OUT, "s3_orientation_unshadowed.csv"), np.column_stack([cols[k] for k in names]),
               delimiter=",", header=",".join(names), comments="", fmt="%.5f")
    summary["unshadowed"] = dict(
        disc_rel_chi0=float(cols["disc_rel_w>0"][0]), disc_rel_chi90=float(cols["disc_rel_w>0"][-1]),
        disc_ratio_0_over_90=float(cols["disc_rel_w>0"][0] / cols["disc_rel_w>0"][-1]),
        disc_ratio_0_over_90_w300=float(cols["disc_rel_w>300"][0] / cols["disc_rel_w>300"][-1]),
        disc_ratio_0_over_90_w500=float(cols["disc_rel_w>500"][0] / cols["disc_rel_w>500"][-1]),
        csi_ratio_0_over_90=float(cols["csi_cylinder_rel"][0] / cols["csi_cylinder_rel"][-1]),
        csi_rel_chi0=float(cols["csi_cylinder_rel"][0]), csi_rel_chi90=float(cols["csi_cylinder_rel"][-1]),
        veto_top_over_side_chi0=float(cols["veto_top_over_side_rollmean"][0]),
        veto_top_over_side_chi90=float(cols["veto_top_over_side_rollmean"][-1]),
        isotropic_check_disc_mean=float(np.trapezoid(cols["disc_rel_w>0"] * np.sin(chi), chi)),
        effective_areas_cm2=dict(
            nai_one_disc_orientation_avg=S_nai / 4, nai_18=18 * S_nai / 4,
            nai_one_disc_faces_only=A_FACE / 2, csi_one_orientation_avg=S_csi / 4, csi_18=18 * S_csi / 4,
            veto_panel_crossings=(TOP_PANELS + 2 * SIDE_PANELS_PER_AXIS) / 2))

    # ---------------- with the Earth shadow: pointing-dependent template ----------------
    a = wt.vsun / vobs
    zc = np.array([0.0, 0.0, 1.0])
    eN = zc - (zc @ a) * a
    eN /= np.linalg.norm(eN)
    eE = np.cross(zc, a)
    eE /= np.linalg.norm(eE)
    pointings = {
        "pointing = apex (chi = 0)": a,
        "chi = 45 deg (towards equatorial-plane direction)": (a + eE) / math.sqrt(2),
        "chi = 90 deg, in the equatorial plane": eE,
        "chi = 90 deg, towards the celestial pole side": eN,
    }
    nu, nO = 72, 72
    u = (np.arange(nu) + 0.5) * 2 * np.pi / nu
    dOm = (np.arange(nO) + 0.5) * 2 * np.pi / nO
    ra_a, _ = C.vec_to_radec(wt.vsun)
    U, DO = np.meshgrid(u, dOm, indexing="ij")
    rh, _ = C.sat_vectors(U.ravel(), DO.ravel() + math.radians(ra_a), C.INC_DEG)
    T_iso = wt.evaluate(rh).reshape(nu, nO)                     # isotropically sensitive detector
    node_iso = T_iso - T_iso.mean(axis=1, keepdims=True)
    shadow_rows = [dict(case="isotropic detector (sphere)", mean=float(T_iso.mean()), unshadowed=1.0,
                        rms_node_part=float(node_iso.std()), p2p_max_over_nodes=float((T_iso.max(axis=0) - T_iso.min(axis=0)).max()))]
    curves = {"isotropic detector": T_iso[:, np.argmin(np.abs(np.degrees(dOm) - 270.0))]}
    iso_flat = tab.mean_speed / 2.0
    for name, p in pointings.items():
        open_rate = float(flat_rate(tab, quad, [math.acos(np.clip(p @ a, -1, 1))])[0])
        blocked = shadowed_flat(tab, sh, rh, a, p)
        Td = ((open_rate - blocked) / iso_flat).reshape(nu, nO)  # relative to the isotropic-flux rate n<w>A/2
        node = Td - Td.mean(axis=1, keepdims=True)
        shadow_rows.append(dict(case=name, mean=float(Td.mean()), unshadowed=open_rate / iso_flat,
                                rms_node_part=float(node.std()),
                                p2p_max_over_nodes=float((Td.max(axis=0) - Td.min(axis=0)).max()),
                                corr_node_part_with_isotropic=float(np.corrcoef(node.ravel(), node_iso.ravel())[0, 1])))
        curves[name] = Td[:, np.argmin(np.abs(np.degrees(dOm) - 270.0))]
    summary["shadowed_thin_disc"] = shadow_rows
    with open(os.path.join(OUT, "s3_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    # ---------------- supplementary figure ----------------
    P.apply()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax = axs[0]
    ax.plot(chi_deg, cols["disc_rel_w>0"], color=P.BLUE, label="thin disc (NaI face), all speeds")
    ax.plot(chi_deg, cols["disc_rel_w>300"], color=P.ORANGE, label="thin disc, w > 300 km/s")
    ax.plot(chi_deg, cols["csi_cylinder_rel"], color=P.AQUA, label="CsI cylinder 19 cm x 4 cm")
    ax.plot(chi_deg, cols["veto_top_over_side_rollmean"] / cols["veto_top_over_side_rollmean"].mean(), color=P.YELLOW,
            label="veto top / side panel rate ratio (normalised to its mean)")
    ax.axhline(1.0, color=P.AXIS, lw=0.8)
    ax.set_xlabel("chi: angle between pointing axis and wind apex (deg)")
    ax.set_ylabel("DM crossing rate / isotropic-flux rate")
    ax.set_title("(a) No Earth shadow: pointing dependence alone")
    ax.set_xlim(0, 90)
    ax.legend(loc="lower left", fontsize=8)
    ax = axs[1]
    for k, (name, y) in enumerate(curves.items()):
        ax.plot(np.degrees(u), y, color=P.CAT[k], label=name, lw=2.0 if k else 2.6)
    ax.set_xlabel("u: argument of latitude (deg), node with the deepest modulation")
    ax.set_ylabel("rate / isotropic-flux unshadowed rate")
    ax.set_title("(b) Opaque Earth: one orbit, thin disc at four pointings")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", fontsize=7.6)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    P.note(fig, "Simulation (forecast): geometric crossing rates only (every crossing counted), SHM v0 = 238 km/s; no detector response.")
    fig.savefig(os.path.join(FIG, "figS1_orientation.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
