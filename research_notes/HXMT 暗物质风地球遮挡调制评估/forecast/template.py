"""
template.py -- the DM-rate template T(t) for an (isotropically sensitive) detector
on the satellite, normalised to the unshadowed, Sun-only-velocity rate:

    S(t) = S0 * T(t),      S0 = f * rho/m * <w>_ref * A_eff,
    <w>_ref = mean DM speed for an observer moving with the Sun's velocity
              (354.1 km/s for v0 = 238 km/s).

    T(t) = [ Phi_E(psi_E, |v_E-frame|) + (v_sat . a_E) * K(psi_E, |v_E-frame|) ] / <w>_ref

  psi_E  : angle between the satellite's geocentric direction and the apex of the
           EARTH's motion through the halo (Sun + Earth orbital velocity),
  Phi_E  : flux/n reaching the satellite when arrival directions inside the Earth
           disc are blocked (opaque Earth) or slowed (sigma/m given),
  K      : first-order "ram" kernel for the satellite's own 7.6 km/s velocity.

The blocking test is done in the Earth frame (that is where the shadow cone is
a cone); the flux through the detector is evaluated in the satellite frame to
first order in v_sat/w.  Validated against brute-force Monte Carlo in s1.
"""
import math

import numpy as np

import common as C


class WindTemplate:
    def __init__(self, v0=C.V0_DEFAULT, vesc=C.VESC_DEFAULT, a_km=C.A_ORBIT, wthr=0.0,
                 sigma_over_m=None, npsi=361, vgrid=None, with_atm=False):
        self.v0, self.vesc, self.a_km = v0, vesc, a_km
        self.vsun = C.vsun_eq(v0)                                  # equatorial, km/s
        self.vsun_speed = float(np.linalg.norm(self.vsun))
        ref = C.FluxTable(self.vsun_speed, v0, vesc)
        self.ref_flux = ref.flux_above(wthr)                       # <w>_ref (above threshold)
        self.ref_mean_speed = ref.mean_speed
        self.sh = C.Shadow(a_km, with_atm=with_atm or (sigma_over_m is not None))
        self.psi = np.linspace(0.0, math.pi, npsi)
        if vgrid is None:
            vgrid = np.arange(self.vsun_speed - 50.0, self.vsun_speed + 50.01, 5.0)
        self.vgrid = np.asarray(vgrid, dtype=float)
        Phi = np.empty((npsi, len(self.vgrid)))
        K = np.empty_like(Phi)
        Un = np.empty(len(self.vgrid))
        for j, v in enumerate(self.vgrid):
            tab = C.FluxTable(v, v0, vesc, nw=1201, nmu=1001)
            tot = tab.flux_above(wthr)
            Un[j] = tot
            Phi[:, j] = tot - self.sh.blocked(tab, self.psi, wthr, sigma_over_m)
            K[:, j] = self.sh.ram_kernel(tab, self.psi, sigma_over_m)
        self.unshadowed = Un / self.ref_flux                        # annual modulation of the open flux
        self._Phi = C.Table2D(self.psi, self.vgrid, Phi / self.ref_flux)
        self._K = C.Table2D(self.psi, self.vgrid, K / self.ref_flux)
        self._Un = Un / self.ref_flux

    # -- pieces ---------------------------------------------------------
    def F(self, psi, vobs=None):
        """Shadowed flux / reference flux at angle psi (rad) for Earth-frame speed vobs."""
        vobs = self.vsun_speed if vobs is None else vobs
        return self._Phi(psi, np.broadcast_to(vobs, np.shape(psi)))

    def Kram(self, psi, vobs=None):
        vobs = self.vsun_speed if vobs is None else vobs
        return self._K(psi, np.broadcast_to(vobs, np.shape(psi)))

    def open_flux(self, vobs):
        return np.interp(vobs, self.vgrid, self._Un)

    # -- full template ----------------------------------------------------
    def evaluate(self, rhat, vsat=None, vobs_vec=None, return_parts=False):
        """
        rhat     : (3, N) unit geocentric satellite direction (inertial equatorial)
        vsat     : (3, N) satellite velocity (km/s) or None (no ram term)
        vobs_vec : (3,) or (3, N) velocity of the EARTH through the halo (km/s);
                   default: Sun only (static apex).
        """
        if vobs_vec is None:
            vobs_vec = self.vsun
        vobs_vec = np.asarray(vobs_vec, dtype=float)
        if vobs_vec.ndim == 1:
            vobs_vec = vobs_vec[:, None]
        speed = np.sqrt((vobs_vec ** 2).sum(axis=0))
        ahat = vobs_vec / speed
        cpsi = np.clip((rhat * ahat).sum(axis=0), -1.0, 1.0)
        psi = np.arccos(cpsi)
        speed_b = np.broadcast_to(speed, psi.shape)
        T = self._Phi(psi, speed_b)
        ram = 0.0
        if vsat is not None:
            ram = (vsat * ahat).sum(axis=0) * self._K(psi, speed_b)
        if return_parts:
            return T + ram, T, ram, psi
        return T + ram

    # -- Omega-independent part (function of latitude only) ---------------
    def lat_profile(self, nlat=181, nra=720):
        """
        T0(lat) = average of the static (Sun-only) template over the right ascension
        of the satellite at fixed latitude (= average over the node Omega).
        Returns (lat_grid_rad, T0).
        """
        lat = np.linspace(-math.pi / 2, math.pi / 2, nlat)
        ra = (np.arange(nra) + 0.5) * 2 * math.pi / nra
        _, dec_a = C.vec_to_radec(self.vsun)
        dec_a = math.radians(dec_a)
        cpsi = (np.sin(lat)[:, None] * math.sin(dec_a)
                + np.cos(lat)[:, None] * math.cos(dec_a) * np.cos(ra)[None, :])
        T = self.F(np.arccos(np.clip(cpsi, -1, 1)))
        return lat, T.mean(axis=1)
