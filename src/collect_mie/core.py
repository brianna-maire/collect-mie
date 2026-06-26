"""
Mie angular scattering and solid-angle integration using miepython.

This module is the numerical core: it turns bead optics (index, size, wavelength)
into (1) intensity vs polar scattering angle and (2) the integral of intensity
over a polar-angle band, which we treat as a toy model for light collected into
forward (FSC-like) and side (SSC-like) detector cones.

Units convention (must match miepython):
    - wavelength_vacuum and diameter share the same length unit (here we use nm in CLI).
    - n_particle is the complex refractive index of the sphere (absolute, not pre-divided).
    - n_medium is the real index of the surrounding medium; miepython converts to
      relative index and size parameter in the medium internally.
"""

from __future__ import annotations

import math
from typing import Literal

import miepython
import numpy as np

Polarization = Literal["unpolarized", "parallel", "perpendicular"]
SignalMode = Literal["absolute_cross_section", "phase_function"]

# NumPy 2.0 renamed trapz -> trapezoid; keep compatibility with older NumPy.
try:
    from numpy import trapezoid
except ImportError:  # pragma: no cover
    from numpy import trapz as trapezoid


def _intensity_components(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    mu: np.ndarray,
    signal_mode: SignalMode,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Call miepython for parallel / perpendicular scattered intensity [1/sr].

    mu is cos(theta) where theta is the polar angle between incident beam and
    observation direction (0 = straight forward).
    signal_mode controls miepython normalization:
      - absolute_cross_section: norm='qsca' and geometric-area scaling
      - phase_function: norm='albedo' (shape-only, area-independent)
    """
    norm = "qsca" if signal_mode == "absolute_cross_section" else "albedo"
    return miepython.intensities(
        n_particle,
        diameter,
        wavelength_vacuum,
        mu,
        n_env=n_medium,
        norm=norm,
    )


def intensity_for_polarization(
    ipar: np.ndarray,
    iper: np.ndarray,
    polarization: Polarization,
) -> np.ndarray:
    """Combine miepython's parallel (|S2|²) and perpendicular (|S1|²) channels."""
    if polarization == "unpolarized":
        return (ipar + iper) / 2.0
    if polarization == "parallel":
        return ipar
    return iper


def _apply_signal_mode_scaling(
    intensity_per_sr: np.ndarray, diameter: float, signal_mode: SignalMode
) -> np.ndarray:
    """
    Convert the angular intensity to either shape-only or area-scaled signal.

    For absolute_cross_section mode we multiply by geometric area π(d/2)^2 so
    integrated band signals scale with scattering cross-section (same length^2 units
    as diameter/wavelength inputs).
    """
    if signal_mode == "absolute_cross_section":
        area = np.pi * (diameter / 2.0) ** 2
        return intensity_per_sr * area
    return intensity_per_sr


def angular_intensity_curve(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    theta_min_deg: float = 0.1,
    theta_max_deg: float = 180.0,
    n_points: int = 3600,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample scattered intensity vs polar scattering angle θ (0° = forward).

    diameter and wavelength_vacuum must use the same length units (e.g. nm).
    miepython expects vacuum wavelength and absolute particle refractive index.
    """
    theta_rad = np.linspace(np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg), n_points)
    # miepython uses mu = cos(theta), not theta itself.
    mu = np.cos(theta_rad)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    iplot = intensity_for_polarization(ipar, iper, polarization)
    iplot = _apply_signal_mode_scaling(iplot, diameter, signal_mode)
    return np.rad2deg(theta_rad), iplot


def integrate_polar_band(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    theta_min_deg: float,
    theta_max_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_points: int = 4000,
) -> float:
    """
    Integrate scattered intensity over an azimuthally symmetric polar-angle band.

    For intensity I(θ) per unit solid angle [1/sr], collecting over all azimuth φ:

      ∫ I dΩ = ∫_0^{2π} ∫_{θ_min}^{θ_max} I(θ) sin θ dθ dφ
             = 2π ∫_{θ_min}^{θ_max} I(θ) sin θ dθ

    This models a detector that accepts all φ at polar angles between the given
    θ bounds (a coaxial cone / annulus in θ). Real cytometers also depend on NA,
    obscuration bars, and 3D optics; those are only approximated here by choosing θ ranges.

    Returns a scalar proportional to collected scatter for that band; comparison across
    diameters uses the same integral, so relative curves are meaningful.
    """
    if theta_min_deg >= theta_max_deg:
        raise ValueError("theta_min_deg must be less than theta_max_deg")
    theta_rad = np.linspace(np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg), n_points)
    mu = np.cos(theta_rad)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    i_int = intensity_for_polarization(ipar, iper, polarization)
    i_int = _apply_signal_mode_scaling(i_int, diameter, signal_mode)
    integrand = i_int * np.sin(theta_rad)
    return float(2.0 * np.pi * trapezoid(integrand, theta_rad))


def integrate_detector_cone(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_points: int = 4000,
) -> float:
    """
    Integrate over a single cone around a detector axis (not a full azimuthal band).

    Cone is defined by axis polar angle detector_center_deg and half-angle alpha.
    For each polar angle theta, only a subset of azimuth phi is accepted:
      cos(gamma) = cos(theta)cos(theta0) + sin(theta)sin(theta0)cos(phi) >= cos(alpha)
    and the allowed azimuth width W(theta) is integrated numerically:
      ∫ I(theta) W(theta) sin(theta) dtheta.
    """
    if detector_half_angle_deg <= 0:
        raise ValueError("detector_half_angle_deg must be positive")
    if detector_half_angle_deg >= 180:
        raise ValueError("detector_half_angle_deg must be < 180")

    theta0 = math.radians(detector_center_deg)
    alpha = math.radians(detector_half_angle_deg)
    cos_alpha = math.cos(alpha)

    theta = np.linspace(0.0, np.pi, n_points)
    mu = np.cos(theta)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    i_int = intensity_for_polarization(ipar, iper, polarization)
    i_int = _apply_signal_mode_scaling(i_int, diameter, signal_mode)

    width = _detector_azimuth_width(theta, theta0, alpha, cos_alpha)
    sin_theta = np.sin(theta)
    integrand = i_int * width * sin_theta
    return float(trapezoid(integrand, theta))


def integrate_detector_cone_rect_mask(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_x_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_points: int = 4000,
    n_phi: int = 720,
) -> float:
    """
    Integrate collected scatter over the intersection of:

    1) A full right circular cone (lens NA) about the detector axis at polar angle
       ``detector_center_deg`` with half-angle ``detector_half_angle_deg``.
    2) A symmetric rectangular mask in lab coordinates (no roll about detector +y):
       |arctan2(k_x, k_y)| <= mask_half_angle_x_deg (deg),
       |arctan2(k_z, k_y)| <= mask_half_angle_z_deg (deg),
       where (k_x, k_y, k_z) is the scattered propagation direction in a fixed lab
       basis: laser +x, detector +y (sample toward objective), vertical +z.

    Mie polar angle θ and azimuth φ stay the usual incident-centered convention used
    elsewhere in this module; directions are mapped to lab with a fixed orthogonal
    transform consistent with ``integrate_detector_cone`` at θ0 = 90°.

    For each θ, accepted φ arc length is computed by uniform quadrature on [0, 2π).
    """
    if detector_half_angle_deg <= 0:
        raise ValueError("detector_half_angle_deg must be positive")
    if detector_half_angle_deg >= 180:
        raise ValueError("detector_half_angle_deg must be < 180")
    if mask_half_angle_x_deg <= 0 or mask_half_angle_z_deg <= 0:
        raise ValueError("mask half-angles (deg) must be positive")
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")

    theta0 = math.radians(detector_center_deg)
    alpha = math.radians(detector_half_angle_deg)
    cos_alpha = math.cos(alpha)
    lim_x = math.radians(mask_half_angle_x_deg)
    lim_z = math.radians(mask_half_angle_z_deg)

    theta = np.linspace(0.0, np.pi, n_points)
    mu = np.cos(theta)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    i_int = intensity_for_polarization(ipar, iper, polarization)
    i_int = _apply_signal_mode_scaling(i_int, diameter, signal_mode)

    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / float(n_phi)
    th_grid, ph_grid = np.meshgrid(theta, phis, indexing="ij")

    kx = np.cos(th_grid)
    ky = np.sin(th_grid) * np.cos(ph_grid)
    kz = np.sin(th_grid) * np.sin(ph_grid)

    cos_gamma = np.cos(th_grid) * math.cos(theta0) + np.sin(th_grid) * math.sin(
        theta0
    ) * np.cos(ph_grid)
    in_lens = cos_gamma >= cos_alpha

    ax = np.arctan2(kx, ky)
    az = np.arctan2(kz, ky)
    in_rect = (np.abs(ax) <= lim_x) & (np.abs(az) <= lim_z)
    width = np.sum(in_lens & in_rect, axis=1) * dphi

    integrand = i_int * width * np.sin(theta)
    return float(trapezoid(integrand, theta))


def _fsc_rect_bar_membership(
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    lim_y: float,
    lim_z: float,
) -> np.ndarray:
    """
    True where direction lies inside the FSC obscuration bar (fixed lab frame).

    Bar centered on lab +x (y=0, z=0): |arctan2(k_y, k_x)| <= lim_y and
    |arctan2(k_z, k_x)| <= lim_z. SSC uses the same lab frame centered on +y.
    """
    ay = np.arctan2(ky, kx)
    az = np.arctan2(kz, kx)
    return (np.abs(ay) <= lim_y) & (np.abs(az) <= lim_z)


def integrate_detector_cone_minus_fsc_rect_bar(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_y_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_points: int = 4000,
    n_phi: int = 720,
) -> float:
    """
    Integrate collected scatter over (lens cone) \\ (FSC rect obscuration bar).

    The bar is a symmetric rectangle in the fixed lab frame (laser +x, detector +y,
    vertical +z), centered on +x with half-angles |arctan2(k_y, k_x)| and
    |arctan2(k_z, k_x)| — configured as ``mask_half_angle_y_deg`` /
    ``mask_half_angle_z_deg`` in YAML.
    """
    if detector_half_angle_deg <= 0:
        raise ValueError("detector_half_angle_deg must be positive")
    if detector_half_angle_deg >= 180:
        raise ValueError("detector_half_angle_deg must be < 180")
    if mask_half_angle_y_deg <= 0 or mask_half_angle_z_deg <= 0:
        raise ValueError("mask half-angles (deg) must be positive")
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")

    theta0 = math.radians(detector_center_deg)
    alpha = math.radians(detector_half_angle_deg)
    cos_alpha = math.cos(alpha)
    lim_y = math.radians(mask_half_angle_y_deg)
    lim_z = math.radians(mask_half_angle_z_deg)

    theta = np.linspace(0.0, np.pi, n_points)
    mu = np.cos(theta)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    i_int = intensity_for_polarization(ipar, iper, polarization)
    i_int = _apply_signal_mode_scaling(i_int, diameter, signal_mode)

    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / float(n_phi)
    th_grid, ph_grid = np.meshgrid(theta, phis, indexing="ij")

    kx = np.cos(th_grid)
    ky = np.sin(th_grid) * np.cos(ph_grid)
    kz = np.sin(th_grid) * np.sin(ph_grid)

    cos_gamma = np.cos(th_grid) * math.cos(theta0) + np.sin(th_grid) * math.sin(
        theta0
    ) * np.cos(ph_grid)
    in_lens = cos_gamma >= cos_alpha

    in_bar = _fsc_rect_bar_membership(kx, ky, kz, lim_y, lim_z)
    width = np.sum(in_lens & ~in_bar, axis=1) * dphi

    integrand = i_int * width * np.sin(theta)
    return float(trapezoid(integrand, theta))


def integrate_detector_cone_minus_rect_mask(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_x_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_points: int = 4000,
    n_phi: int = 720,
) -> float:
    """
    Integrate collected scatter over the lens cone minus a symmetric rectangular mask.

    Same lab frame and mask definition as ``integrate_detector_cone_rect_mask``;
    accepted solid angle is (lens cone) \\ (rect mask).
    """
    if detector_half_angle_deg <= 0:
        raise ValueError("detector_half_angle_deg must be positive")
    if detector_half_angle_deg >= 180:
        raise ValueError("detector_half_angle_deg must be < 180")
    if mask_half_angle_x_deg <= 0 or mask_half_angle_z_deg <= 0:
        raise ValueError("mask half-angles (deg) must be positive")
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")

    theta0 = math.radians(detector_center_deg)
    alpha = math.radians(detector_half_angle_deg)
    cos_alpha = math.cos(alpha)
    lim_x = math.radians(mask_half_angle_x_deg)
    lim_z = math.radians(mask_half_angle_z_deg)

    theta = np.linspace(0.0, np.pi, n_points)
    mu = np.cos(theta)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    i_int = intensity_for_polarization(ipar, iper, polarization)
    i_int = _apply_signal_mode_scaling(i_int, diameter, signal_mode)

    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / float(n_phi)
    th_grid, ph_grid = np.meshgrid(theta, phis, indexing="ij")

    kx = np.cos(th_grid)
    ky = np.sin(th_grid) * np.cos(ph_grid)
    kz = np.sin(th_grid) * np.sin(ph_grid)

    cos_gamma = np.cos(th_grid) * math.cos(theta0) + np.sin(th_grid) * math.sin(
        theta0
    ) * np.cos(ph_grid)
    in_lens = cos_gamma >= cos_alpha

    ax = np.arctan2(kx, ky)
    az = np.arctan2(kz, ky)
    in_rect = (np.abs(ax) <= lim_x) & (np.abs(az) <= lim_z)
    width = np.sum(in_lens & ~in_rect, axis=1) * dphi

    integrand = i_int * width * np.sin(theta)
    return float(trapezoid(integrand, theta))


def integrate_detector_annular_cone_minus_rect_mask(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    mask_half_angle_x_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_points: int = 4000,
    n_phi: int = 720,
) -> float:
    """
    Integrate over an annular lens cone minus a symmetric rectangular mask.

    Collection solid angle is (outer cone \\ inner obscuration) \\ (rect mask).
    FSC collection helpers do not use this stacking; they apply either circular
    ``na_inner`` or rect-bar obscuration, not both (see ``fsc_collection``).
    """
    if inner_half_angle_deg < 0:
        raise ValueError("inner_half_angle_deg must be >= 0")
    if outer_half_angle_deg <= 0:
        raise ValueError("outer_half_angle_deg must be positive")
    if inner_half_angle_deg >= outer_half_angle_deg:
        raise ValueError("inner_half_angle_deg must be smaller than outer_half_angle_deg")
    if mask_half_angle_x_deg <= 0 or mask_half_angle_z_deg <= 0:
        raise ValueError("mask half-angles (deg) must be positive")
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")

    theta0 = math.radians(detector_center_deg)
    cos_out = math.cos(math.radians(outer_half_angle_deg))
    cos_in = math.cos(math.radians(inner_half_angle_deg))
    lim_x = math.radians(mask_half_angle_x_deg)
    lim_z = math.radians(mask_half_angle_z_deg)

    theta = np.linspace(0.0, np.pi, n_points)
    mu = np.cos(theta)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    i_int = intensity_for_polarization(ipar, iper, polarization)
    i_int = _apply_signal_mode_scaling(i_int, diameter, signal_mode)

    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / float(n_phi)
    th_grid, ph_grid = np.meshgrid(theta, phis, indexing="ij")

    kx = np.cos(th_grid)
    ky = np.sin(th_grid) * np.cos(ph_grid)
    kz = np.sin(th_grid) * np.sin(ph_grid)

    cos_gamma = np.cos(th_grid) * math.cos(theta0) + np.sin(th_grid) * math.sin(
        theta0
    ) * np.cos(ph_grid)
    in_outer = cos_gamma >= cos_out
    in_inner = cos_gamma >= cos_in

    ax = np.arctan2(kx, ky)
    az = np.arctan2(kz, ky)
    in_rect = (np.abs(ax) <= lim_x) & (np.abs(az) <= lim_z)
    width = np.sum(in_outer & ~in_inner & ~in_rect, axis=1) * dphi

    integrand = i_int * width * np.sin(theta)
    return float(trapezoid(integrand, theta))


def _detector_azimuth_width(
    theta: np.ndarray, theta0: float, alpha: float, cos_alpha: float
) -> np.ndarray:
    """Allowed azimuth width W(theta) for a cone around axis theta0."""
    sin_theta = np.sin(theta)
    sin_theta0 = math.sin(theta0)
    cos_theta = np.cos(theta)
    cos_theta0 = math.cos(theta0)

    width = np.zeros_like(theta)
    denom = sin_theta * sin_theta0
    near_zero = np.abs(denom) < 1e-12

    if np.any(near_zero):
        gamma = np.arccos(np.clip(cos_theta[near_zero] * cos_theta0, -1.0, 1.0))
        width[near_zero] = np.where(gamma <= alpha, 2.0 * np.pi, 0.0)

    regular = ~near_zero
    if np.any(regular):
        k = (cos_alpha - cos_theta[regular] * cos_theta0) / denom[regular]
        width[regular] = np.where(
            k <= -1.0,
            2.0 * np.pi,
            np.where(k >= 1.0, 0.0, 2.0 * np.arccos(np.clip(k, -1.0, 1.0))),
        )
    return width


def integrate_detector_annular_cone(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_points: int = 4000,
) -> float:
    """Integrate over cone annulus: outer cone minus inner obscuration cone."""
    if inner_half_angle_deg < 0:
        raise ValueError("inner_half_angle_deg must be >= 0")
    if outer_half_angle_deg <= 0:
        raise ValueError("outer_half_angle_deg must be positive")
    if inner_half_angle_deg >= outer_half_angle_deg:
        raise ValueError("inner_half_angle_deg must be smaller than outer_half_angle_deg")

    theta0 = math.radians(detector_center_deg)
    alpha_out = math.radians(outer_half_angle_deg)
    alpha_in = math.radians(inner_half_angle_deg)
    cos_out = math.cos(alpha_out)
    cos_in = math.cos(alpha_in)

    theta = np.linspace(0.0, np.pi, n_points)
    mu = np.cos(theta)
    ipar, iper = _intensity_components(
        n_particle, diameter, wavelength_vacuum, n_medium, mu, signal_mode
    )
    i_int = intensity_for_polarization(ipar, iper, polarization)
    i_int = _apply_signal_mode_scaling(i_int, diameter, signal_mode)

    width_out = _detector_azimuth_width(theta, theta0, alpha_out, cos_out)
    width_in = _detector_azimuth_width(theta, theta0, alpha_in, cos_in)
    width = np.clip(width_out - width_in, 0.0, 2.0 * np.pi)

    integrand = i_int * width * np.sin(theta)
    return float(trapezoid(integrand, theta))


def diameter_sweep(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    theta_min_deg: float,
    theta_max_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
) -> np.ndarray:
    """One integrate_polar_band per diameter (used for FSC/SSC vs size curves)."""
    out = np.empty_like(diameters, dtype=float)
    for i, d in enumerate(diameters):
        out[i] = integrate_polar_band(
            n_particle,
            float(d),
            wavelength_vacuum,
            n_medium,
            theta_min_deg,
            theta_max_deg,
            polarization=polarization,
            signal_mode=signal_mode,
        )
    return out


def diameter_sweep_detector_cone(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
) -> np.ndarray:
    """One integrate_detector_cone per diameter (single-cone detector geometry)."""
    out = np.empty_like(diameters, dtype=float)
    for i, d in enumerate(diameters):
        out[i] = integrate_detector_cone(
            n_particle,
            float(d),
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            detector_half_angle_deg,
            polarization=polarization,
            signal_mode=signal_mode,
        )
    return out


def diameter_sweep_detector_cone_rect_mask(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_x_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> np.ndarray:
    """One integrate_detector_cone_rect_mask per diameter."""
    out = np.empty_like(diameters, dtype=float)
    for i, d in enumerate(diameters):
        out[i] = integrate_detector_cone_rect_mask(
            n_particle,
            float(d),
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            detector_half_angle_deg,
            mask_half_angle_x_deg,
            mask_half_angle_z_deg,
            polarization=polarization,
            signal_mode=signal_mode,
            n_phi=n_phi,
        )
    return out


def diameter_sweep_detector_annular_cone(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
) -> np.ndarray:
    """One integrate_detector_annular_cone per diameter."""
    out = np.empty_like(diameters, dtype=float)
    for i, d in enumerate(diameters):
        out[i] = integrate_detector_annular_cone(
            n_particle,
            float(d),
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            outer_half_angle_deg,
            inner_half_angle_deg,
            polarization=polarization,
            signal_mode=signal_mode,
        )
    return out


def diameter_sweep_detector_cone_minus_fsc_rect_bar(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_y_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> np.ndarray:
    """One integrate_detector_cone_minus_fsc_rect_bar per diameter."""
    out = np.empty_like(diameters, dtype=float)
    for i, d in enumerate(diameters):
        out[i] = integrate_detector_cone_minus_fsc_rect_bar(
            n_particle,
            float(d),
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            detector_half_angle_deg,
            mask_half_angle_y_deg,
            mask_half_angle_z_deg,
            polarization=polarization,
            signal_mode=signal_mode,
            n_phi=n_phi,
        )
    return out


def diameter_sweep_detector_cone_minus_rect_mask(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_x_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> np.ndarray:
    """One integrate_detector_cone_minus_rect_mask per diameter."""
    out = np.empty_like(diameters, dtype=float)
    for i, d in enumerate(diameters):
        out[i] = integrate_detector_cone_minus_rect_mask(
            n_particle,
            float(d),
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            detector_half_angle_deg,
            mask_half_angle_x_deg,
            mask_half_angle_z_deg,
            polarization=polarization,
            signal_mode=signal_mode,
            n_phi=n_phi,
        )
    return out


def diameter_sweep_detector_annular_cone_minus_rect_mask(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    mask_half_angle_x_deg: float,
    mask_half_angle_z_deg: float,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> np.ndarray:
    """One integrate_detector_annular_cone_minus_rect_mask per diameter."""
    out = np.empty_like(diameters, dtype=float)
    for i, d in enumerate(diameters):
        out[i] = integrate_detector_annular_cone_minus_rect_mask(
            n_particle,
            float(d),
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            outer_half_angle_deg,
            inner_half_angle_deg,
            mask_half_angle_x_deg,
            mask_half_angle_z_deg,
            polarization=polarization,
            signal_mode=signal_mode,
            n_phi=n_phi,
        )
    return out


def normalize_relative(values: np.ndarray, mode: Literal["max", "first"] = "max") -> np.ndarray:
    """Dimensionless scaling so plots can overlay shape without absolute calibration."""
    if values.size == 0:
        return values
    if mode == "first":
        denom = float(values[0]) if values[0] != 0 else np.finfo(float).eps
    else:
        denom = float(np.max(values)) if np.max(values) != 0 else np.finfo(float).eps
    return values / denom
