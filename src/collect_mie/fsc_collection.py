"""FSC outer-cone integration with circular or rectangular obscuration (either/or)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from collect_mie.core import (
    Polarization,
    SignalMode,
    diameter_sweep_detector_annular_cone,
    diameter_sweep_detector_cone_minus_fsc_rect_bar,
    integrate_detector_annular_cone,
    integrate_detector_cone_minus_fsc_rect_bar,
)


class FscCollectionConfig(Protocol):
    n_medium: float
    fsc_center_deg: float
    fsc_mask_half_angle_y_deg: float | None
    fsc_mask_half_angle_z_deg: float | None
    fsc_rect_mask_n_phi: int


def fsc_uses_rect_mask(mask_y: float | None, mask_z: float | None) -> bool:
    """True when both rectangular bar half-angles are set in config."""
    return mask_y is not None and mask_z is not None


def integrate_fsc_collection(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    mask_half_angle_y_deg: float | None,
    mask_half_angle_z_deg: float | None,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> float:
    """
    Integrate FSC over the outer NA cone with either/or obscuration.

    - Rect bar set: (outer cone) \\ (lab +x rect bar); ``inner`` ignored.
    - Rect bar omitted: (outer cone) \\ (inner circular stop).
    """
    if fsc_uses_rect_mask(mask_half_angle_y_deg, mask_half_angle_z_deg):
        return integrate_detector_cone_minus_fsc_rect_bar(
            n_particle,
            diameter,
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            outer_half_angle_deg,
            mask_half_angle_y_deg,
            mask_half_angle_z_deg,
            polarization=polarization,
            signal_mode=signal_mode,
            n_phi=n_phi,
        )
    return integrate_detector_annular_cone(
        n_particle,
        diameter,
        wavelength_vacuum,
        n_medium,
        detector_center_deg,
        outer_half_angle_deg,
        inner_half_angle_deg,
        polarization=polarization,
        signal_mode=signal_mode,
    )


def diameter_sweep_fsc_collection(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    mask_half_angle_y_deg: float | None,
    mask_half_angle_z_deg: float | None,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> np.ndarray:
    """Diameter sweep using circular or rectangular obscuration per mask settings."""
    if fsc_uses_rect_mask(mask_half_angle_y_deg, mask_half_angle_z_deg):
        return diameter_sweep_detector_cone_minus_fsc_rect_bar(
            n_particle,
            diameters,
            wavelength_vacuum,
            n_medium,
            detector_center_deg,
            outer_half_angle_deg,
            mask_half_angle_y_deg,
            mask_half_angle_z_deg,
            polarization=polarization,
            signal_mode=signal_mode,
            n_phi=n_phi,
        )
    return diameter_sweep_detector_annular_cone(
        n_particle,
        diameters,
        wavelength_vacuum,
        n_medium,
        detector_center_deg,
        outer_half_angle_deg,
        inner_half_angle_deg,
        polarization=polarization,
        signal_mode=signal_mode,
    )


def integrate_fsc_from_config(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    cfg: FscCollectionConfig,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    *,
    polarization: Polarization,
    signal_mode: SignalMode,
) -> float:
    """Single-point FSC integral using optional mask fields on a config model."""
    return integrate_fsc_collection(
        n_particle,
        diameter,
        wavelength_vacuum,
        cfg.n_medium,
        cfg.fsc_center_deg,
        outer_half_angle_deg,
        inner_half_angle_deg,
        cfg.fsc_mask_half_angle_y_deg,
        cfg.fsc_mask_half_angle_z_deg,
        polarization=polarization,
        signal_mode=signal_mode,
        n_phi=cfg.fsc_rect_mask_n_phi,
    )


def diameter_sweep_fsc_from_config(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    cfg: FscCollectionConfig,
    outer_half_angle_deg: float,
    inner_half_angle_deg: float,
    *,
    polarization: Polarization,
    signal_mode: SignalMode,
) -> np.ndarray:
    """Diameter sweep FSC using optional mask fields on a config model."""
    return diameter_sweep_fsc_collection(
        n_particle,
        diameters,
        wavelength_vacuum,
        cfg.n_medium,
        cfg.fsc_center_deg,
        outer_half_angle_deg,
        inner_half_angle_deg,
        cfg.fsc_mask_half_angle_y_deg,
        cfg.fsc_mask_half_angle_z_deg,
        polarization=polarization,
        signal_mode=signal_mode,
        n_phi=cfg.fsc_rect_mask_n_phi,
    )
