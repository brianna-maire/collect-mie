"""SSC cone vs optional rectangular-mask integration helpers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from collect_mie.core import (
    Polarization,
    SignalMode,
    diameter_sweep_detector_cone,
    diameter_sweep_detector_cone_rect_mask,
    integrate_detector_cone,
    integrate_detector_cone_rect_mask,
)


class SscCollectionConfig(Protocol):
    n_medium: float
    ssc_center_deg: float
    ssc_mask_half_angle_x_deg: float | None
    ssc_mask_half_angle_z_deg: float | None
    ssc_rect_mask_n_phi: int


def ssc_uses_rect_mask(mask_x: float | None, mask_z: float | None) -> bool:
    """True when both rectangular mask half-angles are set in config."""
    return mask_x is not None and mask_z is not None


def integrate_ssc_collection(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_x_deg: float | None,
    mask_half_angle_z_deg: float | None,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> float:
    """Integrate SSC over lens cone only, or cone ∩ rect mask when both mask angles are set."""
    if ssc_uses_rect_mask(mask_half_angle_x_deg, mask_half_angle_z_deg):
        return integrate_detector_cone_rect_mask(
            n_particle,
            diameter,
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
    return integrate_detector_cone(
        n_particle,
        diameter,
        wavelength_vacuum,
        n_medium,
        detector_center_deg,
        detector_half_angle_deg,
        polarization=polarization,
        signal_mode=signal_mode,
    )


def diameter_sweep_ssc_collection(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    n_medium: float,
    detector_center_deg: float,
    detector_half_angle_deg: float,
    mask_half_angle_x_deg: float | None,
    mask_half_angle_z_deg: float | None,
    *,
    polarization: Polarization = "unpolarized",
    signal_mode: SignalMode = "absolute_cross_section",
    n_phi: int = 720,
) -> np.ndarray:
    """Diameter sweep using cone-only or masked collection per mask settings."""
    if ssc_uses_rect_mask(mask_half_angle_x_deg, mask_half_angle_z_deg):
        return diameter_sweep_detector_cone_rect_mask(
            n_particle,
            diameters,
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
    return diameter_sweep_detector_cone(
        n_particle,
        diameters,
        wavelength_vacuum,
        n_medium,
        detector_center_deg,
        detector_half_angle_deg,
        polarization=polarization,
        signal_mode=signal_mode,
    )


def integrate_ssc_from_config(
    n_particle: complex,
    diameter: float,
    wavelength_vacuum: float,
    cfg: SscCollectionConfig,
    detector_half_angle_deg: float,
    *,
    polarization: Polarization,
    signal_mode: SignalMode,
) -> float:
    """Single-point SSC integral using optional mask fields on a config model."""
    return integrate_ssc_collection(
        n_particle,
        diameter,
        wavelength_vacuum,
        cfg.n_medium,
        cfg.ssc_center_deg,
        detector_half_angle_deg,
        cfg.ssc_mask_half_angle_x_deg,
        cfg.ssc_mask_half_angle_z_deg,
        polarization=polarization,
        signal_mode=signal_mode,
        n_phi=cfg.ssc_rect_mask_n_phi,
    )


def diameter_sweep_ssc_from_config(
    n_particle: complex,
    diameters: np.ndarray,
    wavelength_vacuum: float,
    cfg: SscCollectionConfig,
    detector_half_angle_deg: float,
    *,
    polarization: Polarization,
    signal_mode: SignalMode,
) -> np.ndarray:
    """Diameter sweep SSC using optional mask fields on a config model."""
    return diameter_sweep_ssc_collection(
        n_particle,
        diameters,
        wavelength_vacuum,
        cfg.n_medium,
        cfg.ssc_center_deg,
        detector_half_angle_deg,
        cfg.ssc_mask_half_angle_x_deg,
        cfg.ssc_mask_half_angle_z_deg,
        polarization=polarization,
        signal_mode=signal_mode,
        n_phi=cfg.ssc_rect_mask_n_phi,
    )
