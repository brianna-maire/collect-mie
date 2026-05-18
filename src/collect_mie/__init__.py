"""
Public API for Mie-based scatter curves and defaults.

See README.md for physics background; implementation lives in collect_mie.core.
"""

from collect_mie.defaults import (
    DEFAULT_N_MEDIUM,
    DEFAULT_N_PARTICLE_REAL,
    DEFAULT_FSC_CENTER_DEG,
    DEFAULT_FSC_NA_INNER,
    DEFAULT_FSC_NA_OUTER,
    DEFAULT_SSC_CENTER_DEG,
    DEFAULT_SSC_NA,
    DEFAULT_SSC_MASK_HALF_ANGLE_X_DEG,
    DEFAULT_SSC_MASK_HALF_ANGLE_Z_DEG,
    DEFAULT_WAVELENGTH_NM,
)
from collect_mie.core import (
    angular_intensity_curve,
    diameter_sweep,
    diameter_sweep_detector_annular_cone,
    diameter_sweep_detector_cone,
    diameter_sweep_detector_cone_rect_mask,
    integrate_detector_annular_cone,
    integrate_detector_cone,
    integrate_detector_cone_rect_mask,
    integrate_polar_band,
    normalize_relative,
)

__all__ = [
    "DEFAULT_WAVELENGTH_NM",
    "DEFAULT_N_MEDIUM",
    "DEFAULT_N_PARTICLE_REAL",
    "DEFAULT_FSC_CENTER_DEG",
    "DEFAULT_FSC_NA_OUTER",
    "DEFAULT_FSC_NA_INNER",
    "DEFAULT_SSC_CENTER_DEG",
    "DEFAULT_SSC_NA",
    "DEFAULT_SSC_MASK_HALF_ANGLE_X_DEG",
    "DEFAULT_SSC_MASK_HALF_ANGLE_Z_DEG",
    "angular_intensity_curve",
    "diameter_sweep",
    "diameter_sweep_detector_annular_cone",
    "diameter_sweep_detector_cone",
    "diameter_sweep_detector_cone_rect_mask",
    "integrate_detector_annular_cone",
    "integrate_detector_cone",
    "integrate_detector_cone_rect_mask",
    "integrate_polar_band",
    "normalize_relative",
]
