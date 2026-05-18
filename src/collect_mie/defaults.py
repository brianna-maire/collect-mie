"""
Instrument-style defaults for wavelength and polar-angle bands.

Angles are polar scattering θ in degrees from the incident beam (0° = forward).
They approximate cytometer collection only roughly; tune them to your optics or
calibration beads when comparing to experiment.
"""

DEFAULT_WAVELENGTH_NM = 488.0
DEFAULT_N_MEDIUM = 1.3374
DEFAULT_N_PARTICLE_REAL = 1.602

DEFAULT_FSC_CENTER_DEG = 0.0
# Outer NA sets collection half-angle; inner NA models obscuration stop.
DEFAULT_FSC_NA_OUTER = 0.34
DEFAULT_FSC_NA_INNER = 0.23

DEFAULT_SSC_CENTER_DEG = 90.0
# Numerical aperture that defines SSC collection half-angle as:
#   alpha = asin(NA / n_medium)
DEFAULT_SSC_NA = 1.29

# Rectangular lab mask (half-angles, deg) intersected with the SSC lens cone;
# see integrate_detector_cone_rect_mask in core.py.
DEFAULT_SSC_MASK_HALF_ANGLE_X_DEG = 68.0
DEFAULT_SSC_MASK_HALF_ANGLE_Z_DEG = 74.0
