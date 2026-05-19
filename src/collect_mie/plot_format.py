"""Consistent numeric formatting for plot titles, legends, and labels."""

from __future__ import annotations


def fmt_na(value: float) -> str:
    """Numerical aperture (2 decimal places)."""
    return f"{float(value):.2f}"


def fmt_deg(value: float) -> str:
    """Angle in degrees (1 decimal place)."""
    return f"{float(value):.1f}"


def fmt_n(value: float) -> str:
    """Refractive index (4 decimal places)."""
    return f"{float(value):.4f}"


def fmt_particle_n(n_real: float, n_imag: float = 0.0) -> str:
    """Particle refractive index, optional imaginary part."""
    if n_imag == 0:
        return fmt_n(n_real)
    return f"{fmt_n(n_real)}{float(n_imag):+.4f}j"


def format_ssc_rect_mask_note(
    mask_x: float | None,
    mask_z: float | None,
    *,
    prefix: str = ", ",
) -> str:
    """Title/legend suffix when SSC uses NA cone ∩ rectangular mask."""
    if mask_x is None or mask_z is None:
        return ""
    return (
        f"{prefix}NA cone ∩ rect mask "
        f"(mask_x={fmt_deg(mask_x)}°, mask_z={fmt_deg(mask_z)}°)"
    )
