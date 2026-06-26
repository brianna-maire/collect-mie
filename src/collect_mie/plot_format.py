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


def format_beam_waist_note(
    waist_um_y: float,
    waist_um_z: float,
    *,
    prefix: str = ", ",
) -> str:
    """Excitation waist description for titles (lab frame, µm)."""
    if abs(waist_um_y - waist_um_z) <= 1e-6 * max(abs(waist_um_y), abs(waist_um_z), 1.0):
        body = f"Gaussian beam (w₀={waist_um_y:g} µm)"
    else:
        body = (
            f"elliptical Gaussian (w_y={waist_um_y:g} µm, w_z={waist_um_z:g} µm)"
        )
    return f"{prefix}{body}"


def format_fsc_rect_mask_note(
    mask_y: float | None,
    mask_z: float | None,
    *,
    prefix: str = ", ",
) -> str:
    """Title/legend suffix when FSC uses outer NA cone minus lab +x rect bar."""
    if mask_y is None or mask_z is None:
        return ""
    return (
        f"{prefix}NA cone w\\ rect bar"
        f"(mask_y={fmt_deg(mask_y)}°, mask_z={fmt_deg(mask_z)}°)"
    )


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
