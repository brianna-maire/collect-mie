"""Shared geometry helpers and signal-mode mapping."""

from __future__ import annotations

import math


def parse_comma_separated_floats(text: str) -> list[float]:
    """Parse '1.59, 1.602, 1.62' into floats."""
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one value")
    return [float(x) for x in parts]


def signal_mode_value(mode: str) -> str:
    """Map config spelling to internal enum spelling."""
    return mode.replace("-", "_")


def resolve_ssc_band_deg(
    *, ssc_na: float, ssc_center_deg: float, n_medium: float
) -> tuple[float, float]:
    """Return (ssc_theta_min, ssc_theta_max) in degrees from center and NA."""
    alpha_deg = ssc_half_angle_deg(ssc_na, n_medium)
    lo = float(ssc_center_deg) - alpha_deg
    hi = float(ssc_center_deg) + alpha_deg
    return lo, hi


def ssc_half_angle_deg(ssc_na: float, n_medium: float) -> float:
    """Convert NA to cone half-angle alpha (deg) in medium."""
    na = float(ssc_na)
    n_medium = float(n_medium)
    if na <= 0:
        raise SystemExit("ssc_na must be positive.")
    if n_medium <= 0:
        raise SystemExit("n_medium must be positive.")
    if na > n_medium:
        raise SystemExit("ssc_na must be <= n_medium for alpha=asin(NA/n_medium).")
    return math.degrees(math.asin(na / n_medium))


def fsc_half_angles_deg(
    fsc_na_outer: float, fsc_na_inner: float, n_medium: float
) -> tuple[float, float]:
    """Convert FSC outer/inner NAs to half-angles (alpha_outer, alpha_inner)."""
    na_out = float(fsc_na_outer)
    na_in = float(fsc_na_inner)
    n_medium = float(n_medium)
    if na_out <= 0:
        raise SystemExit("fsc_na_outer must be positive.")
    if na_in < 0:
        raise SystemExit("fsc_na_inner must be >= 0.")
    if na_in >= na_out:
        raise SystemExit("fsc_na_inner must be smaller than fsc_na_outer.")
    if n_medium <= 0:
        raise SystemExit("n_medium must be positive.")
    if na_out > n_medium:
        raise SystemExit("fsc_na_outer must be <= n_medium for alpha=asin(NA/n_medium).")
    alpha_outer = math.degrees(math.asin(na_out / n_medium))
    alpha_inner = math.degrees(math.asin(na_in / n_medium))
    return alpha_outer, alpha_inner


def resolve_fsc_half_angles_deg(
    *,
    fsc_na_outer: float,
    fsc_na_inner: float,
    n_medium: float,
    mask_half_angle_y_deg: float | None = None,
    mask_half_angle_z_deg: float | None = None,
) -> tuple[float, float]:
    """
    FSC half-angles for titles and sweeps.

    When both rect-mask half-angles are set, ``na_inner`` is ignored for
    integration and ``0`` is used here so ``na_outer`` may be below the default
    ``na_inner`` without error.
    """
    uses_rect_mask = (
        mask_half_angle_y_deg is not None and mask_half_angle_z_deg is not None
    )
    na_inner = 0.0 if uses_rect_mask else fsc_na_inner
    return fsc_half_angles_deg(fsc_na_outer, na_inner, n_medium)
