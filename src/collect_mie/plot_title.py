"""
Build multi-line figure titles from resolved YAML config (one line per section).

Geometry (FSC / SSC / beam) belongs in the title; legends should identify curves only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from collect_mie.plot_format import (
    fmt_deg,
    fmt_na,
    fmt_n,
    fmt_particle_n,
    format_beam_waist_note,
    format_fsc_rect_mask_note,
    format_ssc_rect_mask_note,
)

# Human-readable labels for ``run.command`` (shown as ``Analysis: …``).
COMMAND_ALIASES: dict[str, str] = {
    "plot-angle": "Mie angular intensity",
    "plot-diameter": "Mie integrated scatter vs particle diameter",
    "plot-refractive-index": "SSC vs particle diameter",
    "plot-ssc-vs-na": "SSC vs numerical aperture",
    "plot-diameter-ssc-rect-mask": "SSC cone vs rectangular mask",
    "plot-diameter-fsc-rect-mask": "FSC annular NA vs rectangular bar",
    "compare-ssc": "SSC data vs Mie model",
    "compare-fsc": "FSC data vs Mie model",
    "plot-angle-beam": "GLMT angular intensity",
    "plot-diameter-beam": "GLMT integrated scatter vs diameter",
    "compare-ssc-beam": "SSC data vs GLMT model",
    "compare-fsc-beam": "FSC data vs GLMT model",
}


class _HasMedium(Protocol):
    wavelength_nm: float
    n_medium: float
    n_imag: float
    polarization: str
    signal_mode: str


class _HasParticle(Protocol):
    n_real: float
    n_imag: float


class _HasFsc(Protocol):
    fsc_center_deg: float
    fsc_na_outer: float
    fsc_na_inner: float
    fsc_mask_half_angle_y_deg: float | None
    fsc_mask_half_angle_z_deg: float | None


class _HasSscGeometry(Protocol):
    ssc_center_deg: float
    ssc_mask_half_angle_x_deg: float | None
    ssc_mask_half_angle_z_deg: float | None


class _HasSsc(_HasSscGeometry, Protocol):
    ssc_na: float


class _HasBeam(Protocol):
    beam_profile: str
    plane_wave_waist_ratio: float

    def beam_waists_um(self) -> tuple[float, float]: ...


@dataclass
class TitleContext:
    """Per-run flags and precomputed geometry passed into title builders."""

    uses_fsc: bool = False
    uses_ssc: bool = False
    fsc_alpha_outer: float | None = None
    fsc_alpha_inner: float | None = None
    ssc_alpha: float | None = None
    extra_lines: list[str] = field(default_factory=list)


def analysis_alias(command: str) -> str:
    """Return display name for ``run.command``."""
    return COMMAND_ALIASES.get(command, command)


def _line_analysis(command: str) -> str:
    return f"Analysis: {analysis_alias(command)}"


def _line_mie(cfg: _HasMedium & _HasParticle) -> str:
    return (
        f"Optics: λ={cfg.wavelength_nm:g} nm (vac), n={fmt_particle_n(cfg.n_real, cfg.n_imag)}, "
        f"n_medium={fmt_n(cfg.n_medium)}, pol={cfg.polarization}, signal={cfg.signal_mode}"
    )


def _line_beam(cfg: _HasBeam) -> str:
    wy, wz = cfg.beam_waists_um()
    waist = format_beam_waist_note(wy, wz, prefix="")
    return f"Excitation:{waist}; PW ratio={cfg.plane_wave_waist_ratio:g}"


def _fsc_collection_note(cfg: _HasFsc) -> str:
    mask = format_fsc_rect_mask_note(
        cfg.fsc_mask_half_angle_y_deg,
        cfg.fsc_mask_half_angle_z_deg,
        prefix="",
    )
    if mask:
        return mask
    return "annular NA"


def _line_fsc(cfg: _HasFsc, ctx: TitleContext) -> str:
    parts = [f"FSC: center={fmt_deg(cfg.fsc_center_deg)}°"]
    if ctx.fsc_alpha_outer is not None and ctx.fsc_alpha_inner is not None:
        parts.append(
            f"α_out={fmt_deg(ctx.fsc_alpha_outer)}°, α_in={fmt_deg(ctx.fsc_alpha_inner)}°"
        )
    parts.append(f"collection={_fsc_collection_note(cfg)}")
    return ", ".join(parts)


def _ssc_collection_note(cfg: _HasSscGeometry) -> str:
    mask = format_ssc_rect_mask_note(
        cfg.ssc_mask_half_angle_x_deg,
        cfg.ssc_mask_half_angle_z_deg,
        prefix="",
    )
    return mask if mask else "NA cone only"


def _line_ssc_na_sweep(cfg: _HasSscGeometry) -> str:
    """SSC geometry when NA is swept (no fixed ``ssc_na`` or α)."""
    return (
        f"SSC: center={fmt_deg(cfg.ssc_center_deg)}°, "
        f"collection={_ssc_collection_note(cfg)}"
    )


def _line_ssc(cfg: _HasSsc, ctx: TitleContext) -> str:
    parts = [
        f"SSC: center={fmt_deg(cfg.ssc_center_deg)}°, NA={fmt_na(cfg.ssc_na)}",
    ]
    if ctx.ssc_alpha is not None:
        parts.append(f"α={fmt_deg(ctx.ssc_alpha)}°")
    parts.append(f"collection={_ssc_collection_note(cfg)}")
    return ", ".join(parts)


def _format_float_list(values: list[float]) -> str:
    return ", ".join(f"{v:g}" for v in values)


def _line_command(command: str, cfg: BaseModel) -> str | None:
    if command == "plot-angle":
        c = cfg  # PlotAngleConfig
        return f"d={c.diameter_um:g} µm, θ={c.theta_min_deg:g}–{c.theta_max_deg:g}°"
    if command == "plot-angle-beam":
        c = cfg
        return f"d={c.diameter_um:g} µm, θ={c.theta_min_deg:g}–{c.theta_max_deg:g}°"
    if command in ("plot-diameter", "plot-diameter-beam"):
        c = cfg
        return f""
    if command == "plot-refractive-index":
        c = cfg
        return f""
    if command == "plot-ssc-vs-na":
        c = cfg
        return f""
    if command == "plot-diameter-ssc-rect-mask":
        c = cfg
        return f"d={c.d_min_um:g}–{c.d_max_um:g} µm, normalize={c.normalize}"
    if command == "plot-diameter-fsc-rect-mask":
        c = cfg
        return f"d={c.d_min_um:g}–{c.d_max_um:g} µm, normalize={c.normalize}"
    if command in ("compare-ssc", "compare-fsc", "compare-ssc-beam", "compare-fsc-beam"):
        c = cfg
        parts = [f"normalize={c.normalize}"]
        if getattr(c, "data_source", "manifest") == "table":
            parts.insert(0, f"points={Path(c.points_manifest).name}")
        else:
            parts.insert(0, f"manifest={Path(c.manifest).name}")
            parts.append(f"summary={c.channel_summary}")
            if c.median_gate != "none":
                parts.append(f"gate={c.median_gate}±{c.median_gate_log_decades:g} decades")
            if c.median_error == "bootstrap":
                parts.append(f"CI={c.median_ci_percent:g}% bootstrap")
        return ", ".join(parts)
    return None


def build_figure_title(
    command: str,
    cfg: BaseModel,
    ctx: TitleContext | None = None,
) -> str:
    """Assemble title lines for a resolved config and command name."""
    ctx = ctx or TitleContext()
    lines: list[str] = [_line_analysis(command)]

    if hasattr(cfg, "wavelength_nm") and hasattr(cfg, "n_real"):
        lines.append(_line_mie(cfg))  # type: ignore[arg-type]

    if hasattr(cfg, "beam_waists_um"):
        lines.append(_line_beam(cfg))  # type: ignore[arg-type]

    if ctx.uses_fsc and hasattr(cfg, "fsc_center_deg"):
        lines.append(_line_fsc(cfg, ctx))  # type: ignore[arg-type]

    if ctx.uses_ssc and hasattr(cfg, "ssc_center_deg"):
        if hasattr(cfg, "ssc_na"):
            lines.append(_line_ssc(cfg, ctx))  # type: ignore[arg-type]
        else:
            lines.append(_line_ssc_na_sweep(cfg))  # type: ignore[arg-type]

    cmd_line = _line_command(command, cfg)
    if cmd_line:
        lines.append(cmd_line)

    lines.extend(ctx.extra_lines)
    return "\n".join(lines)


# Default line spacing for single-axis figures (fontsize 9), figure coordinates.
_FIGURE_TITLE_LINE_SPACING = 0.033
# Tighter spacing for multi-subplot / figure-level titles.
_FIGURE_TITLE_LINE_SPACING_MULTI = 0.027
# Nudge title down within the reserved band (fraction of one line height).
_FIGURE_TITLE_TOP_INSET_LINES = 0.75
_TITLE_BAND_PAD = 0.006


def _scaled_line_spacing(line_spacing: float, fontsize: int) -> float:
    return line_spacing * (fontsize / 9.0)


def _line_extent(line_spacing: float, fontsize: int) -> float:
    """Approximate height of one title line in figure coordinates."""
    return _scaled_line_spacing(line_spacing, fontsize) * 0.92


def _figure_title_band_height(
    n_lines: int,
    line_spacing: float,
    fontsize: int,
) -> float:
    """Reserve space for ``n_lines`` stacked title lines."""
    extent = _line_extent(line_spacing, fontsize)
    if n_lines <= 1:
        return extent + _TITLE_BAND_PAD
    gap = _scaled_line_spacing(line_spacing, fontsize)
    return (n_lines - 1) * gap + extent + _TITLE_BAND_PAD


def _draw_figure_title_lines(
    fig: Any,
    title: str,
    *,
    y_top: float,
    fontsize: int,
    line_spacing: float,
) -> None:
    """Draw title lines in figure coordinates; ``Analysis:`` line is bold."""
    lines = title.split("\n")
    gap = _scaled_line_spacing(line_spacing, fontsize)
    y = y_top
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 and line.startswith("Analysis:") else "normal"
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight=weight,
            transform=fig.transFigure,
        )
        if i + 1 < len(lines):
            y -= gap


def apply_figure_title(
    fig: Any,
    title: str,
    *,
    ax: Any | None = None,
    use_suptitle: bool = False,
    suptitle_y: float = 1.0,
    rect_bottom: float = 0.0,
    title_fontsize: int = 9,
    band_trim_lines: float = 0.0,
    title_line_spacing: float | None = None,
) -> None:
    """Place a multi-line title above axes (``Analysis:`` line in bold)."""
    del suptitle_y
    if not use_suptitle and ax is None:
        raise ValueError("provide ax or use_suptitle=True")

    if title_line_spacing is None:
        line_spacing = (
            _FIGURE_TITLE_LINE_SPACING
            if ax is not None
            else _FIGURE_TITLE_LINE_SPACING_MULTI
        )
    else:
        line_spacing = title_line_spacing

    if fig.get_layout_engine() is not None:
        fig.set_layout_engine(None)
    n_lines = max(1, title.count("\n") + 1)
    band = _figure_title_band_height(n_lines, line_spacing, title_fontsize)
    gap = _scaled_line_spacing(line_spacing, title_fontsize)
    band -= band_trim_lines * gap
    band = max(_line_extent(line_spacing, title_fontsize) + _TITLE_BAND_PAD, band)
    axes_top = 1.0 - band
    fig.tight_layout(rect=(0.0, rect_bottom, 1.0, axes_top))
    y_top = (
        axes_top
        + band
        - _scaled_line_spacing(line_spacing, title_fontsize)
        * _FIGURE_TITLE_TOP_INSET_LINES
    )
    _draw_figure_title_lines(
        fig, title, y_top=y_top, fontsize=title_fontsize, line_spacing=line_spacing
    )
