"""Shared backend for comparing experimental .fcs medians to Mie model curves."""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator
from pydantic import BaseModel

from collect_mie.common import (
    resolve_fsc_half_angles_deg,
    resolve_ssc_band_deg,
    ssc_half_angle_deg,
    signal_mode_value,
)
from collect_mie.fsc_collection import fsc_uses_rect_mask
from collect_mie.config import load_config
from collect_mie.config_schema import CompareFscConfig, CompareSscConfig
from collect_mie.core import normalize_relative
from collect_mie.fsc_collection import diameter_sweep_fsc_from_config
from collect_mie.fcs_io import (
    LogHistogramPeakResult,
    channel_summary_and_bounds,
    load_manifest_rows,
    load_points_manifest,
    read_channel,
)
from collect_mie.plot_title import (
    TitleContext,
    apply_figure_title,
    build_figure_title,
)
from collect_mie.run_config import resolve_config_path, write_run_record
from collect_mie.ssc_collection import diameter_sweep_ssc_from_config

CompareChannelConfig = CompareSscConfig | CompareFscConfig
ScatterChannel = Literal["fsc", "ssc"]


class _CompareMedianFields(Protocol):
    data_source: str
    channel_summary: str
    median_error: str
    median_gate: str
    median_gate_log_decades: float
    median_gate_min_events: int
    peak_histogram_bins: int
    peak_selection: str
    peak_prominence_fraction: float
    peak_smooth_bins: int
    median_ci_percent: float
    median_bootstrap_n: int
    median_bootstrap_max_events: int


def resolve_histogram_output(
    primary_output: str | None,
    explicit: str | None,
) -> str | None:
    """Histogram PNG path: explicit config, else derived from ``output``."""
    if explicit:
        return explicit
    if primary_output:
        path = Path(primary_output)
        return str(path.with_name(f"{path.stem}_histograms{path.suffix}"))
    return None


def resolve_ssc_histogram_output(
    primary_output: str | None, explicit: str | None
) -> str | None:
    """Backward-compatible alias for histogram path derivation."""
    return resolve_histogram_output(primary_output, explicit)


def _relative_to_fitted(observed: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """Fractional residual ``(observed - fitted) / fitted`` (NaN where ``|fitted|`` ~ 0)."""
    obs = np.asarray(observed, dtype=float)
    fit = np.asarray(fitted, dtype=float)
    scale = max(float(np.max(np.abs(fit))), float(np.max(np.abs(obs))), 1.0)
    eps = np.finfo(float).eps * scale
    safe_fit = np.where(np.abs(fit) > eps, fit, np.nan)
    return (obs - fit) / safe_fit


def _relative_yerr_to_fitted(
    yerr: np.ndarray | None, fitted: np.ndarray
) -> np.ndarray | None:
    """Scale absolute observed uncertainties to fractional error vs fitted."""
    if yerr is None:
        return None
    fit = np.asarray(fitted, dtype=float)
    scale = max(float(np.max(np.abs(fit))), 1.0)
    eps = np.finfo(float).eps * scale
    return np.asarray(yerr, dtype=float) / np.where(np.abs(fit) > eps, np.abs(fit), np.nan)


def fit_metrics(observed: np.ndarray, fitted: np.ndarray) -> tuple[float, float]:
    """R² through origin and RMSE for observed vs fitted values at manifest diameters."""
    obs = np.asarray(observed, dtype=float)
    fit = np.asarray(fitted, dtype=float)
    resid = obs - fit
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(obs, obs))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return r2, rmse


def least_squares_scale(data: np.ndarray, model: np.ndarray) -> float:
    """Scalar ``s`` minimizing sum((data - s * model)**2) with no intercept."""
    m = np.asarray(model, dtype=float)
    d = np.asarray(data, dtype=float)
    denom = float(np.dot(m, m))
    if denom <= 0:
        raise SystemExit(
            "least_squares calibration: model values are all zero at manifest diameters"
        )
    return float(np.dot(d, m) / denom)


def _yerr_absolute(
    medians: np.ndarray, lows: np.ndarray | None, highs: np.ndarray | None
) -> np.ndarray | None:
    if lows is None or highs is None:
        return None
    lower = np.maximum(medians - lows, 0.0)
    upper = np.maximum(highs - medians, 0.0)
    return np.vstack([lower, upper])


def _normalize_median_bounds(
    medians: np.ndarray,
    lows: np.ndarray | None,
    highs: np.ndarray | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply the same normalization as ``normalize_relative`` to medians and bounds."""
    y = normalize_relative(medians, mode=mode)  # type: ignore[arg-type]
    if lows is None or highs is None:
        return y, None
    eps = float(np.finfo(float).eps)
    if mode == "first":
        denom = float(medians[0]) if medians[0] != 0 else eps
    else:
        denom = float(np.max(medians)) if np.max(medians) != 0 else eps
    low_n = lows / denom
    high_n = highs / denom
    lower = np.maximum(y - low_n, np.maximum(y * 1e-3, eps))
    upper = np.maximum(high_n - y, np.maximum(y * 1e-3, eps))
    return y, np.vstack([lower, upper])


def _scaled_prediction_sweep(
    cfg: CompareChannelConfig,
    *,
    diam_um: np.ndarray,
    n_particle: complex,
    wl_nm: float,
    smode: str,
    scale: float,
    channel: ScatterChannel,
    fsc_alpha_outer: float = 0.0,
    fsc_alpha_inner: float = 0.0,
    ssc_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense diameter sweep with model × LS calibration scale (instrument units)."""
    diam_nm = diam_um * 1000.0
    if channel == "fsc":
        raw = diameter_sweep_fsc_from_config(
            n_particle,
            diam_nm,
            wl_nm,
            cfg,
            fsc_alpha_outer,
            fsc_alpha_inner,
            polarization=cfg.polarization,
            signal_mode=smode,
        )
    else:
        raw = diameter_sweep_ssc_from_config(
            n_particle,
            diam_nm,
            wl_nm,
            cfg,
            ssc_alpha,
            polarization=cfg.polarization,
            signal_mode=smode,
        )
    return diam_um, raw * scale


def _prepare_compare_trace(
    medians: np.ndarray,
    lows: np.ndarray | None,
    highs: np.ndarray | None,
    model_raw: np.ndarray,
    normalize_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float | None]:
    """
    Return (experimental y, model y, yerr, ls_scale).

    ``ls_scale`` is set when ``normalize_mode == least_squares``; experimental
    values stay in instrument units and the model is multiplied by the scale.
    """
    if normalize_mode == "least_squares":
        scale = least_squares_scale(medians, model_raw)
        return medians, model_raw * scale, _yerr_absolute(medians, lows, highs), scale
    exp_y, yerr = _normalize_median_bounds(medians, lows, highs, normalize_mode)
    model_y = normalize_relative(model_raw, mode=normalize_mode)  # type: ignore[arg-type]
    return exp_y, model_y, yerr, None


def _channel_summary_kwargs(cfg: _CompareMedianFields) -> dict[str, object]:
    return {
        "summary": cfg.channel_summary,
        "error": cfg.median_error,
        "gate": cfg.median_gate,
        "half_decades": cfg.median_gate_log_decades,
        "min_events": cfg.median_gate_min_events,
        "peak_bins": cfg.peak_histogram_bins,
        "peak_selection": cfg.peak_selection,
        "peak_prominence_fraction": cfg.peak_prominence_fraction,
        "peak_smooth_bins": cfg.peak_smooth_bins,
        "ci_percent": cfg.median_ci_percent,
        "n_boot": cfg.median_bootstrap_n,
        "max_events": cfg.median_bootstrap_max_events,
    }


def _summarize_channel_values(
    values_per_file: list[np.ndarray], cfg: _CompareMedianFields
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    list[float | None],
    list[tuple[float, float] | None],
    list[LogHistogramPeakResult | None],
]:
    with warnings.catch_warnings():
        warnings.simplefilter("default", category=UserWarning)
        return channel_summary_and_bounds(
            values_per_file,
            **_channel_summary_kwargs(cfg),  # type: ignore[arg-type]
        )


def _summary_kind_label(cfg: _CompareMedianFields) -> str:
    if cfg.channel_summary == "peak_gated_median":
        return "peak-gated median"
    return "median"


def _data_legend_label(cfg: _CompareMedianFields, channel: str) -> str:
    if cfg.data_source == "table":
        return f"Data: {channel}"
    return f"Data: {channel} {_summary_kind_label(cfg)}{_median_legend_suffix(cfg)}"


def _median_legend_suffix(cfg: _CompareMedianFields) -> str:
    if cfg.median_error == "bootstrap":
        return f" ({cfg.median_ci_percent:g}% bootstrap CI)"
    return ""


def _style_compare_median_axis(ax: plt.Axes) -> None:
    """Major + minor grid on the compare (median vs diameter) panel."""
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2.0, 10.0)))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(which="minor", axis="x", alpha=0.35, linestyle=":", linewidth=0.8)
    ax.grid(which="minor", axis="y", alpha=0.35, linestyle=":", linewidth=0.8)


def _scatter_median(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    if yerr is not None:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            color=color,
            label=label,
            capsize=3,
            linestyle="none",
            markersize=6,
            zorder=5,
        )
    else:
        ax.scatter(x, y, color=color, marker=marker, zorder=5, label=label)


def _save_figure(fig: plt.Figure, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _format_ls_cal_line(channel: str, scale: float, r2: float, rmse: float) -> str:
    return f"{channel}: R²={r2:.3f}, RMSE={rmse:.4g}, scale={scale:.4g}"


def _draw_ls_fit_panels(
    ax_parity: plt.Axes,
    ax_resid: plt.Axes,
    *,
    diam_um: np.ndarray,
    name: str,
    observed: np.ndarray,
    fitted: np.ndarray,
    yerr: np.ndarray | None,
    color: str,
) -> None:
    """Parity and residual panels for one channel's LS fit."""
    resid = _relative_to_fitted(observed, fitted)
    rel_yerr = _relative_yerr_to_fitted(yerr, fitted)

    lo = float(min(np.min(observed), np.min(fitted)))
    hi = float(max(np.max(observed), np.max(fitted)))
    if lo >= hi:
        hi = lo * 1.01
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax_parity.plot(lim, lim, "k--", linewidth=1, alpha=0.6)
    if yerr is not None:
        ax_parity.errorbar(
            fitted,
            observed,
            yerr=yerr,
            fmt="o",
            color=color,
            capsize=3,
            linestyle="none",
        )
    else:
        ax_parity.scatter(fitted, observed, color=color, s=36, zorder=5)
    for i, d_um in enumerate(diam_um):
        ax_parity.annotate(
            f"{d_um:g}",
            (fitted[i], observed[i]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=7,
            alpha=0.85,
        )
    ax_parity.set_xlim(lim)
    ax_parity.set_ylim(lim)
    ax_parity.set_aspect("equal", adjustable="box")
    ax_parity.set_xlabel("Fitted (Mie × scale)")
    ax_parity.set_ylabel(f"Observed median ({name})")
    ax_parity.set_title(f"{name}: parity")
    ax_parity.grid(True, alpha=0.3)

    valid = np.isfinite(resid)
    if rel_yerr is not None:
        ax_resid.errorbar(
            diam_um[valid],
            resid[valid],
            yerr=rel_yerr[valid],
            fmt="o",
            color=color,
            capsize=3,
            linestyle="none",
        )
    else:
        ax_resid.scatter(diam_um[valid], resid[valid], color=color, s=36, zorder=5)
    ax_resid.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax_resid.set_xlabel("Diameter (µm)")
    ax_resid.set_ylabel("(observed − fitted) / fitted")
    ax_resid.set_title(f"{name}: relative residuals")
    ax_resid.grid(True, alpha=0.3)


def _peak_selection_label(selection: str) -> str:
    if selection == "highest_prominence":
        return "highest prominence"
    return "rightmost prominent"


def _histogram_title_lines(
    cfg: _CompareMedianFields, channel: ScatterChannel, channel_label: str
) -> list[str]:
    channel_name = channel.upper()
    lines = [f"Analysis: {channel_name} histograms — {channel_label}"]
    if cfg.channel_summary == "peak_gated_median":
        lines.append(
            f"summary={cfg.channel_summary}, peak: "
            f"{_peak_selection_label(cfg.peak_selection)}, "
            f"prom≥{cfg.peak_prominence_fraction:.0%} of max, "
            f"smooth={cfg.peak_smooth_bins} bins, "
            f"hist={cfg.peak_histogram_bins} bins"
        )
        lines.append(
            f"gate: ±{cfg.median_gate_log_decades:g} log decades "
            f"(min {cfg.median_gate_min_events} events)"
        )
    else:
        lines.append(f"summary={cfg.channel_summary}")
        if cfg.median_gate == "log_decades":
            lines.append(
                f"gate: median-centered ±{cfg.median_gate_log_decades:g} log decades "
                f"(min {cfg.median_gate_min_events} events)"
            )
        else:
            lines.append("gate: none")
    if cfg.median_error == "bootstrap":
        lines.append(
            f"median CI: {cfg.median_ci_percent:g}% bootstrap "
            f"(n={cfg.median_bootstrap_n})"
        )
    return lines


def _plot_histograms(
    *,
    cfg: _CompareMedianFields,
    channel: ScatterChannel,
    diam_um: np.ndarray,
    paths: list[str],
    channel_label: str,
    values_per_file: list[np.ndarray],
    summaries: np.ndarray,
    summary_label: str,
    gate_bounds: list[tuple[float, float] | None],
    peak_centers: list[float | None],
    bins: int,
    hist_output: str | None,
) -> None:
    n = len(paths)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()

    for i in range(n):
        ax = axes_flat[i]
        vals = values_per_file[i]
        summary = float(summaries[i])
        peak = peak_centers[i]
        positive = vals[vals > 0]
        if positive.size == 0:
            ax.text(
                0.5,
                0.5,
                "no positive events",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        else:
            lo = float(np.min(positive))
            hi = float(np.max(positive))
            if lo >= hi:
                hi = lo * 1.01
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
            ax.hist(
                positive,
                bins=bin_edges,
                color="C1",
                alpha=0.75,
                edgecolor="white",
                linewidth=0.3,
            )
            bounds = gate_bounds[i]
            if bounds is not None:
                gate_lo, gate_hi = bounds
                ax.axvline(gate_lo, color="0.45", linestyle=":", linewidth=1.2)
                ax.axvline(gate_hi, color="0.45", linestyle=":", linewidth=1.2)
            if peak is not None and peak > 0:
                ax.axvline(peak, color="C2", linestyle="-.", linewidth=1.2)
            if summary > 0:
                ax.axvline(summary, color="C3", linestyle="--", linewidth=1.5)
            ax.set_xscale("log")
        ax.set_title(f"d = {diam_um[i]:g} µm\n{Path(paths[i]).name}", fontsize=9)
        ax.set_xlabel(channel_label)
        ax.set_ylabel("Events")
        note_lines = [f"{summary_label} = {summary:.6g}"]
        if peak is not None:
            note_lines.append(f"peak = {peak:.6g}")
        bounds = gate_bounds[i]
        if bounds is not None:
            gate_lo, gate_hi = bounds
            note_lines.append(f"gate [{gate_lo:.4g}, {gate_hi:.4g}]")
        ax.text(
            0.03,
            0.97,
            "\n".join(note_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        ax.grid(True, alpha=0.25)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    summary_line = (
        f"{summary_label} (used in compare plot)"
        if cfg.channel_summary == "peak_gated_median"
        else summary_label
    )
    legend_handles = [
        Patch(facecolor="C1", alpha=0.75, edgecolor="white", label="events"),
        Line2D(
            [0], [0], color="0.45", linestyle=":", linewidth=1.2, label="gate bounds"
        ),
        Line2D(
            [0], [0], color="C2", linestyle="-.", linewidth=1.2, label="peak center"
        ),
        Line2D(
            [0],
            [0],
            color="C3",
            linestyle="--",
            linewidth=1.5,
            label=summary_line,
        ),
    ]
    apply_figure_title(
        fig,
        "\n".join(_histogram_title_lines(cfg, channel, channel_label)),
        use_suptitle=True,
        rect_bottom=0.07,
        title_fontsize=10,
        title_line_spacing=0.015,
        band_trim_lines=0.0,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=8,
        framealpha=0.9,
    )

    if hist_output:
        _save_figure(fig, hist_output)
    else:
        plt.show()


@dataclass
class ExperimentalTrace:
    diam_um: np.ndarray
    medians: np.ndarray
    lows: np.ndarray | None
    highs: np.ndarray | None
    channel_label: str
    from_table: bool
    paths: list[str] | None = None
    values_per_file: list[np.ndarray] | None = None
    peak_centers: list[float | None] | None = None
    gate_bounds: list[tuple[float, float] | None] | None = None


def _load_experimental_from_manifest(
    cfg: CompareChannelConfig,
    *,
    channel: ScatterChannel,
    channel_name: str,
) -> ExperimentalTrace:
    assert cfg.manifest is not None
    rows = load_manifest_rows(cfg.manifest)
    diam_exp_um = np.array([r[0] for r in rows])
    order = np.argsort(diam_exp_um)
    diam_exp_um = diam_exp_um[order]
    paths = [rows[i][1] for i in order]

    col_label, values_per_file = _load_channel_values_per_file(
        paths,
        channel_name,
        channel_naming=cfg.channel_naming,
        channel_label=channel,
    )
    med, lo, hi, peak_centers, gate_bounds, _ = _summarize_channel_values(
        values_per_file, cfg
    )
    return ExperimentalTrace(
        diam_um=diam_exp_um,
        medians=med,
        lows=lo,
        highs=hi,
        channel_label=col_label,
        from_table=False,
        paths=paths,
        values_per_file=values_per_file,
        peak_centers=peak_centers,
        gate_bounds=gate_bounds,
    )


def _load_experimental_from_table(
    cfg: CompareChannelConfig,
    *,
    channel_name: str,
) -> ExperimentalTrace:
    assert cfg.points_manifest is not None
    rows = load_points_manifest(cfg.points_manifest)
    diam_exp_um = np.array([r[0] for r in rows])
    med = np.array([r[1] for r in rows])
    order = np.argsort(diam_exp_um)
    return ExperimentalTrace(
        diam_um=diam_exp_um[order],
        medians=med[order],
        lows=None,
        highs=None,
        channel_label=channel_name,
        from_table=True,
    )


def _load_experimental_trace(
    cfg: CompareChannelConfig,
    *,
    channel: ScatterChannel,
    channel_name: str,
) -> ExperimentalTrace:
    if cfg.data_source == "table":
        return _load_experimental_from_table(cfg, channel_name=channel_name)
    return _load_experimental_from_manifest(
        cfg, channel=channel, channel_name=channel_name
    )


def _load_channel_values_per_file(
    paths: list[str],
    channel_name: str,
    *,
    channel_naming: str,
    channel_label: ScatterChannel,
) -> tuple[str, list[np.ndarray]]:
    col, values = read_channel(
        paths[0], channel_name, channel_naming=channel_naming  # type: ignore[arg-type]
    )
    per_file: list[np.ndarray] = [values]
    for p in paths[1:]:
        next_col, next_values = read_channel(
            p, channel_name, channel_naming=channel_naming  # type: ignore[arg-type]
        )
        if next_col != col:
            raise ValueError(
                f"{channel_label.upper()} column name mismatch: "
                f"{col!r} vs {next_col!r} in {p!r}"
            )
        per_file.append(next_values)
    return col, per_file


def _model_raw_at_manifest(
    cfg: CompareChannelConfig,
    *,
    channel: ScatterChannel,
    diam_nm: np.ndarray,
    n_particle: complex,
    wl_nm: float,
    smode: str,
    fsc_alpha_outer: float,
    fsc_alpha_inner: float,
    ssc_alpha: float,
) -> np.ndarray:
    if channel == "fsc":
        return diameter_sweep_fsc_from_config(
            n_particle,
            diam_nm,
            wl_nm,
            cfg,
            fsc_alpha_outer,
            fsc_alpha_inner,
            polarization=cfg.polarization,
            signal_mode=smode,
        )
    return diameter_sweep_ssc_from_config(
        n_particle,
        diam_nm,
        wl_nm,
        cfg,
        ssc_alpha,
        polarization=cfg.polarization,
        signal_mode=smode,
    )


def _model_curve_label(channel: ScatterChannel, *, use_ls: bool, prediction: bool) -> str:
    suffix = " (LS scaled)" if use_ls else ""
    if channel == "fsc":
        return f"Mie FSC {'prediction' if prediction else 'band'}{suffix}"
    return f"Mie SSC {'prediction' if prediction else 'calibration'}{suffix}"


def _compare_plot_style(channel: ScatterChannel, *, use_ls: bool) -> dict[str, str]:
    """Line/marker styling aligned with legacy compare-fcs SSC panel conventions."""
    data_color = "C0" if channel == "fsc" else "C1"
    if use_ls:
        return {
            "data_color": data_color,
            "data_marker": "s",
            "model_color": "k",
            "model_linestyle": "--",
        }
    return {
        "data_color": data_color,
        "data_marker": "s",
        "model_color": data_color,
        "model_linestyle": "-",
    }


def run_compare(
    cfg: CompareChannelConfig,
    *,
    command_name: str,
    channel: ScatterChannel,
    channel_name: str,
    config_path: str,
) -> None:
    """Compare experimental scatter medians to the Mie model for one channel."""
    trace = _load_experimental_trace(cfg, channel=channel, channel_name=channel_name)
    diam_exp_um = trace.diam_um
    med = trace.medians
    lo, hi = trace.lows, trace.highs

    fsc_alpha_outer = 0.0
    fsc_alpha_inner = 0.0
    ssc_alpha = 0.0
    uses_fsc_rect_mask = False
    if channel == "fsc":
        uses_fsc_rect_mask = fsc_uses_rect_mask(
            cfg.fsc_mask_half_angle_y_deg, cfg.fsc_mask_half_angle_z_deg
        )
        fsc_alpha_outer, fsc_alpha_inner = resolve_fsc_half_angles_deg(
            fsc_na_outer=cfg.fsc_na_outer,
            fsc_na_inner=cfg.fsc_na_inner,
            n_medium=cfg.n_medium,
            mask_half_angle_y_deg=cfg.fsc_mask_half_angle_y_deg,
            mask_half_angle_z_deg=cfg.fsc_mask_half_angle_z_deg,
        )
    else:
        ssc_alpha = ssc_half_angle_deg(cfg.ssc_na, cfg.n_medium)
        ssc_min, ssc_max = resolve_ssc_band_deg(
            ssc_na=cfg.ssc_na,
            ssc_center_deg=cfg.ssc_center_deg,
            n_medium=cfg.n_medium,
        )
        if ssc_min >= ssc_max:
            raise SystemExit("SSC band: min angle must be less than max angle")

    wl_nm = cfg.wavelength_nm
    n_particle = complex(cfg.n_real, cfg.n_imag)
    diam_nm = diam_exp_um * 1000.0
    smode = signal_mode_value(cfg.signal_mode)

    model_raw = _model_raw_at_manifest(
        cfg,
        channel=channel,
        diam_nm=diam_nm,
        n_particle=n_particle,
        wl_nm=wl_nm,
        smode=smode,
        fsc_alpha_outer=fsc_alpha_outer,
        fsc_alpha_inner=fsc_alpha_inner,
        ssc_alpha=ssc_alpha,
    )
    exp_y, model_y, yerr, ls_scale = _prepare_compare_trace(
        med, lo, hi, model_raw, cfg.normalize
    )

    use_instrument_units = cfg.normalize == "least_squares"
    style = _compare_plot_style(channel, use_ls=use_instrument_units)

    ls_diag: tuple[str, np.ndarray, np.ndarray, np.ndarray | None, str] | None = None
    if use_instrument_units and ls_scale is not None:
        ls_diag = (channel_name, exp_y, model_y, yerr, style["data_color"])

    embed_ls_panels = ls_diag is not None
    n_rows = 1 + (1 if embed_ls_panels else 0)

    if embed_ls_panels:
        fig = plt.figure(figsize=(9.0, 3.4 * n_rows), layout="constrained")
        gs = fig.add_gridspec(n_rows, 2)
        ax_compare = fig.add_subplot(gs[0, :])
        ax_parity = fig.add_subplot(gs[1, 0])
        ax_resid = fig.add_subplot(gs[1, 1])
    else:
        fig, ax_compare = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_parity = ax_resid = None  # type: ignore[assignment]

    diam_pred_um = np.linspace(cfg.d_min_um, cfg.d_max_um, cfg.n_diameters)

    if use_instrument_units and ls_scale is not None:
        pred_um, pred = _scaled_prediction_sweep(
            cfg,
            diam_um=diam_pred_um,
            n_particle=n_particle,
            wl_nm=wl_nm,
            smode=smode,
            scale=ls_scale,
            channel=channel,
            fsc_alpha_outer=fsc_alpha_outer,
            fsc_alpha_inner=fsc_alpha_inner,
            ssc_alpha=ssc_alpha,
        )
        ax_compare.plot(
            pred_um,
            pred,
            color=style["model_color"],
            linestyle=style["model_linestyle"],
            linewidth=1.2,
            label=_model_curve_label(channel, use_ls=True, prediction=True),
            zorder=2,
        )
    else:
        ax_compare.plot(
            diam_exp_um,
            model_y,
            color=style["model_color"],
            linestyle=style["model_linestyle"],
            label=_model_curve_label(channel, use_ls=False, prediction=False),
        )

    data_label = _data_legend_label(cfg, channel_name)
    _scatter_median(
        ax_compare,
        diam_exp_um,
        exp_y,
        yerr,
        color=style["data_color"],
        marker=style["data_marker"],
        label=data_label,
    )
    ax_compare.set_xlabel("Diameter (µm)")
    rel_ylabel = "Relative FSC" if channel == "fsc" else "Relative SSC"
    ax_compare.set_ylabel(
        f"{'Median' if trace.from_table else _summary_kind_label(cfg).title()} {channel_name}"
        if use_instrument_units
        else rel_ylabel
    )
    ax_compare.legend(loc="best")
    ax_compare.set_yscale("log")
    _style_compare_median_axis(ax_compare)

    extra_title_lines: list[str] = []
    if use_instrument_units and ls_scale is not None:
        r2, rmse = fit_metrics(exp_y, model_y)
        extra_title_lines.append(
            _format_ls_cal_line(channel_name, ls_scale, r2, rmse)
        )

    if embed_ls_panels and ls_diag is not None and ax_parity is not None and ax_resid is not None:
        name, obs, fit, err, color = ls_diag
        _draw_ls_fit_panels(
            ax_parity,
            ax_resid,
            diam_um=diam_exp_um,
            name=name,
            observed=obs,
            fitted=fit,
            yerr=err,
            color=color,
        )

    title_ctx = TitleContext(
        uses_fsc=channel == "fsc",
        uses_ssc=channel == "ssc",
        fsc_alpha_outer=fsc_alpha_outer if channel == "fsc" else None,
        fsc_alpha_inner=(
            fsc_alpha_inner if channel == "fsc" and not uses_fsc_rect_mask else None
        ),
        ssc_alpha=ssc_alpha if channel == "ssc" else None,
        extra_lines=extra_title_lines,
    )
    apply_figure_title(
        fig,
        build_figure_title(command_name, cfg, title_ctx),
        use_suptitle=True,
    )

    if cfg.output:
        _save_figure(fig, cfg.output)
    else:
        plt.show()

    if not trace.from_table and trace.values_per_file is not None:
        hist_output = resolve_histogram_output(cfg.output, cfg.histogram_output)
        _plot_histograms(
            cfg=cfg,
            channel=channel,
            diam_um=diam_exp_um,
            paths=trace.paths or [],
            channel_label=trace.channel_label,
            values_per_file=trace.values_per_file,
            summaries=med,
            summary_label=_summary_kind_label(cfg),
            gate_bounds=trace.gate_bounds or [],
            peak_centers=trace.peak_centers or [],
            bins=cfg.histogram_bins,
            hist_output=hist_output,
        )

    if cfg.write_run_record:
        write_run_record(
            cfg.write_run_record,
            command_name=command_name,
            config_path=config_path,
            resolved=cfg,
        )


def _main_for_command(
    command_name: str,
    channel: ScatterChannel,
    model_cls: type[BaseModel],
    *,
    config_path: str | None = None,
    argv: list[str] | None = None,
) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, command_name)
    assert isinstance(cfg, model_cls)
    channel_name = cfg.fsc_channel if channel == "fsc" else cfg.ssc_channel
    run_compare(
        cfg,
        command_name=command_name,
        channel=channel,
        channel_name=channel_name,
        config_path=path,
    )
