"""Compare experimental .fcs medians to Mie model curves at manifest diameters."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator

from collect_mie.common import (
    fsc_half_angles_deg,
    resolve_ssc_band_deg,
    ssc_half_angle_deg,
    signal_mode_value,
)
from collect_mie.config import load_config
from collect_mie.config_schema import CompareFcsConfig
from collect_mie.core import normalize_relative
from collect_mie.fsc_collection import diameter_sweep_fsc_from_config
from collect_mie.ssc_collection import diameter_sweep_ssc_from_config
from collect_mie.fcs_io import (
    LogHistogramPeakResult,
    channel_summary_and_bounds,
    load_manifest_rows,
    read_channel,
)
from collect_mie.plot_title import (
    TitleContext,
    apply_figure_title,
    build_figure_title,
)
from collect_mie.run_config import resolve_config_path, write_run_record


def resolve_ssc_histogram_output(
    primary_output: str | None, explicit: str | None
) -> str | None:
    """Histogram PNG path: explicit config, else derived from ``output``."""
    if explicit:
        return explicit
    if primary_output:
        path = Path(primary_output)
        return str(path.with_name(f"{path.stem}_ssc_histograms{path.suffix}"))
    return None


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
    cfg: CompareFcsConfig,
    *,
    diam_um: np.ndarray,
    n_particle: complex,
    wl_nm: float,
    smode: str,
    scale: float,
    channel: Literal["fsc", "ssc"],
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


def _channel_summary_kwargs(cfg: CompareFcsConfig) -> dict[str, object]:
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
    values_per_file: list[np.ndarray], cfg: CompareFcsConfig
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


def _summary_kind_label(cfg: CompareFcsConfig) -> str:
    if cfg.channel_summary == "peak_gated_median":
        return "peak-gated median"
    return "median"


def _data_legend_label(cfg: CompareFcsConfig, channel: str) -> str:
    return f"Data: {channel} {_summary_kind_label(cfg)}{_median_legend_suffix(cfg)}"


def _median_legend_suffix(cfg: CompareFcsConfig) -> str:
    if cfg.median_error == "bootstrap":
        return f" ({cfg.median_ci_percent:g}% bootstrap CI)"
    return ""


def _style_compare_median_axis(ax: plt.Axes) -> None:
    """Major + minor grid on the top compare (median vs diameter) panel."""
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


def _ssc_histogram_title_lines(cfg: CompareFcsConfig, channel_label: str) -> list[str]:
    lines = [f"Analysis: SSC histograms — {channel_label}"]
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


def _plot_ssc_histograms(
    *,
    cfg: CompareFcsConfig,
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
        "\n".join(_ssc_histogram_title_lines(cfg, channel_label)),
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


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, "compare-fcs")
    assert isinstance(cfg, CompareFcsConfig)

    rows = load_manifest_rows(cfg.manifest)
    diam_exp_um = np.array([r[0] for r in rows])
    order = np.argsort(diam_exp_um)
    diam_exp_um = diam_exp_um[order]
    paths = [rows[i][1] for i in order]

    plot_fsc = cfg.fsc_channel is not None

    if plot_fsc:
        fsc_per_file = [
            read_channel(p, cfg.fsc_channel, channel_naming=cfg.channel_naming)[1]
            for p in paths
        ]
        fsc_med, fsc_lo, fsc_hi, _, _, _ = _summarize_channel_values(fsc_per_file, cfg)
    ssc_col, ssc_values = read_channel(
        paths[0], cfg.ssc_channel, channel_naming=cfg.channel_naming
    )
    ssc_per_file: list[np.ndarray] = [ssc_values]
    for p in paths[1:]:
        col, values = read_channel(
            p, cfg.ssc_channel, channel_naming=cfg.channel_naming
        )
        if col != ssc_col:
            raise ValueError(
                f"SSC column name mismatch: {ssc_col!r} vs {col!r} in {p!r}"
            )
        ssc_per_file.append(values)
    (
        ssc_med,
        ssc_lo,
        ssc_hi,
        ssc_peak_centers,
        ssc_gate_bounds,
        _,
    ) = _summarize_channel_values(ssc_per_file, cfg)
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

    fsc_ls_scale: float | None = None
    if plot_fsc:
        fsc_alpha_outer, fsc_alpha_inner = fsc_half_angles_deg(
            cfg.fsc_na_outer, cfg.fsc_na_inner, cfg.n_medium
        )
        fsc_model_raw = diameter_sweep_fsc_from_config(
            n_particle,
            diam_nm,
            wl_nm,
            cfg,
            fsc_alpha_outer,
            fsc_alpha_inner,
            polarization=cfg.polarization,
            signal_mode=smode,
        )
        fsc_exp, fsc_model, fsc_yerr, fsc_ls_scale = _prepare_compare_trace(
            fsc_med, fsc_lo, fsc_hi, fsc_model_raw, cfg.normalize
        )

    ssc_model_raw = diameter_sweep_ssc_from_config(
        n_particle,
        diam_nm,
        wl_nm,
        cfg,
        ssc_alpha,
        polarization=cfg.polarization,
        signal_mode=smode,
    )
    ssc_exp, ssc_model, ssc_yerr, ssc_ls_scale = _prepare_compare_trace(
        ssc_med, ssc_lo, ssc_hi, ssc_model_raw, cfg.normalize
    )

    use_instrument_units = cfg.normalize == "least_squares"
    model_label_suffix = " (LS scaled)" if use_instrument_units else ""

    ls_diag_channels: list[
        tuple[str, np.ndarray, np.ndarray, np.ndarray | None, str]
    ] = []
    if use_instrument_units:
        if plot_fsc and fsc_ls_scale is not None:
            ls_diag_channels.append(
                (str(cfg.fsc_channel), fsc_exp, fsc_model, fsc_yerr, "C0")
            )
        if ssc_ls_scale is not None:
            ls_diag_channels.append(
                (cfg.ssc_channel, ssc_exp, ssc_model, ssc_yerr, "C1")
            )

    embed_ls_panels = bool(ls_diag_channels)
    n_compare_rows = 2 if plot_fsc else 1
    n_rows = n_compare_rows + (len(ls_diag_channels) if embed_ls_panels else 0)

    if embed_ls_panels:
        fig = plt.figure(figsize=(9.0, 3.4 * n_rows), layout="constrained")
        gs = fig.add_gridspec(n_rows, 2)
        row = 0
        ax_fsc = fig.add_subplot(gs[row, :]) if plot_fsc else None
        if plot_fsc:
            row += 1
        ax_ssc = fig.add_subplot(gs[row, :])
        if ax_fsc is not None:
            ax_ssc.sharex(ax_fsc)
        row += 1
        ls_panel_axes: list[tuple[plt.Axes, plt.Axes]] = []
        for _ in ls_diag_channels:
            ls_panel_axes.append(
                (fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1]))
            )
            row += 1
    elif plot_fsc:
        fig, (ax_fsc, ax_ssc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ls_panel_axes = []
    else:
        fig, ax_ssc = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_fsc = None
        ls_panel_axes = []

    diam_pred_um = np.linspace(cfg.d_min_um, cfg.d_max_um, cfg.n_diameters)

    if plot_fsc:
        if use_instrument_units and fsc_ls_scale is not None:
            fsc_pred_um, fsc_pred = _scaled_prediction_sweep(
                cfg,
                diam_um=diam_pred_um,
                n_particle=n_particle,
                wl_nm=wl_nm,
                smode=smode,
                scale=fsc_ls_scale,
                channel="fsc",
                fsc_alpha_outer=fsc_alpha_outer,
                fsc_alpha_inner=fsc_alpha_inner,
            )
            ax_fsc.plot(
                fsc_pred_um,
                fsc_pred,
                color="C0",
                linewidth=1.2,
                label=f"Mie FSC prediction{model_label_suffix}",
                zorder=2,
            )
        else:
            ax_fsc.plot(
                diam_exp_um,
                fsc_model,
                color="C0",
                label=f"Mie FSC band{model_label_suffix}",
            )
        fsc_label = _data_legend_label(cfg, str(cfg.fsc_channel))
        _scatter_median(
            ax_fsc,
            diam_exp_um,
            fsc_exp,
            fsc_yerr,
            color="C0",
            marker="o",
            label=fsc_label,
        )
        ax_fsc.set_ylabel(
            f"{_summary_kind_label(cfg).title()} {cfg.fsc_channel}"
            if use_instrument_units
            else "Relative FSC"
        )
        ax_fsc.legend(loc="best")
        ax_fsc.set_yscale("log")
        _style_compare_median_axis(ax_fsc)

    if use_instrument_units and ssc_ls_scale is not None:
        ssc_pred_um, ssc_pred = _scaled_prediction_sweep(
            cfg,
            diam_um=diam_pred_um,
            n_particle=n_particle,
            wl_nm=wl_nm,
            smode=smode,
            scale=ssc_ls_scale,
            channel="ssc",
            ssc_alpha=ssc_alpha,
        )
        ax_ssc.plot(
            ssc_pred_um,
            ssc_pred,
            color="k",
            linestyle="--",
            linewidth=1.2,
            label=f"Mie SSC prediction{model_label_suffix}",
            zorder=2,
        )
    else:
        ax_ssc.plot(
            diam_exp_um,
            ssc_model,
            color="C1",
            label=f"Mie SSC calibration{model_label_suffix}",
        )
    ssc_label = _data_legend_label(cfg, cfg.ssc_channel)
    _scatter_median(
        ax_ssc,
        diam_exp_um,
        ssc_exp,
        ssc_yerr,
        color="C1",
        marker="s",
        label=ssc_label,
    )
    ax_ssc.set_xlabel("Diameter (µm)")
    ax_ssc.set_ylabel(
        f"{_summary_kind_label(cfg).title()} {cfg.ssc_channel}"
        if use_instrument_units
        else "Relative SSC"
    )
    ax_ssc.legend(loc="best")
    ax_ssc.set_yscale("log")
    _style_compare_median_axis(ax_ssc)

    extra_title_lines: list[str] = []
    if use_instrument_units:
        if plot_fsc and fsc_ls_scale is not None:
            fsc_r2, fsc_rmse = fit_metrics(fsc_exp, fsc_model)
            extra_title_lines.append(
                _format_ls_cal_line(
                    str(cfg.fsc_channel), fsc_ls_scale, fsc_r2, fsc_rmse
                )
            )
        if ssc_ls_scale is not None:
            ssc_r2, ssc_rmse = fit_metrics(ssc_exp, ssc_model)
            extra_title_lines.append(
                _format_ls_cal_line(cfg.ssc_channel, ssc_ls_scale, ssc_r2, ssc_rmse)
            )

    for (ax_parity, ax_resid), (name, obs, fit, yerr, color) in zip(
        ls_panel_axes, ls_diag_channels
    ):
        _draw_ls_fit_panels(
            ax_parity,
            ax_resid,
            diam_um=diam_exp_um,
            name=name,
            observed=obs,
            fitted=fit,
            yerr=yerr,
            color=color,
        )
    title_ctx = TitleContext(
        uses_fsc=plot_fsc,
        uses_ssc=True,
        fsc_alpha_outer=fsc_alpha_outer if plot_fsc else None,
        fsc_alpha_inner=fsc_alpha_inner if plot_fsc else None,
        ssc_alpha=ssc_alpha,
        extra_lines=extra_title_lines,
    )
    apply_figure_title(
        fig,
        build_figure_title("compare-fcs", cfg, title_ctx),
        use_suptitle=True,
    )

    hist_output = resolve_ssc_histogram_output(cfg.output, cfg.ssc_histogram_output)

    if cfg.output:
        _save_figure(fig, cfg.output)
    else:
        plt.show()

    _plot_ssc_histograms(
        cfg=cfg,
        diam_um=diam_exp_um,
        paths=paths,
        channel_label=ssc_col,
        values_per_file=ssc_per_file,
        summaries=ssc_med,
        summary_label=_summary_kind_label(cfg),
        gate_bounds=ssc_gate_bounds,
        peak_centers=ssc_peak_centers,
        bins=cfg.ssc_histogram_bins,
        hist_output=hist_output,
    )

    if cfg.write_run_record:
        write_run_record(
            cfg.write_run_record,
            command_name="compare-fcs",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
