"""Read Flow Cytometry Standard (.fcs) files and summarize channels for model comparison."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

MedianError = Literal["none", "bootstrap"]
ChannelGate = Literal["none", "log_decades"]
ChannelSummary = Literal["median", "peak_gated_median"]
PeakSelection = Literal["highest_prominence", "rightmost_prominent"]
GateCenter = Literal["median", "peak"]

try:
    import fcsparser
except ImportError as err:  # pragma: no cover
    raise ImportError("fcsparser is required for FCS support") from err


def find_channel_column(df: Any, name: str) -> str:
    """
    Match user-provided channel label (e.g. 'FSC-A') to an actual DataFrame column.

    fcsparser column names come from FCS metadata ($PnS vs $PnN); spelling varies by
    acquisition software, so we allow exact match first then substring match.
    """
    name_l = name.strip().lower()
    cols = list(df.columns)
    for c in cols:
        if str(c).strip().lower() == name_l:
            return c
    for c in cols:
        if name_l in str(c).lower():
            return c
    raise KeyError(f"No column matching {name!r}. Available: {cols}")


def read_channel(
    path: str, channel: str, *, channel_naming: str = "$PnS"
) -> tuple[str, np.ndarray]:
    """Load one numeric channel; returns resolved column name and finite event values."""
    _, data = fcsparser.parse(path, channel_naming=channel_naming)
    col = find_channel_column(data, channel)
    values = np.asarray(data[col], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite events in {path!r} column {col!r}")
    return col, values


def _positive_finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values[np.isfinite(values) & (values > 0)], dtype=float)


def _smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    w = int(window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(arr, kernel, mode="same")


def _local_maxima_indices(counts: np.ndarray) -> np.ndarray:
    """Indices of local maxima in a 1D count array."""
    if counts.size < 3:
        return np.array([], dtype=int)
    peaks: list[int] = []
    for i in range(1, counts.size - 1):
        left, mid, right = counts[i - 1], counts[i], counts[i + 1]
        if mid > left and mid >= right:
            peaks.append(i)
        elif mid >= left and mid > right:
            peaks.append(i)
    return np.asarray(peaks, dtype=int)


def _peak_prominences(counts: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """Prominence of each peak index: height above the higher enclosing minimum."""
    prominences = np.empty(peak_indices.size, dtype=float)
    for j, idx in enumerate(peak_indices):
        left_min = float(np.min(counts[: idx + 1]))
        right_min = float(np.min(counts[idx:]))
        base = max(left_min, right_min)
        prominences[j] = float(counts[idx]) - base
    return prominences


@dataclass(frozen=True)
class LogHistogramPeakResult:
    """Peak-find outcome for one file's log histogram."""

    center: float
    prominence: float | None
    relative_prominence: float | None
    used_median_fallback: bool


def log_histogram_peak_find(
    values: np.ndarray,
    *,
    bins: int = 200,
    smooth_bins: int = 3,
    prominence_fraction: float = 0.05,
    selection: PeakSelection = "rightmost_prominent",
) -> LogHistogramPeakResult:
    """
    Estimate the bright population center from a log-spaced event histogram.

    Finds local maxima on smoothed counts, filters by relative prominence, then picks
    the highest-prominence or rightmost qualifying peak. Falls back to the median of
    positive events when no peak passes the prominence filter.
    """
    positive = _positive_finite(values)
    if positive.size == 0:
        raise ValueError("no positive events for peak finding")

    lo = float(np.min(positive))
    hi = float(np.max(positive))
    if lo >= hi:
        return LogHistogramPeakResult(
            center=lo,
            prominence=None,
            relative_prominence=None,
            used_median_fallback=True,
        )

    edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    counts, _ = np.histogram(positive, bins=edges)
    counts = _smooth_1d(counts.astype(float), smooth_bins)
    centers = np.sqrt(edges[:-1] * edges[1:])

    peak_idx = _local_maxima_indices(counts)
    if peak_idx.size == 0:
        warnings.warn(
            "log histogram peak find: no local maxima; using median of positive events",
            stacklevel=2,
        )
        return LogHistogramPeakResult(
            center=float(np.median(positive)),
            prominence=None,
            relative_prominence=None,
            used_median_fallback=True,
        )

    prominences = _peak_prominences(counts, peak_idx)
    max_prom = float(np.max(prominences))
    if max_prom <= 0:
        warnings.warn(
            "log histogram peak find: zero prominence; using median of positive events",
            stacklevel=2,
        )
        return LogHistogramPeakResult(
            center=float(np.median(positive)),
            prominence=None,
            relative_prominence=None,
            used_median_fallback=True,
        )

    min_prom = prominence_fraction * max_prom
    keep = prominences >= min_prom
    if not np.any(keep):
        keep = prominences >= float(np.max(prominences)) * 0.5

    candidates = peak_idx[keep]
    cand_prom = prominences[keep]

    if selection == "highest_prominence":
        pick = int(np.argmax(cand_prom))
    else:
        pick = int(np.argmax(candidates))

    best = int(candidates[pick])
    sel_prom = float(cand_prom[pick])
    return LogHistogramPeakResult(
        center=float(centers[best]),
        prominence=sel_prom,
        relative_prominence=sel_prom / max_prom,
        used_median_fallback=False,
    )


def log_histogram_peak_center(
    values: np.ndarray,
    *,
    bins: int = 200,
    smooth_bins: int = 3,
    prominence_fraction: float = 0.05,
    selection: PeakSelection = "rightmost_prominent",
) -> float:
    """Return only the peak center from :func:`log_histogram_peak_find`."""
    return log_histogram_peak_find(
        values,
        bins=bins,
        smooth_bins=smooth_bins,
        prominence_fraction=prominence_fraction,
        selection=selection,
    ).center


def gate_log_decades(
    values: np.ndarray,
    *,
    half_decades: float,
    min_events: int = 100,
    center: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Keep positive events within [center/10^w, center×10^w].

    ``center`` defaults to the median of positive events when omitted.
    If fewer than ``min_events`` pass the gate, returns all positive events and still
    reports the gate bounds from the chosen center.
    """
    positive = _positive_finite(values)
    if positive.size == 0:
        raise ValueError("no positive events to gate")

    if center is None:
        center_f = float(np.median(positive))
    else:
        center_f = float(center)
    lo = center_f / (10.0**half_decades)
    hi = center_f * (10.0**half_decades)
    gated = positive[(positive >= lo) & (positive <= hi)]
    if gated.size < min_events:
        warnings.warn(
            f"log-decade gate [{lo:.6g}, {hi:.6g}] kept {gated.size} events "
            f"(< median_gate_min_events={min_events}); using all positive events",
            stacklevel=2,
        )
        return positive, lo, hi
    return gated, lo, hi


def apply_channel_gate(
    values_per_file: list[np.ndarray],
    gate: ChannelGate,
    *,
    half_decades: float,
    min_events: int,
    gate_center: GateCenter = "median",
    peak_bins: int = 200,
    peak_selection: PeakSelection = "rightmost_prominent",
    peak_prominence_fraction: float = 0.05,
    peak_smooth_bins: int = 3,
) -> tuple[list[np.ndarray], list[tuple[float, float] | None], list[float | None]]:
    """
    Optionally gate each file's events before channel summaries.

    Returns gated values, optional (lo, hi) bounds per file, and optional peak centers
    used when ``gate_center='peak'``.
    """
    if gate == "none":
        return values_per_file, [None] * len(values_per_file), [None] * len(values_per_file)

    gated: list[np.ndarray] = []
    bounds: list[tuple[float, float] | None] = []
    peak_centers: list[float | None] = []
    for values in values_per_file:
        center: float | None = None
        if gate_center == "peak":
            center = log_histogram_peak_find(
                values,
                bins=peak_bins,
                smooth_bins=peak_smooth_bins,
                prominence_fraction=peak_prominence_fraction,
                selection=peak_selection,
            ).center
            peak_centers.append(center)
        else:
            peak_centers.append(None)

        g, lo, hi = gate_log_decades(
            values,
            half_decades=half_decades,
            min_events=min_events,
            center=center,
        )
        gated.append(g)
        bounds.append((lo, hi))
    return gated, bounds, peak_centers


def _peak_window_gate(
    values: np.ndarray,
    *,
    half_decades: float,
    min_events: int,
    peak_bins: int,
    peak_selection: PeakSelection,
    peak_prominence_fraction: float,
    peak_smooth_bins: int,
) -> tuple[np.ndarray, float, float, LogHistogramPeakResult]:
    """Gate positive events in a log-decade window centered on the histogram peak."""
    peak = log_histogram_peak_find(
        values,
        bins=peak_bins,
        smooth_bins=peak_smooth_bins,
        prominence_fraction=peak_prominence_fraction,
        selection=peak_selection,
    )
    gated, lo, hi = gate_log_decades(
        values,
        half_decades=half_decades,
        min_events=min_events,
        center=peak.center,
    )
    return gated, lo, hi, peak


def channel_summary_and_bounds(
    values_per_file: list[np.ndarray],
    summary: ChannelSummary,
    error: MedianError,
    gate: ChannelGate,
    *,
    half_decades: float,
    min_events: int,
    peak_bins: int = 200,
    peak_selection: PeakSelection = "rightmost_prominent",
    peak_prominence_fraction: float = 0.05,
    peak_smooth_bins: int = 3,
    ci_percent: float = 95.0,
    n_boot: int = 2000,
    max_events: int = 20_000,
    rng: np.random.Generator | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    list[float | None],
    list[tuple[float, float] | None],
    list[LogHistogramPeakResult | None],
]:
    """
    Per-file channel summaries and optional bootstrap CI bounds.

    ``median``: optional ``log_decades`` gate centered on the ungated median, then median.
    ``peak_gated_median``: peak-centered log-decade window (always), then median of gated
    events, using ``half_decades`` / ``min_events`` for the window width.

    Returns ``(summaries, lows, highs, peak_centers, gate_bounds, peak_results)``.
    """
    n = len(values_per_file)
    peak_centers: list[float | None] = [None] * n
    gate_bounds: list[tuple[float, float] | None] = [None] * n
    peak_results: list[LogHistogramPeakResult | None] = [None] * n

    if summary == "median":
        if gate == "log_decades":
            gated, gate_bounds, _ = apply_channel_gate(
                values_per_file,
                gate,
                half_decades=half_decades,
                min_events=min_events,
                gate_center="median",
            )
        else:
            gated = values_per_file
        return _summaries_from_gated(
            gated,
            error,
            ci_percent=ci_percent,
            n_boot=n_boot,
            max_events=max_events,
            rng=rng,
            peak_centers=peak_centers,
            gate_bounds=gate_bounds,
            peak_results=peak_results,
        )

    gated_list: list[np.ndarray] = []
    for i, values in enumerate(values_per_file):
        g, lo, hi, peak = _peak_window_gate(
            values,
            half_decades=half_decades,
            min_events=min_events,
            peak_bins=peak_bins,
            peak_selection=peak_selection,
            peak_prominence_fraction=peak_prominence_fraction,
            peak_smooth_bins=peak_smooth_bins,
        )
        gated_list.append(g)
        peak_centers[i] = peak.center
        gate_bounds[i] = (lo, hi)
        peak_results[i] = peak

    return _summaries_from_gated(
        gated_list,
        error,
        ci_percent=ci_percent,
        n_boot=n_boot,
        max_events=max_events,
        rng=rng,
        peak_centers=peak_centers,
        gate_bounds=gate_bounds,
        peak_results=peak_results,
    )


def _summaries_from_gated(
    gated: list[np.ndarray],
    error: MedianError,
    *,
    ci_percent: float,
    n_boot: int,
    max_events: int,
    rng: np.random.Generator | None,
    peak_centers: list[float | None],
    gate_bounds: list[tuple[float, float] | None],
    peak_results: list[LogHistogramPeakResult | None],
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    list[float | None],
    list[tuple[float, float] | None],
    list[LogHistogramPeakResult | None],
]:
    summaries = np.empty(len(gated), dtype=float)
    if error == "none":
        for i, values in enumerate(gated):
            summaries[i] = float(np.median(values))
        return summaries, None, None, peak_centers, gate_bounds, peak_results

    lows = np.empty(len(gated), dtype=float)
    highs = np.empty(len(gated), dtype=float)
    rng = rng or np.random.default_rng()
    for i, values in enumerate(gated):
        med, lo, hi = bootstrap_median_ci(
            values,
            ci_percent=ci_percent,
            n_boot=n_boot,
            max_events=max_events,
            rng=rng,
        )
        summaries[i] = med
        lows[i] = lo
        highs[i] = hi
    return summaries, lows, highs, peak_centers, gate_bounds, peak_results


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    ci_percent: float = 95.0,
    n_boot: int = 2000,
    max_events: int = 20_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Median and two-sided bootstrap percentile CI for the median.

    Large event lists are subsampled to ``max_events`` before resampling.
    """
    rng = rng or np.random.default_rng()
    sample = np.asarray(values, dtype=float)
    if sample.size > max_events:
        sample = rng.choice(sample, size=max_events, replace=False)

    med = float(np.median(sample))
    if sample.size < 2:
        return med, med, med

    n = sample.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(sample[idx], axis=1)
    tail = (100.0 - ci_percent) / 2.0
    lo, hi = np.percentile(boot_medians, [tail, 100.0 - tail])
    return med, float(lo), float(hi)


def channel_median_and_bounds(
    values_per_file: list[np.ndarray],
    error: MedianError,
    *,
    ci_percent: float = 95.0,
    n_boot: int = 2000,
    max_events: int = 20_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Per-file medians and optional bootstrap CI bounds (legacy median-only API)."""
    summaries, lows, highs, _, _, _ = channel_summary_and_bounds(
        values_per_file,
        "median",
        error,
        "none",
        half_decades=0.5,
        min_events=1,
        ci_percent=ci_percent,
        n_boot=n_boot,
        max_events=max_events,
        rng=rng,
    )
    return summaries, lows, highs


def median_channel(path: str, channel: str, *, channel_naming: str = "$PnS") -> float:
    """Median of one numeric channel over gated/unfiltered events (caller may gate upstream)."""
    _, values = read_channel(path, channel, channel_naming=channel_naming)
    return float(np.median(values))


def load_points_manifest(path: str) -> list[tuple[float, float]]:
    """
    Parse a diameter + median table (CSV or whitespace):

      diameter_um  median

    Lines starting with # are ignored.
    """
    rows: list[tuple[float, float]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(f"Bad points manifest line: {line!r}")
            rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise ValueError(f"No rows in points manifest {path!r}")
    return rows


def load_manifest_rows(path: str) -> list[tuple[float, str]]:
    """
    Parse a simple manifest: two columns per line (CSV or whitespace):

      diameter_um  /path/to/file.fcs

    Lines starting with # are ignored.
    """
    rows: list[tuple[float, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(f"Bad manifest line: {line!r}")
            rows.append((float(parts[0]), " ".join(parts[1:])))
    if not rows:
        raise ValueError(f"No rows in manifest {path!r}")
    return rows
