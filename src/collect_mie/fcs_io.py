"""Read Flow Cytometry Standard (.fcs) files and summarize channels for model comparison."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

MedianError = Literal["none", "bootstrap"]
ChannelGate = Literal["none", "log_decades"]

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


def gate_log_decades(
    values: np.ndarray,
    *,
    half_decades: float,
    min_events: int = 100,
) -> tuple[np.ndarray, float, float]:
    """
    Keep positive events within [median/10^w, median×10^w] using the ungated median.

    If fewer than ``min_events`` pass the gate, returns all positive events and still
    reports the gate bounds from the first-pass median.
    """
    positive = np.asarray(values[np.isfinite(values) & (values > 0)], dtype=float)
    if positive.size == 0:
        raise ValueError("no positive events to gate")

    center = float(np.median(positive))
    lo = center / (10.0**half_decades)
    hi = center * (10.0**half_decades)
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
) -> tuple[list[np.ndarray], list[tuple[float, float] | None]]:
    """Optionally gate each file's events before median / bootstrap summaries."""
    if gate == "none":
        return values_per_file, [None] * len(values_per_file)

    gated: list[np.ndarray] = []
    bounds: list[tuple[float, float] | None] = []
    for values in values_per_file:
        g, lo, hi = gate_log_decades(
            values, half_decades=half_decades, min_events=min_events
        )
        gated.append(g)
        bounds.append((lo, hi))
    return gated, bounds


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
    """Per-file medians and optional bootstrap CI bounds from the same events."""
    medians = np.empty(len(values_per_file), dtype=float)
    if error == "none":
        for i, values in enumerate(values_per_file):
            medians[i] = float(np.median(values))
        return medians, None, None

    lows = np.empty(len(values_per_file), dtype=float)
    highs = np.empty(len(values_per_file), dtype=float)
    rng = rng or np.random.default_rng()
    for i, values in enumerate(values_per_file):
        med, lo, hi = bootstrap_median_ci(
            values,
            ci_percent=ci_percent,
            n_boot=n_boot,
            max_events=max_events,
            rng=rng,
        )
        medians[i] = med
        lows[i] = lo
        highs[i] = hi
    return medians, lows, highs


def median_channel(path: str, channel: str, *, channel_naming: str = "$PnS") -> float:
    """Median of one numeric channel over gated/unfiltered events (caller may gate upstream)."""
    _, values = read_channel(path, channel, channel_naming=channel_naming)
    return float(np.median(values))


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
