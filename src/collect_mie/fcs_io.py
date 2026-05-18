"""Read Flow Cytometry Standard (.fcs) files and summarize channels for model comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

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


def median_channel(path: str, channel: str, *, channel_naming: str = "$PnS") -> float:
    """Median of one numeric channel over gated/unfiltered events (caller may gate upstream)."""
    _, data = fcsparser.parse(path, channel_naming=channel_naming)
    col = find_channel_column(data, channel)
    values = np.asarray(data[col], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite events in {path!r} column {col!r}")
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
            rows.append((float(parts[0]), parts[1]))
    if not rows:
        raise ValueError(f"No rows in manifest {path!r}")
    return rows
