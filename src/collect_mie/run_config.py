"""YAML dispatch and run-record helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.figure
import yaml
from pydantic import BaseModel

from collect_mie.config import CONFIG_MODELS, _load_yaml, load_config

COMMAND_TARGETS: dict[str, str] = {
    "plot-angle": "collect_mie.plot_angle:main",
    "plot-diameter": "collect_mie.plot_diameter:main",
    "plot-refractive-index": "collect_mie.plot_refractive_index:main",
    "plot-ssc-vs-na": "collect_mie.plot_ssc_vs_na:main",
    "plot-diameter-ssc-rect-mask": "collect_mie.plot_diameter_ssc_rect_mask:main",
    "plot-diameter-fsc-rect-mask": "collect_mie.plot_diameter_fsc_rect_mask:main",
    "compare-fcs": "collect_mie.compare_fcs:main",
}


def resolve_config_path(argv: list[str]) -> str:
    """
    Parse argv that contains only a config file path.

    Accepts: CONFIG.yaml | --config CONFIG.yaml | --config=CONFIG.yaml
    """
    if not argv:
        raise SystemExit(
            "Usage: collect-mie CONFIG.yaml\n"
            "       collect-mie --config CONFIG.yaml"
        )
    if len(argv) == 1:
        arg = argv[0]
        if arg in ("-h", "--help"):
            raise SystemExit(
                "Usage: collect-mie CONFIG.yaml\n"
                "       collect-mie --config CONFIG.yaml\n\n"
                "All run parameters come from the YAML file (see examples/*_run.example.yaml)."
            )
        if arg.startswith("--config="):
            path = arg.partition("=")[2]
            if path:
                return path
        if not arg.startswith("-"):
            return arg
    if len(argv) == 2 and argv[0] in ("--config", "-c") and not argv[1].startswith("-"):
        return argv[1]
    raise SystemExit(
        "Only a YAML config path is accepted on the command line. "
        "Set all other options in the config file."
    )


def load_run_command(config_path: str) -> str:
    """Return run.command from a config file."""
    raw = _load_yaml(config_path)
    run = raw.get("run")
    if not isinstance(run, dict):
        raise SystemExit(f"Config {config_path} must include a run: section with command.")
    cmd = run.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        raise SystemExit(f"Config {config_path} must set run.command (e.g. plot-diameter).")
    cmd = cmd.strip()
    if cmd not in CONFIG_MODELS:
        valid = ", ".join(sorted(CONFIG_MODELS))
        raise SystemExit(f"Unknown run.command {cmd!r}. Valid commands: {valid}")
    return cmd


def dispatch_config(config_path: str) -> None:
    """Load run.command from config and invoke the matching plot main."""
    command = load_run_command(config_path)
    target = COMMAND_TARGETS.get(command)
    if target is None:
        valid = ", ".join(sorted(COMMAND_TARGETS))
        raise SystemExit(f"Unknown run.command {command!r}. Valid commands: {valid}")

    mod_name, _, func_name = target.partition(":")
    mod = __import__(mod_name, fromlist=["_"])
    fn = getattr(mod, func_name)
    fn(config_path=config_path)


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directories for a file path when they do not exist."""
    out = Path(path)
    parent = out.parent
    if parent != out:
        parent.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(
    fig: matplotlib.figure.Figure,
    path: str | Path,
    *,
    dpi: int = 150,
    **kwargs: Any,
) -> Path:
    """Save a matplotlib figure, creating parent directories first."""
    out = ensure_parent_dir(path)
    fig.savefig(out, dpi=dpi, **kwargs)
    return out


def write_run_record(
    path: str,
    *,
    command_name: str,
    config_path: str,
    resolved: BaseModel,
) -> None:
    """Persist run metadata and resolved configuration for reproducibility."""
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_name,
        "config_path": config_path,
        "resolved_config": resolved.model_dump(mode="python"),
    }
    out = ensure_parent_dir(path)
    out.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
