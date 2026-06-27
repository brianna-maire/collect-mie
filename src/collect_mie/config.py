"""Load and validate YAML configs into Pydantic models."""

from __future__ import annotations

from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from collect_mie.config_schema import (
    CompareFscConfig,
    CompareSscConfig,
    PlotAngleConfig,
    PlotDiameterConfig,
    PlotDiameterFscRectMaskConfig,
    PlotDiameterSscRectMaskConfig,
    PlotRefractiveIndexConfig,
    PlotSscVsNaConfig,
)

T = TypeVar("T", bound=BaseModel)

CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "plot-angle": PlotAngleConfig,
    "plot-diameter": PlotDiameterConfig,
    "plot-refractive-index": PlotRefractiveIndexConfig,
    "plot-ssc-vs-na": PlotSscVsNaConfig,
    "plot-diameter-ssc-rect-mask": PlotDiameterSscRectMaskConfig,
    "plot-diameter-fsc-rect-mask": PlotDiameterFscRectMaskConfig,
    "compare-ssc": CompareSscConfig,
    "compare-fsc": CompareFscConfig,
}


def load_config(config_path: str, command_name: str) -> BaseModel:
    """Merge YAML sections and validate into the model for ``command_name``."""
    model_cls = CONFIG_MODELS.get(command_name)
    if model_cls is None:
        valid = ", ".join(sorted(CONFIG_MODELS))
        raise SystemExit(f"Unknown command {command_name!r}. Valid: {valid}")
    raw = _load_yaml(config_path)
    flat = merge_config_dict(
        raw,
        command_name=command_name,
        include_fsc=command_name
        in ("plot-diameter", "compare-fsc", "plot-diameter-fsc-rect-mask"),
        include_ssc=command_name
        in (
            "plot-diameter",
            "plot-refractive-index",
            "compare-ssc",
            "plot-diameter-ssc-rect-mask",
        ),
    )
    try:
        return model_cls.model_validate(flat)
    except ValidationError as err:
        raise SystemExit(
            f"Invalid configuration in {config_path}:\n{err}"
        ) from err


def merge_config_dict(
    raw: dict[str, Any],
    *,
    command_name: str,
    include_fsc: bool,
    include_ssc: bool,
) -> dict[str, Any]:
    """
    Flatten nested YAML sections into field names matching config models.

    Merge order (later wins):
    common (deprecated) -> mie -> fsc/ssc -> command section(s) -> args -> run.args
    """
    out: dict[str, Any] = {}
    _merge(out, raw.get("common"))  # deprecated; use mie:
    _merge(out, raw.get("mie"))

    if include_fsc:
        _merge_prefixed(
            out,
            raw.get("fsc"),
            prefix="fsc_",
            aliases={
                "center_deg": "center_deg",
                "na_outer": "na_outer",
                "na_inner": "na_inner",
                "mask_half_angle_y_deg": "mask_half_angle_y_deg",
                "mask_half_angle_z_deg": "mask_half_angle_z_deg",
            },
        )
    if include_ssc:
        _merge_prefixed(
            out,
            raw.get("ssc"),
            prefix="ssc_",
            aliases={
                "center_deg": "center_deg",
                "na": "na",
                "mask_half_angle_x_deg": "mask_half_angle_x_deg",
                "mask_half_angle_z_deg": "mask_half_angle_z_deg",
            },
        )
    elif command_name == "plot-ssc-vs-na":
        _merge_ssc_na_sweep_section(out, raw.get("ssc"))

    cmd_us = command_name.replace("-", "_")
    if command_name == "plot-diameter-fsc-rect-mask":
        _merge_fsc_rect_mask_plot_section(out, raw.get(cmd_us))
        _merge_plot_diameter_sweep_fields(out, raw.get("plot_diameter"))
    else:
        _merge(out, raw.get(command_name))
        _merge(out, raw.get(cmd_us))
    _merge(out, raw.get("args"))

    run = raw.get("run")
    if isinstance(run, dict):
        _merge(out, run.get("args"))

    return out


def _merge(dst: dict[str, Any], src: Any) -> None:
    if isinstance(src, dict):
        dst.update(src)


def _merge_plot_diameter_sweep_fields(dst: dict[str, Any], src: Any) -> None:
    """Copy diameter-sweep keys from a ``plot_diameter:`` block."""
    if not isinstance(src, dict):
        return
    for key in ("d_min_um", "d_max_um", "n_diameters", "normalize"):
        if key in src:
            dst[key] = src[key]


def _merge_fsc_rect_mask_plot_section(dst: dict[str, Any], src: Any) -> None:
    if not isinstance(src, dict):
        return
    flat = {
        "center_deg": "fsc_center_deg",
        "fsc_center_deg": "fsc_center_deg",
        "na_outer": "fsc_na_outer",
        "fsc_na_outer": "fsc_na_outer",
        "na_inner": "fsc_na_inner",
        "fsc_na_inner": "fsc_na_inner",
        "mask_half_angle_y_deg": "fsc_mask_half_angle_y_deg",
        "mask_half_angle_z_deg": "fsc_mask_half_angle_z_deg",
        "rect_mask_n_phi": "fsc_rect_mask_n_phi",
    }
    for key, value in src.items():
        dest = flat.get(key, key if key.startswith("fsc_") else None)
        if dest is not None:
            dst[dest] = value
        elif key in ("d_min_um", "d_max_um", "n_diameters", "normalize", "output", "write_run_record"):
            dst[key] = value


def _merge_ssc_na_sweep_section(dst: dict[str, Any], src: Any) -> None:
    if not isinstance(src, dict):
        return
    flat = {
        "center_deg": "ssc_center_deg",
        "ssc_center_deg": "ssc_center_deg",
        "na_min": "na_min",
        "na_max": "na_max",
        "n_na": "n_na",
        "mask_half_angle_x_deg": "ssc_mask_half_angle_x_deg",
        "mask_half_angle_z_deg": "ssc_mask_half_angle_z_deg",
        "rect_mask_n_phi": "ssc_rect_mask_n_phi",
    }
    for key, value in src.items():
        dest = flat.get(key, key if key.startswith("ssc_") else None)
        if dest is not None:
            dst[dest] = value


def _merge_prefixed(
    dst: dict[str, Any],
    src: Any,
    *,
    prefix: str,
    aliases: dict[str, str],
) -> None:
    if not isinstance(src, dict):
        return
    for k, v in src.items():
        if k.startswith(prefix):
            dst[k] = v
            continue
        mapped = aliases.get(k)
        if mapped is not None:
            dst[prefix + mapped] = v
        else:
            dst[prefix + k] = v


def _resolve_config_path(path: str) -> Any:
    from pathlib import Path

    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        alt = Path("examples") / candidate.name
        if alt.is_file():
            return alt.resolve()
    raise SystemExit(f"Config file not found: {path}")


def _load_yaml(path: str) -> dict[str, Any]:
    text = _resolve_config_path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must contain a YAML mapping at top level: {path}")
    return data
