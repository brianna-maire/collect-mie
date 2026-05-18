"""Overlay SSC vs bead diameter for multiple particle refractive indices."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from collect_mie.common import (
    resolve_ssc_band_deg,
    ssc_half_angle_deg,
    signal_mode_value,
)
from collect_mie.config import load_config
from collect_mie.config_schema import PlotRefractiveIndexConfig
from collect_mie.core import normalize_relative
from collect_mie.ssc_collection import (
    diameter_sweep_ssc_from_config,
    ssc_uses_rect_mask,
)
from collect_mie.run_config import resolve_config_path, write_run_record


def _apply_normalize_overlay(
    stacked: np.ndarray,
    mode: str,
    n_ref_display: float,
) -> tuple[np.ndarray, str]:
    eps = float(np.finfo(float).eps)
    if mode == "none":
        return stacked.copy(), "Integrated SSC (model units)"
    if mode == "global-max":
        mx = float(np.max(stacked))
        denom = mx if mx > 0 else 1.0
        return stacked / denom, "Normalized SSC (÷ global max over all curves)"
    if mode == "ref-first":
        ref = stacked[0].astype(float, copy=False)
        scale = max(float(np.max(np.abs(stacked))), eps)
        ref_safe = np.where(np.abs(ref) < eps * scale, eps * scale, ref)
        return stacked / ref_safe[np.newaxis, :], f"SSC / SSC (n={n_ref_display:g})"
    if mode == "max":
        return (
            np.array([normalize_relative(row, mode="max") for row in stacked]),
            "Relative integrated SSC (each curve ÷ own max)",
        )
    return (
        np.array([normalize_relative(row, mode="first") for row in stacked]),
        "Relative integrated SSC (each curve ÷ own first point)",
    )


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, "plot-refractive-index")
    assert isinstance(cfg, PlotRefractiveIndexConfig)

    ssc_alpha = ssc_half_angle_deg(cfg.ssc_na, cfg.n_medium)
    ssc_min, ssc_max = resolve_ssc_band_deg(
        ssc_na=cfg.ssc_na,
        ssc_center_deg=cfg.ssc_center_deg,
        n_medium=cfg.n_medium,
    )
    if ssc_min >= ssc_max:
        raise SystemExit("SSC band: min angle must be less than max angle")

    wl_nm = cfg.wavelength_nm
    diam_um = np.linspace(cfg.d_min_um, cfg.d_max_um, cfg.n_diameters)
    diam_nm = diam_um * 1000.0
    smode = signal_mode_value(cfg.signal_mode)

    raw_rows: list[np.ndarray] = []
    for n_real in cfg.n_real_list:
        n_particle = complex(n_real, cfg.n_imag)
        ssc_raw = diameter_sweep_ssc_from_config(
            n_particle,
            diam_nm,
            wl_nm,
            cfg,
            ssc_alpha,
            polarization=cfg.polarization,
            signal_mode=smode,
        )
        raw_rows.append(ssc_raw)

    stacked = np.asarray(raw_rows)
    plot_y, y_label = _apply_normalize_overlay(
        stacked, cfg.normalize, cfg.n_real_list[0]
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, n_real in enumerate(cfg.n_real_list):
        label = f"n={n_real:g}"
        if cfg.n_imag != 0:
            label += f"{cfg.n_imag:+g}j"
        ax.plot(diam_um, plot_y[i], label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Diameter (µm)")
    ax.set_ylabel(y_label)
    if ssc_uses_rect_mask(cfg.ssc_mask_half_angle_x_deg, cfg.ssc_mask_half_angle_z_deg):
        mask_note = (
            f", NA cone ∩ rect mask (mask_x={cfg.ssc_mask_half_angle_x_deg:g}°, "
            f"mask_z={cfg.ssc_mask_half_angle_z_deg:g}°)"
        )
    else:
        mask_note = ", NA cone"
    ax.set_title(
        f"SSC vs diameter λ={wl_nm:g} nm (vacuum), n_medium={cfg.n_medium:g}\n"
        f"SSC center={cfg.ssc_center_deg:g}°, alpha={ssc_alpha:.2f}°{mask_note}"
    )
    ax.legend(title="Particle index")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", axis="x", alpha=0.5, linestyle=":", linewidth=0.9)
    ax.grid(which="minor", axis="y", alpha=0.5, linestyle="-", linewidth=0.9)
    fig.tight_layout()

    if cfg.output:
        fig.savefig(cfg.output, dpi=150)
    else:
        plt.show()

    if cfg.write_run_record:
        write_run_record(
            cfg.write_run_record,
            command_name="plot-refractive-index",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
