"""Compare FSC annular NA vs outer cone \\ rectangular obscuration bar vs bead diameter."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from collect_mie.common import fsc_half_angles_deg, signal_mode_value
from collect_mie.config import load_config
from collect_mie.config_schema import PlotDiameterFscRectMaskConfig
from collect_mie.core import (
    diameter_sweep_detector_annular_cone,
    diameter_sweep_detector_cone_minus_fsc_rect_bar,
    normalize_relative,
)
from collect_mie.plot_title import TitleContext, apply_figure_title, build_figure_title
from collect_mie.run_config import resolve_config_path, save_figure, write_run_record


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, "plot-diameter-fsc-rect-mask")
    assert isinstance(cfg, PlotDiameterFscRectMaskConfig)

    fsc_alpha_outer, fsc_alpha_inner = fsc_half_angles_deg(
        cfg.fsc_na_outer, cfg.fsc_na_inner, cfg.n_medium
    )
    wl_nm = cfg.wavelength_nm
    diam_um = np.linspace(cfg.d_min_um, cfg.d_max_um, cfg.n_diameters)
    diam_nm = diam_um * 1000.0
    n_particle = complex(cfg.n_real, cfg.n_imag)
    sweep_kw = dict(
        polarization=cfg.polarization,
        signal_mode=signal_mode_value(cfg.signal_mode),
    )

    annular_raw = diameter_sweep_detector_annular_cone(
        n_particle,
        diam_nm,
        wl_nm,
        cfg.n_medium,
        cfg.fsc_center_deg,
        fsc_alpha_outer,
        fsc_alpha_inner,
        **sweep_kw,
    )
    bar_raw = diameter_sweep_detector_cone_minus_fsc_rect_bar(
        n_particle,
        diam_nm,
        wl_nm,
        cfg.n_medium,
        cfg.fsc_center_deg,
        fsc_alpha_outer,
        cfg.fsc_mask_half_angle_y_deg,
        cfg.fsc_mask_half_angle_z_deg,
        n_phi=cfg.fsc_rect_mask_n_phi,
        **sweep_kw,
    )

    annular_rel = normalize_relative(annular_raw, mode=cfg.normalize)
    bar_rel = normalize_relative(bar_raw, mode=cfg.normalize)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(diam_um, annular_rel, label="annular NA")
    ax.plot(diam_um, bar_rel, label="outer cone \\ rect bar")

    ax.set_yscale("log")
    ax.set_xlabel("Diameter (µm)")
    ax.set_ylabel("Relative integrated FSC")
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", axis="x", alpha=0.5, linestyle=":", linewidth=0.9)
    ax.grid(which="minor", axis="y", alpha=0.5, linestyle="-", linewidth=0.9)
    apply_figure_title(
        fig,
        build_figure_title(
            "plot-diameter-fsc-rect-mask",
            cfg,
            TitleContext(
                uses_fsc=True,
                fsc_alpha_outer=fsc_alpha_outer,
                fsc_alpha_inner=fsc_alpha_inner,
            ),
        ),
        ax=ax,
    )

    if cfg.output:
        save_figure(fig, cfg.output, dpi=150)
    else:
        plt.show()

    if cfg.write_run_record:
        write_run_record(
            cfg.write_run_record,
            command_name="plot-diameter-fsc-rect-mask",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
