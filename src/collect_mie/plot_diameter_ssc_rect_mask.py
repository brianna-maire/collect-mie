"""Compare SSC lens cone vs cone ∩ rectangular lab mask vs bead diameter."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from collect_mie.common import ssc_half_angle_deg, signal_mode_value
from collect_mie.config import load_config
from collect_mie.config_schema import PlotDiameterSscRectMaskConfig
from collect_mie.core import (
    diameter_sweep_detector_cone,
    diameter_sweep_detector_cone_rect_mask,
    normalize_relative,
)
from collect_mie.run_config import resolve_config_path, write_run_record


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, "plot-diameter-ssc-rect-mask")
    assert isinstance(cfg, PlotDiameterSscRectMaskConfig)

    ssc_alpha = ssc_half_angle_deg(cfg.ssc_na, cfg.n_medium)
    wl_nm = cfg.wavelength_nm
    diam_um = np.linspace(cfg.d_min_um, cfg.d_max_um, cfg.n_diameters)
    diam_nm = diam_um * 1000.0
    n_particle = complex(cfg.n_real, cfg.n_imag)
    sweep_kw = dict(
        polarization=cfg.polarization,
        signal_mode=signal_mode_value(cfg.signal_mode),
    )

    cone_raw = diameter_sweep_detector_cone(
        n_particle,
        diam_nm,
        wl_nm,
        cfg.n_medium,
        cfg.ssc_center_deg,
        ssc_alpha,
        **sweep_kw,
    )
    masked_raw = diameter_sweep_detector_cone_rect_mask(
        n_particle,
        diam_nm,
        wl_nm,
        cfg.n_medium,
        cfg.ssc_center_deg,
        ssc_alpha,
        cfg.ssc_mask_half_angle_x_deg,
        cfg.ssc_mask_half_angle_z_deg,
        n_phi=cfg.ssc_rect_mask_n_phi,
        **sweep_kw,
    )

    cone_rel = normalize_relative(cone_raw, mode=cfg.normalize)
    masked_rel = normalize_relative(masked_raw, mode=cfg.normalize)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        diam_um,
        cone_rel,
        label=f"SSC cone only (center={cfg.ssc_center_deg:g}°, alpha={ssc_alpha:.2f}°)",
    )
    ax.plot(
        diam_um,
        masked_rel,
        label=(
            f"SSC cone ∩ rect mask (alpha={ssc_alpha:.2f}°, "
            f"mask_x={cfg.ssc_mask_half_angle_x_deg:g}°, "
            f"mask_z={cfg.ssc_mask_half_angle_z_deg:g}°)"
        ),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Diameter (µm)")
    ax.set_ylabel("Relative integrated SSC")
    ax.set_title(
        f"Mie SSC vs diameter λ={wl_nm:g} nm (vacuum), n={cfg.n_real:g}, "
        f"n_medium={cfg.n_medium:g}\n"
        f"polarization={cfg.polarization}, signal_mode={cfg.signal_mode}"
    )
    ax.legend()
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
            command_name="plot-diameter-ssc-rect-mask",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
