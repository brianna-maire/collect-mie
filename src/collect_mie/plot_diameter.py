"""Plot relative integrated scatter vs bead diameter for FSC and/or SSC cones."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from collect_mie.common import (
    fsc_half_angles_deg,
    resolve_ssc_band_deg,
    ssc_half_angle_deg,
    signal_mode_value,
)
from collect_mie.config import load_config
from collect_mie.config_schema import PlotDiameterConfig
from collect_mie.core import diameter_sweep_detector_annular_cone, normalize_relative
from collect_mie.ssc_collection import (
    diameter_sweep_ssc_from_config,
    ssc_uses_rect_mask,
)
from collect_mie.run_config import resolve_config_path, write_run_record


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, "plot-diameter")
    assert isinstance(cfg, PlotDiameterConfig)

    want_fsc = cfg.bands in ("both", "fsc")
    want_ssc = cfg.bands in ("both", "ssc")

    fsc_alpha_outer = fsc_alpha_inner = 0.0
    if want_fsc:
        fsc_alpha_outer, fsc_alpha_inner = fsc_half_angles_deg(
            cfg.fsc_na_outer, cfg.fsc_na_inner, cfg.n_medium
        )

    ssc_alpha = 0.0
    if want_ssc:
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
    n_particle = complex(cfg.n_real, cfg.n_imag)
    smode = signal_mode_value(cfg.signal_mode)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if want_fsc:
        fsc_raw = diameter_sweep_detector_annular_cone(
            n_particle,
            diam_nm,
            wl_nm,
            cfg.n_medium,
            cfg.fsc_center_deg,
            fsc_alpha_outer,
            fsc_alpha_inner,
            polarization=cfg.polarization,
            signal_mode=smode,
        )
        fsc_rel = normalize_relative(fsc_raw, mode=cfg.normalize)
        ax.plot(
            diam_um,
            fsc_rel,
            label=(
                f"FSC annular cone center={cfg.fsc_center_deg:g}°, "
                f"alpha_out={fsc_alpha_outer:.2f}°, alpha_in={fsc_alpha_inner:.2f}°"
            ),
        )

    if want_ssc:
        ssc_raw = diameter_sweep_ssc_from_config(
            n_particle,
            diam_nm,
            wl_nm,
            cfg,
            ssc_alpha,
            polarization=cfg.polarization,
            signal_mode=smode,
        )
        ssc_rel = normalize_relative(ssc_raw, mode=cfg.normalize)
        if ssc_uses_rect_mask(
            cfg.ssc_mask_half_angle_x_deg, cfg.ssc_mask_half_angle_z_deg
        ):
            mask_note = (
                f", rect mask (mask_x={cfg.ssc_mask_half_angle_x_deg:g}°, "
                f"mask_z={cfg.ssc_mask_half_angle_z_deg:g}°)"
            )
        else:
            mask_note = ""
        ax.plot(
            diam_um,
            ssc_rel,
            label=(
                f"SSC center={cfg.ssc_center_deg:g}°, alpha={ssc_alpha:.2f}°{mask_note}"
            ),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Diameter (µm)")
    if cfg.bands == "fsc":
        ax.set_ylabel("Relative integrated FSC")
    elif cfg.bands == "ssc":
        ax.set_ylabel("Relative integrated SSC")
    else:
        ax.set_ylabel("Relative integrated scatter")

    ax.set_title(
        f"Mie Model λ={wl_nm:g} nm (vacuum), n={cfg.n_real:g}, n_medium={cfg.n_medium:g}, "
        f"bands={cfg.bands}\n"
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
            command_name="plot-diameter",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
