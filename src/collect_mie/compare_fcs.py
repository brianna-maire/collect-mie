"""Compare experimental .fcs medians to Mie model curves at manifest diameters."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from collect_mie.common import (
    fsc_half_angles_deg,
    resolve_ssc_band_deg,
    ssc_half_angle_deg,
    signal_mode_value,
)
from collect_mie.config import load_config
from collect_mie.config_schema import CompareFcsConfig
from collect_mie.core import diameter_sweep_detector_annular_cone, normalize_relative
from collect_mie.ssc_collection import diameter_sweep_ssc_from_config
from collect_mie.fcs_io import load_manifest_rows, median_channel
from collect_mie.run_config import resolve_config_path, write_run_record


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

    fsc_med = np.array(
        [
            median_channel(p, cfg.fsc_channel, channel_naming=cfg.channel_naming)
            for p in paths
        ]
    )
    ssc_med = np.array(
        [
            median_channel(p, cfg.ssc_channel, channel_naming=cfg.channel_naming)
            for p in paths
        ]
    )

    fsc_exp = normalize_relative(fsc_med, mode=cfg.normalize)
    ssc_exp = normalize_relative(ssc_med, mode=cfg.normalize)

    fsc_alpha_outer, fsc_alpha_inner = fsc_half_angles_deg(
        cfg.fsc_na_outer, cfg.fsc_na_inner, cfg.n_medium
    )
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

    fsc_model_raw = diameter_sweep_detector_annular_cone(
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
    ssc_model_raw = diameter_sweep_ssc_from_config(
        n_particle,
        diam_nm,
        wl_nm,
        cfg,
        ssc_alpha,
        polarization=cfg.polarization,
        signal_mode=smode,
    )

    fsc_model = normalize_relative(fsc_model_raw, mode=cfg.normalize)
    ssc_model = normalize_relative(ssc_model_raw, mode=cfg.normalize)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax0.plot(diam_exp_um, fsc_model, color="C0", label="Mie FSC band")
    ax0.scatter(diam_exp_um, fsc_exp, color="C0", marker="o", zorder=5, label="FCS median")
    ax0.set_ylabel("Relative FSC")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    ax1.plot(diam_exp_um, ssc_model, color="C1", label="Mie SSC band")
    ax1.scatter(diam_exp_um, ssc_exp, color="C1", marker="s", zorder=5, label="FCS median")
    ax1.set_xlabel("Diameter (µm)")
    ax1.set_ylabel("Relative SSC")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        f"λ={wl_nm:g} nm vacuum  n={cfg.n_real:g}  n_medium={cfg.n_medium:g}  "
        f"FSC(center={cfg.fsc_center_deg:g}°, NA_out={cfg.fsc_na_outer:g}, NA_in={cfg.fsc_na_inner:g})  "
        f"SSC(center={cfg.ssc_center_deg:g}°, NA={cfg.ssc_na:g}) -> [{ssc_min:g},{ssc_max:g}]°"
    )
    fig.tight_layout()

    if cfg.output:
        fig.savefig(cfg.output, dpi=150)
    else:
        plt.show()

    if cfg.write_run_record:
        write_run_record(
            cfg.write_run_record,
            command_name="compare-fcs",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
