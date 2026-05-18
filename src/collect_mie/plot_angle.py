"""Plot Mie scattered intensity vs polar angle."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from collect_mie.common import signal_mode_value
from collect_mie.config import load_config
from collect_mie.config_schema import PlotAngleConfig
from collect_mie.core import angular_intensity_curve
from collect_mie.run_config import resolve_config_path, write_run_record


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, "plot-angle")
    assert isinstance(cfg, PlotAngleConfig)

    wl_nm = cfg.wavelength_nm
    d_nm = cfg.diameter_um * 1000.0
    n_particle = complex(cfg.n_real, cfg.n_imag)

    theta_deg, intensity = angular_intensity_curve(
        n_particle,
        d_nm,
        wl_nm,
        cfg.n_medium,
        theta_min_deg=cfg.theta_min_deg,
        theta_max_deg=cfg.theta_max_deg,
        n_points=cfg.n_points,
        polarization=cfg.polarization,
        signal_mode=signal_mode_value(cfg.signal_mode),
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(theta_deg, intensity, lw=1.2)
    ax.set_xlabel(r"Polar scattering angle $\theta$ (deg)")
    ax.set_ylabel(r"Intensity (1/sr)")
    ax.set_title(
        f"Mie scattering  λ={wl_nm:g} nm vacuum  d={cfg.diameter_um:g} µm  "
        f"n={cfg.n_real:g}{'' if cfg.n_imag == 0 else f'{cfg.n_imag:+g}'}  "
        f"n_medium={cfg.n_medium:g}"
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if cfg.output:
        fig.savefig(cfg.output, dpi=150)
    else:
        plt.show()

    if cfg.write_run_record:
        write_run_record(
            cfg.write_run_record,
            command_name="plot-angle",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
