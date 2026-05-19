"""Plot integrated SSC vs side-detector NA for several bead diameters."""

from __future__ import annotations

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from collect_mie.common import ssc_half_angle_deg, signal_mode_value
from collect_mie.config import load_config
from collect_mie.config_schema import PlotSscVsNaConfig
from collect_mie.core import normalize_relative
from collect_mie.ssc_collection import integrate_ssc_from_config
from collect_mie.plot_format import fmt_deg, fmt_na, fmt_n, fmt_particle_n, format_ssc_rect_mask_note
from collect_mie.run_config import resolve_config_path, write_run_record


def _apply_normalize_na_sweep(
    stacked: np.ndarray,
    mode: str,
    d_ref_um: float,
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
        return stacked / ref_safe[np.newaxis, :], f"SSC / SSC (d={d_ref_um:g} µm)"
    if mode == "max":
        return (
            np.array([normalize_relative(row, mode="max") for row in stacked]),
            "Relative integrated SSC (each curve ÷ own max)",
        )
    return (
        np.array([normalize_relative(row, mode="first") for row in stacked]),
        "Relative integrated SSC (each curve ÷ own first NA)",
    )


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    path = config_path or resolve_config_path(
        argv if argv is not None else sys.argv[1:]
    )
    cfg = load_config(path, "plot-ssc-vs-na")
    assert isinstance(cfg, PlotSscVsNaConfig)

    na_hi = min(float(cfg.na_max), float(cfg.n_medium))
    if na_hi < cfg.na_max:
        warnings.warn(
            f"Capping NA sweep maximum to {na_hi:g} (na_max {cfg.na_max} > "
            f"n_medium {cfg.n_medium}); increase n_medium to reach higher NA.",
            stacklevel=1,
        )
    if na_hi <= cfg.na_min:
        raise SystemExit(
            f"Effective NA maximum ({na_hi}) must exceed na_min ({cfg.na_min}); "
            "check na_max and n_medium."
        )

    na_values = np.linspace(cfg.na_min, na_hi, cfg.n_na)
    wl_nm = cfg.wavelength_nm
    n_particle = complex(cfg.n_real, cfg.n_imag)
    smode = signal_mode_value(cfg.signal_mode)

    raw_rows: list[np.ndarray] = []
    for d_um in cfg.diameter_um_list:
        d_nm = d_um * 1000.0
        row = np.empty(na_values.shape, dtype=float)
        for i, na in enumerate(na_values):
            alpha_deg = ssc_half_angle_deg(na, cfg.n_medium)
            row[i] = integrate_ssc_from_config(
                n_particle,
                d_nm,
                wl_nm,
                cfg,
                alpha_deg,
                polarization=cfg.polarization,
                signal_mode=smode,
            )
        raw_rows.append(row)

    stacked = np.asarray(raw_rows)
    plot_y, y_label = _apply_normalize_na_sweep(
        stacked, cfg.normalize, cfg.diameter_um_list[0]
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, d_um in enumerate(cfg.diameter_um_list):
        ax.plot(na_values, plot_y[i], label=f"d={d_um:g} µm")

    ax.set_yscale("log")
    ax.set_xlabel("SSC numerical aperture")
    ax.set_ylabel(y_label)
    coll_note = format_ssc_rect_mask_note(
        cfg.ssc_mask_half_angle_x_deg,
        cfg.ssc_mask_half_angle_z_deg,
        prefix="",
    )
    if not coll_note:
        coll_note = "NA cone"
    ax.set_title(
        f"SSC vs NA  λ={wl_nm:g} nm (vacuum), n={fmt_particle_n(cfg.n_real, cfg.n_imag)}, "
        f"n_medium={fmt_n(cfg.n_medium)}\n"
        f"SSC center={fmt_deg(cfg.ssc_center_deg)}°, collection={coll_note},\n"
        f"polarization={cfg.polarization}, signal_mode={cfg.signal_mode}"
    )
    ax.legend(title="Diameter")
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
            command_name="plot-ssc-vs-na",
            config_path=path,
            resolved=cfg,
        )


if __name__ == "__main__":
    main()
