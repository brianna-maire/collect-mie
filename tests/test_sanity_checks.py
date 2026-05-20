import math

import numpy as np
import pytest

from collect_mie.common import fsc_half_angles_deg, ssc_half_angle_deg
from collect_mie.plot_refractive_index import _apply_normalize_overlay
from collect_mie.config import merge_config_dict
from collect_mie.config_schema import PlotDiameterSscRectMaskConfig
from collect_mie.run_config import load_run_command, resolve_config_path
from pydantic import ValidationError
from collect_mie.core import (
    integrate_detector_annular_cone,
    integrate_detector_cone,
    integrate_detector_cone_rect_mask,
    integrate_polar_band,
)


def test_ssc_cone_solid_angle_from_na():
    """Geometry check: NA -> half-angle -> cone solid angle."""
    ssc_na = 1.29
    n_medium = 1.33
    alpha_deg = ssc_half_angle_deg(ssc_na, n_medium)
    alpha_rad = math.radians(alpha_deg)
    omega = 2.0 * math.pi * (1.0 - math.cos(alpha_rad))

    assert abs(omega - 4.753828330125695) < 1e-9


def test_ssc_400nm_exceeds_40nm_with_cone_integration():
    """Regression test for the original issue: larger bead should scatter more."""
    n_particle = 1.59 + 0j
    wavelength_nm = 488.0
    n_medium = 1.33
    ssc_center_deg = 90.0
    ssc_na = 1.29
    alpha_deg = ssc_half_angle_deg(ssc_na, n_medium)

    ssc_40 = integrate_detector_cone(
        n_particle,
        40.0,
        wavelength_nm,
        n_medium,
        ssc_center_deg,
        alpha_deg,
        n_points=2500,
    )
    ssc_400 = integrate_detector_cone(
        n_particle,
        400.0,
        wavelength_nm,
        n_medium,
        ssc_center_deg,
        alpha_deg,
        n_points=2500,
    )

    assert ssc_400 > ssc_40


def test_cone_collection_is_smaller_than_axisymmetric_band():
    """
    For nearly isotropic scattering, cone/band ratio should match solid-angle ratio.

    This catches accidental reversion to full-azimuth band integration for SSC.
    """
    n_particle = 1.59 + 0j
    wavelength_nm = 488.0
    n_medium = 1.33
    ssc_center_deg = 90.0
    ssc_na = 1.29
    alpha_deg = ssc_half_angle_deg(ssc_na, n_medium)

    # Use tiny particle for near-Rayleigh angular isotropy.
    diameter_nm = 5.0
    cone = integrate_detector_cone(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        ssc_center_deg,
        alpha_deg,
        n_points=2500,
    )
    band = integrate_polar_band(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        ssc_center_deg - alpha_deg,
        ssc_center_deg + alpha_deg,
        n_points=2500,
    )

    alpha_rad = math.radians(alpha_deg)
    expected = (2.0 * math.pi * (1.0 - math.cos(alpha_rad))) / (
        4.0 * math.pi * math.sin(alpha_rad)
    )
    observed = cone / band

    # Allow small numerical/anisotropy differences.
    assert abs(observed - expected) < 0.03


def test_resolve_config_path_accepts_positional_and_flag():
    assert resolve_config_path(["foo.yaml"]) == "foo.yaml"
    assert resolve_config_path(["--config", "bar.yaml"]) == "bar.yaml"
    assert resolve_config_path(["--config=bar.yaml"]) == "bar.yaml"


def test_resolve_config_path_rejects_extra_arguments():
    import pytest

    with pytest.raises(SystemExit):
        resolve_config_path(["a.yaml", "--wavelength-nm", "405"])


def test_load_run_command_from_example_yaml(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "run:\n  command: plot-angle\n  args:\n    output: out.png\n",
        encoding="utf-8",
    )
    assert load_run_command(str(cfg)) == "plot-angle"


def test_ssc_rect_mask_wide_approaches_pure_cone():
    """Very loose rectangular mask should recover the NA cone integral."""
    n_particle = 1.59 + 0j
    wavelength_nm = 488.0
    n_medium = 1.33
    ssc_center_deg = 90.0
    ssc_na = 1.29
    alpha_deg = ssc_half_angle_deg(ssc_na, n_medium)
    diameter_nm = 40.0

    cone = integrate_detector_cone(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        ssc_center_deg,
        alpha_deg,
        n_points=2000,
    )
    masked = integrate_detector_cone_rect_mask(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        ssc_center_deg,
        alpha_deg,
        mask_half_angle_x_deg=89.5,
        mask_half_angle_z_deg=89.5,
        n_points=2000,
        n_phi=1440,
    )
    assert abs(masked - cone) / max(cone, 1e-30) < 0.02


def test_ssc_rect_mask_tight_is_smaller_than_cone():
    n_particle = 1.59 + 0j
    wavelength_nm = 488.0
    n_medium = 1.33
    ssc_center_deg = 90.0
    ssc_na = 1.29
    alpha_deg = ssc_half_angle_deg(ssc_na, n_medium)
    diameter_nm = 40.0

    cone = integrate_detector_cone(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        ssc_center_deg,
        alpha_deg,
        n_points=2000,
    )
    masked = integrate_detector_cone_rect_mask(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        ssc_center_deg,
        alpha_deg,
        mask_half_angle_x_deg=5.0,
        mask_half_angle_z_deg=5.0,
        n_points=2000,
        n_phi=1440,
    )
    assert masked < 0.5 * cone


def test_plot_diameter_ssc_rect_mask_yaml_merges_ssc_mask_keys():
    raw = {
        "mie": {"wavelength_nm": 405, "n_real": 1.602},
        "ssc": {
            "na": 1.2,
            "mask_half_angle_x_deg": 60.0,
            "mask_half_angle_z_deg": 70.0,
        },
        "plot_diameter_ssc_rect_mask": {"ssc_rect_mask_n_phi": 360},
    }
    flat = merge_config_dict(
        raw,
        command_name="plot-diameter-ssc-rect-mask",
        include_fsc=False,
        include_ssc=True,
    )
    cfg = PlotDiameterSscRectMaskConfig.model_validate(flat)
    assert cfg.wavelength_nm == 405
    assert cfg.ssc_na == 1.2
    assert cfg.ssc_mask_half_angle_x_deg == 60.0
    assert cfg.ssc_mask_half_angle_z_deg == 70.0
    assert cfg.ssc_rect_mask_n_phi == 360


def test_plot_diameter_without_mask_uses_cone_integration():
    from collect_mie.config_schema import PlotDiameterConfig
    from collect_mie.ssc_collection import diameter_sweep_ssc_from_config

    flat = {
        "wavelength_nm": 488.0,
        "n_medium": 1.33,
        "n_real": 1.59,
        "ssc_center_deg": 90.0,
        "ssc_na": 1.29,
        "d_min_um": 0.04,
        "d_max_um": 0.08,
        "n_diameters": 5,
    }
    cfg = PlotDiameterConfig.model_validate(flat)
    n_particle = 1.59 + 0j
    diam_nm = np.array([40.0, 80.0])
    alpha = ssc_half_angle_deg(cfg.ssc_na, cfg.n_medium)
    cone = diameter_sweep_ssc_from_config(
        n_particle, diam_nm, 488.0, cfg, alpha,
        polarization="unpolarized", signal_mode="absolute_cross_section",
    )
    from collect_mie.core import diameter_sweep_detector_cone

    expected = diameter_sweep_detector_cone(
        n_particle, diam_nm, 488.0, cfg.n_medium, cfg.ssc_center_deg, alpha,
    )
    np.testing.assert_allclose(cone, expected)


def test_plot_diameter_with_mask_uses_rect_integration():
    from collect_mie.config_schema import PlotDiameterConfig
    from collect_mie.ssc_collection import diameter_sweep_ssc_from_config

    flat = {
        "wavelength_nm": 488.0,
        "n_medium": 1.33,
        "n_real": 1.59,
        "ssc_center_deg": 90.0,
        "ssc_na": 1.29,
        "ssc_mask_half_angle_x_deg": 5.0,
        "ssc_mask_half_angle_z_deg": 5.0,
        "d_min_um": 0.04,
        "d_max_um": 0.08,
        "n_diameters": 5,
        "ssc_rect_mask_n_phi": 360,
    }
    cfg = PlotDiameterConfig.model_validate(flat)
    n_particle = 1.59 + 0j
    diam_nm = np.array([40.0])
    alpha = ssc_half_angle_deg(cfg.ssc_na, cfg.n_medium)
    masked = diameter_sweep_ssc_from_config(
        n_particle, diam_nm, 488.0, cfg, alpha,
        polarization="unpolarized", signal_mode="absolute_cross_section",
    )
    from collect_mie.core import diameter_sweep_detector_cone_rect_mask

    expected = diameter_sweep_detector_cone_rect_mask(
        n_particle, diam_nm, 488.0, cfg.n_medium, cfg.ssc_center_deg, alpha,
        5.0, 5.0, n_phi=360,
    )
    np.testing.assert_allclose(masked, expected)


def test_ssc_mask_requires_both_half_angles():
    from collect_mie.config_schema import PlotDiameterConfig

    with pytest.raises(ValidationError):
        PlotDiameterConfig.model_validate(
            {
                "wavelength_nm": 488,
                "n_real": 1.59,
                "ssc_na": 1.29,
                "ssc_mask_half_angle_x_deg": 68.0,
            }
        )


def test_unknown_config_key_rejected():
    raw = {
        "mie": {"wavelength_nm": 488, "n_real": 1.602},
        "plot_angle": {"diameter_um": 1.0, "not_a_real_key": 1},
    }
    flat = merge_config_dict(
        raw, command_name="plot-angle", include_fsc=False, include_ssc=False
    )
    with pytest.raises(ValidationError):
        from collect_mie.config_schema import PlotAngleConfig

        PlotAngleConfig.model_validate(flat)


def test_fsc_annular_cone_matches_forward_band_for_axis_aligned_geometry():
    """
    FSC axis at 0° with inner/outer half-angles should match simple forward band.

    This sanity check ensures annular cone integration behaves as expected in the
    common axis-aligned FSC geometry.
    """
    n_particle = 1.59 + 0j
    wavelength_nm = 488.0
    n_medium = 1.33
    alpha_out, alpha_in = fsc_half_angles_deg(0.34, 0.23, n_medium)

    diameter_nm = 40.0
    annular = integrate_detector_annular_cone(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        detector_center_deg=0.0,
        outer_half_angle_deg=alpha_out,
        inner_half_angle_deg=alpha_in,
        n_points=2500,
    )
    band = integrate_polar_band(
        n_particle,
        diameter_nm,
        wavelength_nm,
        n_medium,
        theta_min_deg=alpha_in,
        theta_max_deg=alpha_out,
        n_points=2500,
    )
    assert abs(annular - band) / max(abs(band), 1e-30) < 0.01


def test_ssc_indices_global_max_scales_to_one():
    stacked = np.array([[1.0, 2.0], [4.0, 2.0]])
    y, _ = _apply_normalize_overlay(stacked, "global-max", 0.0)
    assert abs(float(np.max(y)) - 1.0) < 1e-9
    np.testing.assert_allclose(y[1, 0], 1.0)


def test_common_section_deprecated_alias_for_mie():
    from collect_mie.config_schema import PlotAngleConfig

    raw = {
        "common": {"wavelength_nm": 532, "n_real": 1.5},
        "plot_angle": {"diameter_um": 1.0},
    }
    flat = merge_config_dict(
        raw, command_name="plot-angle", include_fsc=False, include_ssc=False
    )
    cfg = PlotAngleConfig.model_validate(flat)
    assert cfg.wavelength_nm == 532
    assert cfg.n_real == 1.5


def test_mie_section_overrides_common():
    from collect_mie.config_schema import PlotAngleConfig

    raw = {
        "common": {"wavelength_nm": 532, "n_real": 1.4},
        "mie": {"wavelength_nm": 488, "n_real": 1.602},
        "plot_angle": {"diameter_um": 1.0},
    }
    flat = merge_config_dict(
        raw, command_name="plot-angle", include_fsc=False, include_ssc=False
    )
    cfg = PlotAngleConfig.model_validate(flat)
    assert cfg.wavelength_nm == 488
    assert cfg.n_real == 1.602


def test_plot_ssc_vs_na_yaml_merges_plot_section_and_ssc_block():
    from collect_mie.config_schema import PlotSscVsNaConfig

    raw = {
        "mie": {"wavelength_nm": 405, "n_real": 1.602},
        "ssc": {"na_min": 1.0, "na_max": 1.25, "n_na": 10},
        "plot_ssc_vs_na": {"diameter_um_list": [0.5, 1.0]},
    }
    flat = merge_config_dict(
        raw,
        command_name="plot-ssc-vs-na",
        include_fsc=False,
        include_ssc=False,
    )
    cfg = PlotSscVsNaConfig.model_validate(flat)
    assert cfg.wavelength_nm == 405
    assert cfg.na_max == 1.25
    assert cfg.diameter_um_list == [0.5, 1.0]


def test_plot_format_helpers():
    from collect_mie.plot_format import (
        fmt_deg,
        fmt_na,
        fmt_n,
        fmt_particle_n,
        format_ssc_rect_mask_note,
    )

    assert fmt_na(1.29) == "1.29"
    assert fmt_deg(68.456) == "68.5"
    assert fmt_n(1.602) == "1.6020"
    assert fmt_particle_n(1.602, 0.0) == "1.6020"
    assert fmt_particle_n(1.602, -0.001) == "1.6020-0.0010j"
    assert format_ssc_rect_mask_note(None, 75.0) == ""
    assert format_ssc_rect_mask_note(68.0, 75.0) == (
        ", NA cone ∩ rect mask (mask_x=68.0°, mask_z=75.0°)"
    )
    assert format_ssc_rect_mask_note(68.0, 75.0, prefix="") == (
        "NA cone ∩ rect mask (mask_x=68.0°, mask_z=75.0°)"
    )


def test_resolve_ssc_histogram_output_derives_from_primary():
    from collect_mie.compare_fcs import resolve_ssc_histogram_output

    assert (
        resolve_ssc_histogram_output("out/compare_fcs.png", None)
        == "out/compare_fcs_ssc_histograms.png"
    )
    assert (
        resolve_ssc_histogram_output("out/compare_fcs.png", "out/custom.png")
        == "out/custom.png"
    )
    assert resolve_ssc_histogram_output(None, None) is None


def test_fit_metrics_through_origin():
    from collect_mie.compare_fcs import fit_metrics

    observed = np.array([2.0, 4.0, 6.0])
    fitted = np.array([2.0, 4.0, 6.0])
    r2, rmse = fit_metrics(observed, fitted)
    assert r2 == 1.0
    assert rmse == 0.0

    fitted2 = np.array([1.0, 2.0, 3.0])
    r2, rmse = fit_metrics(observed, fitted2)
    assert r2 < 1.0
    assert rmse > 0.0


def test_least_squares_scale():
    from collect_mie.compare_fcs import least_squares_scale

    data = np.array([2.0, 4.0, 6.0])
    model = np.array([1.0, 2.0, 3.0])
    assert least_squares_scale(data, model) == 2.0


def test_prepare_compare_trace_least_squares():
    from collect_mie.compare_fcs import _prepare_compare_trace

    med = np.array([10.0, 20.0])
    model = np.array([1.0, 3.0])
    exp_y, model_y, yerr, scale = _prepare_compare_trace(med, None, None, model, "least_squares")
    assert scale == 7.0
    np.testing.assert_allclose(exp_y, med)
    np.testing.assert_allclose(model_y, model * 7.0)
    assert yerr is None


def test_compare_fcs_rejects_least_squares_with_phase_function():
    from collect_mie.config_schema import CompareFcsConfig

    with pytest.raises(ValidationError):
        CompareFcsConfig.model_validate(
            {
                "manifest": "m.txt",
                "normalize": "least_squares",
                "signal_mode": "phase-function",
            }
        )


def test_gate_log_decades_keeps_center_band():
    from collect_mie.fcs_io import gate_log_decades

    values = np.logspace(0, 4, 500)  # 1 .. 10000
    gated, lo, hi = gate_log_decades(values, half_decades=0.5, min_events=10)
    assert lo < np.median(values) < hi
    assert np.all(gated >= lo)
    assert np.all(gated <= hi)
    assert gated.size < values.size


def test_apply_channel_gate_none_passthrough():
    from collect_mie.fcs_io import apply_channel_gate

    vals = [np.array([1.0, 2.0])]
    out, bounds, peaks = apply_channel_gate(vals, "none", half_decades=0.5, min_events=1)
    assert bounds == [None]
    assert peaks == [None]
    np.testing.assert_array_equal(out[0], vals[0])


def test_log_histogram_peak_center_finds_bright_mode():
    from collect_mie.fcs_io import log_histogram_peak_center

    rng = np.random.default_rng(0)
    background = rng.lognormal(mean=2.0, sigma=0.4, size=8000)
    cells = rng.lognormal(mean=4.5, sigma=0.15, size=1200)
    values = np.concatenate([background, cells])
    peak = log_histogram_peak_center(
        values,
        bins=120,
        smooth_bins=5,
        prominence_fraction=0.05,
        selection="rightmost_prominent",
    )
    assert peak > float(np.median(values))
    assert abs(np.log10(peak) - 4.5) < 0.6


def test_channel_summary_peak_gated_median_above_background():
    from collect_mie.fcs_io import channel_summary_and_bounds

    rng = np.random.default_rng(1)
    background = rng.lognormal(mean=2.0, sigma=0.35, size=6000)
    cells = rng.lognormal(mean=4.2, sigma=0.12, size=1500)
    values = np.concatenate([background, cells])
    med, _, _, peaks, bounds = channel_summary_and_bounds(
        [values],
        "peak_gated_median",
        "none",
        "none",
        half_decades=0.35,
        min_events=50,
        peak_bins=100,
    )
    assert peaks[0] is not None
    assert bounds[0] is not None
    assert float(med[0]) > float(np.median(values))


def test_gate_log_decades_accepts_explicit_center():
    from collect_mie.fcs_io import gate_log_decades

    values = np.logspace(0, 4, 500)
    center = float(np.percentile(values, 90))
    gated, lo, hi = gate_log_decades(
        values, half_decades=0.25, min_events=10, center=center
    )
    assert lo < center < hi
    assert np.median(gated) > float(np.median(values)) * 0.5


def test_bootstrap_median_ci_brackets_median():
    from collect_mie.fcs_io import bootstrap_median_ci

    rng = np.random.default_rng(0)
    values = rng.normal(10.0, 2.0, size=500)
    med, lo, hi = bootstrap_median_ci(
        values, ci_percent=95.0, n_boot=500, max_events=500, rng=rng
    )
    assert lo <= med <= hi
    assert lo < hi


def test_channel_median_and_bounds_bootstrap():
    from collect_mie.fcs_io import channel_median_and_bounds

    rng = np.random.default_rng(1)
    vals = [rng.normal(5.0, 1.0, 200)]
    med, lo, hi = channel_median_and_bounds(
        vals, "bootstrap", ci_percent=90.0, n_boot=400, max_events=200, rng=rng
    )
    assert lo[0] <= med[0] <= hi[0]
    assert lo[0] < hi[0]


def test_normalize_median_bounds_matches_normalize_relative():
    from collect_mie.compare_fcs import _normalize_median_bounds

    med = np.array([10.0, 20.0, 40.0])
    lo = np.array([8.0, 15.0, 30.0])
    hi = np.array([12.0, 25.0, 50.0])
    y, yerr = _normalize_median_bounds(med, lo, hi, "max")
    np.testing.assert_allclose(y, [0.25, 0.5, 1.0])
    assert yerr is not None
    assert yerr.shape == (2, 3)


def test_compare_fcs_config_allows_omitted_fsc_channel():
    from collect_mie.config_schema import CompareFcsConfig

    cfg = CompareFcsConfig.model_validate(
        {
            "manifest": "examples/compare_manifest.txt",
            "ssc_channel": "SSC-A",
        }
    )
    assert cfg.fsc_channel is None


def test_compare_fcs_config_accepts_diameter_sweep():
    from collect_mie.config_schema import CompareFcsConfig

    cfg = CompareFcsConfig.model_validate(
        {
            "manifest": "m.txt",
            "d_min_um": 0.1,
            "d_max_um": 0.5,
            "n_diameters": 50,
        }
    )
    assert cfg.d_max_um == 0.5


def test_load_manifest_rows_joins_path_with_spaces(tmp_path):
    from collect_mie.fcs_io import load_manifest_rows

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("0.151 fcs/NA129_full/151 nm.fcs\n")
    rows = load_manifest_rows(str(manifest))
    assert rows == [(0.151, "fcs/NA129_full/151 nm.fcs")]


def test_ssc_indices_ref_first_divides_by_first_index_row():
    stacked = np.array([[2.0, 4.0], [4.0, 4.0]])
    y, _ = _apply_normalize_overlay(stacked, "ref-first", 1.59)
    np.testing.assert_allclose(y[0], [1.0, 1.0])
    np.testing.assert_allclose(y[1], [2.0, 1.0])


def test_ensure_parent_dir_creates_nested_directories(tmp_path):
    from collect_mie.run_config import ensure_parent_dir

    target = tmp_path / "deep" / "nested" / "out.png"
    assert ensure_parent_dir(target) == target
    assert (tmp_path / "deep" / "nested").is_dir()
    assert not target.exists()


def test_save_figure_creates_parent_directories(tmp_path):
    import matplotlib.pyplot as plt

    from collect_mie.run_config import save_figure

    out = tmp_path / "plots" / "angle.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    save_figure(fig, out, dpi=50)
    plt.close(fig)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_write_run_record_creates_parent_directories(tmp_path):
    from collect_mie.config_schema import PlotAngleConfig
    from collect_mie.run_config import write_run_record

    record = tmp_path / "records" / "run.yaml"
    cfg = PlotAngleConfig(diameter_um=1.0)
    write_run_record(
        str(record),
        command_name="plot-angle",
        config_path="examples/plot_angle_run.example.yaml",
        resolved=cfg,
    )
    assert record.is_file()
    text = record.read_text(encoding="utf-8")
    assert "plot-angle" in text
    assert "diameter_um" in text
