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


def test_ssc_indices_ref_first_divides_by_first_index_row():
    stacked = np.array([[2.0, 4.0], [4.0, 4.0]])
    y, _ = _apply_normalize_overlay(stacked, "ref-first", 1.59)
    np.testing.assert_allclose(y[0], [1.0, 1.0])
    np.testing.assert_allclose(y[1], [2.0, 1.0])
