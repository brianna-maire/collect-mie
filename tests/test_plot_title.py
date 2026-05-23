"""Figure title lines: analysis aliases, optics, geometry, sweep params."""

from collect_mie.config import load_config
import matplotlib.pyplot as plt

from collect_mie.plot_title import (
    COMMAND_ALIASES,
    TitleContext,
    _FIGURE_TITLE_LINE_SPACING,
    _FIGURE_TITLE_LINE_SPACING_MULTI,
    _figure_title_band_height,
    analysis_alias,
    apply_figure_title,
    build_figure_title,
)
from collect_mie.config_schema import PlotDiameterConfig, PlotSscVsNaConfig


def test_command_aliases_cover_run_commands():
    assert analysis_alias("plot-diameter") == COMMAND_ALIASES["plot-diameter"]
    assert "Compare:" not in COMMAND_ALIASES["compare-fcs"]
    assert "Sweep:" not in COMMAND_ALIASES.values()


def test_plot_diameter_title_includes_geometry_not_sweep_prefix():
    cfg = PlotDiameterConfig(
        d_min_um=0.04,
        d_max_um=0.4,
        bands="ssc",
        normalize="max",
    )
    title = build_figure_title(
        "plot-diameter",
        cfg,
        TitleContext(uses_ssc=True, ssc_alpha=48.2),
    )
    lines = title.split("\n")
    assert lines[0] == "Analysis: Mie integrated scatter vs diameter"
    assert lines[1].startswith("Optics:")
    assert any(line.startswith("SSC:") for line in lines)
    assert lines[-1] == "d=0.04–0.4 µm, bands=ssc, normalize=max"
    assert "Sweep:" not in title
    assert "n_pts=" not in title


def test_plot_ssc_vs_na_title_without_fixed_ssc_na():
    cfg = PlotSscVsNaConfig(
        na_min=1.0,
        na_max=1.3,
        ssc_mask_half_angle_x_deg=67.5,
        ssc_mask_half_angle_z_deg=90.0,
    )
    title = build_figure_title("plot-ssc-vs-na", cfg, TitleContext(uses_ssc=True))
    lines = title.split("\n")
    ssc_line = next(line for line in lines if line.startswith("SSC:"))
    assert "NA=1.00" not in ssc_line
    assert "collection=NA cone ∩ rect mask" in ssc_line
    assert lines[-1].startswith("NA=1–1.3")


def test_figure_title_band_matches_line_count():
    assert _figure_title_band_height(1, _FIGURE_TITLE_LINE_SPACING, 9) < (
        _figure_title_band_height(6, _FIGURE_TITLE_LINE_SPACING, 9)
    )
    assert _figure_title_band_height(4, _FIGURE_TITLE_LINE_SPACING_MULTI, 10) < (
        _figure_title_band_height(4, _FIGURE_TITLE_LINE_SPACING, 9)
    )


def test_analysis_line_is_bold():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    title = "Analysis: Test plot\nOptics: λ=488 nm"
    apply_figure_title(fig, title, ax=ax)
    texts = [t for t in fig.texts if t.get_text().startswith("Analysis:")]
    assert len(texts) == 1
    assert texts[0].get_fontweight() == "bold"
    other = [t for t in fig.texts if t.get_text().startswith("Optics:")]
    assert other[0].get_fontweight() == "normal"
    plt.close(fig)


def test_figure_title_sits_above_axes_without_using_y_equals_one():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    apply_figure_title(fig, "line1\nline2\nline3", use_suptitle=True)
    n_lines = 3
    axes_top = 1.0 - _figure_title_band_height(
        n_lines, _FIGURE_TITLE_LINE_SPACING_MULTI, 9
    )
    assert ax.get_position().y1 <= axes_top + 0.01
    assert getattr(fig, "_suptitle", None) is None
    plt.close(fig)


def test_compare_fcs_title_from_example_config():
    cfg = load_config("examples/compare_fcs_run.example.yaml", "compare-fcs")
    title = build_figure_title(
        "compare-fcs",
        cfg,
        TitleContext(
            uses_fsc=True,
            uses_ssc=True,
            fsc_alpha_outer=12.3,
            fsc_alpha_inner=4.5,
            ssc_alpha=48.2,
            extra_lines=["SSC-A: R²=0.99, RMSE=1e3, scale=2.5"],
        ),
    )
    assert title.startswith("Analysis: FCS vs Mie model\n")
    assert "Compare:" not in title
    assert "Prediction sweep:" not in title
    assert "SSC-A: R²=0.99" in title
    assert any(line.startswith("FSC:") for line in title.split("\n"))
    assert any(line.startswith("SSC:") for line in title.split("\n"))
