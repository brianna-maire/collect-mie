"""Pydantic models for YAML run configuration (replaces argparse schemas)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from collect_mie.defaults import (
    DEFAULT_FSC_CENTER_DEG,
    DEFAULT_FSC_MASK_HALF_ANGLE_Y_DEG,
    DEFAULT_FSC_MASK_HALF_ANGLE_Z_DEG,
    DEFAULT_FSC_NA_INNER,
    DEFAULT_FSC_NA_OUTER,
    DEFAULT_N_MEDIUM,
    DEFAULT_N_PARTICLE_REAL,
    DEFAULT_SSC_CENTER_DEG,
    DEFAULT_SSC_MASK_HALF_ANGLE_X_DEG,
    DEFAULT_SSC_MASK_HALF_ANGLE_Z_DEG,
    DEFAULT_SSC_NA,
    DEFAULT_WAVELENGTH_NM,
)

Polarization = Literal["unpolarized", "parallel", "perpendicular"]
SignalModeCli = Literal["absolute-cross-section", "phase-function"]
BandsChoice = Literal["both", "fsc", "ssc"]
NormalizeSimple = Literal["max", "first"]
CompareNormalize = Literal["max", "first", "least_squares"]
CompareDataSource = Literal["manifest", "table"]
MedianError = Literal["none", "bootstrap"]
ChannelGate = Literal["none", "log_decades"]
ChannelSummary = Literal["median", "peak_gated_median"]
PeakSelection = Literal["highest_prominence", "rightmost_prominent"]
NormalizeOverlay = Literal["none", "max", "first", "global-max", "ref-first"]
ChannelNaming = Literal["$PnS", "$PnN"]


def _parse_float_list(value: object) -> list[float]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        if not parts:
            raise ValueError("expected at least one value")
        return [float(x) for x in parts]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError("expected comma-separated string or list of numbers")


FloatList = Annotated[list[float], Field(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RunOutputFields(_Strict):
    output: str | None = None
    write_run_record: str | None = None


class MediumOptics(_Strict):
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    n_medium: float = DEFAULT_N_MEDIUM
    n_imag: float = 0.0
    polarization: Polarization = "unpolarized"
    signal_mode: SignalModeCli = "absolute-cross-section"

    @field_validator("wavelength_nm", "n_medium")
    @classmethod
    def positive_medium(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class ParticleOptics(_Strict):
    n_real: float = DEFAULT_N_PARTICLE_REAL


class FscBand(_Strict):
    fsc_center_deg: float = DEFAULT_FSC_CENTER_DEG
    fsc_na_outer: float = DEFAULT_FSC_NA_OUTER
    fsc_na_inner: float = DEFAULT_FSC_NA_INNER


class FscOptionalRectMask(_Strict):
    """
    When both mask half-angles are set in YAML (fsc: mask_half_angle_y/z_deg),
    FSC integration uses (outer NA cone) \\ lab rect bar centered on +x; na_inner is
    ignored. Bar limits are |arctan2(k_y, k_x)| and |arctan2(k_z, k_x)| in the same
    fixed lab frame as the SSC mask (which is centered on +y with mask_x / mask_z).
    """

    fsc_mask_half_angle_y_deg: float | None = None
    fsc_mask_half_angle_z_deg: float | None = None
    fsc_rect_mask_n_phi: int = Field(default=720, ge=8)

    @model_validator(mode="after")
    def check_mask_pair(self) -> FscOptionalRectMask:
        y, z = self.fsc_mask_half_angle_y_deg, self.fsc_mask_half_angle_z_deg
        if (y is None) ^ (z is None):
            raise ValueError(
                "set both fsc_mask_half_angle_y_deg and fsc_mask_half_angle_z_deg "
                "in fsc:, or omit both to use circular na_inner obscuration"
            )
        if y is not None and (y <= 0 or z <= 0):
            raise ValueError("mask half-angles must be positive when set")
        return self


class FscRectMaskRequired(_Strict):
    """Required mask defaults for plot-diameter-fsc-rect-mask comparison runs."""

    fsc_mask_half_angle_y_deg: float = DEFAULT_FSC_MASK_HALF_ANGLE_Y_DEG
    fsc_mask_half_angle_z_deg: float = DEFAULT_FSC_MASK_HALF_ANGLE_Z_DEG
    fsc_rect_mask_n_phi: int = Field(default=720, ge=8)


class FscBandMixin(FscBand, FscOptionalRectMask):
    """FSC band + optional rectangular mask subtraction (shared by collection helpers)."""


class SscBand(_Strict):
    ssc_center_deg: float = DEFAULT_SSC_CENTER_DEG
    ssc_na: float = DEFAULT_SSC_NA


class SscOptionalRectMask(_Strict):
    """
    When both mask half-angles are set in YAML (ssc: mask_half_angle_*_deg),
    SSC integration uses cone ∩ rectangular mask; otherwise cone only.
    """

    ssc_mask_half_angle_x_deg: float | None = None
    ssc_mask_half_angle_z_deg: float | None = None
    ssc_rect_mask_n_phi: int = Field(default=720, ge=8)

    @model_validator(mode="after")
    def check_mask_pair(self) -> SscOptionalRectMask:
        x, z = self.ssc_mask_half_angle_x_deg, self.ssc_mask_half_angle_z_deg
        if (x is None) ^ (z is None):
            raise ValueError(
                "set both ssc_mask_half_angle_x_deg and ssc_mask_half_angle_z_deg "
                "in ssc:, or omit both for cone-only collection"
            )
        if x is not None and (x <= 0 or z <= 0):
            raise ValueError("mask half-angles must be positive when set")
        return self


class SscRectMaskRequired(_Strict):
    """Required mask defaults for plot-diameter-ssc-rect-mask comparison runs."""

    ssc_mask_half_angle_x_deg: float = DEFAULT_SSC_MASK_HALF_ANGLE_X_DEG
    ssc_mask_half_angle_z_deg: float = DEFAULT_SSC_MASK_HALF_ANGLE_Z_DEG
    ssc_rect_mask_n_phi: int = Field(default=720, ge=8)


class SscBandMixin(SscBand, SscOptionalRectMask):
    """SSC band + optional rectangular mask (shared by collection helpers)."""


class DiameterSweep(_Strict):
    d_min_um: float = 0.04
    d_max_um: float = 0.40
    n_diameters: int = Field(default=120, ge=2)

    @model_validator(mode="after")
    def check_diameter_range(self) -> DiameterSweep:
        if self.d_min_um <= 0 or self.d_max_um <= self.d_min_um:
            raise ValueError("require 0 < d_min_um < d_max_um")
        return self


class PlotAngleConfig(MediumOptics, ParticleOptics, RunOutputFields):
    diameter_um: float
    theta_min_deg: float = 0.1
    theta_max_deg: float = 180.0
    n_points: int = Field(default=3600, ge=2)

    @field_validator("diameter_um")
    @classmethod
    def positive_diameter(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("diameter_um must be positive")
        return v


class PlotDiameterConfig(
    MediumOptics, ParticleOptics, FscBandMixin, SscBandMixin, DiameterSweep, RunOutputFields
):
    bands: BandsChoice = "both"
    normalize: NormalizeSimple = "max"


class PlotRefractiveIndexConfig(MediumOptics, SscBandMixin, DiameterSweep, RunOutputFields):
    n_real_list: FloatList
    normalize: NormalizeOverlay = "none"

    @field_validator("n_real_list", mode="before")
    @classmethod
    def coerce_n_real_list(cls, v: object) -> list[float]:
        return _parse_float_list(v)


class PlotSscVsNaConfig(MediumOptics, ParticleOptics, SscOptionalRectMask, RunOutputFields):
    ssc_center_deg: float = DEFAULT_SSC_CENTER_DEG
    na_min: float = 1.0
    na_max: float = 1.4
    n_na: int = Field(default=41, ge=2)
    diameter_um_list: FloatList = Field(default_factory=lambda: [0.5, 1.0, 3.0, 6.0])
    normalize: NormalizeOverlay = "none"

    @field_validator("diameter_um_list", mode="before")
    @classmethod
    def coerce_diameter_list(cls, v: object) -> list[float]:
        return _parse_float_list(v)

    @field_validator("diameter_um_list")
    @classmethod
    def positive_diameters(cls, v: list[float]) -> list[float]:
        if any(d <= 0 for d in v):
            raise ValueError("all diameters must be positive")
        return v

    @model_validator(mode="after")
    def check_na_sweep(self) -> PlotSscVsNaConfig:
        if self.na_min <= 0 or self.na_max <= self.na_min:
            raise ValueError("require 0 < na_min < na_max")
        return self


class PlotDiameterSscRectMaskConfig(
    MediumOptics, ParticleOptics, SscBand, SscRectMaskRequired, DiameterSweep, RunOutputFields
):
    normalize: NormalizeSimple = "max"


class PlotDiameterFscRectMaskConfig(
    MediumOptics, ParticleOptics, FscBand, FscRectMaskRequired, DiameterSweep, RunOutputFields
):
    normalize: NormalizeSimple = "max"


class CompareChannelBase(DiameterSweep, RunOutputFields):
    data_source: CompareDataSource = "manifest"
    manifest: str | None = None
    points_manifest: str | None = None
    channel_naming: ChannelNaming = "$PnS"
    normalize: CompareNormalize = "max"
    histogram_output: str | None = None
    histogram_bins: int = Field(default=50, gt=0)
    median_error: MedianError = "none"
    median_ci_percent: float = Field(default=95.0, gt=0, lt=100)
    median_bootstrap_n: int = Field(default=2000, ge=100)
    median_bootstrap_max_events: int = Field(default=20_000, ge=100)
    channel_summary: ChannelSummary = "peak_gated_median"
    median_gate: ChannelGate = "none"
    median_gate_log_decades: float = Field(default=0.5, gt=0)
    median_gate_min_events: int = Field(default=100, ge=1)
    peak_histogram_bins: int = Field(default=200, gt=0)
    peak_selection: PeakSelection = "rightmost_prominent"
    peak_prominence_fraction: float = Field(default=0.05, gt=0, le=1)
    peak_smooth_bins: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def check_data_source(self) -> CompareChannelBase:
        if self.data_source == "manifest":
            if not self.manifest:
                raise ValueError("data_source=manifest requires manifest")
        elif self.data_source == "table":
            if not self.points_manifest:
                raise ValueError("data_source=table requires points_manifest")
        return self

    @model_validator(mode="after")
    def check_least_squares_signal_mode(self) -> CompareChannelBase:
        if self.normalize == "least_squares" and self.signal_mode != "absolute-cross-section":
            raise ValueError(
                "normalize=least_squares requires mie.signal_mode=absolute-cross-section"
            )
        return self


class CompareSscConfig(MediumOptics, ParticleOptics, SscBandMixin, CompareChannelBase):
    ssc_channel: str = "SSC-A"


class CompareFscConfig(MediumOptics, ParticleOptics, FscBandMixin, CompareChannelBase):
    fsc_channel: str
