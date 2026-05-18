# collect-mie

Helpers for **Mie scattering** from homogeneous spheres, aimed at flow-cytometry-style reasoning: scattered intensity vs polar angle, integrated scatter vs bead diameter for configurable forward/side collection cones, and optional comparison to `.fcs` channel medians.

## Background: Mie scattering theory

Classic Mie theory (Gustav Mie, 1908; equivalent formulations by Lorenz) gives the exact solution for a plane electromagnetic wave incident on a homogeneous, isotropic sphere embedded in a uniform dielectric medium. The scattered far field is expanded in vector spherical harmonics; for each polarization one obtains scattering amplitudes $S_1(\theta)$ and $S_2(\theta)$ as functions of polar scattering angle $\theta$ between the incident beam and the observation direction.

### What enters the model

- **Particle refractive index** (possibly complex $n + \mathrm{i}k$ for absorption): controls phase lag inside the bead and thus interference in the scattered wave.
- **Medium refractive index** $n_\mathrm{medium}$: sets wavelength in the fluid $\lambda_\mathrm{med} = \lambda_\mathrm{vac}/n_\mathrm{medium}$ and Snell-type physics inside miepython's formulation.
- **Size parameter** $x = \pi d / \lambda_\mathrm{med}$ for diameter $d$: compares bead size to wavelength in the medium. Small $x$ $\Rightarrow$ Rayleigh-like dipolar scattering; larger $x$ $\Rightarrow$ rich angular structure (rainbow-like features, deep minima).

### What this package computes

We use **[miepython](https://github.com/scottprahl/miepython)** to obtain scattered intensity per unit solid angle (effectively proportional to \(|S_i|^2\) with miepython’s chosen normalization).

By default, configs use `signal_mode: absolute-cross-section`, which scales the angular signal by geometric area $\pi(d/2)^2$ so integrated band outputs track size-dependent scatter magnitude. `signal_mode: phase-function` is also available for shape-only angular redistribution (diameter scaling suppressed by normalization).

#### Integrating over detector solid angle

For `plot-angle`, the model only samples $I(\theta)$ on a polar grid (no collection geometry).

For FSC/SSC-style outputs, we integrate the same angular intensity over a finite solid angle that stands in for light collected by a detector cone (or cone minus obscuration). Mie scattering is treated as azimuthally symmetric in $\theta$ and $\phi$ about the incident beam; the instrument geometry picks which $(\theta,\phi)$ directions contribute.

In general the collected signal is a scalar proportional to

$$
\int I(\theta)\,W(\theta)\,\sin\theta\,\mathrm{d}\theta,
$$

where $W(\theta)$ is the allowed azimuth width (radians) at polar angle $\theta$ for directions that hit the detector optics. Intensity $I$ already includes polarization averaging and `signal_mode` scaling from miepython.

**FSC (forward scatter)** — **`integrate_detector_annular_cone`** in `src/collect_mie/core.py`:

- Detector axis at polar angle `fsc_center_deg` (default **$0^\circ$**, on the laser axis).
- Outer and inner half-angles from `fsc_na_outer`and `fsc_na_inner` via $\alpha = \arcsin(\mathrm{NA}/n_\mathrm{medium})$.
- Collection is the outer right circular cone minus an inner obscuration cone (annulus in solid angle). At each $\theta$, $W(\theta) = W_\mathrm{out}(\theta) - W_\mathrm{in}(\theta)$, clipped to $[0,2\pi]$.
- For the default axis-aligned FSC geometry ($0^\circ$ center), this reduces to integrating over a polar band $\theta \in [\alpha_\mathrm{inner},\alpha_\mathrm{outer}]$ with full $2\pi$ azimuth.

**SSC (side scatter)** — **`integrate_detector_cone`**:

- Detector axis at `ssc_center_deg` (default **$90^\circ$**, perpendicular to the beam in the scattering plane).
- A single lens NA cone: half-angle `ssc_na` $\Rightarrow$ $\alpha = \arcsin(\mathrm{NA}/n_\mathrm{medium})$.
- At each $\theta$, only azimuths $\phi$ with $\cos\gamma \ge \cos\alpha$ are accepted, where $\gamma$ is the angle between the scattered direction and the detector axis (spherical law of cosines). That yields a $\theta$-dependent $W(\theta)$ rather than a simple polar band.

**SSC + rectangular cuvette mask** — **`integrate_detector_cone_rect_mask`**:

- Same lens cone as above, intersected with a symmetric rectangular acceptance in a fixed lab frame (no roll about the detector axis):
  - laser **$+x$**, detector / collection **$+y$**, vertical **$+z$**.
  - Scattered direction $\mathbf{k}=(k_x,k_y,k_z)$ from Mie $(\theta,\phi)$ with $k_x=\cos\theta$, $k_y=\sin\theta\cos\phi$, $k_z=\sin\theta\sin\phi$.
  - Mask: $|\arctan2(k_x,k_y)| \le$ `ssc_mask_half_angle_x_deg` and $|\arctan2(k_z,k_y)| \le$ `ssc_mask_half_angle_z_deg` (direct half-angle caps, not an NA-derived ellipse).
- At each $\theta$, $W(\theta)$ is estimated by uniform quadrature on $\phi\in[0,2\pi)$ (`ssc_rect_mask_n_phi`, default 720): count $\phi$ samples that lie in **both** the lens cone and the rectangle, times $\Delta\phi$.


NA and half-angle formulas are summarized under [Default Parameters](#default-parameters) below.

The following analyses can be selected with `run.command` in a YAML file:

| **`run.command`** | What it does |
|---------------|----------------|
| [`plot-angle`](#plot-angle) | Intensity vs $\theta$ for one bead size. |
| [`plot-diameter`](#plot-diameter) | FSC annular cone and/or SSC cone vs bead diameter. |
| [`plot-refractive-index`](#plot-refractive-index) | SSC vs diameter for several particle indices. |
| [`plot-ssc-vs-na`](#plot-ssc-vs-na) | Integrated SSC vs side-detector NA for several diameters. |
| [`plot-diameter-ssc-rect-mask`](#plot-diameter-ssc-rect-mask) | SSC cone only vs cone ∩ rectangular cuvette mask vs diameter. |
| [`compare-fcs`](#compare-fcs) | Overlay FCS medians from `.fcs` files on the same model at manifest diameters. |



### Relation to a real cytometer (limitations of model)

Forward scatter (FSC) and side scatter (SSC) signals depend on laser wavelength, optical power density, numerical aperture**, obscuration geometry, polarization, detector optics and flow velocity. This code only uses cone/annular-cone collection geometry around detector axes; it **does not** ray-trace an instrument. Routines here are **order-of-magnitude knobs** for exploring relative scatter magnitude and bead size trends. This does not perform a calibration.

Additional effects ignored here include multiple scattering, non-spherical beads, index dispersion with wavelength, saturation, and gain differences between acquisitions — hence `compare-fcs` normalizes experiment and model together when overlaying curves.

### References

- C. F. Bohren & D. R. Huffman, *Absorption and Scattering of Light by Small Particles* (Wiley).
- **[miepython docs](https://miepython.readthedocs.io/en/latest/#)** for normalization conventions and polarization definitions.

---

## Install

From the repository root:

```bash
pip install -e .
```

Dependencies: `numpy`, `matplotlib`, `miepython`, `fcsparser`, `PyYAML`, `pydantic`.

## Running an analysis

All runs are driven by a **YAML config file**. The shell accepts only a config file path:

```bash
collect-mie examples/plot_diameter_run.example.yaml
collect-mie --config examples/plot_diameter_run.example.yaml
python -m collect_mie examples/plot_diameter_run.example.yaml
```

The command specifying an analysis is read from `run.command` in the file. Every other parameter must appear in YAML (`mie:`, `fsc:`, `ssc:`, a command section, and/or `run.args:`). Values are validated by Pydantic models in `src/collect_mie/config_schema.py`; unknown keys are rejected.

`collect-mie --help` lists valid `run.command` values.

Run all example configs:

```bash
python examples/run_example_configs.py
```

### Minimal config definition

```yaml
run:
  command: plot-diameter # required
  args:
    output: path/to/figure.png # optional; omit to show interactive plot
    write_run_record: path/to/record.yaml # optional

mie:
  wavelength_nm: 488
  n_medium: 1.3374
  n_real: 1.602
  # ...

plot_diameter: # section name = command with underscores
  bands: both
  d_min_um: 0.04
```

**Merge order** (later wins): `mie` → `fsc` / `ssc` → command section (`plot_diameter:` or `plot-diameter:`) → top-level `args:` → `run.args:`.

In `fsc:` / `ssc:`, use short keys (`center_deg`, `na`, `na_outer`, …); they are mapped to flat names (`fsc_center_deg`, `ssc_na`, …) before validation.

## Default Parameters

These match `src/collect_mie/defaults.py` and `config_schema.py` unless overridden in YAML.

| Field | Default |
|-------|---------|
| `wavelength_nm` | 488 |
| `n_medium` | 1.3374 |
| `n_real` | 1.6020 |
| `n_imag` | unset |
| `polarization` | unpolarized |
| `signal_mode` | absolute-cross-section |
| `fsc_center_deg` | 0 |
| `fsc_na_outer` | 0.34 |
| `fsc_na_inner` | 0.23 |
| `ssc_center_deg` | 90 |
| `ssc_na` | 1.29 |
| `ssc_mask_half_angle_x_deg` | unset (cone only; set both mask angles in `ssc:` to enable mask) |
| `ssc_mask_half_angle_z_deg` | unset |
| `ssc_rect_mask_n_phi` | 720 (when mask is active) |

**Geometry constraints:** each NA in $\alpha = \arcsin(\mathrm{NA}/n_\mathrm{medium})$ must satisfy $\mathrm{NA} \leq n_\mathrm{medium}$. Require `fsc_na_inner` < `fsc_na_outer`.

Particle index conventions match `miepython`: vacuum wavelength, absolute $n_\mathrm{particle}$ passed to `miepython.intensities`.

Cone half-angle from NA (single cone):

$$
\alpha = \arcsin\left(\frac{\mathrm{NA}}{n_\mathrm{medium}}\right), \qquad
\Omega_\mathrm{cone}=2\pi\left(1-\cos\alpha\right).
$$

FSC annulus (outer cone minus inner obscuration):

$$
\Omega_\mathrm{annulus}=2\pi\left(\cos\alpha_\mathrm{inner}-\cos\alpha_\mathrm{outer}\right).
$$

For axis-aligned FSC (center $0^\circ$), this annular cone reduces to
$$\theta\in[\alpha_\mathrm{inner},\alpha_\mathrm{outer}]$$
with
$$\alpha=\arcsin(\mathrm{NA}/n_\mathrm{medium}).$$

---

## Configuration fields

Field names below are the flat keys after merging YAML sections (what Pydantic validates). Authoritative definitions and validators live in `src/collect_mie/config_schema.py`.

### Run output: `run.args:`

| Field | Meaning |
|-------|---------|
| `output` | Save PNG path. Omit to open an interactive window. |
| `write_run_record` | After the run, write resolved config + timestamp YAML. |

---

### Mie model: `mie:`

Base parameters for the homogeneous-sphere Mie calculation (**[miepython](https://github.com/scottprahl/miepython)**): wavelength, medium and particle indices, polarization, and signal scaling (detector geometry lives in `fsc:` / `ssc:`).

| Field | Type / choices | Meaning |
|-------|----------------|---------|
| `wavelength_nm` | float | Vacuum wavelength (nm); must be $> 0$. |
| `n_medium` | float | Fluid refractive index. |
| `n_real` | float | Real part of particle refractive index. Not used by `plot-refractive-index` (use `n_real_list` there). |
| `n_imag` | float | Imaginary part of particle index (absorption). |
| `polarization` | `unpolarized`, `parallel`, `perpendicular` | miepython parallel / perpendicular / average (see miepython docs). |
| `signal_mode` | `absolute-cross-section`, `phase-function` | `phase-function`: `norm='albedo'`, shape only. `absolute-cross-section`: `norm='qsca'` then multiply by $\pi(d/2)^2$ for size comparisons. |


### FSC: `fsc:` → `fsc_*` (`plot-diameter`, `compare-fcs`)

| YAML key | Flat field | Meaning |
|----------|------------|---------|
| `center_deg` | `fsc_center_deg` | FSC detector axis $\theta$ (deg). |
| `na_outer` | `fsc_na_outer` | Outer NA → $\alpha_\mathrm{outer}$. |
| `na_inner` | `fsc_na_inner` | Inner obscuration NA; must be &lt; `na_outer`. |

### SSC: `ssc:` → `ssc_*` (most side-scatter commands)

| YAML key | Flat field | Meaning |
|----------|------------|---------|
| `center_deg` | `ssc_center_deg` | SSC detector axis $\theta$ (deg). |
| `na` | `ssc_na` | Side NA; cone half-angle $\alpha=\arcsin(\mathrm{NA}/n_\mathrm{medium})$. |
| `mask_half_angle_x_deg` | `ssc_mask_half_angle_x_deg` | Optional. With `mask_half_angle_z_deg`, SSC uses cone ∩ rectangular mask; omit both for cone only. |
| `mask_half_angle_z_deg` | `ssc_mask_half_angle_z_deg` | Optional. Partner to `mask_half_angle_x_deg`. |
| `rect_mask_n_phi` | `ssc_rect_mask_n_phi` | Azimuth samples when mask is active (default 720). |


Note: `plot-diameter-ssc-rect-mask` always plots cone vs masked on one figure (mask defaults apply). Other commands plot a single SSC curve: masked if both mask angles are in `ssc:`, otherwise cone-only.

---
## Analysis-Specific Fields

### `plot-angle`

Plots scattered intensity (1/sr) vs polar angle $\theta$ for a single bead diameter. This is the angular “fingerprint” of Mie scattering from one sphere: forward peak, side lobes, and minima, on a log $y$-axis. Use it to inspect how `signal_mode`, `polarization`, and refractive index affect the phase function before integrating over a detector cone. No FSC/SSC collection geometry is applied.

Requires `diameter_um`. Uses `mie:` only (no FSC/SSC sections).

| Field | Default | Meaning |
|-------|---------|---------|
| `diameter_um` | *(required)* | Sphere diameter (µm). |
| `theta_min_deg` | 0.1 | Minimum $\theta$ on plot (deg). |
| `theta_max_deg` | 180 | Maximum $\theta$ (deg). |
| `n_points` | 3600 | Samples along $\theta$. |

---

### `plot-diameter`

Sweeps bead diameter and plots relative integrated scatter for forward and/or side collection. FSC uses an annular cone (outer NA minus inner obscuration) on the forward axis; SSC integrates over a lens NA cone about the side axis (or cone ∩ rectangular mask** if both `mask_half_angle_*_deg` are set in `ssc:`). Choose `bands` to plot FSC only, SSC only, or both. Curves are normalized per trace (`max` or `first`) for shape-vs-size comparisons.

Uses `mie`, `fsc`, `ssc`, and `plot_diameter:` (or `plot-diameter:`).

| Field | Default | Meaning |
|-------|---------|---------|
| `d_min_um` | 0.04 | Minimum diameter (µm); require $0 < \texttt{d\_min\_um} < \texttt{d\_max\_um}$. |
| `d_max_um` | 0.40 | Maximum diameter (µm). |
| `n_diameters` | 120 | Linear diameter samples. |
| `bands` | `both` | `both`, `fsc`, or `ssc` — which cones to compute. |
| `normalize` | `max` | `max` or `first` per plotted trace. |

---

### `plot-refractive-index`

Overlays SSC vs bead diameter for several particle refractive indices (`n_real_list`). Each curve is the same SSC cone (or masked collection if configured in `ssc:`), so you can compare how index affects collected side scatter across size. `normalize` controls whether curves are comparable in amplitude (`none`, `global-max`, `ref-first`) or normalized independently (`max`, `first`). Does not use a single `n_real` in `mie:` — indices come only from the list.

Uses `mie` (no `n_real`), `ssc`, and command section. `n_imag` applies to every index in the list.

| Field | Default | Meaning |
|-------|---------|---------|
| `n_real_list` | *(required)* | List of real indices, e.g. `[1.59, 1.602, 1.62]` or a comma-separated string. |
| `d_min_um`, `d_max_um`, `n_diameters` | same as `plot-diameter` | Diameter sweep. |
| `normalize` | `none` | `none`, `global-max`, `ref-first`, `max`, or `first` (see `plot_refractive_index.py` docstring behavior). |

---

### `plot-ssc-vs-na`

Plots integrated SSC vs side-detector numerical aperture*with one curve per bead diameter. At each NA sample, the model builds a cone half-angle from that NA and integrates Mie scatter into it (optionally intersected with the rectangular cuvette mask if both mask half-angles are in `ssc:`). The x-axis is NA, not diameter — useful for exploring how collection aperture affects SSC at fixed bead sizes. NA values above `n_medium` are capped with a warning.

Uses `mie` and `plot_ssc_vs_na:` and/or `ssc:` for `na_min`, `na_max`, `n_na`, `center_deg`, and optional mask keys.

| Field | Default | Meaning |
|-------|---------|---------|
| `ssc_center_deg` | 90 | SSC axis (deg); NA is the x-axis. |
| `na_min` | 1.0 | Minimum NA on sweep. |
| `na_max` | 1.4 | Maximum NA; capped at `n_medium` with a warning. |
| `n_na` | 41 | Number of NA samples (linear). |
| `diameter_um_list` | `[0.5, 1, 3, 6]` | One curve per diameter (µm). |
| `normalize` | `none` | Same choices as `plot-refractive-index`; `ref-first` uses the first diameter. |

---

### `plot-diameter-ssc-rect-mask`

Dedicated comparison of two SSC collection models vs bead diameter on one figure: (1) lens NA cone only, and (2) the same cone intersected with a symmetric rectangular lab mask (cuvette-style limits on $|arctan2(k_x,k_y)|$ and $|arctan2(k_z,k_y)|$). Both curves share the same `ssc_na`, center angle, and bead optics; only the solid angle differs. Use this command when you want to see the mask effect explicitly; other commands show one collection mode (masked or cone-only) per run. Mask half-angles default to 68° / 74° if omitted from YAML.

Uses `mie`, `ssc`, and command section.

| Field | Default | Meaning |
|-------|---------|---------|
| `ssc_mask_half_angle_x_deg` | 68 | Mask on $\|\arctan2(k_x, k_y)\|$ (deg). |
| `ssc_mask_half_angle_z_deg` | 74 | Mask on $\|\arctan2(k_z, k_y)\|$ (deg). |
| `ssc_rect_mask_n_phi` | 720 | Azimuth samples for mask ∩ cone integration. |
| `d_min_um`, `d_max_um`, `n_diameters`, `normalize` | same as `plot-diameter` | |

---

### `compare-fcs`

Compares experimental forward/side scatter to the Mie model at the same nominal bead diameters. Reads a manifest of diameter + `.fcs` file paths, plots median FSC and SSC from each file against model FSC (annular cone) and SSC (cone or masked) evaluated only at those diameters. Normalization is applied separately to each trace so instrument gain and model scale can be aligned for shape comparison. Two stacked panels: FSC on top, SSC on bottom.

Uses `mie`, `fsc`, `ssc`, and `compare_fcs:`.

| Field | Default | Meaning |
|-------|---------|---------|
| `manifest` | *(required)* | Text file: nominal diameter (µm) and `.fcs` path per line. |
| `fsc_channel` | `FSC-A` | FSC column for median. |
| `ssc_channel` | `SSC-A` | SSC column for median. |
| `channel_naming` | `$PnS` | `$PnS` or `$PnN` for FCS keyword lookup. |
| `normalize` | `max` | `max` or `first`; applied separately to each experimental and model trace. |

Model curves use manifest diameters only (not a dense sweep).

**Manifest format:** one bead per line — diameter (µm) and path (whitespace- or comma-separated). Lines starting with `#` are ignored.

---

## Example configs

Templates under `examples/`:

- `plot_angle_run.example.yaml`
- `plot_diameter_run.example.yaml`
- `refractive_index_run.example.yaml`
- `plot_ssc_vs_na_run.example.yaml`
- `plot_diameter_ssc_rect_mask_run.example.yaml`
- `compare_fcs_run.example.yaml`

```bash
collect-mie examples/plot_diameter_run.example.yaml
python examples/run_example_configs.py # all *_run.example.yaml
```

`compare-fcs` is skipped by the batch runner unless `examples/compare_manifest.txt` exists (see `compare_manifest.example.txt`).
