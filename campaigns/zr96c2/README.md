# zr96c2: A=96 isobar collisions at 200 GeV, campaign 2

Event-by-event viscous hydrodynamics for ⁹⁶Ru+⁹⁶Ru and ⁹⁶Zr+⁹⁶Zr collisions
at √s_NN = 200 GeV, on a 2000-point design over 20 model parameters (initial
state, nuclear shape, transport, freeze-out) plus a 360-point extension,
particlized with three δf models. The data are meant for Bayesian calibration against the STAR isobar
measurements and for nuclear-structure studies.

This directory is the *simulation package*: the description of the data plus
what software needs to read them. Analysis runs through two repositories:

- **soft-sector-data-curation** holds the curated experimental database
  (HEPData records with acceptances, centrality definitions and methods) and
  the model-independent pipeline that computes each observable with the
  record's exact cuts.
- **soft-sector-sim-adapters** holds `sim_adapters.iebe_music`, the reader
  for this output format.

`setup.sh` clones both at the versions this package was tested with.

## Contents

| file | |
|---|---|
| `README.md` | this description |
| `simulation.yaml` | manifest read by the pipeline: format, systems, variants, design, adapter, pinned software |
| `design_lhs2000_A96_c2.csv` | the design points: the 2000-point hypercube and the extension `design_2001`–`design_2360` (`label` = `design_<NNNN>` ↔ data directory `dp<NNNN>`) |
| `cross_sections_a96.csv` | per-design-point inelastic cross section and b_max (centrality anchor), for the simulated points dp 1–1000 and 2001–2360 |
| `setup.sh` | fetches the two repositories (`./setup.sh --venv` also builds a Python environment) |

The data are in `../merged/dp<NNNN>/` (see [Data format](#data-format)).

## Quick start

```bash
./setup.sh --venv                     # clones into ./software, builds ./software/venv
source software/venv/bin/activate
cd software/soft-sector-data-curation

# All default observables (manifest 'selection') for one design point and δf model:
python3 scripts/run_pipeline.py --sim-package ../.. --design-point 5 --variant CE \
    --type flow_vn particle_yields particle_spectra --reaction '^(Ru\+Ru|Zr\+Zr)$' \
    --energy 200 --output dp0005_CE.h5
```

For the whole campaign, use one pipeline process per design point and δf
model, with the manifest's default selection. Rerun the command later to
compute only new or re-merged points. Then stack the outputs into one table
(runs × observables, design parameters, aligned experimental points) for
emulation:

```bash
python3 scripts/run_campaign.py --sim-package ../.. --out-dir ~/zr96c2_runs --jobs 8
python3 scripts/collect_campaign.py ~/zr96c2_runs --output ~/zr96c2_table.h5
```

`--sim-package` takes this directory. If the data are not at `../merged`
(copied or mounted elsewhere), add `--data-root <path>`. If this directory is
read-only, run `SOFTWARE_DIR=<writable dir> ./setup.sh`; the pipeline then
finds the adapter next to its own checkout.

Each run writes an HDF5 file with one group per collision system. A group
holds `values`, `errors`, `covariance` (block jackknife over events) and
`labels`, plus `exp_<i>` groups with the matching experimental points. The
attrs record the package, adapter and analysis versions.

From Python:

```python
from analysis.sim_package import SimPackage          # soft-sector-data-curation
pkg = SimPackage("<this directory>")
qv = pkg.load(5, "CE")       # QVectorData: Qn[event, species, harmonic, pT]
pkg.anchor(5)                # {'b_max': ..., 'sigma_tot': ...}

from sim_adapters.iebe_music import open_simulation  # adapter only, plain numpy
arrays = open_simulation("../merged").load(5, "CE")
```

## Physics

| stage | setting |
|---|---|
| nuclei | A = 96 deformed Woods–Saxon: radius R, diffuseness a, β₂, γ, β₃, with event-by-event shape fluctuations (σ_plane, σ_β₃; `shape_distribution plane`) |
| initial state | TRENTo, σ_NN = 4.23 fm²; reduced thickness p, nucleon width w, multiplicity fluctuations σ_k, normalization |
| pre-equilibrium | free streaming to τ_fs, then the **full** T^μν (e, u^μ, π^μν, Π) handed to hydro |
| hydrodynamics | MUSIC, 2+1D boost-invariant, η/s(T) with a kink, ζ/s(T) peaked; τ_π factor 5 |
| particlization | iSS at T_freeze, oversampled; each surface particlized three times, with the same seeds: **Grad22** (22-moment), **CE** (Chapman–Enskog), **PTB** (positive-definite, Pratt–Torrieri–Bernhard) |
| afterburner | resonance decays only (no hadronic rescattering) |
| output | hadronic_afterburner_toolkit Q-vectors of 13 identified species |

The sampler has no proton number, so the A=96 nucleus is Ru or Zr only
through its shape parameters. The same design point serves both systems.

**Design.** A 2000-point maximin Latin hypercube in the unit cube (seed
8675309), mapped linearly onto the prior ranges. Columns:

| parameter | range | | parameter | range |
|---|---|---|---|---|
| `normalization` | 6 – 13 | | `bulk_viscosity_3_max` | 0.01 – 0.25 |
| `reduced-thickness` | −0.7 – 0.7 | | `bulk_viscosity_3_T_peak_in_GeV` | 0.12 – 0.30 |
| `nucleon-width` (fm) | 0.5 – 1.5 | | `bulk_viscosity_3_width_in_GeV` | 0.025 – 0.15 |
| `sigma_k` | 0.3 – 2.0 | | `T_freeze` (GeV) | 0.135 – 0.165 |
| `tau` (τ_fs, fm/c) | 0.3 – 2.0 | | `WS_radius` (fm) | 5.0 – 5.7 |
| `shear_viscosity_3_eta_over_s_at_kink` | 0.01 – 0.20 | | `WS_diffusiveness` (fm) | 0.45 – 0.60 |
| `..._low_T_slope_in_GeV` | −2.0 – 0.0 | | `beta_2` | 0.0 – 0.4 |
| `..._high_T_slope_in_GeV` | 0.0 – 2.0 | | `gamma` (deg) | 0 – 60 |
| `..._T_kink_in_GeV` | 0.13 – 0.30 | | `beta_3` | 0.0 – 0.4 |
| | | | `sigma_plane`, `sigma_beta_3` | 0 – 0.08 each |

Fixed: `shear_relax_time_factor` 5, `bulk_viscosity_3_lambda_asymm` 0.
The design is campaign 1's hypercube with only `normalization` remapped
(3–10 → 6–13), so `design_<k>` is the same normalized point in both
campaigns, and in the A=76 campaigns.

**Extension (design_2001–2360).** 360 points added after the first analysis
of design_0001–1000, as 180 pairs: the two points of a pair (consecutive
labels) share the 13 medium parameters and differ only in the 7
nuclear-shape parameters. design_2001–2240 are focused on the region the
first analysis favours, design_2241–2360 fill the space maximin; labels are
in priority order. Four ranges are wider than in the hypercube:
`nucleon-width` 0.35 – 1.5 fm, `tau` 0.15 – 2.0 fm/c, `WS_radius` 4.8 –
5.7 fm, `WS_diffusiveness` 0.40 – 0.66 fm. Events use the same event ids and
seeds as the original points (common random numbers: an event id has the
same draws at every design point).

**Events.** For each design point, impact parameters are drawn uniformly in
the disk b < b_max (about 2000 draws). Draws without a collision, or without
a fluid, have no event group; they are listed with `status` in `csv.tgz`.
The adapter appends them as empty events, so the loaded sample covers the
fraction f = π b_max² / σ_inel of the inelastic cross section. The pipeline
takes σ_inel from `cross_sections_a96.csv` and refuses a table whose b_max
differs from the campaign's.

## Data format

`../merged/dp<NNNN>/`:

- `dp<NNNN>.h5`: one group `spvn_results_<event_id>` per event with a fluid.
  - Hydro-level datasets: eccentricities and momentum anisotropy vs τ, ⟨pT⟩ estimators.
  - Sub-groups `deltaf_Grad22`, `deltaf_CE`, `deltaf_PTB`, each holding for the 13 species (π±, K±, p, p̄, Λ, Λ̄, Ξ∓, Ω∓, φ):
    - `particle_<pid>_vndata_diff_y_-30_30_boostinv.dat`: 100 pT bins × (mean pT, E dN/d³p, Re/Im Q₁…Q₅ per particle, count);
    - `particle_<pid>_vndata_y_-30_30_boostinv.dat`: pT-integrated table; its last row holds the number of iSS samples.
  - Attrs `deltaf_models`, `toolkit_pT_min` = 0, `toolkit_pT_max` = 5, `toolkit_npT` = 100: bins of 0.05 GeV.
  - Yields are boost-invariant over a rapidity width of 10.
- `csv.tgz`: drawn events (`event_id, b, b_max, b_over_bmax, mult, status, seed_add`).
- `merge.log`, `sources.txt`: merge bookkeeping.

The adapter docstring (`sim_adapters/iebe_music.py`) gives the exact
conversion to Q-vectors.

## Status and caveats

- **Live campaign.** Design points are merged as jobs return, and a merged
  file can be replaced by a larger one. dp0001–1000 and dp2001–2360 are
  simulated; design_1001–2000 are not. The pipeline output records which
  package and versions were used; rerun a design point when its files change.
- **Peripheral events at long τ_fs.** For τ_fs ≳ 1.3 fm/c, the most
  peripheral fluids live under 0.25 fm/c. Particles are then emitted at the
  hand-off with the raw free-streaming π^μν, which PTB applies in full. This
  produces upturns in PTB observables beyond about 60% centrality. Calibrate
  up to 60%, or treat the δf spread there as model uncertainty.
- The simulation configuration, its changes from campaign 1 and the
  validation of the chain (δf closure, reproduction of grid events,
  bookkeeping) are documented with the simulation code:
  [`CAMPAIGN2_HANDOFF.md`](https://github.com/mluzum/iEBE-MUSIC/blob/NME/CAMPAIGN2_HANDOFF.md) and
  [`design/A96_priors.md`](https://github.com/mluzum/iEBE-MUSIC/blob/NME/design/A96_priors.md) in the iEBE-MUSIC
  fork (branch `NME`).
