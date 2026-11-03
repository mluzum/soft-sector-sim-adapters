# Observables from JETSCAPE QnVector output

This guide is for computing experimental observables from the output of the
JETSCAPE QnVector writer (`final_output_*.dat`, one hydro event per file).
Each run handles one collision system at one design point. It compares the
simulation with every curated measurement of that system that you select,
and it reads the event files as they are, with no conversion step.

Two repositories do the work:

| repository | role |
|---|---|
| [soft-sector-data-curation](https://github.com/mluzum/soft-sector-data-curation) | the database of curated measurements (HEPData tables with their centrality definitions, acceptances and methods) and the analysis code that computes each of them from a simulation |
| [soft-sector-sim-adapters](https://github.com/mluzum/soft-sector-sim-adapters) | readers for simulation formats (here `sim_adapters/jetscape_qnvector.py`) and the one-command wrapper `tools/qnvector_observables.py` |

## 1. Setup

You need Python 3.10 or newer. Clone both repositories into the same parent
directory, because the wrapper looks for the analysis repository next to its
own checkout:

```bash
git clone https://github.com/mluzum/soft-sector-data-curation.git
git clone https://github.com/mluzum/soft-sector-sim-adapters.git

python3 -m venv venv && . venv/bin/activate
pip install -r soft-sector-data-curation/requirements.txt matplotlib

python3 soft-sector-sim-adapters/tests/test_jetscape_qnvector.py    # all "ok"
```

If the analysis checkout lives elsewhere, set `SOFT_SECTOR_ANALYSIS=<path>`
or pass `--analysis-repo <path>`. To update later, run `git pull` in both
checkouts.

Every example below runs from the parent directory and uses this shorthand:

```bash
QV="python3 soft-sector-sim-adapters/tools/qnvector_observables.py"
```

## 2. Input

One run reads one design point of one system, given in either of two forms:

- **a directory** of `final_output_<...>_<id>.dat` (or `.dat.gz`) files,
  where the trailing integer of each name is the event id; or
- **a tar archive** of such files (`.tar`, `.tar.gz`, `.tgz`, `.gz.tar`).
  It is read as a stream and never unpacked.

Both give identical results. On storage that is slow with many small files,
the compressed archive can be much faster. On our `/data` disk, 40k O+O
events take about 6 minutes from a `.tar.gz` and about 30 minutes from the
directory.

The reader checks each file's header and grid, and works with:
- **any `Norder`**: harmonics 1 to Norder−1 are stored;
- **any number of oversamples**, read from each file;
- **missing event ids**: these count as failed runs and are dropped.

A file that is truncated or malformed is skipped with a warning naming
it, and `pipeline.log` gives the number of events loaded.

## 3. Choosing observables

### List what exists for a system

```bash
$QV --list --system "O O" --energy 5360
$QV --list --system "Pb Pb" --energy 5020
$QV --list --system "p Pb" --energy 5020
```

This prints one table per observable type, with one row per curated
measurement table: `ID`, value type, dimensionality (integrated or
differential in pT or η), method, harmonic, centrality classes, kinematic
window and HEPData table. `--system` is matched against the records'
reaction names, such as `O O --> CHARGED X` or `P PB --> ...`; matching
ignores case. `--reaction-regex` overrides that match.

### Write a selection and edit it

```bash
$QV --write-manifest oo5360.yaml --system "O O" --energy 5360
```

This writes every record of the system to a YAML manifest with
`select: true`. Set `select: false` on the records you do not want. Each
record entry carries enough to decide: `value_type`, `hepdata_file`,
`dimensionality`, `method`, `harmonic`, `cumulant_order` and
`centrality_classes`.

```yaml
records:
- id: obs_f21238f2
  fpath: data/flow_vn/atlas_flow_vn_OO_NeNe_5360gev_ins2967110.yaml
  hepdata_file: figure_7a_(3).yaml
  value_type: $v_{2}${4} (O+O)
  dimensionality: integrated
  method: cumulant
  harmonic: 2
  cumulant_order: 4
  select: true
  ...
```

A manifest per system, kept with your analysis, documents exactly what was
compared.

### Finer searches

The analysis repository's `scripts/search_records.py` filters further, by
`--dimensionality`, `--method`, `--harmonic`, `--cumulant-order` and
`--value-type-regex`. It writes a manifest in the same format, one
observable type at a time. Run it from inside `soft-sector-data-curation`:

```bash
cd soft-sector-data-curation
python3 scripts/search_records.py --type flow_vn --reaction '^O O' --energy 5360 \
    --cumulant-order 4 -o ../oo5360_v4.yaml
```

To combine searches of different types, concatenate the `records:` lists of
their manifests into one file. `--reaction` is a case-insensitive regex.
Avoid `*` in it: older versions of the script rewrite `*` as `.*`.

The types are the subdirectories of `soft-sector-data-curation/data/`.
These are computed by default:
- `flow_vn`
- `particle_yields`
- `particle_spectra`
- `mean_pT`
- `observable_ratios`
- `flow_harmonic_correlations`

`transverse_energy` and `radial_flow` are not in the default list, and the
wrapper has not been tested with them. Transverse energy needs neutral
particles as well, and the files hold only charged hadrons plus a few
identified species.

## 4. Computing

```bash
$QV results_OO/dp0007.tar.gz --system "O O" --energy 5360 \
    --manifest oo5360.yaml --out out/OO_dp0007
```

Leave out `--manifest` to compute every record of the system (of the
default types). Add `--max-events 300` for a quick test of a new dataset,
which takes a minute.

| option | |
|---|---|
| `--forward-beam p` | asymmetric systems: the beam moving toward +y. The simulation is taken to be in the nucleon–nucleon centre-of-mass frame, and records measured in the lab frame are shifted into it. |
| `--anchor 1.0` | centrality anchor. The default, 1.0, means minimum-bias events that span the inelastic cross section. A sample covering only the central fraction `f` of σ_inel takes `--anchor f`; uniform-b sampling takes `--anchor sigma_tot=<fm²>,b_max=<fm>`. |
| `--expected-events N` | ids 0…N−1 were generated, so the run summary lists the missing ones |
| `--species 211 -211 ...` | identified PDG ids to load (default π±, K±, p, p̄). Fewer species use less memory. |
| `--workers N` | parallel file readers (default min(16, cores)) |
| `-- <args>` | passed to `run_pipeline.py`, e.g. `-- --n-jk-blocks 40 --save-jackknife-replicates` |

Centrality is assigned separately for each record. Events are ranked by the
multiplicity or ET in that record's centrality-estimator window, for
example ATLAS FCal or CMS HF, and percentiles are taken relative to the
anchor. Statistical errors come from a jackknife over 20 blocks of events.

**Cost.** For O+O with 39,734 events, π±, K± and p/p̄ loaded, and all 214
records, reading the archive takes about 6 minutes. The analysis took 20
minutes on analysis commit 5b7014810 and 65 on b1fef19bb, because b1fef19bb
added the pair-gap v_n{2} correlations. A faster version of those
correlations is coming in the next analysis commit. Peak memory is about
28 GB, and `observables.h5` is about 3 MB.
A manifest that selects fewer records is faster. Memory grows with
the number of events, the number of species and `Norder`: storing
harmonics up to 8 roughly doubles it.

**Several design points or systems.** Each run is independent. Loop over
them in whatever way suits your workflow, for example:

```bash
for d in results_OO/dp*.tar.gz; do
    $QV "$d" --system "O O" --energy 5360 --manifest oo5360.yaml \
        --out out/OO_$(basename "$d" .tar.gz)
done
```

## 5. Output

`--out DIR` contains:

| file | |
|---|---|
| `observables.h5` | every computed value with its jackknife error and replicates (for the covariance), next to the experimental data |
| `observables.pdf` | simulation (line and band) against data (points), one panel per measurement table and centrality class |
| `observables_table.csv` | the same numbers as a flat table: system, table, value_type, class, x, sim, sim_err, data, data_err |
| `pipeline.log` | the full log of the analysis |
| `sim_package/simulation.yaml` | the settings of this run (input, system, anchor). It is kept for provenance and lets the run be repeated with `run_pipeline.py --sim-package`. |

The HDF5 file has one group per system and energy, for example
`O_O_-->_CHARGED_X_5360p0`:

| dataset | |
|---|---|
| `labels` | one label per value, e.g. `vn2_single_diff_pT pT 1.00-1.20 0-5% [$v_{2}$ 2PC/figure_1a_...]` |
| `values`, `errors` | simulation result and jackknife error |
| `jk_replicates` | the 20 delete-one-block jackknife replicates of `values`, shape (20, n). They give the covariance of all values in the group, which come from the same events (see below). |
| `jk_bias` | jackknife bias estimate |
| `label_record` | index *k* of the measurement each value belongs to |
| `exp_k/values`, `exp_k/errors` | the data of measurement *k*, in the same order as its values. The attributes give `record_file`, `hepdata_file`, `value_type`, `cent_classes` and `status`. |

The file's attributes record the provenance: the command line, both
repositories' commits, the adapter version and the run settings.

The dense covariance is not stored, because it would take n² values (370 MB
for the full O+O set) and has rank at most 19. Rebuild it exactly from the
replicates with `analysis.jackknife.covariance_from_replicates`, or by hand
as (B−1)/B Σ_b (r_b − r̄)(r_b − r̄)ᵀ. To store the dense matrix instead,
add `-- --covariance full`. With an analysis checkout older than b1fef19bb,
the wrapper writes `covariance` directly.

```python
import h5py, numpy as np

g = h5py.File("out/OO_dp0007/observables.h5")["O_O_-->_CHARGED_X_5360p0"]
labels = g["labels"].asstr()[:]
sim, err = g["values"][:], g["errors"][:]
if "covariance" in g:
    cov = g["covariance"][:]
else:
    r = g["jk_replicates"][:]                       # (n_blocks, n_values)
    d = r - r.mean(axis=0)
    cov = (len(r) - 1) / len(r) * d.T @ d           # = covariance_from_replicates(r)
rec = g["label_record"][:]
for k in np.unique(rec):
    i = np.flatnonzero(rec == k)
    e = g[f"exp_{k}"]
    print(e.attrs["value_type"], e.attrs["hepdata_file"], len(i), "values")
    data, data_err = e["values"][:], e["errors"][:]    # aligned with sim[i]
```

To plot several outputs together, run
`python3 soft-sector-sim-adapters/tools/plot_vs_data.py a.h5 b.h5 --out plots --png`.

## 6. What the format allows

- **Harmonics.** Only 1 to Norder−1 are available. With `Norder 4`, any
  v₄ or higher comes out NaN. v₂{4} is built without the Q₄
  self-correlation term. Records that need v_n{6}, v_n{8}, nonlinear
  response or symmetric cumulants need higher harmonics in the files:
  `Norder 9` (n ≤ 8) covers everything in the database.
- **Axes.** Charged hadrons (pid 9999) are binned in pseudorapidity, and
  identified species in rapidity. The analysis maps windows between the two
  where needed.
- **Acceptance.** The grid limits what can be computed. A window beyond
  the stored |η| (5.0 in the O+O run) is cut off: ALICE V0A extends to 5.1.
- **High pT.** With hydrodynamics plus Cooper–Frye, few particles are
  produced above about 4 GeV. Differential bins there have large errors,
  or are meaningless in peripheral classes, so check `errors` before
  fitting them.

## 7. Example: O+O at 5.36 TeV

`campaigns/oo5360_tau03_pos/` describes the first production run: τ₀ = 0.3
fm/c, nuclear-structure configurations with positive weights, minimum bias,
500 oversamples. Its README covers what was checked. On that run, the
command `$QV <archive> --system "O O" --energy 5360 --out oo5360` computes
214 records:

- **flow:** ALICE, ATLAS and CMS v₂ and v₃ (two- and four-particle
  cumulants, integrated and vs pT; the ATLAS v₄(pT) records need more
  harmonics);
- **multiplicity:** ATLAS and CMS dN_ch/dη, vs η and centrality;
- **spectra:** ATLAS charged-hadron pT spectra at |η| < 0.5, 1.5 < |η| <
  2.25 and |y_π| < 0.5. The forward y_π table is held back because its
  HEPData normalization is inconsistent;
- **⟨pT⟩:** ATLAS ⟨pT⟩ vs centrality.

The Ne+Ne/O+O ratios are left out, because they need a Ne+Ne run as well.
