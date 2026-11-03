# soft-sector-sim-adapters

Readers that turn simulation output formats into the containers of the
soft-sector-data-curation pipeline, plus the
*simulation packages* that describe each simulation campaign.

The analysis repository is model-independent. It computes every observable
from a curated experimental record (acceptance, centrality definition,
method) applied to one of two in-memory containers:

- **2D**, `QVectorData`: boost-invariant Q-vectors per event, species, harmonic and pT bin;
- **3D**, `QGridData`: Q-vectors on an η × pT grid.

An adapter is the only model-specific code. It reads one output format and
returns the container's arrays. It imports only numpy and its own readers,
never the analysis repository, so the two evolve independently. The
contract is documented in the analysis repository's
`docs/sim_interface.md` §4.

## Layout

| path | |
|---|---|
| `sim_adapters/iebe_music.py` | merged iEBE-MUSIC campaigns (TRENTo + MUSIC 2D + iSS + decays, hadronic_afterburner_toolkit Q-vectors; flat and per-δf-model layouts) |
| `sim_adapters/jetscape_qnvector.py` | JETSCAPE QnVector final-state text files, read directly (3D: charged hadrons on an η × pT grid, identified species on rapidity × pT grids; one event per file, in a directory or a tar archive) |
| `campaigns/<name>/` | one simulation package per campaign: `simulation.yaml`, a README for users, design and cross-section tables |
| `tools/deploy_package.py` | installs a campaign's package next to its data, pinning both repositories |
| `tools/qnvector_observables.py` | one command from a set of JETSCAPE QnVector events (one design point of one system) to observables vs data: lists or selects the records of a system (`--list`, `--write-manifest`, `--manifest`), writes the package, runs the pipeline, plots |
| `docs/jetscape_qnvector.md` | user guide for JETSCAPE QnVector output |
| `tools/plot_vs_data.py` | simulation vs experiment for every record in a `run_pipeline.py` output (PDF + CSV table) |
| `templates/setup.sh` | the package's setup script (clones both repositories at the pinned commits) |
| `tests/` | adapter tests on synthetic campaign directories (`python3 tests/test_iebe_music.py`, `python3 tests/test_jetscape_qnvector.py`) |

Campaigns: `zr96c2` (A=96 isobars, 200 GeV, campaign 2), `zr96c1`
(campaign 1, superseded) and `oo5360_tau03_pos` (O+O at 5.36 TeV, JETSCAPE).

## JETSCAPE QnVector output in one command

**Guide for users: [`docs/jetscape_qnvector.md`](docs/jetscape_qnvector.md)**, covering setup,
choosing observables per system, running, and reading the output.

For one set of events (a directory of `final_output_*.dat` files or a tar
archive of them), with this repository and `soft-sector-data-curation`
checked out side by side:

```bash
python3 tools/qnvector_observables.py <events dir or .tar.gz> \
    --system "O O" --energy 5360 --out oo5360/
```

This writes `oo5360/observables.h5`, which holds every curated record of that system and
energy with jackknife errors. It also writes `observables.pdf`, which plots simulation
against data, `observables_table.csv`, and the generated `sim_package/`.
The defaults are minimum-bias events (`--anchor 1.0`) and failed runs
dropped. Pass `--max-events 300` for a quick test, and `--forward-beam p`
for p+Pb in the nucleon–nucleon frame. `--list` and `--write-manifest`
show or select the records of a system, and `--manifest` restricts a run
to the selection. Arguments after `--` go to `run_pipeline.py`. All
options: `--help`.

## Adding a campaign

1. Write `campaigns/<name>/simulation.yaml` (copy `zr96c2`'s) and a
   `README.md` describing the physics and the data.
2. Add whatever the adapter options name: cross-section table, design CSV.
3. Commit this repository and the analysis repository, then run:

   ```bash
   python3 tools/deploy_package.py <name> <results dir>/sim_package
   python3 ../soft-sector-data-curation/scripts/check_sim_package.py <results dir>/sim_package
   ```

   `deploy_package.py` refuses uncommitted changes, because the pinned
   commits must contain the tested code. Rerun it after either repository
   changes in a way the package should pick up. Once the repositories are
   published, pass `--analysis-url` and `--adapters-url`, or set each
   checkout's `origin` remote.

## Adding a format

Write `sim_adapters/<format>.py` with `API_VERSION = 1` and
`open_simulation(data_root, package_dir, **options)`. The latter returns an
object with `design_points()`, `load(design_point, variant)`, and optionally
`anchor(design_point)` and `source_files(design_point)`. `load` returns a
dict of the container fields (anything else under `meta`). The analysis
side validates it: its `scripts/check_sim_package.py` is the first test.
For physics validation, push toy events with known flow through the
adapter and compare with the analysis repository's toy-MC harness.

## License

MIT (`LICENSE`).
