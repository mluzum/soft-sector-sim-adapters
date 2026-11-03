# oo5360_tau03_pos: O+O at 5.36 TeV, τ₀ = 0.3 fm/c, minimum bias

The production run `resultsOO_QVec_tau03_big_pos`. It has one parameter
point (design point 0), with these settings:

- τ₀ = 0.3 fm/c;
- the oxygen configurations come from a nuclear-structure calculation,
  using the positive-weight set (`pos`);
- the events are minimum bias, and each hydro event is sampled 500 times
  with SMASH as afterburner.

Each event file is the output of the JETSCAPE QnVector writer. It holds
dN and Q₁–Q₃ on a pT × rapidity grid with 30 pT bins (0–6 GeV) and 50
rapidity bins (−5 to 5), all oversamples summed. The format is described
in `sim_adapters/jetscape_qnvector.py`.

## Use

In one command, without a deployed package:

```sh
python3 tools/qnvector_observables.py /data/mluzum/Andreas/resultsOO_QVec_tau03_big_pos.gz.tar \
    --system "O O" --energy 5360 --expected-events 40000 --out oo5360/
```

Or, with this package:

```sh
./setup.sh --venv                     # after deployment; see the repository README
python scripts/run_pipeline.py --sim-package <this directory> \
    [--data-root <dir holding the archive>] --output oo5360.h5
```

`run_pipeline.py` computes only the types given with `--type` (default
`flow_vn`); add `--type flow_vn particle_yields`. The manifest's
`selection` (those two types, O+O at 5360 GeV) is applied by
`run_campaign.py` and `check_sim_package.py`. Pass
`--species-grids` for records of identified particles: π±, K± and p/p̄
are loaded.

Loading the archive takes about 3–6 minutes, depending on whether the
file is in the page cache. A full run of all 214 records takes about 13
minutes and peaks at about 28 GB of memory. Reading the
loose directory instead takes about 30 minutes, because `/data` is slow
with many small files. To read the directory, map `sources` to it.

## What the analysis has to know

- **Harmonics 1–3 only** (`Norder 4`). v₄ and higher come out NaN.
  v₂{4} is built without the Q₄ self-correlation terms. Records that
  need v_n{6}, v_n{8} or symmetric cumulants are skipped. Later runs
  will store more harmonics, and the adapter reads any `Norder`.
- **Axes.** Charged hadrons (pid 9999) are binned in pseudorapidity.
  Identified species are binned in rapidity y. The adapter passes them
  as `eta_edges` and `y_edges`, so they cannot be mixed up.
- **Charged hadrons** are everything charged that remains after SMASH.
  It is not yet known whether SMASH performs weak decays here. This
  matters for comparisons with p/p̄ and with "primary" charged-particle
  yields.
- **Missing events.** Indices 0–39999 were generated, and 266 runs
  failed without writing a file. They are dropped, which leaves 39734
  events. `meta['missing_event_ids']` lists them.
- **Centrality anchor.** The sample is minimum bias, so it spans the
  whole inelastic cross section (`f = 1`). Centrality is assigned per
  record from the multiplicity or ET in the record's estimator window.
- **Oversampling.** The adapter infers the number of oversamples per
  event from the file (`nsamples`). Larger systems such as Pb+Pb will
  use 200 oversamples, which needs no change.
