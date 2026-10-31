# zr96c1: A=96 isobar collisions at 200 GeV, campaign 1 (superseded)

**Use [zr96c2](../zr96c2/README.md) for physics.** This campaign is kept for
comparison and transfer studies: it has the same 2000-point hypercube, so
`design_<k>` is the same normalized point in both campaigns. Its
configuration has two known problems, both fixed in campaign 2:

- **Hand-off.** Only the free-streamed energy density was handed to MUSIC
  (`Initial_profile 92`). The fluid started at rest, so the pre-flow and part
  of the energy were lost, and the flow is too small at the same parameters.
- **δf.** One particlization per surface, with the linear CE δf, which iSS's
  sampler silently clamps to [−1, 1].

Other differences from zr96c2:

- identified-particle pT grid 0.2–2.0 GeV in 41 bins;
- `normalization` prior 3–10.

Everything else matches zr96c2: TRENTo with σ_NN = 4.23 fm², the same nuclei
and b_max, decays-only afterburner, and the same seeds and events. The
centrality anchor uses the same `cross_sections_a96.csv`. The two
configurations are compared in
[`CAMPAIGN2_HANDOFF.md`](https://github.com/mluzum/iEBE-MUSIC/blob/NME/CAMPAIGN2_HANDOFF.md) of the iEBE-MUSIC fork.

## Use

As for zr96c2 (`./setup.sh --venv`, then `run_pipeline.py --sim-package <this
directory> --design-point N ...`), without `--variant`: the files hold a
single particlization, in the flat layout described in
`sim_adapters/iebe_music.py`.
