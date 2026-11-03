#!/usr/bin/env python3
"""Observables of one JETSCAPE QnVector simulation set, compared to data.

    # what can be compared for a system
    python tools/qnvector_observables.py --list --system "O O" --energy 5360
    python tools/qnvector_observables.py --write-manifest oo.yaml --system "O O" --energy 5360

    # compute (all records of the system, or the ones selected in a manifest)
    python tools/qnvector_observables.py <events> --system "O O" --energy 5360 --out DIR
    python tools/qnvector_observables.py <events> --system "O O" --energy 5360 \\
        --manifest oo.yaml --out DIR

<events> is one design point of one collision system: a directory of
``final_output_*.dat[.gz]`` event files, or a tar archive of them (read as a
stream, never unpacked).  The script

1. writes ``DIR/sim_package/simulation.yaml``, a one-design-point package
   for ``sim_adapters.jetscape_qnvector`` (kept for provenance and reruns);
2. runs ``scripts/run_pipeline.py`` of soft-sector-data-curation on it.
   Every curated record of that system and energy whose type is in
   ``--types`` (or every record selected in ``--manifest``) is computed,
   with jackknife errors, into ``DIR/observables.h5``;
3. plots simulation vs data (``tools/plot_vs_data.py``):
   ``DIR/observables.pdf`` and ``DIR/observables_table.csv``.

``--list`` prints the curated records of the system and energy, one table
per type.  ``--write-manifest`` writes them to a YAML manifest with
``select: true`` on every record; set ``select: false`` on the ones to
leave out and pass the file to ``--manifest``.

Centrality is assigned per record from the multiplicity (or ET) in that
record's estimator window.  The default anchor, 1.0, says the events are
minimum bias: the sample spans the inelastic cross section.  For a sample
restricted to small impact parameters give its fraction ``f`` of sigma_inel,
or ``sigma_tot=<fm^2>,b_max=<fm>`` for uniform-b sampling.

Asymmetric systems (p+Pb, ...): the simulation is taken to be in the
nucleon-nucleon centre-of-mass frame, and ``--forward-beam`` names the beam
moving toward +y (e.g. ``p``).  Records measured in the lab frame are
shifted into it by the analysis.

Arguments after ``--`` go to run_pipeline.py unchanged, e.g.
``-- --n-jk-blocks 40 --save-jackknife-replicates``.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ANALYSIS_REPO = "soft-sector-data-curation"
ADAPTER = "sim_adapters.jetscape_qnvector"
TYPES = ["flow_vn", "particle_yields", "particle_spectra", "mean_pT",
         "observable_ratios", "flow_harmonic_correlations"]


def find_analysis(arg: str | None) -> Path:
    for c in (arg, os.environ.get("SOFT_SECTOR_ANALYSIS"), REPO.parent / ANALYSIS_REPO):
        if c and (Path(c) / "scripts" / "run_pipeline.py").is_file():
            return Path(c).resolve()
    sys.exit(f"cannot find the {ANALYSIS_REPO} checkout next to {REPO}: "
             f"pass --analysis-repo or set SOFT_SECTOR_ANALYSIS")


def parse_anchor(s: str):
    try:
        return float(s)
    except ValueError:
        return s            # 'n_ref=...', 'sigma_tot=...,b_max=...': resolve_anchor


def reaction_regex(system: str) -> str:
    """Records' reactions read 'O O --> CHARGED X', 'P PB --> ...'.  No '*':
    older search_records.py rewrote it as '.*'."""
    return "^" + r"\s+".join(re.escape(t) for t in system.split()) + r"\s"


def write_package(pkg: Path, events: Path, *, name: str, system: str, energy: float,
                  forward_beam: str | None = None, anchor=1.0, expected_events=None,
                  species=None, workers=None, max_events=None) -> Path:
    """A one-design-point simulation package (design point 0) for ``events``."""
    opts = {"sources": {0: str(events)}, "anchor": anchor, "missing": "drop"}
    for k, v in (("expected_events", expected_events), ("species", species),
                 ("workers", workers), ("max_events", max_events)):
        if v is not None:
            opts[k] = list(v) if k == "species" else v
    sysent = {"reaction": system, "energy_gev": float(energy)}
    if forward_beam:
        sysent.update(frame="nn_cms", forward_beam=forward_beam)
    manifest = {
        "format_version": 1,
        "name": name,
        "title": f"{system} at {energy:g} GeV: {events}",
        "container": "3d",
        "data_root": str(events.parent),
        "adapter": {"module": ADAPTER, "repo": REPO.name, "options": opts},
        "systems": [sysent],
        "variants": None,
    }
    pkg.mkdir(parents=True, exist_ok=True)
    path = pkg / "simulation.yaml"
    with open(path, "w") as fh:
        fh.write("# Written by tools/qnvector_observables.py; read by run_pipeline.py "
                 "--sim-package.\n")
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return path


def search(analysis: Path, types, rx: str, energy: float, output: Path | None = None):
    """Run search_records.py per type: print the tables, or merge the
    manifests into ``output``.  Returns the number of records found."""
    script = analysis / "scripts" / "search_records.py"
    merged, n = None, 0
    with tempfile.TemporaryDirectory() as tmp:
        for t in types:
            base = [sys.executable, str(script), "--type", t, "--reaction", rx,
                    "--energy", f"{energy:g}"]
            if output is None:
                r = subprocess.run(base, cwd=analysis, capture_output=True, text=True)
                lines = [l for l in r.stdout.splitlines() if l.startswith("obs_")]
                if r.returncode != 0:
                    sys.exit(f"search_records.py --type {t} failed:\n{r.stderr}")
                if lines:
                    print(f"\n== {t}: {len(lines)} records\n" + r.stdout.rstrip())
                n += len(lines)
                continue
            out = Path(tmp) / f"{t}.yaml"
            r = subprocess.run(base + ["--output", str(out)], cwd=analysis,
                               capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"search_records.py --type {t} failed:\n{r.stderr}")
            if not out.is_file():
                continue
            m = yaml.safe_load(out.read_text()) or {}
            recs = m.get("records") or []
            n += len(recs)
            if merged is None:
                merged = dict(m, records=[])
            merged["records"] += recs
    if output is not None and merged is not None:
        merged["description"] = (f"records matching reaction {rx!r} at {energy:g} GeV "
                                 f"(types {' '.join(types)}); set select: false to skip one")
        with open(output, "w") as fh:
            yaml.safe_dump(merged, fh, sort_keys=False)
    return n


def manifest_types(path: Path) -> list[str]:
    """Record types (data/<type>/...) of the selected records of a manifest."""
    m = yaml.safe_load(Path(path).read_text()) or {}
    types = []
    for r in m.get("records") or []:
        if r.get("select", True) is False:
            continue
        parts = Path(str(r.get("fpath", ""))).parts
        t = parts[parts.index("data") + 1] if "data" in parts[:-1] else None
        if t and t not in types:
            types.append(t)
    if not types:
        sys.exit(f"{path}: no selected records")
    return types


def run_logged(cmd, cwd, log: Path) -> int:
    with open(log, "w") as fh:
        fh.write(" ".join(map(str, cmd)) + "\n\n")
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        held = []           # a group's header, shown unless the group is skipped
        for line in p.stdout:
            fh.write(line)
            s = line.strip()
            if not s or s.startswith(("Group:", "====")):
                held.append(line)
                continue
            if "does not simulate" in s:
                held = []
                continue
            sys.stdout.write("".join(held) + line)
            held = []
        return p.wait()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("events", nargs="?",
                    help="directory of event files, or a tar archive of them")
    ap.add_argument("--system", required=True, help="reaction, e.g. 'O O', 'Pb Pb', 'p Pb'")
    ap.add_argument("--energy", required=True, type=float, help="sqrt(s_NN) in GeV")
    ap.add_argument("--out", help="output directory")
    sel = ap.add_argument_group("record selection")
    sel.add_argument("--list", action="store_true",
                     help="print the curated records of the system and energy, then exit")
    sel.add_argument("--write-manifest", metavar="FILE",
                     help="write them to an editable manifest, then exit")
    sel.add_argument("--manifest", metavar="FILE",
                     help="compute only the records selected in this manifest")
    sel.add_argument("--types", nargs="+",
                     help=f"record types (data/ subdirectories; default {' '.join(TYPES)}, "
                          f"or those of --manifest)")
    sel.add_argument("--reaction-regex",
                     help="regex on the records' reaction (default from --system)")
    ap.add_argument("--forward-beam", help="asymmetric systems: the beam toward +y")
    ap.add_argument("--anchor", default="1.0", help="centrality anchor (default 1.0: minimum bias)")
    ap.add_argument("--expected-events", type=int,
                    help="number of event indices generated (reports the missing ones)")
    ap.add_argument("--species", nargs="+", type=int,
                    help="identified PDG ids to load (default pi+-, K+-, p, pbar)")
    ap.add_argument("--workers", type=int, help="reader processes (default min(16, cores))")
    ap.add_argument("--max-events", type=int, help="read only the first N events (quick test)")
    ap.add_argument("--name", help="name in the output provenance (default: events basename)")
    ap.add_argument("--analysis-repo", help=f"{ANALYSIS_REPO} checkout (default: next to this one)")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    analysis = find_analysis(a.analysis_repo)
    rx = a.reaction_regex or reaction_regex(a.system)
    if a.list or a.write_manifest:
        types = a.types or TYPES
        out = Path(a.write_manifest).resolve() if a.write_manifest else None
        n = search(analysis, types, rx, a.energy, out)
        print(f"\n{n} records for reaction ~ {rx!r} at {a.energy:g} GeV"
              + (f" -> {out}" if out else ""))
        return
    if not a.events or not a.out:
        ap.error("give <events> and --out (or --list / --write-manifest)")
    if a.manifest:
        manifest = Path(a.manifest).resolve()
        types = a.types or manifest_types(manifest)
    else:
        manifest, types = None, a.types or TYPES

    events = Path(a.events).resolve()
    if not events.exists():
        sys.exit(f"{events} does not exist")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or re.sub(r"(\.tar)?(\.gz|\.tgz)?(\.tar)?$", "", events.name)

    pkg = out / "sim_package"
    write_package(pkg, events, name=name, system=a.system, energy=a.energy,
                  forward_beam=a.forward_beam, anchor=parse_anchor(a.anchor),
                  expected_events=a.expected_events, species=a.species,
                  workers=a.workers, max_events=a.max_events)
    h5 = out / "observables.h5"
    cmd = [sys.executable, analysis / "scripts" / "run_pipeline.py",
           "--sim-package", pkg, "--design-point", "0", "--adapter-path", REPO,
           "--type", *types, "--energy", f"{a.energy:g}", "--reaction", rx,
           "--species-grids", "--output", h5]
    if manifest:
        cmd += ["--manifest", manifest]
    cmd += extra
    print(f"{a.system} at {a.energy:g} GeV from {events}\n  package {pkg}\n"
          f"  log {out / 'pipeline.log'}", flush=True)
    rc = run_logged(cmd, analysis, out / "pipeline.log")
    if rc != 0:
        sys.exit(f"run_pipeline.py failed (exit {rc}); see {out / 'pipeline.log'}")
    if not h5.is_file():
        sys.exit(f"no records of {a.system} at {a.energy:g} GeV were computed "
                 f"(types {' '.join(types)}); see {out / 'pipeline.log'}")
    if not a.no_plots:
        rc = subprocess.call([sys.executable, str(HERE / "plot_vs_data.py"), str(h5),
                              "--out", str(out), "--stem", "observables"])
        if rc != 0:
            sys.exit(f"plot_vs_data.py failed (exit {rc})")
    print(f"done: {h5}")


if __name__ == "__main__":
    main()
