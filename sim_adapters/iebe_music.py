"""Adapter for merged iEBE-MUSIC campaigns (2D boost-invariant).

Reads the merged per-design-point files written by the GeSe campaign
pipeline (iEBE-MUSIC fork, github.com/mluzum/iEBE-MUSIC: TRENTo -> free
streaming -> MUSIC -> iSS -> decays -> hadronic_afterburner_toolkit) and
returns the arrays of the soft-sector pipeline's 2D container.

Layout of ``<data_root>``::

    dp<NNNN>/dp<NNNN>.h5   one group per event, ``spvn_results_<event_id>``
    dp<NNNN>/csv.tgz       drawn-event table(s): event_id, b, b_max,
                           b_over_bmax, mult, status (ok / no_collision;
                           no_trento rows, where TRENTo itself did not run,
                           were never drawn and are skipped)

Each event group holds, for 13 identified species, a pT-differential table
``particle_<pid>_vndata_diff_y_-30_30_boostinv.dat`` (n_pT x 13: mean pT,
E dN/d^3p, Q1..Q5 real/imag, total_NpT) and a pT-integrated table
``particle_<pid>_vndata_y_-30_30_boostinv.dat`` (7 x 5, last row
``99, total_N, n_samples, 0, 0``).  The differential table stores
per-particle averages <e^{in phi}> and counts, so the accumulated Q-vector
of a pT bin is exactly ``Q_n = total_NpT * (Qn_real + i Qn_imag)``;
``n_samples`` (iSS resamplings) is returned per event and divided out
downstream.  The yield is boost-invariant over a rapidity width of 10
(``rap_max = 5``).

Two layouts exist:

- campaign 1 (zr96, ge76): the tables sit directly in the event group, on
  the toolkit grid 0.2-2.0 GeV in 41 bins (not recorded in the file);
- campaign 2 onward: one sub-group ``deltaf_<model>`` per delta f model
  (attr ``deltaf_models``) and the toolkit grid in attrs
  ``toolkit_pT_min``, ``toolkit_pT_max``, ``toolkit_npT``.  A variant
  (delta f model) must be chosen.

Drawn events without a fluid (no collision, or hit without freeze-out:
status ``no_surface``) have no group; with ``include_empty`` they are
appended as empty events, so the loaded sample is every draw with b < b_max
and covers the fraction f = pi b_max^2 / sigma_inel of the inelastic cross
section.  ``anchor`` returns that from the per-design-point cross-section
table (columns ``dp``, ``b_max``, ``sigma_inel_fm2``), after checking that
the table's b_max is the one the campaign drew with.  Draws whose hydro
failed (status ``hydro_failed``: lost job, crash) are not events and are
left out (``meta["n_failed"]``).

Standard output (iEBE-MUSIC docs/standard_output.md, format
"iebe-music-qn" version 1, 2D): ``dp<NNNN>/qn_dp<NNNN>.h5`` with the exact
per-event sums of every final particle -- ``/grid`` (pt_edges, species_pdg,
species_mass, species_charge, species_nmax), ``/events`` (the collided
events that reached particlization: status ok or no_surface, event_id,
weight, ...), ``/drawn`` (every drawn id and its fate) and one group
``deltaf_<model>`` per variant (Q [E, species, 11, pT], ptsum [E, species,
4, pT], mtsum, nsamples; attrs dy_width, collection, sums).  The species
rows are disjoint and include "other charged" (PDG 0: Sigma+-, nuclei,
...), so charged hadrons are the sum of the rows with non-zero charge; a
row's harmonics above its ``species_nmax`` are not stored (returned with
the container as missing, never as Q_n = 0).  no_surface rows (all sums
0, nsamples 0) are empty events (nsamples 1).  ``rap_max`` is dy_width / 2.
The anchor is n_ref = (collided events loaded) / frac_below_bmax (the
cross-section table), b_max checked against the file's; draws whose hydro
failed are not events, as above.  Importance-sampled files (root attrs
selection_format 1 and the selection summary, /drawn the accepted draws
with weight, p_acc, centrality, status): the events' weights, the failed
draws' weight to the survivors of their selection bin, and the selection's
anchor, as below.  A variant written with the optional HBT sums
(docs/standard_output.md section 9: /grid/hbt_* and hbt_num, hbt_pairs,
hbt_N) returns them as ``hbt`` (the pipeline's analysis.hbt.HBTSums
fields; soft-sector-data-curation with femtoscopy support): per event,
like-sign pion pair sums sum_{i != j} cos(q.dx~) and N (N - 1) per row of
species, single-particle K_T bin, lab azimuth bin and q point.

Importance-sampled campaigns (iEBE-MUSIC docs/importance_sampling.md)
simulate only the draws accepted with a probability P from their
initial-state centrality, each with the weight w = 1/P.  Their merged file
says so (root attrs ``selection_format``, ``n_drawn``, ``n_collided``,
``covered_centrality``, ``selection``: the summary as JSON), every event
group carries ``event_weight`` (absent: 1), and ``selection_dp<NNNN>.csv``
next to it lists every accepted draw with its weight and ``status``:

- ``ok``: its group is in the file, used with its weight;
- ``no_surface``: a collision with nothing to particlize, an empty event
  with its weight (``include_empty``);
- ``failed``: nothing came back.  Never an empty event: with
  ``failed="reweight"`` (the default) the weight of the failed draws goes to
  the surviving ones of the same selection bin (the summary's
  ``centrality_bins``, else 5% bins of the initial-state centrality), which
  is exact if failures are random within a bin; ``failed="error"`` refuses
  the design point.

The returned ``event_weights`` enter every event average, and ``anchor``
gives the campaign's exact total, n_ref = n_collided / coverage (the
collided draws of its first phase over the fraction of the inelastic cross
section that b_max covers), plus ``covered_cent`` when the selection
accepts no events beyond some centrality inside that fraction.
"""

from __future__ import annotations

import csv
import io
import json
import re
import tarfile
import warnings
from pathlib import Path

import h5py
import numpy as np

API_VERSION = 1
__version__ = "1.4.0"

# PDG id, short name, mass (GeV).  Order defines the species axis.
SPECIES = [
    (211,    "pi+",       0.13957),
    (-211,   "pi-",       0.13957),
    (321,    "K+",        0.49368),
    (-321,   "K-",        0.49368),
    (2212,   "p",         0.93827),
    (-2212,  "pbar",      0.93827),
    (3122,   "Lambda",    1.115683),
    (-3122,  "Lambdabar", 1.115683),
    (3312,   "Xi-",       1.32171),
    (-3312,  "Xibar+",    1.32171),
    (3334,   "Omega-",    1.67245),
    (-3334,  "Omegabar+", 1.67245),
    (333,    "phi",       1.019461),
]
PDG_IDS = [s[0] for s in SPECIES]
SPECIES_NAMES = [s[1] for s in SPECIES]
MASSES = np.array([s[2] for s in SPECIES], dtype=float)

# Toolkit pT grid of the campaign-1 files, which do not record it
# (npT = 41, pT_min = 0.2, pT_max = 2.0 in their parameter dictionaries).
LEGACY_PT_EDGES = np.linspace(0.2, 2.0, 41 + 1)

# Boost-invariant rapidity half-width of the yields.
RAP_MAX = 5.0

N_HARM = 6   # slot 0 counts, slots 1..5 Q1..Q5


def _dataset_name(pid: int, differential: bool) -> str:
    kind = "diff_y_-30_30_boostinv" if differential else "y_-30_30_boostinv"
    return f"particle_{pid}_vndata_{kind}.dat"


def _event_id(key: str) -> int:
    m = re.search(r"(\d+)$", key)
    return int(m.group(1)) if m else -1


def pt_edges(g: h5py.Group) -> np.ndarray:
    """Toolkit bin edges recorded in the event group, else the campaign-1 grid."""
    if "toolkit_npT" in g.attrs:
        return np.linspace(float(g.attrs["toolkit_pT_min"]),
                           float(g.attrs["toolkit_pT_max"]),
                           int(g.attrs["toolkit_npT"]) + 1)
    return LEGACY_PT_EDGES


def models(g: h5py.Group) -> list[str]:
    """Delta f models of an event group ([] for the flat campaign-1 layout)."""
    return [k[len("deltaf_"):] for k in g.keys() if k.startswith("deltaf_")]


def flow_group(g: h5py.Group, variant: str | None) -> h5py.Group:
    """The group holding the identified-particle tables of one delta f model."""
    have = models(g)
    if not have:
        if variant is not None:
            raise ValueError(f"{g.name} has a single particlization; "
                             f"variant {variant!r} cannot be selected")
        return g
    if variant is None:
        raise ValueError(f"{g.name} holds delta f models {have}; choose one")
    if variant not in have:
        raise ValueError(f"{g.name}: no delta f model {variant!r} (have {have})")
    return g["deltaf_" + variant]


def read_event(g: h5py.Group, n_pt: int):
    """Identified-particle tables of one event (or model) group.

    Returns ``qn`` (n_species, 6, n_pt) complex (slot 0 the raw count
    total_NpT, slots 1..5 the accumulated N <e^{in phi}>, not divided by
    n_samples), ``sum_pt`` (n_species, n_pt) = total_NpT * mean pT, and the
    number of iSS resamplings.
    """
    n_sp = len(SPECIES)
    qn = np.zeros((n_sp, N_HARM, n_pt), dtype=np.complex128)
    sum_pt = np.zeros((n_sp, n_pt), dtype=float)

    # The integrated table's bookkeeping row is species-independent.
    integ = g[_dataset_name(SPECIES[0][0], differential=False)][...]
    n_s = int(round(float(integ[6, 2])))
    if n_s <= 0:
        n_s = 1

    for i, (pid, _name, _mass) in enumerate(SPECIES):
        d = g[_dataset_name(pid, differential=True)][...]
        n = d[:, 12].astype(float)
        qn[i, 0, :] = n
        for harm in range(1, N_HARM):
            qn[i, harm, :] = n * (d[:, 2 * harm] + 1j * d[:, 2 * harm + 1])
        sum_pt[i, :] = n * d[:, 0]
    return qn, sum_pt, n_s


def check_pt_grid(sum_pt: np.ndarray, counts: np.ndarray, edges: np.ndarray) -> None:
    """Raise if the stored per-bin mean pT does not lie inside ``edges``.

    Column 0 of the differential table is the mean pT of the particles in a
    bin, so its count-weighted average over events and species must fall
    inside the bin; this catches a file read with the wrong grid.
    """
    s = sum_pt.reshape(-1, sum_pt.shape[-1]).sum(axis=0)
    n = counts.reshape(-1, counts.shape[-1]).sum(axis=0)
    filled = n > 0
    mean = np.divide(s, n, out=np.zeros_like(s), where=filled)
    inside = (mean >= edges[:-1]) & (mean < edges[1:])
    bad = np.flatnonzero(filled & ~inside)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"pT grid mismatch: bin {i} has mean pT {mean[i]:.4f} GeV outside "
            f"[{edges[i]:.4f}, {edges[i + 1]:.4f}); {bad.size} of "
            f"{int(filled.sum())} filled bins disagree with the grid "
            f"{edges[0]:.3f}-{edges[-1]:.3f} GeV")


def read_drawn_events(csv_tgz) -> dict[str, np.ndarray]:
    """The drawn-event table of ``csv.tgz`` (event_id, b, b_max, b_over_bmax, mult, status).

    ``status``: ok, no_collision, no_surface (a collision without freeze-out;
    campaigns before November 2026 wrote ok), hydro_failed (lost or crashed).

    A row with status ``no_trento`` (TRENTo itself did not run: no b, no
    mult) is not a drawn event -- neither a collision nor an empty draw --
    and is skipped, as the campaign's cross-section collector does
    (GeSe scripts/cross_section/collect_xsec.py); ``n_not_run`` counts them.
    Counting it as an empty event would shift the centrality anchor."""
    rows = []
    with tarfile.open(csv_tgz, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            rows.extend(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
    if not rows:
        raise ValueError(f"no CSV rows in {csv_tgz}")
    n_not_run = sum(1 for r in rows if r.get("status") == "no_trento" or r.get("mult", "") == "")
    rows = [r for r in rows if not (r.get("status") == "no_trento" or r.get("mult", "") == "")]

    def col(name):
        return np.asarray([float(r[name]) for r in rows], dtype=float)

    return {
        "event_id": np.asarray([int(r["event_id"]) for r in rows], dtype=np.int64),
        "b": col("b"),
        "b_max": col("b_max"),
        "b_over_bmax": col("b_over_bmax"),
        "mult": col("mult"),
        "status": np.asarray([r["status"] for r in rows]),
        "n_not_run": n_not_run,
    }


SELECTION_FORMAT = 1
FAILED_MODES = ("reweight", "error")
# |selection coverage - frac_below_bmax of the cross-section table| allowed
# (the table's statistical error is ~1e-3)
COVERAGE_TOLERANCE = 0.005


def selection_summary(h5_path) -> dict | None:
    """The selection summary of an importance-sampled design point (root
    attrs of its merged file), or None for a campaign without selection."""
    with h5py.File(h5_path, "r") as f:
        if "selection_format" not in f.attrs:
            return None
        fmt = int(f.attrs["selection_format"])
        if fmt != SELECTION_FORMAT:
            raise ValueError(f"{h5_path}: selection_format {fmt}, this adapter reads "
                             f"{SELECTION_FORMAT}")
        summary = json.loads(f.attrs["selection"])
        for key in ("n_drawn", "n_collided", "b_max"):
            if key in f.attrs:
                summary.setdefault(key, f.attrs[key].item() if hasattr(f.attrs[key], "item")
                                   else f.attrs[key])
    return summary


def read_selection(h5_path) -> dict[str, np.ndarray]:
    """``selection_dp<NNNN>.csv`` next to the merged file: every accepted draw
    (event_id, weight, p_acc, centrality, status)."""
    h5_path = Path(h5_path)
    path = h5_path.with_name(f"selection_{h5_path.stem}.csv")
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the event list of an importance-sampled "
                                f"design point (merge_dp_h5.py --selection)")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "status" not in rows[0]:
        raise ValueError(f"{path} has no status column: merge with merge_dp_h5.py --selection")
    bad = sorted({r["status"] for r in rows} - {"ok", "no_surface", "failed"})
    if bad:
        raise ValueError(f"{path}: unknown status {bad}")
    return {
        "event_id": np.asarray([int(r["event_id"]) for r in rows], dtype=np.int64),
        "weight": np.asarray([float(r["weight"]) for r in rows], dtype=float),
        "p_acc": np.asarray([float(r["p_acc"]) for r in rows], dtype=float),
        "centrality": np.asarray([float(r["centrality"]) for r in rows], dtype=float),
        "status": np.asarray([r["status"] for r in rows]),
    }


def survival_factors(centrality, weight, failed, edges) -> np.ndarray:
    """Per accepted draw, the factor that gives the failed draws' weight to
    the surviving ones of the same centrality bin: sum of all weights over
    the survivors' (1 for a failed draw, which is dropped).  Bins below the
    first and beyond the last edge are bins too.  A bin whose draws all
    failed cannot be corrected (ValueError)."""
    centrality = np.asarray(centrality, dtype=float)
    weight = np.asarray(weight, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    edges = np.asarray(sorted(float(e) for e in edges))
    bins = np.searchsorted(edges, centrality, side="right")
    factor = np.ones(len(weight))
    for b in np.unique(bins[failed]):
        sel = bins == b
        alive = sel & ~failed
        if not alive.any():
            lo = edges[b - 1] if b > 0 else -np.inf
            hi = edges[b] if b < len(edges) else np.inf
            raise ValueError(f"every accepted draw with initial-state centrality in "
                             f"[{lo:g}, {hi:g})% failed: re-run them")
        factor[alive] = weight[sel].sum() / weight[alive].sum()
    return factor


def read_design_point(h5_path, variant: str | None = None, drawn_csv=None,
                      include_empty: bool = True, failed: str = "reweight") -> dict:
    """Arrays of one merged design-point file, in the 2D container layout.

    An importance-sampled design point (``selection_summary``) also returns
    ``event_weights``; its empty events come from its selection list, not
    from ``drawn_csv``.  ``failed``: "reweight" or "error" (module docstring).
    """
    if failed not in FAILED_MODES:
        raise ValueError(f"failed={failed!r}: one of {FAILED_MODES}")
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        keys = sorted((k for k in f.keys() if k.startswith("spvn_results_")), key=_event_id)
        if not keys:
            raise ValueError(f"{h5_path}: no events")
        edges = pt_edges(f[keys[0]])
        qn_list, spt_list, ns_list, ids, wts = [], [], [], [], []
        for k in keys:
            if not np.array_equal(pt_edges(f[k]), edges):
                raise ValueError(f"{h5_path}: event {k} has a different pT grid")
            qn, spt, ns = read_event(flow_group(f[k], variant), len(edges) - 1)
            qn_list.append(qn)
            spt_list.append(spt)
            ns_list.append(ns)
            ids.append(_event_id(k))
            wts.append(float(f[k].attrs.get("event_weight", 1.0)))

    qn = np.stack(qn_list, axis=0)
    sum_pt = np.stack(spt_list, axis=0)
    nsamples = np.asarray(ns_list, dtype=float)
    event_ids = np.asarray(ids, dtype=np.int64)
    weights = np.asarray(wts, dtype=float)
    check_pt_grid(sum_pt, qn[:, :, 0, :].real, edges)
    meta = {"source": str(h5_path)}

    def pad(pad_ids, pad_w=None):
        nonlocal qn, sum_pt, nsamples, event_ids, weights
        n = len(pad_ids)
        if n:
            qn = np.concatenate([qn, np.zeros((n,) + qn.shape[1:], qn.dtype)])
            sum_pt = np.concatenate([sum_pt, np.zeros((n,) + sum_pt.shape[1:], sum_pt.dtype)])
            nsamples = np.concatenate([nsamples, np.ones(n)])
            event_ids = np.concatenate([event_ids, np.asarray(pad_ids, dtype=np.int64)])
            weights = np.concatenate([weights, np.ones(n) if pad_w is None else pad_w])
        return n

    summary = selection_summary(h5_path)
    if summary is None:
        if np.any(weights != 1.0):
            raise ValueError(f"{h5_path}: events carry event_weight but the file has no "
                             f"selection summary (merge with merge_dp_h5.py --selection)")
        n_empty = n_failed = 0
        if drawn_csv is not None:
            drawn = read_drawn_events(drawn_csv)
            # Drawn events without a group become empty events (they land
            # at the peripheral end of any multiplicity-ranked centrality),
            # except draws whose hydro failed: those are no events at all.
            have = set(int(e) for e in event_ids)
            missing = [i for i, eid in enumerate(drawn["event_id"]) if int(eid) not in have]
            lost = [i for i in missing if drawn["status"][i] == "hydro_failed"]
            n_failed = len(lost)
            if include_empty:
                n_empty = pad(drawn["event_id"][sorted(set(missing) - set(lost))])
        meta.update(event_ids=event_ids, n_empty=n_empty, n_failed=n_failed)
        out_weights = None
    else:
        sel = read_selection(h5_path)
        pos = {int(e): i for i, e in enumerate(sel["event_id"])}
        rows = np.asarray([pos.get(int(e), -1) for e in event_ids])
        if np.any(rows < 0):
            raise ValueError(f"{h5_path}: events {event_ids[rows < 0][:5].tolist()} are not in "
                             f"the selection list")
        if np.any(sel["status"][rows] != "ok"):
            raise ValueError(f"{h5_path}: events with a group are listed as not ok")
        if not np.allclose(weights, sel["weight"][rows], rtol=1e-6, atol=0.0):
            raise ValueError(f"{h5_path}: event_weight attrs differ from the selection list")
        is_failed = sel["status"] == "failed"
        n_failed = int(is_failed.sum())
        factor = np.ones(len(sel["weight"]))
        if n_failed:
            if failed == "error":
                raise ValueError(f"{h5_path}: {n_failed} accepted draws failed; re-run them "
                                 f"(or load with failed='reweight')")
            bins = (summary.get("selection") or {}).get("centrality_bins") or np.arange(0, 101, 5)
            factor = survival_factors(sel["centrality"], sel["weight"], is_failed, bins)
            share = float(sel["weight"][is_failed].sum() / sel["weight"].sum())
            warnings.warn(f"{h5_path.name}: {n_failed} accepted draws failed ({100 * share:.2f}% "
                          f"of the weight); their weight goes to the survivors of their "
                          f"selection bin (largest factor {factor.max():.4f})", stacklevel=2)
        weights = weights * factor[rows]
        n_empty = 0
        if include_empty:
            empty = np.flatnonzero(sel["status"] == "no_surface")
            n_empty = pad(sel["event_id"][empty], sel["weight"][empty] * factor[empty])
        meta.update(event_ids=event_ids, n_empty=n_empty, n_failed=n_failed,
                    sum_weights=float(weights.sum()), max_survival_factor=float(factor.max()))
        out_weights = weights

    out = {
        "Qn": qn,
        "pt_bin_edges": edges,
        "masses": MASSES.copy(),
        "species_names": list(SPECIES_NAMES),
        "pdg_ids": list(PDG_IDS),
        "nsamples": nsamples,
        "rap_max": RAP_MAX,
        "sum_pt": sum_pt,
        "meta": meta,
    }
    if out_weights is not None:
        out["event_weights"] = out_weights
    return out


# Standard output (format "iebe-music-qn" version 1): species names by PDG id.
STANDARD_FORMAT = "iebe-music-qn"
STANDARD_NAMES = {211: "pi+", -211: "pi-", 321: "K+", -321: "K-", 2212: "p", -2212: "pbar",
                  3122: "Lambda", -3122: "Lambdabar", 3312: "Xi-", -3312: "Xibar+", 3334: "Omega-",
                  -3334: "Omegabar+", 311: "K0", -311: "K0bar", 0: "other_charged"}
STATUS = {"ok": 0, "no_collision": 1, "no_surface": 2, "hydro_failed": 3, "failed": 4}


def standard_variants(h5_path) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        return [v for v in str(f.attrs.get("variants", "")).split(",") if v]


def read_standard_output(h5_path, variant: str | None = None, include_empty: bool = True,
                         failed: str = "reweight") -> dict:
    """Arrays of one standard-output design-point file (module docstring),
    in the 2D container layout.  An importance-sampled file (root attrs
    selection_format 1, selection: the summary) returns ``event_weights``;
    its /drawn lists the accepted draws, and those that failed (status
    hydro_failed or failed) give their weight to the survivors of their
    selection bin (``failed="reweight"``) or refuse the design point
    (``failed="error"``), as for the toolkit files."""
    if failed not in FAILED_MODES:
        raise ValueError(f"failed={failed!r}: one of {FAILED_MODES}")
    h5_path = Path(h5_path)
    summary = selection_summary(h5_path)
    with h5py.File(h5_path, "r") as f:
        if f.attrs.get("format") != STANDARD_FORMAT or int(f.attrs.get("format_version", -1)) != 1:
            raise ValueError(f"{h5_path}: not {STANDARD_FORMAT} format version 1")
        if str(f.attrs.get("mode")) != "2D":
            raise ValueError(f"{h5_path}: mode {f.attrs.get('mode')!r}; this reader takes 2D files")
        names = [v for v in str(f.attrs.get("variants", "")).split(",") if v]
        if variant is None:
            if len(names) != 1:
                raise ValueError(f"{h5_path}: choose a variant of {names}")
            variant = names[0]
        if variant not in names:
            raise ValueError(f"{h5_path}: variant {variant!r} not in {names}")
        grp = f[f"deltaf_{variant}" if f"deltaf_{variant}" in f else "deltaf_default"]
        if str(grp.attrs.get("sums", "")) != "raw over nsamples":
            raise ValueError(f"{h5_path}: sums {grp.attrs.get('sums')!r}, expected 'raw over nsamples'")
        g = f["grid"]
        edges = np.asarray(g["pt_edges"][:], dtype=float)
        pdg = [int(x) for x in g["species_pdg"][:]]
        masses = np.asarray(g["species_mass"][:], dtype=float)
        charge = [int(x) for x in g["species_charge"][:]]
        nmax = [int(x) for x in g["species_nmax"][:]]
        harmonics = np.asarray(g["harmonics"][:], dtype=int)
        if not np.array_equal(harmonics, np.arange(len(harmonics))):
            raise ValueError(f"{h5_path}: harmonics {harmonics.tolist()}, expected 0..n")
        unknown = [p for p in pdg if p not in STANDARD_NAMES]
        if unknown:
            raise ValueError(f"{h5_path}: unknown species {unknown}")
        ev = f["events"]
        status = np.asarray(ev["status"][:], dtype=int)
        event_ids = np.asarray(ev["event_id"][:], dtype=np.int64)
        weights = np.asarray(ev["weight"][:], dtype=float)
        if summary is None and np.any(weights != 1.0):
            raise ValueError(f"{h5_path}: weights other than 1 without a selection")
        if not set(status.tolist()) <= {STATUS["ok"], STATUS["no_surface"]}:
            raise ValueError(f"{h5_path}: /events holds statuses {sorted(set(status.tolist()))}")
        qn = np.asarray(grp["Q"][:], dtype=np.complex128)
        ptsum = np.asarray(grp["ptsum"][:], dtype=float)
        mtsum = np.asarray(grp["mtsum"][:], dtype=float)
        ns = np.asarray(grp["nsamples"][:], dtype=float)
        dy_width = float(grp.attrs["dy_width"])
        dr = f["drawn"]
        drawn_status = np.asarray(dr["status"][:], dtype=int)
        drawn = {"event_id": np.asarray(dr["event_id"][:], dtype=np.int64), "status": drawn_status}
        if summary is not None:
            drawn.update(weight=np.asarray(dr["weight"][:], dtype=float),
                         centrality=np.asarray(dr["centrality"][:], dtype=float))
        attrs = {k: f.attrs[k] for k in ("n_drawn", "n_collided", "n_no_collision", "n_accepted", "b_max")
                 if k in f.attrs}
        hbt = _read_hbt(g, grp, h5_path) if "hbt_num" in grp else None
    empty = status == STATUS["no_surface"]
    if np.any(ns[~empty] <= 0) or np.any(ns[empty] != 0) or np.any(qn[empty] != 0):
        raise ValueError(f"{h5_path}: ok rows need nsamples > 0, no_surface rows nsamples 0 and no sums")
    if hbt is not None:
        if hbt["num"].shape[0] != len(status):
            raise ValueError(f"{h5_path}: HBT sums for {hbt['num'].shape[0]} events, /events has {len(status)}")
        if np.any(hbt["num"][empty] != 0) or np.any(hbt["N"][empty] != 0):
            raise ValueError(f"{h5_path}: no_surface rows with HBT sums")
    for k, n in enumerate(nmax):                 # harmonics above a row's nmax: not stored
        if np.any(qn[:, k, n + 1:, :] != 0):
            raise ValueError(f"{h5_path}: species {pdg[k]} has sums above its nmax {n}")
    keep = np.ones(len(status), bool) if include_empty else ~empty
    ns = np.where(empty, 1.0, ns)
    counts = qn[:, :, 0, :].real
    check_pt_grid(ptsum[:, :, 0, :], counts, edges)
    is_failed = np.isin(drawn_status, (STATUS["hydro_failed"], STATUS["failed"]))
    n_failed = int(is_failed.sum())
    meta = {"source": str(h5_path), "event_ids": event_ids[keep], "n_empty": int(np.sum(empty & keep)),
            "n_failed": n_failed, "format": f"{STANDARD_FORMAT} 1", **{k: float(v) for k, v in attrs.items()}}
    out_weights = None
    if summary is None:
        if n_failed:
            warnings.warn(f"{h5_path.name}: {n_failed} collided draws have no events (hydro failed or "
                          f"lost); they are left out", stacklevel=2)
    else:
        # the accepted draws (/drawn) and the events: every row a draw that
        # ran, with its weight; failed draws give their weight to the
        # survivors of their selection bin
        pos = {int(e): i for i, e in enumerate(drawn["event_id"])}
        rows = np.asarray([pos.get(int(e), -1) for e in event_ids])
        if np.any(rows < 0):
            raise ValueError(f"{h5_path}: events {event_ids[rows < 0][:5].tolist()} are not accepted draws")
        if not np.array_equal(drawn_status[rows], status):
            raise ValueError(f"{h5_path}: /events and /drawn disagree on the events' status")
        if not np.allclose(weights, drawn["weight"][rows], rtol=1e-6, atol=0.0):
            raise ValueError(f"{h5_path}: /events weights differ from /drawn")
        lost = set(np.flatnonzero(~np.isin(drawn_status, (STATUS["ok"], STATUS["no_surface"])) & ~is_failed))
        if lost:
            raise ValueError(f"{h5_path}: accepted draws with status "
                             f"{sorted(set(drawn_status[sorted(lost)].tolist()))}")
        if len(set(rows.tolist())) + n_failed != len(drawn_status):
            raise ValueError(f"{h5_path}: accepted draws that are neither events nor failed")
        factor = np.ones(len(drawn_status))
        if n_failed:
            if failed == "error":
                raise ValueError(f"{h5_path}: {n_failed} accepted draws failed; re-run them "
                                 f"(or load with failed='reweight')")
            bins = (summary.get("selection") or {}).get("centrality_bins") or np.arange(0, 101, 5)
            factor = survival_factors(drawn["centrality"], drawn["weight"], is_failed, bins)
            share = float(drawn["weight"][is_failed].sum() / drawn["weight"].sum())
            warnings.warn(f"{h5_path.name}: {n_failed} accepted draws failed ({100 * share:.2f}% of the "
                          f"weight); their weight goes to the survivors of their selection bin (largest "
                          f"factor {factor.max():.4f})", stacklevel=2)
        weights = weights * factor[rows]
        out_weights = weights[keep]
        meta.update(sum_weights=float(out_weights.sum()), max_survival_factor=float(factor.max()))
    out = {
        "Qn": qn[keep],
        "pt_bin_edges": edges,
        "masses": masses,
        "species_names": [STANDARD_NAMES[p] for p in pdg],
        "pdg_ids": pdg,
        "species_charge": charge,
        "species_nmax": nmax,
        "nsamples": ns[keep],
        "rap_max": dy_width / 2.0,
        "sum_pt": ptsum[keep, :, 0, :],
        "sum_pt2": ptsum[keep, :, 1, :],
        "sum_mT": mtsum[keep],
        "meta": meta,
    }
    if out_weights is not None:
        out["event_weights"] = out_weights
    if hbt is not None:
        out["hbt"] = dict(hbt, num=hbt["num"][keep], pairs=hbt["pairs"][keep], N=hbt["N"][keep])
    return out


HBT_GRID = ("species", "rows", "kt_edges", "phi_edges", "q", "q_weight", "q_directions", "q_magnitudes",
            "q_shell_edges")


def _read_hbt(grid, grp, h5_path) -> dict:
    """The optional HBT sums of a variant (docs/standard_output.md section
    9): the grid entries, hbt_num [E, R, K, F, nq], hbt_pairs [E, R, K, F],
    hbt_N [E, Sp, K, F] and the variant's hbt_y_cut."""
    missing = [k for k in HBT_GRID if f"hbt_{k}" not in grid] + \
        [k for k in ("hbt_pairs", "hbt_N") if k not in grp] + \
        (["hbt_y_cut"] if "hbt_y_cut" not in grp.attrs else [])
    if missing:
        raise ValueError(f"{h5_path}: HBT sums without {missing}")
    out = {k: np.asarray(grid[f"hbt_{k}"][:]) for k in HBT_GRID}
    out["species"] = [int(x) for x in out["species"]]
    out["num"] = np.asarray(grp["hbt_num"][:])
    out["pairs"] = np.asarray(grp["hbt_pairs"][:], dtype=float)
    out["N"] = np.asarray(grp["hbt_N"][:], dtype=float)
    out["y_cut"] = float(grp.attrs["hbt_y_cut"])
    E, R, K, F, nq = out["num"].shape
    if (out["pairs"].shape != (E, R, K, F) or out["N"].shape != (E, len(out["species"]), K, F)
            or out["rows"].shape != (R, len(out["species"])) or out["q"].shape != (nq, 3)
            or len(out["kt_edges"]) != K + 1 or len(out["phi_edges"]) != F + 1):
        raise ValueError(f"{h5_path}: HBT sums with inconsistent shapes")
    return out


class Campaign:
    """One merged campaign directory (the adapter's simulation object)."""

    container = "2d"

    def __init__(self, data_root, package_dir=None, *, xsec_table=None,
                 include_empty=True, failed="reweight"):
        self.data_root = Path(data_root)
        self.package_dir = Path(package_dir) if package_dir is not None else None
        self.include_empty = bool(include_empty)
        if failed not in FAILED_MODES:
            raise ValueError(f"failed={failed!r}: one of {FAILED_MODES}")
        self.failed = failed
        self.xsec_table = None
        if xsec_table:
            p = Path(xsec_table)
            if not p.is_absolute() and self.package_dir is not None:
                p = self.package_dir / p
            if not p.is_file():
                raise FileNotFoundError(f"cross-section table {p}")
            self.xsec_table = p
        self._xsec = None

    def _dir(self, design_point: int) -> Path:
        return self.data_root / f"dp{int(design_point):04d}"

    def _h5(self, design_point: int) -> Path:
        dp = f"dp{int(design_point):04d}"
        return self.data_root / dp / f"{dp}.h5"

    def _standard(self, design_point: int) -> Path:
        dp = f"dp{int(design_point):04d}"
        return self.data_root / dp / f"qn_{dp}.h5"

    def _is_standard(self, design_point: int) -> bool:
        std, old = self._standard(design_point).is_file(), self._h5(design_point).is_file()
        if std and old:
            raise ValueError(f"dp{int(design_point):04d}: both a standard-output file and a toolkit file; "
                             f"move one away")
        return std

    def design_points(self) -> list[int]:
        out = []
        for d in self.data_root.glob("dp[0-9][0-9][0-9][0-9]"):
            if (d / f"{d.name}.h5").is_file() or (d / f"qn_{d.name}.h5").is_file():
                out.append(int(d.name[2:]))
        return sorted(out)

    def source_files(self, design_point: int) -> list[Path]:
        if self._is_standard(design_point):
            return [self._standard(design_point)]
        d = self._dir(design_point)
        extra = [d / "csv.tgz", d / f"selection_dp{int(design_point):04d}.csv"]
        return [self._h5(design_point)] + [p for p in extra if p.is_file()]

    def variants(self, design_point: int) -> list[str]:
        """Delta f models stored for a design point ([] for campaign 1)."""
        if self._is_standard(design_point):
            return standard_variants(self._standard(design_point))
        with h5py.File(self._h5(design_point), "r") as f:
            return models(f[next(iter(f.keys()))])

    def load(self, design_point: int, variant: str | None = None) -> dict:
        if self._is_standard(design_point):
            return read_standard_output(self._standard(design_point), variant, include_empty=self.include_empty,
                                        failed=self.failed)
        h5 = self._h5(design_point)
        if not h5.is_file():
            raise FileNotFoundError(h5)
        csv_tgz = self._dir(design_point) / "csv.tgz"
        return read_design_point(h5, variant,
                                 drawn_csv=csv_tgz if csv_tgz.is_file() else None,
                                 include_empty=self.include_empty, failed=self.failed)

    def _xsec_row(self, design_point: int) -> dict:
        if self._xsec is None:
            with open(self.xsec_table, newline="") as fh:
                self._xsec = {int(r["dp"]): r for r in csv.DictReader(fh)}
        row = self._xsec.get(int(design_point))
        if row is None:
            raise KeyError(f"{self.xsec_table}: no row for design point {design_point}")
        return row

    def anchor(self, design_point: int):
        """``{"b_max", "sigma_tot"}`` (f = pi b_max^2 / sigma_inel), or None without a table.

        Needs the drawn-event table: without it the empty draws cannot be
        added and the coverage is unknown.

        An importance-sampled design point: ``{"n_ref": n_collided /
        coverage}`` from its selection summary (the exact count of collided
        draws, over the fraction of sigma_inel that b_max covers), with
        ``covered_cent`` when the selection accepts nothing beyond some
        centrality; a cross-section table, if given, must agree on b_max.
        """
        if self._is_standard(design_point):
            return self._standard_anchor(design_point)
        h5 = self._h5(design_point)
        summary = selection_summary(h5) if h5.is_file() else None
        if summary is not None:
            return self._selection_anchor(design_point, summary)
        if self.xsec_table is None:
            return None
        if not self.include_empty:
            raise ValueError("anchor() assumes the empty draws are loaded "
                             "(include_empty=True)")
        row = self._xsec_row(design_point)
        b_max = float(row["b_max"])
        sigma = float(row["sigma_inel_fm2"])
        csv_tgz = self._dir(design_point) / "csv.tgz"
        if not csv_tgz.is_file():
            raise FileNotFoundError(
                f"{csv_tgz}: no drawn-event table, so the empty draws are "
                f"missing and the cross-section coverage is unknown")
        drawn = np.unique(read_drawn_events(csv_tgz)["b_max"])
        if len(drawn) != 1 or abs(drawn[0] - b_max) > 1e-4 * b_max:
            raise ValueError(f"dp{int(design_point):04d}: campaign b_max {drawn} "
                             f"!= {b_max} in {self.xsec_table}")
        return {"b_max": b_max, "sigma_tot": sigma}

    def _standard_anchor(self, design_point: int):
        """Standard output, minimum bias: n_ref = the collided events loaded
        (ok and no_surface rows) over the fraction of sigma_inel below b_max
        (the table's frac_below_bmax), b_max checked against the file's.
        Importance-sampled: the selection's anchor (_selection_anchor)."""
        summary = selection_summary(self._standard(design_point))
        if summary is not None:
            return self._selection_anchor(design_point, summary)
        if self.xsec_table is None:
            return None
        if not self.include_empty:
            raise ValueError("anchor() assumes the empty (no_surface) events are loaded (include_empty=True)")
        dp = f"dp{int(design_point):04d}"
        row = self._xsec_row(design_point)
        b_max = float(row["b_max"])
        frac = row.get("frac_below_bmax")
        if frac in (None, ""):
            raise ValueError(f"{self.xsec_table}: no frac_below_bmax for {dp}")
        with h5py.File(self._standard(design_point), "r") as f:
            file_b_max = float(f.attrs["b_max"])
            n_rows = int(f["events"]["event_id"].shape[0])
        if abs(file_b_max - b_max) > 1e-4 * b_max:
            raise ValueError(f"{dp}: campaign b_max {file_b_max} != {b_max} in {self.xsec_table}")
        return {"n_ref": n_rows / float(frac)}

    def _selection_anchor(self, design_point: int, summary: dict) -> dict:
        if not self.include_empty:
            raise ValueError("anchor() assumes the empty (no_surface) events are loaded "
                             "(include_empty=True)")
        dp = f"dp{int(design_point):04d}"
        coverage = float(summary.get("coverage", 1.0))
        n_collided = float(summary["n_collided"])
        if not (0 < coverage <= 1 and n_collided > 0):
            raise ValueError(f"{dp}: selection summary coverage {coverage}, n_collided {n_collided}")
        lo, hi = (float(c) for c in summary.get("covered_centrality", (0.0, 100.0)))
        if lo > 0:
            raise ValueError(f"{dp}: the selection accepts no events below {lo:g}% centrality, "
                             f"so the percentiles of the others are unknown")
        if self.xsec_table is not None:
            row = self._xsec_row(design_point)
            b_max = float(row["b_max"])
            if abs(float(summary["b_max"]) - b_max) > 1e-4 * b_max:
                raise ValueError(f"{dp}: campaign b_max {summary['b_max']} != {b_max} in "
                                 f"{self.xsec_table}")
            # the fraction of sigma_inel below b_max sets n_ref: a selection
            # made with another coverage (select_events.py --coverage) puts
            # every percentile off by their ratio
            frac = row.get("frac_below_bmax")
            if frac not in (None, "") and abs(float(frac) - coverage) > COVERAGE_TOLERANCE:
                raise ValueError(
                    f"{dp}: the selection was made with coverage {coverage:g}, but b_max "
                    f"{b_max:g} fm covers {float(frac):.4f} of sigma_inel ({self.xsec_table.name} "
                    f"frac_below_bmax): every centrality would be off by {float(frac) / coverage:.3f}; "
                    f"redo the selection with select_events.py --coverage {float(frac):.4f}")
        spec = {"n_ref": n_collided / coverage}
        # covered_centrality ends at min(acceptance range, 100 x coverage)
        # (chain 1028d08): only an acceptance cut inside the drawn range is a
        # cut of the selection; the end of the b_max coverage is the anchor's
        # own coverage (sum w / n_ref), whose straddling classes are flagged
        # partial as in unweighted campaigns.
        if hi < 100.0 * coverage - 1e-3:
            spec["covered_cent"] = hi
        return spec


def open_simulation(data_root, package_dir=None, **options) -> Campaign:
    """Adapter entry point (soft-sector simulation-package API 1)."""
    return Campaign(data_root, package_dir, **options)
