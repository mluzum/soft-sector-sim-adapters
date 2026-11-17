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
__version__ = "1.1.0"

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

    def design_points(self) -> list[int]:
        out = []
        for d in self.data_root.glob("dp[0-9][0-9][0-9][0-9]"):
            if (d / f"{d.name}.h5").is_file():
                out.append(int(d.name[2:]))
        return sorted(out)

    def source_files(self, design_point: int) -> list[Path]:
        d = self._dir(design_point)
        extra = [d / "csv.tgz", d / f"selection_dp{int(design_point):04d}.csv"]
        return [self._h5(design_point)] + [p for p in extra if p.is_file()]

    def variants(self, design_point: int) -> list[str]:
        """Delta f models stored for a design point ([] for campaign 1)."""
        with h5py.File(self._h5(design_point), "r") as f:
            return models(f[next(iter(f.keys()))])

    def load(self, design_point: int, variant: str | None = None) -> dict:
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
