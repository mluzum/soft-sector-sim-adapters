"""Adapter for merged iEBE-MUSIC campaigns (2D boost-invariant).

Reads the merged per-design-point files written by the GeSe campaign
pipeline (iEBE-MUSIC fork, github.com/mluzum/iEBE-MUSIC: TRENTo -> free
streaming -> MUSIC -> iSS -> decays -> hadronic_afterburner_toolkit) and
returns the arrays of the soft-sector pipeline's 2D container.

Layout of ``<data_root>``::

    dp<NNNN>/dp<NNNN>.h5   one group per event, ``spvn_results_<event_id>``
    dp<NNNN>/csv.tgz       drawn-event table(s): event_id, b, b_max,
                           b_over_bmax, mult, status (ok / no_collision)

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

Drawn events without a fluid (no collision, or hit without freeze-out) have
no group; with ``include_empty`` they are appended as empty events, so the
loaded sample is every draw with b < b_max and covers the fraction
f = pi b_max^2 / sigma_inel of the inelastic cross section.  ``anchor``
returns that from the per-design-point cross-section table (columns
``dp``, ``b_max``, ``sigma_inel_fm2``), after checking that the table's
b_max is the one the campaign drew with.
"""

from __future__ import annotations

import csv
import io
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np

API_VERSION = 1
__version__ = "1.0.0"

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
    """The drawn-event table of ``csv.tgz`` (event_id, b, b_max, b_over_bmax, mult, status)."""
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

    def col(name):
        return np.asarray([float(r[name]) for r in rows], dtype=float)

    return {
        "event_id": np.asarray([int(r["event_id"]) for r in rows], dtype=np.int64),
        "b": col("b"),
        "b_max": col("b_max"),
        "b_over_bmax": col("b_over_bmax"),
        "mult": col("mult"),
        "status": np.asarray([r["status"] for r in rows]),
    }


def read_design_point(h5_path, variant: str | None = None, drawn_csv=None,
                      include_empty: bool = True) -> dict:
    """Arrays of one merged design-point file, in the 2D container layout."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=_event_id)
        if not keys:
            raise ValueError(f"{h5_path}: no events")
        edges = pt_edges(f[keys[0]])
        qn_list, spt_list, ns_list, ids = [], [], [], []
        for k in keys:
            if not np.array_equal(pt_edges(f[k]), edges):
                raise ValueError(f"{h5_path}: event {k} has a different pT grid")
            qn, spt, ns = read_event(flow_group(f[k], variant), len(edges) - 1)
            qn_list.append(qn)
            spt_list.append(spt)
            ns_list.append(ns)
            ids.append(_event_id(k))

    qn = np.stack(qn_list, axis=0)
    sum_pt = np.stack(spt_list, axis=0)
    nsamples = np.asarray(ns_list, dtype=float)
    event_ids = np.asarray(ids, dtype=np.int64)
    check_pt_grid(sum_pt, qn[:, :, 0, :].real, edges)

    n_empty = 0
    if drawn_csv is not None and include_empty:
        drawn_ids = read_drawn_events(drawn_csv)["event_id"]
        # Drawn events without a group become empty events; they land at
        # the peripheral end of any multiplicity-ranked centrality.
        have = set(int(e) for e in event_ids)
        pad = [i for i, eid in enumerate(drawn_ids) if int(eid) not in have]
        n_empty = len(pad)
        if pad:
            qn = np.concatenate([qn, np.zeros((n_empty,) + qn.shape[1:], qn.dtype)])
            sum_pt = np.concatenate(
                [sum_pt, np.zeros((n_empty,) + sum_pt.shape[1:], sum_pt.dtype)])
            nsamples = np.concatenate([nsamples, np.ones(n_empty)])
            event_ids = np.concatenate([event_ids, drawn_ids[pad]])

    return {
        "Qn": qn,
        "pt_bin_edges": edges,
        "masses": MASSES.copy(),
        "species_names": list(SPECIES_NAMES),
        "pdg_ids": list(PDG_IDS),
        "nsamples": nsamples,
        "rap_max": RAP_MAX,
        "sum_pt": sum_pt,
        "meta": {"event_ids": event_ids, "n_empty": n_empty,
                 "source": str(h5_path)},
    }


class Campaign:
    """One merged campaign directory (the adapter's simulation object)."""

    container = "2d"

    def __init__(self, data_root, package_dir=None, *, xsec_table=None,
                 include_empty=True):
        self.data_root = Path(data_root)
        self.package_dir = Path(package_dir) if package_dir is not None else None
        self.include_empty = bool(include_empty)
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
        csv_tgz = self._dir(design_point) / "csv.tgz"
        return [self._h5(design_point)] + ([csv_tgz] if csv_tgz.is_file() else [])

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
                                 include_empty=self.include_empty)

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
        """
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


def open_simulation(data_root, package_dir=None, **options) -> Campaign:
    """Adapter entry point (soft-sector simulation-package API 1)."""
    return Campaign(data_root, package_dir, **options)
