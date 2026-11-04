"""Adapter for JETSCAPE QnVector final-state files (3D: rapidity x pT grids).

Reads the text files written by the JETSCAPE framework's QnVector
final-state writer directly (no conversion step) and returns the arrays of
the soft-sector pipeline's 3D container, one grid per identified species
plus the combined charged grid.

File format (one hydro event per file, e.g. ``final_output_OO_12321059_17.dat``)::

    #	JETSCAPE_FINAL_STATE	QnVector
    	# PID of Charged particle 9999 	 y(Charged) = pseudo-rapidity 	 y(PID) = rapidity
    #	pTmin	0	pTmax	6	NpT	30	rapmin	-5	rapmax	5	Nrap	50	Norder	4
    #	PID of Charged particle 9999 	 y(Charged) = pseudo-rapidity 	 y(PID) = rapidity
    #	pid	pT	pT_err	y	y_err	ET	dNdpTdy	dNdpTdy_err	vncos	vncos_err	vnsin	vnsin_err	dN
    #	Event	1
    <NpT x Nrap rows per species>
    #	Event	1 End

The grid is dense: one row per (species, pT bin, rapidity bin), species in
contiguous blocks, pT the outer and rapidity the inner index; empty cells
hold the bin centres and zeros.  Columns: pid, mean pT, its error, mean
rapidity, its error, ET, dN/dpTdy, its error, then for n = 1..Norder-1
the four columns <cos n phi>, error, <sin n phi>, error, and last the raw
count dN summed over the oversamples of the event.  Established on the
O+O set (resultsOO_QVec_tau03_big_pos, November 2026) and checked on every
file read:

- ``dN/dpTdy = dN / (N_os * dpT * dy)`` exactly, so the number of
  oversamples ``N_os`` (500 there) is read from each file;
- ``<cos n phi>``, ``<sin n phi>`` are per-particle averages, so the
  accumulated Q-vector is ``Q_n = dN (<cos n phi> + i <sin n phi>)``;
- ``ET = sum(E sin theta) / N_os``, with E sin theta for every species
  (no baryon mass correction);
- pid 9999 (all charged hadrons) is binned in pseudorapidity, identified
  species in rapidity (the header says so; it is checked).

The 9999 grid becomes the ``combined`` grid (``eta_edges``,
``positives_only`` False: both charges); identified species become
rapidity-binned species grids (``y_edges``; analysis repository >=
8daeca5b0).  ``Qn``, ``sum_pt`` and ``sum_ET`` are sums over the
oversamples and ``nsamples`` their number, as the container expects.

Sources.  A design point's events are either a directory of event files
or a tar archive of them (``.tar``, ``.tar.gz``, ``.tgz``, ``.gz.tar``),
read as a stream.  Individual event files may be gzipped (``.dat.gz``).
On a disk array an archive reads several times faster than loose files
(fewer, larger reads, and gzip shrinks this format about 4x).

Events.  The event id is the last integer of the file name and is passed
on as ``meta['event_ids']`` (jackknife blocks follow it).  Files that are
incomplete (no ``End`` marker) or malformed are skipped with a warning and
listed in ``meta['skipped_files']``.  With ``expected_events`` (the number
of drawn events, ids 0..N-1) the ids without a file are listed in
``meta['missing_event_ids']``; ``missing="empty"`` loads them as empty
events (genuine collisions without particles), ``missing="drop"`` (the
default) leaves them out (failed runs).

Command line, for a quick look at a source::

    python -m sim_adapters.jetscape_qnvector <directory or archive> [--max-events N]
"""

from __future__ import annotations

import csv
import gzip
import os
import re
import sys
import tarfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

API_VERSION = 1
__version__ = "1.0.0"

FORMAT_TAG = "JETSCAPE_FINAL_STATE"
CHARGED_PID = 9999
COLUMNS = ["pid", "pT", "pT_err", "y", "y_err", "ET", "dNdpTdy", "dNdpTdy_err",
           "vncos", "vncos_err", "vnsin", "vnsin_err", "dN"]

# Identified species loaded by default: the ones curated records use.
# 22 (decay photons), 111 and 2112 are in the files but opt-in.
DEFAULT_SPECIES = (211, -211, 321, -321, 2212, -2212)

EVENT_FILE_RE = r"final_output_.*?(\d+)\.dat(\.gz)?$"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".gz.tar", ".tar")

# Column indices of the fixed part.
_C_PID, _C_PT, _C_PTERR, _C_Y, _C_YERR, _C_ET, _C_DNDPTDY = range(7)
_C_V0 = 8                      # first <cos n phi> column


class FormatError(ValueError):
    """A file that is not a well-formed QnVector event file."""


# ── one event file ──────────────────────────────────────────────────────────

def parse_header(lines: list[str]) -> dict:
    """Grid and layout of an event file from its leading comment lines."""
    grid = None
    names = None
    eta_charged = False
    for ln in lines:
        tok = ln.replace("#", " ").split()
        if not tok:
            continue
        if "pTmin" in tok:
            kv = dict(zip(tok[0::2], tok[1::2]))
            try:
                grid = {"pt_min": float(kv["pTmin"]), "pt_max": float(kv["pTmax"]),
                        "n_pt": int(kv["NpT"]), "rap_min": float(kv["rapmin"]),
                        "rap_max": float(kv["rapmax"]), "n_rap": int(kv["Nrap"]),
                        "n_order": int(kv["Norder"])}
            except (KeyError, ValueError) as e:
                raise FormatError(f"grid line {ln.strip()!r}: {e}") from None
        elif tok[0] == "pid":
            names = tok
        elif "pseudo-rapidity" in ln and str(CHARGED_PID) in ln:
            eta_charged = True
    if not lines or FORMAT_TAG not in lines[0] or "QnVector" not in lines[0]:
        raise FormatError(f"not a {FORMAT_TAG} QnVector file (first line {lines[0][:60]!r})"
                          if lines else "empty file")
    if grid is None:
        raise FormatError("no pTmin/pTmax/NpT/rapmin/rapmax/Nrap/Norder line")
    if names != COLUMNS:
        raise FormatError(f"column line {names} != {COLUMNS}")
    if not eta_charged:
        raise FormatError("header does not state that pid 9999 is binned in pseudo-rapidity")
    grid["n_harm"] = grid["n_order"] - 1          # harmonics 1..Norder-1
    grid["n_col"] = 8 + 4 * grid["n_harm"] + 1
    return grid


def grid_key(grid: dict) -> tuple:
    return tuple(grid[k] for k in ("pt_min", "pt_max", "n_pt", "rap_min",
                                   "rap_max", "n_rap", "n_order"))


def pt_edges(grid: dict) -> np.ndarray:
    return np.linspace(grid["pt_min"], grid["pt_max"], grid["n_pt"] + 1)


def rap_edges(grid: dict) -> np.ndarray:
    return np.linspace(grid["rap_min"], grid["rap_max"], grid["n_rap"] + 1)


def _decode(raw: bytes, name: str) -> str:
    if name.endswith(".gz") or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("ascii")


def parse_event(raw: bytes, name: str = "<event>", species=None) -> dict:
    """Arrays of one event file (bytes of the file, gzipped or not).

    Returns ``grid`` (header), ``nsamples`` (NaN for an event without
    particles, whose normalization cannot be read), ``pids`` (file order),
    and per requested PDG (all when ``species`` is None) a dict of float32
    arrays on (n_rap, n_pt): ``dN``, ``mean_pt``, ``ET`` (the file's
    per-oversample value), and ``cos``, ``sin`` (n_rap, n_pt, n_harm).
    Raises :class:`FormatError` on anything unexpected.
    """
    text = _decode(raw, name)
    head = text[:4096].splitlines()[:8]
    grid = parse_header(head)
    starts = [m.start() for m in re.finditer(r"^#\s*Event\s+\d+\s*$", text, re.M)]
    if len(starts) != 1:
        raise FormatError(f"{name}: {len(starts)} 'Event' blocks (expected 1)")
    end = re.search(r"^#\s*Event\s+\d+\s+End\s*$", text, re.M)
    if end is None:
        raise FormatError(f"{name}: no 'Event ... End' line (incomplete file)")
    body = text[text.index("\n", starts[0]) + 1:end.start()]
    vals = np.array(body.split(), dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise FormatError(f"{name}: non-finite values")
    n_pt, n_rap, n_col, n_h = grid["n_pt"], grid["n_rap"], grid["n_col"], grid["n_harm"]
    cells = n_pt * n_rap
    if vals.size % (n_col * cells):
        raise FormatError(f"{name}: {vals.size} numbers is not a whole number of "
                          f"{n_pt}x{n_rap} species blocks of {n_col} columns")
    blocks = vals.reshape(-1, n_pt, n_rap, n_col).transpose(0, 2, 1, 3)   # (sp, rap, pt, col)
    pids = blocks[:, 0, 0, _C_PID].astype(int)
    if np.any(blocks[..., _C_PID] != pids[:, None, None]):
        raise FormatError(f"{name}: pid changes inside a species block")
    if len(set(pids.tolist())) != len(pids):
        raise FormatError(f"{name}: repeated species blocks {pids.tolist()}")

    dn = blocks[..., -1]
    if np.any(dn < 0) or np.any(dn != np.round(dn)):
        raise FormatError(f"{name}: dN not a non-negative integer")
    filled = dn > 0
    # Cell positions: the mean pT and rapidity of filled cells lie in their bins
    # (catches a wrong row order or grid); the values carry 6 significant digits.
    pe, ye = pt_edges(grid), rap_edges(grid)
    tol = 1e-5 * max(abs(grid["pt_max"]), abs(grid["rap_max"]), 1.0)
    ipt = np.broadcast_to(np.arange(n_pt)[None, None, :], dn.shape)
    iy = np.broadcast_to(np.arange(n_rap)[None, :, None], dn.shape)
    pt_v, y_v = blocks[..., _C_PT], blocks[..., _C_Y]
    bad = filled & ((pt_v < pe[ipt] - tol) | (pt_v > pe[ipt + 1] + tol)
                    | (y_v < ye[iy] - tol) | (y_v > ye[iy + 1] + tol))
    if bad.any():
        s, r, p = (int(a[0]) for a in np.nonzero(bad))
        raise FormatError(f"{name}: pid {pids[s]} cell (y bin {r}, pT bin {p}) has mean "
                          f"pT {pt_v[s, r, p]}, y {y_v[s, r, p]} outside the bin")
    # Number of oversamples from the yield normalization.
    nsamples = np.nan
    if filled.any():
        dndpt = blocks[..., _C_DNDPTDY][filled]
        if np.any(dndpt <= 0):
            raise FormatError(f"{name}: dN > 0 with dN/dpTdy <= 0")
        width = (pe[1] - pe[0]) * (ye[1] - ye[0])
        ratio = dn[filled] / dndpt / width
        ns = float(np.median(ratio))
        if np.max(np.abs(ratio / ns - 1)) > 1e-3 or abs(ns - round(ns)) > 1e-3 * ns:
            raise FormatError(f"{name}: dN / (dN/dpTdy dpT dy) is not one integer "
                              f"(range {ratio.min():.4f}-{ratio.max():.4f})")
        nsamples = float(round(ns))

    want = pids.tolist() if species is None else [int(p) for p in species]
    out = {"grid": grid, "nsamples": nsamples, "pids": pids.tolist(), "species": {}}
    pos = {int(p): i for i, p in enumerate(pids)}
    for p in want:
        if p not in pos:
            raise FormatError(f"{name}: no pid {p} (file has {pids.tolist()})")
        b = blocks[pos[p]]
        out["species"][p] = {
            "dN": b[..., -1].astype(np.float32),
            "mean_pt": b[..., _C_PT].astype(np.float32),
            "ET": b[..., _C_ET].astype(np.float32),
            "cos": b[..., _C_V0:_C_V0 + 4 * n_h:4].astype(np.float32),
            "sin": b[..., _C_V0 + 2:_C_V0 + 4 * n_h:4].astype(np.float32),
        }
    return out


def read_event_file(path, species=None) -> dict:
    path = Path(path)
    return parse_event(path.read_bytes(), path.name, species)


# ── sources: a directory or an archive of event files ───────────────────────

def is_archive(path) -> bool:
    return Path(path).is_file() and any(str(path).endswith(s) for s in ARCHIVE_SUFFIXES)


def event_id(name: str, pattern: str = EVENT_FILE_RE) -> int | None:
    m = re.search(pattern, os.path.basename(name))
    return int(m.group(1)) if m else None


def _iter_directory(path: Path, pattern: str):
    """(event id, name, None) for the event files of a directory (read by the worker)."""
    for ent in os.scandir(path):
        if ent.is_file():
            eid = event_id(ent.name, pattern)
            if eid is not None:
                yield eid, ent.path, None


def _iter_archive(path: Path, pattern: str):
    """(event id, member name, bytes) for the event files of an archive, streamed."""
    with tarfile.open(path, "r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            eid = event_id(m.name, pattern)
            if eid is None:
                continue
            yield eid, m.name, tf.extractfile(m).read()


class _Blocks:
    """Event arrays collected in blocks of large arrays, in arrival order.

    Thousands of small per-event arrays would fragment the heap (the peak
    memory of a read was twice the data); blocks are freed back to the
    system as :meth:`assemble` builds the final arrays.
    """

    def __init__(self, size: int = 1024):
        self.size = size
        self.store = None
        self.ids, self.ns = [], []

    def add(self, eid: int, ev: dict):
        sp = ev["species"]
        if self.store is None:
            self.store = {p: {f: [] for f in d} for p, d in sp.items()}
        k = len(self.ids) % self.size
        for p, d in sp.items():
            for f, a in d.items():
                blocks = self.store[p][f]
                if k == 0:
                    blocks.append(np.empty((self.size,) + a.shape, a.dtype))
                blocks[-1][k] = a
        self.ids.append(int(eid))
        self.ns.append(ev["nsamples"])

    def assemble(self):
        """(sorted event ids, nsamples, {pdg: {field: (n_ev, ...) array}})."""
        ids = np.asarray(self.ids, dtype=np.int64)
        n = len(ids)
        order = np.argsort(ids, kind="stable")
        dest = np.empty(n, dtype=np.int64)
        dest[order] = np.arange(n)          # arrival index -> sorted position
        out = {}
        for p in list(self.store):
            out[p] = {}
            for f in list(self.store[p]):
                blocks = self.store[p].pop(f)
                full = np.empty((n,) + blocks[0].shape[1:], blocks[0].dtype)
                for b in range(len(blocks)):
                    lo = b * self.size
                    hi = min(n, lo + self.size)
                    full[dest[lo:hi]] = blocks[b][:hi - lo]
                    blocks[b] = None
                out[p][f] = full
            del self.store[p]
        return ids[order], np.asarray(self.ns, dtype=float)[order], out


def _worker(args):
    eid, name, raw, species = args
    try:
        if raw is None:
            with open(name, "rb") as fh:
                raw = fh.read()
        ev = parse_event(raw, os.path.basename(name), species)
        return eid, name, ev, None
    except (FormatError, ValueError, OSError, EOFError, UnicodeDecodeError) as e:
        return eid, name, None, f"{type(e).__name__}: {e}"


def read_source(source, species=DEFAULT_SPECIES, *, combined=True, workers=None,
                pattern: str = EVENT_FILE_RE, max_events: int | None = None,
                progress: bool = False) -> dict:
    """Parse every event file of a directory or archive.

    Returns ``grid``, ``event_ids`` (sorted), ``nsamples``, ``species``
    {pdg: {field: (n_ev, n_rap, n_pt[, n_harm]) float32}} for the requested
    species plus 9999 when ``combined``, and ``skipped`` [(name, reason)].
    """
    source = Path(source)
    want = [int(p) for p in species] + ([CHARGED_PID] if combined else [])
    if len(set(want)) != len(want):
        raise ValueError(f"species listed twice: {want}")
    if is_archive(source):
        items = _iter_archive(source, pattern)
    elif source.is_dir():
        items = _iter_directory(source, pattern)
    else:
        raise FileNotFoundError(f"{source}: neither a directory nor a tar archive")
    workers = workers or min(16, os.cpu_count() or 1)

    blocks, skipped, grid0, seen = _Blocks(), [], None, {}
    max_inflight = 4 * workers

    def collect(res):
        nonlocal grid0
        eid, name, ev, err = res
        if err is not None:
            skipped.append((name, err))
            return
        if eid in seen:
            raise ValueError(f"{source}: event id {eid} in both {seen[eid]} and {name}")
        if grid0 is None:
            grid0 = ev["grid"]
        elif grid_key(ev["grid"]) != grid_key(grid0):
            skipped.append((name, f"grid {grid_key(ev['grid'])} != {grid_key(grid0)}"))
            return
        seen[eid] = name
        blocks.add(eid, ev)

    n_sub = 0
    with ProcessPoolExecutor(workers) as ex:
        pending = []
        for eid, name, raw in items:
            if max_events is not None and n_sub >= max_events:
                break
            pending.append(ex.submit(_worker, (eid, name, raw, want)))
            n_sub += 1
            if len(pending) >= max_inflight:
                collect(pending.pop(0).result())
            if progress and n_sub % 2000 == 0:
                print(f"  {source.name}: {n_sub} files", file=sys.stderr, flush=True)
        for fut in pending:
            collect(fut.result())

    if skipped:
        warnings.warn(f"{source}: skipped {len(skipped)} file(s), e.g. "
                      f"{skipped[0][0]}: {skipped[0][1]}")
    if not seen:
        raise ValueError(f"{source}: no readable event files matching {pattern!r}")
    ids, ns, fields = blocks.assemble()
    return {"grid": grid0, "event_ids": ids, "nsamples": ns, "species": fields,
            "skipped": skipped, "n_files": len(ids) + len(skipped)}


# ── container arrays ────────────────────────────────────────────────────────

def _fill_nsamples(ns: np.ndarray, where: str) -> np.ndarray:
    """Events without particles carry no normalization: give them the common value."""
    ok = np.isfinite(ns)
    if not ok.any():
        raise ValueError(f"{where}: no event with particles, nsamples unknown")
    vals, counts = np.unique(ns[ok], return_counts=True)
    ns = ns.copy()
    ns[~ok] = vals[np.argmax(counts)]
    return ns


def grids_to_container(parsed: dict, dtype=np.complex64, where: str = "source") -> dict:
    """Multi-species 3D container dict from :func:`read_source` output.

    Consumes (empties) ``parsed['species']`` to keep the peak memory low.
    """
    grid = parsed["grid"]
    ns = _fill_nsamples(parsed["nsamples"], where)
    pe, ye = pt_edges(grid), rap_edges(grid)
    sp = parsed["species"]
    if CHARGED_PID in sp:
        cent_mult = sp[CHARGED_PID]["dN"].sum(axis=(1, 2), dtype=np.float64) / ns
    else:     # no combined grid: charged identified species
        cent_mult = sum(sp[p]["dN"].sum(axis=(1, 2), dtype=np.float64)
                        for p in sp if p in (211, -211, 321, -321, 2212, -2212)) / ns
    ns3 = ns[:, None, None]

    real = np.finfo(dtype).dtype            # float32 for complex64

    def one(p):
        # Products are written straight into the output arrays (ufunc out= /
        # dtype=), computed in the same precision as before: no full-size
        # temporaries, whose allocate-and-free churn on a 40k-event set
        # (~0.7 GB each, five per species) drives memory pressure.
        d = sp.pop(p)
        n_ev, n_r, n_p, n_h = d["cos"].shape
        qn = np.empty((n_ev, n_r, n_p, n_h + 1), dtype=dtype)
        dn = d.pop("dN")
        qn[..., 0] = dn
        np.multiply(d.pop("cos"), dn[..., None], out=qn.real[..., 1:], dtype=real)
        np.multiply(d.pop("sin"), dn[..., None], out=qn.imag[..., 1:], dtype=real)
        out = {"Qn": qn, "pt_edges": pe.copy(), "cent_mult": cent_mult, "nsamples": ns,
               "positives_only": False,
               "sum_pt": np.multiply(dn, d.pop("mean_pt"), dtype=np.float64),
               "sum_ET": np.multiply(d.pop("ET"), ns3, dtype=np.float64)}
        out["eta_edges" if p == CHARGED_PID else "y_edges"] = ye.copy()
        return out

    species = {}
    for p in [p for p in list(sp) if p != CHARGED_PID]:
        species[p] = one(p)
    comb = one(CHARGED_PID) if CHARGED_PID in sp else None
    return {"species": species, "combined": comb}


def _pad_empty(c: dict, ids: np.ndarray, missing: np.ndarray) -> tuple[dict, np.ndarray]:
    """Append empty events (``missing`` ids) to every grid, in event-id order.

    The empty events get no particles and the common number of oversamples.
    """
    allids = np.concatenate([ids, missing])
    order = np.argsort(allids, kind="stable")
    n_add = len(missing)
    grids = list(c["species"].values()) + ([c["combined"]] if c["combined"] else [])
    g0 = grids[0]
    ns = np.concatenate([g0["nsamples"], np.full(n_add, np.median(g0["nsamples"]))])[order]
    cm = np.concatenate([g0["cent_mult"], np.zeros(n_add)])[order]
    for g in grids:
        for k in ("Qn", "sum_pt", "sum_ET"):
            v = g[k]
            g[k] = np.concatenate([v, np.zeros((n_add,) + v.shape[1:], v.dtype)])[order]
        g["nsamples"], g["cent_mult"] = ns, cm
    return c, allids[order]


# ── simulation-package adapter ──────────────────────────────────────────────

class Campaign:
    """Design points of QnVector event sets (the adapter's simulation object).

    Options (``simulation.yaml`` ``adapter.options``):

    ``sources``
        Where each design point's events are, relative to ``data_root``:
        a mapping ``{design point: path}``, or a pattern with a ``{dp}``
        field such as ``"dp{dp:04d}"`` or ``"OO_{dp}.tar.gz"``.  A path is
        a directory of event files or an archive; a pattern also matches
        ``<name>`` plus an archive suffix.  Default ``"dp{dp:04d}"``.
    ``species``
        Identified PDG codes to load (default pi+-, K+-, p, pbar); the
        charged grid 9999 is always loaded as the combined grid.
    ``dtype``
        ``complex64`` (default; window sums accumulate in complex128) or
        ``complex128``.
    ``expected_events``
        Number of drawn events per design point (ids 0..N-1), or a mapping
        per design point.  Ids without a file are reported in ``meta``.
    ``missing``
        ``drop`` (default: missing ids are failed runs) or ``empty`` (they
        are collisions without particles and are loaded as empty events;
        needs ``expected_events``).
    ``anchor``
        Centrality anchor of every design point, anything
        ``resolve_anchor`` accepts (e.g. ``{f: 1.0}``), or
    ``anchor_table``
        CSV relative to the package with a ``dp`` column and the anchor
        keys (``f``, ``n_ref``, ``sigma_tot``, ``b_max``, ``sigma_eff``).
    ``workers``
        Parallel reader processes (default min(16, cores)).
    ``file_pattern``
        Regex of event file names; group 1 is the event id.
    ``max_events``
        Read only the first N event files of each source (quick tests);
        missing events are then not reported.
    """

    container = "3d"

    def __init__(self, data_root, package_dir=None, *, sources="dp{dp:04d}",
                 species=DEFAULT_SPECIES, dtype="complex64", expected_events=None,
                 missing="drop", anchor=None, anchor_table=None, workers=None,
                 file_pattern=EVENT_FILE_RE, max_events=None):
        self.data_root = Path(data_root)
        self.package_dir = Path(package_dir) if package_dir is not None else None
        self.sources = sources
        self.species = tuple(int(p) for p in species)
        if CHARGED_PID in self.species:
            raise ValueError("pid 9999 is always loaded as the combined grid; "
                             "leave it out of 'species'")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.complex64, np.complex128):
            raise ValueError(f"dtype must be complex64 or complex128, got {dtype}")
        if missing not in ("drop", "empty"):
            raise ValueError(f"missing must be 'drop' or 'empty', got {missing!r}")
        if missing == "empty" and expected_events is None:
            raise ValueError("missing='empty' needs expected_events")
        self.expected_events = expected_events
        self.missing = missing
        self._anchor = anchor
        self.anchor_table = None
        if anchor_table:
            p = Path(anchor_table)
            if not p.is_absolute() and self.package_dir is not None:
                p = self.package_dir / p
            if not p.is_file():
                raise FileNotFoundError(f"anchor table {p}")
            self.anchor_table = p
        self.workers = workers
        self.file_pattern = file_pattern
        self.max_events = None if max_events is None else int(max_events)

    # design points -----------------------------------------------------------
    def _resolve(self, rel) -> Path | None:
        p = self.data_root / str(rel)
        if p.exists():
            return p
        for s in ARCHIVE_SUFFIXES:
            q = Path(str(p) + s)
            if q.is_file():
                return q
        return None

    def _source_map(self) -> dict[int, Path]:
        if isinstance(self.sources, dict):
            out = {}
            for dp, rel in self.sources.items():
                p = self._resolve(rel)
                if p is None:
                    raise FileNotFoundError(f"design point {dp}: {self.data_root / str(rel)}")
                out[int(dp)] = p
            return out
        pat = str(self.sources)
        parent = self.data_root / os.path.dirname(pat)
        base = os.path.basename(pat)
        if "{dp" not in base:
            raise ValueError(f"sources pattern {pat!r}: the last path component "
                             f"needs a {{dp}} field")
        head, _, rest = base.partition("{dp")
        tail = rest.partition("}")[2]
        rx = re.compile(re.escape(head) + r"(\d+)" + re.escape(tail)
                        + "(" + "|".join(re.escape(s) for s in ARCHIVE_SUFFIXES) + ")?$")
        out = {}
        if parent.is_dir():
            for ent in sorted(os.scandir(parent), key=lambda e: e.name):
                m = rx.match(ent.name)
                if m and (ent.is_dir() or is_archive(ent.path)):
                    dp = int(m.group(1))
                    # a directory wins over its archive (both may exist)
                    if dp not in out or Path(ent.path).is_dir():
                        out[dp] = Path(ent.path)
        return out

    def design_points(self) -> list[int]:
        return sorted(self._source_map())

    def source(self, design_point: int) -> Path:
        m = self._source_map()
        if int(design_point) not in m:
            raise FileNotFoundError(f"no source for design point {design_point} "
                                    f"(sources {self.sources!r} under {self.data_root})")
        return m[int(design_point)]

    def source_files(self, design_point: int) -> list[Path]:
        # The directory itself (its mtime changes when files are added or
        # removed): stat-ing tens of thousands of files would be slow.
        return [self.source(design_point)]

    # loading -----------------------------------------------------------------
    def _expected(self, design_point: int):
        e = self.expected_events
        if isinstance(e, dict):
            e = e.get(int(design_point), e.get(str(design_point)))
        return None if e is None else int(e)

    def load(self, design_point: int, variant=None, *, max_events=None) -> dict:
        if variant is not None:
            raise ValueError(f"this simulation has no variants (got {variant!r})")
        if max_events is None:
            max_events = self.max_events
        src = self.source(design_point)
        parsed = read_source(src, self.species, combined=True, workers=self.workers,
                             pattern=self.file_pattern, max_events=max_events)
        ids = parsed["event_ids"]
        skipped = parsed["skipped"]
        grid = parsed["grid"]
        ns_read = parsed["nsamples"]
        c = grids_to_container(parsed, dtype=self.dtype, where=str(src))
        n_exp = self._expected(design_point)
        missing_ids = np.array([], dtype=np.int64)
        n_empty = 0
        if n_exp is not None and max_events is None:
            if ids.max() >= n_exp:
                raise ValueError(f"{src}: event id {ids.max()} >= expected_events {n_exp}")
            missing_ids = np.setdiff1d(np.arange(n_exp), ids)
            if self.missing == "empty" and missing_ids.size and max_events is None:
                c, ids = _pad_empty(c, ids, missing_ids)
                n_empty = int(missing_ids.size)
        c["meta"] = {
            "event_ids": ids,
            "source": str(src),
            "n_files_read": int(len(parsed["event_ids"])),
            "skipped_files": [f"{n}: {r}" for n, r in skipped],
            "missing_event_ids": missing_ids,
            "missing_policy": self.missing,
            "n_empty": n_empty,
            "nsamples_values": sorted(set(np.unique(ns_read[np.isfinite(ns_read)]).tolist())),
            "n_events_without_particles": int(np.sum(~np.isfinite(ns_read))),
            "grid": dict(grid),
            "harmonics": list(range(1, grid["n_harm"] + 1)),
            "rapidity_axis": {"9999": "eta", "identified": "y"},
            "adapter": f"jetscape_qnvector {__version__}",
        }
        return c

    # centrality anchor -------------------------------------------------------
    def anchor(self, design_point: int):
        if self.anchor_table is not None:
            with open(self.anchor_table, newline="") as fh:
                rows = {int(r["dp"]): r for r in csv.DictReader(fh)}
            r = rows.get(int(design_point))
            if r is None:
                raise KeyError(f"{self.anchor_table}: no row for design point {design_point}")
            return {k: float(v) for k, v in r.items()
                    if k in ("f", "n_ref", "sigma_tot", "b_max", "sigma_eff") and v not in ("", None)}
        return self._anchor


def open_simulation(data_root, package_dir=None, **options) -> Campaign:
    """Adapter entry point (soft-sector simulation-package API 1)."""
    return Campaign(data_root, package_dir, **options)


# ── command line: summary of a source ───────────────────────────────────────

def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="directory of event files or tar archive")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--species", type=int, nargs="*", default=list(DEFAULT_SPECIES))
    a = ap.parse_args(argv)
    import time
    t = time.time()
    p = read_source(a.source, a.species, workers=a.workers, max_events=a.max_events,
                    progress=True)
    dt = time.time() - t
    g = p["grid"]
    ns = p["nsamples"]
    print(f"{a.source}: {len(p['event_ids'])} events read in {dt:.0f} s, "
          f"{len(p['skipped'])} skipped")
    print(f"  grid: pT {g['pt_min']}-{g['pt_max']} GeV in {g['n_pt']} bins, "
          f"rapidity {g['rap_min']}-{g['rap_max']} in {g['n_rap']} bins, "
          f"harmonics 1-{g['n_harm']}")
    print(f"  event ids {p['event_ids'].min()}-{p['event_ids'].max()}, "
          f"nsamples {sorted(set(ns[np.isfinite(ns)].tolist()))}, "
          f"{int(np.sum(~np.isfinite(ns)))} events without particles")
    for pid, d in p["species"].items():
        n = d["dN"].sum(axis=(1, 2), dtype=np.float64) / np.nan_to_num(ns, nan=1.0)
        print(f"  pid {pid:6d}: <dN per oversample> {n.mean():9.2f}  (min {n.min():.1f}, max {n.max():.1f})")
    for name, why in p["skipped"][:10]:
        print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    _main()
