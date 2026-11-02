"""Tests of the merged iEBE-MUSIC adapter on synthetic campaign directories.

Run with ``python -m pytest tests`` or ``python tests/test_iebe_music.py``.
"""

import io
import math
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_adapters import iebe_music as im  # noqa: E402

EDGES = np.linspace(0.0, 5.0, 101)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])


def _tables(n_pt, centres, scale=1.0, count=10.0, nsamp=4):
    diff = np.zeros((n_pt, 13), dtype=np.float32)
    diff[:, 0] = centres
    diff[:, 4] = 0.05 * scale          # Re Q2 per particle
    diff[:, 7] = -0.02 * scale         # Im Q3 per particle
    diff[:, 12] = count
    integ = np.zeros((7, 5), dtype=np.float32)
    integ[6] = [99, 1000, nsamp, 0, 0]
    return diff, integ


def _write(g, diff, integ):
    for pid in im.PDG_IDS:
        g[f"particle_{pid}_vndata_diff_y_-30_30_boostinv.dat"] = diff
        g[f"particle_{pid}_vndata_y_-30_30_boostinv.dat"] = integ


def _drawn(path, ids, b_max, not_run=()):
    text = "event_id,b,b_max,b_over_bmax,mult,status,seed_add\n"
    text += "".join(f"{i},{0.1 * i},{b_max},{0.1 * i / b_max},1.0,ok,0\n" for i in ids)
    text += "".join(f"{i},,{b_max},,,no_trento,0\n" for i in not_run)
    data = text.encode()
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("events_drawn_0.csv")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))


def _campaign2(root, dp=7, events=(3, 7), drawn=(3, 5, 7, 9), b_max=12.0, not_run=(11,)):
    d = Path(root) / f"dp{dp:04d}"
    d.mkdir(parents=True)
    with h5py.File(d / f"dp{dp:04d}.h5", "w") as f:
        for ev in events:
            g = f.create_group(f"spvn_results_{ev}")
            g.attrs["toolkit_pT_min"] = 0.0
            g.attrs["toolkit_pT_max"] = 5.0
            g.attrs["toolkit_npT"] = 100
            g.attrs["deltaf_models"] = np.bytes_("CE,PTB")
            _write(g.create_group("deltaf_CE"), *_tables(100, CENTRES, 1.0))
            _write(g.create_group("deltaf_PTB"), *_tables(100, CENTRES, 2.0))
    _drawn(d / "csv.tgz", drawn, b_max, not_run)


def test_campaign2_layout_and_padding():
    with tempfile.TemporaryDirectory() as tmp:
        _campaign2(tmp)
        sim = im.open_simulation(tmp)
        assert sim.design_points() == [7]
        assert sim.variants(7) == ["CE", "PTB"]
        a = sim.load(7, "PTB")
        assert np.allclose(a["pt_bin_edges"], EDGES)
        assert a["Qn"].shape == (4, 13, 6, 100) and a["Qn"].dtype == np.complex128
        # two events with a fluid, two empty draws appended after them; the
        # no_trento row (event 11: TRENTo did not run) is not a draw
        assert list(a["meta"]["event_ids"]) == [3, 7, 5, 9]
        assert im.read_drawn_events(Path(tmp) / "dp0007" / "csv.tgz")["n_not_run"] == 1
        assert a["meta"]["n_empty"] == 2
        assert np.allclose(a["Qn"][:2, :, 2, :], 10.0 * 0.05 * 2.0)
        assert np.allclose(a["Qn"][:2, :, 3, :], 10.0 * -0.02j * 2.0)
        assert np.all(a["Qn"][2:] == 0) and np.all(a["sum_pt"][2:] == 0)
        assert list(a["nsamples"]) == [4, 4, 1, 1]
        assert np.allclose(a["sum_pt"][:2], 10.0 * CENTRES.astype(np.float32))
        assert a["rap_max"] == 5.0 and a["pdg_ids"][:2] == [211, -211]
        ce = sim.load(7, "CE")
        assert np.allclose(ce["Qn"][:2, :, 2, :], 10.0 * 0.05)
        for bad in (None, "Grad22"):
            try:
                sim.load(7, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"variant {bad!r} must be refused")
        no_pad = im.open_simulation(tmp, include_empty=False).load(7, "CE")
        assert no_pad["Qn"].shape[0] == 2
        names = [p.name for p in sim.source_files(7)]
        assert names == ["dp0007.h5", "csv.tgz"]


def test_campaign1_flat_layout():
    legacy = im.LEGACY_PT_EDGES
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "dp0001"
        d.mkdir()
        with h5py.File(d / "dp0001.h5", "w") as f:
            _write(f.create_group("spvn_results_1"),
                   *_tables(41, 0.5 * (legacy[:-1] + legacy[1:]), count=5.0, nsamp=2))
        sim = im.open_simulation(tmp)
        a = sim.load(1, None)
        assert np.allclose(a["pt_bin_edges"], legacy) and a["Qn"].shape == (1, 13, 6, 41)
        assert sim.variants(1) == []
        try:
            sim.load(1, "CE")
        except ValueError:
            pass
        else:
            raise AssertionError("a variant on a flat file must be refused")


def test_pt_grid_check():
    counts = np.full((2, 100), 100.0)
    im.check_pt_grid(counts * (CENTRES - 0.002), counts, EDGES)
    try:
        im.check_pt_grid(counts * (CENTRES - 0.002), counts, np.linspace(0.05, 4.05, 101))
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong grid must be rejected")


def test_anchor():
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg"
        pkg.mkdir()
        (pkg / "xsec.csv").write_text("campaign,dp,b_max,sigma_inel_fm2\n"
                                      "a96,7,12.0,500.0\na96,8,11.0,450.0\n")
        data = Path(tmp) / "merged"
        _campaign2(data, dp=7, b_max=12.0)
        _campaign2(data, dp=8, b_max=11.5)
        sim = im.open_simulation(data, pkg, xsec_table="xsec.csv")
        spec = sim.anchor(7)
        assert spec == {"b_max": 12.0, "sigma_tot": 500.0}
        assert abs(math.pi * spec["b_max"] ** 2 / spec["sigma_tot"] - math.pi * 144 / 500) < 1e-12
        for dp, err in ((8, ValueError), (9, KeyError)):
            try:
                sim.anchor(dp)
            except err:
                pass
            else:
                raise AssertionError(f"dp {dp} must raise {err.__name__}")
        os.remove(data / "dp0007" / "csv.tgz")
        try:
            sim.anchor(7)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("anchor without the drawn-event table must fail")
        assert im.open_simulation(data, pkg).anchor(7) is None


def test_real_files_if_present():
    """pT grid of real campaign files.

    Set IEBE_MUSIC_TEST_DATA to merged campaign directories
    (os.pathsep-separated); skipped otherwise.
    """
    roots = [r for r in os.environ.get("IEBE_MUSIC_TEST_DATA", "").split(os.pathsep) if r]
    for root in roots:
        h5 = Path(root) / "dp0001" / "dp0001.h5"
        if not h5.is_file():
            continue
        with h5py.File(h5, "r") as f:
            keys = list(f.keys())[:30]
            edges = im.pt_edges(f[keys[0]])
            have = im.models(f[keys[0]])
            variant = have[0] if have else None
            ev = [im.read_event(im.flow_group(f[k], variant), len(edges) - 1) for k in keys]
        im.check_pt_grid(np.stack([e[1] for e in ev]),
                         np.stack([e[0][:, 0, :].real for e in ev]), edges)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
