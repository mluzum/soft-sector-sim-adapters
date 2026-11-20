"""Tests of the merged iEBE-MUSIC adapter on synthetic campaign directories.

Run with ``python -m pytest tests`` or ``python tests/test_iebe_music.py``.
"""

import io
import json
import math
import os
import sys
import tarfile
import tempfile
import warnings
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


def _drawn(path, ids, b_max, not_run=(), status=None):
    status = status or {}
    text = "event_id,b,b_max,b_over_bmax,mult,status,seed_add\n"
    text += "".join(f"{i},{0.1 * i},{b_max},{0.1 * i / b_max},1.0,{status.get(i, 'ok')},0\n"
                    for i in ids)
    text += "".join(f"{i},,{b_max},,,no_trento,0\n" for i in not_run)
    data = text.encode()
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("events_drawn_0.csv")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))


def _campaign2(root, dp=7, events=(3, 7), drawn=(3, 5, 7, 9), b_max=12.0, not_run=(11,),
               status=None, weights=None):
    d = Path(root) / f"dp{dp:04d}"
    d.mkdir(parents=True)
    with h5py.File(d / f"dp{dp:04d}.h5", "w") as f:
        for ev in events:
            g = f.create_group(f"spvn_results_{ev}")
            if weights and ev in weights:
                g.attrs["event_weight"] = weights[ev]
            g.attrs["toolkit_pT_min"] = 0.0
            g.attrs["toolkit_pT_max"] = 5.0
            g.attrs["toolkit_npT"] = 100
            g.attrs["deltaf_models"] = np.bytes_("CE,PTB")
            _write(g.create_group("deltaf_CE"), *_tables(100, CENTRES, 1.0))
            _write(g.create_group("deltaf_PTB"), *_tables(100, CENTRES, 2.0))
    _drawn(d / "csv.tgz", drawn, b_max, not_run, status)


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


def test_hydro_failed_draws_are_not_events():
    """A draw whose hydro failed is no empty event (unweighted campaigns)."""
    with tempfile.TemporaryDirectory() as tmp:
        _campaign2(tmp, status={9: "hydro_failed", 5: "no_surface"})
        a = im.open_simulation(tmp).load(7, "CE")
        assert list(a["meta"]["event_ids"]) == [3, 7, 5]
        assert a["meta"]["n_empty"] == 1 and a["meta"]["n_failed"] == 1
        assert "event_weights" not in a


# accepted draws: id -> (weight, initial-state centrality, status)
SELECTED = {3: (1.0, 5.0, "ok"), 7: (4.0, 20.0, "ok"), 12: (4.0, 30.0, "ok"),
            15: (4.0, 45.0, "no_surface"), 20: (4.0, 25.0, "failed"), 30: (10.0, 60.0, "ok")}


def _selected_campaign(root, dp=7, covered=(0.0, 100.0), selected=SELECTED, coverage=0.9):
    events = [e for e, (_, _, st) in selected.items() if st == "ok"]
    _campaign2(root, dp=dp, events=events, drawn=list(selected), not_run=(),
               weights={e: selected[e][0] for e in events})
    d = Path(root) / f"dp{dp:04d}"
    summary = {"format_version": 1, "coverage": coverage, "b_max": 12.0, "n_drawn": 100,
               "n_collided": 90, "n_no_collision": 10, "n_accepted": len(selected),
               "covered_centrality": list(covered),
               "selection": {"mode": "bins", "centrality_bins": [0, 10, 50]}}
    with h5py.File(d / f"dp{dp:04d}.h5", "a") as f:
        f.attrs["selection_format"] = 1
        for k in ("n_drawn", "n_collided", "b_max"):
            f.attrs[k] = summary[k]
        f.attrs["covered_centrality"] = list(covered)
        f.attrs["selection"] = json.dumps(summary)
    rows = "".join(f"{e},{w},{1 / w},{c},1.0,0.5,{st}\n" for e, (w, c, st) in selected.items())
    (d / f"selection_dp{dp:04d}.csv").write_text(
        "event_id,weight,p_acc,centrality,estimator,u,status\n" + rows)


def test_importance_sampled_design_point():
    with tempfile.TemporaryDirectory() as tmp:
        _selected_campaign(tmp)
        sim = im.open_simulation(tmp)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a = sim.load(7, "CE")
        assert any("1 accepted draws failed" in str(w.message) for w in caught)
        # groups in id order, then the no_surface draw as an empty event
        assert list(a["meta"]["event_ids"]) == [3, 7, 12, 30, 15]
        assert a["meta"]["n_empty"] == 1 and a["meta"]["n_failed"] == 1
        assert np.all(a["Qn"][4] == 0) and a["nsamples"][4] == 1
        # the failed draw's weight (4, bin 10-50%) goes to 7, 12 and 15: x 16/12;
        # the draw beyond the last bin edge (60%) is a bin of its own
        assert np.allclose(a["event_weights"], [1.0, 16 / 3, 16 / 3, 10.0, 16 / 3])
        assert abs(a["event_weights"].sum() - sum(w for w, _, _ in SELECTED.values())) < 1e-12
        assert a["meta"]["sum_weights"] == a["event_weights"].sum()
        try:
            im.open_simulation(tmp, failed="error").load(7, "CE")
        except ValueError as e:
            assert "re-run" in str(e)
        else:
            raise AssertionError("failed='error' must refuse a design point with failed draws")
        assert sim.anchor(7) == {"n_ref": 100.0}               # 90 collided / coverage 0.9
        names = [p.name for p in sim.source_files(7)]
        assert names == ["dp0007.h5", "csv.tgz", "selection_dp0007.csv"]
        no_empty = im.open_simulation(tmp, include_empty=False)
        assert len(no_empty.load(7, "CE")["event_weights"]) == 4
        try:
            no_empty.anchor(7)
        except ValueError:
            pass
        else:
            raise AssertionError("an anchor without the no_surface events must fail")


def test_selection_anchor_and_checks():
    with tempfile.TemporaryDirectory() as tmp:
        _selected_campaign(Path(tmp) / "cut", covered=(0.0, 70.0), coverage=1.0)
        assert im.open_simulation(Path(tmp) / "cut").anchor(7) == {"n_ref": 90.0, "covered_cent": 70.0}
        # covered_centrality ends at 100 x coverage without an acceptance cut:
        # that is the b_max coverage, not a cut of the selection
        _selected_campaign(Path(tmp) / "bmax", covered=(0.0, 87.1), coverage=0.871)
        assert im.open_simulation(Path(tmp) / "bmax").anchor(7) == {"n_ref": 90 / 0.871}
        _selected_campaign(Path(tmp) / "both", covered=(0.0, 70.0), coverage=0.871)
        assert im.open_simulation(Path(tmp) / "both").anchor(7) == {"n_ref": 90 / 0.871,
                                                                   "covered_cent": 70.0}
        _selected_campaign(Path(tmp) / "central", covered=(5.0, 100.0))
        try:
            im.open_simulation(Path(tmp) / "central").anchor(7)
        except ValueError as e:
            assert "below 5%" in str(e)
        else:
            raise AssertionError("a selection without the most central events must be refused")
        pkg = Path(tmp) / "pkg"
        pkg.mkdir()
        (pkg / "xsec.csv").write_text("campaign,dp,b_max,sigma_inel_fm2\na,7,11.0,500.0\n")
        try:
            im.open_simulation(Path(tmp) / "cut", pkg, xsec_table="xsec.csv").anchor(7)
        except ValueError as e:
            assert "b_max" in str(e)
        else:
            raise AssertionError("a b_max other than the table's must be refused")
        # the selection's coverage must be the table's fraction below b_max
        (pkg / "frac.csv").write_text("campaign,dp,b_max,sigma_inel_fm2,frac_below_bmax\n"
                                      "a,7,12.0,500.0,0.871\n")
        ok = im.open_simulation(Path(tmp) / "bmax", pkg, xsec_table="frac.csv").anchor(7)
        assert ok == {"n_ref": 90 / 0.871}
        try:
            im.open_simulation(Path(tmp) / "cut", pkg, xsec_table="frac.csv").anchor(7)   # coverage 1.0
        except ValueError as e:
            assert "--coverage 0.8710" in str(e)
        else:
            raise AssertionError("a coverage other than the table's frac_below_bmax must be refused")
        dead = {**SELECTED, 7: (4.0, 20.0, "failed"), 12: (4.0, 30.0, "failed"),
                15: (4.0, 45.0, "failed")}
        _selected_campaign(Path(tmp) / "dead", selected=dead)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im.open_simulation(Path(tmp) / "dead").load(7, "CE")
        except ValueError as e:
            assert "re-run" in str(e)
        else:
            raise AssertionError("a bin whose draws all failed must be refused")
        # weights in the file but no selection summary: refused
        _campaign2(Path(tmp) / "nosum", events=(3, 7), weights={7: 2.0})
        try:
            im.open_simulation(Path(tmp) / "nosum").load(7, "CE")
        except ValueError as e:
            assert "selection" in str(e)
        else:
            raise AssertionError("event weights without a selection summary must be refused")


STD_PDG = [211, -211, 321, -321, 2212, -2212, 3122, -3122, 3312, -3312, 3334, -3334, 311, -311, 0]
STD_CHARGE = [1, -1, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 0, 0, 1]
STD_NMAX = [10, 10, 10, 10, 10, 10, 6, 6, 10, 10, 10, 10, 6, 6, 10]
STD_EDGES = np.sort(np.concatenate([np.round(np.arange(0, 3.0001, 0.05), 10), [0.27, 3.25, 3.5, 3.75, 4.0, 4.25,
                                                                                4.5, 4.75, 5.0, 10.0]]))


def _standard(data, dp=583, b_max=12.4, statuses=(0, 2, 0, 0), drawn=((0, 0), (1, 1), (2, 2), (3, 0), (4, 0)),
              variants=("Grad22", "CE"), weights=None, selection=None):
    """A standard-output file (iEBE-MUSIC docs/standard_output.md, 2D):
    rows = the collided events (status 0 ok, 2 no_surface); drawn = (event id,
    status) of every draw."""
    rng = np.random.default_rng(dp)
    d = Path(data) / f"dp{dp:04d}"
    d.mkdir(parents=True, exist_ok=True)
    ids = [e for e, st in drawn if st in (0, 2)]
    E, S, P = len(statuses), len(STD_PDG), len(STD_EDGES) - 1
    centres = 0.5 * (STD_EDGES[:-1] + STD_EDGES[1:])
    with h5py.File(d / f"qn_dp{dp:04d}.h5", "w") as f:
        f.attrs.update(format="iebe-music-qn", format_version=1, mode="2D", variants=",".join(variants),
                       afterburner="decay", b_max=b_max, n_drawn=len(drawn), n_collided=len(ids),
                       n_no_collision=sum(st == 1 for _, st in drawn))
        if selection:                     # {"summary": dict, "weight": [...], "centrality": [...]} per drawn row
            f.attrs["selection_format"] = 1
            f.attrs["selection"] = json.dumps(selection["summary"])
            f.attrs["n_collided"] = selection["summary"]["n_collided"]
        g = f.create_group("grid")
        g["pt_edges"] = STD_EDGES
        g["species_pdg"] = np.array(STD_PDG, np.int32)
        g["species_mass"] = np.array([0.138, 0.138, 0.494, 0.494, 0.938, 0.938, 1.116, 1.116, 1.315, 1.315, 1.672,
                                      1.672, 0.494, 0.494, 1.1894])
        g["species_charge"] = np.array(STD_CHARGE, np.int32)
        g["species_nmax"] = np.array(STD_NMAX, np.int32)
        g["harmonics"] = np.arange(11, dtype=np.int32)
        ev = f.create_group("events")
        ev["event_id"] = np.array(ids, np.int64)
        ev["status"] = np.array(statuses, np.uint8)
        ev["weight"] = np.ones(E) if weights is None else np.asarray(weights, float)
        dr = f.create_group("drawn")
        dr["event_id"] = np.array([e for e, _ in drawn], np.int64)
        dr["status"] = np.array([st for _, st in drawn], np.uint8)
        if selection:
            dr["weight"] = np.asarray(selection["weight"], float)
            dr["centrality"] = np.asarray(selection["centrality"], float)
            dr["p_acc"] = 1.0 / dr["weight"][:]
        for v in variants:
            vg = f.create_group(f"deltaf_{v}")
            vg.attrs.update(collection="all", dy_width=10.0, sums="raw over nsamples")
            counts = rng.poisson(20.0, (E, S, P)).astype(float)
            q = np.zeros((E, S, 11, P), np.complex64)
            q[:, :, 0, :] = counts
            for n in range(1, 11):
                q[:, :, n, :] = counts * (0.05 / n) * np.exp(1j * rng.uniform(0, 2 * np.pi, (E, S, 1)))
            for k, n in enumerate(STD_NMAX):
                q[:, k, n + 1:, :] = 0
            ptsum = np.zeros((E, S, 4, P), np.float32)
            for k in range(4):
                ptsum[:, :, k, :] = counts * centres ** (k + 1)
            ns = np.full(E, 5, np.int32)
            empty = np.asarray(statuses) == 2
            q[empty], ptsum[empty], ns[empty] = 0, 0, 0
            vg["Q"], vg["ptsum"], vg["nsamples"] = q, ptsum, ns
            vg["mtsum"] = ptsum[:, :, 0, :] * 1.1
    return d


def test_standard_output():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "merged"
        _standard(data)
        pkg = Path(tmp) / "pkg"
        pkg.mkdir()
        (pkg / "xsec.csv").write_text("campaign,dp,b_max,sigma_inel_fm2,frac_below_bmax\n"
                                      "a96,583,12.4,500.0,0.8\n")
        sim = im.open_simulation(data, pkg, xsec_table="xsec.csv")
        assert sim.design_points() == [583] and sim.variants(583) == ["Grad22", "CE"]
        a = sim.load(583, "CE")
        assert a["Qn"].shape == (4, 15, 11, 70) and a["rap_max"] == 5.0
        assert a["species_names"][-1] == "other_charged" and a["pdg_ids"][-1] == 0
        assert a["species_charge"] == STD_CHARGE and a["species_nmax"] == STD_NMAX
        assert list(a["nsamples"]) == [5, 1, 5, 5]                     # the empty event: nsamples 1, no sums
        assert np.all(a["Qn"][1] == 0) and a["meta"]["n_empty"] == 1
        c = 0.5 * (STD_EDGES[:-1] + STD_EDGES[1:])
        assert np.allclose(a["sum_pt2"], a["Qn"][:, :, 0, :].real * c ** 2, rtol=1e-5)   # ptsum k = 2
        assert a["sum_pt2"].shape == a["sum_mT"].shape == (4, 15, 70)
        assert sim.anchor(583) == {"n_ref": 4 / 0.8}                   # collided events over frac_below_bmax
        assert len(im.open_simulation(data, pkg, xsec_table="xsec.csv", include_empty=False).load(583, "CE")
                   ["nsamples"]) == 3
        for bad, err in ((dict(variant=None), ValueError), (dict(variant="PTB"), ValueError)):
            try:
                sim.load(583, bad["variant"])
            except err:
                pass
            else:
                raise AssertionError(f"{bad} must raise")
        (pkg / "xsec.csv").write_text("campaign,dp,b_max,sigma_inel_fm2,frac_below_bmax\n"
                                      "a96,583,13.0,500.0,0.8\n")
        try:
            im.open_simulation(data, pkg, xsec_table="xsec.csv").anchor(583)
        except ValueError:
            pass
        else:
            raise AssertionError("a b_max other than the table's must raise")
        # an importance-sampled file: /drawn the accepted draws (ids 10-14),
        # one failed (id 12, weight 4, in the 0-10% bin with id 10, weight 4)
        sel = {"summary": {"coverage": 0.8, "n_collided": 300, "n_drawn": 320, "b_max": 12.4,
                           "covered_centrality": [0.0, 80.0], "selection": {"centrality_bins": [0, 10, 50]}},
               "weight": [4.0, 10.0, 4.0, 10.0, 25.0], "centrality": [3.0, 20.0, 6.0, 30.0, 60.0]}
        _standard(Path(tmp) / "sel", statuses=(0, 0, 2, 0),
                  drawn=((10, 0), (11, 0), (12, 4), (13, 2), (14, 0)), weights=[4.0, 10.0, 10.0, 25.0],
                  selection=sel)
        ssim = im.open_simulation(Path(tmp) / "sel", pkg, xsec_table="xsec.csv")
        (pkg / "xsec.csv").write_text("campaign,dp,b_max,sigma_inel_fm2,frac_below_bmax\n"
                                      "a96,583,12.4,500.0,0.8\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b = ssim.load(583, "CE")
        assert any("failed" in str(x.message) for x in w)
        assert list(b["event_weights"]) == [8.0, 10.0, 10.0, 25.0]   # id 12's weight to id 10
        assert b["meta"]["n_failed"] == 1 and b["meta"]["n_empty"] == 1
        assert abs(b["meta"]["sum_weights"] - 53.0) < 1e-12
        assert ssim.anchor(583) == {"n_ref": 300 / 0.8}
        try:
            im.open_simulation(Path(tmp) / "sel", failed="error").load(583, "CE")
        except ValueError:
            pass
        else:
            raise AssertionError("failed='error' must refuse a failed draw")
        # refusal: sums above a row's nmax
        bad = _standard(Path(tmp) / "nmax") / "qn_dp0583.h5"
        with h5py.File(bad, "r+") as f:
            q = f["deltaf_CE"]["Q"][:]
            q[0, 6, 8, :] = 1.0                                       # Lambda Q8: above its nmax 6
            f["deltaf_CE"]["Q"][...] = q
        try:
            im.read_standard_output(bad, "CE")
        except ValueError:
            pass
        else:
            raise AssertionError("sums above a row's nmax must be refused")


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
