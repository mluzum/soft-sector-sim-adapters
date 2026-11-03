"""Tests of the JETSCAPE QnVector adapter on synthetic event files.

The synthetic files are written from particle lists in the format of the
real files (see the module docstring), so the Q-vectors the adapter returns
can be compared with sums over the particles themselves.

Run with ``python -m pytest tests`` or ``python tests/test_jetscape_qnvector.py``.
"""

import gzip
import os
import sys
import tarfile
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_adapters import jetscape_qnvector as jq  # noqa: E402

GRID = dict(pt_min=0.0, pt_max=3.0, n_pt=6, rap_min=-2.0, rap_max=2.0, n_rap=8, n_order=4)
ORDER = [22, 111, 2112, 211, 2212, 321, -211, -2212, -321, 9999]   # file order of the real files
NSAMP = 7


def _particles(rng, n=400):
    """Per-species particle lists (pT, rapidity, phi) inside the grid, with some v2."""
    out = {}
    for pid in ORDER:
        k = rng.poisson(n if pid == 9999 else n // 8)
        pt = rng.uniform(GRID["pt_min"], GRID["pt_max"], k)
        y = rng.uniform(GRID["rap_min"], GRID["rap_max"], k)
        phi = rng.uniform(-np.pi, np.pi, k)
        phi = phi - 0.1 * np.sin(2 * phi)          # v2 ~ 0.05
        out[pid] = (pt, y, phi)
    return out


def _g(x):
    return f"{x:g}"


def write_event(path, parts, nsamples=NSAMP, complete=True, swap_axes=False):
    """Write one event file the way the JETSCAPE QnVector writer does."""
    pe = np.linspace(GRID["pt_min"], GRID["pt_max"], GRID["n_pt"] + 1)
    ye = np.linspace(GRID["rap_min"], GRID["rap_max"], GRID["n_rap"] + 1)
    dpt, dy = pe[1] - pe[0], ye[1] - ye[0]
    lines = [
        "#\tJETSCAPE_FINAL_STATE\tQnVector\t",
        "\t# PID of Charged particle 9999 \t y(Charged) = pseudo-rapidity \t y(PID) = rapidity\t",
        "#\tpTmin\t{pt_min:g}\tpTmax\t{pt_max:g}\tNpT\t{n_pt}\trapmin\t{rap_min:g}\trapmax\t{rap_max:g}"
        "\tNrap\t{n_rap}\tNorder\t{n_order}".format(**GRID),
        "\t# PID of Charged particle 9999 \t y(Charged) = pseudo-rapidity \t y(PID) = rapidity\t",
        "#\t" + "\t".join(jq.COLUMNS),
        "#\tEvent\t1",
    ]
    for pid in ORDER:
        pt, y, phi = parts[pid]
        ip = np.clip(np.digitize(pt, pe) - 1, 0, GRID["n_pt"] - 1)
        iy = np.clip(np.digitize(y, ye) - 1, 0, GRID["n_rap"] - 1)
        outer, inner = (range(GRID["n_rap"]), range(GRID["n_pt"])) if swap_axes else \
                       (range(GRID["n_pt"]), range(GRID["n_rap"]))
        for a in outer:
            for b in inner:
                i, j = (b, a) if swap_axes else (a, b)
                sel = (ip == i) & (iy == j)
                n = int(sel.sum())
                if n == 0:
                    row = [pid, 0.5 * (pe[i] + pe[i + 1]), 0, 0.5 * (ye[j] + ye[j + 1])] + [0] * 17
                else:
                    p, r, f = pt[sel], y[sel], phi[sel]
                    row = [pid, p.mean(), p.std() / np.sqrt(n), r.mean(), r.std() / np.sqrt(n),
                           p.sum() / nsamples, n / (nsamples * dpt * dy),
                           np.sqrt(n) / (nsamples * dpt * dy)]
                    for h in (1, 2, 3):
                        c, s = np.cos(h * f), np.sin(h * f)
                        row += [c.mean(), c.std() / np.sqrt(n), s.mean(), s.std() / np.sqrt(n)]
                    row += [n]
                lines.append(" ".join(_g(x) for x in row))
    if complete:
        lines.append("#\tEvent\t1 End ")
    else:
        lines = lines[: len(lines) // 2]
    text = "\n".join(lines) + "\n"
    path = Path(path)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)


def expected_qn(parts, pid):
    """(n_rap, n_pt, 4) accumulated Q-vectors from the particles."""
    pe = np.linspace(GRID["pt_min"], GRID["pt_max"], GRID["n_pt"] + 1)
    ye = np.linspace(GRID["rap_min"], GRID["rap_max"], GRID["n_rap"] + 1)
    pt, y, phi = parts[pid]
    ip = np.clip(np.digitize(pt, pe) - 1, 0, GRID["n_pt"] - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, GRID["n_rap"] - 1)
    q = np.zeros((GRID["n_rap"], GRID["n_pt"], 4), complex)
    for h in range(4):
        np.add.at(q[..., h], (iy, ip), np.exp(1j * h * phi))
    spt = np.zeros((GRID["n_rap"], GRID["n_pt"]))
    np.add.at(spt, (iy, ip), pt)
    return q, spt


def _campaign(root, ids, rng, dp=3, gz=False):
    d = Path(root) / f"dp{dp:04d}"
    d.mkdir(parents=True)
    parts = {}
    for i in ids:
        parts[i] = _particles(rng)
        write_event(d / f"final_output_OO_123_{i}.dat{'.gz' if gz else ''}", parts[i])
    return d, parts


def test_qn_from_particles():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        _d, parts = _campaign(tmp, [0, 1, 4], rng)
        camp = jq.open_simulation(tmp, None, workers=2)
        assert camp.design_points() == [3]
        c = camp.load(3)
        assert list(c["meta"]["event_ids"]) == [0, 1, 4]
        assert sorted(c["species"]) == sorted(jq.DEFAULT_SPECIES)
        comb = c["combined"]
        assert "eta_edges" in comb and "y_edges" not in comb and comb["positives_only"] is False
        assert all("y_edges" in g and "eta_edges" not in g for g in c["species"].values())
        np.testing.assert_allclose(comb["nsamples"], NSAMP)
        for k, i in enumerate([0, 1, 4]):
            for pid, g in list(c["species"].items()) + [(9999, comb)]:
                q, spt = expected_qn(parts[i], pid)
                np.testing.assert_allclose(g["Qn"][k], q, atol=2e-4 * max(1, np.abs(q).max()))
                np.testing.assert_allclose(g["sum_pt"][k], spt, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(g["sum_ET"][k], spt, rtol=1e-5, atol=1e-5)  # ET = pT here
            np.testing.assert_allclose(comb["cent_mult"][k], len(parts[i][9999][0]) / NSAMP)


def test_complex128_and_extra_species():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        _campaign(tmp, [0], rng)
        c = jq.open_simulation(tmp, None, species=[211, 111, 22], dtype="complex128",
                               workers=1).load(3)
        assert sorted(c["species"]) == [22, 111, 211]
        assert c["combined"]["Qn"].dtype == np.complex128
        try:
            jq.open_simulation(tmp, None, species=[9999])
        except ValueError:
            pass
        else:
            raise AssertionError("9999 in species must be refused")


def test_archive_and_gz_agree_with_directory():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        d, parts = _campaign(Path(tmp) / "a", [2, 5, 9], rng)
        ref = jq.open_simulation(Path(tmp) / "a", None, workers=2).load(3)
        # gzipped tar named like the production archives, files under a top directory
        b = Path(tmp) / "b"
        b.mkdir()
        with tarfile.open(b / "dp0003.gz.tar", "w:gz") as tf:
            for f in sorted(d.iterdir()):
                tf.add(f, arcname=f"results/{f.name}")
        arc = jq.open_simulation(b, None, workers=2)
        assert arc.design_points() == [3] and jq.is_archive(arc.source(3))
        c = arc.load(3)
        np.testing.assert_array_equal(c["meta"]["event_ids"], ref["meta"]["event_ids"])
        np.testing.assert_array_equal(c["combined"]["Qn"], ref["combined"]["Qn"])
        # individually gzipped files
        g = Path(tmp) / "g"
        for i in parts:
            (g / "dp0003").mkdir(parents=True, exist_ok=True)
            write_event(g / "dp0003" / f"final_output_OO_123_{i}.dat.gz", parts[i])
        c = jq.open_simulation(g, None, workers=2).load(3)
        np.testing.assert_array_equal(c["combined"]["Qn"], ref["combined"]["Qn"])


def test_incomplete_and_missing_events():
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as tmp:
        d, _parts = _campaign(tmp, [0, 1, 3], rng)
        write_event(d / "final_output_OO_123_5.dat", _particles(rng), complete=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = jq.open_simulation(tmp, None, expected_events=7, workers=2).load(3)
        assert any("skipped 1 file" in str(x.message) for x in w)
        m = c["meta"]
        assert list(m["event_ids"]) == [0, 1, 3]
        assert len(m["skipped_files"]) == 1 and "final_output_OO_123_5.dat" in m["skipped_files"][0]
        assert list(m["missing_event_ids"]) == [2, 4, 5, 6]
        # missing ids as empty events, in id order
        c = jq.open_simulation(tmp, None, expected_events=7, missing="empty", workers=2).load(3)
        ids = c["meta"]["event_ids"]
        assert list(ids) == list(range(7)) and c["meta"]["n_empty"] == 4
        for g in list(c["species"].values()) + [c["combined"]]:
            assert g["Qn"].shape[0] == 7
            assert np.all(g["Qn"][[2, 4, 5, 6]] == 0) and np.all(g["Qn"][[0, 1, 3], ..., 0].real.sum(axis=(1, 2)) > 0)
            np.testing.assert_allclose(g["nsamples"], NSAMP)
        assert np.all(c["combined"]["cent_mult"][[2, 4, 5, 6]] == 0)
        try:
            jq.open_simulation(tmp, None, missing="empty")
        except ValueError:
            pass
        else:
            raise AssertionError("missing='empty' without expected_events must be refused")


def test_blocks_reorder_events():
    """Events arriving out of order across several blocks come out sorted by id."""
    b = jq._Blocks(size=2)
    arrival = [7, 3, 9, 1, 5]
    for eid in arrival:
        b.add(eid, {"nsamples": float(eid), "species": {211: {"dN": np.full((2, 3), eid, np.float32)}}})
    ids, ns, out = b.assemble()
    assert list(ids) == sorted(arrival) and list(ns) == sorted(arrival)
    assert [int(x[0, 0]) for x in out[211]["dN"]] == sorted(arrival)


def test_format_checks():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "final_output_x_1.dat"
        write_event(p, _particles(rng), swap_axes=True)          # rapidity outer: wrong order
        try:
            jq.read_event_file(p)
        except jq.FormatError as e:
            assert "outside the bin" in str(e)
        else:
            raise AssertionError("swapped pT/rapidity order not detected")
        write_event(p, _particles(rng))
        ev = jq.read_event_file(p, species=None)
        assert ev["pids"] == ORDER and ev["nsamples"] == NSAMP and ev["grid"]["n_harm"] == 3
        txt = p.read_text().replace("pseudo-rapidity", "rapidity")
        p.write_text(txt)
        try:
            jq.read_event_file(p)
        except jq.FormatError as e:
            assert "pseudo-rapidity" in str(e)
        else:
            raise AssertionError("header without the 9999 eta statement accepted")


def test_sources_mapping_and_anchor():
    rng = np.random.default_rng(6)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = root / "resultsOO_x"
        d.mkdir()
        write_event(d / "final_output_OO_1_0.dat", _particles(rng))
        camp = jq.open_simulation(root, None, sources={1: "resultsOO_x"}, anchor={"f": 0.9})
        assert camp.design_points() == [1] and camp.source_files(1) == [d]
        assert camp.anchor(1) == {"f": 0.9}
        pkg = root / "pkg"
        pkg.mkdir()
        (pkg / "anchors.csv").write_text("dp,sigma_tot,b_max\n1,140.5,9.0\n")
        camp = jq.open_simulation(root, pkg, sources={1: "resultsOO_x"}, anchor_table="anchors.csv")
        assert camp.anchor(1) == {"sigma_tot": 140.5, "b_max": 9.0}


def test_container_contract_if_analysis_present():
    """The analysis repository's validator accepts the arrays (when checked out alongside)."""
    ana = Path(__file__).resolve().parents[2] / "soft-sector-data-curation"
    if not (ana / "analysis" / "sim_package.py").is_file():
        print("  (analysis repository not found; skipped)")
        return
    sys.path.insert(0, str(ana))
    try:
        from analysis.sim_package import build_container
    except Exception as e:            # missing yaml/scipy, or an older analysis checkout
        print(f"  (analysis import failed: {e}; skipped)")
        return
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        _campaign(tmp, [0, 1], rng)
        c = jq.open_simulation(tmp, None, workers=1).load(3)
        try:
            obj = build_container(c, "3d")
        except Exception as e:
            if "y_edges" in str(e):   # analysis older than 8daeca5b0
                print(f"  (analysis predates y_edges grids: {e}; skipped)")
                return
            raise
        assert sorted(obj.grids) == sorted(jq.DEFAULT_SPECIES) and obj.combined is not None


def test_wrapper_package_if_analysis_present():
    """tools/qnvector_observables.py writes a package the analysis opens and loads."""
    ana = Path(__file__).resolve().parents[2] / "soft-sector-data-curation"
    if not (ana / "analysis" / "sim_package.py").is_file():
        print("  (analysis repository not found; skipped)")
        return
    sys.path.insert(0, str(ana))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    try:
        from analysis.sim_package import SimPackage
        import qnvector_observables as qo
    except Exception as e:
        print(f"  (import failed: {e}; skipped)")
        return
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        d, _ = _campaign(tmp, [0, 2, 3], rng)
        for events in (d, Path(tmp) / "events.tar.gz"):
            if events.suffix == ".gz":
                with tarfile.open(events, "w:gz") as tf:
                    tf.add(d, arcname="dp0003")
            pkg = Path(tmp) / f"pkg_{events.name}"
            qo.write_package(pkg, events.resolve(), name="t", system="p Pb", energy=5020,
                             forward_beam="p", anchor=0.8, expected_events=4, workers=1)
            sp = SimPackage(pkg, adapter_path=str(Path(__file__).resolve().parents[1]))
            assert sp.design_points() == [0]
            c = sp.load(0)
            assert c.combined.n_events == 3
            assert sp.anchor(0) == 0.8
            assert list(sp.frames.values()) == [{"frame": "nn_cms", "forward_beam": "p"}]


def test_real_file_if_present():
    """Layout of a real file against a plain loadtxt of it."""
    real = Path("/data/mluzum/Andreas/resultsOO_QVec_tau03_big_pos/final_output_OO_12321059_0.dat")
    if not real.is_file():
        print("  (real file not present; skipped)")
        return
    raw = np.loadtxt(real, comments="#")
    ev = jq.read_event_file(real, species=None)
    assert ev["nsamples"] == 500 and ev["grid"]["n_harm"] == 3
    for pid in (2212, 9999):
        blk = raw[raw[:, 0] == pid].reshape(30, 50, 21)            # pT outer, rapidity inner
        d = ev["species"][pid]
        np.testing.assert_array_equal(d["dN"], blk[:, :, -1].T)
        np.testing.assert_allclose(d["cos"][..., 1], blk[:, :, 12].T, rtol=1e-6)
        np.testing.assert_allclose(d["sin"][..., 2], blk[:, :, 18].T, rtol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
