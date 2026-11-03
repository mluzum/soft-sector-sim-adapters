#!/usr/bin/env python3
"""Simulation vs experiment for every record in a run_pipeline.py output.

    python plot_vs_data.py <pipeline output .h5> [more .h5 ...] [--out DIR] [--png]

One panel per record table. The x axis is pT, eta or centrality, whichever
the labels vary; a table spanning several centrality classes vs pT or eta
gets one panel per class. Simulation: line + jackknife band; data: points
with total errors. Writes <out>/<stem>.pdf (all pages), optionally one
PNG per page, and <out>/<stem>_table.csv with sim, data and pull
(sim - data) / sqrt(err_sim^2 + err_data^2) for every point.
"""

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.mathtext import MathTextParser
import numpy as np

SIM = "#2a78d6"          # categorical slot 1
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

CENT_RE = re.compile(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)%")
PT_RE = re.compile(r"\bpT (\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)")
ETA_RE = re.compile(r"\beta (-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)")


def _rng(rx, s):
    m = rx.search(s)
    return (float(m.group(1)), float(m.group(2))) if m else None


_MATHTEXT = MathTextParser("path")


def _tex(s):
    """A value_type as a matplotlib title: LaTeX that mathtext cannot parse
    (e.g. \\lvert, or unbalanced $) falls back to plain text."""
    s = s.replace(r"\text{", r"\mathrm{").replace(r"\lvert", "|").replace(r"\rvert", "|")
    if s.count("$") % 2 == 0:
        try:
            _MATHTEXT.parse(s)
            return s
        except ValueError:
            pass
    return re.sub(r"\\([{}|])", r"\1", s.replace("$", ""))


def read_groups(path):
    out = []
    with h5py.File(path, "r") as f:
        for gk in f:
            g = f[gk]
            if "labels" not in g:
                continue
            labs = [l.decode() if isinstance(l, bytes) else str(l) for l in g["labels"][:]]
            lr = g["label_record"][:]
            sim, serr = g["values"][:], g["errors"][:]
            recs = OrderedDict()
            for r in sorted(set(lr.tolist())):
                ii = np.where(lr == r)[0]
                e = g[f"exp_{r}"]
                ev, ee = e["values"][:], e["errors"][:]
                if len(ev) != len(ii):
                    print(f"skip {gk} exp_{r}: {len(ii)} labels, {len(ev)} data points")
                    continue
                recs[r] = dict(attrs={k: (v.decode() if isinstance(v, bytes) else v)
                                      for k, v in e.attrs.items()},
                               labels=[labs[i].split(" [")[0] for i in ii],
                               sim=sim[ii], serr=serr[ii], exp=ev, eerr=ee)
            title = f"{g.attrs.get('reaction', gk).split(' -->')[0].replace(' ', '+')}"
            try:
                title += f", {float(g.attrs.get('energy')):g} GeV"
            except (TypeError, ValueError):
                pass
            out.append((gk, title, recs))
    return out


def panels_of(rec):
    """Split a record into (subtitle, xlabel, x, xerr, index) panels."""
    labs = rec["labels"]
    pts = [_rng(PT_RE, l) for l in labs]
    etas = [_rng(ETA_RE, l) for l in labs]
    cents = [_rng(CENT_RE, l) for l in labs]
    if all(p for p in pts) and len(set(pts)) > 1:
        axis, xlabel = pts, r"$p_T$ (GeV)"
    elif all(e for e in etas) and len(set(etas)) > 1:
        axis, xlabel = etas, r"$\eta$"
    elif all(c for c in cents):
        lo = np.array([c[0] for c in cents]); hi = np.array([c[1] for c in cents])
        return [("", "centrality (%)", (lo + hi) / 2, (hi - lo) / 2, np.arange(len(labs)))]
    else:
        return [("", "point", np.arange(len(labs)), None, np.arange(len(labs)))]
    out = []
    for c in OrderedDict.fromkeys(cents):
        idx = np.array([i for i, cc in enumerate(cents) if cc == c])
        lo = np.array([axis[i][0] for i in idx]); hi = np.array([axis[i][1] for i in idx])
        sub = f"{c[0]:g}-{c[1]:g}%" if c else ""
        out.append((sub, xlabel, (lo + hi) / 2, (hi - lo) / 2, idx))
    return out


def draw(ax, rec, panel, exp_name):
    sub, xlabel, x, xerr, idx = panel
    s, se = rec["sim"][idx], rec["serr"][idx]
    d, de = rec["exp"][idx], rec["eerr"][idx]
    order = np.argsort(x)
    x, xerr, s, se, d, de = (a[order] if a is not None else None
                             for a in (x, xerr, s, se, d, de))
    ok = np.isfinite(s)
    if ok.any():
        ax.fill_between(x[ok], (s - se)[ok], (s + se)[ok], color=SIM, alpha=0.25, lw=0)
        ax.plot(x[ok], s[ok], color=SIM, lw=1.6, label="Simulation")
    okd = np.isfinite(d)
    ax.errorbar(x[okd], d[okd], yerr=de[okd], xerr=None if xerr is None else xerr[okd],
                fmt="o", ms=3.5, color=INK, ecolor=INK, elinewidth=0.9, capsize=0,
                label=exp_name)
    both = ok & okd
    chi = None
    if both.any():
        pull = (s - d)[both] / np.sqrt(se[both] ** 2 + de[both] ** 2)
        chi = float(np.mean(pull ** 2))
    vt = _tex(str(rec["attrs"].get("value_type", "")))
    src = Path(str(rec["attrs"].get("hepdata_file", ""))).stem
    head = vt + (f"   {sub}" if sub else "")
    ax.set_title(head, fontsize=9, color=INK, loc="left")
    note = src + ("" if ok.any() else "   (no simulation value)")
    if chi is not None:
        note += f"   $\\chi^2/N$ = {chi:.1f}"
    ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=6.5, color=MUTED,
            va="top", ha="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b5b4ae")
    if xlabel == "centrality (%)":
        ax.set_xlim(left=0)
    # y range from the data, widened for simulation points that are not
    # statistically noise-dominated; a wild band is clipped, not shown whole.
    # Positive data spanning more than two decades (spectra) go on a log
    # axis, limited to the data so a vanishing simulation tail is cut off.
    if okd.any() and np.all(d[okd] > 0) and d[okd].max() > 100 * d[okd].min():
        ax.set_yscale("log")
        ax.set_ylim(d[okd].min() / 5, d[okd].max() * 5)
    elif okd.any():
        lo, hi = np.nanmin((d - de)[okd]), np.nanmax((d + de)[okd])
        good = ok & (se < max(hi - lo, 1e-12))
        if good.any():
            lo, hi = min(lo, np.min(s[good])), max(hi, np.max(s[good]))
        pad = 0.15 * (hi - lo if hi > lo else abs(hi) or 1.0)
        ax.set_ylim(lo - pad, hi + 2 * pad)
    return [(sub, float(x[i]), float(s[i]), float(se[i]), float(d[i]), float(de[i]))
            for i in range(len(x))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("h5", nargs="+", help="run_pipeline.py outputs (plotted in order)")
    ap.add_argument("--stem", help="output file stem (default: first input's)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--png", action="store_true", help="also write one PNG per page")
    ap.add_argument("--per-page", type=int, default=12)
    ap.add_argument("--match", default=None, help="regex on value_type or HEPData table")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = args.stem or Path(args.h5[0]).stem
    rows = []
    jobs = []
    groups = [g for h in args.h5 for g in read_groups(h)]
    for gk, title, recs in groups:
        for r, rec in recs.items():
            key = f"{rec['attrs'].get('value_type', '')} {rec['attrs'].get('hepdata_file', '')}"
            if args.match and not re.search(args.match, key):
                continue
            exp_name = Path(str(rec["attrs"].get("record_file", "data"))).name.split("_")[0].upper()
            for p in panels_of(rec):
                jobs.append((title, rec, p, exp_name, r))

    ncol = 4
    pages = [jobs[i:i + args.per_page] for i in range(0, len(jobs), args.per_page)]
    with PdfPages(out / f"{stem}.pdf") as pdf:
        for k, page in enumerate(pages):
            nrow = -(-len(page) // ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.9 * nrow + 0.7), squeeze=False)
            for ax, (title, rec, p, exp_name, r) in zip(axes.flat, page):
                for row in draw(ax, rec, p, exp_name):
                    rows.append([title, rec["attrs"].get("hepdata_file", ""),
                                 rec["attrs"].get("value_type", ""), *row])
            for ax in list(axes.flat)[len(page):]:
                ax.set_visible(False)
            names = sorted({j[3] for j in page})
            handles = [Line2D([], [], color=SIM, lw=1.6),
                       Line2D([], [], color=INK, marker="o", ms=3.5, ls="none")]
            fig.legend(handles, ["Simulation", " / ".join(names)], loc="upper right",
                       ncol=2, frameon=False, fontsize=9)
            fig.suptitle(f"{page[0][0]}: simulation vs data  ({k + 1}/{len(pages)})",
                         x=0.01, ha="left", fontsize=11, color=INK)
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
            pdf.savefig(fig)
            if args.png:
                fig.savefig(out / f"{stem}_p{k + 1:02d}.png", dpi=110)
            plt.close(fig)
    with open(out / f"{stem}_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "table", "value_type", "class", "x", "sim", "sim_err",
                    "data", "data_err"])
        w.writerows(rows)
    print(f"{len(jobs)} panels on {len(pages)} pages -> {out / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
