#!/usr/bin/env python3
"""Install a campaign's simulation package next to its data.

    python tools/deploy_package.py zr96c2 <results dir>/sim_package

copies ``campaigns/<campaign>/`` to the destination, pins the analysis and
adapter repositories at their current commits (appended to
``simulation.yaml`` as ``software:``) and writes ``setup.sh``, which clones
them at those commits.  Both checkouts must be clean: the pin must contain
the code that was tested.  Repository URLs default to each checkout's
``origin`` remote (GitHub SSH remotes are written as HTTPS, which anyone
can clone), else its local path; ``--analysis-url`` / ``--adapters-url``
override them.
"""

import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent      # this repository
ANALYSIS_REPO = "soft-sector-data-curation"
ADAPTERS_REPO = "soft-sector-sim-adapters"


def git(repo: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} in {repo}: {r.stderr.strip()}")
    return r.stdout.strip()


def pin(repo: Path, url: str | None, allow_dirty: bool) -> tuple[str, str]:
    if git(repo, "status", "--porcelain", "--untracked-files=no") and not allow_dirty:
        raise SystemExit(f"{repo} has uncommitted changes; commit them first "
                         f"(the pin must contain the tested code)")
    commit = git(repo, "rev-parse", "HEAD")
    if url is None:
        r = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"],
                           capture_output=True, text=True)
        url = r.stdout.strip() if r.returncode == 0 else str(repo.resolve())
    m = re.match(r"(?:ssh://)?git@github\.com[:/](.+?)(?:\.git)?$", url)
    if m:
        url = f"https://github.com/{m.group(1)}.git"
    return url, commit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", help="directory name under campaigns/")
    ap.add_argument("dest", help="package directory to write (e.g. <results>/sim_package)")
    ap.add_argument("--analysis-repo", default=str(HERE.parent / ANALYSIS_REPO),
                    help="checkout of soft-sector-data-curation to pin (default: sibling)")
    ap.add_argument("--analysis-url", default=None)
    ap.add_argument("--adapters-url", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="pin HEAD even if a checkout has uncommitted changes")
    a = ap.parse_args()

    src = HERE / "campaigns" / a.campaign
    if not (src / "simulation.yaml").is_file():
        raise SystemExit(f"no campaign {a.campaign!r} under {HERE / 'campaigns'}")
    manifest_text = (src / "simulation.yaml").read_text()
    manifest = yaml.safe_load(manifest_text)
    if "software" in manifest:
        raise SystemExit(f"{src}/simulation.yaml must not set 'software' (deploy writes it)")

    dest = Path(a.dest)
    if dest.exists() and any(dest.iterdir()) and not (dest / "simulation.yaml").exists():
        raise SystemExit(f"{dest} exists and is not a simulation package; refusing to write")
    data_root = (dest / manifest.get("data_root", ".")).resolve()
    if not data_root.is_dir():
        print(f"warning: data_root {data_root} does not exist", file=sys.stderr)

    an_url, an_commit = pin(Path(a.analysis_repo), a.analysis_url, a.allow_dirty)
    ad_url, ad_commit = pin(HERE, a.adapters_url, a.allow_dirty)

    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name != "simulation.yaml":
            shutil.copy2(f, dest / f.name)
    stamp = datetime.date.today().isoformat()
    software = (
        f"\n# Software this package was tested with (tools/deploy_package.py, {stamp});\n"
        f"# setup.sh clones these commits.\n"
        f"software:\n"
        f"  analysis: {{repo: {ANALYSIS_REPO}, url: '{an_url}', commit: {an_commit}}}\n"
        f"  adapters: {{repo: {ADAPTERS_REPO}, url: '{ad_url}', commit: {ad_commit}}}\n")
    (dest / "simulation.yaml").write_text(manifest_text.rstrip("\n") + "\n" + software)

    sel = manifest.get("selection") or {}
    words = []
    if sel.get("type"):
        words += ["--type", *sel["type"]]
    for key in ("reaction", "energy"):
        if sel.get(key) is not None:
            words += [f"--{key}", str(sel[key])]
    variants = manifest.get("variants") or []
    sub = {
        "@@ANALYSIS_URL@@": an_url, "@@ANALYSIS_COMMIT@@": an_commit,
        "@@ADAPTERS_URL@@": ad_url, "@@ADAPTERS_COMMIT@@": ad_commit,
        "@@EXAMPLE_DP@@": "1", "@@EXAMPLE_DP4@@": "0001",
        "@@EXAMPLE_VARIANT@@": f"--variant {variants[0]} " if variants else "",
        "@@EXAMPLE_SELECTION@@": shlex.join(words),
    }
    text = (HERE / "templates" / "setup.sh").read_text()
    for k, v in sub.items():
        text = text.replace(k, v)
    (dest / "setup.sh").write_text(text)
    os.chmod(dest / "setup.sh", 0o755)
    print(f"{a.campaign} -> {dest}\n  analysis {an_commit[:10]} ({an_url})\n"
          f"  adapters {ad_commit[:10]} ({ad_url})")


if __name__ == "__main__":
    main()
