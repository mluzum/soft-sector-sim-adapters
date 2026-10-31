#!/usr/bin/env bash
# Fetch the software that reads this simulation package, at the versions it
# was tested with (written by soft-sector-sim-adapters/tools/deploy_package.py).
#
#   ./setup.sh           clone (or update) both repositories into ./software
#   ./setup.sh --venv    ... and build ./software/venv with the Python dependencies
#
# SOFTWARE_DIR=<dir> ./setup.sh puts the checkouts elsewhere (e.g. when this
# directory is read-only); the pipeline finds the adapter next to its own
# checkout, so nothing else changes.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sw="${SOFTWARE_DIR:-$here/software}"

ANALYSIS_URL="@@ANALYSIS_URL@@"
ANALYSIS_COMMIT="@@ANALYSIS_COMMIT@@"
ADAPTERS_URL="@@ADAPTERS_URL@@"
ADAPTERS_COMMIT="@@ADAPTERS_COMMIT@@"

fetch() {  # url commit dir
    if [ -d "$3/.git" ]; then
        git -C "$3" fetch --quiet origin
    else
        git clone --quiet "$1" "$3"
    fi
    git -C "$3" -c advice.detachedHead=false checkout --quiet "$2"
    echo "$3 at $(git -C "$3" rev-parse --short HEAD)"
}

mkdir -p "$sw"
fetch "$ANALYSIS_URL" "$ANALYSIS_COMMIT" "$sw/soft-sector-data-curation"
fetch "$ADAPTERS_URL" "$ADAPTERS_COMMIT" "$sw/soft-sector-sim-adapters"

py=python3
if [ "${1:-}" = "--venv" ]; then
    python3 -m venv "$sw/venv"
    "$sw/venv/bin/pip" install --quiet --upgrade pip
    "$sw/venv/bin/pip" install --quiet -r "$sw/soft-sector-data-curation/requirements.txt"
    py="$sw/venv/bin/python"
    echo "Python environment: source $sw/venv/bin/activate"
fi

cat <<EOF

Check the package:
  cd $sw/soft-sector-data-curation
  $py scripts/check_sim_package.py $here
Compute observables for one design point:
  $py scripts/run_pipeline.py --sim-package $here --design-point @@EXAMPLE_DP@@ @@EXAMPLE_VARIANT@@\\
      @@EXAMPLE_SELECTION@@ --output dp@@EXAMPLE_DP4@@.h5
For every design point and variant (rerun to pick up new data), then one table:
  $py scripts/run_campaign.py --sim-package $here --out-dir <runs dir> --jobs 8
  $py scripts/collect_campaign.py <runs dir> --output table.h5
EOF
