#!/usr/bin/env bash
# Download WM-811K (spec 3.2). Needs KAGGLE_USERNAME and KAGGLE_KEY in the environment
# (never commit them) and the kaggle CLI: pip install kaggle
set -euo pipefail

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
: "${KAGGLE_KEY:?set KAGGLE_KEY}"
command -v kaggle >/dev/null || { echo "kaggle CLI not found: pip install kaggle" >&2; exit 1; }

mkdir -p data/raw
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw --unzip
ls -lh data/raw

if command -v sha256sum >/dev/null; then sha=$(sha256sum data/raw/LSWMD.pkl | cut -d' ' -f1)
else sha=$(shasum -a 256 data/raw/LSWMD.pkl | cut -d' ' -f1); fi
echo "SHA-256 of LSWMD.pkl: $sha"

recorded=$(grep -oE 'SHA-256 of `LSWMD\.pkl`:\*\* *`[0-9a-fA-F]{64}`' docs/results/data_report.md 2>/dev/null \
           | grep -oE '[0-9a-fA-F]{64}' || true)
if [[ -z "$recorded" ]]; then
  echo "No hash on record yet: scripts/prepare_data.py will record it in docs/results/data_report.md."
elif [[ "$(echo "$recorded" | tr '[:upper:]' '[:lower:]')" == "$(echo "$sha" | tr '[:upper:]' '[:lower:]')" ]]; then
  echo "Matches the team's recorded hash."
else
  echo "MISMATCH with recorded hash $recorded. Do not load this file; ask the team." >&2
  exit 1
fi
