#!/usr/bin/env bash
# Download WM-811K (spec 3.2). Credentials come from the environment (never commit them):
#   KAGGLE_API_TOKEN                  new-style API token (kaggle.com -> Settings -> API Tokens), or
#   KAGGLE_USERNAME + KAGGLE_KEY      legacy kaggle.json values
# Needs the kaggle CLI >= 2: pip install "kaggle>=2"
set -euo pipefail

if [[ -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY)." >&2; exit 1
fi
command -v kaggle >/dev/null || { echo 'kaggle CLI not found: pip install "kaggle>=2"' >&2; exit 1; }

mkdir -p data/raw
# Positional dataset argument: kaggle 2.x removed the old -d flag.
kaggle datasets download qingyi/wm811k-wafer-map -p data/raw --unzip
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
