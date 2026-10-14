#!/usr/bin/env bash
# Download WM-811K (spec 3.2). With credentials in the environment (never commit them) it uses
# the kaggle CLI >= 2:
#   KAGGLE_API_TOKEN                  new-style API token (kaggle.com -> Settings -> API Tokens), or
#   KAGGLE_USERNAME + KAGGLE_KEY      legacy kaggle.json values
# Without them it downloads the public archive directly (works as of 14 Oct 2026).
set -euo pipefail

mkdir -p data/raw
if [[ -n "${KAGGLE_API_TOKEN:-}" || ( -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ) ]] && command -v kaggle >/dev/null; then
  # Positional dataset argument: kaggle 2.x removed the old -d flag.
  kaggle datasets download qingyi/wm811k-wafer-map -p data/raw --unzip
else
  # The dataset is public: Kaggle's download endpoint redirects to its own storage without a login.
  echo "No Kaggle credentials/CLI: downloading the public archive directly."
  curl -sS -L --fail -o data/raw/archive.zip "https://www.kaggle.com/api/v1/datasets/download/qingyi/wm811k-wafer-map"
  python -I -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extract('LSWMD.pkl', sys.argv[2])" data/raw/archive.zip data/raw
  rm -f data/raw/archive.zip
fi
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
