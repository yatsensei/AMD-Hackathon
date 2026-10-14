#!/usr/bin/env bash
# E0 end to end on the AMD GPU (spec 6.1). Run inside the ROCm container after bootstrap.sh.
#   1. builds data/processed/ if missing (download + prepare_data)
#   2. trains E0: 5 seeds in parallel on the MI300X
#   3. aggregates the leaderboard, exports the best-val seed as models/v0, evaluates it on val
# Then commit: docs/results/{leaderboard.csv,v0_val/,data_report.md,env_amd.json}, models/v0/
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ! -f data/processed/wafers_s64.npz ]]; then
  [[ -f data/raw/LSWMD.pkl ]] || scripts/download_data.sh
  python scripts/prepare_data.py --jobs "$(nproc)"
fi

N="${N:-5}" scripts/run_sweep.sh configs/e0_baseline.yaml

best=$(python - <<'PY'
import json, pathlib
runs = [(json.loads(p.read_text())["val_macro_f1"], p.parent) for p in pathlib.Path("experiments").glob("e0_baseline_s*/summary.json")]
print(max(runs)[1])
PY
)
echo "best E0 seed by val macro-F1: $best"
version="${VERSION:-v0}"
python -m wafer.export --runs "$best" --version "$version"
python -m wafer.evaluate --model "models/$version"
python scripts/bench_latency.py --model "models/$version" || true
echo
echo "Done. Commit docs/results/ and models/$version/ (weights are small; see spec 9.4)."
