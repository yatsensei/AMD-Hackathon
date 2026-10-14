#!/usr/bin/env bash
# Run every config x seed, N training processes at a time on the one GPU (spec 4.2, 5.7).
#   scripts/run_sweep.sh configs/tier1/*.yaml            # N defaults to 8
#   N=4 scripts/run_sweep.sh configs/e0_baseline.yaml
# Logs: experiments/logs/<config>_s<seed>.log. Watch the GPU with rocm-smi.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${N:-8}"
[[ $# -gt 0 ]] || { echo "usage: $0 configs/*.yaml" >&2; exit 1; }
mkdir -p experiments/logs

jobs_file=$(mktemp)
for cfg in "$@"; do
  for seed in $(python -c "import sys,yaml; print(' '.join(map(str, yaml.safe_load(open(sys.argv[1])).get('seeds', [0]))))" "$cfg"); do
    echo "$cfg $seed" >> "$jobs_file"
  done
done
echo "$(wc -l < "$jobs_file") runs, $N at a time"

status=0
xargs -P "$N" -L 1 bash -c 'name=$(basename "$0" .yaml); python -m wafer.train --config "$0" --seed "$1" \
  > "experiments/logs/${name}_s$1.log" 2>&1 && echo "ok   $0 seed $1" || { echo "FAIL $0 seed $1 (see experiments/logs/${name}_s$1.log)"; exit 255; }' \
  < "$jobs_file" || status=$?
rm -f "$jobs_file"
python scripts/aggregate.py || true
exit $status
