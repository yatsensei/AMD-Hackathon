#!/usr/bin/env bash
# Run inside the ROCm container (or on a host whose Python already has ROCm PyTorch).
# Installs the repo's CPU-side dependencies WITHOUT touching torch, then checks the GPU and
# records the environment in docs/results/env_amd.json (commit it: it backs the AMD claim).
set -euo pipefail
cd "$(dirname "$0")/../.."

python - <<'PY'
import sys, torch
if not torch.version.hip:
    sys.exit("This Python's torch has no ROCm/HIP support (torch %s). Use the rocm/pytorch image "
             "(scripts/amd/start_container.sh); never pip install the CUDA wheel." % torch.__version__)
print("torch", torch.__version__, "hip", torch.version.hip)
PY

# requirements.txt never lists torch, so pip leaves the ROCm build alone.
python -m pip install -q -r requirements.txt
python -m pip install -q kaggle
python scripts/check_gpu.py --out docs/results/env_amd.json
echo
echo "Ready. Next: scripts/amd/run_e0.sh"
