#!/usr/bin/env bash
# Run on the AMD Developer Cloud instance (host). Starts -- or re-attaches to -- the ROCm
# PyTorch container with this repo mounted at /workspace. Pip installs survive restarts
# because the container is kept (named "wafer"), not removed.
#
#   scripts/amd/start_container.sh            # then, inside: scripts/amd/bootstrap.sh
#   ROCM_IMAGE=rocm/pytorch:<tag> scripts/amd/start_container.sh   # pin an image
set -euo pipefail

IMAGE="${ROCM_IMAGE:-rocm/pytorch:latest}"
NAME="${CONTAINER_NAME:-wafer}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Re-attaching to existing container '$NAME'"
  exec docker start -ai "$NAME"
fi

echo "Starting $IMAGE as '$NAME' with $REPO at /workspace"
exec docker run -it --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  -e KAGGLE_API_TOKEN -e KAGGLE_USERNAME -e KAGGLE_KEY \
  -v "$REPO":/workspace -w /workspace \
  "$IMAGE" bash
