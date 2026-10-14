"""Runtime environment: ROCm defaults and the env.json every run writes (spec 4.1, 5.6)."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def configure_rocm_env() -> None:
    """MIOpen defaults. Call before the first convolution runs (safe to call anywhere).

    FIND_MODE=FAST avoids very slow first epochs while MIOpen searches kernels; the user DB
    must be writable (inside Docker the default location may not be).
    """
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    db = Path(os.environ.setdefault("MIOPEN_USER_DB_PATH", str(REPO_ROOT / ".miopen")))
    db.mkdir(parents=True, exist_ok=True)


def git_commit() -> tuple[str, bool]:
    """(short commit, dirty) or ("unknown", True) outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return commit, bool(status)
    except Exception:
        return "unknown", True


def env_info() -> dict:
    import numpy as np
    import torch
    commit, dirty = git_commit()
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "host": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu_available": torch.cuda.is_available(),
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info
