"""Load AMD Token Factory settings from the environment. Never log the API key."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from wafer.env import REPO_ROOT


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    model: str
    classifier: str = "cnn"


def load_dotenv(path: Path | None = None) -> None:
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    env_path = path or (REPO_ROOT / ".env")
    if env_path.exists():
        _load(env_path)


def load_config(*, require_model: bool = True) -> AgentConfig:
    """Read AMD_API_KEY / AMD_BASE_URL / AMD_MODEL (and WAFER_CLASSIFIER).

    Falls back to RADEON_API_KEY for the key only, for teammate .env files that still
    use the older name. Never returns or prints the key in error messages.
    """
    load_dotenv()
    key = os.environ.get("AMD_API_KEY") or os.environ.get("RADEON_API_KEY")
    base = os.environ.get("AMD_BASE_URL", "").rstrip("/")
    model = os.environ.get("AMD_MODEL", "").strip()
    classifier = (os.environ.get("WAFER_CLASSIFIER") or "cnn").strip().lower()

    missing = []
    if not key:
        missing.append("AMD_API_KEY (or RADEON_API_KEY)")
    if not base:
        missing.append("AMD_BASE_URL")
    if require_model and not model:
        missing.append("AMD_MODEL")
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: " + ", ".join(missing)
            + ". Copy .env.example to .env and set them. "
            "List available model IDs with GET /v1/models — do not guess."
        )
    return AgentConfig(api_key=key, base_url=base, model=model, classifier=classifier)
