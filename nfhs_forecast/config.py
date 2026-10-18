"""Load config.yaml and resolve paths relative to the repo root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config and add absolute paths under cfg['abs_paths']."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parent
    cfg["abs_paths"] = {k: (root / v) for k, v in cfg["paths"].items()}
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("processed_dir", "outputs_dir", "models_dir"):
        cfg["abs_paths"][key].mkdir(parents=True, exist_ok=True)


def config_hash(cfg: dict) -> str:
    """Stable hash of the config (minus resolved paths), stored with frozen models."""
    clean = {k: v for k, v in cfg.items() if k != "abs_paths"}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def region_of(cfg: dict) -> dict[str, str]:
    """Map canonical state name -> region."""
    return {state: region for region, states in cfg["regions"].items() for state in states}
