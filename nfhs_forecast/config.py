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


#: Config sections that only steer analyses run AFTER the NFHS-6 evaluation (post-run
#: tables, figures, report, release). They cannot affect training or the frozen
#: models, so they are left out of the config hash (DECISIONS D35).
POST_RUN_SECTIONS = ("post_run",)


def config_hash(cfg: dict) -> str:
    """Stable hash of the config (minus resolved paths and post-run sections), stored with frozen models."""
    clean = {k: v for k, v in cfg.items() if k != "abs_paths" and k not in POST_RUN_SECTIONS}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def write_csv(df, path: str | Path, **kwargs) -> None:
    """DataFrame.to_csv with '\\n' line endings on every OS (DECISIONS D25)."""
    kwargs.setdefault("index", False)
    df.to_csv(path, lineterminator="\n", encoding="utf-8", **kwargs)


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text with '\\n' line endings on every OS."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json(obj, path: str | Path) -> None:
    write_text(path, json.dumps(obj, indent=1, default=str) + "\n")


def region_of(cfg: dict) -> dict[str, str]:
    """Map canonical state name -> region."""
    return {state: region for region, states in cfg["regions"].items() for state in states}
