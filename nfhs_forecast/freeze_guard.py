"""Checks that make scripts/evaluate_nfhs6.py a genuine one-shot test (DECISIONS D26).

The NFHS-6 evaluation may run only when
  1. the git tag `evaluation.freeze_tag` (default `freeze-v1`) exists and
     points at HEAD,
  2. the working tree is clean (no modified, staged or untracked files;
     git-ignored files such as data/processed/ are allowed), and
  3. it has never run before: the run marker `nfhs6_evaluation_run.json`
     does not exist in the output directory.
The marker is created with exclusive-create mode BEFORE any NFHS-6 value is
scored, so a run that crashes half-way still counts as the one run.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RUN_MARKER = "nfhs6_evaluation_run.json"


class FreezeCheckError(RuntimeError):
    """The repository is not in the frozen, tagged, clean state the test requires."""


class AlreadyRunError(RuntimeError):
    """The one-shot NFHS-6 evaluation has already been run."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=False)
    except FileNotFoundError as e:  # git not installed
        raise FreezeCheckError("git is not available; cannot verify the freeze tag") from e


def check_freeze(repo: str | Path, tag: str) -> dict:
    """Verify tag == HEAD and a clean tree. Returns commit info; raises FreezeCheckError."""
    repo = Path(repo)
    if _git(repo, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        raise FreezeCheckError(f"{repo} is not a git working tree")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    res = _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
    if res.returncode != 0 or not res.stdout.strip():
        raise FreezeCheckError(f"git tag '{tag}' does not exist; review the freeze and tag it first")
    tag_commit = res.stdout.strip()
    if tag_commit != head:
        raise FreezeCheckError(f"git tag '{tag}' points at {tag_commit[:12]}, but HEAD is {head[:12]}; "
                               f"check out the tagged commit")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=normal").stdout.strip()
    if dirty:
        lines = dirty.splitlines()
        shown = "\n  ".join(lines[:10]) + ("\n  ..." if len(lines) > 10 else "")
        raise FreezeCheckError(f"working tree is not clean ({len(lines)} changes):\n  {shown}")
    return {"head_commit": head, "freeze_tag": tag, "tag_commit": tag_commit}


def check_not_already_run(out_dir: str | Path) -> Path:
    marker = Path(out_dir) / RUN_MARKER
    if marker.exists():
        raise AlreadyRunError(f"{marker} exists: the one-shot NFHS-6 evaluation has already been run. "
                              "Any re-run must be justified and logged in DECISIONS.md, and the "
                              "marker removed by hand.")
    return marker


def claim_run(out_dir: str | Path, info: dict) -> Path:
    """Create the run marker atomically (fails if it exists) before any scoring."""
    marker = Path(out_dir) / RUN_MARKER
    record = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "python": platform.python_version(), **info}
    try:
        with open(marker, "x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, indent=1, default=str) + "\n")
    except FileExistsError as e:
        raise AlreadyRunError(f"{marker} appeared while starting; refusing to run twice") from e
    return marker


def finish_run(marker: str | Path, **updates) -> None:
    marker = Path(marker)
    record = json.loads(marker.read_text(encoding="utf-8"))
    record.update(updates)
    record["status"] = "completed"
    record["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(marker, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, indent=1, default=str) + "\n")
