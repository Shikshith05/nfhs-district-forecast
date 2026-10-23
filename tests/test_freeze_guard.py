"""Freeze integrity: an OS-independent training hash and the one-shot guards (DECISIONS D25, D26)."""

import json
import os
import subprocess

import pandas as pd
import pytest

from nfhs_forecast import freeze_guard, models


def _tiny_train():
    return pd.DataFrame({"district_uid": ["s|b", "s|a", "s|a"], "indicator_id": ["x", "x", "x"],
                         "round": [4, 4, 5], "value": [12.5, 10.0, 11.25]})


def test_training_hash_independent_of_os_line_separator(monkeypatch):
    df = _tiny_train()
    monkeypatch.setattr(os, "linesep", "\r\n")
    h_windows = models.training_data_hash(df)
    monkeypatch.setattr(os, "linesep", "\n")
    h_unix = models.training_data_hash(df)
    assert h_windows == h_unix


def test_training_hash_golden_value():
    # sha256 of "district_uid,indicator_id,round,value\ns|a,x,4,10\ns|a,x,5,11.25\ns|b,x,4,12.5\n"
    import hashlib
    blob = b"district_uid,indicator_id,round,value\ns|a,x,4,10\ns|a,x,5,11.25\ns|b,x,4,12.5\n"
    assert models.training_data_hash(_tiny_train()) == hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------
# git-based checks
# ---------------------------------------------------------------------
def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "core.autocrlf=false", *args], check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_check_freeze_requires_tag(repo):
    with pytest.raises(freeze_guard.FreezeCheckError, match="does not exist"):
        freeze_guard.check_freeze(repo, "freeze-v1")


def test_check_freeze_passes_on_tagged_clean_head(repo):
    git(repo, "tag", "-a", "freeze-v1", "-m", "freeze")
    info = freeze_guard.check_freeze(repo, "freeze-v1")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    assert info["head_commit"] == head == info["tag_commit"]


def test_check_freeze_rejects_tag_not_at_head(repo):
    git(repo, "tag", "freeze-v1")                     # lightweight tags work too
    (repo / "b.txt").write_text("b\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "later")
    with pytest.raises(freeze_guard.FreezeCheckError, match="HEAD"):
        freeze_guard.check_freeze(repo, "freeze-v1")


@pytest.mark.parametrize("change", ["modify", "untracked"])
def test_check_freeze_rejects_dirty_tree(repo, change):
    git(repo, "tag", "-a", "freeze-v1", "-m", "freeze")
    if change == "modify":
        (repo / "a.txt").write_text("changed\n")
    else:
        (repo / "new.txt").write_text("new\n")
    with pytest.raises(freeze_guard.FreezeCheckError, match="not clean"):
        freeze_guard.check_freeze(repo, "freeze-v1")


def test_check_freeze_ignores_gitignored_files(repo):
    (repo / ".gitignore").write_text("processed/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "ignore")
    git(repo, "tag", "-a", "freeze-v1", "-m", "freeze")
    (repo / "processed").mkdir()
    (repo / "processed" / "panel.parquet").write_text("x")
    freeze_guard.check_freeze(repo, "freeze-v1")


def test_check_freeze_outside_git(tmp_path):
    with pytest.raises(freeze_guard.FreezeCheckError):
        freeze_guard.check_freeze(tmp_path, "freeze-v1")


def test_run_marker_is_exclusive(tmp_path):
    freeze_guard.check_not_already_run(tmp_path)
    marker = freeze_guard.claim_run(tmp_path, {"head_commit": "abc"})
    rec = json.loads(marker.read_text())
    assert rec["status"] == "started" and rec["head_commit"] == "abc"
    with pytest.raises(freeze_guard.AlreadyRunError):
        freeze_guard.check_not_already_run(tmp_path)
    with pytest.raises(freeze_guard.AlreadyRunError):
        freeze_guard.claim_run(tmp_path, {})
    freeze_guard.finish_run(marker, outcome="scored")
    rec = json.loads(marker.read_text())
    assert rec["status"] == "completed" and rec["head_commit"] == "abc" and rec["outcome"] == "scored"
