"""scripts/gate_b.ps1 driven end to end in a temporary repo with STUB pre-flight / evaluation scripts.

The real evaluation is never run: the stubs replace both scripts inside tmp_path.
"""

import json
import shutil
import subprocess
import sys

import pytest

from tests.conftest import REPO

PS = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(PS is None, reason="PowerShell not available")

EVAL_STUB = '''import json, pathlib, sys
out = pathlib.Path("outputs"); out.mkdir(exist_ok=True)
log = pathlib.Path("eval_calls.txt"); log.write_text(log.read_text() + "x" if log.exists() else "x")
mode = pathlib.Path("eval_mode.txt").read_text().strip()
if mode == "fail_before":
    sys.exit(6)
(out / "nfhs6_evaluation_run.json").write_text(json.dumps({"status": "started"}))
if mode == "fail_after":
    sys.exit(1)
(out / "results_nfhs5to6.csv").write_text("a,b\\n1,2\\n")
(out / "predictions_nfhs5to6.parquet").write_bytes(b"PAR1")
(out / "nfhs6_evaluation_run.json").write_text(json.dumps({"status": "completed"}))
'''


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "outputs" / "models").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "gate_b.ps1", tmp_path / "scripts" / "gate_b.ps1")
    (tmp_path / "scripts" / "evaluate_nfhs6.py").write_text(EVAL_STUB)
    (tmp_path / "scripts" / "preflight.py").write_text(
        "import pathlib, sys\nsys.exit(int(pathlib.Path('preflight_code.txt').read_text()))\n")
    (tmp_path / "outputs" / "models" / "manifest.json").write_text(json.dumps(
        {"train_hash": "abc", "n_core_districts": 1, "n_test_units": 2, "n_stable_units": 1}))
    (tmp_path / ".gitignore").write_text("outputs/*.parquet\npreflight_code.txt\neval_mode.txt\neval_calls.txt\n")
    (tmp_path / "preflight_code.txt").write_text("0")
    (tmp_path / "eval_mode.txt").write_text("ok")
    for cmd in (["init", "-q"], ["config", "user.name", "t"], ["config", "user.email", "t@example.com"],
                ["add", "-A"], ["commit", "-q", "-m", "init"]):
        assert git(tmp_path, *cmd).returncode == 0
    return tmp_path


def run_gate(repo, answer="YES"):
    return subprocess.run([PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                           str(repo / "scripts" / "gate_b.ps1"), "-RepoRoot", str(repo), "-Python", sys.executable],
                          input=answer + "\n", capture_output=True, text=True, timeout=120)


def has_tag(repo):
    return git(repo, "tag", "-l", "freeze-v1").stdout.strip() == "freeze-v1"


def test_gate_b_parses():
    r = subprocess.run([PS, "-NoProfile", "-Command",
                        "$e=$null; [System.Management.Automation.Language.Parser]::ParseFile("
                        f"'{REPO / 'scripts' / 'gate_b.ps1'}', [ref]$null, [ref]$e) | Out-Null; $e.Count"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "0", r.stdout + r.stderr


def test_gate_b_happy_path(fake_repo):
    head = git(fake_repo, "rev-parse", "HEAD").stdout.strip()
    r = run_gate(fake_repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GATE B DONE" in r.stdout
    assert git(fake_repo, "rev-parse", "freeze-v1^{commit}").stdout.strip() == head   # tag on the frozen commit
    files = git(fake_repo, "show", "--name-only", "--format=", "HEAD").stdout.split()
    assert {"outputs/nfhs6_evaluation_run.json", "outputs/predictions_nfhs5to6.parquet",
            "outputs/results_nfhs5to6.csv"} <= set(files)
    assert (fake_repo / "eval_calls.txt").read_text() == "x"                          # ran exactly once


def test_gate_b_needs_yes(fake_repo):
    r = run_gate(fake_repo, answer="yes")
    assert r.returncode == 10 and not has_tag(fake_repo) and not (fake_repo / "eval_calls.txt").exists()


def test_gate_b_stops_on_preflight_failure(fake_repo):
    (fake_repo / "preflight_code.txt").write_text("1")
    r = run_gate(fake_repo)
    assert r.returncode != 0 and "pre-flight failed" in r.stdout
    assert not has_tag(fake_repo) and not (fake_repo / "eval_calls.txt").exists()


@pytest.mark.parametrize("mode,expect", [("fail_before", "BEFORE scoring"), ("fail_after", "AFTER claiming")])
def test_gate_b_evaluation_failure_is_not_retried(fake_repo, mode, expect):
    (fake_repo / "eval_mode.txt").write_text(mode)
    before = git(fake_repo, "rev-parse", "HEAD").stdout
    r = run_gate(fake_repo)
    assert r.returncode != 0 and expect in r.stdout
    assert (fake_repo / "eval_calls.txt").read_text() == "x"                          # no retry
    assert git(fake_repo, "rev-parse", "HEAD").stdout == before                       # nothing committed
