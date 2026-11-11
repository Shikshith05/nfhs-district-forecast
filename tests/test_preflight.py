"""scripts/preflight.py check functions (synthetic data and temporary git repos)."""

import importlib.util
import subprocess

import pandas as pd
import pytest

from tests.conftest import REPO


def _load():
    spec = importlib.util.spec_from_file_location("preflight", REPO / "scripts" / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "core.autocrlf=false", *args], check=True, capture_output=True, text=True)


def commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "DECISIONS.md").write_text("# Decisions\n\n**D31. Old.** text\n", encoding="utf-8")
    commit(tmp_path, "init")
    return tmp_path


def add_decision(repo, text):
    p = repo / "DECISIONS.md"
    p.write_text(p.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_clean_tree_and_not_yet_run(repo):
    assert pf.check_clean_tree(repo)[0] == pf.PASS
    (repo / "x.txt").write_text("x")
    assert pf.check_clean_tree(repo)[0] == pf.FAIL
    marker = repo / "outputs" / "run.json"
    assert pf.check_not_yet_run(repo, "freeze-v1", marker)[0] == pf.PASS
    git(repo, "tag", "freeze-v1")
    assert pf.check_not_yet_run(repo, "freeze-v1", marker)[0] == pf.FAIL


def test_preregistration_order_pass_same_commit_and_fail(repo):
    add_decision(repo, "\n**D32. Pre-registration — ü.** plan\n")
    commit(repo, "prereg")
    (repo / "outputs" / "res.csv").write_text("a\n")
    commit(repo, "results")
    assert pf.check_preregistration(repo, {"D32": ["outputs/res.csv"]})[0] == pf.PASS
    # a later decision appended after it does not count as a change
    add_decision(repo, "\n## New section\n\n*Intro text after the pre-registration.*\n\n**D33. Later.** more\n")
    commit(repo, "d33")
    assert pf.check_preregistration(repo, {"D32": ["outputs/res.csv"]})[0] == pf.PASS
    # editing the pre-registered text afterwards fails
    p = repo / "DECISIONS.md"
    p.write_text(p.read_text(encoding="utf-8").replace("plan", "plan, edited"), encoding="utf-8")
    commit(repo, "edit")
    status, msg = pf.check_preregistration(repo, {"D32": ["outputs/res.csv"]})
    assert status == pf.FAIL and "changed" in msg


def test_preregistration_results_before_plan_fail(repo):
    (repo / "outputs" / "res.csv").write_text("a\n")
    commit(repo, "results first")
    add_decision(repo, "\n**D32. Plan.** written late\n")
    commit(repo, "prereg late")
    assert pf.check_preregistration(repo, {"D32": ["outputs/res.csv"]})[0] == pf.FAIL
    assert pf.check_preregistration(repo, {"D40": []})[0] == pf.FAIL     # missing decision


def test_preregistration_same_commit_warns(repo):
    add_decision(repo, "\n**D34. Plan.** x\n")
    (repo / "outputs" / "dev.csv").write_text("a\n")
    commit(repo, "together")
    assert pf.check_preregistration(repo, {"D34": ["outputs/dev.csv"]})[0] == pf.WARN


def _units():
    return pd.DataFrame({
        "unit_uid": ["s|a", "s|b", "s|p", "s|t"],
        "unit_type": ["identity", "identity", "inherited_remnant", "identity_train_only"],
        "test_unit": [True, True, True, False], "train_unit": [True, False, False, True],
        "stable_unit": [True, False, False, False]})


def _review(rows):
    return pd.DataFrame(rows, columns=["transition", "parent_uid", "child_uid", "confidence"])


def test_crosswalk_check():
    u = _units()
    ok = _review([("4to5", None, "s|b", "unresolved"), ("5to6", None, "s|new", "unresolved")])
    assert pf.check_crosswalk(ok, u)[0] == pf.PASS
    warn = _review([("5to6", "s|p", "s|c", "low")])
    assert pf.check_crosswalk(warn, u)[0] == pf.WARN
    medium = _review([("5to6", "s|a", "s|a", "medium")])            # medium is not weak
    assert pf.check_crosswalk(medium, u)[0] == pf.PASS
    for bad in ([("5to6", "s|a", "s|c", "low")],                   # touches a stable district
                [("4to5", None, "s|t", "unresolved")],             # training unit with unknown 4->5 lineage
                [("5to6", "s|b", "s|c", "unresolved")]):           # unflagged test unit
        assert pf.check_crosswalk(_review(bad), u)[0] == pf.FAIL


def test_crosswalk_check_on_repo_outputs():
    out = REPO / "outputs"
    status, _ = pf.check_crosswalk(pd.read_csv(out / "crosswalk_review.csv"), pd.read_csv(out / "harmonised_units.csv"))
    assert status in (pf.PASS, pf.WARN)
