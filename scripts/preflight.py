"""Pre-flight checks before Gate B (DECISIONS D35). Prints PASS / WARN / FAIL per check.

    .venv\\Scripts\\python scripts\\preflight.py            # all checks
    .venv\\Scripts\\python scripts\\preflight.py --fast     # skip the full test suite

Checks
  1. git working tree clean (tracked outputs and frozen models committed)
  2. freeze tag absent and the NFHS-6 run marker absent (Gate B not yet done)
  3. full pytest suite passes
  4. leakage tests pass (run separately so a failure is named)
  5. the pipeline rebuilds the frozen training-data hash (run_pipeline.py
     --hash-only into a temporary directory; nothing in the repo is written)
     and config.yaml still has the frozen config hash
  6. no unresolved / low-confidence crosswalk case touches a stable-boundary
     test district; no unresolved 4->5 case touches a training unit; any
     5->6 case touching a test unit concerns a flagged (inherited) unit
  7. the pre-registrations D32 and D34 were committed before the results they
     govern, and their text has not changed since
Exit code 0 only if no check FAILs (WARN does not fail). This script never
reads NFHS-6 outcome values and never runs the NFHS-6 evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
#: Tests that enforce the no-leakage rule (HARD RULE 1).
LEAKAGE_TESTS = [
    "tests/test_leakage.py",
    "tests/test_end_to_end.py::test_predictions_use_only_round_t",
    "tests/test_harmonise.py::test_no_nfhs6_values_used_for_units_or_extra_variance",
    "tests/test_models_a3.py::test_seen_state_cv_rejects_round6",
]
#: Pre-registration -> result files it governs (first commit adding any of them must come later).
PREREG_RESULTS = {
    "D32": ["outputs/paired_tests_nfhs4to5.csv", "outputs/secondary_nfhs4to5.csv",
            "outputs/results_nfhs4to5_seenstate.csv", "outputs/results_nfhs5to6.csv"],
    "D34": ["outputs/gate_c_dev/coverage.csv", "outputs/gate_c/coverage.csv", "outputs/post_run/coverage.csv"],
}


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False)


# ---------------------------------------------------------------------
# Individual checks: each returns (status, message)
# ---------------------------------------------------------------------
def check_clean_tree(repo: Path) -> tuple[str, str]:
    r = git(repo, "status", "--porcelain", "--untracked-files=normal")
    if r.returncode != 0:
        return FAIL, f"git status failed: {r.stderr.strip()}"
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    if lines:
        return FAIL, f"{len(lines)} uncommitted change(s): " + "; ".join(lines[:5])
    return PASS, "working tree clean"


def check_not_yet_run(repo: Path, tag: str, marker: Path) -> tuple[str, str]:
    problems = []
    if git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0:
        problems.append(f"tag {tag} already exists")
    if marker.exists():
        problems.append(f"{marker.name} exists (the NFHS-6 evaluation already ran)")
    return (FAIL, "; ".join(problems)) if problems else (PASS, f"tag {tag} and run marker absent")


def run_pytest(repo: Path, args: list[str]) -> tuple[str, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
                       cwd=repo, capture_output=True, text=True)
    last = [x for x in r.stdout.strip().splitlines() if x.strip()]
    summary = last[-1] if last else r.stderr.strip()[-200:]
    return (PASS if r.returncode == 0 else FAIL), summary


def _temp_config(repo: Path, tmp: Path) -> Path:
    raw = yaml.safe_load((repo / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {k: str((repo / v).resolve()) for k, v in raw["paths"].items()}
    raw["paths"].update({"processed_dir": str(tmp / "processed"), "outputs_dir": str(tmp / "outputs"),
                         "models_dir": str(tmp / "outputs" / "models")})
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def check_hash(repo: Path) -> tuple[str, str]:
    from nfhs_forecast import models
    from nfhs_forecast.config import config_hash, load_config
    manifest = json.loads((repo / "outputs" / "models" / "manifest.json").read_text(encoding="utf-8"))
    frozen = models.load_frozen(repo / "outputs" / "models" / "frozen_models.pkl")
    if frozen.train_hash != manifest["train_hash"]:
        return FAIL, f"frozen_models.pkl hash {frozen.train_hash} != manifest {manifest['train_hash']}"
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _temp_config(repo, Path(tmp))
        r = subprocess.run([sys.executable, str(repo / "scripts" / "run_pipeline.py"), "--config", str(cfg_path),
                            "--hash-only"], cwd=tmp, capture_output=True, text=True)
    m = re.search(r"TRAIN_HASH=([0-9a-f]+)", r.stdout)
    if r.returncode != 0 or not m:
        return FAIL, f"pipeline failed: {(r.stderr or r.stdout).strip()[-300:]}"
    rebuilt = m.group(1)
    if rebuilt != frozen.train_hash:
        return FAIL, f"rebuilt training hash {rebuilt} != frozen {frozen.train_hash}"
    ch = config_hash(load_config(repo / "config.yaml"))
    if ch != frozen.config_hash:
        return FAIL, f"training hash {rebuilt} OK, but config hash {ch} != frozen {frozen.config_hash}"
    return PASS, f"training hash {rebuilt} and config hash {ch} match the frozen models"


def check_crosswalk(review: pd.DataFrame, units: pd.DataFrame) -> tuple[str, str]:
    weak = review[review.confidence.isin(["low", "unresolved"])]
    stable = set(units.loc[units.stable_unit.astype(bool), "unit_uid"])
    test = set(units.loc[units.test_unit.astype(bool), "unit_uid"])
    train = set(units.loc[units.train_unit.astype(bool), "unit_uid"])
    flagged = set(units.loc[units.test_unit.astype(bool) & ~units.unit_type.isin(["identity"]), "unit_uid"])

    def touched(df):
        return set(df.child_uid.dropna()) | set(df.parent_uid.dropna())
    bad_stable = touched(weak) & stable
    bad_train = set(weak.loc[weak.transition == "4to5", "child_uid"].dropna()) & train
    t56 = touched(weak[weak.transition == "5to6"]) & test
    bad_56 = t56 - flagged
    if bad_stable or bad_train or bad_56:
        parts = []
        if bad_stable:
            parts.append(f"{len(bad_stable)} stable test district(s): {sorted(bad_stable)[:5]}")
        if bad_train:
            parts.append(f"{len(bad_train)} training unit(s) with unresolved 4->5 lineage: {sorted(bad_train)[:5]}")
        if bad_56:
            parts.append(f"{len(bad_56)} unflagged test unit(s) with weak 5->6 lineage: {sorted(bad_56)[:5]}")
        return FAIL, "; ".join(parts)
    n4 = len(set(weak.loc[weak.transition == "4to5", "child_uid"].dropna()) & test)
    msg = (f"{len(weak)} weak cases; none touches a stable district or a training unit; {n4} test units lack "
           "4->5 lineage (training side only)")
    if t56:
        return WARN, msg + f"; weak 5->6 edge(s) touch flagged test unit(s) {sorted(t56)} (scored by inheritance)"
    return PASS, msg


def _first_commit_adding(repo: Path, pattern: str, path: str) -> str | None:
    r = git(repo, "log", "--reverse", "--format=%H", "-S", pattern, "--", path)
    commits = r.stdout.split()
    return commits[0] if commits else None


def _first_commit_with_file(repo: Path, path: str) -> str | None:
    r = git(repo, "log", "--reverse", "--format=%H", "--diff-filter=A", "--", path)
    commits = r.stdout.split()
    return commits[0] if commits else None


def _block(text: str, dec: str) -> str:
    m = re.search(rf"\n\*\*{dec}\.(.*?)(?=\n\*\*D\d+\.|\n#+ |\Z)", text, flags=re.S)
    return m.group(0).strip() if m else ""


def _is_strict_ancestor(repo: Path, a: str, b: str) -> bool:
    return a != b and git(repo, "merge-base", "--is-ancestor", a, b).returncode == 0


def check_preregistration(repo: Path, prereg: dict | None = None) -> tuple[str, str]:
    prereg = prereg or PREREG_RESULTS
    head_text = (repo / "DECISIONS.md").read_text(encoding="utf-8")
    msgs, status = [], PASS
    for dec, files in prereg.items():
        c = _first_commit_adding(repo, f"**{dec}.", "DECISIONS.md")
        if c is None:
            return FAIL, f"{dec} is not in the committed DECISIONS.md"
        then = git(repo, "show", f"{c}:DECISIONS.md").stdout
        if _block(then, dec) != _block(head_text, dec):
            return FAIL, f"{dec} text changed after its pre-registration commit {c[:8]}"
        for f in files:
            fc = _first_commit_with_file(repo, f)
            if fc is None:
                continue
            if fc == c:
                status = WARN if status == PASS else status
                msgs.append(f"{dec} and {f} first appear in the same commit {c[:8]}")
            elif not _is_strict_ancestor(repo, c, fc):
                return FAIL, f"{f} (commit {fc[:8]}) was committed before {dec} (commit {c[:8]})"
        msgs.append(f"{dec} committed in {c[:8]}, unchanged since")
    return status, "; ".join(msgs)


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--fast", action="store_true", help="skip the full test suite (leakage tests still run)")
    ap.add_argument("--json", default=None, help="also write the results to this JSON file")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    cfg = yaml.safe_load((repo / "config.yaml").read_text(encoding="utf-8"))
    tag = cfg["evaluation"].get("freeze_tag", "freeze-v1")
    out = repo / cfg["paths"]["outputs_dir"]

    checks = [
        ("git tree clean", lambda: check_clean_tree(repo)),
        ("Gate B not yet run", lambda: check_not_yet_run(repo, tag, out / "nfhs6_evaluation_run.json")),
    ]
    if not args.fast:
        checks.append(("full test suite", lambda: run_pytest(repo, [])))
    checks += [
        ("leakage tests", lambda: run_pytest(repo, LEAKAGE_TESTS)),
        ("training-data hash reproduces", lambda: check_hash(repo)),
        ("crosswalk cases vs test districts",
         lambda: check_crosswalk(pd.read_csv(out / "crosswalk_review.csv"), pd.read_csv(out / "harmonised_units.csv"))),
        ("pre-registration D32/D34 order", lambda: check_preregistration(repo)),
        ("git tree still clean", lambda: check_clean_tree(repo)),
    ]
    results = []
    for name, fn in checks:
        try:
            status, msg = fn()
        except Exception as e:  # a crashing check is a failed check
            status, msg = FAIL, f"{type(e).__name__}: {e}"
        results.append({"check": name, "status": status, "message": msg})
        print(f"[{status}] {name}: {msg}", flush=True)
    n_fail = sum(r["status"] == FAIL for r in results)
    print(f"\nPRE-FLIGHT {'FAILED' if n_fail else 'PASSED'}: {n_fail} failure(s), "
          f"{sum(r['status'] == WARN for r in results)} warning(s).")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
