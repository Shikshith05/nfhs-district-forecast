"""Synthetic end-to-end run: LOSO CV, metrics, freeze, and the NFHS-6 evaluation script.

Uses synthetic data only; the real NFHS-6 evaluation is never run by the tests.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from nfhs_forecast import evaluate, freeze_guard, models
from nfhs_forecast.config import config_hash
from tests.conftest import REPO, make_synthetic_panel


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _freeze(cfg, panel, inds):
    uids = sorted(panel.district_uid.unique())
    train = panel[panel["round"].isin([4, 5])]
    trans = models.build_transition(train, 4, 5, inds, uids, cfg)
    spec = evaluate.feature_spec(trans, inds, cfg["models"]["feature_sets"])
    frozen = models.freeze(train, trans, cfg, spec, config_hash(cfg))
    return frozen, train, uids


def test_loso_metrics_and_bootstrap(small_cfg):
    panel, inds = make_synthetic_panel(small_cfg)
    train = panel[panel["round"].isin([4, 5])]
    uids = sorted(panel.district_uid.unique())
    preds = evaluate.loso_cv(train, inds, uids, small_cfg, verbose=False)
    assert set(evaluate.METHOD_ORDER) <= set(preds.method)
    res = evaluate.metric_table(preds, evaluate.METHOD_ORDER, bootstrap_reps=50, seed=1)
    overall = res[res.indicator_id == "OVERALL"].set_index("method")
    assert overall.loc["persistence", "skill"] == 0
    # data are generated with regression to the mean, so RTM must beat persistence
    assert overall.loc["rtm", "skill"] > 0.3
    assert (res.skill_lo <= res.skill + 1e-9).all() and (res.skill <= res.skill_hi + 1e-9).all()
    assert (overall.noise_mse > 0).all()


def test_freeze_roundtrip(small_cfg, tmp_path):
    panel, inds = make_synthetic_panel(small_cfg)
    frozen, train, _ = _freeze(small_cfg, panel, inds)
    path = tmp_path / "m.pkl"
    models.save_frozen(frozen, path)
    again = models.load_frozen(path)
    assert again.train_hash == models.training_data_hash(train)
    assert again.indicators == inds


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "core.autocrlf=false", *args], check=True, capture_output=True, text=True)


def _commit_and_tag(repo, msg):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    _git(repo, "tag", "-f", "-a", "freeze-v1", "-m", msg)


def test_evaluate_nfhs6_script_on_synthetic_data(small_cfg):
    root = Path(small_cfg["abs_paths"]["outputs_dir"]).parent
    panel, inds = make_synthetic_panel(small_cfg)
    (root / "processed").mkdir(exist_ok=True)
    panel_path = root / "processed" / "panel_model.parquet"
    panel.to_parquet(panel_path, index=False)
    frozen, _, _ = _freeze(small_cfg, panel, inds)
    models.save_frozen(frozen, root / "out" / "models" / "frozen_models.pkl")

    script = _load_script("evaluate_nfhs6")
    cfg_path = str(root / "config.yaml")
    run = ["--config", cfg_path, "--confirm-frozen", "--repo", str(root)]
    assert script.main(["--config", cfg_path]) == 2                 # refuses without confirmation
    assert script.main(run) == 6                                    # not a git repo

    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "freeze")
    assert script.main(run) == 6                                    # no freeze tag
    _git(root, "tag", "-a", "freeze-v1", "-m", "freeze")
    (root / "notes.txt").write_text("uncommitted\n")
    assert script.main(run) == 6                                    # dirty working tree
    (root / "notes.txt").unlink()

    # tampering with the NFHS-4/5 training data after the freeze must abort the test
    tampered = panel.copy()
    tampered.loc[tampered["round"] == 5, "value"] += 1.0
    tampered.to_parquet(panel_path, index=False)
    _commit_and_tag(root, "tampered")
    assert script.main(run) == 3
    assert not (root / "out" / freeze_guard.RUN_MARKER).exists()   # an aborted check is not a run

    panel.to_parquet(panel_path, index=False)
    _commit_and_tag(root, "restored")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    assert script.main(run) == 0
    res = pd.read_csv(root / "out" / "results_nfhs5to6.csv")
    assert "state_trend_oracle" not in set(res.method)
    assert set(res.method) == set(evaluate.HONEST_METHODS)
    assert (root / "out" / "skill_by_indicator_nfhs6.png").exists()
    assert (root / "out" / "results_nfhs5to6_stable.csv").exists()
    for name in ("secondary_nfhs5to6", "paired_tests_nfhs5to6", "paired_summary_nfhs5to6"):
        assert (root / "out" / f"{name}.csv").exists() and (root / "out" / f"{name}_stable.csv").exists()
    pt = pd.read_csv(root / "out" / "paired_tests_nfhs5to6.csv")
    assert set(pt.comparison) == {f"{a} vs {b}" for a, b in evaluate.PAIRED_COMPARISONS}
    log = json.loads((root / "out" / freeze_guard.RUN_MARKER).read_text())
    assert log["status"] == "completed" and log["head_commit"] == head and log["hash_check"]

    # the one-shot test refuses to run a second time
    _commit_and_tag(root, "after run")
    assert script.main(run) == 5


def test_predictions_use_only_round_t(small_cfg):
    """Changing round-6 values must not change any 5 -> 6 prediction."""
    panel, inds = make_synthetic_panel(small_cfg)
    frozen, _, uids = _freeze(small_cfg, panel, inds)
    def predict(p):
        t = models.build_transition(p, 5, 6, inds, uids, small_cfg, for_training=False)
        return models.predict_methods(frozen.fitted, t.inputs_only(), small_cfg).y_pred.to_numpy()
    p2 = panel.copy()
    p2.loc[p2["round"] == 6, "value"] = np.random.default_rng(0).uniform(0, 100, (p2["round"] == 6).sum())
    assert np.allclose(predict(panel), predict(p2), equal_nan=True)
