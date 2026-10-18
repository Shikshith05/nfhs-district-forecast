"""Synthetic end-to-end run: LOSO CV, metrics, freeze, and the NFHS-6 evaluation script.

Uses synthetic data only; the real NFHS-6 evaluation is never run by the tests.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from nfhs_forecast import evaluate, models
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
    frozen = models.freeze(train, trans, cfg, evaluate.feature_spec(trans, inds), config_hash(cfg))
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


def test_evaluate_nfhs6_script_on_synthetic_data(small_cfg):
    root = Path(small_cfg["abs_paths"]["outputs_dir"]).parent
    panel, inds = make_synthetic_panel(small_cfg)
    (root / "processed").mkdir(exist_ok=True)
    panel.to_parquet(root / "processed" / "panel_model.parquet", index=False)
    frozen, _, _ = _freeze(small_cfg, panel, inds)
    models.save_frozen(frozen, root / "out" / "models" / "frozen_models.pkl")

    script = _load_script("evaluate_nfhs6")
    cfg_path = str(root / "config.yaml")
    assert script.main(["--config", cfg_path]) == 2                 # refuses without confirmation
    assert script.main(["--config", cfg_path, "--confirm-frozen"]) == 0
    res = pd.read_csv(root / "out" / "results_nfhs5to6.csv")
    assert "state_trend_oracle" not in set(res.method)
    assert set(res.method) == set(evaluate.HONEST_METHODS)
    assert (root / "out" / "skill_by_indicator_nfhs6.png").exists()

    # tampering with the NFHS-4/5 training data after the freeze must abort the test
    panel.loc[panel["round"] == 5, "value"] += 1.0
    panel.to_parquet(root / "processed" / "panel_model.parquet", index=False)
    assert script.main(["--config", cfg_path, "--confirm-frozen"]) == 3


def test_predictions_use_only_round_t(small_cfg):
    """Changing round-6 values must not change any 5 -> 6 prediction."""
    panel, inds = make_synthetic_panel(small_cfg)
    frozen, _, uids = _freeze(small_cfg, panel, inds)
    def predict(p):
        t = models.build_transition(p, 5, 6, inds, uids, small_cfg, for_training=False)
        return models.predict_methods(frozen.fitted, models.Transition(5, 6, t.prev, t.se_prev, t.meta),
                                      small_cfg).y_pred.to_numpy()
    p2 = panel.copy()
    p2.loc[p2["round"] == 6, "value"] = np.random.default_rng(0).uniform(0, 100, (p2["round"] == 6).sum())
    assert np.allclose(predict(panel), predict(p2), equal_nan=True)
