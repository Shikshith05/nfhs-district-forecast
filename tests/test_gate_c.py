"""Gate C tools (conformal, FDR surprises, forecastability) on synthetic data only."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from nfhs_forecast import conformal, forecastability, surprise
from tests.conftest import REPO


def _preds(n=3000, seed=0, shift=False, method="m", se_fn=None, resid_sd=None, n_ind=1, state_prefix="s"):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.6, 1.0, n) if shift else rng.uniform(0, 1, n)
    y_prev = 20 + 60 * x
    se = se_fn(x, rng) if se_fn else np.full(n, 2.0)
    sd = resid_sd(x) if resid_sd else np.full(n, 3.0)
    y_pred = y_prev + 5
    y_true = y_pred + rng.normal(0, sd)
    ind = np.array([f"i{k % n_ind}" for k in range(n)])
    return pd.DataFrame({"district_uid": [f"{state_prefix}{k % 20}|d{k}" for k in range(n)], "indicator_id": ind,
                         "method": method, "y_pred": y_pred, "y_prev": y_prev, "y_true": y_true,
                         "se_prev": se, "se_true": se, "state": [f"{state_prefix}{k % 20}" for k in range(n)]})


def test_conformal_quantile_and_weighted_equal_weights():
    s = np.arange(1, 101, dtype=float)
    assert conformal.conformal_quantile(s, 0.1) == 91.0           # ceil(101 * 0.9) = 91
    assert conformal.conformal_quantile(s[:5], 0.1) == np.inf     # too few points
    assert conformal.weighted_quantile(s, np.ones(100), 1.0, 0.1) == 91.0


def test_split_conformal_covers_exchangeable_data():
    cal, tgt = _preds(seed=1), _preds(seed=2)
    iv = conformal.conformal_intervals(cal, tgt, alpha=0.1, variants=("split", "normalised"))
    cov = conformal.coverage_table(iv, 0.1)
    for v in ("split", "normalised"):
        c = cov[(cov.variant == v) & (cov.indicator_id == "OVERALL")].coverage.iloc[0]
        assert abs(c - 0.9) < 0.03


def test_weighted_conformal_repairs_covariate_shift():
    sd = lambda x: 1 + 8 * x  # noqa: E731  residuals grow with x; the target has only large x
    cal = _preds(seed=3, resid_sd=sd)
    tgt = _preds(seed=4, shift=True, resid_sd=sd)
    iv = conformal.conformal_intervals(cal, tgt, alpha=0.1, variants=("split", "weighted"))
    cov = conformal.coverage_table(iv, 0.1).set_index(["variant", "indicator_id"]).coverage
    assert cov[("split", "OVERALL")] < 0.85                       # split under-covers under shift
    assert abs(cov[("weighted", "OVERALL")] - 0.9) < 0.04
    assert cov[("weighted", "OVERALL")] > cov[("split", "OVERALL")]


def test_surprises_find_planted_shocks_and_naive_prefers_noisy():
    se_fn = lambda x, rng: rng.choice([1.0, 8.0], len(x))  # noqa: E731
    cal = _preds(n=2000, seed=5, se_fn=se_fn)
    cal["y_true"] = cal.y_pred + np.random.default_rng(6).normal(0, np.sqrt(4 + cal.se_true ** 2))
    tgt = _preds(n=2000, seed=7, se_fn=se_fn)
    rng = np.random.default_rng(8)
    tgt["y_true"] = tgt.y_pred + rng.normal(0, np.sqrt(4 + tgt.se_true ** 2))
    planted = rng.choice(np.where(tgt.se_true == 1.0)[0], 20, replace=False)
    tgt.loc[planted, "y_true"] += 25
    tab, summ = surprise.detect_surprises(cal, tgt, "m", q=0.1)
    hits = set(tab.loc[tab.surprise, "district_uid"])
    planted_ids = set(tgt.loc[planted, "district_uid"])
    assert len(hits & planted_ids) >= 18
    assert len(hits - planted_ids) <= 4                           # FDR 10% of ~20
    assert summ["naive_topk_mean_se_true"] > summ["bh_mean_se_true"]
    assert tab.surprise_conformal.sum() >= 15


def test_forecastability_decomposition_and_district_table():
    rng = np.random.default_rng(0)
    rows = []
    hard = {"i0": 1.0, "i1": 4.0, "i2": 9.0}
    for d in range(60):
        for ind, sd in hard.items():
            y_prev = rng.uniform(20, 80)
            y_true = y_prev + rng.normal(0, 10)
            for m, yp in (("persistence", y_prev), ("m", y_true + rng.normal(0, sd))):
                rows.append(dict(district_uid=f"s|d{d}", state="s", indicator_id=ind, method=m, y_pred=yp,
                                 y_prev=y_prev, y_true=y_true, se_true=1.0))
    p = pd.DataFrame(rows)
    dec = forecastability.decompose(p, "m")
    assert dec["r2_indicator_only"] > 0.2 and dec["r2_district_only"] < dec["r2_indicator_only"]
    dt = forecastability.district_table(p, "m", min_indicators=3)
    assert (dt.ceiling >= dt.skill - 1e-9).mean() > 0.9 and len(dt) == 60
    it = forecastability.indicator_table(p, "m").set_index("indicator_id")
    assert it.loc["i0", "skill"] > it.loc["i2", "skill"]


def test_gate_c_script_dev_mode(tmp_path, cfg):
    out = tmp_path / "out"
    out.mkdir()
    rng = np.random.default_rng(0)
    parts = []
    for m, noise in (("persistence", 8.0), ("rtm", 3.0), ("fh_rtm", 3.0), ("elastic_net", 2.5), ("lightgbm", 2.8)):
        d = _preds(n=1200, seed=11, method=m, n_ind=3)
        d["y_pred"] = d.y_true + rng.normal(0, noise, len(d))
        parts.append(d)
    pd.concat(parts).to_parquet(out / "predictions_nfhs4to5.parquet", index=False)
    pd.DataFrame({"indicator_id": "OVERALL", "method": ["rtm", "elastic_net", "lightgbm"],
                  "rmse": [3.0, 2.4, 2.6]}).to_csv(out / "results_nfhs4to5.csv", index=False)
    raw = yaml.safe_load(yaml.safe_dump({k: v for k, v in cfg.items() if k != "abs_paths"}))
    raw["paths"].update({"outputs_dir": "out", "processed_dir": "processed"})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gc", REPO / "scripts" / "gate_c_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfgp = str(tmp_path / "config.yaml")
    assert mod.main(["--config", cfgp]) == 2                       # needs --dev or --predictions
    assert mod.main(["--config", cfgp, "--dev"]) == 0
    dev = out / "gate_c_dev"
    for f in ("coverage.csv", "surprises_elastic_net.csv", "surprises_rtm.csv", "forecastability_districts.csv",
              "forecastability_decomposition.json", "gate_c_summary.md"):
        assert (dev / f).exists(), f
    assert json.loads((dev / "surprise_summary.json").read_text())["primary_method"] == "elastic_net"
    cov = pd.read_csv(dev / "coverage.csv")
    split = cov[(cov.variant == "split") & (cov.indicator_id == "OVERALL")].coverage
    assert ((split > 0.8) & (split < 0.97)).all()
