"""Task A3: Fay-Herriot, spatial features, feature-set ablations, secondary metrics, paired tests."""

import numpy as np
import pandas as pd
import pytest

from nfhs_forecast import baselines as bl
from nfhs_forecast import evaluate, models, spatial
from tests.conftest import make_synthetic_panel


# ---------------------------------------------------------------------
# Fay-Herriot
# ---------------------------------------------------------------------
def _fh_data(seed=0, s2=16.0, n_per=80):
    rng = np.random.default_rng(seed)
    st = np.repeat(["a", "b", "c"], n_per)
    mu = {"a": 30.0, "b": 50.0, "c": 70.0}
    truth = np.array([mu[s] for s in st]) + rng.normal(0, np.sqrt(s2), len(st))
    se = rng.uniform(2, 8, len(st))
    return truth + rng.normal(0, se), se, st, truth


def test_fay_herriot_recovers_variance_and_beats_raw():
    y, se, st, truth = _fh_data()
    theta, gamma, info = bl.fay_herriot(y, se, st, return_info=True)
    assert 8 < info["sigma2_u"] < 32
    assert np.mean((theta - truth) ** 2) < 0.5 * np.mean((y - truth) ** 2)
    assert np.all((gamma >= 0) & (gamma <= 1))
    # noisier districts are shrunk more
    assert np.corrcoef(se, gamma)[0, 1] < -0.9


def test_fay_herriot_no_area_effect_shrinks_to_state_means():
    rng = np.random.default_rng(3)
    st = np.repeat(["a", "b"], 100)
    y = np.where(st == "a", 20.0, 60.0) + rng.normal(0, 5, 200)
    theta, gamma, info = bl.fay_herriot(y, np.full(200, 5.0), st, return_info=True)
    assert info["sigma2_u"] < 3
    assert np.std(theta[st == "a"]) < 0.5 * np.std(y[st == "a"])


def test_fay_herriot_passes_missing_through():
    y, se, st, _ = _fh_data(n_per=20)
    y[0], se[1] = np.nan, np.nan
    theta = bl.fay_herriot(y, se, st)
    assert np.isnan(theta[0]) and theta[1] == y[1]


# ---------------------------------------------------------------------
# Spatial neighbours
# ---------------------------------------------------------------------
def _square(code, x, y):
    ring = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]]
    return {"properties": {"censuscode": float(code)}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_adjacency_shared_edges_corner_and_island():
    feats = [_square(1, 0, 0), _square(2, 1, 0), _square(3, 0, 1), _square(4, 1, 1),
             _square(5, 10, 10)]                       # island
    adj = spatial.adjacency_from_features(feats, k_fallback=1)
    assert adj["1"] >= {"2", "3"} and "4" not in adj["1"]    # a corner (1 shared vertex) is not adjacency
    assert len(adj["5"]) == 1                                 # island -> nearest centroid
    assert spatial.norm_code(532.0) == "532" and spatial.norm_code(None) is None


def test_neighbour_means_use_codes_and_siblings():
    adj = {"1": {"2"}, "2": {"1"}, "3": set()}
    vals = pd.DataFrame({"x": [10.0, 20.0, 30.0, np.nan, 99.0]},
                        index=["s|a", "s|b", "s|a2", "s|c", "s|new"])
    codes = pd.Series({"s|a": "1", "s|b": "2", "s|a2": "1", "s|c": "2", "s|new": None})
    out = spatial.neighbour_means(vals, codes, adj)
    assert out.loc["s|a", "x"] == pytest.approx(25.0)        # b (code 2) + sibling a2 (code 1); c is NaN
    assert out.loc["s|b", "x"] == pytest.approx(20.0)        # a, a2 (code 1); c is NaN
    assert np.isnan(out.loc["s|new", "x"])                   # no code -> no spatial feature


# ---------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------
def test_feature_sets(cfg):
    panel, inds = make_synthetic_panel(cfg)
    train = panel[panel["round"] < 6]
    uids = sorted(panel.district_uid.unique())
    t = models.build_transition(train, 4, 5, inds, uids, cfg)
    spec = evaluate.feature_spec(t, inds, list(models.FEATURE_SETS))
    cols = {fs: set(models.make_features(t.prev, t.meta, "ind_a", spec, fs, t.nbr).columns)
            for fs in models.FEATURE_SETS}
    assert cols["own"] == {"prev__ind_a"}
    assert cols["cross"] == {f"prev__{i}" for i in inds}
    assert cols["cross_spatial"] == cols["cross"] | {"nbr_mean__target", "nbr_diff__target"}
    assert any(c.startswith("state__") for c in cols["full"])
    assert not any(c.startswith("state__") for c in cols["nostate"])
    assert "state_mean__target" in cols["nostate"] and any(c.startswith("region__") for c in cols["nostate"])
    with pytest.raises(ValueError):
        models.make_features(t.prev, t.meta, "ind_a", spec, "bogus")


def test_fitted_variants_and_new_baselines_predict(small_cfg):
    panel, inds = make_synthetic_panel(small_cfg)
    train = panel[panel["round"] < 6]
    uids = sorted(panel.district_uid.unique())
    t = models.build_transition(train, 4, 5, inds, uids, small_cfg)
    spec = evaluate.feature_spec(t, inds, ["full", "nostate"])
    fitted = models.fit_methods(t, small_cfg, spec)
    p = models.predict_methods(fitted, t.inputs_only(), small_cfg)
    expect = {"persistence", "national_trend", "state_trend", "rtm", "fh_rtm", "fay_herriot",
              "elastic_net", "lightgbm", "elastic_net__nostate", "lightgbm__nostate"}
    assert set(p.method) == expect
    nat = p[p.method == "national_trend"].set_index(["district_uid", "indicator_id"]).y_pred
    per = p[p.method == "persistence"].set_index(["district_uid", "indicator_id"]).y_pred
    shift = (nat - per).round(6)[(nat > 0) & (nat < 100)].groupby(level="indicator_id").nunique()
    assert (shift == 1).all()   # one national shift per indicator (unclipped rows)


def test_seen_state_cv_rejects_round6(cfg):
    panel, inds = make_synthetic_panel(cfg)
    with pytest.raises(models.LeakageError):
        evaluate.seen_state_cv(panel, inds, sorted(panel.district_uid.unique()), cfg)


# ---------------------------------------------------------------------
# Metrics and tests
# ---------------------------------------------------------------------
def _toy_preds(seed=0, n_states=8, per_state=10, inds=("i1", "i2"), noise_b=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_states):
        for d in range(per_state):
            for ind in inds:
                y_prev = rng.uniform(20, 80)
                y_true = y_prev + 5 + rng.normal(0, 3)
                preds = {"persistence": y_prev, "national_trend": y_prev + 5,
                         "rtm": y_prev + 5 + rng.normal(0, 2), "good": y_true + rng.normal(0, noise_b),
                         "same_as_rtm": None}
                preds["same_as_rtm"] = preds["rtm"]
                for m, v in preds.items():
                    rows.append(dict(district_uid=f"s{s}|d{d}", indicator_id=ind, method=m, y_pred=v,
                                     y_prev=y_prev, y_true=y_true, se_prev=2.0, se_true=2.0, state=f"s{s}"))
    return pd.DataFrame(rows)


def test_metric_table_mean_row_is_unweighted_mean():
    p = _toy_preds()
    t = evaluate.metric_table(p, ["persistence", "rtm", "good"], bootstrap_reps=50, seed=1)
    for m in ("rtm", "good"):
        per = t[(t.method == m) & ~t.indicator_id.isin(["OVERALL", "MEAN"])].skill.mean()
        row = t[(t.method == m) & (t.indicator_id == "MEAN")].iloc[0]
        assert row.skill == pytest.approx(per)
        assert row.skill_lo <= row.skill <= row.skill_hi


def test_secondary_metrics():
    p = _toy_preds()
    s = evaluate.secondary_metrics(p, ["persistence", "national_trend", "rtm", "good"], bootstrap_reps=50)
    nat = s[(s.method == "national_trend") & (s.indicator_id == "OVERALL")].iloc[0]
    assert nat.skill_vs_national == pytest.approx(0.0)
    good = s[(s.method == "good") & (s.indicator_id == "i1")].iloc[0]
    assert good.skill_vs_national > 0.5 and good.skill_vs_rtm > 0.5
    # noise share: (2^2 + 2^2) / var(change) with var(change) ~ 3^2
    sub = p[(p.method == "persistence") & (p.indicator_id == "i1")]
    assert good.noise_share_change == pytest.approx(8.0 / (sub.y_true - sub.y_prev).var())
    assert good.ceiling_vs_rtm < good.ceiling_vs_persistence


def test_bh_adjust_known_values():
    p = [0.01, 0.04, 0.03, 0.20, np.nan]
    adj = evaluate.bh_adjust(p)
    assert np.allclose(adj[:4], [0.04, 0.0533333, 0.0533333, 0.20])
    assert np.isnan(adj[4])


def test_paired_tests_identical_and_better():
    p = _toy_preds()
    t = evaluate.paired_tests(p, [("same_as_rtm", "rtm"), ("good", "rtm")], bootstrap_reps=200, seed=2)
    same = t[t.comparison == "same_as_rtm vs rtm"]
    assert np.allclose(same.diff_mse, 0) and not same.beats_reference.any()
    good = t[(t.comparison == "good vs rtm") & (t.indicator_id != "OVERALL")]
    assert good.beats_reference.all() and (good.diff_hi < 0).all()
    s = evaluate.paired_summary(t).set_index("comparison")
    assert s.loc["good vs rtm", "significantly_better"] == 2
    assert s.loc["same_as_rtm vs rtm", "significantly_better"] == 0
