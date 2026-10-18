"""Features, ML models and the frozen-model artifact.

HARD RULE: NFHS-6 (round 6) values never enter training, tuning, feature
engineering or model selection. Every training entry point calls
assert_no_round6() on its input.
"""

from __future__ import annotations

import hashlib
import pickle
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import TEST_ROUND
from . import baselines as bl
from .config import region_of

ML_METHODS = ("elastic_net", "lightgbm")


class LeakageError(AssertionError):
    """Raised when NFHS-6 data reaches a training code path."""


def assert_no_round6(df: pd.DataFrame, context: str = "training") -> None:
    """Fail loudly if any round-6 row is present in data meant for training."""
    if "round" not in df.columns:
        raise LeakageError(f"{context}: data has no 'round' column; cannot verify no-leakage rule")
    rounds = pd.to_numeric(df["round"], errors="coerce")
    if rounds.isna().any() or (rounds >= TEST_ROUND).any():
        bad = sorted(set(df.loc[rounds.isna() | (rounds >= TEST_ROUND), "round"].astype(str)))
        raise LeakageError(f"{context}: round(s) {bad} present; NFHS-{TEST_ROUND} must never be used "
                           "for training, tuning, feature engineering or model selection")


# =====================================================================
# Transition data (round t -> t+1, wide by indicator)
# =====================================================================
@dataclass
class Transition:
    t_prev: int
    t_next: int
    prev: pd.DataFrame                 # district_uid x indicator, values at t_prev
    se_prev: pd.DataFrame
    meta: pd.DataFrame                 # district_uid -> state, region
    next: pd.DataFrame | None = None   # values at t_next (None when only predicting)
    se_next: pd.DataFrame | None = None

    def subset(self, uids) -> "Transition":
        uids = list(uids)
        return Transition(self.t_prev, self.t_next, self.prev.loc[uids], self.se_prev.loc[uids],
                          self.meta.loc[uids],
                          None if self.next is None else self.next.loc[uids],
                          None if self.se_next is None else self.se_next.loc[uids])


def _wide(panel: pd.DataFrame, rnd: int, col: str, uids, indicators) -> pd.DataFrame:
    sub = panel[panel["round"] == rnd]
    return (sub.pivot_table(index="district_uid", columns="indicator_id", values=col, aggfunc="first")
            .reindex(index=uids, columns=indicators).astype(float))


def build_transition(panel: pd.DataFrame, t_prev: int, t_next: int | None, indicators, uids,
                     cfg: dict, for_training: bool = True) -> Transition:
    """Wide matrices for a transition. panel must already carry an `se` column.

    for_training=True (the default) enforces the no-leakage rule on the
    whole input frame. Only scripts/evaluate_nfhs6.py builds a 5 -> 6
    transition, with for_training=False, and only to score frozen models.
    """
    if for_training:
        assert_no_round6(panel, "build_transition(for_training=True)")
        if t_next is not None and t_next >= TEST_ROUND:
            raise LeakageError(f"cannot build a training transition with target round {t_next}")
    uids, indicators = list(uids), list(indicators)
    regions = region_of(cfg)
    states = pd.Series([u.split("|")[0] for u in uids], index=uids)
    meta = pd.DataFrame({"state": states, "region": states.map(regions).fillna("other")})
    tr = Transition(t_prev, t_next if t_next is not None else t_prev + 1,
                    _wide(panel, t_prev, "value", uids, indicators),
                    _wide(panel, t_prev, "se", uids, indicators), meta)
    if t_next is not None and (panel["round"] == t_next).any():
        tr.next = _wide(panel, t_next, "value", uids, indicators)
        tr.se_next = _wide(panel, t_next, "se", uids, indicators)
    return tr


# =====================================================================
# Features
# =====================================================================
@dataclass
class FeatureSpec:
    indicators: list[str]
    states: list[str]
    regions: list[str]


def make_features(prev: pd.DataFrame, meta: pd.DataFrame, target: str, spec: FeatureSpec) -> pd.DataFrame:
    """Previous-round values of all included indicators, the target's state mean,
    one-hot state and one-hot region. Uses round-t data only."""
    X = prev[spec.indicators].add_prefix("prev__").copy()
    X["state_mean__target"] = prev[target].groupby(meta.state).transform("mean")
    for s in spec.states:
        X[f"state__{s}"] = (meta.state == s).astype(float)
    for r in spec.regions:
        X[f"region__{r}"] = (meta.region == r).astype(float)
    return X


def fit_elastic_net(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: dict, seed: int) -> Pipeline:
    ecfg = cfg["models"]["elastic_net"]
    k = min(ecfg["inner_cv_folds"], len(np.unique(groups)))
    splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("enet", ElasticNetCV(l1_ratio=ecfg["l1_ratio"], alphas=ecfg["n_alphas"], cv=splits,
                              max_iter=ecfg["max_iter"], random_state=seed)),
    ])
    model.fit(X, y)
    return model


def fit_lightgbm(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int) -> LGBMRegressor:
    model = LGBMRegressor(**cfg["models"]["lightgbm"], random_state=seed, n_jobs=1, verbose=-1)
    model.fit(X, y)
    return model


# =====================================================================
# Fit / predict all methods
# =====================================================================
@dataclass
class FittedMethods:
    spec: FeatureSpec
    per_indicator: dict = field(default_factory=dict)


def fit_methods(train: Transition, cfg: dict, spec: FeatureSpec,
                fit_ml: bool = True) -> FittedMethods:
    """Fit state trend, RTM and the ML models on training pairs (t_next must be < 6)."""
    if train.t_next >= TEST_ROUND or train.next is None:
        raise LeakageError(f"fit_methods: training target round {train.t_next} is not allowed")
    seed = cfg["random_seed"]
    fitted = FittedMethods(spec)
    for ind in spec.indicators:
        y_prev, y_next = train.prev[ind], train.next[ind]
        ok = y_prev.notna() & y_next.notna()
        sc, nat = bl.fit_state_change(y_prev[ok], y_next[ok], train.meta.state[ok])
        entry = {"state_change": sc, "national_change": nat, "rtm": bl.fit_rtm(y_prev[ok], y_next[ok]),
                 "n_train": int(ok.sum())}
        if fit_ml:
            X = make_features(train.prev, train.meta, ind, spec)[ok]
            target = (y_next - y_prev)[ok].to_numpy()
            groups = train.meta.state[ok].to_numpy()
            entry["elastic_net"] = fit_elastic_net(X, target, groups, cfg, seed)
            entry["lightgbm"] = fit_lightgbm(X, target, cfg, seed)
        fitted.per_indicator[ind] = entry
    return fitted


def predict_methods(fitted: FittedMethods, data: Transition, cfg: dict,
                    include_eb: bool = True, deff_values: list[float] | None = None) -> pd.DataFrame:
    """Long table of predictions for round t+1: district_uid, indicator_id, method, y_pred.

    Uses only round-t information from `data` (prev, se_prev, meta).
    """
    damping = cfg["models"].get("state_trend_damping", 1.0)
    rows = []
    uids = data.prev.index
    for ind, e in fitted.per_indicator.items():
        y_prev = data.prev[ind].to_numpy()
        preds = {
            "persistence": bl.persistence(y_prev),
            "state_trend": bl.state_trend(y_prev, data.meta.state, e["state_change"],
                                          e["national_change"], damping),
            "rtm": bl.rtm(y_prev, *e["rtm"]),
        }
        if include_eb:
            preds["eb_shrinkage"] = bl.eb_shrinkage(y_prev, data.se_prev[ind].to_numpy(), data.meta.state)
        for m in ML_METHODS:
            if m in e:
                X = make_features(data.prev, data.meta, ind, fitted.spec)
                preds[m] = y_prev + e[m].predict(X)
        for m, p in preds.items():
            p = np.where(np.isnan(y_prev), np.nan, np.clip(p, 0, 100))
            rows.append(pd.DataFrame({"district_uid": uids, "indicator_id": ind, "method": m, "y_pred": p}))
    return pd.concat(rows, ignore_index=True)


# =====================================================================
# Frozen artifact
# =====================================================================
def training_data_hash(train_panel: pd.DataFrame) -> str:
    """Hash of the exact training data (rounds 4-5) so evaluate_nfhs6.py can detect drift."""
    assert_no_round6(train_panel, "training_data_hash")
    cols = ["district_uid", "indicator_id", "round", "value"]
    d = train_panel[cols].sort_values(cols[:3]).reset_index(drop=True)
    blob = d.to_csv(index=False, float_format="%.6g").encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class FrozenModels:
    fitted: FittedMethods
    core_districts: list[str]
    indicators: list[str]
    train_hash: str
    config_hash: str
    created_utc: str
    versions: dict


def freeze(train_panel: pd.DataFrame, train: Transition, cfg: dict, spec: FeatureSpec,
           config_hash: str) -> FrozenModels:
    """Fit all methods on the full NFHS-4 -> NFHS-5 data and wrap them for later use."""
    assert_no_round6(train_panel, "freeze")
    import lightgbm
    import sklearn
    fitted = fit_methods(train, cfg, spec)
    return FrozenModels(
        fitted=fitted, core_districts=list(train.prev.index), indicators=list(spec.indicators),
        train_hash=training_data_hash(train_panel), config_hash=config_hash,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        versions={"python": platform.python_version(), "sklearn": sklearn.__version__,
                  "lightgbm": lightgbm.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    )


def save_frozen(frozen: FrozenModels, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(frozen, f)


def load_frozen(path: str | Path) -> FrozenModels:
    with open(path, "rb") as f:
        return pickle.load(f)
