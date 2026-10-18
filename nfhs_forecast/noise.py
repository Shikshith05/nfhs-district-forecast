"""Sampling-noise model: assumed standard errors for district estimates.

The fact sheets do not publish sample sizes, so every SE here rests on the
ASSUMED per-district sample sizes and design effect in config.yaml (noise:).
SE = 100 * sqrt(p (1 - p) / n * deff) for a percentage p.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def se_percent(value_pct, n, deff: float = 2.0):
    """Standard error (percentage points) of a percentage estimate.

    p is clipped to [0.5/n, 1 - 0.5/n] so that 0% and 100% estimates still
    get a positive SE (a continuity correction).
    """
    v = np.asarray(value_pct, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0):
        raise ValueError("n must be positive")
    if deff <= 0:
        raise ValueError("deff must be positive")
    p = np.clip(v / 100.0, 0.5 / n, 1 - 0.5 / n)
    se = 100.0 * np.sqrt(p * (1 - p) / n * deff)
    return np.where(np.isnan(v), np.nan, se)


def assumed_n(se_group: pd.Series, small_sample: pd.Series | None, cfg: dict) -> np.ndarray:
    """Per-row assumed n: the group's n, capped at small_sample_n for flagged rows."""
    ncfg = cfg["noise"]
    sizes = ncfg["sample_size"]
    unknown = set(se_group.dropna().unique()) - set(sizes)
    if unknown:
        raise KeyError(f"se_group(s) without an assumed sample size: {sorted(unknown)}")
    n = se_group.map(sizes).astype(float).to_numpy()
    if small_sample is not None:
        flag = small_sample.fillna(False).astype(bool).to_numpy()
        n = np.where(flag, np.minimum(n, ncfg["small_sample_n"]), n)
    return n


def add_se(panel: pd.DataFrame, codebook: pd.DataFrame, cfg: dict,
           deff: float | None = None) -> pd.DataFrame:
    """Add n_assumed and se columns for indicators that have an se_group."""
    deff = cfg["noise"]["deff"] if deff is None else deff
    groups = codebook.dropna(subset=["se_group"]).set_index("indicator_id").se_group
    out = panel.copy()
    out["se_group"] = out.indicator_id.map(groups)
    has = out.se_group.notna()
    out["n_assumed"] = np.nan
    out["se"] = np.nan
    if has.any():
        n = assumed_n(out.loc[has, "se_group"], out.loc[has, "small_sample_flag"], cfg)
        out.loc[has, "n_assumed"] = n
        out.loc[has, "se"] = se_percent(out.loc[has, "value"].to_numpy(), n, deff)
    return out
