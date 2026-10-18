import numpy as np
import pandas as pd
import pytest

from nfhs_forecast import noise


def test_se_percent_formula():
    # p = 0.5, n = 100, deff = 2 -> 100 * sqrt(0.25 / 100 * 2)
    assert noise.se_percent(50, 100, 2.0) == pytest.approx(100 * np.sqrt(0.25 / 100 * 2))
    assert noise.se_percent(20, 400, 1.0) == pytest.approx(100 * np.sqrt(0.2 * 0.8 / 400))


def test_se_percent_scales_with_deff_and_n():
    base = noise.se_percent(30, 200, 2.0)
    assert noise.se_percent(30, 200, 3.0) == pytest.approx(base * np.sqrt(1.5))
    assert noise.se_percent(30, 800, 2.0) == pytest.approx(base / 2)


def test_se_percent_boundaries_are_positive_and_nan_passthrough():
    se = noise.se_percent(np.array([0.0, 100.0, np.nan]), np.array([100, 100, 100]), 2.0)
    assert se[0] > 0 and se[1] > 0 and se[0] == pytest.approx(se[1])
    assert np.isnan(se[2])


def test_se_percent_rejects_bad_inputs():
    with pytest.raises(ValueError):
        noise.se_percent(50, 0, 2.0)
    with pytest.raises(ValueError):
        noise.se_percent(50, 100, 0)


def test_assumed_n_caps_small_sample_rows(cfg):
    groups = pd.Series(["children", "children", "women"])
    flags = pd.Series([False, True, pd.NA], dtype="boolean")
    n = noise.assumed_n(groups, flags, cfg)
    sizes = cfg["noise"]["sample_size"]
    assert n.tolist() == [sizes["children"], cfg["noise"]["small_sample_n"], sizes["women"]]


def test_assumed_n_unknown_group(cfg):
    with pytest.raises(KeyError):
        noise.assumed_n(pd.Series(["aliens"]), None, cfg)


def test_add_se_inflates_flagged_rows(cfg):
    panel = pd.DataFrame({"indicator_id": ["x", "x", "y"], "value": [40.0, 40.0, 40.0],
                          "small_sample_flag": pd.array([False, True, False], dtype="boolean")})
    cb = pd.DataFrame({"indicator_id": ["x"], "se_group": ["women"]})
    out = noise.add_se(panel, cb, cfg)
    assert out.se.iloc[1] > out.se.iloc[0]
    assert np.isnan(out.se.iloc[2])        # no se_group -> no SE
