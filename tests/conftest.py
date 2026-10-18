import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nfhs_forecast import load, noise  # noqa: E402
from nfhs_forecast.config import load_config  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO / "config.yaml")


@pytest.fixture(scope="session")
def raw45():
    return load.read_nfhs45(FIXTURES / "nfhs45_small.csv")


@pytest.fixture(scope="session")
def raw6():
    return load.read_nfhs6(FIXTURES / "nfhs6_small.csv")


def make_synthetic_panel(cfg, n_states=4, n_districts=8, seed=0):
    """Synthetic 3-round panel (rounds 4, 5, 6) with SEs, plus its codebook."""
    rng = np.random.default_rng(seed)
    indicators = {"ind_a": "children", "ind_b": "women", "ind_c": "mothers"}
    rows = []
    for s in range(n_states):
        state = f"state{s}"
        shift = rng.normal(0, 4)
        for d in range(n_districts):
            uid = f"{state}|district{d}"
            for ind in indicators:
                y4 = rng.uniform(20, 80)
                y5 = np.clip(0.8 * y4 + 15 + shift + rng.normal(0, 3), 0, 100)
                y6 = np.clip(0.8 * y5 + 15 + shift + rng.normal(0, 3), 0, 100)
                for rnd, v in ((4, y4), (5, y5), (6, y6)):
                    rows.append({"state": state, "district": f"district{d}", "district_uid": uid,
                                 "round": rnd, "indicator_id": ind, "value": round(v, 1),
                                 "small_sample_flag": (rnd == 6 and d == 0)})
    panel = pd.DataFrame(rows)
    panel["small_sample_flag"] = panel.small_sample_flag.astype("boolean")
    codebook = pd.DataFrame({"indicator_id": list(indicators), "se_group": list(indicators.values()),
                             "include": True})
    return noise.add_se(panel, codebook, cfg), list(indicators)


@pytest.fixture
def small_cfg(cfg, tmp_path):
    """Config with tiny models and paths inside tmp_path (written to disk for the scripts)."""
    raw = {k: v for k, v in cfg.items() if k != "abs_paths"}
    raw = yaml.safe_load(yaml.safe_dump(raw))
    raw["paths"].update({"processed_dir": "processed", "outputs_dir": "out", "models_dir": "out/models"})
    raw["models"]["lightgbm"]["n_estimators"] = 20
    raw["models"]["lightgbm"]["min_child_samples"] = 5
    raw["models"]["elastic_net"]["n_alphas"] = 10
    raw["models"]["elastic_net"]["inner_cv_folds"] = 3
    raw["evaluation"]["bootstrap_reps"] = 50
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)
