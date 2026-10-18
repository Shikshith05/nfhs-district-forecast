"""The NFHS-6 no-leakage rule must be enforced by code, not convention."""

import pandas as pd
import pytest

from nfhs_forecast import evaluate, models
from tests.conftest import make_synthetic_panel


def test_assert_no_round6_passes_on_rounds_4_5():
    models.assert_no_round6(pd.DataFrame({"round": [4, 5, 5]}))


@pytest.mark.parametrize("rounds", [[4, 5, 6], [6], ["6"], [4, None]])
def test_assert_no_round6_raises(rounds):
    with pytest.raises(models.LeakageError):
        models.assert_no_round6(pd.DataFrame({"round": rounds}))


def test_assert_no_round6_requires_round_column():
    with pytest.raises(models.LeakageError):
        models.assert_no_round6(pd.DataFrame({"value": [1.0]}))


def test_training_entry_points_reject_round6(cfg):
    panel, inds = make_synthetic_panel(cfg)
    uids = sorted(panel.district_uid.unique())
    with pytest.raises(models.LeakageError):
        models.build_transition(panel, 4, 5, inds, uids, cfg)          # full panel incl. round 6
    with pytest.raises(models.LeakageError):
        evaluate.loso_cv(panel, inds, uids, cfg, verbose=False)
    with pytest.raises(models.LeakageError):
        models.training_data_hash(panel)
    train = panel[panel["round"] < 6]
    with pytest.raises(models.LeakageError):
        models.build_transition(train, 5, 6, inds, uids, cfg)          # target round 6


def test_fit_methods_rejects_round6_target(cfg):
    panel, inds = make_synthetic_panel(cfg)
    uids = sorted(panel.district_uid.unique())
    t56 = models.build_transition(panel, 5, 6, inds, uids, cfg, for_training=False)
    spec = evaluate.feature_spec(t56, inds)
    with pytest.raises(models.LeakageError):
        models.fit_methods(t56, cfg, spec, fit_ml=False)
