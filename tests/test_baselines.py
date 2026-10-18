import numpy as np
import pytest

from nfhs_forecast import baselines as bl


def test_persistence_returns_copy():
    y = np.array([10.0, np.nan, 30.0])
    p = bl.persistence(y)
    assert np.array_equal(p, y, equal_nan=True)
    p[0] = 99
    assert y[0] == 10.0


def test_state_trend_uses_state_change_and_national_fallback():
    sc, nat = bl.fit_state_change([10, 20, 30, 40], [15, 25, 31, 41], ["a", "a", "b", "b"])
    assert sc == {"a": 5.0, "b": 1.0} and nat == pytest.approx(3.0)
    p = bl.state_trend([50, 50, 50], ["a", "b", "zz"], sc, nat)
    assert p.tolist() == [55.0, 51.0, 53.0]
    assert bl.state_trend([50], ["a"], sc, nat, damping=0.5).tolist() == [52.5]


def test_state_change_leave_one_out():
    chg = bl.state_change_leave_one_out([0, 0, 0, 0], [1, 3, 5, 10], ["a", "a", "a", "b"])
    assert chg[0] == pytest.approx(4.0)      # mean of 3 and 5
    assert chg[2] == pytest.approx(2.0)      # mean of 1 and 3
    assert chg[3] == pytest.approx(np.mean([1, 3, 5, 10]))   # alone in state -> national mean


def test_rtm_recovers_linear_relation():
    x = np.linspace(10, 90, 50)
    a, b = bl.fit_rtm(x, 5 + 0.7 * x)
    assert a == pytest.approx(5) and b == pytest.approx(0.7)
    assert bl.rtm([50], a, b)[0] == pytest.approx(40.0)


def test_rtm_handles_nan_and_degenerate():
    a, b = bl.fit_rtm([10, np.nan, 30, 50], [12, 99, 32, np.nan])
    assert b == pytest.approx(1.0) and a == pytest.approx(2.0)


def test_eb_shrinks_toward_state_mean_more_when_noisier():
    rng = np.random.default_rng(1)
    states = np.repeat(["a", "b"], 20)
    y = np.concatenate([rng.normal(30, 5, 20), rng.normal(60, 5, 20)])
    se = np.full(40, 3.0)
    se[0] = 15.0                                   # one very noisy district
    theta, B, tau2 = bl.eb_shrinkage(y, se, states, return_weights=True)
    m_a = y[:20].mean()
    assert tau2 > 0
    assert np.all(np.abs(theta[:20] - m_a) <= np.abs(y[:20] - m_a) + 1e-9)   # never moves away
    assert B[0] < B[1]                             # noisier -> smaller weight on own value
    assert abs(theta[0] - m_a) < abs(y[0] - m_a)


def test_eb_no_noise_leaves_values_and_missing_se_passthrough():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    st = np.array(["a", "a", "b", "b"])
    assert np.allclose(bl.eb_shrinkage(y, np.full(4, 1e-9), st), y)
    out = bl.eb_shrinkage(y, np.array([1.0, np.nan, 1.0, 1.0]), st)
    assert out[1] == 20.0
