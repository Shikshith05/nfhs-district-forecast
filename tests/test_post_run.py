"""Post-run analysis library on synthetic data (no NFHS-6 values are involved)."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from nfhs_forecast import post_run as pr
from tests.conftest import REPO


def _period(seed, slope=0.8, shift=5.0, noise=3.0, n_states=12, per=25, inds=("i1", "i2")):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_states):
        for d in range(per):
            for ind in inds:
                x = rng.uniform(20, 80)
                y = 50 + slope * (x - 50) + shift + rng.normal(0, noise)
                for m, yp in (("persistence", x), ("rtm", 50 + 0.8 * (x - 50) + 5)):
                    rows.append(dict(district_uid=f"s{s}|d{d}", indicator_id=ind, method=m, y_pred=yp, y_prev=x,
                                     y_true=y, se_prev=2.0, se_true=2.0, state=f"s{s}"))
    return pd.DataFrame(rows)


def test_structural_break_detects_planted_change_only():
    cal = _period(0)
    same = pr.structural_break(cal, _period(1), "rtm", reps=300, seed=1)
    assert not same.break_any.any()
    changed = pr.structural_break(cal, _period(2, slope=0.4, shift=-3.0), "rtm", reps=300, seed=1)
    assert changed.slope_break.all() and changed.shift_break.all()
    r = changed.iloc[0]
    assert r.slope_45 == pytest.approx(0.8, abs=0.05) and r.slope_56 == pytest.approx(0.4, abs=0.05)
    assert r.slope_diff_lo < r.slope_diff < r.slope_diff_hi


def test_unit_groups_and_comparison():
    g = pr.unit_group(pd.Series(["identity", "identity", "inherited_remnant", "identity_train_only"]),
                      pd.Series([True, False, False, False]))
    assert g.tolist() == ["stable", "changed_4to5", "flagged", "changed_4to5"]
    res = pd.DataFrame({"indicator_id": ["OVERALL"] * 3 + ["i1"] * 3 + ["i2"] * 3,
                        "method": ["persistence", "rtm", "elastic_net"] * 3,
                        "skill": [0, 0.6, 0.65, 0, 0.5, 0.7, 0, 0.4, 0.3], "rmse": [1.0] * 9,
                        "skill_lo": [0.0] * 9, "skill_hi": [1.0] * 9})
    res56 = res.assign(skill=res.skill * 0.5)
    comp, summ = pr.comparison(res, res56, "elastic_net")
    assert summ["best_method_45"] == summ["best_method_56"] == "elastic_net"
    assert summ["spearman_methods_overall_skill"] == pytest.approx(1.0)
    assert comp.loc[(comp.method == "rtm") & (comp.indicator_id == "i1"), "skill_change"].iloc[0] == pytest.approx(-0.25)


def test_out_marks_dryrun(tmp_path):
    out = pr.Out(tmp_path, dryrun=True)
    p = out.csv("x.csv", pd.DataFrame({"a": [1]}))
    assert p.name == "DRYRUN_x.csv" and pd.read_csv(p).run_mode.eq("DRYRUN").all()
    assert json.loads(out.json("s.json", {"k": 1}).read_text())["run_mode"] == "DRYRUN"
    assert out.md("r.md", "hello").read_text(encoding="utf-8").startswith("> **DRYRUN")
    real = pr.Out(tmp_path / "real")
    assert real.csv("x.csv", pd.DataFrame({"a": [1]})).name == "x.csv"


def test_post_run_real_mode_refuses_dryrun_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location("post_run_script", REPO / "scripts" / "post_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    marker = tmp_path / "run.json"
    t = pd.DataFrame({"run_mode": ["DRYRUN"]})
    assert "DRYRUN" in mod.check_real_inputs(tmp_path / "p.parquet", marker, t)
    assert "DRYRUN" in mod.check_real_inputs(tmp_path / "dryrun" / "p.parquet", marker, pd.DataFrame())
    assert "not found" in mod.check_real_inputs(tmp_path / "p.parquet", marker, pd.DataFrame())
    marker.write_text(json.dumps({"status": "started"}))
    assert "completed" in mod.check_real_inputs(tmp_path / "p.parquet", marker, pd.DataFrame())
    marker.write_text(json.dumps({"status": "completed"}))
    assert mod.check_real_inputs(tmp_path / "p.parquet", marker, pd.DataFrame()) is None
