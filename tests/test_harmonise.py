"""Boundary harmonisation: crosswalk, components, weighted aggregation, SE propagation, flags."""

import copy

import numpy as np
import pandas as pd
import pytest

from nfhs_forecast import harmonise, noise


# ---------------------------------------------------------------------
# Synthetic name map and node panel
# ---------------------------------------------------------------------
def _nm_row(state, name, method="exact", flag=False, has4=True, has5=True, audit=1.0, in45=True, in6=True):
    key = state.lower()
    return dict(state=state, district=name, district_uid=f"{key}|{name}", state_key=key,
                state_nfhs6=state if in6 else None, district_nfhs6=name if in6 else None,
                state_nfhs45=state if in45 else None, district_nfhs45=name if in45 else None,
                method=method, score=100.0, boundary_change_nfhs6=flag, has_nfhs4=has4, has_nfhs5=has5,
                audit_pairs=20.0, audit_exact=audit, audit_within_0_5=audit, core=False, exclusion_reason="")


def make_name_map():
    rows = [
        _nm_row("S", "a"), _nm_row("S", "a2"), _nm_row("S", "a3"),             # stable
        _nm_row("S", "b", has4=False),                                          # changed 4->5, unchanged 5->6
        _nm_row("S", "p", flag=True, audit=np.nan),                             # retained name, flagged
        _nm_row("S", "c", method="unmatched", flag=True, has4=False, has5=False, audit=np.nan, in45=False),
        _nm_row("S", "q", method="not_in_nfhs6", audit=np.nan, in6=False),      # absent from NFHS-6
        _nm_row("S", "x", audit=0.1),                                           # audit failure
        _nm_row("T", "t1"), _nm_row("T", "t2"),
        _nm_row("T", "p1", flag=True, audit=np.nan), _nm_row("T", "p2", flag=True, audit=np.nan),
        _nm_row("T", "c2", method="unmatched", flag=True, has4=False, has5=False, audit=np.nan, in45=False),
    ]
    nm = pd.DataFrame(rows)
    nm["core"] = nm.district.isin(["a", "a2", "a3", "t1", "t2"])
    return nm


def make_node_panel(cfg, nm, seed=0, inds=("i1", "i2")):
    rng = np.random.default_rng(seed)
    rows = []
    for r in nm.itertuples():
        for ind in inds:
            base = rng.uniform(20, 80)
            for rnd in (4, 5, 6):
                present = ((rnd == 4 and r.has_nfhs4) or (rnd == 5 and r.has_nfhs5)
                           or (rnd == 6 and r.district_nfhs6 is not None))
                if present:
                    rows.append(dict(state=r.state, district=r.district, district_uid=r.district_uid, round=rnd,
                                     indicator_id=ind, value=round(base + 3 * (rnd - 4) + rng.normal(0, 2), 1),
                                     small_sample_flag=False, boundary_change_flag=False))
    p = pd.DataFrame(rows)
    p["small_sample_flag"] = p.small_sample_flag.astype("boolean")
    cb = pd.DataFrame({"indicator_id": list(inds), "se_group": "women", "include": True})
    return noise.add_se(p, cb, cfg)


@pytest.fixture
def hcfg(cfg):
    c = copy.deepcopy({k: v for k, v in cfg.items() if k != "abs_paths"})
    c["abs_paths"] = cfg["abs_paths"]
    return c


def run(hcfg, nm, panel, supplied=None):
    cw = harmonise.build_crosswalk(nm, hcfg)
    cw, problems = harmonise.apply_supplied_lineage(cw, (supplied or {}).get("lineage"), nm, hcfg)
    hp, units, cw = harmonise.harmonise_panel(panel, nm, cw, hcfg, supplied)
    return hp, units.set_index("unit_uid"), cw, problems


# ---------------------------------------------------------------------
# Aggregation and weights
# ---------------------------------------------------------------------
def test_aggregate_reproduces_parent_of_a_split():
    w = np.array([0.5, 0.3, 0.2])
    children = np.array([40.0, 60.0, 25.0])
    parent = float(np.dot(w, children))
    se = np.array([2.0, 3.0, 4.0])
    val, se_s, var_w, cov = harmonise.aggregate(children, se, w)
    assert val == pytest.approx(parent) and cov == pytest.approx(1.0)
    assert se_s == pytest.approx(np.sqrt(np.sum(w ** 2 * se ** 2)))
    assert var_w == 0.0


def test_aggregate_weight_uncertainty_and_missing_children():
    w = np.array([0.6, 0.4])
    val, se_s, var_w, _ = harmonise.aggregate([10.0, 30.0], [1.0, 1.0], w, weight_cv=0.1)
    assert val == pytest.approx(18.0)
    assert var_w == pytest.approx((0.1 * 0.6) ** 2 * 8 ** 2 + (0.1 * 0.4) ** 2 * 12 ** 2)
    # one child missing: renormalise if coverage is enough, else NaN
    val, _, _, cov = harmonise.aggregate([10.0, np.nan], [1.0, 1.0], w, min_coverage=0.5)
    assert val == pytest.approx(10.0) and cov == pytest.approx(0.6)
    val, _, _, _ = harmonise.aggregate([10.0, np.nan], [1.0, 1.0], w, min_coverage=0.8)
    assert np.isnan(val)
    with pytest.raises(ValueError):
        harmonise.aggregate([1.0, 2.0], [1.0, 1.0], [0.5, 0.6])


def test_population_weights_sum_to_one_and_require_all_members():
    pop = {"a": (300.0, False), "b": (100.0, True)}
    w, est = harmonise.population_weights(["a", "b"], pop)
    assert w.sum() == pytest.approx(1.0) and w.tolist() == pytest.approx([0.75, 0.25]) and est
    assert harmonise.population_weights(["a", "zz"], pop) is None


def test_connected_components_split_merge_chain():
    comp = harmonise.connected_components([("5:p", "6:p"), ("5:p", "6:c"),           # split
                                           ("5:m1", "6:m"), ("5:m2", "6:m"),         # merge
                                           ("5:u", "6:u")])                          # unchanged
    assert comp["5:p"] == comp["6:p"] == comp["6:c"]
    assert comp["5:m1"] == comp["5:m2"] == comp["6:m"]
    assert len({comp["5:p"], comp["5:m1"], comp["5:u"]}) == 3


# ---------------------------------------------------------------------
# Crosswalk from repo evidence
# ---------------------------------------------------------------------
def test_crosswalk_relations_and_confidence(hcfg):
    cw = harmonise.build_crosswalk(make_name_map(), hcfg).set_index(["transition", "child_uid"])
    assert cw.loc[("5to6", "s|a"), "relation"] == "unchanged" and cw.loc[("5to6", "s|a"), "confidence"] == "high"
    assert cw.loc[("5to6", "s|p"), "relation"] == "retained_name_boundary_changed"
    assert cw.loc[("5to6", "s|p"), "confidence"] == "medium"
    # the only flagged retained-name district in state S is a low-confidence candidate parent of c
    assert cw.loc[("5to6", "s|c"), "parent_uid"] == "s|p" and cw.loc[("5to6", "s|c"), "confidence"] == "low"
    # two candidates in state T: unresolved, no parent invented
    assert cw.loc[("5to6", "t|c2"), "confidence"] == "unresolved" and pd.isna(cw.loc[("5to6", "t|c2"), "parent_uid"])
    assert cw.loc[("4to5", "s|b"), "confidence"] == "unresolved"
    assert cw.source.notna().all() and cw.confidence.isin(harmonise.CONFIDENCE_RANK).all()


# ---------------------------------------------------------------------
# Units and the harmonised panel
# ---------------------------------------------------------------------
def test_units_without_supplied_files(hcfg):
    nm = make_name_map()
    hp, u, cw, _ = run(hcfg, nm, make_node_panel(hcfg, nm))
    assert u.loc["s|a", "unit_type"] == "identity" and u.loc["s|a", "stable_unit"] and u.loc["s|a", "train_unit"]
    assert u.loc["s|b", "test_unit"] and not u.loc["s|b", "train_unit"] and not u.loc["s|b", "stable_unit"]
    assert u.loc["s|p", "unit_type"] == "inherited_remnant" and u.loc["s|p", "train_unit"]
    assert not u.loc["s|p", "stable_unit"]
    assert u.loc["s|q", "train_unit"] and not u.loc["s|q", "test_unit"]
    assert "s|x" not in u.index                       # audit failure: excluded everywhere
    assert "s|c" not in u.index and "t|c2" not in u.index   # low / unresolved lineage is not used
    # the remnant's round-6 value carries extra variance; its training rows do not
    p = hp[hp.district_uid == "s|p"].set_index(["round", "indicator_id"])
    assert (p.loc[6, "extra_var"] > 0).all() and (p.loc[[4, 5], "extra_var"] == 0).all()
    assert np.allclose(p.loc[6, "se"] ** 2, p.loc[6, "se_sampling"] ** 2 + p.loc[6, "extra_var"])
    assert p.loc[6, "harmonised"].all() and not p.loc[5, "harmonised"].any()
    assert cw.loc[cw.child_uid.eq("s|c"), "used"].eq(False).all()


def test_review_lists_every_low_or_unresolved_case(hcfg):
    nm = make_name_map()
    _, u, cw, problems = run(hcfg, nm, make_node_panel(hcfg, nm))
    rv = harmonise.review_table(cw, u.reset_index(), problems, hcfg)
    low = cw[cw.confidence.isin(["low", "unresolved"])]
    assert len(low) and set(zip(low.transition, low.child_uid)) <= set(zip(rv.transition, rv.child_uid))
    assert rv.suggested_action.str.len().gt(0).all()


def test_supplied_split_with_population_weights_aggregates(hcfg):
    nm = make_name_map()
    panel = make_node_panel(hcfg, nm)
    lineage = pd.DataFrame([
        dict(transition="5to6", state="S", parent_district="p", child_district="p", relation="split",
             weight="", source="LGD test", confidence="high"),
        dict(transition="5to6", state="S", parent_district="p", child_district="c", relation="split",
             weight="", source="LGD test", confidence="high")])
    pop = pd.DataFrame(dict(vintage=["nfhs6", "nfhs6"], state=["S", "S"], district=["p", "c"],
                            population=[700.0, 300.0], estimated=[False, False]))
    hp, u, cw, problems = run(hcfg, nm, panel, {"lineage": lineage, "population": pop})
    assert not problems
    agg = u[u.unit_type == "aggregated"]
    assert len(agg) == 1
    uid = agg.index[0]
    assert agg.loc[uid, "test_unit"] and not agg.loc[uid, "train_unit"]
    assert u.loc["s|p", "train_unit"] and not u.loc["s|p", "test_unit"]   # parent stays a training unit
    for ind in ("i1", "i2"):
        v6 = panel[(panel["round"] == 6) & (panel.indicator_id == ind)].set_index("district_uid")
        got = hp[(hp.district_uid == uid) & (hp["round"] == 6) & (hp.indicator_id == ind)].iloc[0]
        assert got.value == pytest.approx(0.7 * v6.loc["s|p", "value"] + 0.3 * v6.loc["s|c", "value"])
        assert got.se == pytest.approx(np.sqrt(0.49 * v6.loc["s|p", "se"] ** 2 + 0.09 * v6.loc["s|c", "se"] ** 2))
        # rounds 4/5 of the component are the parent's own values (one NFHS-5 member)
        v5 = panel[(panel["round"] == 5) & (panel.indicator_id == ind) & (panel.district_uid == "s|p")].value.iloc[0]
        assert hp[(hp.district_uid == uid) & (hp["round"] == 5) & (hp.indicator_id == ind)].value.iloc[0] == v5
    assert cw.loc[cw.child_uid.eq("s|c") & cw.transition.eq("5to6"), "used"].all()


def test_supplied_split_without_weights_inherits(hcfg):
    nm = make_name_map()
    panel = make_node_panel(hcfg, nm)
    lineage = pd.DataFrame([dict(transition="5to6", state="S", parent_district="p", child_district="c",
                                 relation="split", weight="", source="LGD test", confidence="high")])
    hp, u, _, _ = run(hcfg, nm, panel, {"lineage": lineage})
    assert u.loc["s|c", "unit_type"] == "inherited_child" and u.loc["s|c", "parent_uid"] == "s|p"
    assert u.loc["s|p", "unit_type"] == "inherited_remnant"
    c = hp[hp.district_uid == "s|c"].set_index(["round", "indicator_id"])
    p = hp[hp.district_uid == "s|p"].set_index(["round", "indicator_id"])
    assert np.allclose(c.loc[5, "value"], p.loc[5, "value"])            # inputs inherited from the parent
    assert (c.loc[5, "value_method"] == "inherited").all()
    share = hcfg["harmonise"]["inherit_var_share"]
    assert np.allclose(c.loc[6, "extra_var"] / share["child"], p.loc[6, "extra_var"] / share["remnant"])


def test_unknown_supplied_name_is_reported(hcfg):
    nm = make_name_map()
    lineage = pd.DataFrame([dict(transition="5to6", state="S", parent_district="nowhere", child_district="c",
                                 relation="split", weight="", source="x", confidence="high")])
    _, problems = harmonise.apply_supplied_lineage(harmonise.build_crosswalk(nm, hcfg), lineage, nm, hcfg)
    assert problems and "not found" in problems[0]["review_reason"]


def test_inherited_nfhs4_from_census2011_parent(hcfg):
    nm = make_name_map()
    panel = make_node_panel(hcfg, nm)
    lineage = pd.DataFrame([dict(transition="4to5", state="S", parent_district="old", child_district="b",
                                 relation="split", weight="", source="Census test", confidence="high")])
    n4 = pd.DataFrame(dict(state=["S", "S"], district=["old", "old"], indicator_id=["i1", "i2"],
                           value=[33.0, 44.0]))
    hp, u, _, _ = run(hcfg, nm, panel, {"lineage": lineage, "nfhs4_census2011": n4})
    assert u.loc["s|b", "train_unit"] and u.loc["s|b", "nfhs4_inherited"]
    b4 = hp[(hp.district_uid == "s|b") & (hp["round"] == 4)].set_index("indicator_id")
    assert b4.loc["i1", "value"] == 33.0 and b4.loc["i2", "value"] == 44.0
    assert (b4.value_method == "inherited").all() and (b4.extra_var > 0).all()


def test_no_nfhs6_values_used_for_units_or_extra_variance(hcfg):
    """Unit membership, eligibility and extra variance must not depend on NFHS-6 values."""
    nm = make_name_map()
    panel = make_node_panel(hcfg, nm)
    p2 = panel.copy()
    r6 = p2["round"] == 6
    p2.loc[r6, "value"] = np.random.default_rng(5).uniform(0, 100, r6.sum())
    p2 = noise.add_se(p2.drop(columns=["se", "se_group", "n_assumed"]),
                      pd.DataFrame({"indicator_id": ["i1", "i2"], "se_group": "women"}), hcfg)
    h1, u1, _, _ = run(hcfg, nm, panel)
    h2, u2, _, _ = run(hcfg, nm, p2)
    pd.testing.assert_frame_equal(u1, u2)
    key = ["district_uid", "round", "indicator_id"]
    a, b = h1.set_index(key).sort_index(), h2.set_index(key).sort_index()
    assert np.allclose(a.extra_var, b.extra_var)
    assert a.loc[a.index.get_level_values("round") < 6].value.equals(b.loc[b.index.get_level_values("round") < 6].value)


def test_state_reconstruction_check():
    vals = pd.DataFrame(dict(district_uid=["s|a", "s|b"], indicator_id="i1", value=[10.0, 20.0]))
    pops = pd.DataFrame(dict(district_uid=["s|a", "s|b"], population=[1.0, 3.0]))
    pub = pd.DataFrame(dict(state=["s"], indicator_id=["i1"], value=[17.9]))
    out = harmonise.check_state_reconstruction(vals, pops, pub, tol=0.5)
    assert out.rebuilt.iloc[0] == pytest.approx(17.5) and out.within_tol.iloc[0]
