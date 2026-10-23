"""District boundary harmonisation across NFHS rounds (DECISIONS D27-D31).

Three boundary vintages exist:
  census2011  NFHS-4 sampling frame (640 Census 2011 districts)
  nfhs5       the 707 NFHS-5 districts; India.csv gives NFHS-4 values on these
              boundaries only where IIPS judged them comparable (has_nfhs4)
  nfhs6       the NFHS-6 districts (boundary_change_single_round flags changes)

Lineage crosswalk
-----------------
`build_crosswalk` lists one edge per (parent, child) district pair for the
transitions 4to5 (census2011 -> nfhs5) and 5to6 (nfhs5 -> nfhs6), each with
its relation, SOURCE and CONFIDENCE (high / medium / low / unresolved).
Edges come from evidence in the repository only:
  * 5to6 `unchanged`: names matched within state and NFHS-6 flag False
    (high; alias matches are high only if the NFHS-5 audit confirmed them);
  * 5to6 `retained_name_boundary_changed`: matched name, NFHS-6 flag True
    (medium: the district continues but its territory changed; the districts
    carved out of it are not recorded in the repo);
  * 5to6 new NFHS-6 districts: the parent is NOT in the repo. If exactly one
    flagged retained-name district exists in the same state it is recorded as
    a low-confidence candidate; otherwise the edge is unresolved;
  * 4to5 `unchanged`: India.csv has NFHS-4 values (IIPS comparability);
  * 4to5 districts without NFHS-4: lineage and NFHS-4 values on Census 2011
    boundaries are not in the repo (unresolved).
Lineage is never invented: anything below `harmonise.min_confidence` is not
used and is written to outputs/crosswalk_review.csv for a human. Official
lineage (LGD / Census), populations and NFHS-4 Census-2011 values can be
supplied as CSVs (data/lineage/README.md); supplied rows replace repo rows.

Analysis units
--------------
The used 5to6 edges form a bipartite graph between NFHS-5 and NFHS-6
districts; each connected component is one analysis unit:
  identity           1 NFHS-5 district == 1 NFHS-6 district;
  inherited_remnant  1 NFHS-5 district, its NFHS-6 namesake has lost (or
                     gained) territory; NFHS-6 value scored against the
                     forecast from the whole NFHS-5 district, extra variance;
  inherited_child    split without population weights: each NFHS-6 child is
                     scored against its parent's forecast (inputs copied from
                     the parent, flagged), extra variance;
  aggregated         split / merge / multi-parent with population weights:
                     every round is re-aggregated to the component,
                         value = sum_i w_i y_i,
                         se^2  = sum_i w_i^2 se_i^2 + sum_i (cv w_i)^2 (y_i - value)^2
                     (the second term is weight uncertainty, cv = harmonise.weight_cv).
Training (NFHS-4 -> NFHS-5) uses NFHS-5 districts whose NFHS-4 and NFHS-5
values refer to the same boundaries; NFHS-4 values inherited from a Census
2011 parent (supplied data only) are flagged and carry extra variance.

Leakage contract: weights come from supplied Census/LGD populations, never
from NFHS-6 outcomes. The extra variance for inherited values is a share of
the within-state between-district variance estimated on NFHS-5 values only.
NFHS-6 values are aggregated only so they can be scored; unit membership and
eligibility use NFHS-6 metadata (names, boundary flags), never NFHS-6 values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import noise
from .clean import normalise_name

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "unresolved": 0}
MATCHED = ("exact", "exact_nospace", "alias", "fuzzy")
CROSSWALK_COLUMNS = ["transition", "state", "parent_vintage", "parent_uid", "parent_name", "child_vintage",
                     "child_uid", "child_name", "relation", "weight", "source", "confidence", "used",
                     "review_reason"]


# =====================================================================
# Supplied files (optional)
# =====================================================================
def _read_optional(path) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    return df if len(df) else None


def load_supplied(cfg: dict) -> dict:
    """Read the optional supplied lineage / population / NFHS-4 Census-2011 files."""
    P = cfg["abs_paths"]
    out = {"lineage": _read_optional(P.get("lineage_supplied")),
           "population": _read_optional(P.get("population_supplied")),
           "nfhs4_census2011": _read_optional(P.get("nfhs4_census2011_supplied"))}
    if out["population"] is not None:
        pop = out["population"]
        pop["population"] = pd.to_numeric(pop.population, errors="coerce")
        pop["estimated"] = pop.get("estimated", "true").astype(str).str.lower().isin(["true", "1", "yes"])
    if out["nfhs4_census2011"] is not None:
        out["nfhs4_census2011"]["value"] = pd.to_numeric(out["nfhs4_census2011"].value, errors="coerce")
    return out


# =====================================================================
# Crosswalk
# =====================================================================
def _audit_status(nm: pd.DataFrame, thr: float) -> pd.Series:
    a = nm.audit_within_0_5
    return pd.Series(np.select([a.isna(), a >= thr], ["unaudited", "pass"], "fail"), index=nm.index)


def build_crosswalk(name_map: pd.DataFrame, cfg: dict, codes: dict | None = None) -> pd.DataFrame:
    """Lineage edges supported by the repository (see module docstring).

    codes: optional {district_uid: Census 2011 code} (from India.csv), used
    only to annotate districts whose boundaries changed between NFHS-4 and -5.
    """
    thr = cfg["matching"]["audit_min_district_agreement"]
    nm = name_map.copy()
    nm["district_code"] = nm.district_uid.map(codes or {})
    nm["audit_status"] = _audit_status(nm, thr)
    rows = []
    matched = nm[nm.method.isin(MATCHED)]
    for r in matched.itertuples():
        if r.boundary_change_nfhs6:
            rel, conf = "retained_name_boundary_changed", "medium"
            src = (f"name match ({r.method}) + NFHS-6 boundary_change_single_round=True; districts carved "
                   "out of / added to it are not recorded in the repo")
        else:
            rel = "unchanged"
            conf = "high" if (r.method != "alias" or r.audit_status == "pass") else "medium"
            src = f"name match ({r.method}) + NFHS-6 boundary_change_single_round=False"
            if r.method == "alias":
                src += f"; alias in config.yaml, NFHS-5 audit {r.audit_status}"
        rows.append(dict(transition="5to6", state=r.state, parent_vintage="nfhs5", parent_uid=r.district_uid,
                         parent_name=r.district_nfhs45, child_vintage="nfhs6", child_uid=r.district_uid,
                         child_name=r.district_nfhs6, relation=rel, weight=np.nan, source=src, confidence=conf))
    flagged = matched[matched.boundary_change_nfhs6]
    for r in nm[nm.method == "unmatched"].itertuples():
        cands = flagged[flagged.state_key == r.state_key]
        base = dict(transition="5to6", state=r.state, parent_vintage="nfhs5", child_vintage="nfhs6",
                    child_uid=r.district_uid, child_name=r.district_nfhs6, weight=np.nan)
        if len(cands) == 1:
            c = cands.iloc[0]
            rows.append({**base, "parent_uid": c.district_uid, "parent_name": c.district_nfhs45,
                         "relation": "split_candidate_single_flagged_parent", "confidence": "low",
                         "source": "inferred: the only boundary-flagged retained-name district in the state; "
                                   "NOT verified against LGD/Census"})
        else:
            names = "; ".join(cands.district_nfhs45.astype(str)) or "none"
            rows.append({**base, "parent_uid": None, "parent_name": None, "relation": "new_district_parent_unknown",
                         "confidence": "unresolved",
                         "source": f"not in India.csv; flagged candidate parents in state: {names}"})
    for r in nm[nm.method == "not_in_nfhs6"].itertuples():
        rows.append(dict(transition="5to6", state=r.state, parent_vintage="nfhs5", parent_uid=r.district_uid,
                         parent_name=r.district_nfhs45, child_vintage="nfhs6", child_uid=None, child_name=None,
                         relation="absent_in_nfhs6", weight=np.nan, confidence="high",
                         source="district not in the NFHS-6 file"))
    # 4 -> 5: India.csv reports NFHS-4 on NFHS-5 boundaries where IIPS judged it comparable.
    for r in nm[nm.district_nfhs45.notna()].itertuples():
        base = dict(transition="4to5", state=r.state, parent_vintage="census2011", child_vintage="nfhs5",
                    child_uid=r.district_uid, child_name=r.district_nfhs45, weight=np.nan)
        if r.has_nfhs4:
            rows.append({**base, "parent_uid": r.district_uid, "parent_name": r.district_nfhs45,
                         "relation": "unchanged", "confidence": "high",
                         "source": "India.csv has an NFHS-4 value on the NFHS-5 boundary (IIPS comparability)"})
        else:
            code = r.district_code if pd.notna(r.district_code) else None
            note = (f"keeps Census 2011 code {code} (likely the truncated parent)" if code else
                    "no Census 2011 code (district created after 2011)")
            rows.append({**base, "parent_uid": None, "parent_name": None,
                         "relation": "boundary_changed_4to5_parent_unknown", "confidence": "unresolved",
                         "source": f"India.csv NFHS-4 column 'na' ({note}); NFHS-4 Census-2011 values and "
                                   "lineage not in the repo"})
    cw = pd.DataFrame(rows)
    for c in CROSSWALK_COLUMNS:
        if c not in cw.columns:
            cw[c] = None
    return cw[CROSSWALK_COLUMNS[:-2]]


def _name_index(name_map: pd.DataFrame, vintage: str) -> dict:
    """(state_key, normalised district name) -> district_uid for one vintage."""
    idx = {}
    cols = ["district_nfhs45", "district"] if vintage == "nfhs5" else ["district_nfhs6", "district"]
    sub = name_map[name_map[cols[0]].notna()]
    for r in sub.itertuples():
        for c in cols:
            idx.setdefault((r.state_key, normalise_name(getattr(r, c))), r.district_uid)
    return idx


def apply_supplied_lineage(cw: pd.DataFrame, supplied: pd.DataFrame | None, name_map: pd.DataFrame,
                           cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Merge supplied official lineage into the crosswalk.

    Supplied 5to6 rows replace every repo edge pointing at the same NFHS-6 child;
    supplied 4to5 rows replace the repo edge of the same NFHS-5 child. Names
    that cannot be resolved are returned as problems (and go to review).
    """
    if supplied is None:
        return cw, []
    st_alias = cfg["matching"].get("state_aliases", {})
    state_key = lambda s: st_alias.get(normalise_name(s), normalise_name(s))  # noqa: E731
    idx5, idx6 = _name_index(name_map, "nfhs5"), _name_index(name_map, "nfhs6")
    state_name = name_map.groupby("state_key").state.first().to_dict()
    new, problems = [], []
    for r in supplied.itertuples():
        t = r.transition.strip()
        sk = state_key(r.state)
        conf = (getattr(r, "confidence", "") or "high").strip().lower()
        w = pd.to_numeric(getattr(r, "weight", ""), errors="coerce")
        if t == "5to6":
            p = idx5.get((sk, normalise_name(r.parent_district)))
            c = idx6.get((sk, normalise_name(r.child_district)))
            if p is None or c is None:
                problems.append({"transition": t, "state": r.state, "parent_name": r.parent_district,
                                 "child_name": r.child_district,
                                 "review_reason": "supplied lineage: name not found "
                                                  f"({'parent' if p is None else 'child'})"})
                continue
            new.append(dict(transition=t, state=state_name.get(sk, r.state), parent_vintage="nfhs5", parent_uid=p,
                            parent_name=r.parent_district, child_vintage="nfhs6", child_uid=c,
                            child_name=r.child_district, relation=r.relation, weight=w,
                            source=f"supplied: {r.source}", confidence=conf))
        elif t == "4to5":
            c = idx5.get((sk, normalise_name(r.child_district)))
            if c is None:
                problems.append({"transition": t, "state": r.state, "parent_name": r.parent_district,
                                 "child_name": r.child_district,
                                 "review_reason": "supplied lineage: NFHS-5 child name not found"})
                continue
            new.append(dict(transition=t, state=state_name.get(sk, r.state), parent_vintage="census2011",
                            parent_uid=f"{sk}|{normalise_name(r.parent_district)}", parent_name=r.parent_district,
                            child_vintage="nfhs5", child_uid=c, child_name=r.child_district, relation=r.relation,
                            weight=w, source=f"supplied: {r.source}", confidence=conf))
        else:
            problems.append({"transition": t, "state": r.state, "parent_name": r.parent_district,
                             "child_name": r.child_district,
                             "review_reason": f"supplied lineage: unknown transition '{t}'"})
    if not new:
        return cw, problems
    new = pd.DataFrame(new)
    drop = pd.Series(False, index=cw.index)
    for t in ("5to6", "4to5"):
        kids = set(new.loc[new.transition == t, "child_uid"])
        drop |= (cw.transition == t) & cw.child_uid.isin(kids)
    out = pd.concat([cw[~drop], new], ignore_index=True)
    return out.drop_duplicates(["transition", "parent_uid", "child_uid"], keep="last").reset_index(drop=True), problems


# =====================================================================
# Components
# =====================================================================
def connected_components(edges: list[tuple[str, str]]) -> dict[str, int]:
    """Union-find over node ids; returns node -> component number (deterministic order)."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = sorted({find(x) for x in parent})
    num = {r: i for i, r in enumerate(roots)}
    return {x: num[find(x)] for x in parent}


def population_weights(members: list[str], pop: dict[str, tuple[float, bool]]) -> tuple[np.ndarray, bool] | None:
    """Normalised population weights for the members of one vintage, or None if any is missing."""
    vals = [pop.get(m) for m in members]
    if any(v is None or not np.isfinite(v[0]) or v[0] <= 0 for v in vals):
        return None
    p = np.array([v[0] for v in vals], float)
    return p / p.sum(), any(v[1] for v in vals)


def aggregate(values, ses, weights, weight_cv: float = 0.0, min_coverage: float = 0.8):
    """Population-weighted aggregate of child estimates with SE propagation.

    Returns (value, se_sampling, var_weights, coverage). Missing children are
    dropped and the weights renormalised if the remaining weight share is at
    least `min_coverage`; otherwise the value is NaN.
        value = sum w_i y_i
        se_sampling^2 = sum w_i^2 se_i^2                    (independent samples)
        var_weights   = sum (weight_cv * w_i)^2 (y_i - value)^2   (estimated weights; delta method)
    """
    y = np.asarray(values, float)
    se = np.asarray(ses, float)
    w = np.asarray(weights, float)
    if len(y) == 0 or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to 1")
    ok = ~np.isnan(y)
    coverage = float(w[ok].sum())
    if coverage < min_coverage or coverage <= 0:
        return np.nan, np.nan, np.nan, coverage
    wn = w[ok] / coverage
    value = float(np.sum(wn * y[ok]))
    se_ok = np.where(np.isnan(se[ok]), 0.0, se[ok])
    var_s = float(np.sum(wn ** 2 * se_ok ** 2))
    var_w = float(np.sum((weight_cv * wn) ** 2 * (y[ok] - value) ** 2))
    return value, np.sqrt(var_s), var_w, coverage


def within_state_tau2(panel: pd.DataFrame, uids, rnd: int = 5) -> pd.Series:
    """Per-indicator between-district variance within states (method of moments):
    var(y - state mean) - mean(se^2), floored at 0. Round-`rnd` values only (default NFHS-5)."""
    p = panel[(panel["round"] == rnd) & panel.district_uid.isin(set(uids))].dropna(subset=["value"])
    if p.empty:
        return pd.Series(dtype=float)
    p = p.assign(state_=p.district_uid.str.split("|").str[0])
    p = p.assign(dev=p.value - p.groupby(["indicator_id", "state_"]).value.transform("mean"))
    n = p.groupby(["indicator_id", "state_"]).value.transform("size")
    p = p[n > 1]
    v = p.groupby("indicator_id").dev.var(ddof=1)
    noise = p.groupby("indicator_id").se.apply(lambda s: float(np.nanmean(s ** 2)))
    return (v - noise).clip(lower=0.0)


# =====================================================================
# Harmonised panel
# =====================================================================
def _unit_rows(src: pd.DataFrame, uid: str, rounds, method: str, extra_var=None) -> pd.DataFrame:
    rows = src[src["round"].isin(rounds)].copy()
    rows["district_uid"] = uid
    rows["value_method"] = method
    if extra_var is not None:
        rows["extra_var"] = rows.indicator_id.map(extra_var).fillna(0.0)
    return rows


def harmonise_panel(node_panel: pd.DataFrame, name_map: pd.DataFrame, crosswalk: pd.DataFrame, cfg: dict,
                    supplied: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build analysis units and their long panel (see module docstring).

    node_panel: the model panel at NFHS-5 / NFHS-6 district level WITH an `se`
    column (noise.add_se). Returns (unit_panel, units, crosswalk). unit_panel keeps the
    panel columns (district_uid is the unit id) and adds value_method
    (observed / aggregated / inherited), harmonised, se_sampling, extra_var
    and parent_uid (bootstrap cluster). units has one row per unit with
    train_unit / test_unit / stable_unit flags. crosswalk is the input with
    `used` (edge used to build a unit) and `review_reason` filled in.
    """
    hcfg = cfg["harmonise"]
    supplied = supplied or {}
    min_rank = CONFIDENCE_RANK[hcfg["min_confidence"]]
    thr = cfg["matching"]["audit_min_district_agreement"]
    nm = name_map.copy()
    nm["audit_status"] = _audit_status(nm, thr)
    info = nm.set_index("district_uid")
    share = hcfg["inherit_var_share"]
    weight_cv, min_cov = hcfg["weight_cv"], hcfg["min_weight_coverage"]

    panel = node_panel.copy()
    panel["value_method"] = "observed"
    panel["extra_var"] = 0.0
    panel["se_sampling"] = panel["se"]
    cw = crosswalk.copy()
    cw["conf_rank"] = cw.confidence.map(CONFIDENCE_RANK).fillna(0)
    cw["used"] = False
    cw["review_reason"] = None

    # ---------------- NFHS-5 districts usable for training (4 -> 5) ----------------
    bad_audit = set(nm.loc[nm.audit_status == "fail", "district_uid"])
    has5 = set(nm.loc[nm.has_nfhs5.astype(bool) & nm.district_nfhs45.notna(), "district_uid"]) - bad_audit
    has4 = set(nm.loc[nm.has_nfhs4.astype(bool), "district_uid"]) & has5
    unaudited = set(nm.loc[nm.audit_status == "unaudited", "district_uid"])
    train = set(has4) if hcfg["train_include_unaudited"] else has4 - unaudited
    e45 = cw[(cw.transition == "4to5") & (cw.conf_rank >= min_rank)]
    cw.loc[e45.index[e45.relation.eq("unchanged") & e45.child_uid.isin(train)], "used"] = True
    tau2 = within_state_tau2(panel, sorted(has4 & set(nm.loc[nm.core, "district_uid"]) or has4), rnd=5)

    # NFHS-4 inherited from Census-2011 parents (supplied values + supplied 4to5 lineage only)
    n4 = supplied.get("nfhs4_census2011")
    inherited4 = set()
    if n4 is not None and hcfg["train_on_inherited_nfhs4"]:
        st_alias = cfg["matching"].get("state_aliases", {})
        n4 = n4.assign(parent_uid=[f"{st_alias.get(normalise_name(s), normalise_name(s))}|{normalise_name(d)}"
                                   for s, d in zip(n4.state, n4.district)])
        vals4 = n4.pivot_table(index="parent_uid", columns="indicator_id", values="value", aggfunc="first")
        sup = e45[e45.source.astype(str).str.startswith("supplied") & ~e45.relation.eq("unchanged")]
        for child, grp in sup.groupby("child_uid"):
            if child in has4 or child not in has5 or not grp.parent_uid.isin(vals4.index).all():
                continue
            w = grp.weight.astype(float).to_numpy() if len(grp) > 1 else np.array([1.0])
            if np.isnan(w).any() or not np.isclose(w.sum(), 1.0):
                continue
            new = panel[(panel.district_uid == child) & (panel["round"] == 5)].copy()
            new["round"] = 4
            pv = vals4.reindex(index=grp.parent_uid.tolist(), columns=new.indicator_id.unique())
            agg = (pv.mul(w, axis=0)).sum(min_count=len(w))
            new["value"] = new.indicator_id.map(agg)
            new["value_method"] = "inherited"
            new["extra_var"] = new.indicator_id.map(tau2 * share["child"]).fillna(0.0)
            new["se_sampling"] = noise.se_percent(new.value.to_numpy(), new.n_assumed.to_numpy(),
                                                    cfg["noise"]["deff"])
            new["se"] = np.sqrt(new.se_sampling ** 2 + new.extra_var)
            new["boundary_change_flag"] = True
            panel = pd.concat([panel[~((panel.district_uid == child) & (panel["round"] == 4))], new],
                              ignore_index=True)
            inherited4.add(child)
            cw.loc[grp.index, "used"] = True
        train |= inherited4

    # ---------------- 5 -> 6 components ----------------
    e56 = cw[(cw.transition == "5to6") & (cw.conf_rank >= min_rank) & cw.parent_uid.notna() & cw.child_uid.notna()]
    e56 = e56[e56.parent_uid.isin(has5)]
    comp = connected_components([("5:" + p, "6:" + c) for p, c in zip(e56.parent_uid, e56.child_uid)])
    pop = {}
    if supplied.get("population") is not None:
        st_alias = cfg["matching"].get("state_aliases", {})
        idx = {"nfhs5": _name_index(name_map, "nfhs5"), "nfhs6": _name_index(name_map, "nfhs6")}
        for r in supplied["population"].itertuples():
            v = r.vintage.strip()
            uid = idx.get(v, {}).get((st_alias.get(normalise_name(r.state), normalise_name(r.state)),
                                      normalise_name(r.district)))
            if uid is not None:
                pop[("5:" if v == "nfhs5" else "6:") + uid] = (float(r.population), bool(r.estimated))

    units, extra = [], []
    by_comp: dict[int, list[str]] = {}
    for node, k in comp.items():
        by_comp.setdefault(k, []).append(node)
    stable_core = set(nm.loc[nm.core, "district_uid"])
    for k, nodes in sorted(by_comp.items()):
        p5 = sorted(n[2:] for n in nodes if n.startswith("5:"))
        c6 = sorted(n[2:] for n in nodes if n.startswith("6:"))
        edges = e56[e56.parent_uid.isin(p5) | e56.child_uid.isin(c6)]
        conf = min(edges.confidence, key=lambda c: CONFIDENCE_RANK[c])
        state = info.loc[p5[0], "state"]
        base = dict(state=state, members_nfhs5="; ".join(p5), members_nfhs6="; ".join(c6), confidence=conf,
                    component=k)
        if len(p5) == 1 and len(c6) == 1:
            u, v = p5[0], c6[0]
            if edges.relation.isin(["unchanged", "renamed"]).all():
                if u != v:   # renamed only (supplied lineage): same territory, rows copied under the NFHS-6 uid
                    extra.append(_unit_rows(panel[panel.district_uid == u], v, [4, 5], "observed"))
                    if u in train:
                        train.discard(u)
                        train.add(v)
                units.append({**base, "unit_uid": v, "unit_type": "identity" if u == v else "identity_renamed",
                              "parent_uid": u, "test_unit": True})
            else:
                rows = panel[(panel.district_uid == v) & (panel["round"] == 6)]
                ev = (tau2 * share["remnant" if u == v else "child"])
                idx_ = rows.index
                panel.loc[idx_, "extra_var"] = rows.indicator_id.map(ev).fillna(0.0).to_numpy()
                panel.loc[idx_, "value_method"] = "observed_changed_boundary"
                if u != v:   # renamed-and-changed: inputs copied from the parent under the child's uid
                    extra.append(_unit_rows(panel[panel.district_uid == u], v, [4, 5], "inherited"))
                units.append({**base, "unit_uid": v, "unit_type": "inherited_remnant" if u == v else "inherited_child",
                              "parent_uid": u, "test_unit": True})
            cw.loc[edges.index, "used"] = True
            continue
        w5 = population_weights(p5, {n[2:]: pop[n] for n in pop if n.startswith("5:")}) if len(p5) > 1 else (np.ones(1), False)
        w6 = population_weights(c6, {n[2:]: pop[n] for n in pop if n.startswith("6:")}) if len(c6) > 1 else (np.ones(1), False)
        if w5 is not None and w6 is not None:
            uid = f"{p5[0].split('|')[0]}|h:" + "+".join(x.split("|")[1] for x in p5)
            for rounds, members, (w, est) in (([4, 5], p5, w5), ([6], c6, w6)):
                sub = panel[panel.district_uid.isin(members) & panel["round"].isin(rounds)]
                for (ind, rnd), g in sub.groupby(["indicator_id", "round"]):
                    g = g.set_index("district_uid").reindex(members)
                    val, se_s, var_w, _ = aggregate(g.value, g.se, w, weight_cv if est else 0.0, min_cov)
                    first = g.dropna(subset=["state"]).iloc[0]
                    row = first.to_dict()
                    row.update(district_uid=uid, district=f"{'+'.join(members)}", value=val, se_sampling=se_s,
                               extra_var=var_w if np.isfinite(var_w) else 0.0,
                               se=np.sqrt(se_s ** 2 + (var_w if np.isfinite(var_w) else 0.0)),
                               value_method="aggregated", round=rnd, indicator_id=ind,
                               small_sample_flag=bool(g.small_sample_flag.fillna(False).any()),
                               boundary_change_flag=True)
                    extra.append(pd.DataFrame([row]))
            units.append({**base, "unit_uid": uid, "unit_type": "aggregated", "parent_uid": uid, "test_unit": True})
            cw.loc[edges.index, "used"] = True
        elif len(p5) == 1 and hcfg["split_without_weights"] == "inherit":
            u = p5[0]
            for v in c6:
                rows = panel[(panel.district_uid == v) & (panel["round"] == 6)]
                ev = tau2 * share["remnant" if v == u else "child"]
                panel.loc[rows.index, "extra_var"] = rows.indicator_id.map(ev).fillna(0.0).to_numpy()
                panel.loc[rows.index, "value_method"] = "observed_changed_boundary"
                if v != u:
                    extra.append(_unit_rows(panel[panel.district_uid == u], v, [4, 5], "inherited"))
                units.append({**base, "unit_uid": v, "unit_type": "inherited_remnant" if v == u else "inherited_child",
                              "parent_uid": u, "test_unit": True})
            cw.loc[edges.index, "used"] = True
        else:
            units.append({**base, "unit_uid": None, "unit_type": "unresolved_no_weights", "parent_uid": None,
                          "test_unit": False})
            cw.loc[edges.index, "review_reason"] = ("component needs population weights "
                                                    "(data/lineage/population.csv)")

    if extra:
        panel = pd.concat([panel] + extra, ignore_index=True)
    panel["se"] = np.sqrt(panel.se_sampling.astype(float) ** 2 + panel.extra_var.astype(float))
    panel["harmonised"] = ~panel.value_method.eq("observed")

    u = pd.DataFrame(units)
    known = set(u.unit_uid.dropna())
    train_only = sorted(train - known)
    if train_only:
        u = pd.concat([u, pd.DataFrame({"unit_uid": train_only, "state": [info.loc[x, "state"] for x in train_only],
                                        "members_nfhs5": train_only, "members_nfhs6": "", "confidence": "high",
                                        "unit_type": "identity_train_only", "parent_uid": train_only,
                                        "test_unit": False})], ignore_index=True)
    u = u.dropna(subset=["unit_uid"]).drop_duplicates("unit_uid")
    u["train_unit"] = u.unit_uid.isin(train)
    u["nfhs4_inherited"] = u.unit_uid.isin(inherited4)
    u["stable_unit"] = u.unit_uid.isin(stable_core) & u.unit_type.eq("identity")
    u["audit_status"] = u.unit_uid.map(nm.set_index("district_uid").audit_status).fillna("n/a")
    # test units need NFHS-5 inputs; train units need both rounds (checked again in the pipeline)
    u = u.sort_values(["state", "unit_uid"]).reset_index(drop=True)
    parent = u.set_index("unit_uid").parent_uid
    panel = panel[panel.district_uid.isin(set(u.unit_uid))].copy()
    panel["parent_uid"] = panel.district_uid.map(parent)
    panel = panel.sort_values(["district_uid", "indicator_id", "round"]).reset_index(drop=True)
    return panel, u, cw.drop(columns="conf_rank")


# =====================================================================
# Review and checks
# =====================================================================
def review_table(crosswalk: pd.DataFrame, units: pd.DataFrame, problems: list[dict], cfg: dict) -> pd.DataFrame:
    """Every low-confidence / unresolved / unused-but-needed lineage case, for a human to check."""
    min_rank = CONFIDENCE_RANK[cfg["harmonise"]["min_confidence"]]
    cw = crosswalk.copy()
    rank = cw.confidence.map(CONFIDENCE_RANK).fillna(0)
    reasons = cw.get("review_reason", pd.Series(None, index=cw.index)).astype(object)
    action = pd.Series("", index=cw.index, dtype=object)
    low = rank < min_rank
    reasons = reasons.where(reasons.notna(), np.where(low, "confidence below harmonise.min_confidence", None))
    m = cw.relation.eq("retained_name_boundary_changed")
    reasons = reasons.where(~m | reasons.notna(), "boundary changed but carved-out/added districts unknown; "
                                                   "scored by inheritance with extra variance")
    action[cw.relation.eq("new_district_parent_unknown")] = ("supply parent(s) in data/lineage/lineage_supplied.csv "
                                                             "(LGD) and populations in data/lineage/population.csv")
    action[cw.relation.eq("split_candidate_single_flagged_parent")] = ("confirm or correct this parent in "
                                                                       "data/lineage/lineage_supplied.csv")
    action[m] = "list the NFHS-6 districts carved out of / merged into it (lineage_supplied.csv) + populations"
    action[cw.relation.eq("boundary_changed_4to5_parent_unknown")] = (
        "supply Census-2011 parent(s) (lineage_supplied.csv, transition 4to5) and NFHS-4 Census-2011 values "
        "(nfhs4_census2011_values.csv) to use this district in training")
    out = cw.assign(review_reason=reasons, suggested_action=action)
    out = out[out.review_reason.notna()]
    if problems:
        out = pd.concat([out, pd.DataFrame(problems).assign(confidence="unresolved",
                                                            suggested_action="fix the name in the supplied file")],
                        ignore_index=True)
    cols = ["transition", "state", "parent_name", "child_name", "parent_uid", "child_uid", "relation",
            "confidence", "used", "source", "review_reason", "suggested_action"]
    out = out.reindex(columns=cols)
    order = {"unresolved": 0, "low": 1, "medium": 2, "high": 3}
    return (out.assign(_o=out.confidence.map(order)).sort_values(["_o", "transition", "state", "child_name"])
            .drop(columns="_o").reset_index(drop=True))


def check_state_reconstruction(unit_values: pd.DataFrame, weights: pd.DataFrame, state_values: pd.DataFrame,
                               tol: float) -> pd.DataFrame:
    """Rebuild published state values as population-weighted means of district values.

    unit_values: district_uid, indicator_id, value; weights: district_uid, population;
    state_values: state, indicator_id, value (published). Returns one row per
    state x indicator with the rebuilt value, the difference and `within_tol`.
    """
    d = unit_values.merge(weights, on="district_uid")
    d["state"] = d.district_uid.str.split("|").str[0]
    d = d.dropna(subset=["value", "population"])
    rebuilt = (d.assign(wv=d.value * d.population).groupby(["state", "indicator_id"])
               .agg(wv=("wv", "sum"), pop=("population", "sum")).reset_index())
    rebuilt["rebuilt"] = rebuilt.wv / rebuilt["pop"]
    out = state_values.merge(rebuilt[["state", "indicator_id", "rebuilt"]], on=["state", "indicator_id"])
    out["diff"] = out.rebuilt - out.value
    out["within_tol"] = out["diff"].abs() <= tol
    return out


def summarise(units: pd.DataFrame, name_map: pd.DataFrame, cfg: dict) -> dict:
    """Counts reported in outputs/harmonisation_report.md and SUMMARY."""
    thr = cfg["matching"]["audit_min_district_agreement"]
    core = int(name_map.core.sum())
    t = units[units.test_unit]
    return {
        "stable_core_districts": core,
        "train_units": int(units.train_unit.sum()),
        "train_units_added": int(units.train_unit.sum()) - core,
        "train_units_nfhs4_inherited": int(units.nfhs4_inherited.sum()),
        "test_units": int(len(t)),
        "test_units_added": int(len(t)) - int(units.stable_unit.sum()),
        "test_units_by_type": t.unit_type.value_counts().to_dict(),
        "test_identity_without_nfhs4": int((t.unit_type.eq("identity") & ~t.train_unit).sum()),
        "stable_units": int(units.stable_unit.sum()),
        "audit_failed": sorted(f"{r.state} | {r.district} ({r.audit_within_0_5:.0%})" for r in
                               name_map[name_map.audit_within_0_5 < thr].itertuples()),
    }


def write_report(path, summary: dict, crosswalk: pd.DataFrame, review: pd.DataFrame, units: pd.DataFrame,
                 supplied: dict, cfg: dict) -> None:
    """outputs/harmonisation_report.md: what was recovered, from which evidence, what is missing."""
    from .config import write_text
    from .report import md_table
    L = ["# Boundary harmonisation report", "",
         "Generated by `scripts/run_pipeline.py`. Lineage comes only from evidence in the repository "
         "(name matches, India.csv NFHS-4 availability, NFHS-6 boundary flags) plus any supplied files; "
         "nothing is inferred from NFHS-6 values. See `nfhs_forecast/harmonise.py` and DECISIONS D27-D31.", "",
         "## Supplied files", ""]
    for k, label in (("lineage", "lineage_supplied.csv"), ("population", "population.csv"),
                     ("nfhs4_census2011", "nfhs4_census2011_values.csv")):
        df = supplied.get(k)
        L.append(f"- `data/lineage/{label}`: " + ("not supplied" if df is None else f"{len(df)} rows"))
    L += ["", "## Units", "",
          "| | count |", "|---|---|",
          f"| stable-boundary districts (base pipeline core) | {summary['stable_core_districts']} |",
          f"| training units (NFHS-4 -> NFHS-5) | {summary['train_units']} (+{summary['train_units_added']}) |",
          f"| of which NFHS-4 inherited from a Census-2011 parent | {summary['train_units_nfhs4_inherited']} |",
          f"| test units (NFHS-5 -> NFHS-6, harmonised set) | {summary['test_units']} "
          f"(+{summary['test_units_added']} over the stable subset) |"]
    for t, n in sorted(summary["test_units_by_type"].items()):
        L.append(f"| test units: {t} | {n} |")
    L += [f"| test units: identity districts without NFHS-4 (changed 4->5, unchanged 5->6) | "
          f"{summary['test_identity_without_nfhs4']} |", "",
          "Unit types: `identity` = same district in NFHS-5 and NFHS-6; `inherited_remnant` = NFHS-6 district "
          "kept its name but its boundary changed, scored against the forecast for the whole NFHS-5 district "
          "with extra variance; `inherited_child` = new NFHS-6 district scored against its parent's forecast; "
          "`aggregated` = component re-aggregated with population weights; `identity_train_only` = used for "
          "training only (not in NFHS-6, or a member of an aggregated component).", "",
          "## Crosswalk edges", "",
          md_table(crosswalk.groupby(["transition", "relation", "confidence", "used"]).size()
                   .rename("edges").reset_index()), "",
          f"Full crosswalk: `outputs/crosswalk.csv`. Cases for a human to check: `outputs/crosswalk_review.csv` "
          f"({len(review)} rows).", "",
          "## Excluded by the NFHS-5 audit", "",
          "Districts whose NFHS-5 values disagree between India.csv and the NFHS-6 file are excluded from "
          "every set (rule in DECISIONS D12). This now also removes districts that were previously excluded "
          "earlier for lack of NFHS-4 values.", ""]
    L += [f"- {x}" for x in summary["audit_failed"]] + [""]
    L += ["## What is needed to recover more districts", "",
          "1. **Parents of new NFHS-6 districts and the territory changes of flagged districts** "
          "(`data/lineage/lineage_supplied.csv`, transition `5to6`), from the Local Government Directory "
          "(lgdirectory.gov.in, 'Download Only Modifications', districts, 2019-2024) or state gazette notifications.",
          "2. **Populations** of every NFHS-5 and NFHS-6 district in a changed component "
          "(`data/lineage/population.csv`), e.g. Census 2011 populations re-aggregated to current boundaries "
          "via sub-districts. Without them, splits are scored by inheritance (flagged, extra variance).",
          "3. **NFHS-4 district values on Census-2011 boundaries** (`data/lineage/nfhs4_census2011_values.csv`, "
          "from the IIPS NFHS-4 district fact sheets) plus the Census-2011 parent of each NFHS-5 district that "
          "has no NFHS-4 value (transition `4to5`). This is the only way to add the "
          f"{int((crosswalk.relation == 'boundary_changed_4to5_parent_unknown').sum())} districts that changed "
          "between NFHS-4 and NFHS-5 to the training data.", "",
          "Formats: `data/lineage/README.md`."]
    write_text(path, "\n".join(L) + "\n")
