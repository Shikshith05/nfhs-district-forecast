"""Indicator and district matching across NFHS-4/5 (India.csv) and NFHS-6.

Indicators are matched by normalised TEXT only, never by indicator_no
(numbering differs between district and state sheets; DECISIONS D07).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .clean import indicator_key, normalise_indicator_text, normalise_name, squash

SOURCE_45 = "India.csv (NFHS-5 district fact sheets incl. NFHS-4 column; SaiSiddhardhaKalla/NFHS)"
SOURCE_6 = "all_india_nfhs6_district.csv (NFHS-6 provisional; Patra 2026)"
_METHOD_RANK = {"exact": 0, "prefix": 1, "fuzzy": 2}


# =====================================================================
# Indicators
# =====================================================================
_POPULATION_WORDS = {"women", "men", "female", "male", "mothers", "children", "boys", "girls",
                     "households", "population", "public", "private", "home", "urban", "rural"}


def _population_words(norm: str) -> frozenset:
    """Words that change who an indicator is about; fuzzy matches must agree on them."""
    return frozenset(w for w in norm.split() if w in _POPULATION_WORDS)


def _looks_truncated(raw: str) -> bool:
    """Truncated texts in the NFHS-6 file lose their '(%)' / closing bracket."""
    raw = raw.strip()
    return not (raw.endswith(")") or raw.endswith("%"))


def _variants(df: pd.DataFrame, cat_col: str, text_col: str, pattern: str) -> pd.DataFrame:
    v = df.groupby([cat_col, text_col]).size().rename("n_rows").reset_index()
    v.columns = ["category", "raw_text", "n_rows"]
    keys = v.apply(lambda r: indicator_key(r.raw_text, r.category, pattern), axis=1)
    v["sex"] = [k[0] for k in keys]
    v["norm"] = [k[1] for k in keys]
    v["squash"] = [k[2] for k in keys]
    v["truncated"] = v.raw_text.map(_looks_truncated)
    return v


def collapse_within_source(v: pd.DataFrame, min_prefix: int) -> pd.DataFrame:
    """Map every raw variant of one source to a canonical key within that source.

    1. identical normalised keys merge ('tetanus4' == 'tetanus 4');
    2. a truncated variant maps to the unique longer key it is a prefix of;
    3. a minority variant maps to a majority key with fuzz.ratio >= 97
       (e.g. 'kg/m 2' vs 'kg/m2').
    Adds canonical_squash, within_method, within_score.
    """
    v = v.copy()
    keys = v.groupby(["sex", "squash"]).agg(n_rows=("n_rows", "sum"), norm=("norm", "first"),
                                           truncated=("truncated", "all")).reset_index()
    canon: dict[tuple[str, str], tuple[str, str, float]] = {}
    for sex, grp in keys.groupby("sex"):
        full = grp[~grp.truncated]
        full_sq = full.squash.tolist()
        counts = dict(zip(grp.squash, grp.n_rows))
        pop = {q: _population_words(n) for q, n in zip(grp.squash, grp.norm)}
        for row in grp.itertuples():
            target, method, score = row.squash, "exact", 100.0
            if row.truncated and len(row.squash) >= min_prefix:
                longer = [s for s in full_sq if s != row.squash and s.startswith(row.squash)]
                if len(longer) == 1:
                    target, method, score = longer[0], "prefix", 100.0
                elif len(longer) > 1:
                    method = "ambiguous_prefix"
            if method == "exact":
                others = [s for s in full_sq if s != row.squash and counts[s] > 2 * counts[row.squash]
                          and pop[s] == pop[row.squash]]
                if others:
                    best = process.extractOne(row.squash, others, scorer=fuzz.ratio)
                    if best and best[1] >= 97:
                        target, method, score = best[0], "fuzzy", float(best[1])
            canon[(sex, row.squash)] = (target, method, score)
    v["canonical_squash"] = [canon[(s, q)][0] for s, q in zip(v.sex, v.squash)]
    v["within_method"] = [canon[(s, q)][1] for s, q in zip(v.sex, v.squash)]
    v["within_score"] = [canon[(s, q)][2] for s, q in zip(v.sex, v.squash)]
    return v


def _canonical_table(v: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical key with its most frequent full raw text."""
    v = v.copy()
    v["is_self"] = v.squash == v.canonical_squash
    rows = []
    for (sex, cs), grp in v.groupby(["sex", "canonical_squash"]):
        self_rows = grp[grp.is_self & ~grp.truncated]
        pick = (self_rows if len(self_rows) else grp).sort_values("n_rows", ascending=False).iloc[0]
        rows.append({"sex": sex, "canonical_squash": cs, "text": pick.raw_text,
                     "category": pick.category, "n_rows": int(grp.n_rows.sum()),
                     "n_variants": len(grp), "norm": pick.norm})
    return pd.DataFrame(rows)


def match_indicators_across(c6: pd.DataFrame, c45: pd.DataFrame, min_score: float,
                            min_prefix: int) -> pd.DataFrame:
    """One-to-one match of NFHS-6 canonical keys to India.csv canonical keys (same sex)."""
    cands = []
    for r6 in c6.itertuples():
        pool = c45[c45.sex == r6.sex]
        for r45 in pool.itertuples():
            a, b = r6.canonical_squash, r45.canonical_squash
            if a == b:
                cands.append((r6.Index, r45.Index, "exact", 100.0))
                continue
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            if len(short) >= min_prefix and long_.startswith(short):
                cands.append((r6.Index, r45.Index, "prefix", 100.0))
                continue
            score = fuzz.ratio(a, b)
            if score >= min_score and _population_words(r6.norm) == _population_words(r45.norm):
                cands.append((r6.Index, r45.Index, "fuzzy", float(score)))
    cands.sort(key=lambda x: (_METHOD_RANK[x[2]], -x[3]))
    used6, used45, out = set(), set(), []
    for i6, i45, method, score in cands:
        if i6 in used6 or i45 in used45:
            continue
        used6.add(i6)
        used45.add(i45)
        out.append({"i6": i6, "i45": i45, "method": method, "score": score})
    pairs = pd.DataFrame(out, columns=["i6", "i45", "method", "score"])
    m = pd.DataFrame({
        "sex": c6.sex, "squash6": c6.canonical_squash, "text6": c6.text,
        "category6": c6.category, "n_rows6": c6.n_rows,
    })
    m = m.merge(pairs, left_index=True, right_on="i6", how="left")
    m = m.merge(c45[["canonical_squash", "text", "category", "n_rows"]]
                .rename(columns={"canonical_squash": "squash45", "text": "text45",
                                 "category": "category45", "n_rows": "n_rows45"}),
                left_on="i45", right_index=True, how="left")
    only45 = c45.drop(index=list(used45))
    m45 = pd.DataFrame({"sex": only45.sex, "squash45": only45.canonical_squash,
                        "text45": only45.text, "category45": only45.category,
                        "n_rows45": only45.n_rows})
    out = pd.concat([m.drop(columns=["i6", "i45"]), m45], ignore_index=True)
    out["method"] = out.method.fillna("unmatched")
    return out


_STOP = {"the", "a", "an", "of", "in", "who", "with", "for", "to", "and", "or", "by", "that",
         "are", "is", "were", "was", "have", "has", "their", "any", "from", "on", "years",
         "age", "months", "last", "survey", "preceding", "household", "households", "lt",
         "gt", "ge", "le", "at", "time", "they"}


def _slug(text: str, n_words: int = 6) -> str:
    words = [w for w in normalise_indicator_text(text).split() if w not in _STOP]
    return "_".join(words[:n_words]) or "indicator"


def _find_by_text(cross: pd.DataFrame, text: str) -> int | None:
    """Row index of the cross table whose NFHS-6 text matches `text` (exact, else unique substring)."""
    target = squash(normalise_indicator_text(text))
    s6 = cross.squash6.fillna("")
    exact = cross.index[s6 == target]
    if len(exact) == 1:
        return int(exact[0])
    sub = cross.index[s6.str.contains(target, regex=False) & (s6 != "")]
    if len(sub) == 1:
        return int(sub[0])
    return None


def assign_indicator_ids(cross: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Give every cross-round indicator an id: core ids from config, else a text slug."""
    cross = cross.copy()
    cross["indicator_id"] = None
    cross["core"] = False
    cross["domain"] = None
    cross["se_group"] = None
    missing = []
    for item in cfg["indicators"]["core"]:
        idx = _find_by_text(cross, item["text"])
        if idx is None:
            missing.append(item["id"])
            continue
        cross.loc[idx, ["indicator_id", "domain", "se_group"]] = [item["id"], item["domain"], item["se_group"]]
        cross.loc[idx, "core"] = True
    cross.attrs["core_not_found"] = missing
    taken = set(cross.indicator_id.dropna())
    for idx in cross.index[cross.indicator_id.isna()]:
        r = cross.loc[idx]
        text = r.text6 if isinstance(r.text6, str) else r.text45
        base = _slug(text)
        if r.sex:
            base = f"{r.sex}_{base}"
        if r.method == "unmatched":
            base = ("n6only_" if isinstance(r.text6, str) else "n45only_") + base
        sid, k = base, 2
        while sid in taken:
            sid, k = f"{base}_{k}", k + 1
        taken.add(sid)
        cross.loc[idx, "indicator_id"] = sid
    cross["definition_exclusion"] = None
    for item in cfg["indicators"].get("definition_exclusions", []):
        target = squash(normalise_indicator_text(item["text"]))
        hit = cross.squash6.fillna("").str.contains(target, regex=False) & cross.squash6.notna()
        cross.loc[hit, "definition_exclusion"] = item["reason"]
    return cross


def build_indicator_matching(l45: pd.DataFrame, d6: pd.DataFrame, cfg: dict
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (variants, cross).

    variants: every raw text variant in each source with its canonical key and
              the cross-round indicator_id (written to indicator_variants.csv).
    cross:    one row per cross-round indicator (basis of the codebook).
    """
    mcfg = cfg["matching"]
    pat, min_prefix = mcfg["sex_split_pattern"], mcfg["indicator_min_prefix_len"]
    v45 = collapse_within_source(_variants(l45, "Category", "Indicator", pat), min_prefix)
    v6 = collapse_within_source(_variants(d6, "category", "indicator", pat), min_prefix)
    c45, c6 = _canonical_table(v45), _canonical_table(v6)
    cross = match_indicators_across(c6, c45, mcfg["indicator_min_score"], min_prefix)
    cross = assign_indicator_ids(cross, cfg)

    id6 = cross.dropna(subset=["squash6"]).set_index(["sex", "squash6"]).indicator_id
    id45 = cross.dropna(subset=["squash45"]).set_index(["sex", "squash45"]).indicator_id
    v6["source"], v45["source"] = "nfhs6", "nfhs45"
    v6["indicator_id"] = [id6.get((s, c)) for s, c in zip(v6.sex, v6.canonical_squash)]
    v45["indicator_id"] = [id45.get((s, c)) for s, c in zip(v45.sex, v45.canonical_squash)]
    variants = pd.concat([v45, v6], ignore_index=True)
    info = cross.set_index("indicator_id")[["method", "score", "text6", "text45"]]
    variants = variants.join(info.rename(columns={"method": "cross_method", "score": "cross_score",
                                                  "text6": "canonical_text_nfhs6",
                                                  "text45": "canonical_text_nfhs45"}),
                             on="indicator_id")
    cols = ["source", "category", "raw_text", "n_rows", "sex", "norm", "truncated",
            "within_method", "within_score", "indicator_id", "cross_method", "cross_score",
            "canonical_text_nfhs45", "canonical_text_nfhs6"]
    return variants[cols].sort_values(["indicator_id", "source", "n_rows"],
                                      ascending=[True, True, False]), cross


# =====================================================================
# Districts
# =====================================================================
def _state_key(name: str, aliases: dict) -> str:
    k = normalise_name(name)
    return aliases.get(k, k)


def match_districts(l45: pd.DataFrame, d6: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Match NFHS-6 districts to India.csv districts within state; one row per district."""
    mcfg = cfg["matching"]
    st_alias = mcfg.get("state_aliases", {})
    d_alias = mcfg.get("district_aliases", {})

    avail = l45.assign(ok=l45.missing_reason.ne("district_round_not_available"))
    has = avail.groupby(["State", "District", "round"]).ok.any().unstack("round")
    a = has.reset_index().rename(columns={4: "has_nfhs4", 5: "has_nfhs5"})
    a["state_key"] = a.State.map(lambda s: _state_key(s, st_alias))
    a["dkey45"] = a.District.map(normalise_name)
    b = (d6.groupby(["state", "district"]).boundary_change_flag.any()
         .reset_index())
    b["state_key"] = b.state.map(lambda s: _state_key(s, st_alias))
    b["dkey6"] = b.district.map(normalise_name)

    rows, claimed = [], set()
    pending = []
    # pass 1: exact / no-space / alias
    for r in b.itertuples():
        pool = a[a.state_key == r.state_key]
        names = dict(zip(pool.dkey45, pool.index))
        hit, method = None, None
        if r.dkey6 in names:
            hit, method = names[r.dkey6], "exact"
        else:
            sq = {squash(k): i for k, i in names.items()}
            if squash(r.dkey6) in sq:
                hit, method = sq[squash(r.dkey6)], "exact_nospace"
            else:
                target = d_alias.get(r.state_key, {}).get(r.dkey6)
                if target is not None and target in names:
                    hit, method = names[target], "alias"
        if hit is not None and hit not in claimed:
            claimed.add(hit)
            score = 100.0 if method != "alias" else float(fuzz.WRatio(r.dkey6, a.loc[hit, "dkey45"]))
            rows.append((r.Index, hit, method, score))
        else:
            pending.append(r)
    # pass 2: fuzzy among unclaimed
    for r in pending:
        pool = a[(a.state_key == r.state_key) & ~a.index.isin(claimed)]
        method, hit, score = "unmatched", None, np.nan
        if len(pool):
            res = process.extract(r.dkey6, pool.dkey45.tolist(), scorer=fuzz.WRatio, limit=2)
            best = res[0]
            second = res[1][1] if len(res) > 1 else 0
            score = float(best[1])
            if best[1] >= mcfg["district_min_score"] and best[1] - second >= 3:
                hit = pool.index[best[2]]
                method = "fuzzy"
                claimed.add(hit)
        rows.append((r.Index, hit, method, score))

    out = []
    for i6, i45, method, score in rows:
        rb = b.loc[i6]
        ra = a.loc[i45] if i45 is not None else None
        out.append({
            "state_key": rb.state_key, "state_nfhs6": rb.state, "district_nfhs6": rb.district,
            "state_nfhs45": ra.State if ra is not None else None,
            "district_nfhs45": ra.District if ra is not None else None,
            "method": method, "score": score,
            "boundary_change_nfhs6": bool(rb.boundary_change_flag),
            "has_nfhs4": bool(ra.has_nfhs4) if ra is not None else False,
            "has_nfhs5": bool(ra.has_nfhs5) if ra is not None else False,
            "district_uid": f"{rb.state_key}|{rb.dkey6}",
        })
    for i45 in a.index.difference(list(claimed)):
        ra = a.loc[i45]
        out.append({
            "state_key": ra.state_key, "state_nfhs6": None, "district_nfhs6": None,
            "state_nfhs45": ra.State, "district_nfhs45": ra.District,
            "method": "not_in_nfhs6", "score": np.nan, "boundary_change_nfhs6": False,
            "has_nfhs4": bool(ra.has_nfhs4), "has_nfhs5": bool(ra.has_nfhs5),
            "district_uid": f"{ra.state_key}|{ra.dkey45}",
        })
    nm = pd.DataFrame(out)
    nm["state"] = nm.state_nfhs6.fillna(nm.state_nfhs45)
    nm["district"] = nm.district_nfhs6.fillna(nm.district_nfhs45)
    return nm


def finalise_district_status(name_map: pd.DataFrame, audit_by_district: pd.DataFrame,
                             cfg: dict) -> pd.DataFrame:
    """Add audit agreement and the final core / excluded status per district."""
    nm = name_map.merge(audit_by_district, on="district_uid", how="left")
    thr = cfg["matching"]["audit_min_district_agreement"]
    matched = nm.method.isin(["exact", "exact_nospace", "alias", "fuzzy"])
    reasons = []
    for r, m in zip(nm.itertuples(), matched):
        if r.method == "not_in_nfhs6":
            reasons.append("not in NFHS-6 file")
        elif not m:
            reasons.append("no unambiguous match in India.csv (new/renamed district)")
        elif r.boundary_change_nfhs6:
            reasons.append("NFHS-6 boundary_change_single_round == True")
        elif not r.has_nfhs4:
            reasons.append("no NFHS-4 values (boundary changed NFHS-4 -> NFHS-5)")
        elif not r.has_nfhs5:
            reasons.append("no NFHS-5 values")
        elif pd.notna(r.audit_within_0_5) and r.audit_within_0_5 < thr:
            reasons.append(f"NFHS-5 audit agreement {r.audit_within_0_5:.0%} < {thr:.0%} "
                           "(values swapped or misaligned between sources)")
        else:
            reasons.append("")
    nm["exclusion_reason"] = reasons
    nm["core"] = nm.exclusion_reason.eq("")
    return nm


def _id_lookup(variants: pd.DataFrame, source: str) -> dict:
    """{(category, raw_text): indicator_id} for one source."""
    v = variants[variants.source == source]
    return dict(zip(zip(v.category, v.raw_text), v.indicator_id))


# =====================================================================
# Panel
# =====================================================================
def build_panel_long(l45: pd.DataFrame, d6: pd.DataFrame, variants: pd.DataFrame,
                     name_map: pd.DataFrame) -> pd.DataFrame:
    """Tidy long panel: one row per district x indicator x round (all rounds, all districts)."""
    v45 = _id_lookup(variants, "nfhs45")
    v6 = _id_lookup(variants, "nfhs6")

    uid45 = name_map.dropna(subset=["district_nfhs45"]).set_index(["state_nfhs45", "district_nfhs45"])
    p45 = l45.copy()
    p45["indicator_id"] = [v45.get((c, t)) for c, t in zip(p45.Category, p45.Indicator)]
    p45 = p45.join(uid45[["district_uid", "state", "district"]], on=["State", "District"])
    no4 = p45[(p45["round"] == 4)].groupby("district_uid").missing_reason.apply(
        lambda s: (s == "district_round_not_available").all())
    p45["boundary_change_flag"] = (p45["round"] == 5) & p45.district_uid.map(no4).fillna(False).astype(bool)
    p45 = pd.DataFrame({
        "state": p45.state, "district": p45.district, "district_uid": p45.district_uid,
        "district_code": p45.district_code, "round": p45["round"],
        "indicator_id": p45.indicator_id, "indicator_text": p45.Indicator,
        "value": p45.value, "small_sample_flag": pd.array([pd.NA] * len(p45), dtype="boolean"),
        "boundary_change_flag": p45.boundary_change_flag,
        "missing_reason": p45.missing_reason, "source": SOURCE_45,
    })

    uid6 = name_map.dropna(subset=["district_nfhs6"]).set_index(["state_nfhs6", "district_nfhs6"])
    code = p45.dropna(subset=["district_code"]).groupby("district_uid").district_code.first()
    p6 = d6.copy()
    p6["indicator_id"] = [v6.get((c, t)) for c, t in zip(p6.category, p6.indicator)]
    p6 = p6.join(uid6[["district_uid", "state", "district"]], on=["state", "district"],
                 rsuffix="_canon")
    p6 = pd.DataFrame({
        "state": p6.state_canon, "district": p6.district_canon, "district_uid": p6.district_uid,
        "district_code": p6.district_uid.map(code), "round": 6,
        "indicator_id": p6.indicator_id, "indicator_text": p6.indicator,
        "value": p6.value6, "small_sample_flag": p6.small_sample_flag.astype("boolean"),
        "boundary_change_flag": p6.boundary_change_flag,
        "missing_reason": np.where(p6.suppressed6, "suppressed_star",
                                   np.where(p6.value6.isna(), "unparseable", None)),
        "source": SOURCE_6,
    })
    panel = pd.concat([p45, p6], ignore_index=True)

    # Two raw rows mapping to the same district x indicator x round: keep one; NaN if conflicting.
    key = ["district_uid", "indicator_id", "round"]
    dup = panel.duplicated(key, keep=False)
    if dup.any():
        conflict = panel[dup].groupby(key).value.nunique(dropna=True) > 1
        bad = conflict[conflict].index
        panel = panel.drop_duplicates(key, keep="first").reset_index(drop=True)
        if len(bad):
            idx = panel.set_index(key).index.isin(bad)
            panel.loc[idx, "value"] = np.nan
            panel.loc[idx, "missing_reason"] = "conflicting_duplicate"
    panel["round"] = panel["round"].astype(int)
    return panel.sort_values(["state", "district", "indicator_id", "round"]).reset_index(drop=True)


# =====================================================================
# Audit: NFHS-6 file's own NFHS-5 column vs India.csv NFHS-5
# =====================================================================
def audit_nfhs5(l45: pd.DataFrame, d6: pd.DataFrame, variants: pd.DataFrame,
                name_map: pd.DataFrame, cross: pd.DataFrame, min_indicator_agreement: float = 0.9,
                district_indicators: set[str] | None = None
                ) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Compare NFHS-5 values in the two sources for matched district-indicator pairs.

    Returns (pairs, summary, by_district, by_indicator). Uses only NFHS-5 values.
    by_indicator: share of districts agreeing within 0.5 per indicator. An
    indicator below `min_indicator_agreement` disagrees systematically
    (recalculated or mis-parsed), so by_district is computed only over the
    indicators at or above it (and, if given, within `district_indicators`,
    i.e. the core set: a misaligned row in an unused indicator should not drop
    a district).
    """
    matched_ind = set(cross.loc[cross.method.isin(["exact", "prefix", "fuzzy"]), "indicator_id"])
    v45 = _id_lookup(variants, "nfhs45")
    v6 = _id_lookup(variants, "nfhs6")
    nm = name_map[name_map.method.isin(["exact", "exact_nospace", "alias", "fuzzy"])]

    a = l45[l45["round"] == 5].copy()
    a["indicator_id"] = [v45.get((c, t)) for c, t in zip(a.Category, a.Indicator)]
    a = a.merge(nm[["state_nfhs45", "district_nfhs45", "district_uid"]],
                left_on=["State", "District"], right_on=["state_nfhs45", "district_nfhs45"])
    a["india_raw"] = pd.to_numeric(a.value_raw, errors="coerce")
    a = a[["district_uid", "indicator_id", "india_raw", "value", "missing_reason"]].rename(
        columns={"value": "india_clean", "missing_reason": "india_missing_reason"})

    b = d6.copy()
    b["indicator_id"] = [v6.get((c, t)) for c, t in zip(b.category, b.indicator)]
    b = b.merge(nm[["state_nfhs6", "district_nfhs6", "district_uid"]],
                left_on=["state", "district"], right_on=["state_nfhs6", "district_nfhs6"])
    b = b[["district_uid", "indicator_id", "value5_file", "suppressed5"]]

    pairs = a.merge(b, on=["district_uid", "indicator_id"])
    pairs = pairs[pairs.indicator_id.isin(matched_ind)]
    pairs = pairs.drop_duplicates(["district_uid", "indicator_id"])
    pairs["diff"] = pairs.india_clean - pairs.value5_file
    both = pairs.dropna(subset=["india_clean", "value5_file"])
    exact = (both["diff"].abs() < 0.05)
    within = (both["diff"].abs() <= 0.5)

    star = pairs[pairs.suppressed5]
    zero_in_india = pairs.india_raw.eq(0)
    summary = {
        "pairs_compared": int(len(pairs)),
        "pairs_both_numeric": int(len(both)),
        "exact_agreement_rate": float(exact.mean()) if len(both) else np.nan,
        "within_0_5_agreement_rate": float(within.mean()) if len(both) else np.nan,
        "nfhs6file_star_pairs": int(len(star)),
        "nfhs6file_star_and_india_zero": int((star.india_raw == 0).sum()),
        "nfhs6file_star_and_india_nonzero": int((star.india_raw != 0).sum()),
        "india_zero_pairs": int(zero_in_india.sum()),
        "india_zero_and_nfhs6file_star": int((zero_in_india & pairs.suppressed5).sum()),
        "india_zero_and_nfhs6file_zero": int((zero_in_india & pairs.value5_file.eq(0)).sum()),
        "india_zero_and_nfhs6file_positive": int((zero_in_india & pairs.value5_file.gt(0)).sum()),
    }
    both = both.assign(exact=exact, within=within)
    by_ind = both.groupby("indicator_id").agg(
        audit_pairs=("diff", "size"), audit_within_0_5=("within", "mean"),
        audit_mean_diff=("diff", "mean")).reset_index()
    reliable = set(by_ind.loc[by_ind.audit_within_0_5 >= min_indicator_agreement, "indicator_id"])
    summary["indicators_below_agreement_threshold"] = sorted(set(by_ind.indicator_id) - reliable)
    if district_indicators is not None:
        reliable &= set(district_indicators)
    by = both[both.indicator_id.isin(reliable)].groupby("district_uid").agg(
        audit_pairs=("diff", "size"), audit_exact=("exact", "mean"),
        audit_within_0_5=("within", "mean")).reset_index()
    return pairs, summary, by, by_ind


# =====================================================================
# Codebook
# =====================================================================
def build_codebook(cross: pd.DataFrame, panel: pd.DataFrame, core_uids: set[str],
                   cfg: dict, audit_by_indicator: pd.DataFrame | None = None) -> pd.DataFrame:
    """Indicator codebook with include flags.

    Selection uses text matching, definitions, NFHS-4/5 coverage and the NFHS-5
    cross-source audit only; no NFHS-6 values.
    """
    p = panel[panel.district_uid.isin(core_uids) & panel["round"].isin([4, 5])]
    n_core = max(len(core_uids), 1)
    cov = (p.dropna(subset=["value"]).groupby(["indicator_id", "round"]).district_uid.nunique()
           / n_core).unstack("round")
    cb = cross.copy()
    cb["coverage_nfhs4"] = cb.indicator_id.map(cov.get(4, pd.Series(dtype=float))).fillna(0.0)
    cb["coverage_nfhs5"] = cb.indicator_id.map(cov.get(5, pd.Series(dtype=float))).fillna(0.0)
    agree = (audit_by_indicator.set_index("indicator_id").audit_within_0_5
             if audit_by_indicator is not None else pd.Series(dtype=float))
    cb["audit_nfhs5_agreement"] = cb.indicator_id.map(agree)
    min_agree = cfg["indicators"].get("min_audit_agreement", 0.0)
    cb["percent_indicator"] = cb.text6.fillna(cb.text45).fillna("").str.contains(r"\(%\)") | \
        cb.text45.fillna("").str.contains(r"\(%\)")
    conf = np.select([cb.method.isin(["exact", "prefix"]),
                      (cb.method == "fuzzy") & (cb.score >= 97),
                      cb.method == "fuzzy"], ["high", "medium", "low"], default="none")
    cb["match_confidence"] = conf
    min_cov = cfg["indicators"]["min_coverage_rounds_4_5"]

    reasons = []
    for r in cb.itertuples():
        if r.method == "unmatched":
            reasons.append("not present in all three rounds")
        elif r.definition_exclusion:
            reasons.append(f"definition change: {r.definition_exclusion}")
        elif r.match_confidence not in ("high", "medium"):
            reasons.append(f"low-confidence text match ({r.method}, score {r.score:.0f})")
        elif not r.percent_indicator:
            reasons.append("not a percentage (SE formula does not apply)")
        elif r.coverage_nfhs4 < min_cov or r.coverage_nfhs5 < min_cov:
            reasons.append(f"coverage in core districts below {min_cov:.0%} "
                           f"(NFHS-4 {r.coverage_nfhs4:.0%}, NFHS-5 {r.coverage_nfhs5:.0%})")
        elif pd.notna(r.audit_nfhs5_agreement) and r.audit_nfhs5_agreement < min_agree:
            reasons.append(f"NFHS-5 values disagree between sources ({r.audit_nfhs5_agreement:.0%} within "
                           f"0.5 < {min_agree:.0%}): recalculated or mis-parsed, not comparable")
        elif not r.core:
            reasons.append("comparable in all rounds but outside the core set (add to config to include)")
        else:
            reasons.append("")
    cb["exclude_reason"] = reasons
    cb["include"] = cb.exclude_reason.eq("")
    cb["comparable_all_rounds"] = cb.exclude_reason.isin(
        ["", "comparable in all rounds but outside the core set (add to config to include)"])
    out = pd.DataFrame({
        "indicator_id": cb.indicator_id, "include": cb.include, "core": cb.core,
        "domain": cb.domain, "se_group": cb.se_group,
        "text_nfhs4": cb.text45, "text_nfhs5": cb.text45, "text_nfhs6": cb.text6,
        "category_nfhs45": cb.category45, "category_nfhs6": cb.category6, "sex": cb.sex,
        "match_method": cb.method, "match_score": cb.score, "match_confidence": cb.match_confidence,
        "coverage_nfhs4": cb.coverage_nfhs4.round(3), "coverage_nfhs5": cb.coverage_nfhs5.round(3),
        "audit_nfhs5_agreement": cb.audit_nfhs5_agreement.round(3),
        "comparable_all_rounds": cb.comparable_all_rounds, "exclude_reason": cb.exclude_reason,
    })
    order = {"exact": 0, "prefix": 1, "fuzzy": 2, "unmatched": 3}
    out = out.assign(_o=out.match_method.map(order)).sort_values(
        ["include", "comparable_all_rounds", "_o", "indicator_id"],
        ascending=[False, False, True, True]).drop(columns="_o")
    return out.reset_index(drop=True)
