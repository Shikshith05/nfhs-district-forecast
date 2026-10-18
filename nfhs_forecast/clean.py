"""Value, name and indicator-text normalisation, and per-source cleaning.

Known source issues handled here are listed in DECISIONS.md (D04-D09).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------
_MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "-", "--", "—", "–"}


def clean_value(x) -> float:
    """Convert one fact-sheet cell to float.

    '*' (suppressed: fewer than 25 unweighted cases) -> NaN. Brackets and
    parentheses (small-sample values based on 25-49 cases) are stripped, as
    are '%' and thousands separators. Anything unparseable -> NaN.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if "*" in s:
        return np.nan
    s = re.sub(r"[\(\)\[\]\{\}%,]", "", s).strip()
    if s.lower() in _MISSING_TOKENS:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_values(series: pd.Series) -> pd.Series:
    return series.map(clean_value).astype(float)


def is_suppressed(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains(r"\*", regex=True)


# ---------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------
def normalise_name(s: str) -> str:
    """Normalise a state or district name: case, '&' vs 'and', punctuation, spacing."""
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def squash(s: str) -> str:
    """Remove everything except letters and digits ('Bara Banki' == 'Barabanki')."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------------
# Indicator text
# ---------------------------------------------------------------------
_FOOTNOTE_GLUED = re.compile(r"(?<=[a-z\)])\d+(?:\s*,\s*\d+)*")
_FOOTNOTE_TRAILING = re.compile(r"(?:\s\d{1,2})+$")


def normalise_indicator_text(s: str) -> str:
    """Normalise indicator text so that variants of the same indicator compare equal.

    lowercase; comparison symbols -> words; drop '(%)'; drop footnote digits
    glued to a word or ')' ('tetanus4', 'diet10, 11', 'kg/m2)15'); join
    hyphenated words ('drinking-water' == 'drinkingwater', '20-24' == '2024',
    as India.csv lost its hyphens); strip punctuation; collapse whitespace;
    drop trailing 1-2 digit footnote numbers ('tetanus 4').
    """
    s = str(s).lower()
    s = (s.replace("≥", " ge ").replace("≤", " le ")
          .replace(">", " gt ").replace("<", " lt ")
          .replace("&", " and ").replace("’", "'"))
    s = re.sub(r"\(\s*%\s*\)|%", " ", s)
    s = _FOOTNOTE_GLUED.sub("", s)
    s = s.replace("-", "").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _FOOTNOTE_TRAILING.sub("", s)
    return s


def sex_qualifier(text_norm: str, category: str, pattern: str) -> str:
    """'women' / 'men' for indicators whose text repeats across sexes, else ''."""
    if not re.search(pattern, text_norm):
        return ""
    probe = f"{category} {text_norm}".lower()
    if re.search(r"\b(women|female)", probe):
        return "women"
    if re.search(r"\b(men|male)", probe):
        return "men"
    return ""


def indicator_key(text: str, category: str, pattern: str) -> tuple[str, str, str]:
    """Return (sex, normalised text, squashed text) used as the matching key."""
    norm = normalise_indicator_text(text)
    sex = sex_qualifier(norm, category, pattern)
    if sex:
        norm = re.sub(r"^(fe)?male\s*", "", norm)
    return sex, norm, squash(norm)


# ---------------------------------------------------------------------
# India.csv (NFHS-4 and NFHS-5)
# ---------------------------------------------------------------------
def _split_mislabelled_blocks(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Fix districts whose rows carry the wrong 'District' name.

    In India.csv a (State, District) pair can hold two full blocks of rows
    with different DISTRICT keys (e.g. 'Sahibganj' holds 'Sahibganj_7' and
    'Saraikela-kharsawan_24'). A block whose DISTRICT key does not resemble
    the District name is renamed to the key's name.
    """
    fixes = []
    sizes = df.groupby(["State", "District", "DISTRICT"]).size()
    typical = df.groupby(["State", "District"]).size().median()
    for (state, district), grp in sizes.groupby(level=[0, 1]):
        big = grp[grp >= 0.5 * typical]
        if len(big) < 2:
            continue
        for (_, _, key), n in big.items():
            name_part = key.rsplit("_", 1)[0]
            score = fuzz.ratio(normalise_name(name_part), normalise_name(district))
            if score < 80:
                m = (df.State == state) & (df.District == district) & (df.DISTRICT == key)
                df.loc[m, "District"] = name_part
                fixes.append({"state": state, "district_in_source": district,
                              "district_key": key, "renamed_to": name_part, "rows": int(n)})
    return df, fixes


def _district_codes(df: pd.DataFrame, geo: pd.DataFrame | None) -> pd.DataFrame:
    """Census 2011 code per (State, District) via the modal DISTRICT key and the GeoJSON."""
    keyed = df[df.DISTRICT != ""]
    mode = (keyed.groupby(["State", "District"]).DISTRICT
            .agg(lambda s: s.value_counts().index[0]).rename("geo_key"))
    share = (keyed.groupby(["State", "District"]).DISTRICT
             .agg(lambda s: s.value_counts().iloc[0]) /
             df.groupby(["State", "District"]).size()).rename("share")
    codes = pd.concat([mode, share], axis=1).dropna()
    codes = codes[codes.share >= 0.5].drop(columns="share").reset_index()
    st = df.groupby(["State", "District"]).ST_CEN_CD.first().reset_index()
    codes = codes.merge(st, on=["State", "District"], how="left")
    if geo is not None and len(geo):
        g = geo[["DISTRICT", "ST_CEN_CD", "censuscode"]].copy()
        g["ST_CEN_CD"] = g.ST_CEN_CD.astype(str)
        g["censuscode"] = pd.to_numeric(g.censuscode, errors="coerce").astype("Int64").astype(str)
        g = g.rename(columns={"DISTRICT": "geo_key"}).drop_duplicates(["geo_key", "ST_CEN_CD"])
        codes = codes.merge(g, on=["geo_key", "ST_CEN_CD"], how="left")
        codes["district_code"] = codes.censuscode.where(codes.censuscode.notna() & (codes.censuscode != "<NA>"))
    else:
        codes["district_code"] = codes.ST_CEN_CD + "-" + codes.geo_key.str.rsplit("_", n=1).str[-1]
    return codes[["State", "District", "geo_key", "district_code"]]


def clean_nfhs45(raw: pd.DataFrame, cfg: dict, geo: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, dict]:
    """Clean India.csv into long format with one row per district x indicator x round.

    Returns (long, log). `long` has columns State, District, geo_key,
    district_code, Category, Indicator, round, value_raw, value, missing_reason.
    """
    ccfg = cfg["cleaning"]
    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    log: dict = {"raw_rows": len(df),
                 "raw_districts": df[["State", "District"]].drop_duplicates().shape[0],
                 "raw_indicator_texts": df.Indicator.nunique()}

    drop = pd.Series(False, index=df.index)
    for state, district in ccfg.get("excluded_districts_nfhs45", []):
        drop |= (df.State == state) & (df.District == district)
    log["excluded_districts"] = [list(x) for x in ccfg.get("excluded_districts_nfhs45", [])]
    log["excluded_rows"] = int(drop.sum())
    df = df[~drop].copy()

    df, fixes = _split_mislabelled_blocks(df)
    log["renamed_blocks"] = fixes
    codes = _district_codes(df, geo)

    long = df.melt(id_vars=["State", "District", "Category", "Indicator"],
                   value_vars=["NFHS 4", "NFHS 5"], var_name="col", value_name="value_raw")
    long["round"] = long.col.map({"NFHS 4": 4, "NFHS 5": 5}).astype(int)
    long = long.drop(columns="col")
    long["value"] = clean_values(long.value_raw)
    long["missing_reason"] = np.where(long.value.isna(), "unparseable", None)

    # Duplicated district x indicator x round: identical -> keep one; conflicting -> NaN.
    key = ["State", "District", "Indicator", "round"]
    n_before = len(long)
    nvals = long[long.duplicated(key, keep=False)].groupby(key).value.nunique(dropna=False)
    bad = nvals[nvals > 1].index
    long = long.drop_duplicates(key, keep="first").reset_index(drop=True)
    if len(bad):
        idx = long.set_index(key).index.isin(bad)
        long.loc[idx, "value"] = np.nan
        long.loc[idx, "missing_reason"] = "conflicting_duplicate"
    log["duplicate_rows_dropped"] = n_before - len(long)
    log["conflicting_duplicates"] = [dict(zip(key, k)) for k in bad]

    # Zero handling (the parser wrote 'na' and '*' as 0).
    zero = long.value.eq(0)
    dshare = zero.groupby([long.State, long.District, long["round"]]).transform("mean")
    rule1 = dshare >= ccfg["district_zero_share_missing"]
    long.loc[rule1, ["value", "missing_reason"]] = [np.nan, "district_round_not_available"]

    live = long.missing_reason.isna()
    zero = long.value.eq(0)
    ishare = zero[live].groupby([long.Indicator[live], long["round"][live]]).mean()
    ishare = ishare.rename("ishare").reset_index()
    long = long.merge(ishare, on=["Indicator", "round"], how="left")
    rule2 = long.missing_reason.isna() & (long.ishare >= ccfg["indicator_zero_share_missing"])
    long.loc[rule2, ["value", "missing_reason"]] = [np.nan, "indicator_not_in_round"]

    live = long.missing_reason.isna()
    nz = long[live & long.value.gt(0)]
    below1 = (nz.value < 1).groupby([nz.Indicator, nz["round"]]).mean().rename("below1").reset_index()
    long = long.merge(below1, on=["Indicator", "round"], how="left")
    implausible = long.below1.fillna(0) < ccfg["plausible_zero_min_share_below1"]
    rule3 = long.missing_reason.isna() & long.value.eq(0) & implausible
    long.loc[rule3, ["value", "missing_reason"]] = [np.nan, "zero_treated_as_suppressed"]
    long = long.drop(columns=["ishare", "below1"])

    log["zero_rules"] = {
        "district_round_not_available_rows": int(rule1.sum()),
        "district_rounds_not_available": int(
            long[long.missing_reason == "district_round_not_available"]
            [["State", "District", "round"]].drop_duplicates().shape[0]),
        "indicator_not_in_round_rows": int(rule2.sum()),
        "zero_treated_as_suppressed_rows": int(rule3.sum()),
        "zeros_kept_as_true_zero": int((long.value == 0).sum()),
    }
    long = long.merge(codes, on=["State", "District"], how="left")
    log["districts_with_census_code"] = int(codes.district_code.notna().sum())
    log["clean_districts"] = long[["State", "District"]].drop_duplicates().shape[0]
    return long, log


# ---------------------------------------------------------------------
# NFHS-6
# ---------------------------------------------------------------------
def clean_nfhs6(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Clean the NFHS-6 district CSV (still one row per district x indicator).

    Adds value6 (NFHS-6), value5_file (the file's own NFHS-5 column, used only
    for the audit), suppressed6/suppressed5, and boolean flags.
    """
    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    log: dict = {"raw_rows": len(df),
                 "raw_states": df.state.nunique(),
                 "raw_districts": df[["state", "district"]].drop_duplicates().shape[0]}
    excl = cfg["cleaning"].get("excluded_states_nfhs6", [])
    drop = df.state.isin(excl)
    log["excluded_states"] = {s: int((df.state == s).sum()) for s in excl}
    df = df[~drop].copy()

    df["suppressed6"] = is_suppressed(df.nfhs6_value)
    df["suppressed5"] = is_suppressed(df.nfhs5_value)
    df["value6"] = clean_values(df.nfhs6_value)
    df["value5_file"] = clean_values(df.nfhs5_value)
    df["small_sample_flag"] = df.small_sample_flag.str.lower().eq("true")
    df["boundary_change_flag"] = df.boundary_change_single_round.str.lower().eq("true")

    rows_per = df.groupby(["state", "district"]).size()
    log["state_sheet_districts"] = {f"{s} | {d}": int(n) for (s, d), n in rows_per.items()
                                    if n != rows_per.mode().iloc[0]}
    log["suppressed_nfhs6"] = int(df.suppressed6.sum())
    log["suppressed_nfhs5_in_file"] = int(df.suppressed5.sum())
    log["small_sample_rows"] = int(df.small_sample_flag.sum())
    log["boundary_change_districts"] = int(
        df[df.boundary_change_flag][["state", "district"]].drop_duplicates().shape[0])
    log["clean_rows"] = len(df)
    log["clean_districts"] = df[["state", "district"]].drop_duplicates().shape[0]
    return df, log
