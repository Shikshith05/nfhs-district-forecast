"""Public release of the harmonised panel and codebook as CSV (outputs/release/; DECISIONS D40).

Before Gate B the release holds NFHS-4 and NFHS-5 only. NFHS-6 values are
added by scripts/post_run.py after the one-shot evaluation (include_nfhs6=True),
so no NFHS-6 outcome value is copied or used before the test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import write_csv, write_json, write_text

PANEL_COLS = ["unit_uid", "state", "district", "district_code", "round", "indicator_id", "value", "se",
              "se_sampling", "extra_var", "n_assumed", "se_group", "value_method", "harmonised", "parent_uid",
              "small_sample_flag", "missing_reason"]
UNIT_COLS = ["unit_uid", "state", "unit_type", "train_unit", "test_unit", "stable_unit", "parent_uid",
             "members_nfhs5", "members_nfhs6", "confidence", "audit_status", "nfhs4_inherited", "district_code"]
CODEBOOK_COLS = ["indicator_id", "domain", "se_group", "assumed_n", "text_nfhs4", "text_nfhs5", "text_nfhs6",
                 "match_method", "match_confidence", "coverage_nfhs4", "coverage_nfhs5", "audit_nfhs5_agreement"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_release(cfg: dict, include_nfhs6: bool = False, out_dir: Path | None = None) -> Path:
    P = cfg["abs_paths"]
    root = Path(P["outputs_dir"]).parent
    out = Path(out_dir) if out_dir else root / cfg["post_run"]["release_dir"]
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(P["processed_dir"] / "panel_model.parquet")
    units = pd.read_csv(Path(P["outputs_dir"]) / "harmonised_units.csv")
    codebook = pd.read_csv(P["processed_dir"] / "indicator_codebook.csv")
    rounds = [4, 5, 6] if include_nfhs6 else [4, 5]

    p = panel[panel["round"].isin(rounds) & panel.district_uid.isin(set(units.unit_uid))].copy()
    p = p.rename(columns={"district_uid": "unit_uid"})
    p = p[PANEL_COLS].sort_values(["unit_uid", "indicator_id", "round"]).reset_index(drop=True)
    write_csv(p, out / "harmonised_panel.csv", float_format="%.6g")

    codes = (panel[panel["round"] == 5].dropna(subset=["district_code"]).groupby("district_uid").district_code.first())
    u = units.assign(district_code=units.unit_uid.map(codes))[UNIT_COLS]
    write_csv(u, out / "units.csv")

    sizes = cfg["noise"]["sample_size"]
    cb = codebook[codebook.include].assign(assumed_n=lambda d: d.se_group.map(sizes))[CODEBOOK_COLS]
    write_csv(cb.sort_values("indicator_id"), out / "codebook.csv")
    write_csv(pd.read_csv(Path(P["outputs_dir"]) / "crosswalk.csv"), out / "crosswalk.csv")

    write_text(out / "DATA_DICTIONARY.md", data_dictionary(cfg, include_nfhs6))
    write_text(out / "SOURCES_AND_LICENCE.md", SOURCES)
    files = ["harmonised_panel.csv", "units.csv", "codebook.csv", "crosswalk.csv", "DATA_DICTIONARY.md",
             "SOURCES_AND_LICENCE.md"]
    write_json({"rounds": rounds, "includes_nfhs6": include_nfhs6, "n_units": int(u.shape[0]),
                "n_indicators": int(cb.shape[0]), "n_panel_rows": int(p.shape[0]),
                "frozen_train_hash": _manifest(root).get("train_hash"),
                "sha256": {f: _sha(out / f) for f in files}}, out / "release_manifest.json")
    return out


def _manifest(root: Path) -> dict:
    import json
    path = root / "outputs" / "models" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def data_dictionary(cfg: dict, include_nfhs6: bool) -> str:
    sizes = cfg["noise"]["sample_size"]
    groups = "\n".join(f"| `{k}` | {v} |" for k, v in sizes.items())
    status = ("Rounds 4, 5 and 6 (NFHS-6 added after the one-shot evaluation)." if include_nfhs6 else
              "Rounds 4 and 5 only. NFHS-6 values are added after the one-shot evaluation (Gate B), so that no "
              "NFHS-6 outcome is used or distributed before the test.")
    return f"""# Data dictionary: harmonised NFHS district panel

{status}

All standard errors are **model assumptions**, not published figures (DECISIONS D16, D29).
Percentages are 0-100. Missing values are empty cells.

## harmonised_panel.csv (one row per unit x indicator x round)

| column | meaning |
|---|---|
| `unit_uid` | analysis unit: `<state>|<district>` (NFHS-6 name; normalised, lower case) |
| `state`, `district` | display names |
| `district_code` | Census 2011 district code (India.csv / Districts.geojson); empty for districts created after 2011 |
| `round` | 4 = NFHS-4 (2015-16), 5 = NFHS-5 (2019-21), 6 = NFHS-6 (2023-24, provisional) |
| `indicator_id` | see codebook.csv |
| `value` | district estimate (%) |
| `se` | total assumed SE = sqrt(se_sampling^2 + extra_var) |
| `se_sampling` | assumed sampling SE = 100*sqrt(p(1-p)/n*deff), deff = {cfg['noise']['deff']} |
| `extra_var` | added variance for a value standing in for a changed boundary (D29); 0 otherwise |
| `n_assumed` | assumed per-district sample size (see below) |
| `se_group` | population group that sets `n_assumed` |
| `value_method` | `observed`; `observed_changed_boundary` (NFHS-6 namesake of a changed district); `inherited`; `aggregated` |
| `harmonised` | True if `value_method` is not `observed` |
| `parent_uid` | NFHS-5 parent unit (bootstrap cluster for harmonised units) |
| `small_sample_flag` | NFHS-6 fact-sheet marker for 25-49 unweighted cases (round 6 only) |
| `missing_reason` | why a value is missing (suppressed, not measured, district not available, ...) |

Assumed sample sizes (`config.yaml` noise.sample_size; small-sample rows use n = {cfg['noise']['small_sample_n']}):

| se_group | assumed n |
|---|---|
{groups}

## units.csv (one row per analysis unit)

| column | meaning |
|---|---|
| `unit_type` | `identity`, `inherited_remnant`, `inherited_child`, `aggregated`, `identity_train_only` |
| `train_unit` | used to train on NFHS-4 -> NFHS-5 |
| `test_unit` | scored on NFHS-5 -> NFHS-6 |
| `stable_unit` | identical boundaries in all three rounds (stable-boundary subset) |
| `members_nfhs5`, `members_nfhs6` | districts in the unit's lineage component |
| `confidence` | lowest confidence of the lineage edges used |
| `audit_status` | NFHS-5 cross-source audit: pass / unaudited |
| `nfhs4_inherited` | NFHS-4 value inherited from a Census-2011 parent |

## codebook.csv (one row per included indicator)

Indicator texts in each round, matching method and confidence, coverage in the stable districts, NFHS-5
cross-source audit agreement, and the assumed n.

## crosswalk.csv

Every lineage edge with its source, confidence and whether it was used (D27).
"""


SOURCES = """# Sources and licence

## Underlying data

- **NFHS-4 and NFHS-5:** International Institute for Population Sciences (IIPS) and ICF, National Family
  Health Survey district fact sheets (http://rchiips.org/nfhs/), via the community parse
  https://github.com/SaiSiddhardhaKalla/NFHS (`India.csv`). That repository shows **no licence** (checked
  2026-10-23); its README states the data are sourced from IIPS.
- **NFHS-6 (provisional):** IIPS, NFHS-6 (2023-24) State and District Fact Sheets
  (https://iipsindia.ac.in/content/nfhs-projects), via Patra, N. K. (2026), *NFHS-6 Provisional
  District-Level Data (India): Independent Compilation*, v1.0.0
  (https://github.com/nabinpatra007/nfhs6-district-data-india). Its MIT licence covers code only; its data
  notice states that the licence does not apply to NFHS-6 / IIPS data.
- **Boundaries:** Census 2011 district polygons distributed with the SaiSiddhardhaKalla/NFHS data.

## This release

The tables are derived by Team CodeFlow (cleaning, matching, harmonisation, assumed standard errors).
The NFHS estimates remain the work of IIPS and the Ministry of Health and Family Welfare, Government of
India, and are subject to their terms of use. The project repository has no licence file of its own: the
team must choose one for its code and derived tables before publishing them, and must cite IIPS and both
community sources when this release is reused.

**NFHS-6 values are provisional** and were parsed from PDFs by a third party; check them against the
official fact sheets before relying on any single value.
"""
