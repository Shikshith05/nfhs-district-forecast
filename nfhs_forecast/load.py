"""Raw data readers. Everything is read as strings; cleaning happens in clean.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NFHS45_COLUMNS = ["State", "ST_CEN_CD", "District", "DISTRICT", "DT_CEN_CD",
                  "Category", "Indicator", "NFHS 5", "NFHS 4"]
NFHS6_COLUMNS = ["state", "district", "category", "indicator_no", "indicator",
                 "nfhs6_value", "nfhs5_value", "small_sample_flag",
                 "boundary_change_single_round"]


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; found {list(df.columns)}")


def read_nfhs45(path: str | Path) -> pd.DataFrame:
    """India.csv: NFHS-5 district fact sheets with the NFHS-4 comparison column."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    _check_columns(df, NFHS45_COLUMNS, "India.csv")
    return df


def read_nfhs6(path: str | Path) -> pd.DataFrame:
    """all_india_nfhs6_district.csv: NFHS-6 district fact sheets (provisional)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    _check_columns(df, NFHS6_COLUMNS, "NFHS-6 CSV")
    return df


def read_geo_properties(path: str | Path) -> pd.DataFrame:
    """Feature properties (no geometry) of Districts.geojson: Census 2011 districts."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    rows = [feat["properties"] for feat in gj["features"]]
    return pd.DataFrame(rows)
