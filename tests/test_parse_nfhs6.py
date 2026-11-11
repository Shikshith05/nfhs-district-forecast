"""Independent NFHS-6 PDF parser and spot-check (Task A4). Synthetic fact sheets only."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from nfhs_forecast import parse_nfhs6 as pp
from tests.conftest import REPO

SHEET = """National Family Health Survey - 6
District Fact Sheet
Indicators NFHS-6 (2023-24) NFHS-5 (2019-21)
Population and Household Profile
1. Female population age 6 years and above who ever attended school (%) 72.3 70.1
2. Population below age 15 years (%) 24.0 25.4
Marriage and Fertility
3. Women age 20-24 years married before age 18
years (%) (12.5) 18.0
4. Women age 15-19 years who were already mothers or pregnant at the time of the survey (%) * 4.1
5. Children age 12-23 months fully vaccinated based on information from either vaccination card or mother's recall (%) na 80.0
6. Men age 15 years and above who consume alcohol 15 20.1 21.0
7. Total unmet need for family planning (%)
"""


def test_parse_value_markers():
    assert pp.parse_value("12.5")[0] == 12.5
    v, sup, small, na = pp.parse_value("(12.5)")
    assert v == 12.5 and small and not sup
    assert pp.parse_value("*")[1] and np.isnan(pp.parse_value("*")[0])
    assert pp.parse_value("na")[3]


def test_parse_fact_sheet_text():
    rows = pp.parse_fact_sheet_text(SHEET).set_index("indicator_no")
    assert rows.loc[1, "nfhs6_value"] == 72.3 and rows.loc[1, "nfhs5_value"] == 70.1
    assert rows.loc[1, "parse_status"] == "ok"
    assert "(%)" in rows.loc[1, "indicator"]
    # wrapped row
    assert rows.loc[3, "indicator"].endswith("years (%)") and rows.loc[3, "nfhs6_small_sample"]
    assert rows.loc[3, "nfhs6_value"] == 12.5 and rows.loc[3, "nfhs5_value"] == 18.0
    assert rows.loc[4, "nfhs6_suppressed"] and rows.loc[4, "nfhs5_value"] == 4.1
    assert rows.loc[5, "nfhs6_na"]
    # text ends in a number before the values: ambiguous, not used
    assert rows.loc[6, "parse_status"] == "ambiguous"
    # no values at all: incomplete, no value invented
    assert rows.loc[7, "parse_status"] == "incomplete" and np.isnan(rows.loc[7, "nfhs6_value"])


def _sample(rows):
    return pd.DataFrame([dict(state="Kerala", district="Wayanad", indicator_no=n, indicator=t, nfhs6_value=a,
                              nfhs5_value=b) for n, t, a, b in rows])


def test_compare_pairs_agreement_and_statuses():
    parsed = pp.parse_fact_sheet_text(SHEET).assign(state="Kerala", district="Wayanad")
    sample = _sample([
        (1, "Female population age 6 years and above who ever attended school (%)", "72.3", "70.1"),  # agree
        (2, "Population below age 15 years (%)", "24.0", "26.0"),                                    # NFHS-5 differs
        (3, "Women age 20-24 years married before age 18 years (%)", "12.5", "18.0"),                # agree
        (4, "Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)", "*", "4.1"),
        (6, "Men age 15 years and above who consume alcohol (%)", "20.1", "21.0"),                   # ambiguous
        (9, "Something not in the sheet (%)", "1.0", "2.0"),
    ])
    sample.loc[len(sample)] = dict(state="Kerala", district="Idukki", indicator_no=1,
                                   indicator="Population below age 15 years (%)", nfhs6_value="1", nfhs5_value="2")
    res = pp.compare_pairs(sample, parsed).set_index(["district", "indicator_no"])
    w = res.loc["Wayanad"]
    assert w.loc[1, "status"] == "compared" and w.loc[1, "agree_nfhs6"] and w.loc[1, "agree_nfhs5"]
    assert w.loc[2, "agree_nfhs6"] and not w.loc[2, "agree_nfhs5"]
    assert w.loc[3, "agree_nfhs6"] and w.loc[3, "pdf_small_sample_nfhs6"]
    assert w.loc[4, "agree_nfhs6"]                       # both suppressed
    assert w.loc[6, "status"].startswith("pdf_row_") or w.loc[6, "status"].startswith("indicator_")
    assert w.loc[9, "status"] == "indicator_no_match"
    assert res.loc[("Idukki", 1), "status"] == "district_not_parsed"


def test_wilson_ci():
    lo, hi = pp.wilson_ci(95, 100)
    assert 0.88 < lo < 0.95 < hi < 0.99


def _make_pdf(path, text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(14, 8))
    for i, line in enumerate(text.splitlines()):
        fig.text(0.01, 0.97 - i * 0.045, line, fontsize=8, family="DejaVu Sans")
    fig.savefig(path)
    plt.close(fig)


def test_pdf_roundtrip_and_spot_check_script(tmp_path, cfg):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_pdf(pdf_dir / "wayanad.pdf", SHEET)
    rows, info = pp.parse_district_pdf(pdf_dir / "wayanad.pdf")
    if info["status"] != "parsed" or (rows.parse_status == "ok").sum() < 3:
        pytest.skip("matplotlib PDF text layer not extractable on this platform")
    assert rows.set_index("indicator_no").loc[2, "nfhs6_value"] == 24.0

    # the script on a fake NFHS-6 CSV with the same district
    pd.DataFrame({"file": ["wayanad.pdf"], "state": ["Kerala"], "district": ["Wayanad"]}).to_csv(
        pdf_dir / "manifest.csv", index=False)
    csv = pd.DataFrame({"state": "Kerala", "district": "Wayanad", "category": "x",
                        "indicator_no": ["1", "2"], "indicator": [
                            "Female population age 6 years and above who ever attended school (%)",
                            "Population below age 15 years (%)"],
                        "nfhs6_value": ["72.3", "24.5"], "nfhs5_value": ["70.1", "25.4"],
                        "small_sample_flag": "False", "boundary_change_single_round": "False"})
    csv.to_csv(tmp_path / "n6.csv", index=False)
    import yaml
    raw = {k: v for k, v in cfg.items() if k != "abs_paths"}
    raw = yaml.safe_load(yaml.safe_dump(raw))
    raw["paths"]["nfhs6_csv"] = "n6.csv"
    raw["paths"]["outputs_dir"] = "out"
    raw["pdf_check"]["pdf_dir"] = "pdfs"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("spot", REPO / "scripts" / "spot_check_nfhs6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(tmp_path / "config.yaml"), "--all"]) == 0
    res = pd.read_csv(tmp_path / "out" / "nfhs6_spot_check.csv").set_index("indicator_no")
    assert res.loc[1, "agree_nfhs6"] and not res.loc[2, "agree_nfhs6"]
    md = (tmp_path / "out" / "nfhs6_spot_check.md").read_text()
    assert "| NFHS6 | 1 | 2 | 50.0% |" in md


def test_spot_check_script_without_pdfs(tmp_path, cfg):
    spec = importlib.util.spec_from_file_location("spot", REPO / "scripts" / "spot_check_nfhs6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(REPO / "config.yaml"), "--pdf-dir", str(tmp_path),
                     "--out-dir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()
