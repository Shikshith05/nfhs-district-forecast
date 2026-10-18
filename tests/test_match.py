import pytest

from nfhs_forecast import clean, match


@pytest.fixture(scope="module")
def matched(raw45, raw6, cfg):
    l45, _ = clean.clean_nfhs45(raw45, cfg)
    d6, _ = clean.clean_nfhs6(raw6, cfg)
    variants, cross = match.build_indicator_matching(l45, d6, cfg)
    return l45, d6, variants, cross


def _id(variants, source, text):
    v = variants[(variants.source == source) & (variants.raw_text == text)]
    assert len(v), f"variant not found: {text}"
    return v.indicator_id.iloc[0]


def test_footnote_variants_share_an_id(matched):
    _, _, variants, cross = matched
    a = _id(variants, "nfhs6", "Mothers whose last birth was protected against neonatal tetanus4 (%)")
    b = _id(variants, "nfhs6", "Mothers whose last birth was protected against neonatal tetanus 4 (%)")
    c = _id(variants, "nfhs45", "Mothers whose last birth was protected against neonatal tetanus (%)")
    assert a == b == c
    assert cross.set_index("indicator_id").loc[a, "method"] == "exact"


def test_truncated_text_maps_to_full_text_not_to_its_prefix_sibling(matched):
    _, _, variants, _ = matched
    full = _id(variants, "nfhs6", "Children under age 6 months exclusively breastfed or breastfeeding and "
                                  "consuming plain water/other milk only10 (%)")
    trunc = _id(variants, "nfhs6", "Children under age 6 months exclusively breastfed or breastfeeding and "
                                   "consuming plain water/other")
    excl = _id(variants, "nfhs6", "Children under age 6 months exclusively breastfed10 (%)")
    assert full == trunc
    assert excl != full          # 'exclusively breastfed' is a prefix of the longer text but a different indicator


def test_state_sheet_rows_match_by_text_not_number(matched, raw6):
    _, d6, variants, _ = matched
    # Chandigarh (state sheet) numbers institutional births 40, districts number it 35
    assert set(raw6.loc[raw6.indicator == "Institutional births (%)", "indicator_no"]) >= {"35", "40"}
    ids = variants[(variants.source == "nfhs6") & (variants.raw_text == "Institutional births (%)")]
    assert set(ids.indicator_id) == {"inst_births"}
    # indicator_no 18 is TFR on the state sheet; it must not be matched to anything in India.csv
    tfr = _id(variants, "nfhs6", "Total fertility rate (children per woman)")
    assert tfr.startswith("n6only_")


def test_women_and_men_not_merged(matched):
    _, _, variants, _ = matched
    w = _id(variants, "nfhs6", "Women whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)15 (%)")
    m = _id(variants, "nfhs6", "Men whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2) (%)")
    assert w == "women_thin" and m != w
    # India.csv's truncated '(BMI <18)' is matched to the women's indicator by prefix
    assert _id(variants, "nfhs45", "Women whose Body Mass Index (BMI) is below normal (BMI <18)") == w


def test_definition_change_is_flagged(matched):
    _, _, variants, cross = matched
    pid = _id(variants, "nfhs6", "Children age 12-23 months who have received 3 doses of pentavalent vaccine (%)")
    row = cross.set_index("indicator_id").loc[pid]
    assert row.method == "fuzzy"
    assert row.definition_exclusion and "penta" in row.definition_exclusion.lower()


def test_district_matching_methods(matched, cfg):
    l45, d6, _, _ = matched
    nm = match.match_districts(l45, d6, cfg).set_index("district_nfhs6")
    assert nm.loc["Gurugram", "method"] == "alias" and nm.loc["Gurugram", "district_nfhs45"] == "Gurgaon"
    assert nm.loc["Barabanki", "method"] == "exact_nospace"
    assert nm.loc["Hisar", "method"] == "exact"
    assert nm.loc["Saraikela-Kharsawan", "district_nfhs45"] == "Saraikela-kharsawan"
    assert nm.loc["Kasganj", "method"] == "unmatched"


def test_core_districts_and_audit(matched, cfg):
    l45, d6, variants, cross = matched
    nm = match.match_districts(l45, d6, cfg)
    pairs, summary, by_d, by_i = match.audit_nfhs5(l45, d6, variants, nm, cross)
    nm = match.finalise_district_status(nm, by_d, cfg).set_index("district_nfhs6", drop=False)
    assert summary["exact_agreement_rate"] == 1.0
    assert summary["nfhs6file_star_and_india_zero"] == 1          # Hisar exclusive breastfeeding
    assert nm.loc["Gurugram", "core"]
    assert not nm.loc["Charkhi Dadri", "core"]                    # no NFHS-4 values
    assert "NFHS-4" in nm.loc["Charkhi Dadri", "exclusion_reason"]
    assert not nm.loc["Kasganj", "core"]


def test_panel_long_columns_and_uniqueness(matched, cfg):
    l45, d6, variants, cross = matched
    nm = match.match_districts(l45, d6, cfg)
    panel = match.build_panel_long(l45, d6, variants, nm)
    for col in ["state", "district", "district_code", "round", "indicator_id", "indicator_text", "value",
                "small_sample_flag", "boundary_change_flag", "source"]:
        assert col in panel.columns
    assert set(panel["round"]) == {4, 5, 6}
    assert not panel.duplicated(["district_uid", "indicator_id", "round"]).any()
    g = panel[(panel.district == "Gurugram") & (panel.indicator_id == "inst_births")].set_index("round").value
    assert g.to_dict() == {4: 90.2, 5: 99.1, 6: 99.5}
