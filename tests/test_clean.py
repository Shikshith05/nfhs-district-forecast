import math

import numpy as np
import pandas as pd
import pytest

from nfhs_forecast import clean


# ---------------------------------------------------------------- values
@pytest.mark.parametrize("raw, expected", [
    ("45.2", 45.2), (" 7.5 ", 7.5), ("(45.2)", 45.2), ("[12]", 12.0), ("( 3.0 )", 3.0),
    ("1,060", 1060.0), ("12.5%", 12.5), (45, 45.0), (0.0, 0.0), ("0", 0.0),
])
def test_clean_value_numbers_and_brackets(raw, expected):
    assert clean.clean_value(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["*", " * ", "(*)", "12.3*", "", "na", "NA", "n/a", "-", "abc", None])
def test_clean_value_missing(raw):
    assert math.isnan(clean.clean_value(raw))


def test_clean_values_series_and_suppressed_flag():
    s = pd.Series(["10.0", "*", "(20.5)", ""])
    out = clean.clean_values(s)
    assert out.iloc[0] == 10.0 and np.isnan(out.iloc[1]) and out.iloc[2] == 20.5 and np.isnan(out.iloc[3])
    assert clean.is_suppressed(s).tolist() == [False, True, False, False]


# ----------------------------------------------------------------- names
@pytest.mark.parametrize("a, b", [
    ("Dadra & Nagar Haveli", "dadra and nagar haveli"),
    ("Jammu & Kashmir", "Jammu And Kashmir"),
    ("NCT of Delhi", "Nct Of Delhi"),
    ("  Chhattisgarh ", "Chhattisgarh"),
    ("Saraikela-Kharsawan", "Saraikela kharsawan"),
    ("Sant Ravidas Nagar (Bhadohi)", "sant ravidas nagar bhadohi"),
])
def test_normalise_name_equal(a, b):
    assert clean.normalise_name(a) == clean.normalise_name(b)


def test_squash_ignores_spacing():
    assert clean.squash("Bara Banki") == clean.squash("Barabanki")
    assert clean.squash("Medchal Malkajgiri") == clean.squash("MedchalMalkajgiri")
    assert clean.normalise_name("Bara Banki") != clean.normalise_name("Barabanki")


# ------------------------------------------------------- indicator text
@pytest.mark.parametrize("a, b", [
    ("Mothers whose last birth was protected against neonatal tetanus4 (%)",
     "Mothers whose last birth was protected against neonatal tetanus 4 (%)"),
    ("Mothers whose last birth was protected against neonatal tetanus4 (%)",
     "Mothers whose last birth was protected against neonatal tetanus (%)"),
    ("Population living in households with an improved drinking-water source1 (%)",
     "Population living in households with an improved drinkingwater source (%)"),
    ("Women age 20-24 years married before age 18 years (%)",
     "Women age 2024 years married before age 18 years (%)"),
    ("Total children age 6-23 months receiving an adequate diet10, 11 (%)",
     "Total children age 6-23 months receiving an adequate diet (%)"),
    ("Mothers who had an antenatal check-up in the first trimester (%)",
     "Mothers who had an antenatal checkup in the first trimester (%)"),
])
def test_normalise_indicator_text_variants_equal(a, b):
    assert clean.normalise_indicator_text(a) == clean.normalise_indicator_text(b)


def test_normalise_indicator_text_keeps_meaningful_numbers():
    t = clean.normalise_indicator_text("Women age 20-24 years married before age 18 years (%)")
    assert t.endswith("age 18 years")
    assert "2024" in t
    assert clean.normalise_indicator_text("Children under 5 years who are stunted (height-for-age)12 (%)") \
        == "children under 5 years who are stunted heightforage"


def test_normalise_indicator_text_keeps_distinct_indicators_distinct():
    a = clean.normalise_indicator_text("Blood sugar level - high (141-160 mg/dl)16 (%)")
    b = clean.normalise_indicator_text("Blood sugar level - very high (>160 mg/dl)16 (%)")
    assert a != b


def test_sex_qualifier_only_for_split_indicators(cfg):
    pat = cfg["matching"]["sex_split_pattern"]
    assert clean.indicator_key("Blood sugar level - high (141-160 mg/dl)16 (%)", "Men", pat)[0] == "men"
    assert clean.indicator_key("Female Blood sugar level  high (141-160 mg/dl) (%)",
                               "Blood Sugar Level among women", pat)[:2] == \
        clean.indicator_key("Blood sugar level - high (141-160 mg/dl)16 (%)", "Women", pat)[:2]
    # the NFHS-6 parser puts women's indicators under category 'Men'; no qualifier must be applied
    assert clean.indicator_key("Women age 15-24 years who use hygienic methods of protection during "
                               "their menstrual period17 (%)", "Men", pat)[0] == ""


# --------------------------------------------------- source cleaning
def test_clean_nfhs45_zero_rules_and_fixes(raw45, cfg):
    long, log = clean.clean_nfhs45(raw45, cfg)
    get = lambda d, ind, r: long[(long.District == d) & long.Indicator.str.startswith(ind) & (long["round"] == r)].iloc[0]
    # a new district whose NFHS-4 values are all 0 ('na') -> not available
    assert get("Dadri", "Institutional", 4).missing_reason == "district_round_not_available"
    assert np.isnan(get("Dadri", "Institutional", 4).value)
    assert get("Dadri", "Institutional", 5).value == 93.0
    # an indicator that is 0 everywhere in NFHS-4 -> not measured in that round
    assert get("Gurgaon", "Deaths", 4).missing_reason == "indicator_not_in_round"
    # isolated 0 in an indicator with no near-zero values -> suppressed
    assert get("Hisar", "Children under age 6 months", 5).missing_reason == "zero_treated_as_suppressed"
    # 0 in an indicator where small values are common -> true zero kept
    assert get("Hisar", "Male sterilization", 5).value == 0.0
    # mislabelled block split out under its DISTRICT key
    assert "Saraikela-kharsawan" in set(long.District)
    assert log["renamed_blocks"][0]["district_key"] == "Saraikela-kharsawan_24"
    # conflicting duplicate -> NaN, identical duplicate -> kept once
    assert get("Agra", "Institutional", 5).missing_reason == "conflicting_duplicate"
    assert get("Agra", "Institutional", 4).value == 70.0
    assert not long.duplicated(["State", "District", "Indicator", "round"]).any()


def test_clean_nfhs6_stars_brackets_exclusions(raw6, cfg):
    d6, log = clean.clean_nfhs6(raw6, cfg)
    assert "Dadra & Nagar Haveli And Daman & Diu" not in set(d6.state)
    hisar = d6[(d6.district == "Hisar")].set_index("indicator_no")
    assert np.isnan(hisar.loc["61", "value6"]) and hisar.loc["61", "suppressed6"]
    assert hisar.loc["63", "value6"] == 40.2          # '(40.2)' -> 40.2
    assert hisar.loc["63", "small_sample_flag"]
    assert d6.boundary_change_flag.dtype == bool
    assert log["suppressed_nfhs6"] == 1
