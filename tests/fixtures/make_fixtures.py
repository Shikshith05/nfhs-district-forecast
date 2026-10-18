"""Regenerate the small synthetic fixtures in this folder (deterministic).

    .venv\\Scripts\\python tests\\fixtures\\make_fixtures.py

The CSVs mimic the layout and the known quirks of the real sources:
India.csv (NFHS-4/5) and all_india_nfhs6_district.csv (NFHS-6). Values are
made up.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

IND45 = [  # (category, text)
    ("Delivery Care", "Institutional births (%)"),
    ("Maternal and Child Health", "Mothers whose last birth was protected against neonatal tetanus (%)"),
    ("Child Vaccinations and Vitamin A Supplementation",
     "Children age 12-23 months who have received 3 doses of penta or DPT vaccine (%)"),
    ("Nutritional Status of Women", "Women whose Body Mass Index (BMI) is below normal (BMI <18)"),
    ("Child Feeding Practices and Nutritional Status of Children",
     "Children under age 6 months exclusively breastfed (%)"),
    ("Current Use of Family Planning Methods", "Male sterilization (%)"),
    ("Population and Household Profile", "Deaths in the last 3 years registered with the civil authority (%)"),
    ("Marriage and Fertility", "Women age 2024 years married before age 18 years (%)"),
]

# (state, ST_CEN_CD, District, DISTRICT key, DT_CEN_CD, nfhs5 values, nfhs4 values)
D45 = [
    ("Haryana", "6", "Gurgaon", "Gurgaon_1", "1",
     [99.1, 95.0, 88.0, 12.0, 60.0, 0.4, 90.0, 4.0], [90.2, 93.0, 70.0, 15.0, 40.0, 0.3, 0, 6.0]),
    ("Haryana", "6", "Hisar", "Hisar_2", "2",
     [95.5, 92.0, 85.0, 14.0, 0, 0.0, 85.0, 5.0], [80.0, 91.0, 66.0, 18.0, 35.0, 0.2, 0, 8.0]),
    ("Haryana", "6", "Dadri", "", "",  # new district: NFHS-4 'na' written as 0
     [93.0, 90.0, 80.0, 16.0, 55.0, 0.6, 80.0, 6.0], [0, 0, 0, 0, 0, 0, 0, 0]),
    ("Uttar Pradesh", "9", "Bara Banki", "Bara Banki_3", "3",
     [80.0, 88.0, 70.0, 22.0, 50.0, 0.1, 50.0, 20.0], [60.0, 85.0, 50.0, 28.0, 45.0, 0.5, 0, 30.0]),
    ("Uttar Pradesh", "9", "Agra", "Agra_4", "4",
     [85.0, 90.0, 72.0, 20.0, 52.0, 0.2, 60.0, 15.0], [70.0, 86.0, 55.0, 25.0, 47.0, 0.1, 0, 22.0]),
    ("Jharkhand", "20", "Sahibganj", "Sahibganj_7", "7",
     [70.0, 85.0, 65.0, 30.0, 45.0, 0.3, 40.0, 35.0], [50.0, 80.0, 45.0, 35.0, 40.0, 0.2, 0, 45.0]),
    ("Jharkhand", "20", "Sahibganj", "Saraikela-kharsawan_24", "7",  # mislabelled block
     [75.0, 86.0, 68.0, 28.0, 48.0, 0.5, 45.0, 30.0], [55.0, 82.0, 48.0, 33.0, 42.0, 0.4, 0, 40.0]),
    ("Chandigarh", "4", "Chandigarh", "Chandigarh_1", "1",
     [97.0, 94.0, 90.0, 11.0, 58.0, 0.2, 95.0, 3.0], [93.0, 92.0, 80.0, 13.0, 50.0, 0.1, 0, 4.0]),
]

N6_DISTRICT_IND = [  # (no, category, text) in the district-sheet layout
    (31, "Antenatal Care (for last birth in the 5 years before the survey)",
     "Mothers whose last birth was protected against neonatal tetanus4 (%)"),
    (35, "Delivery Care (for births in the 5 years before the survey)", "Institutional births (%)"),
    (49, "Child Vaccinations and Vitamin A Supplementation",
     "Children age 12-23 months who have received 3 doses of pentavalent vaccine (%)"),
    (61, "Child Feeding Practices and Nutritional Status of Children",
     "Children under age 6 months exclusively breastfed10 (%)"),
    (63, "Child Feeding Practices and Nutritional Status of Children",
     "Children under age 6 months exclusively breastfed or breastfeeding and consuming plain water/other milk only10 (%)"),
    (73, "Nutritional Status of Adults (age 15-49 years)",
     "Women whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)15 (%)"),
    (24, "Marriage and Fertility", "Male sterilization (%)"),
    (16, "Marriage and Fertility", "Women age 20-24 years married before age 18 years (%)"),
]
# Chandigarh comes from the STATE sheet: different numbering, extra rows.
N6_STATE_IND = [
    (18, "Marriage and Fertility", "Total fertility rate (children per woman)"),
    (33, "Antenatal Care (for last birth in the 5 years before the survey)",
     "Mothers whose last birth was protected against neonatal tetanus 4 (%)"),
    (40, "Delivery Care (for births in the 5 years before the survey)", "Institutional births (%)"),
    (51, "Child Vaccinations and Vitamin A Supplementation",
     "Children age 12-23 months who have received 3 doses of pentavalent vaccine (%)"),
    (63, "Child Feeding Practices and Nutritional Status of Children",
     "Children under age 6 months exclusively breastfed10 (%)"),
    (65, "Child Feeding Practices and Nutritional Status of Children",
     "Children under age 6 months exclusively breastfed or breastfeeding and consuming plain water/other"),
    (75, "Nutritional Status of Adults (age 15-49 years)",
     "Women whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)15 (%)"),
    (76, "Nutritional Status of Adults (age 15-49 years)",
     "Men whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2) (%)"),
    (26, "Marriage and Fertility", "Male sterilization (%)"),
    (17, "Marriage and Fertility", "Women age 20-24 years married before age 18 years (%)"),
]

# (state, district, nfhs6 values, nfhs5 values, small_sample, boundary_change)
D6 = [
    ("Haryana", "Gurugram", ["97.0", "99.5", "90.0", "65.0", "70.0", "11.0", "0.5", "3.0"],
     ["95.0", "99.1", "88.0", "60.0", "64.0", "12.0", "0.4", "4.0"], False, False),
    ("Haryana", "Hisar", ["93.0", "97.0", "86.0", "*", "(40.2)", "13.0", "0.1", "4.0"],
     ["92.0", "95.5", "85.0", "*", "(38.0)", "14.0", "0.0", "5.0"], True, False),
    ("Haryana", "Charkhi Dadri", ["91.0", "95.0", "82.0", "57.0", "61.0", "15.0", "0.7", "5.0"],
     ["90.0", "93.0", "80.0", "55.0", "59.0", "16.0", "0.6", "6.0"], False, False),
    ("Uttar Pradesh", "Barabanki", ["90.0", "88.0", "75.0", "55.0", "60.0", "20.0", "0.2", "17.0"],
     ["88.0", "80.0", "70.0", "50.0", "54.0", "22.0", "0.1", "20.0"], False, False),
    ("Uttar Pradesh", "Agra", ["91.0", "90.0", "76.0", "56.0", "61.0", "19.0", "0.3", "13.0"],
     ["90.0", "85.0", "72.0", "52.0", "56.0", "20.0", "0.2", "15.0"], False, False),
    ("Uttar Pradesh", "Kasganj", ["89.0", "86.0", "74.0", "54.0", "58.0", "21.0", "0.2", "18.0"],
     ["", "", "", "", "", "", "", ""], False, True),
    ("Jharkhand", "Sahibganj", ["88.0", "78.0", "70.0", "50.0", "55.0", "27.0", "0.4", "30.0"],
     ["85.0", "70.0", "65.0", "45.0", "50.0", "30.0", "0.3", "35.0"], False, False),
    ("Jharkhand", "Saraikela-Kharsawan", ["89.0", "82.0", "72.0", "52.0", "57.0", "25.0", "0.6", "26.0"],
     ["86.0", "75.0", "68.0", "48.0", "52.0", "28.0", "0.5", "30.0"], False, False),
]


def write_nfhs45(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["State", "ST_CEN_CD", "District", "DISTRICT", "DT_CEN_CD", "Category", "Indicator",
                    "NFHS 5", "NFHS 4"])
        for st, sc, d, key, dc, v5, v4 in D45:
            for (cat, txt), a, b in zip(IND45, v5, v4):
                w.writerow([st, sc, d, key, dc, cat, txt, a, b])
        # Conflicting duplicate (as for Surajpur in the real file)
        w.writerow(["Uttar Pradesh", "9", "Agra", "Agra_4", "4", IND45[0][0], IND45[0][1], 99.9, 70.0])


def write_nfhs6(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "district", "category", "indicator_no", "indicator", "nfhs6_value",
                    "nfhs5_value", "small_sample_flag", "boundary_change_single_round"])
        for st, d, v6, v5, small, bc in D6:
            for (no, cat, txt), a, b in zip(N6_DISTRICT_IND, v6, v5):
                if st == "Uttar Pradesh" and no == 31:
                    txt = "Mothers whose last birth was protected against neonatal tetanus 4 (%)"
                if st == "Jharkhand" and no == 63:
                    txt = ("Children under age 6 months exclusively breastfed or breastfeeding and "
                           "consuming plain water/other")
                w.writerow([st, d, cat, no, txt, a, b, small or a.startswith(("*", "(")), bc])
        cvals = ["1.6", "94.0", "98.0", "91.0", "60.0", "62.0", "10.0", "12.5", "0.3", "2.0"]
        c5 = ["1.4", "94.0", "97.0", "90.0", "58.0", "59.0", "11.0", "13.0", "0.2", "3.0"]
        for (no, cat, txt), a, b in zip(N6_STATE_IND, cvals, c5):
            w.writerow(["Chandigarh", "Chandigarh", cat, no, txt, a, b, False, False])
        # Merged UT block that must be excluded
        for (no, cat, txt) in N6_DISTRICT_IND:
            w.writerow(["Dadra & Nagar Haveli And Daman & Diu", "Dadra & Nagar Haveli And Daman & Diu",
                        cat, no, txt, "50.0", "45.0", False, False])


if __name__ == "__main__":
    write_nfhs45(HERE / "nfhs45_small.csv")
    write_nfhs6(HERE / "nfhs6_small.csv")
    print("wrote", HERE / "nfhs45_small.csv", HERE / "nfhs6_small.csv")
