# Data dictionary: harmonised NFHS district panel

Rounds 4 and 5 only. NFHS-6 values are added after the one-shot evaluation (Gate B), so that no NFHS-6 outcome is used or distributed before the test.

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
| `se_sampling` | assumed sampling SE = 100*sqrt(p(1-p)/n*deff), deff = 2.0 |
| `extra_var` | added variance for a value standing in for a changed boundary (D29); 0 otherwise |
| `n_assumed` | assumed per-district sample size (see below) |
| `se_group` | population group that sets `n_assumed` |
| `value_method` | `observed`; `observed_changed_boundary` (NFHS-6 namesake of a changed district); `inherited`; `aggregated` |
| `harmonised` | True if `value_method` is not `observed` |
| `parent_uid` | NFHS-5 parent unit (bootstrap cluster for harmonised units) |
| `small_sample_flag` | NFHS-6 fact-sheet marker for 25-49 unweighted cases (round 6 only) |
| `missing_reason` | why a value is missing (suppressed, not measured, district not available, ...) |

Assumed sample sizes (`config.yaml` noise.sample_size; small-sample rows use n = 37):

| se_group | assumed n |
|---|---|
| `households` | 900 |
| `women` | 1000 |
| `women_subgroup` | 200 |
| `mothers` | 250 |
| `men` | 140 |
| `children` | 300 |
| `children_subgroup` | 70 |

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
