# Gate C analyses

DEVELOPMENT: calibrated on a random half of states, NFHS-4 -> NFHS-5 LOSO; evaluated on the other half

Primary forecaster (rule D34): `elastic_net`; alpha = 0.1; BH q = 0.1.

## C1 Conformal coverage (OVERALL)

| variant | method | n | coverage | nominal | mean_width |
|---|---|---|---|---|---|
| normalised | elastic_net | 7278 | 0.876 | 0.900 | 23.810 |
| normalised | fh_rtm | 7278 | 0.882 | 0.900 | 24.411 |
| normalised | lightgbm | 7278 | 0.877 | 0.900 | 24.157 |
| normalised | rtm | 7278 | 0.885 | 0.900 | 24.989 |
| split | elastic_net | 7278 | 0.877 | 0.900 | 24.186 |
| split | fh_rtm | 7278 | 0.882 | 0.900 | 24.937 |
| split | lightgbm | 7278 | 0.881 | 0.900 | 24.661 |
| split | rtm | 7278 | 0.886 | 0.900 | 25.477 |
| weighted | elastic_net | 7278 | 0.885 | 0.900 | 25.409 |
| weighted | fh_rtm | 7278 | 0.887 | 0.900 | 25.973 |
| weighted | lightgbm | 7278 | 0.893 | 0.900 | 26.581 |
| weighted | rtm | 7278 | 0.892 | 0.900 | 26.497 |

## C2 Surprises

| method | q | tests | surprises_normal | surprises_conformal | naive_k | overlap_naive_vs_bh | naive_topk_mean_se_true | bh_mean_se_true | all_mean_se_true | calibration_z_sd | target_z_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elastic_net | 0.100 | 7275 | 21 | 0 | 21 | 5 | 4.862 | 3.372 | 3.301 | 1.001 | 1.059 |
| rtm | 0.100 | 7275 | 17 | 0 | 17 | 5 | 4.392 | 3.229 | 3.301 | 0.998 | 1.039 |

## C3 Forecastability

Decomposition of log normalised squared error (7275 pairs, 317 districts, 23 indicators): indicator-only R² 0.065, district-only R² 0.071, additive R² 0.137 (unique: indicator 0.065, district 0.071).

Districts mapped: 317 Census-2011 polygons.

Hardest indicators:

| indicator_id | n | mean_noise_share | mean_norm_sq_err | ceiling | skill |
|---|---|---|---|---|---|
| stunting | 317 | 0.224 | 0.553 | 0.776 | 0.447 |
| ifa_100d | 317 | 0.041 | 0.588 | 0.959 | 0.412 |
| women_overweight | 317 | 0.075 | 0.596 | 0.925 | 0.404 |
| teen_pregnancy | 317 | 0.346 | 0.660 | 0.654 | 0.340 |
| early_bf | 317 | 0.063 | 0.860 | 0.937 | 0.140 |

