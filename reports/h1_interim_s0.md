# H1 (INTERIM, seeds [0]): cascade2 - cascade2_placebo

Within-database shifts (Sleep-EDF); ST is development-grade. Points (x100). per protocol: paired subject bootstrap, subject confusion matrices summed over folds and seeds; pooled: equal-weight mean of the shift-protocol macro-F1 differences, one joint bootstrap (10000 resamples) over people, strata SC development (54) and ST (22), 95% percentile interval (EXPERIMENT_PLAN section 7, D-024).

**INTERIM.** Seeds [0]; pooled over 2 of 4 shift protocols (montage, sc2st). Not the confirmatory analysis; changes no plan (owner B6).

## Pooled shift effect (primary statistic)

- macro-F1 -1.7 [-2.4, -1.2], bootstrap p 0.0000; no decision (design incomplete)

## Per protocol (paired subject bootstrap)

| protocol | runs per arm | n subjects | macro-F1 | F1 N1 | summed non-N1 F1 | F1 W | F1 N2 | F1 N3 | F1 REM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| montage | 5 | 54 | -1.9 [-3.1, -0.9] | -1.1 [-2.4, +0.3] | -8.6 [-13.6, -4.0] | -5.3 [-8.4, -2.8] | -1.0 [-2.0, -0.0] | -0.6 [-2.2, +1.1] | -1.7 [-3.3, -0.1] |
| sc2st | 5 | 22 | -1.6 [-2.1, -1.0] | -1.8 [-3.0, -0.6] | -6.1 [-8.1, -4.1] | -3.6 [-4.9, -2.3] | -1.0 [-1.5, -0.5] | -0.9 [-1.7, -0.2] | -0.6 [-1.4, +0.4] |
| in-domain (beside, not pooled) | 5 | 54 | -0.1 [-0.8, +0.5] | +1.5 [+0.1, +3.0] | -2.2 [-4.5, +0.2] | +0.4 [-0.5, +1.4] | -0.9 [-1.5, -0.3] | -1.0 [-2.6, +0.4] | -0.7 [-1.7, +0.3] |

## Secondary: mixed model

- -1.72 points, 95% interval [-4.19, +0.76], 20 run-level rows. variability over seeds and folds on the same people (secondary; no claim rests on it). Run-level macro-F1 ~ arm + (1|protocol) + (1|fold) + (1|seed), REML, shift protocols only; normal 95% interval, approximate.
