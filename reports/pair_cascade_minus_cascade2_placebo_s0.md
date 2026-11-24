## cascade - cascade2_placebo (EXPLORATORY; seeds [0])

Paired subject bootstrap, 10,000 resamples, 95% percentile; only the (fold, seed) runs both arms have. Points (x100). Within-database shifts; ST development-grade.

| protocol | runs per arm | people | macro-F1 | F1 W | F1 N1 | F1 N2 | F1 N3 | F1 REM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| in-domain (cv, Fpz-Cz, std) | 5 | 54 | +0.2 [-0.3, +0.9] | +0.8 [-0.1, +1.8] | +0.5 [-0.9, +2.1] | -0.2 [-0.7, +0.4] | -0.0 [-1.2, +1.2] | -0.0 [-0.9, +0.8] |
| montage: train Fpz-Cz, test Pz-Oz (oracle: Pz-Oz cv) | 5 | 54 | -0.1 [-0.8, +0.6] | -0.4 [-2.2, +1.1] | +0.3 [-1.6, +2.2] | -0.6 [-1.4, +0.2] | +0.6 [-0.7, +1.8] | -0.4 [-1.9, +1.0] |
| cohort shift sc2st, all ST nights (no oracle) | 5 | 22 | -0.6 [-1.2, +0.0] | -1.3 [-2.5, -0.1] | -1.2 [-2.7, +0.1] | -0.2 [-0.7, +0.3] | +0.2 [-0.6, +1.0] | -0.5 [-1.0, +0.1] |
