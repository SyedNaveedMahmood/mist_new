"""Joint shift bootstrap (primary H1 statistic, EXPERIMENT_PLAN section 7, D-024) on simulated people.

Each person has a lasting ability and a lasting response to the arm, per protocol; every seed re-tests the
same people (small run noise on top). The bootstrap resamples people, so the interval must cover the
population effect about 95% of the time despite the seeds being correlated within a person.
"""
import numpy as np
import pytest

from mist.aggregate import joint_shift_bootstrap, scores_batch

SC = [f"SC:{i:02d}" for i in range(54)]
ST = [f"ST:{i:02d}" for i in range(22)]
PROTOS = {"montage": SC, "sc2st": ST, "young2old": SC[36:], "old2young": SC[:18]}
STAGE_N = np.array([200, 60, 400, 120, 160])
DELTA = 0.04            # arm effect on the probability of a correct label
N_SEEDS = 3


def person_cm(c, rng):
    """Correct with prob c per epoch, errors spread uniformly over the other four stages."""
    cm = np.zeros((5, 5), np.int64)
    for j, n in enumerate(STAGE_N):
        k = rng.binomial(n, c[j])
        cm[j, j] = k
        cm[j] += np.bincount(np.delete(np.arange(5), j)[rng.integers(0, 4, n - k)], minlength=5) * (np.arange(5) != j)
    return cm


def simulate(rng):
    a, b = {}, {}
    for p, people in PROTOS.items():
        base = 0.62 + {"montage": -0.05, "sc2st": -0.08, "young2old": -0.03, "old2young": 0.0}[p]
        a[p], b[p] = {}, {}
        for s in people:
            ability = base + rng.normal(0, 0.08) + rng.normal(0, 0.05, 5)          # lasting, per person and stage
            resp = DELTA + rng.normal(0, 0.04)                                        # lasting response to the arm
            ca = cb = 0
            for _ in range(N_SEEDS):                                                  # same person, every seed
                run = rng.normal(0, 0.02, 5)
                ca = ca + person_cm(np.clip(ability + resp + run, 0.02, 0.98), rng)
                cb = cb + person_cm(np.clip(ability + run, 0.02, 0.98), rng)
            a[p][s], b[p][s] = ca, cb
    return a, b


def population_effect():
    """Pooled effect in the large-population limit: macro-F1 of the expected confusion matrices."""
    rng = np.random.default_rng(123)
    eff = []
    for p in PROTOS:
        base = 0.62 + {"montage": -0.05, "sc2st": -0.08, "young2old": -0.03, "old2young": 0.0}[p]
        n = 200_000
        ability = base + rng.normal(0, 0.08, (n, 1)) + rng.normal(0, 0.05, (n, 5))
        resp = DELTA + rng.normal(0, 0.04, (n, 1))
        run = rng.normal(0, 0.02, (n, 5))
        out = []
        for c in (np.clip(ability + resp + run, 0.02, 0.98), np.clip(ability + run, 0.02, 0.98)):
            m = c.mean(0) * STAGE_N
            cm = np.repeat(((STAGE_N - m) / 4)[:, None], 5, 1)
            cm[np.arange(5), np.arange(5)] = m
            out.append(scores_batch(cm)["mf1"])
        eff.append(out[0] - out[1])
    return float(np.mean(eff))


def test_recovers_known_effect_and_pools_equally():
    rng = np.random.default_rng(0)
    a, b = simulate(rng)
    r = joint_shift_bootstrap(a, b, [SC, ST], n_boot=2000)
    truth = population_effect()
    assert truth > 0.02
    assert abs(r["pooled"]["diff"] - truth) < 0.01
    assert r["pooled"]["lo"] < truth < r["pooled"]["hi"] and r["pooled"]["lo"] > 0 and r["pooled"]["p_boot"] < 0.01
    assert r["pooled"]["diff"] == pytest.approx(np.mean([v["diff"] for v in r["per_protocol"].values()]))
    assert r["per_protocol"]["sc2st"]["n_subjects"] == 22 and r["strata_sizes"] == [54, 22]


def test_coverage_near_95_with_people_retested_by_several_seeds():
    truth = population_effect()
    rng = np.random.default_rng(1)
    hits, reps = 0, 300
    for i in range(reps):
        a, b = simulate(rng)
        r = joint_shift_bootstrap(a, b, [SC, ST], n_boot=400, seed=i)
        hits += r["pooled"]["lo"] <= truth <= r["pooled"]["hi"]
    assert 0.92 <= hits / reps <= 0.98, hits / reps


def test_joint_resample_moves_a_person_together_and_checks_inputs():
    rng = np.random.default_rng(3)
    cm = lambda: np.eye(5, dtype=np.int64) * 50 + rng.integers(0, 10, (5, 5))
    a = {"p1": {s: cm() for s in SC[:10]}, "p2": {s: cm() for s in SC[:10]}}
    r = joint_shift_bootstrap(a, a, [SC[:10]], n_boot=50)
    assert r["pooled"] == pytest.approx({"diff": 0, "lo": 0, "hi": 0, "p_boot": 1.0, "n_boot_used": 50})
    with pytest.raises(ValueError):
        joint_shift_bootstrap(a, {"p1": a["p1"]}, [SC[:10]], n_boot=10)            # protocol missing in one arm
    with pytest.raises(ValueError):
        joint_shift_bootstrap(a, a, [SC[:5]], n_boot=10)                           # subject outside every stratum
    with pytest.raises(ValueError):
        joint_shift_bootstrap(a, a, [SC[:10], SC[5:12]], n_boot=10)                # overlapping strata


def test_joint_contrast_matches_paired_difference_and_cancels():
    from mist.aggregate import joint_contrast, paired_diff
    rng = np.random.default_rng(7)
    a, b = simulate(rng)
    one = joint_contrast({"x": a["sc2st"], "y": b["sc2st"]}, [(1, "x"), (-1, "y")], [SC, ST], n_boot=2000)
    ref = paired_diff(a["sc2st"], b["sc2st"], n_boot=2000)
    assert one["mf1"]["diff"] == pytest.approx(ref["mf1"]["diff"])
    assert abs(one["mf1"]["lo"] - ref["mf1"]["lo"]) < 0.01 and abs(one["mf1"]["hi"] - ref["mf1"]["hi"]) < 0.01
    zero = joint_contrast({"x": a["montage"]}, [(1, "x"), (-1, "x")], [SC, ST], n_boot=200)
    assert zero["mf1"] == pytest.approx({"diff": 0, "lo": 0, "hi": 0})
