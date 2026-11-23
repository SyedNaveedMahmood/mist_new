"""Session queue rules (EXPERIMENT_PLAN section 12): 15-minute margin, 1.3 x fit rule, pairs start together,
roll-over keeps the order. Dry runs only: nothing is trained."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dry(tmp_path, plan, times, start="2030-01-01 09:00", end="2030-01-01 13:00", fail=()):
    (tmp_path / "plan.txt").write_text(plan)
    (tmp_path / "times.json").write_text(json.dumps(times))
    extra = [a for f in fail for a in ("--simulate-fail", f)]
    out = subprocess.run([sys.executable, "scripts/session_queue.py", "--end", end, "--now", start, "--dry-run",
                          "--plan", str(tmp_path / "plan.txt"), "--times", str(tmp_path / "times.json"), *extra],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out


T = {"train": {"zzarm|cv": [3000]}, "eval_core": {"zzarm": [0]}}     # 50 min per run, evals free


def test_margin_and_fit_rule(tmp_path):
    # deadline 12:45 -> 225 min; each run needs 1.3 x 50 = 65 min: after 3 runs 75 min are left (4th fits), then 25
    plan = "".join(f"train zz_q{i}_cv0_s0 zzarm cv\n" for i in range(5))
    out = dry(tmp_path, plan, T)
    assert "deadline 2030-01-01 12:45" in out
    assert out.count(": ok (50.0 min)") == 4 and "does not fit: zz_q4_cv0_s0" in out


def test_pair_starts_only_if_both_fit_and_order_is_kept(tmp_path):
    plan = ("train zz_a_cv0_s0 zzarm cv\n&train zz_b_cv0_s0 zzarm cv\n"
            "train zz_a_cv1_s0 zzarm cv\n&train zz_b_cv1_s0 zzarm cv\n"
            "train zz_small_cv0_s0 zzarm cv\n")
    out = dry(tmp_path, plan, T)
    # first pair (130 min with factor) fits; after it 125 min are left < 1.3 x 100: stop, no skipping ahead
    assert "train zz_b_cv0_s0: ok" in out and "does not fit: zz_a_cv1_s0 + zz_b_cv1_s0" in out
    assert "zz_small" not in out


def test_unknown_arm_uses_a_conservative_prior(tmp_path):
    out = dry(tmp_path, "train zz_c_cv0_s0 cascade2 cv\n", {}, end="2030-01-01 12:00")
    assert "does not fit: zz_c_cv0_s0" in out            # prior = full-rate cascade time, > 2.75 h / 1.3


PAIRS = ("train zz_a_cv0_s0 zzarm cv\n&train zz_b_cv0_s0 zzarm cv\n"
         "train zz_a_cv1_s0 zzarm cv\n&train zz_b_cv1_s0 zzarm cv\n")
LONG = "2030-01-01 21:00"


def test_failed_run_is_retried_once_then_ok(tmp_path):
    out = dry(tmp_path, PAIRS, T, end=LONG, fail=["zz_b_cv0_s0:1"])
    assert "zz_b_cv0_s0: failed (rc 1); one retry" in out and "train zz_b_cv0_s0: ok" in out
    assert "skipped" not in out and "train zz_b_cv1_s0: ok" in out


def test_second_failure_skips_the_fold_for_both_arms_and_continues(tmp_path):
    out = dry(tmp_path, PAIRS, T, end=LONG, fail=["zz_a_cv0_s0:2"])
    assert "train zz_a_cv0_s0: FAILED twice (other)" in out
    assert "fold skipped for every arm of the unit (owner rule B3): zz_a_cv0_s0, zz_b_cv0_s0" in out
    assert "train zz_b_cv0_s0" not in out                      # the partner is not trained
    assert "train zz_a_cv1_s0: ok" in out and "train zz_b_cv1_s0: ok" in out


def test_out_of_memory_retries_with_checkpointing(tmp_path):
    out = dry(tmp_path, PAIRS, T, end=LONG, fail=["zz_a_cv0_s0:1:oom"])
    assert "retrying once with encoder checkpointing" in out and "train zz_a_cv0_s0: ok" in out


def test_persistent_gpu_fault_waits_retries_once_then_stops_gpu_work(tmp_path):
    out = dry(tmp_path, PAIRS, T, end=LONG, fail=["zz_a_cv0_s0:2:gpu"])
    assert "GPU or driver fault; waiting 5 minutes, then one retry" in out
    assert "GPU work stops for the night" in out and "zz_a_cv1_s0" not in out.split("GPU work stops")[1]
    assert "no pending evaluations" in out
