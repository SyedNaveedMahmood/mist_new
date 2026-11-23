#!/usr/bin/env python
"""GPU session queue (EXPERIMENT_PLAN section 12). Walks a priority list in order inside a fixed session.

    python scripts/session_queue.py --end "2026-11-24 21:00" --plan experiments/priority.txt [--session 1]
    python scripts/session_queue.py --end ... --plan ... --dry-run [--now "2026-11-24 09:00"]

Rules (owner, 2026-11-23):
  * every GPU process ends 15 minutes before --end (hard deadline = end - 15 min; each process also runs
    under a timeout that expires at the deadline, and a training killed by it moves to runs/_interrupted);
  * a training starts only if the time to the deadline is >= 1.3 x its estimated duration (training plus
    its evaluations; measured times from runs/session_times.json, else the prior below);
  * the walk stops at the first unit that does not fit (it rolls over to the next session, keeping the order
    and the fold interleaving of pairs), then pending evaluations run while they fit, then the queue stops.

Plan file, one item per line ('#' comments):
  train <name> <arm> <evals> [key=value ...]   evals: cv (in-domain, Pz-Oz, sc2st; std, real cell only),
                                               age (the protocol's own test band), none
  &train ...                                   joined to the previous unit: the unit starts only if all fit
  eval <run> <evaluate.py args ...>            one evaluation (skipped if its output exists)
  pending <run> <evaluate.py args ...>         evaluation run only in the pending phase
  check <label> <command ...>                  CPU command; exit 0 continue, 1 jump to <label>, other: stop
  label <name> / stop
A unit whose runs are all trained and evaluated is skipped, so the same plan serves every session.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PY = ".venv/bin/python"
TIMES = Path("runs/session_times.json")
MARGIN_S = 15 * 60
FACTOR = 1.3
PRIOR = {  # seconds, conservative until measured in a session (train, one evaluation)
    "cascade2": (8782 + 600, 600),       # full-rate cascade measured 2026-11-23; replaced by the timing run
    "cascade": (8782, 900),
    "grm": (1218, 300), "gr": (1218, 300), "raw": (1575, 300), "attnsleep": (300, 120),
}
CV_EVALS = [([], "eval_test_cv_{ch}.json"), (["--channel", "Pz-Oz"], "eval_test_cv_Pz-Oz.json"),
            (["--protocol", "sc2st"], "eval_test_sc2st_{ch}.json")]


# --------------------------------------------------------------------------- plan
def parse_plan(path):
    units, labels = [], {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = shlex.split(line)
        kind = tok[0]
        if kind == "label":
            labels[tok[1]] = len(units)
        elif kind == "stop":
            units.append({"kind": "stop"})
        elif kind == "check":
            units.append({"kind": "check", "label": tok[1], "cmd": tok[2:]})
        elif kind in ("eval", "pending"):
            units.append({"kind": kind, "items": [eval_item(tok[1], tok[2:])]})
        elif kind in ("train", "&train"):
            item = train_item(tok[1], tok[2], tok[3], tok[4:])
            if kind == "&train":
                if not units or units[-1]["kind"] != "train":
                    raise ValueError(f"'&train' must follow a train line: {raw}")
                units[-1]["items"].append(item)
            else:
                units.append({"kind": "train", "items": [item]})
        else:
            raise ValueError(f"unknown plan line: {raw}")
    return units, labels


def train_item(name, arm, evals, overrides):
    ch = next((o.split("=", 1)[1] for o in overrides if o.startswith("data.channel=")), "Fpz-Cz")
    proto = next((o.split("=", 1)[1] for o in overrides if o.startswith("split.protocol=")), "cv")
    run = f"runs/{name}"
    if evals == "cv":
        ev = [eval_item(run, args + ["--cells", "core"], out.format(ch=ch), arm) for args, out in CV_EVALS]
    elif evals == "age":
        ev = [eval_item(run, ["--cells", "core"], f"eval_test_{proto}_{ch}.json", arm)]
    elif evals == "none":
        ev = []
    else:
        raise ValueError(f"evals must be cv, age or none: {evals}")
    return {"name": name, "arm": arm, "overrides": overrides, "evals": ev, "block": "age" if evals == "age" else "cv"}


def eval_item(run, args, out=None, arm=None):
    if out is None:                                            # derive the file name as evaluate.py does
        get = lambda f, d: args[args.index(f) + 1] if f in args else d
        run_cfg = Path(run) / "config.yaml"
        ch0, p0 = "Fpz-Cz", "cv"
        if run_cfg.exists():
            import yaml
            c = yaml.safe_load(run_cfg.read_text())
            ch0, p0 = c["data"]["channel"], c["split"]["protocol"]
        tag = get("--tag", "e3" if get("--cells", "all") == "e3" else "test")
        view = get("--view", "std")
        out = f"eval_{tag}_{get('--protocol', p0)}_{get('--channel', ch0)}" + ("" if view == "std" else f"_{view}") + ".json"
    return {"run": run, "args": args, "out": out, "arm": arm}


# ------------------------------------------------------------------- estimates
def load_times():
    return json.loads(TIMES.read_text()) if TIMES.exists() else {}


def arm_key(arm):
    for k in sorted(PRIOR, key=len, reverse=True):
        if arm.startswith(k):
            return k
    return arm


def est_train(item, times):
    key = f"{arm_key(item['arm'])}|{item['block']}"
    got = times.get("train", {}).get(key) or times.get("train", {}).get(f"{arm_key(item['arm'])}|cv")
    t = max(got) if got else PRIOR.get(arm_key(item["arm"]), (8782, 900))[0]
    return t + sum(est_eval(e, times) for e in item["evals"] if not done_eval(e))


def est_eval(e, times):
    arm = e.get("arm") or run_arm(e["run"])
    got = times.get("eval", {}).get(arm_key(arm))
    if "--cells" in e["args"] and e["args"][e["args"].index("--cells") + 1] == "core":
        got = times.get("eval_core", {}).get(arm_key(arm)) or got
    return max(got) if got else PRIOR.get(arm_key(arm), (0, 900))[1]


def run_arm(run):
    p = Path(run) / "config.yaml"
    if p.exists():
        import yaml
        return yaml.safe_load(p.read_text()).get("arm", Path(run).name)
    m = re.match(r"(?:e\d\w*?_)?(.+?)_(?:cv\d|young2old|old2young)", Path(run).name)
    return m.group(1) if m else Path(run).name


def record(kind, key, sec):
    t = load_times()
    t.setdefault(kind, {}).setdefault(key, []).append(round(sec, 1))
    TIMES.parent.mkdir(parents=True, exist_ok=True)
    TIMES.write_text(json.dumps(t, indent=1))


SIM = None          # dry run: set of simulated outputs (nothing is written to disk)


def trained(item):
    p = Path("runs") / item["name"] / "train_summary.json"
    return p.exists() or (SIM is not None and str(p) in SIM)


def done_eval(e):
    p = Path(e["run"]) / e["out"]
    return p.exists() or (SIM is not None and str(p) in SIM)


def mark(path):
    if SIM is not None:
        SIM.add(str(path))


def unit_done(u):
    return all(trained(i) and all(done_eval(e) for e in i["evals"]) for i in u["items"])


# ---------------------------------------------------------------------- clock
class Clock:
    def __init__(self, end, now=None, dry=False):
        self.deadline = end.timestamp() - MARGIN_S
        self.dry, self.t = dry, (now or datetime.now()).timestamp()

    def now(self):
        return self.t if self.dry else time.time()

    def left(self):
        return self.deadline - self.now()

    def advance(self, s):
        self.t += s


def log(msg, clock):
    stamp = datetime.fromtimestamp(clock.now()).strftime("%Y-%m-%d %H:%M")
    print(f"[{stamp}] {msg}", flush=True)


def run_cmd(cmd, logfile, clock, est):
    """Runs cmd until it ends or the deadline passes. Returns (rc, seconds); rc None = killed at deadline."""
    if clock.dry:
        clock.advance(est)
        return 0, est
    t0 = time.time()
    with open(logfile, "a") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=max(clock.left(), 1))
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            rc = None
    return rc, time.time() - t0


# ---------------------------------------------------------------------- steps
def do_eval(e, clock, times, quiet_log):
    if done_eval(e):
        return True
    est = est_eval(e, times)
    if clock.left() < FACTOR * est:
        log(f"no time for eval {e['run']} {e['out']} (est {est / 60:.0f} min)", clock)
        return False
    cmd = [PY, "scripts/evaluate.py", "--run", e["run"], *e["args"]]
    rc, sec = run_cmd(cmd, quiet_log, clock, est)
    if rc == 0:
        mark(Path(e["run"]) / e["out"])
    if rc == 0 and not clock.dry:
        core = "--cells" in e["args"] and e["args"][e["args"].index("--cells") + 1] == "core"
        record("eval_core" if core else "eval", arm_key(e.get("arm") or run_arm(e["run"])), sec)
    log(f"eval {e['run']} -> {e['out']}: {'ok' if rc == 0 else 'FAILED' if rc else 'KILLED at deadline'} "
        f"({sec / 60:.1f} min)", clock)
    return rc == 0


def do_train(item, clock, times):
    name, run = item["name"], Path("runs") / item["name"]
    if not trained(item):
        if run.exists():                                       # leftover of an interrupted run: keep, never reuse
            dest = Path("runs/_interrupted") / f"{name}_{datetime.now():%Y%m%d%H%M%S}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not clock.dry:
                shutil.move(str(run), dest)
            log(f"moved partial {run} -> {dest}", clock)
        cmd = [PY, "scripts/train.py", "--arm", item["arm"], "--name", name, *item["overrides"]]
        est = est_train(item, times) - sum(est_eval(e, times) for e in item["evals"])
        rc, sec = run_cmd(cmd, f"runs/logs/{name}.log", clock, est)
        if rc not in (0, None) and not clock.dry and oom(name) and "model.encoder_checkpoint=true" not in cmd:
            log(f"{name}: out of GPU memory; retrying once with encoder checkpointing (same arithmetic, D-013)", clock)
            dest = Path("runs/_interrupted") / f"{name}_oom_{datetime.now():%Y%m%d%H%M%S}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if run.exists():
                shutil.move(str(run), dest)
            rc, sec2 = run_cmd(cmd + ["model.encoder_checkpoint=true"], f"runs/logs/{name}.log", clock, est)
            sec += sec2
        if rc is None:
            dest = Path("runs/_interrupted") / f"{name}_{datetime.now():%Y%m%d%H%M%S}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if run.exists():
                shutil.move(str(run), dest)
            log(f"train {name}: KILLED at the deadline, moved to {dest}", clock)
            return False
        if rc != 0:
            log(f"train {name}: FAILED (rc {rc}), see runs/logs/{name}.log", clock)
            return False
        mark(run / "train_summary.json")
        if not clock.dry:
            record("train", f"{arm_key(item['arm'])}|{item['block']}", sec)
        log(f"train {name}: ok ({sec / 60:.1f} min)", clock)
    for e in item["evals"]:
        if not do_eval(e, clock, load_times(), "runs/logs/session_eval.log"):
            return False
    return True


def oom(name):
    p = Path(f"runs/logs/{name}.log")
    return p.exists() and "out of memory" in p.read_text()[-20000:].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True, help='session end, "YYYY-MM-DD HH:MM" local time')
    ap.add_argument("--plan", required=True)
    ap.add_argument("--dry-run", action="store_true", help="simulate with estimated durations; runs nothing")
    ap.add_argument("--now", default=None, help="simulated start time for --dry-run")
    ap.add_argument("--times", default=None, help="measured-times file (default runs/session_times.json); "
                                                   "a hypothetical file is useful with --dry-run")
    a = ap.parse_args()
    global TIMES, SIM
    if a.dry_run:
        SIM = set()
    if a.times:
        TIMES = Path(a.times)
    end = datetime.strptime(a.end, "%Y-%m-%d %H:%M")
    clock = Clock(end, datetime.strptime(a.now, "%Y-%m-%d %H:%M") if a.now else None, a.dry_run)
    units, labels = parse_plan(a.plan)
    Path("runs/logs").mkdir(parents=True, exist_ok=True)
    log(f"session queue: deadline {datetime.fromtimestamp(clock.deadline):%Y-%m-%d %H:%M} "
        f"({clock.left() / 3600:.2f} h){' DRY RUN' if a.dry_run else ''}", clock)
    i = 0
    while i < len(units):
        u = units[i]
        if u["kind"] == "stop":
            log("plan: stop", clock); break
        if u["kind"] == "check":
            if a.dry_run:
                log(f"check {' '.join(u['cmd'])} -> (dry run: assumed pass)", clock); i += 1; continue
            rc = subprocess.run(u["cmd"]).returncode
            log(f"check {' '.join(u['cmd'])} -> rc {rc}", clock)
            if rc == 0:
                i += 1; continue
            if rc == 1:
                i = labels[u["label"]]; continue
            log("check could not decide; queue stops", clock); break
        if u["kind"] == "pending":
            i += 1; continue
        if u["kind"] == "eval":
            if not do_eval(u["items"][0], clock, load_times(), "runs/logs/session_eval.log") and not done_eval(u["items"][0]):
                break
            i += 1; continue
        if unit_done(u):
            i += 1; continue
        times = load_times()
        need = sum(est_train(it, times) for it in u["items"] if not (trained(it) and all(done_eval(e) for e in it["evals"])))
        if clock.left() < FACTOR * need:
            log(f"does not fit: {' + '.join(it['name'] for it in u['items'])} (needs {FACTOR} x {need / 60:.0f} min, "
                f"{clock.left() / 60:.0f} min left); rolls over", clock)
            break
        ok = all([do_train(it, clock, times) for it in u["items"]])
        if not ok and clock.left() <= 0:
            break
        i += 1
    # pending evaluations: missing evals of every trained run in the plan, then 'pending' lines
    todo = [e for u in units if u["kind"] == "train" for it in u["items"] if trained(it) for e in it["evals"]]
    todo += [u["items"][0] for u in units if u["kind"] == "pending"]
    seen = set()
    for e in todo:
        key = (e["run"], e["out"])
        if key in seen:
            continue
        seen.add(key)
        run_trained = Path(e["run"], "train_summary.json").exists() or (SIM is not None and
                                                                        str(Path(e["run"], "train_summary.json")) in SIM)
        if not done_eval(e) and run_trained:
            do_eval(e, clock, load_times(), "runs/logs/session_eval.log")
    log("session queue: done", clock)


if __name__ == "__main__":
    sys.exit(main())
