"""Run the frozen Q1A v3 grid (protocol_v3.json). Resumable. Run from <HOME>\\q1a.

--smoke DIR runs one 1-epoch / warmup_steps 5 job per new mode into DIR (engineering check only).
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_grid as g1

OUT = os.path.join(g1.HERE, "results", "runs_v3.jsonl")
VALID = re.compile(r"Valid Loss Cov: \S+, MSE: (\S+)")
META_STEP = re.compile(r"^Step: \d+ cost time: ([0-9.]+)s", re.M)
EPOCH = re.compile(r"^Epoch: \d+ cost time: ([0-9.eE+-]+)", re.M)
VALTIME = re.compile(r"^Validation cost time: ([0-9.eE+-]+)", re.M)


def a_tag(a):
    return f"{a:g}"


def parse_stdout(text):
    vals = []
    for v in VALID.findall(text):
        try:
            x = float(v)
        except ValueError:
            continue
        if x == x and x not in (float("inf"), float("-inf")):
            vals.append(x)
    return dict(best_val_mse=min(vals) if vals else None, val_mse_per_epoch=vals, epochs=len(VALID.findall(text)),
                meta_phase_seconds=round(sum(map(float, META_STEP.findall(text))), 2),
                meta_test_seconds=round(sum(map(float, EPOCH.findall(text))) + sum(map(float, VALTIME.findall(text))), 2))


def run_one(job):
    arm, ds, T, seed = job
    if arm == "FGLS_FULL":
        env_extra = {"Q1_SIGMA_MODE": "fgls", "Q1_FGLS_RIDGE": "1e-3"}
    else:
        env_extra = {"Q1_SIGMA_MODE": "plugin_abs", "Q1_PLUGIN_A": arm[len("PLUGIN_A"):]}
    # the SIGMA_I command is the QDF command with --meta_lr 0, which the protocol requires
    rec = g1.run_one(("SIGMA_I", ds, T, seed), run_id=f"{arm}_{ds}_{T}_{seed}", env_extra=env_extra)
    rec["arm"] = arm
    rec["sigma_mode"] = env_extra["Q1_SIGMA_MODE"]
    work = os.path.join(g1.RUNS, rec["run_id"])
    info = glob.glob(os.path.join(work, "res", "*", "sigma_target.json"))
    rec["sigma_target"] = json.load(open(info[0])) if info else None
    rec.update(parse_stdout(open(os.path.join(work, "stdout.txt"), encoding="utf-8", errors="replace").read()))
    return rec


def all_jobs(proto):
    gr = proto["grid"]
    arms = [f"PLUGIN_A{a_tag(a)}" for a in proto["a_grid"]] + ["FGLS_FULL"]
    return [(arm, ds, T, s) for T in sorted(gr["pred_len"], reverse=True) for ds in gr["datasets"]
            for s in gr["seeds"] for arm in arms]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--smoke", default=None)
    args = p.parse_args()
    proto = json.load(open(os.path.join(g1.HERE, "protocol_v3.json"), encoding="utf-8"))
    out = OUT
    jobs = all_jobs(proto)
    if args.smoke:
        g1.RUNS = os.path.join(g1.HERE, args.smoke)
        out = os.path.join(g1.RUNS, "runs_smoke.jsonl")
        orig = g1.qdf_cmd

        def smoke_cmd(*a, **k):
            c = orig(*a, **k)
            c[c.index("--train_epochs") + 1] = "1"
            c[c.index("--warmup_steps") + 1] = "5"
            return c

        g1.qdf_cmd = smoke_cmd
        jobs = [("PLUGIN_A0.5", "ETTh2", 96, 2021), ("FGLS_FULL", "ETTh2", 96, 2021), ("PLUGIN_A0.01", "ETTh1", 720, 2021),
                ("FGLS_FULL", "ETTh1", 720, 2021)]
    os.makedirs(g1.RUNS, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r["returncode"] == 0 and r["test_mse"] is not None:
                done.add(r["run_id"])
    jobs = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}_{j[3]}" not in done]
    print(f"{len(jobs)} runs to go, {len(done)} already done", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        for fut in as_completed([ex.submit(run_one, j) for j in jobs]):
            rec = fut.result()
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} val={rec['best_val_mse']} "
                  f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
