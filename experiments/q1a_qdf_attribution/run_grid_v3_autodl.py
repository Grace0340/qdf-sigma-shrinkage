"""Run the T=96 part of Q1A v3 on AutoDL (protocol_v3_amend1.json). Resumable.

Jobs: ETTh1/ETTh2, T=96, seeds 2021-2025, arms QDF and SIGMA_I (v1 commands) plus the 8 v3 arms.
"""
import argparse
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

import run_grid as g1
import run_grid_v3 as g3

OUT = os.path.join(g1.HERE, "results", "runs_v3_autodl.jsonl")
PLATFORM = dict(name="autodl", host=platform.node(), python=platform.python_version(), torch=torch.__version__)


def all_jobs(proto):
    gr = proto["grid"]
    arms = ["QDF", "SIGMA_I"] + [f"PLUGIN_A{g3.a_tag(a)}" for a in proto["a_grid"]] + ["FGLS_FULL"]
    return [(arm, ds, 96, s) for ds in gr["datasets"] for s in gr["seeds"] for arm in arms]


def run(job):
    rec = g1.run_one(job) if job[0] in ("QDF", "SIGMA_I") else g3.run_one(job)
    rec["platform"] = PLATFORM
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=14)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    proto = json.load(open(os.path.join(g1.HERE, "protocol_v3.json"), encoding="utf-8"))
    g1.RUNS = os.path.join(g1.HERE, "runs_autodl")
    os.makedirs(g1.RUNS, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            if r["returncode"] == 0 and r["test_mse"] is not None:
                done.add(r["run_id"])
    jobs = [j for j in all_jobs(proto) if f"{j[0]}_{j[1]}_{j[2]}_{j[3]}" not in done]
    print(f"{len(jobs)} runs to go, {len(done)} already done", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            rec = fut.result()
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} val={rec.get('best_val_mse')} "
                  f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
