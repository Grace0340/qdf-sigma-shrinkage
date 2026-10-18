"""Run the frozen Q1A v4 grid (protocol_v4.json) on AutoDL. Resumable.

--smoke DIR runs one 1-epoch / warmup_steps 5 job per new mode into DIR (engineering check only).
"""
import argparse
import glob
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

import run_grid as g1
import run_grid_v3 as g3

OUT = os.path.join(g1.HERE, "results", "runs_v4_autodl.jsonl")
PLATFORM = dict(name="autodl", host=platform.node(), python=platform.python_version(), torch=torch.__version__)


def selected_a():
    s = json.load(open(os.path.join(g1.HERE, "results", "summary_v3_amend1.json"), encoding="utf-8"))
    ds_T = lambda c: (c["cell"].split("-")[0], int(c["cell"].split("-")[1]))
    return {ds_T(c) + (e["seed"],): e["selected_a"] for c in s["cells"] for e in c["selection"]}


def env_for(arm, ds, T, seed, sel):
    if arm.startswith("COVY_A"):
        return {"Q1_SIGMA_MODE": "covy_abs", "Q1_PLUGIN_A": arm[len("COVY_A"):]}
    if arm == "LW":
        return {"Q1_SIGMA_MODE": "lw"}
    if arm == "SPECTRUM_RANDOM":
        return {"Q1_SIGMA_MODE": "spectrum_random", "Q1_PLUGIN_A": f"{sel[(ds, T, seed)]:g}", "Q1_ROT_SEED": str(seed)}
    if arm.startswith("PLUGIN_A"):
        return {"Q1_SIGMA_MODE": "plugin_abs", "Q1_PLUGIN_A": arm[len("PLUGIN_A"):]}
    raise ValueError(arm)


def run_one(job, sel):
    arm, ds, T, seed = job
    if arm in ("QDF", "SIGMA_I"):
        rec = g1.run_one(job)
    else:
        env_extra = env_for(arm, ds, T, seed, sel)
        rec = g1.run_one(("SIGMA_I", ds, T, seed), run_id=f"{arm}_{ds}_{T}_{seed}", env_extra=env_extra)
        rec["arm"] = arm
        rec["sigma_mode"] = env_extra["Q1_SIGMA_MODE"]
        work = os.path.join(g1.RUNS, rec["run_id"])
        info = glob.glob(os.path.join(work, "res", "*", "sigma_target.json"))
        rec["sigma_target"] = json.load(open(info[0])) if info else None
    work = os.path.join(g1.RUNS, rec["run_id"])
    rec.update(g3.parse_stdout(open(os.path.join(work, "stdout.txt"), encoding="utf-8", errors="replace").read()))
    rec["platform"] = PLATFORM
    return rec


def all_jobs(proto):
    gr = proto["grid"]
    a_tags = [g3.a_tag(a) for a in proto["a_grid"]]
    new = [f"COVY_A{a}" for a in a_tags] + ["LW", "SPECTRUM_RANDOM"]
    rerun_720 = ["QDF", "SIGMA_I"] + [f"PLUGIN_A{a}" for a in a_tags]
    jobs = []
    for T in sorted(gr["pred_len"], reverse=True):
        for ds in gr["datasets"]:
            for s in gr["seeds"]:
                for arm in (rerun_720 if T == 720 else []) + new:
                    jobs.append((arm, ds, T, s))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--smoke", default=None)
    p.add_argument("--only", default=None, help="smoke only: comma-separated arms")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    proto = json.load(open(os.path.join(g1.HERE, "protocol_v4.json"), encoding="utf-8"))
    sel = selected_a()
    out = OUT
    g1.RUNS = os.path.join(g1.HERE, "runs_v4")
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
        jobs = [("COVY_A0.5", "ETTh2", 96, 2021), ("LW", "ETTh2", 96, 2021), ("SPECTRUM_RANDOM", "ETTh2", 96, 2021),
                ("COVY_A0.01", "ETTh1", 720, 2021), ("LW", "ETTh1", 720, 2021), ("SPECTRUM_RANDOM", "ETTh1", 720, 2021)]
        if args.only:
            jobs = [j for j in jobs if j[0] in args.only.split(",")]
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
        for fut in as_completed([ex.submit(run_one, j, sel) for j in jobs]):
            rec = fut.result()
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} val={rec.get('best_val_mse')} "
                  f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
