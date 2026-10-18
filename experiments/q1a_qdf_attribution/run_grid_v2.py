"""Run the frozen Q1A v2 grid (protocol_v2.json). Resumable. Run from <HOME>\\q1a."""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_grid as g1

OUT = os.path.join(g1.HERE, "results", "runs_v2.jsonl")
MODES = {"PLUGIN": "plugin", "SCALAR": "scalar", "DIAG": "diag", "TOPK8": "topk", "REINJECT": "learned"}


def learned_a(ds, T, seed):
    hits = glob.glob(os.path.join(g1.RUNS, f"QDF_{ds}_{T}_{seed}", "ck", "*", "A.pth"))
    if len(hits) != 1:
        raise FileNotFoundError(f"learned A for {ds}-{T}-{seed}: {hits}")
    return hits[0]


def run_one(job):
    arm, ds, T, seed = job
    env_extra = {"Q1_SIGMA_MODE": MODES[arm], "Q1_LEARNED_A": learned_a(ds, T, seed), "Q1_TOPK": "8"}
    # the SIGMA_I command is the QDF command with --meta_lr 0, which the protocol requires
    rec = g1.run_one(("SIGMA_I", ds, T, seed), run_id=f"{arm}_{ds}_{T}_{seed}", env_extra=env_extra)
    rec["arm"] = arm
    rec["sigma_mode"] = env_extra["Q1_SIGMA_MODE"]
    info = glob.glob(os.path.join(g1.RUNS, rec["run_id"], "res", "*", "sigma_target.json"))
    rec["sigma_target"] = json.load(open(info[0])) if info else None
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=14)
    args = p.parse_args()
    proto = json.load(open(os.path.join(g1.HERE, "protocol_v2.json"), encoding="utf-8"))
    gr = proto["grid"]
    jobs = [("REINJECT", ds, T, 2021) for T in gr["pred_len"] for ds in gr["datasets"]]
    jobs += [(arm, ds, T, s) for T in sorted(gr["pred_len"], reverse=True) for ds in gr["datasets"]
             for s in gr["seeds"] for arm in ("PLUGIN", "SCALAR", "DIAG", "TOPK8")]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            if r["returncode"] == 0 and r["test_mse"] is not None:
                done.add(r["run_id"])
    jobs = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}_{j[3]}" not in done]
    print(f"{len(jobs)} runs to go, {len(done)} already done", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        for fut in as_completed([ex.submit(run_one, j) for j in jobs]):
            rec = fut.result()
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
