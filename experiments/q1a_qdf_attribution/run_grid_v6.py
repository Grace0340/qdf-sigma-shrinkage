"""Run the frozen Q1B v6 trace-matched shape check (protocol_v6.json) on the AutoDL GPU. Resumable.

Same commands as v5 (run_grid_v5.qdf_cmd); the learned Sigma comes from the v5 QDF run with the same cell and seed.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import run_grid as g1
import run_grid_v3 as g3
import run_grid_v5 as g5

OUT = os.path.join(g1.HERE, "results", "runs_v6.jsonl")
V5_RUNS = os.path.join(g1.HERE, "runs_v5")
MODES = {"LEARNED": "learned", "PLUGIN_MATCHED": "plugin", "PLUGIN_DIAG_MATCHED": "plugin_diag"}


def learned_a(ds, T, seed):
    hits = glob.glob(os.path.join(V5_RUNS, f"QDF_{ds}_{T}_{seed}", "ck", "*", "A.pth"))
    if len(hits) != 1:
        raise FileNotFoundError(f"learned A for {ds}-{T}-{seed}: {hits}")
    return hits[0]


def run_one(job):
    arm, ds, T, seed = job
    env_extra = {"Q1_SIGMA_MODE": MODES[arm], "Q1_LEARNED_A": learned_a(ds, T, seed)}
    rec = g1.run_one(("SIGMA_I", ds, T, seed), run_id=f"{arm}_{ds}_{T}_{seed}", env_extra=env_extra)
    rec["arm"] = arm
    rec["sigma_mode"] = env_extra["Q1_SIGMA_MODE"]
    rec["learned_a_path"] = env_extra["Q1_LEARNED_A"]
    info = glob.glob(os.path.join(g1.RUNS, rec["run_id"], "res", "*", "sigma_target.json"))
    rec["sigma_target"] = json.load(open(info[0])) if info else None
    rec.update(g3.parse_stdout(open(os.path.join(g1.RUNS, rec["run_id"], "stdout.txt"), encoding="utf-8",
                                    errors="replace").read()))
    rec["platform"] = g5.PLATFORM
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--mem_budget", type=int, default=g5.MEM_BUDGET)
    args = p.parse_args()
    proto = json.load(open(os.path.join(g1.HERE, "protocol_v6.json"), encoding="utf-8"))
    g1.qdf_cmd = g5.qdf_cmd
    g1.RUNS = os.path.join(g1.HERE, "runs_v6")
    os.makedirs(g1.RUNS, exist_ok=True)
    cells = [(c.rsplit("-", 1)[0], int(c.rsplit("-", 1)[1])) for c in proto["cells"]]
    jobs = [(arm, ds, T, s) for ds, T in cells for s in proto["seeds"] for arm in proto["arms"]]
    jobs.sort(key=lambda j: (-g5.MEM_GB[(j[1], j[2])], j[3]))
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            if r["returncode"] == 0 and r["test_mse"] is not None:
                done.add(r["run_id"])
    pending = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}_{j[3]}" not in done]
    print(f"{len(pending)} runs to go, {len(done)} already done", flush=True)
    running = {}
    mem = lambda j: g5.MEM_GB[(j[1], j[2])]
    with ThreadPoolExecutor(args.jobs) as ex:
        while pending or running:
            used = sum(mem(j) for j in running.values())
            fit = [j for j in pending if used + mem(j) <= args.mem_budget or not running] if len(running) < args.jobs else []
            if fit:
                pending.remove(fit[0])
                running[ex.submit(run_one, fit[0])] = fit[0]
                continue
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                running.pop(fut)
                rec = fut.result()
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} {rec['seconds']}s "
                      f"[{len(pending)} pending, {len(running)} running]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
