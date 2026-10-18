"""Run the Q1C v8 fair-selection grid (protocol_v8.json) on the AutoDL GPU. Resumable.

Every arm of every cell runs on this GPU, including the ETTh cells that v1-v4 ran on CPU.
A run with any record (success or failure) counts as done; failures are retried only with --retry_failed.
--smoke DIR runs 1-epoch jobs into DIR (engineering check and timing only).
"""
import argparse
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import torch

import run_grid as g1
import run_grid_v3 as g3
import run_grid_v5 as g5
import run_grid_v7 as g7

PROTO = json.load(open(os.path.join(g1.HERE, "protocol_v8.json"), encoding="utf-8"))
OUT = os.path.join(g1.HERE, "results", "runs_v8.jsonl")
RUNS = os.path.join(g1.HERE, "runs_v8")
PLATFORM = dict(name="autodl-gpu", host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

CELLS = [("TQNet", c) for c in PROTO["cells"]["TQNet"]] + [("iTransformer", c) for c in PROTO["cells"]["iTransformer"]]
SEEDS5, SEEDS3, REP_SEED = [2021, 2022, 2023, 2024, 2025], [2021, 2022, 2023], 2021
A_GRID = {"TQNet": [0.01, 0.03, 0.1, 0.25, 0.5, 1, 2], "iTransformer": [0.01, 0.03, 0.1, 0.3, 1]}
M_GRID = {"TQNet": [0.125, 0.25, 0.5, 1, 2, 4, 8], "iTransformer": [0.25, 0.5, 1, 2, 4]}
K_GRID = [0.5, 1, 2]
MEM_ETTH = 3
SMOKE = False


def split(cell):
    ds, T = cell.split("-")
    return ds, int(T)


def seeds(bb, ds):
    return SEEDS5 if ds.startswith("ETTh") or bb == "iTransformer" else SEEDS3


def released(bb, ds, T):
    """(meta_lr, meta steps) of the released scripts; steps = warmup_steps."""
    if ds.startswith("ETTh"):
        return g1.QDF_HP[(ds, T)]["lr_meta"], 300
    _, lr_meta, warm, _, _ = g5.HP[(ds, T)]
    return lr_meta, warm


def arms(bb):
    out = ["SIGMA_I"] + [f"QDF_LR{m:g}" for m in M_GRID[bb]]
    out += [f"PLUGIN_A{a:g}" for a in A_GRID[bb]] + [f"PDIAG_A{a:g}" for a in A_GRID[bb]]
    return out + [f"REPLAY_K{k:g}" for k in K_GRID]


def mem(job):
    _, bb, ds, T, _ = job
    return MEM_ETTH if ds.startswith("ETTh") else g5.MEM_GB[(ds, T)]


# GPU-memory reservation in GB (32 GB card): iTransformer/ECL-96 measured at 9.5 GB, TQNet below 0.5 GB.
def gpu_mem(job):
    _, bb, ds, T, _ = job
    return 9.6 if bb == "iTransformer" else 1.5 if ds == "Weather" else 0.8


def base_cmd(bb, ds, T, seed, work, sigma_identity):
    if bb == "iTransformer":
        g7.SMOKE = SMOKE
        return g7.cmd_for(bb, ds, T, seed, work, sigma_identity)
    c = g1.qdf_cmd(ds, T, seed, work, sigma_identity) if ds.startswith("ETTh") else \
        g5.qdf_cmd(ds, T, seed, work, sigma_identity)
    if SMOKE:
        c[c.index("--train_epochs") + 1] = "1"
    return c


def env_for(arm, bb, ds, T):
    if arm.startswith("PLUGIN_A"):
        return {"Q1_SIGMA_MODE": "plugin_abs", "Q1_PLUGIN_A": arm[len("PLUGIN_A"):]}
    if arm.startswith("PDIAG_A"):
        return {"Q1_SIGMA_MODE": "plugin_diag_abs", "Q1_PLUGIN_A": arm[len("PDIAG_A"):]}
    if arm.startswith("REPLAY_K"):
        lr, steps = released(bb, ds, T)
        return {"Q1_SIGMA_MODE": "replay", "Q1_REPLAY_K": arm[len("REPLAY_K"):],
                "Q1_REPLAY_META_LR": f"{lr:g}", "Q1_REPLAY_STEPS": str(steps)}
    return {}


def run_id(job):
    arm, bb, ds, T, seed = job
    return f"{arm}_{bb}_{ds}_{T}_{seed}"


def run_one(job):
    arm, bb, ds, T, seed = job
    rid = run_id(job)
    work = os.path.join(RUNS, rid)
    os.makedirs(work, exist_ok=True)
    is_qdf = arm.startswith("QDF_LR")
    cmd = base_cmd(bb, ds, T, seed, work, sigma_identity=not is_qdf)
    if is_qdf:
        m = float(arm[len("QDF_LR"):])
        cmd[cmd.index("--meta_lr") + 1] = f"{released(bb, ds, T)[0] * m:g}"
    env_extra = env_for(arm, bb, ds, T)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               **env_extra)
    t0 = time.time()
    with open(os.path.join(work, "stdout.txt"), "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    text = open(os.path.join(work, "stdout.txt"), encoding="utf-8", errors="replace").read()
    mt, n = g1.METRIC.findall(text), g1.TEST_N.findall(text)
    rec = dict(run_id=rid, arm=arm, backbone=bb, dataset=ds, pred_len=T, seed=seed, returncode=rc,
               test_mse=float(mt[-1][0]) if mt else None, test_mae=float(mt[-1][1]) if mt else None,
               test_windows=int(n[-1]) if n else None, seconds=round(time.time() - t0, 1), cmd=cmd[3:],
               finished=datetime.datetime.now().isoformat(timespec="seconds"))
    if env_extra:
        rec["sigma_mode"] = env_extra["Q1_SIGMA_MODE"]
        info = glob.glob(os.path.join(work, "res", "*", "sigma_target.json"))
        rec["sigma_target"] = json.load(open(info[0])) if info else None
    rec.update(g3.parse_stdout(text))
    rec["platform"] = PLATFORM
    return rec


def all_jobs():
    jobs = []
    for bb, c in CELLS:
        ds, T = split(c)
        jobs += [(arm, bb, ds, T, s) for s in seeds(bb, ds) for arm in arms(bb)]
        jobs.append(("SIGMA_I_REP", bb, ds, T, REP_SEED))
    return sorted(jobs, key=lambda j: (-mem(j), j[1], j[2], j[3], j[4]))


def main():
    global SMOKE, RUNS
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--smoke", default=None)
    p.add_argument("--retry_failed", action="store_true")
    p.add_argument("--mem_budget", type=int, default=g5.MEM_BUDGET)
    p.add_argument("--gpu_budget", type=float, default=31.0)
    args = p.parse_args()
    out = OUT
    if args.smoke:
        SMOKE = True
        RUNS = os.path.join(g1.HERE, args.smoke)
        out = os.path.join(RUNS, "runs_smoke.jsonl")
        jobs = [("PDIAG_A0.1", "TQNet", "ETTh1", 96, 2021), ("REPLAY_K1", "TQNet", "ETTh1", 96, 2021),
                ("QDF_LR8", "TQNet", "ETTh1", 96, 2021), ("QDF_LR1", "TQNet", "ETTh1", 96, 2021),
                ("SIGMA_I", "TQNet", "ETTh1", 96, 2021), ("REPLAY_K2", "TQNet", "Weather", 720, 2021),
                ("REPLAY_K1", "TQNet", "ETTh2", 720, 2021), ("QDF_LR4", "iTransformer", "ECL", 96, 2021),
                ("REPLAY_K1", "iTransformer", "ECL", 96, 2021)]
    else:
        jobs = all_jobs()
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if not args.retry_failed or (r["returncode"] == 0 and r["test_mse"] is not None):
                done.add(r["run_id"])
    jobs = [j for j in jobs if run_id(j) not in done]
    print(f"{len(jobs)} runs to go, {len(done)} already recorded", flush=True)
    pending, running = list(jobs), {}
    with ThreadPoolExecutor(args.jobs) as ex:
        while pending or running:
            used = sum(mem(j) for j in running.values())
            gused = sum(gpu_mem(j) for j in running.values())
            fit = [j for j in pending if (used + mem(j) <= args.mem_budget and gused + gpu_mem(j) <= args.gpu_budget)
                   or not running] if len(running) < args.jobs else []
            if fit:
                pending.remove(fit[0])
                running[ex.submit(run_one, fit[0])] = fit[0]
                continue
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                running.pop(fut)
                rec = fut.result()
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} val={rec.get('best_val_mse')} "
                      f"{rec['seconds']}s epochs={rec.get('epochs')} meta={rec.get('meta_phase_seconds')}s "
                      f"[{len(pending)} pending, {len(running)} running]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
