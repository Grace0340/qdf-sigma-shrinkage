"""Run the Q1B v7 second-backbone grid (protocol_v7.json) on the AutoDL GPU. Resumable.

iTransformer and DLinear inside the released QDF code. Backbone hyper-parameters from the official backbone
scripts (research_notes/v5_inputs.md); QDF hyper-parameters from the QDF release's TQNet scripts for the same
(dataset, pred_len), as in run_grid_v5.HP.
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
import run_grid_v4 as g4
import run_grid_v5 as g5

OUT = os.path.join(g1.HERE, "results", "runs_v7.jsonl")
RUNS = os.path.join(g1.HERE, "runs_v7")
PLATFORM = dict(name="autodl-gpu", host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# Official backbone settings: lr, batch_size, train_epochs, patience, extra flags
BACKBONE = {
    ("iTransformer", "ECL"): (0.0005, 16, 10, 3),
    ("iTransformer", "Weather"): (0.0001, 32, 10, 3),
    ("DLinear", "ECL"): (0.001, 16, 10, 3),
    ("DLinear", "Weather"): (0.0001, 16, 10, 3),
}
ARCH = {
    "iTransformer": ["--e_layers", "3", "--d_model", "512", "--d_ff", "512", "--n_heads", "8", "--factor", "1",
                     "--dropout", "0.1", "--activation", "gelu", "--embed", "timeF"],
    "DLinear": ["--moving_avg", "25", "--individual", "0", "--factor", "3", "--dropout", "0.1"],
}
SMOKE = False


def cmd_for(bb, ds, T, seed, work, sigma_identity):
    d = g5.DATA[ds]
    _, lr_meta, warm, tasks, inner = g5.HP[(ds, T)]
    lr, bs, epochs, patience = BACKBONE[(bb, ds)]
    a = [
        "--task_name", "long_term_forecast_meta_ml3", "--is_training", "1",
        "--root_path", os.path.join(g1.HERE, "QDF", "dataset", d["root"]) + os.sep,
        "--data_path", d["file"], "--model_id", f"{ds}_96_{T}", "--model", bb,
        "--data_id", ds, "--data", d["data"], "--features", "M", "--seq_len", "96", "--label_len", "48",
        "--pred_len", str(T), "--enc_in", str(d["enc"]), "--dec_in", str(d["enc"]), "--c_out", str(d["enc"]),
        "--des", bb, "--learning_rate", str(lr), "--lradj", "type1",
        "--train_epochs", "1" if SMOKE else str(epochs), "--patience", str(patience), "--batch_size", str(bs),
        "--test_batch_size", "1", "--itr", "1", "--rec_lambda", "1.0", "--auxi_lambda", "0.0", "--reg_lambda", "0.0",
        "--auxi_batch_size", "64", "--fix_seed", str(seed),
        "--checkpoints", os.path.join(work, "ck"), "--results", os.path.join(work, "res"),
        "--test_results", os.path.join(work, "tr"), "--log_path", os.path.join(work, "log.txt"),
        "--rerun", "1", "--inner_lr", str(lr), "--meta_lr", "0.0" if sigma_identity else str(lr_meta),
        "--meta_inner_steps", str(inner), "--overlap_ratio", "0.0",
        "--num_tasks", str(tasks), "--max_norm", "5.0", "--auxi_loss", "MSE",
        "--model_type", "linear", "--cycle", str(d["cycle"]), "--use_revin", "1",
        "--first_order", "1", "--warmup_steps", str(warm), "--num_workers", "0",
    ] + ARCH[bb]
    return [g1.PY, "-u", os.path.join(g1.HERE, "cpu_launch.py"), os.path.join(g1.HERE, "QDF"), work, "--"] + a


def run_id(job):
    arm, bb, ds, T, seed = job
    return f"{arm}_{bb}_{ds}_{T}_{seed}"


def run_one(job, sel):
    arm, bb, ds, T, seed = job
    rid = run_id(job)
    work = os.path.join(RUNS, rid)
    os.makedirs(work, exist_ok=True)
    env_extra = {}
    if arm not in ("QDF", "SIGMA_I", "SIGMA_I_REP"):
        env_extra = g4.env_for(arm, ds, T, seed, {(ds, T, seed): sel.get((bb, ds, T, seed), 0.1)})
    cmd = cmd_for(bb, ds, T, seed, work, sigma_identity=(arm != "QDF"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               **env_extra)
    t0 = time.time()
    with open(os.path.join(work, "stdout.txt"), "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    text = open(os.path.join(work, "stdout.txt"), encoding="utf-8", errors="replace").read()
    m, n = g1.METRIC.findall(text), g1.TEST_N.findall(text)
    rec = dict(run_id=rid, arm=arm, backbone=bb, dataset=ds, pred_len=T, seed=seed, returncode=rc,
               test_mse=float(m[-1][0]) if m else None, test_mae=float(m[-1][1]) if m else None,
               test_windows=int(n[-1]) if n else None, seconds=round(time.time() - t0, 1), cmd=cmd[3:],
               finished=datetime.datetime.now().isoformat(timespec="seconds"))
    if env_extra:
        rec["sigma_mode"] = env_extra["Q1_SIGMA_MODE"]
        info = glob.glob(os.path.join(work, "res", "*", "sigma_target.json"))
        rec["sigma_target"] = json.load(open(info[0])) if info else None
    rec.update(g3.parse_stdout(text))
    rec["platform"] = PLATFORM
    return rec


def stage_jobs(stage):
    import summarize_v7 as s7
    sel = {}
    if stage == "A":
        jobs = [(arm, *s7.split(c), s) for c in s7.CELLS for s in s7.SEEDS for arm in s7.STAGE_A_ARMS]
        jobs += [("SIGMA_I_REP", *s7.split(c), s7.REP_SEED) for c in s7.CELLS]
    else:
        complete, rows = s7.stage_a(s7.load(OUT))
        if not complete:
            raise SystemExit("stage A incomplete; stage B not started")
        cells = [c for c, r in rows.items() if r["analysis"]]
        for c in cells:
            sel.update({(*s7.split(c), int(s)): a for s, a in rows[c]["selected_a"].items()})
        print(f"stage B analysis cells: {cells}; SPECTRUM_RANDOM a: {sel}", flush=True)
        jobs = [(arm, *s7.split(c), s) for c in cells for s in s7.SEEDS for arm in s7.STAGE_B_ARMS]
    return sorted(jobs, key=lambda j: (-g5.MEM_GB[(j[2], j[3])], j[1], j[4])), sel


def main():
    global SMOKE, RUNS
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--stage", choices=["A", "B"], default="A")
    p.add_argument("--smoke", default=None)
    p.add_argument("--mem_budget", type=int, default=g5.MEM_BUDGET)
    args = p.parse_args()
    out = OUT
    if args.smoke:
        SMOKE = True
        RUNS = os.path.join(g1.HERE, args.smoke)
        out = os.path.join(RUNS, "runs_smoke.jsonl")
        jobs = [(arm, bb, ds, T, 2021) for bb in ("iTransformer", "DLinear")
                for (ds, T) in (("ECL", 96), ("Weather", 96), ("Weather", 720)) for arm in ("QDF", "SIGMA_I")]
        jobs += [("PLUGIN_A0.1", "iTransformer", "ECL", 96, 2021), ("COVY_A0.1", "DLinear", "Weather", 720, 2021),
                 ("SPECTRUM_RANDOM", "iTransformer", "Weather", 720, 2021)]
        sel = {}
    else:
        jobs, sel = stage_jobs(args.stage)
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r["returncode"] == 0 and r["test_mse"] is not None:
                done.add(r["run_id"])
    jobs = [j for j in jobs if run_id(j) not in done]
    print(f"{len(jobs)} runs to go, {len(done)} already done", flush=True)
    pending, running = list(jobs), {}
    mem = lambda j: g5.MEM_GB[(j[2], j[3])]
    with ThreadPoolExecutor(args.jobs) as ex:
        while pending or running:
            used = sum(mem(j) for j in running.values())
            fit = [j for j in pending if used + mem(j) <= args.mem_budget or not running] if len(running) < args.jobs else []
            if fit:
                pending.remove(fit[0])
                running[ex.submit(run_one, fit[0], sel)] = fit[0]
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
