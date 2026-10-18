"""Run the frozen Q1B v5 generalisation grid (protocol_v5.json) on the AutoDL GPU. Resumable.

TQNet with the QDF release's own hyper-parameters (scripts/{ETTm1,ETTm2,Weather,ECL}.sh at eb0693a9).
--smoke DIR runs 1-epoch jobs into DIR (engineering check and timing only).
"""
import argparse
import glob
import json
import os
import platform
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import torch

import run_grid as g1
import run_grid_v3 as g3
import run_grid_v4 as g4

OUT = os.path.join(g1.HERE, "results", "runs_v5.jsonl")
PLATFORM = dict(name="autodl-gpu", host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

DATA = {
    "ETTm1": dict(root="ETT-small", file="ETTm1.csv", data="ETTm1", enc=7, cycle=96, dropout=0.5, bs=32),
    "ETTm2": dict(root="ETT-small", file="ETTm2.csv", data="ETTm2", enc=7, cycle=96, dropout=0.5, bs=32),
    "Weather": dict(root="weather", file="weather.csv", data="custom", enc=21, cycle=144, dropout=0.5, bs=32),
    "ECL": dict(root="electricity", file="electricity.csv", data="custom", enc=321, cycle=168, dropout=0.0, bs=16),
}
# lr = lr_inner, lr_meta, warmup_steps, num_tasks, meta_inner_steps
HP = {
    ("ETTm1", 96): (0.001, 0.1, 500, 2, 1), ("ETTm1", 720): (0.001, 0.2, 500, 2, 5),
    ("ETTm2", 96): (0.0002, 0.2, 500, 1, 5), ("ETTm2", 720): (0.0002, 0.2, 500, 5, 5),
    ("Weather", 96): (0.002, 0.1, 700, 5, 4), ("Weather", 720): (0.002, 0.2, 700, 3, 4),
    ("ECL", 96): (0.005, 0.1, 300, 3, 2), ("ECL", 720): (0.01, 0.05, 300, 1, 1),
}


def qdf_cmd(ds, T, seed, work, sigma_identity):
    d, (lr, lr_meta, warm, tasks, inner) = DATA[ds], HP[(ds, T)]
    a = [
        "--task_name", "long_term_forecast_meta_ml3", "--is_training", "1",
        "--root_path", os.path.join(g1.HERE, "QDF", "dataset", d["root"]) + os.sep,
        "--data_path", d["file"], "--model_id", f"{ds}_96_{T}", "--model", "TQNet",
        "--data_id", ds, "--data", d["data"], "--features", "M", "--seq_len", "96", "--label_len", "48",
        "--pred_len", str(T), "--enc_in", str(d["enc"]), "--dec_in", str(d["enc"]), "--c_out", str(d["enc"]),
        "--factor", "3", "--des", "TQNet", "--learning_rate", str(lr), "--lradj", "type1",
        "--train_epochs", "30", "--patience", "5", "--batch_size", str(d["bs"]), "--test_batch_size", "1",
        "--itr", "1", "--rec_lambda", "1.0", "--auxi_lambda", "0.0", "--reg_lambda", "0.0",
        "--auxi_batch_size", "64", "--fix_seed", str(seed),
        "--checkpoints", os.path.join(work, "ck"), "--results", os.path.join(work, "res"),
        "--test_results", os.path.join(work, "tr"), "--log_path", os.path.join(work, "log.txt"),
        "--rerun", "1", "--inner_lr", str(lr), "--meta_lr", "0.0" if sigma_identity else str(lr_meta),
        "--meta_inner_steps", str(inner), "--overlap_ratio", "0.0",
        "--num_tasks", str(tasks), "--max_norm", "5.0", "--auxi_loss", "MSE",
        "--model_type", "linear", "--cycle", str(d["cycle"]), "--use_revin", "1", "--dropout", str(d["dropout"]),
        "--first_order", "1", "--warmup_steps", str(warm), "--num_workers", "0",
    ]
    return [g1.PY, "-u", os.path.join(g1.HERE, "cpu_launch.py"), os.path.join(g1.HERE, "QDF"), work, "--"] + a


# Resident-memory reservation in GB. The QDF release wraps every meta-task DataLoader in itertools.cycle,
# which caches the whole training set in host memory; the container is capped at 60 GiB (memory.high).
MEM_GB = {("ETTm1", 96): 4, ("ETTm1", 720): 5, ("ETTm2", 96): 4, ("ETTm2", 720): 5,
          ("Weather", 96): 5, ("Weather", 720): 10, ("ECL", 96): 14, ("ECL", 720): 45}
MEM_BUDGET = 50


def run_one(job):
    arm, ds, T, seed = job
    if arm in ("QDF", "SIGMA_I", "SIGMA_I_REP"):
        rec = g1.run_one(("QDF" if arm == "QDF" else "SIGMA_I", ds, T, seed), run_id=f"{arm}_{ds}_{T}_{seed}")
        rec["arm"] = arm
    else:
        env_extra = g4.env_for(arm, ds, T, seed, SEL if arm == "SPECTRUM_RANDOM" and not SMOKE else
                               {(ds, T, seed): 0.25})
        rec = g1.run_one(("SIGMA_I", ds, T, seed), run_id=f"{arm}_{ds}_{T}_{seed}", env_extra=env_extra)
        rec["arm"] = arm
        rec["sigma_mode"] = env_extra["Q1_SIGMA_MODE"]
        info = glob.glob(os.path.join(g1.RUNS, rec["run_id"], "res", "*", "sigma_target.json"))
        rec["sigma_target"] = json.load(open(info[0])) if info else None
    rec.update(g3.parse_stdout(open(os.path.join(g1.RUNS, rec["run_id"], "stdout.txt"), encoding="utf-8",
                                    errors="replace").read()))
    rec["platform"] = PLATFORM
    return rec


SEL = {}
SMOKE = False


def stage_jobs(stage):
    """Stage A: all cells; stage B: analysis cells of a complete stage A (protocol_v5.json)."""
    import summarize_v5 as s5
    if stage == "A":
        cells, arms = s5.PROTO["cells"], s5.STAGE_A_ARMS
    else:
        complete, rows = s5.stage_a(s5.load(OUT))
        if not complete:
            raise SystemExit("stage A incomplete; stage B not started")
        cells, arms = [c for c, r in rows.items() if r["analysis"]], s5.STAGE_B_ARMS
        for c in cells:
            ds, T = s5.split(c)
            SEL.update({(ds, T, int(s)): a for s, a in rows[c]["selected_a"].items()})
        print(f"stage B analysis cells: {cells}; SPECTRUM_RANDOM a: {SEL}", flush=True)
    jobs = [(arm, *s5.split(c), s) for c in cells for s in s5.SEEDS for arm in arms]
    return sorted(jobs, key=lambda j: (-MEM_GB[(j[1], j[2])], j[3]))


def main():
    global SMOKE
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--stage", choices=["A", "B"], default="A")
    p.add_argument("--smoke", default=None)
    p.add_argument("--mem_budget", type=int, default=MEM_BUDGET)
    args = p.parse_args()
    globals()["MEM_BUDGET"] = args.mem_budget
    g1.qdf_cmd = qdf_cmd
    g1.RUNS = os.path.join(g1.HERE, "runs_v5")
    out = OUT
    if args.smoke:
        SMOKE = True
        jobs = [("PLUGIN_A0.25", "ECL", 96, 2021), ("COVY_A0.25", "ECL", 96, 2021),
                ("SPECTRUM_RANDOM", "Weather", 720, 2021), ("PLUGIN_A0.25", "ETTm2", 720, 2021),
                ("QDF", "Weather", 96, 2021), ("QDF", "Weather", 720, 2021)]
        g1.RUNS = os.path.join(g1.HERE, args.smoke)
        out = os.path.join(g1.RUNS, "runs_smoke.jsonl")

        def smoke_cmd(*a, **k):
            c = qdf_cmd(*a, **k)
            c[c.index("--train_epochs") + 1] = "1"
            return c

        g1.qdf_cmd = smoke_cmd
    else:
        jobs = stage_jobs(args.stage)
    os.makedirs(g1.RUNS, exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r["returncode"] == 0 and r["test_mse"] is not None:
                done.add(r["run_id"])
    jobs = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}_{j[3]}" not in done]
    print(f"{len(jobs)} runs to go, {len(done)} already done", flush=True)
    pending, running = list(jobs), {}
    mem = lambda j: MEM_GB[(j[1], j[2])]
    with ThreadPoolExecutor(args.jobs) as ex:
        while pending or running:
            used = sum(mem(j) for j in running.values())
            fit = [j for j in pending if used + mem(j) <= MEM_BUDGET or not running] if len(running) < args.jobs else []
            if fit:
                pending.remove(fit[0])
                running[ex.submit(run_one, fit[0])] = fit[0]
                continue
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                running.pop(fut)
                rec = fut.result()
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} val={rec.get('best_val_mse')} "
                      f"{rec['seconds']}s epochs={rec.get('epochs')} meta={rec.get('meta_phase_seconds')}s "
                      f"[{len(pending)} pending, {len(running)} running]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
