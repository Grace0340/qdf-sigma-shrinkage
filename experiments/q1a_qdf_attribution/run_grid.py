"""Run the frozen Q1A grid (protocol.json). Resumable: finished runs are skipped.

Run from the short junction path, e.g. <HOME>\\q1a, to stay under MAX_PATH.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("Q1_PY") or (os.path.join(HERE, ".venv", "Scripts", "python.exe") if os.name == "nt" else sys.executable)
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results", "runs.jsonl")

# scripts/ETTh{1,2}.sh at QDF eb0693a9
QDF_HP = {
    ("ETTh1", 96): dict(lr=0.005, lr_inner=0.005, lr_meta=0.2, num_tasks=3, meta_inner_steps=3),
    ("ETTh1", 720): dict(lr=0.005, lr_inner=0.005, lr_meta=0.01, num_tasks=5, meta_inner_steps=2),
    ("ETTh2", 96): dict(lr=0.0005, lr_inner=0.0005, lr_meta=0.2, num_tasks=1, meta_inner_steps=2),
    ("ETTh2", 720): dict(lr=0.0005, lr_inner=0.0005, lr_meta=0.2, num_tasks=2, meta_inner_steps=4),
}


def qdf_cmd(ds, T, seed, work, sigma_identity):
    hp = QDF_HP[(ds, T)]
    meta_lr = 0.0 if sigma_identity else hp["lr_meta"]
    a = [
        "--task_name", "long_term_forecast_meta_ml3", "--is_training", "1",
        "--root_path", os.path.join(HERE, "QDF", "dataset", "ETT-small") + os.sep,
        "--data_path", f"{ds}.csv", "--model_id", f"{ds}_96_{T}", "--model", "TQNet",
        "--data_id", ds, "--data", ds, "--features", "M", "--seq_len", "96", "--label_len", "48",
        "--pred_len", str(T), "--enc_in", "7", "--dec_in", "7", "--c_out", "7", "--factor", "3",
        "--des", "TQNet", "--learning_rate", str(hp["lr"]), "--lradj", "type1",
        "--train_epochs", "30", "--patience", "5", "--batch_size", "32", "--test_batch_size", "1",
        "--itr", "1", "--rec_lambda", "1.0", "--auxi_lambda", "0.0", "--reg_lambda", "0.0",
        "--auxi_batch_size", "64", "--fix_seed", str(seed),
        "--checkpoints", os.path.join(work, "ck"), "--results", os.path.join(work, "res"),
        "--test_results", os.path.join(work, "tr"), "--log_path", os.path.join(work, "log.txt"),
        "--rerun", "1", "--inner_lr", str(hp["lr_inner"]), "--meta_lr", str(meta_lr),
        "--meta_inner_steps", str(hp["meta_inner_steps"]), "--overlap_ratio", "0.0",
        "--num_tasks", str(hp["num_tasks"]), "--max_norm", "5.0", "--auxi_loss", "MSE",
        "--model_type", "linear", "--cycle", "24", "--use_revin", "1", "--dropout", "0.5",
        "--first_order", "1", "--warmup_steps", "300", "--num_workers", "0",
    ]
    return [PY, "-u", os.path.join(HERE, "cpu_launch.py"), os.path.join(HERE, "QDF"), work, "--"] + a


def tqnet_cmd(ds, T, seed, work):
    a = [
        "--is_training", "1", "--root_path", os.path.join(HERE, "TQNet", "dataset") + os.sep,
        "--data_path", f"{ds}.csv", "--model_id", f"{ds}_96_{T}", "--model", "TQNet", "--data", ds,
        "--features", "M", "--seq_len", "96", "--pred_len", str(T), "--enc_in", "7", "--cycle", "24",
        "--train_epochs", "30", "--patience", "5", "--dropout", "0.5", "--itr", "1",
        "--batch_size", "256", "--learning_rate", "0.001", "--random_seed", str(seed), "--num_workers", "0",
    ]
    return [PY, "-u", os.path.join(HERE, "cpu_launch.py"), os.path.join(HERE, "TQNet"), work, "--"] + a


METRIC = re.compile(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)")
TEST_N = re.compile(r"^test (\d+)\s*$", re.M)


def run_one(job, run_id=None, env_extra=None):
    arm, ds, T, seed = job
    rid = run_id or f"{arm}_{ds}_{T}_{seed}"
    work = os.path.join(RUNS, rid)
    os.makedirs(work, exist_ok=True)
    if arm == "TQNET_OFFICIAL":
        cmd = tqnet_cmd(ds, T, seed, work)
    else:
        cmd = qdf_cmd(ds, T, seed, work, sigma_identity=(arm == "SIGMA_I"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               **(env_extra or {}))
    t0 = time.time()
    with open(os.path.join(work, "stdout.txt"), "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    text = open(os.path.join(work, "stdout.txt"), encoding="utf-8", errors="replace").read()
    m = METRIC.findall(text)
    n = TEST_N.findall(text)
    stub = json.load(open(os.path.join(work, "stub_report.json"))) if os.path.exists(os.path.join(work, "stub_report.json")) else None
    rec = dict(run_id=rid, arm=arm, dataset=ds, pred_len=T, seed=seed, returncode=rc,
               test_mse=float(m[-1][0]) if m else None, test_mae=float(m[-1][1]) if m else None,
               test_windows=int(n[-1]) if n else None, seconds=round(time.time() - t0, 1),
               stub_runtime_use=(stub or {}).get("runtime_use"), cmd=cmd[3:],
               finished=datetime.datetime.now().isoformat(timespec="seconds"))
    return rec


def done_ids():
    if not os.path.exists(OUT):
        return set()
    ids = set()
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        if r["returncode"] == 0 and r["test_mse"] is not None:
            ids.add(r["run_id"])
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=16)
    args = p.parse_args()
    proto = json.load(open(os.path.join(HERE, "protocol.json"), encoding="utf-8"))
    g = proto["grid"]
    jobs = [(arm, ds, T, s) for T in sorted(g["pred_len"], reverse=True) for ds in g["datasets"]
            for s in g["seeds"] for arm in ("QDF", "SIGMA_I", "TQNET_OFFICIAL")]
    skip = done_ids()
    jobs = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}_{j[3]}" not in skip]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"{len(jobs)} runs to go, {len(skip)} already done", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        for fut in as_completed([ex.submit(run_one, j) for j in jobs]):
            rec = fut.result()
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{rec['run_id']}: rc={rec['returncode']} mse={rec['test_mse']} n={rec['test_windows']} {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
