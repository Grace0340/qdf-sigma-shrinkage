"""Run the v9 fresh-seed replication (protocol_v9.json): the v8 grid with seeds 2026-2028, on machine 697. Resumable.

Arms, grids, commands, selection and failure handling are those of run_grid_v8.py; only seeds, output paths
and the platform record change. --xm runs the cross-machine check instead: SIGMA_I and QDF_LR1 at seed 2021 in
the eight cells, under the v8 run ids, into results/runs_v9_xm.jsonl and runs_v9_xm/.
"""
import json
import os
import platform
import sys

import torch

import run_grid_v8 as g8

HERE = g8.g1.HERE
g8.PROTO = json.load(open(os.path.join(HERE, "protocol_v9.json"), encoding="utf-8"))
g8.OUT = os.path.join(HERE, "results", "runs_v9.jsonl")
g8.RUNS = os.path.join(HERE, "runs_v9")
g8.PLATFORM = dict(name="autodl-gpu-697", host=platform.node(), python=platform.python_version(),
                   torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
SEEDS = [2026, 2027, 2028]
XM_SEED = 2021
g8.REP_SEED = SEEDS[0]
g8.seeds = lambda bb, ds: SEEDS


def xm_jobs():
    jobs = []
    for bb, c in g8.CELLS:
        ds, T = g8.split(c)
        jobs += [(arm, bb, ds, T, XM_SEED) for arm in ("SIGMA_I", "QDF_LR1")]
    return sorted(jobs, key=lambda j: (-g8.mem(j), j[1], j[2], j[3], j[0]))


if __name__ == "__main__":
    if "--xm" in sys.argv:
        sys.argv.remove("--xm")
        g8.OUT = os.path.join(HERE, "results", "runs_v9_xm.jsonl")
        g8.RUNS = os.path.join(HERE, "runs_v9_xm")
        g8.all_jobs = xm_jobs
    sys.exit(g8.main())
