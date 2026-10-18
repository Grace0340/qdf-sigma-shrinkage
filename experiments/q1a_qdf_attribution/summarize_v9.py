"""Summarize runs_v9.jsonl under the frozen rules of protocol_v9.json -> results/summary_v9.json.

The per-cell quantities and verdict rules are those of summarize_v8.py (with 3 seeds, min_pos = 3 of 3);
Q9.0 counts the cells in which released QDF passes the original analysis-cell rule on the fresh seeds.
The cross-machine check compares runs_v9_xm.jsonl with runs_v8.jsonl run by run.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

import run_grid_v9 as g9
import summarize_v8 as s8

g8 = g9.g8
OUT = os.path.join(g9.HERE, "results", "summary_v9.json")
V8 = os.path.join(g9.HERE, "results", "runs_v8.jsonl")
XM = os.path.join(g9.HERE, "results", "runs_v9_xm.jsonl")


def wilcoxon(rows, a, b):
    d = [r[f"mean_{a}"] - r[f"mean_{b}"] for r in rows.values()
         if r[f"mean_{a}"] is not None and r[f"mean_{b}"] is not None]
    return dict(mean=float(np.mean(d)) if d else None, n=len(d),
                p=float(stats.wilcoxon(d).pvalue) if len(d) >= 2 and any(d) else None)


def cross_machine():
    v8, xm = s8.load(V8), s8.load(XM)
    out = []
    for rid, r in sorted(xm.items()):
        b = v8.get(rid)
        if b is None or not s8.ok(r) or not s8.ok(b):
            out.append(dict(run_id=rid, comparable=False))
            continue
        out.append(dict(run_id=rid, comparable=True, mse_v8=b["test_mse"], mse_697=r["test_mse"],
                        rel_diff=abs(r["test_mse"] - b["test_mse"]) / b["test_mse"],
                        identical=r["test_mse"] == b["test_mse"] and r.get("best_val_mse") == b.get("best_val_mse")))
    cmp_ = [x for x in out if x["comparable"]]
    return dict(n=len(out), n_comparable=len(cmp_), n_identical=sum(x["identical"] for x in cmp_),
                max_rel_diff=max((x["rel_diff"] for x in cmp_), default=None), runs=out)


def main():
    recs = s8.load(g8.OUT)
    complete, rows = s8.cell_rows(recs)
    v = s8.verdicts(rows) if complete else "INCOMPLETE"
    if complete:
        v = {("Q9" + k[2:] if k.startswith("Q8.") else k): x for k, x in v.items()}
        v["Q9.0_cells_passing_analysis_rule"] = v["counts"]["q80_entry_rule_QDF_LR1"]
        v["secondary_wilcoxon_REPLAY_K1_minus_QDF_VAL"] = wilcoxon(rows, "REPLAY_K1", "QDF_VAL")
        v["secondary_wilcoxon_PDIAG_VAL_minus_QDF_VAL"] = wilcoxon(rows, "PDIAG_VAL", "QDF_VAL")
    out = dict(protocol_id=g8.PROTO["protocol_id"], complete=complete, n_records=len(recs), verdicts=v,
               cross_machine=cross_machine() if os.path.exists(XM) else None, cells=rows)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    for c, r in rows.items():
        f = lambda x: "  -  " if x is None else f"{100 * x:+.2f}"
        print(f"{c:22s} QDF1 {f(r['mean_QDF_LR1'])} ({r['pos_QDF_LR1']}) QDFv {f(r['mean_QDF_VAL'])} "
              f"PLv {f(r['mean_PLUGIN_VAL'])} ({r['pos_PLUGIN_VAL']}) PDv {f(r['mean_PDIAG_VAL'])} "
              f"RP1 {f(r['mean_REPLAY_K1'])} ({r['pos_REPLAY_K1']}) RPv {f(r['mean_REPLAY_VAL'])} "
              f"missing {r['missing']} failed {len(r['failed'])}")
    if out["cross_machine"]:
        x = out["cross_machine"]
        print(f"cross-machine: {x['n_identical']}/{x['n_comparable']} identical, max rel diff {x['max_rel_diff']}")
    print("complete" if complete else "INCOMPLETE", json.dumps(v, default=str))


if __name__ == "__main__":
    sys.exit(main())
