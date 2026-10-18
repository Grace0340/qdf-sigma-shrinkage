"""Summarize runs_v8.jsonl under the frozen rules of protocol_v8.json -> results/summary_v8.json."""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

import run_grid_v8 as g8

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "summary_v8.json")


def load(path=g8.OUT):
    recs = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            recs[r["run_id"]] = r
    return recs


def ok(r):
    return r is not None and r["returncode"] == 0 and r["test_mse"] is not None and math.isfinite(r["test_mse"])


def select(recs, bb, ds, T, seed, prefix, grid, prefer=None):
    """Lowest best validation MSE; ties to `prefer`, then to the smaller grid value. Failed runs are ineligible."""
    cand = []
    for v in grid:
        r = recs.get(g8.run_id((f"{prefix}{v:g}", bb, ds, T, seed)))
        if ok(r) and r.get("best_val_mse") is not None:
            cand.append((r["best_val_mse"], 0 if v == prefer else 1, v, r))
    if not cand:
        return None, None
    _, _, v, r = min(cand, key=lambda c: (c[0], c[1], c[2]))
    return v, r


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), None
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def cell_rows(recs):
    rows, complete = {}, True
    for bb, c in g8.CELLS:
        ds, T = g8.split(c)
        sd = g8.seeds(bb, ds)
        need = [g8.run_id((a, bb, ds, T, s)) for s in sd for a in g8.arms(bb)]
        need.append(g8.run_id(("SIGMA_I_REP", bb, ds, T, g8.REP_SEED)))
        missing = [i for i in need if i not in recs]
        complete &= not missing
        g, sel = {k: [] for k in ("QDF_LR1", "QDF_VAL", "PLUGIN_VAL", "PDIAG_VAL", "REPLAY_K1", "REPLAY_VAL")}, \
            {k: [] for k in ("QDF_VAL", "PLUGIN_VAL", "PDIAG_VAL", "REPLAY_VAL")}
        failed = sorted(i for i in need if i in recs and not ok(recs[i]))
        for s in sd:
            base = recs.get(g8.run_id(("SIGMA_I", bb, ds, T, s)))
            if not ok(base):
                continue
            gain = lambda r: None if not ok(r) else (base["test_mse"] - r["test_mse"]) / base["test_mse"]
            picks = {
                "QDF_VAL": select(recs, bb, ds, T, s, "QDF_LR", g8.M_GRID[bb], prefer=1),
                "PLUGIN_VAL": select(recs, bb, ds, T, s, "PLUGIN_A", g8.A_GRID[bb]),
                "PDIAG_VAL": select(recs, bb, ds, T, s, "PDIAG_A", g8.A_GRID[bb]),
                "REPLAY_VAL": select(recs, bb, ds, T, s, "REPLAY_K", g8.K_GRID),
            }
            g["QDF_LR1"].append(gain(recs.get(g8.run_id(("QDF_LR1", bb, ds, T, s)))))
            g["REPLAY_K1"].append(gain(recs.get(g8.run_id(("REPLAY_K1", bb, ds, T, s)))))
            for k, (v, r) in picks.items():
                g[k].append(gain(r))
                sel[k].append(v)
        rep = recs.get(g8.run_id(("SIGMA_I_REP", bb, ds, T, g8.REP_SEED)))
        b0 = recs.get(g8.run_id(("SIGMA_I", bb, ds, T, g8.REP_SEED)))
        nu = abs(rep["test_mse"] - b0["test_mse"]) / b0["test_mse"] if ok(rep) and ok(b0) else None
        row = dict(backbone=bb, cell=c, seeds=sd, missing=len(missing), failed=failed, nu=nu, selected=sel, g=g)
        for k, v in g.items():
            vv = [x for x in v if x is not None]
            row[f"mean_{k}"], row[f"ci_{k}"] = ci(vv) if vv else (None, None)
            row[f"pos_{k}"] = sum(x > 0 for x in vv)
            row[f"n_{k}"] = len(vv)
        for ref in ("QDF_VAL", "QDF_LR1"):
            m = row[f"mean_{ref}"]
            for k in ("PLUGIN_VAL", "PDIAG_VAL", "REPLAY_K1", "REPLAY_VAL", "QDF_LR1"):
                row[f"R_{k}_vs_{ref}"] = row[f"mean_{k}"] / m if m and m > 0 and row[f"mean_{k}"] is not None else None
        d = [a - b for a, b in zip(g["PDIAG_VAL"], g["PLUGIN_VAL"]) if a is not None and b is not None]
        row["D_diag"], row["pos_D_diag"], row["neg_D_diag"] = (float(np.mean(d)) if d else None,
                                                               sum(x > 0 for x in d), sum(x < 0 for x in d))
        row["min_pos"] = 4 if len(sd) == 5 else 3
        q = row["mean_QDF_LR1"]
        row["q80_entry_rule"] = bool(q is not None and q >= max(0.005, 3 * (nu or 0)) and row["pos_QDF_LR1"] >= row["min_pos"])
        rows[c if bb == "TQNet" else f"{bb}/{c}"] = row
    return complete, rows


def verdicts(rows):
    n, k = len(rows), math.ceil(2 * len(rows) / 3)
    R = lambda r, a: r.get(f"R_{a}_vs_QDF_VAL")
    rec = lambda r, a: R(r, a) is not None and R(r, a) >= 0.8 and r[f"pos_{a}"] >= r["min_pos"]
    fail = lambda r, a: R(r, a) is None or R(r, a) < 0.5
    cnt = lambda f: sum(bool(f(r)) for r in rows.values())
    v = {"n": n, "k": k}
    v["Q8.1"] = ("SURROGATE_MATCHES_TUNED_QDF" if cnt(lambda r: rec(r, "PLUGIN_VAL")) >= k else
                 "TUNED_QDF_BETTER" if cnt(lambda r: fail(r, "PLUGIN_VAL")) >= k else "MIXED")
    v["Q8.2"] = ("DIAG_BETTER" if cnt(lambda r: r["D_diag"] is not None and r["D_diag"] > 0 and r["pos_D_diag"] >= r["min_pos"]) >= k else
                 "DIAG_WORSE" if cnt(lambda r: r["D_diag"] is not None and r["D_diag"] < 0 and r["neg_D_diag"] >= r["min_pos"]) >= k else
                 "NO_DIFFERENCE")
    for a, key in (("REPLAY_K1", "Q8.3_K1"), ("REPLAY_VAL", "Q8.3_VAL")):
        v[key] = ("REPLAY_RECOVERS" if cnt(lambda r: rec(r, a)) >= k else
                  "REPLAY_FAILS" if cnt(lambda r: fail(r, a)) >= k else "PARTIAL")
    v["counts"] = {f"recovers_{a}": cnt(lambda r, a=a: rec(r, a)) for a in ("PLUGIN_VAL", "PDIAG_VAL", "REPLAY_K1", "REPLAY_VAL")}
    v["counts"]["q80_entry_rule_QDF_LR1"] = cnt(lambda r: r["q80_entry_rule"])
    diffs = [r["mean_PLUGIN_VAL"] - r["mean_QDF_VAL"] for r in rows.values()
             if r["mean_PLUGIN_VAL"] is not None and r["mean_QDF_VAL"] is not None]
    v["secondary_wilcoxon_PLUGIN_VAL_minus_QDF_VAL"] = dict(
        mean=float(np.mean(diffs)) if diffs else None,
        p=float(stats.wilcoxon(diffs).pvalue) if len(diffs) >= 2 and any(diffs) else None, n=len(diffs))
    return v


def main():
    recs = load()
    complete, rows = cell_rows(recs)
    out = dict(protocol_id=g8.PROTO["protocol_id"], complete=complete, n_records=len(recs),
               verdicts=verdicts(rows) if complete else "INCOMPLETE", cells=rows)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    for c, r in rows.items():
        f = lambda x: "  -  " if x is None else f"{100 * x:+.2f}"
        print(f"{c:22s} QDF1 {f(r['mean_QDF_LR1'])} QDFv {f(r['mean_QDF_VAL'])} PLv {f(r['mean_PLUGIN_VAL'])} "
              f"PDv {f(r['mean_PDIAG_VAL'])} RP1 {f(r['mean_REPLAY_K1'])} RPv {f(r['mean_REPLAY_VAL'])} "
              f"missing {r['missing']} failed {len(r['failed'])}")
    print("complete" if complete else "INCOMPLETE", json.dumps(out["verdicts"], default=str))


if __name__ == "__main__":
    sys.exit(main())
