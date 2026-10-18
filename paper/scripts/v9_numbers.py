# Numbers and table rows of protocol v9 (fresh seeds 2026-2028) for paper/main.tex and paper/supplementary.tex.
# Reads v9_pulled/results/runs_v9.jsonl, runs_v9_xm.jsonl, summary_v9.json and results/runs_v8.jsonl;
# no training or model evaluation. Recomputes the frozen v9 statistics from the run records, checks them
# against summary_v9.json, and adds the two post hoc analyses (validation choice between the surrogates,
# v8+v9 pooled seeds), which are descriptive only.
#   python paper/scripts/v9_numbers.py
import json
import math
import os
import re

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
EXP = os.environ.get("Q1A_DIR", os.path.join(PAPER, "..", "experiments", "q1a_qdf_attribution"))
RES9 = os.path.join(EXP, "v9_pulled", "results")
OUT = os.path.join(PAPER, "generated")

CELLS = [("TQNet", "ETTh1", 96), ("TQNet", "ETTh2", 96), ("TQNet", "ETTh2", 720), ("TQNet", "ETTm1", 96),
         ("TQNet", "ETTm2", 96), ("TQNet", "ETTm2", 720), ("TQNet", "Weather", 720), ("iTransformer", "ECL", 96)]
M_GRID = {"TQNet": [0.125, 0.25, 0.5, 1, 2, 4, 8], "iTransformer": [0.25, 0.5, 1, 2, 4]}
A_GRID = {"TQNet": [0.01, 0.03, 0.1, 0.25, 0.5, 1, 2], "iTransformer": [0.01, 0.03, 0.1, 0.3, 1]}
K_GRID = [0.5, 1, 2]
SEEDS9 = [2026, 2027, 2028]


def seeds8(bb, ds):
    return [2021, 2022, 2023, 2024, 2025] if ds.startswith("ETTh") or bb == "iTransformer" else [2021, 2022, 2023]


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        recs[r["run_id"]] = r
    return recs


def ok(r):
    return (r is not None and r["returncode"] == 0 and r.get("test_mse") is not None and math.isfinite(r["test_mse"])
            and r.get("best_val_mse") is not None and math.isfinite(r["best_val_mse"]))


rid = lambda arm, bb, ds, T, s: f"{arm}_{bb}_{ds}_{T}_{s}"
ckey = lambda bb, ds, T: f"{ds}-{T}" if bb == "TQNet" else f"{bb}/{ds}-{T}"


def select(recs, bb, ds, T, s, prefix, grid, prefer=None):
    cand = [(r["best_val_mse"], 0 if v == prefer else 1, v, r) for v in grid
            for r in [recs.get(rid(f"{prefix}{v:g}", bb, ds, T, s))] if ok(r)]
    return min(cand, key=lambda c: c[:3])[2:] if cand else (None, None)


def cell(recs, bb, ds, T, seeds):
    g = {k: [] for k in ("QDF_LR1", "QDF_VAL", "PLUGIN_VAL", "PDIAG_VAL", "REPLAY_K1", "SURR_VAL")}
    sel = {"QDF_VAL": [], "PLUGIN_VAL": []}
    for s in seeds:
        base = recs[rid("SIGMA_I", bb, ds, T, s)]
        assert ok(base)
        gain = lambda r: (base["test_mse"] - r["test_mse"]) / base["test_mse"]
        mq, rq = select(recs, bb, ds, T, s, "QDF_LR", M_GRID[bb], prefer=1)
        ap, rp = select(recs, bb, ds, T, s, "PLUGIN_A", A_GRID[bb])
        _, rd = select(recs, bb, ds, T, s, "PDIAG_A", A_GRID[bb])
        rr = recs[rid("REPLAY_K1", bb, ds, T, s)]
        g["QDF_LR1"].append(gain(recs[rid("QDF_LR1", bb, ds, T, s)]))
        g["QDF_VAL"].append(gain(rq))
        g["PLUGIN_VAL"].append(gain(rp))
        g["PDIAG_VAL"].append(gain(rd))
        g["REPLAY_K1"].append(gain(rr))
        g["SURR_VAL"].append(gain(min((rp, rr), key=lambda x: x["best_val_mse"])))
        sel["QDF_VAL"].append(mq)
        sel["PLUGIN_VAL"].append(ap)
    out = {k: dict(mean=float(np.mean(v)), pos=int(sum(x > 0 for x in v)), n=len(v), g=v) for k, v in g.items()}
    out["sel"] = sel
    return out


R9 = load(os.path.join(RES9, "runs_v9.jsonl"))
R8 = load(os.path.join(EXP, "results", "runs_v8.jsonl"))
S9 = json.load(open(os.path.join(RES9, "summary_v9.json"), encoding="utf-8"))
XM = load(os.path.join(RES9, "runs_v9_xm.jsonl"))
assert len(R9) == 590 and all(ok(r) for r in R9.values())

C9, CP = {}, {}
for bb, ds, T in CELLS:
    k = ckey(bb, ds, T)
    C9[k] = cell(R9, bb, ds, T, SEEDS9)
    CP[k] = cell({**R8, **R9}, bb, ds, T, seeds8(bb, ds) + SEEDS9)
    s = S9["cells"][k]
    for a in ("QDF_LR1", "QDF_VAL", "PLUGIN_VAL", "PDIAG_VAL", "REPLAY_K1"):
        assert abs(C9[k][a]["mean"] - s[f"mean_{a}"]) < 1e-12 and C9[k][a]["pos"] == s[f"pos_{a}"], (k, a)
keys = list(C9)

# cross-machine check
xm_same = sum(R8[i]["test_mse"] == r["test_mse"] and R8[i]["best_val_mse"] == r["best_val_mse"] for i, r in XM.items())
assert xm_same == len(XM) == 16


def R(c, a, ref="QDF_VAL"):
    return c[a]["mean"] / c[ref]["mean"]


def recovers(c, a, min_pos):
    return R(c, a) >= 0.8 and c[a]["pos"] >= min_pos


mp9 = lambda c: 3
mpP = lambda c: c["QDF_VAL"]["n"] - 1
rec = lambda C, a, mp: sum(recovers(C[k], a, mp(C[k])) for k in keys)
gap = lambda C, a: [C[k][a]["mean"] - C[k]["QDF_VAL"]["mean"] for k in keys]
wil = lambda d: float(stats.wilcoxon(d).pvalue)
agg = lambda C, a: sum(C[k][a]["mean"] for k in keys) / sum(C[k]["QDF_VAL"]["mean"] for k in keys)
entry = [k for k in keys if C9[k]["QDF_LR1"]["pos"] == 3 and C9[k]["QDF_LR1"]["mean"] >= 0.005]
v = S9["verdicts"]
assert rec(C9, "PLUGIN_VAL", mp9) == v["counts"]["recovers_PLUGIN_VAL"]
assert rec(C9, "REPLAY_K1", mp9) == v["counts"]["recovers_REPLAY_K1"]
assert len(entry) == v["Q9.0_cells_passing_analysis_rule"]


def pct(x, d=2):
    s = f"{100 * x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def neg(s):
    return re.sub(r"(?<![\w.$])-(?=\d)", r"\\ensuremath{-}", s)


def sc(verdict):
    return "\\textsc{" + verdict.lower().replace("_", "\\_") + "}"


def pval(p):
    return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"


C8 = {ckey(bb, ds, T): cell(R8, bb, ds, T, seeds8(bb, ds)) for bb, ds, T in CELLS}
mp8 = lambda c: 4 if c["QDF_VAL"]["n"] == 5 else 3
assert rec(C8, "PLUGIN_VAL", mp8) == 6 and rec(C8, "REPLAY_K1", mp8) == 6
assert set(keys) - set(entry) == {"ETTm1-96", "Weather-720"}, entry
rec_in = lambda a, ref, cells: sum(C9[k][a]["mean"] / C9[k][ref]["mean"] >= 0.8 and C9[k][a]["pos"] == 3 for k in cells)

plug_good = [k for k in keys if recovers(C9[k], "PLUGIN_VAL", 3)]
rep_good = [k for k in keys if recovers(C9[k], "REPLAY_K1", 3)]
mac = {
    "NineRuns": len(R9), "NineXM": len(XM), "NineEntry": len(entry),
    "NineQPos": sum(C9[k]["QDF_LR1"]["mean"] > 0 for k in keys),
    "NineQAllPos": sum(C9[k]["QDF_LR1"]["pos"] == 3 for k in keys),
    "NineRecPlug": rec(C9, "PLUGIN_VAL", mp9), "NineRecRep": rec(C9, "REPLAY_K1", mp9),
    "NineVOne": sc(v["Q9.1"]), "NineVTwo": sc(v["Q9.2"]), "NineVThreeA": sc(v["Q9.3_K1"]),
    "NineDPlugVal": pct(np.mean(gap(C9, "PLUGIN_VAL"))), "NineDPlugValP": pval(wil(gap(C9, "PLUGIN_VAL"))),
    "NineDRepVal": pct(np.mean(gap(C9, "REPLAY_K1"))), "NineDRepValP": pval(wil(gap(C9, "REPLAY_K1"))),
    "NineAggPlug": f"{100 * agg(C9, 'PLUGIN_VAL'):.0f}", "NineAggRep": f"{100 * agg(C9, 'REPLAY_K1'):.0f}",
    "NinePlugGoodMin": f"{min(R(C9[k], 'PLUGIN_VAL') for k in plug_good):.2f}",
    "NinePlugGoodMax": f"{max(R(C9[k], 'PLUGIN_VAL') for k in plug_good):.2f}",
    "NineRepGoodMin": f"{min(R(C9[k], 'REPLAY_K1') for k in rep_good):.2f}",
    "NineRepGoodMax": f"{max(R(C9[k], 'REPLAY_K1') for k in rep_good):.2f}",
    "NineHtwoQOne": pct(C9["ETTh2-96"]["QDF_LR1"]["mean"]), "NineHtwoQVal": pct(C9["ETTh2-96"]["QDF_VAL"]["mean"]),
    "NineHtwoSelM": "/".join(f"{m:g}" for m in C9["ETTh2-96"]["sel"]["QDF_VAL"]),
    "NineSurrRec": rec(C9, "SURR_VAL", mp9), "NineSurrD": pct(np.mean(gap(C9, "SURR_VAL"))),
    "PoolRecPlug": rec(CP, "PLUGIN_VAL", mpP), "PoolRecRep": rec(CP, "REPLAY_K1", mpP),
    "PoolRecSurr": rec(CP, "SURR_VAL", mpP),
    "PoolDPlugVal": pct(np.mean(gap(CP, "PLUGIN_VAL"))), "PoolDRepVal": pct(np.mean(gap(CP, "REPLAY_K1"))),
    "NineRecPlugEntry": rec_in("PLUGIN_VAL", "QDF_VAL", entry), "NineRecRepEntry": rec_in("REPLAY_K1", "QDF_VAL", entry),
    "NineRecPlugOne": rec_in("PLUGIN_VAL", "QDF_LR1", keys), "NineRecRepOne": rec_in("REPLAY_K1", "QDF_LR1", keys),
    "EightAggPlug": f"{100 * agg(C8, 'PLUGIN_VAL'):.0f}", "EightAggRep": f"{100 * agg(C8, 'REPLAY_K1'):.0f}",
    "NineWthQOne": pct(C9["Weather-720"]["QDF_LR1"]["mean"]),
    "NineHtwoLongQOne": pct(C9["ETTh2-720"]["QDF_LR1"]["mean"]),
    "NineHtwoLongQVal": pct(C9["ETTh2-720"]["QDF_VAL"]["mean"]),
    "NineSecMin": f"{min(r['seconds'] for r in R9.values()):.0f}",
    "NineSecMax": f"{max(r['seconds'] for r in R9.values()):.0f}",
    "NineSecMed": f"{np.median([r['seconds'] for r in R9.values()]):.0f}",
    "NineSecSum": f"{sum(r['seconds'] for r in R9.values()) / 3600:.0f}",
}
assert set(plug_good) == {"ETTh1-96", "ETTh2-720", "Weather-720"}, plug_good
open(os.path.join(OUT, "v9.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v9_numbers.py -- do not edit by hand.\n"
    + "".join(neg(f"\\newcommand{{\\num{k}}}{{{x}}}\n") for k, x in mac.items()))
json.dump(dict(macros=mac, entry=entry, plug_good=plug_good, rep_good=rep_good,
               cells={k: {a: {kk: vv for kk, vv in C9[k][a].items()} for a in C9[k] if a != "sel"} | {"sel": C9[k]["sel"]}
                      for k in keys},
               pooled={k: {a: CP[k][a]["mean"] for a in CP[k] if a != "sel"} for k in keys}),
          open(os.path.join(OUT, "v9.json"), "w", encoding="utf-8"), indent=1, default=float)

# Table I: v8 columns (rel., val., R_val, R_rep) from v8_table1_rows.tex, plus v9 (val., R_val, R_rep)
v8rows = [l for l in open(os.path.join(OUT, "v8_table1_rows.tex"), encoding="utf-8") if "&" in l]
rows = []
for k, line in zip(keys, v8rows):
    p = [x.strip() for x in line.replace("\\\\", "").split("&")]
    c = C9[k]
    rows.append(f"{p[0]} & {p[1]} & {p[2]} & {p[3]} & {p[5]} & {pct(c['QDF_VAL']['mean'])} & "
                f"{R(c, 'PLUGIN_VAL'):.2f}\\,({c['PLUGIN_VAL']['pos']}/3) & {R(c, 'REPLAY_K1'):.2f}\\,({c['REPLAY_K1']['pos']}/3) \\\\")
open(os.path.join(OUT, "v89_table1_rows.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v9_numbers.py -- do not edit by hand.\n" + neg("\n".join(rows)) + "\n")


# Extended results: all v9 arms with 95% t-intervals and selected m
def cc(c, a):
    x = np.asarray(c[a]["g"])
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f"{pct(x.mean())}\\,$\\pm$\\,{pct(h)}\\,({c[a]['pos']})"


rows = [f"{p[0]} & {cc(C9[k], 'QDF_LR1')} & {cc(C9[k], 'QDF_VAL')} & {cc(C9[k], 'PLUGIN_VAL')} & {cc(C9[k], 'PDIAG_VAL')} & "
        f"{cc(C9[k], 'REPLAY_K1')} & {'/'.join(f'{m:g}' for m in C9[k]['sel']['QDF_VAL'])} \\\\"
        for k, p in zip(keys, [[x.strip() for x in l.split("&")] for l in v8rows])]
open(os.path.join(OUT, "v9_supp_rows.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v9_numbers.py -- do not edit by hand.\n" + neg("\n".join(rows)) + "\n")

# Extended results: per-setting controls formerly in Table I (v8 R_diag and the earlier R_Cy, R_rot) plus v9 R_diag
rows = []
for k, line in zip(keys, v8rows):
    p = [x.strip() for x in line.replace("\\\\", "").split("&")]
    rows.append(f"{p[0]} & {p[4]} & {R(C9[k], 'PDIAG_VAL'):.2f} & {p[6]} & {p[7]} \\\\")
open(os.path.join(OUT, "controls_rows.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v9_numbers.py -- do not edit by hand.\n" + neg("\n".join(rows)) + "\n")
for k, x in mac.items():
    print(f"{k:16s} {x}")
