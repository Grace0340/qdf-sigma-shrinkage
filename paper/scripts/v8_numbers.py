# Numbers, table rows and figure data of protocol v8 for paper/main.tex and paper/supplementary.tex.
# Reads results/runs_v8.jsonl, results/summary_v8.json, results/extras_v8.json and the saved Sigma files of
# v8_pulled/runs_v8; no training or model evaluation. Recomputes every v8 statistic from the run records and
# checks it against summary_v8.json.
#   python paper/scripts/v8_numbers.py
import glob
import json
import math
import os
import pickle
import re

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
# Q1A_DIR: a short path to experiments/q1a_qdf_attribution, needed on Windows where the run directories
# exceed MAX_PATH below the paper folder.
EXP = os.environ.get("Q1A_DIR", os.path.join(PAPER, "..", "experiments", "q1a_qdf_attribution"))
RES = os.path.join(EXP, "results")
RUNS = os.path.join(EXP, "v8_pulled", "runs_v8")
OUT = os.path.join(PAPER, "generated")
EPS = 1e-6

CELLS = [("TQNet", "ETTh1", 96), ("TQNet", "ETTh2", 96), ("TQNet", "ETTh2", 720), ("TQNet", "ETTm1", 96),
         ("TQNet", "ETTm2", 96), ("TQNet", "ETTm2", 720), ("TQNet", "Weather", 720), ("iTransformer", "ECL", 96)]
M_GRID = {"TQNet": ["0.125", "0.25", "0.5", "1", "2", "4", "8"], "iTransformer": ["0.25", "0.5", "1", "2", "4"]}
A_GRID = {"TQNet": ["0.01", "0.03", "0.1", "0.25", "0.5", "1", "2"], "iTransformer": ["0.01", "0.03", "0.1", "0.3", "1"]}
K_GRID = ["0.5", "1", "2"]


def ckey(bb, ds, T):
    return f"{ds}-{T}" if bb == "TQNet" else f"{bb}/{ds}-{T}"


def load_runs(fn):
    recs = {}
    with open(os.path.join(RES, fn), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["run_id"]] = r
    return recs


def ok(r):
    return (r is not None and r["returncode"] == 0 and r.get("test_mse") is not None and math.isfinite(r["test_mse"])
            and r.get("best_val_mse") is not None and math.isfinite(r["best_val_mse"]))


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(m), float(h)


R8 = load_runs("runs_v8.jsonl")
S8 = json.load(open(os.path.join(RES, "summary_v8.json"), encoding="utf-8"))


def rid(arm, bb, ds, T, s):
    return f"{arm}_{bb}_{ds}_{T}_{s}"


def select(arm_prefix, grid, bb, ds, T, s, prefer=None):
    cand = []
    for v in grid:
        r = R8.get(rid(f"{arm_prefix}{v}", bb, ds, T, s))
        if ok(r):
            cand.append((r["best_val_mse"], 0 if v == prefer else 1, float(v), v, r))
    cand.sort(key=lambda c: c[:3])
    return cand[0][3], cand[0][4]


rep = {"cells": {}}
for bb, ds, T in CELLS:
    key = ckey(bb, ds, T)
    seeds = S8["cells"][key]["seeds"]
    base = {s: R8[rid("SIGMA_I", bb, ds, T, s)]["test_mse"] for s in seeds}
    g = lambda r, s: (base[s] - r["test_mse"]) / base[s]
    arms = {}
    for m in M_GRID[bb]:
        arms[f"QDF_LR{m}"] = [g(R8[rid(f"QDF_LR{m}", bb, ds, T, s)], s) for s in seeds]
    for a in A_GRID[bb]:
        arms[f"PLUGIN_A{a}"] = [g(R8[rid(f"PLUGIN_A{a}", bb, ds, T, s)], s) for s in seeds]
        arms[f"PDIAG_A{a}"] = [g(R8[rid(f"PDIAG_A{a}", bb, ds, T, s)], s) for s in seeds]
    for k in K_GRID:
        arms[f"REPLAY_K{k}"] = [g(R8[rid(f"REPLAY_K{k}", bb, ds, T, s)], s) for s in seeds]
    sel = {}
    for name, pre, grid, prefer in (("QDF_VAL", "QDF_LR", M_GRID[bb], "1"), ("PLUGIN_VAL", "PLUGIN_A", A_GRID[bb], None),
                                    ("PDIAG_VAL", "PDIAG_A", A_GRID[bb], None), ("REPLAY_VAL", "REPLAY_K", K_GRID, None)):
        vals, gs = [], []
        for s in seeds:
            v, r = select(pre, grid, bb, ds, T, s, prefer)
            vals.append(v)
            gs.append(g(r, s))
        arms[name] = gs
        sel[name] = vals
    # cross-check against the frozen summary
    for name in ("QDF_LR1", "QDF_VAL", "PLUGIN_VAL", "PDIAG_VAL", "REPLAY_K1", "REPLAY_VAL"):
        assert np.allclose(arms[name], S8["cells"][key]["g"][name], atol=1e-12), (key, name)
    for name, v in sel.items():
        assert [float(x) for x in v] == [float(x) for x in S8["cells"][key]["selected"][name]], (key, name)
    nu = S8["cells"][key]["nu"]
    rep["cells"][key] = dict(bb=bb, ds=ds, T=T, seeds=seeds, nu=nu, sel=sel,
                             mse_I=float(np.mean(list(base.values()))),
                             arms={k: dict(g=v, mean=float(np.mean(v)), ci=tci(v)[1], pos=int(sum(x > 0 for x in v)),
                                           n=len(v)) for k, v in arms.items()})

# replay time and S0 time from the run directories
rt, s0t = [], []
for f in glob.glob(os.path.join(RUNS, "REPLAY_K*", "res", "*", "sigma_target.json")):
    j = json.load(open(f))
    rt.append(j["replay_seconds"])
    s0t.append(j["s0_seconds"])
rep["replay_seconds"] = [min(rt), max(rt), len(rt)]
rep["s0_seconds"] = [min(s0t), max(s0t)]
meta = [R8[rid("QDF_LR1", bb, ds, T, s)]["meta_phase_seconds"] for bb, ds, T in CELLS
        for s in rep["cells"][ckey(bb, ds, T)]["seeds"]]
rep["qdf_meta_phase_seconds"] = [min(meta), max(meta)]


# shape of the zero-parameter replay against the learned Sigma of the same seed (QDF_LR1)
def learned_sigma(path):
    class Holder:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, st):
            self.__dict__.update(st)

    class Unp(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith(("torch", "collections", "builtins", "_codecs", "numpy")):
                return super().find_class(mod, name)
            return Holder

    import torch
    shim = type("shim", (), {"Unpickler": Unp, "load": pickle.load})
    A = torch.load(path, weights_only=False, map_location="cpu", pickle_module=shim)
    P = A._parameters["L_param"].detach().double().numpy()
    L = np.tril(P) + EPS * np.eye(P.shape[0])
    return L @ L.T


def nshape(M):
    return M * (M.shape[0] / np.trace(M))


shape = []
for bb, ds, T in CELLS:
    for s in rep["cells"][ckey(bb, ds, T)]["seeds"]:
        a = glob.glob(os.path.join(RUNS, rid("QDF_LR1", bb, ds, T, s), "ck", "*", "A.pth"))
        p = glob.glob(os.path.join(RUNS, rid("REPLAY_K1", bb, ds, T, s), "res", "*", "sigma_target.npy"))
        q = glob.glob(os.path.join(RUNS, rid("PLUGIN_A1", bb, ds, T, s), "res", "*", "sigma_target.npy"))
        if not (a and p and q):
            shape.append(dict(cell=ckey(bb, ds, T), seed=s, missing=True))
            continue
        Sl, Sr = learned_sigma(a[0]), np.load(p[0]).astype(np.float64)
        I = np.eye(T)
        e = np.linalg.norm(nshape(Sl) - nshape(Sr)) / np.linalg.norm(nshape(Sl) - I)
        shape.append(dict(cell=ckey(bb, ds, T), seed=s, err=float(e),
                          dev_learned=float(np.linalg.norm(nshape(Sl) - I) / math.sqrt(T)),
                          dev_replay=float(np.linalg.norm(nshape(Sr) - I) / math.sqrt(T))))
rep["replay_k1_vs_learned"] = shape

# extras (49 learned Sigma of v4, v5, v7) grouped by setting
EX = json.load(open(os.path.join(RES, "extras_v8.json"), encoding="utf-8"))
groups = {}
for r in EX:
    groups.setdefault((r["backbone"], r["dataset"], r["T"]), []).append(r)
rep["extras"] = []
for (bb, ds, T), rs in groups.items():
    f = lambda key: [x[key]["err"] for x in rs]
    rep["extras"].append(dict(bb=bb, ds=ds, T=T, n=len(rs), dev=[min(x["dev_from_I"] for x in rs),
                                                                 max(x["dev_from_I"] for x in rs)],
                              fo=[min(f("first_order")), max(f("first_order"))],
                              sq=[min(f("sqrt_family")), max(f("sqrt_family"))],
                              rp=[min(f("replay")), max(f("replay"))],
                              ratio=[min(x["diag_over_off_slope"] for x in rs), max(x["diag_over_off_slope"] for x in rs)],
                              replay_best=all(x["replay"]["err"] < min(x["first_order"]["err"], x["sqrt_family"]["err"])
                                              for x in rs)))

# cross-platform: SIGMA_I and QDF of the same cell and seed in v4 (ETTh, CPU), v5 and v7 (GPU, machine 703)
old = {}
for fn, default_bb in (("runs_v3_autodl.jsonl", "TQNet"), ("runs_v4_autodl.jsonl", "TQNet"), ("runs_v5.jsonl", "TQNet"),
                       ("runs_v7.jsonl", None)):
    plat = "cpu" if "autodl.jsonl" in fn else "gpu703"
    for r in load_runs(fn).values():
        if (r["arm"] in ("SIGMA_I", "QDF") and r["returncode"] == 0 and r.get("test_mse") is not None
                and math.isfinite(r["test_mse"])):
            bb = r.get("backbone") or default_bb
            old[(r["arm"], bb, r["dataset"], r["pred_len"], r["seed"])] = (r["test_mse"], plat, fn)
xp = {}
for bb, ds, T in CELLS:
    for s in rep["cells"][ckey(bb, ds, T)]["seeds"]:
        for arm_old, arm_new in (("SIGMA_I", "SIGMA_I"), ("QDF", "QDF_LR1")):
            o = old.get((arm_old, bb, ds, T, s))
            if o:
                new = R8[rid(arm_new, bb, ds, T, s)]["test_mse"]
                xp.setdefault(o[1], []).append(dict(cell=ckey(bb, ds, T), seed=s, arm=arm_new,
                                                    rel=(new - o[0]) / o[0], src=o[2]))
rep["cross_platform"] = {k: dict(n=len(v), max_abs=max(abs(x["rel"]) for x in v),
                                 cells=sorted({x["cell"] for x in v})) for k, v in xp.items()}

# ---- aggregate statistics --------------------------------------------------
C = rep["cells"]
keys = list(C)
mean = lambda k, arm: C[k]["arms"][arm]["mean"]
R = lambda k, arm, den: mean(k, arm) / mean(k, den)
min_pos = lambda k: 4 if len(C[k]["seeds"]) == 5 else 3
rec = lambda arm, den: sum(R(k, arm, den) >= 0.8 and C[k]["arms"][arm]["pos"] >= min_pos(k) for k in keys)
agg = {}
for arm in ("PLUGIN_VAL", "PDIAG_VAL", "REPLAY_K1", "REPLAY_VAL"):
    for den in ("QDF_VAL", "QDF_LR1"):
        agg[f"rec_{arm}_{den}"] = rec(arm, den)
for arm in ("PLUGIN_VAL", "REPLAY_K1"):
    for den in ("QDF_VAL", "QDF_LR1"):
        d = np.array([mean(k, arm) - mean(k, den) for k in keys])
        agg[f"diff_{arm}_{den}"] = dict(mean=float(d.mean()), neg=int((d < 0).sum()),
                                        p=float(stats.wilcoxon(d).pvalue))
d = np.array([mean(k, "QDF_VAL") - mean(k, "QDF_LR1") for k in keys])
agg["diff_QDF_VAL_QDF_LR1"] = dict(mean=float(d.mean()), neg=int((d < 0).sum()), p=float(stats.wilcoxon(d).pvalue))
agg["qdf_val_selected_m1"] = sum(x == "1" for k in keys for x in C[k]["sel"]["QDF_VAL"])
agg["qdf_val_n"] = sum(len(C[k]["seeds"]) for k in keys)
# oracle: best single grid value on test data
for pre, grid in (("PLUGIN_A", A_GRID), ("QDF_LR", M_GRID)):
    agg[f"oracle_{pre}"] = {k: max(((v, mean(k, f"{pre}{v}")) for v in grid[C[k]["bb"]]), key=lambda t: t[1])
                            for k in keys}
dd = {k: np.array(C[k]["arms"]["PDIAG_VAL"]["g"]) - np.array(C[k]["arms"]["PLUGIN_VAL"]["g"]) for k in keys}
agg["diag_minus_plain"] = {k: dict(mean=float(v.mean()), pos=int((v > 0).sum()), neg=int((v < 0).sum())) for k, v in dd.items()}
agg["diag_minus_plain_wilcoxon_p"] = float(stats.wilcoxon([v.mean() for v in dd.values()]).pvalue)
agg["nu_all_zero"] = all(C[k]["nu"] == 0 for k in keys)
agg["n_runs"] = len(R8)
agg["n_ok"] = sum(ok(r) for r in R8.values())
rep["agg"] = agg
rep["verdicts"] = S8["verdicts"]

os.makedirs(OUT, exist_ok=True)
json.dump(rep, open(os.path.join(OUT, "v8.json"), "w", encoding="utf-8"), indent=1, default=float)


def pct(x, d=2):
    s = f"{100 * x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def neg(s):
    return re.sub(r"(?<![\w.$])-(?=\d)", r"\\ensuremath{-}", s)


def nm(k, short=True):
    c = C[k]
    n = f"{c['ds']}-{c['T']}".replace("Weather", "Wth")
    return ("iT " + n) if c["bb"] != "TQNet" else n


# ---- main Table I rows: v8 plus the two earlier controls --------------------
old1 = json.load(open(os.path.join(OUT, "stats.json"), encoding="utf-8"))
t1_old = {}
for line in open(os.path.join(OUT, "table1_rows.tex"), encoding="utf-8"):
    if "&" in line:
        p = [x.strip() for x in line.replace("\\\\", "").split("&")]
        t1_old[(p[0], p[1])] = (p[5], p[6])
rows = []
for k in keys:
    c = C[k]
    a = c["arms"]
    oldk = ("TQ" if c["bb"] == "TQNet" else "iT", f"{c['ds']}-{c['T']}".replace("Weather", "Wth"))
    rcy, rrot = t1_old[oldk]
    rows.append(f"{nm(k)} & {pct(a['QDF_LR1']['mean'])} & {pct(a['QDF_VAL']['mean'])} & "
                f"{R(k, 'PLUGIN_VAL', 'QDF_VAL'):.2f}\\,({a['PLUGIN_VAL']['pos']}/{a['PLUGIN_VAL']['n']}) & "
                f"{R(k, 'PDIAG_VAL', 'QDF_VAL'):.2f} & "
                f"{R(k, 'REPLAY_K1', 'QDF_VAL'):.2f}\\,({a['REPLAY_K1']['pos']}/{a['REPLAY_K1']['n']}) & "
                f"{rcy} & {rrot} \\\\")
open(os.path.join(OUT, "v8_table1_rows.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v8_numbers.py -- do not edit by hand.\n" + neg("\n".join(rows)) + "\n")


# ---- supplement: all v8 arms with 95% t-intervals ---------------------------
def cc(k, arm):
    a = C[k]["arms"][arm]
    return f"{pct(a['mean'])}\\,$\\pm$\\,{pct(a['ci'])}\\,({a['pos']})"


rows = [f"{nm(k)} & {C[k]['mse_I']:.3f} & {cc(k, 'QDF_LR1')} & {cc(k, 'QDF_VAL')} & {cc(k, 'PLUGIN_VAL')} & "
        f"{cc(k, 'PDIAG_VAL')} & {cc(k, 'REPLAY_K1')} & {cc(k, 'REPLAY_VAL')} \\\\" for k in keys]
open(os.path.join(OUT, "v8_supp_rows.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v8_numbers.py -- do not edit by hand.\n" + neg("\n".join(rows)) + "\n")


# ---- supplement: dose responses (mean gain per grid value) ------------------
def grid_row(k, pre, grid):
    return " & ".join(pct(mean(k, f"{pre}{v}")) for v in grid)


def union_row(k, pre, union, grid):
    return " & ".join(pct(mean(k, f"{pre}{v}")) if v in grid else "--" for v in union)


M_ALL = M_GRID["TQNet"]
A_ALL = sorted(set(A_GRID["TQNet"]) | set(A_GRID["iTransformer"]), key=float)
rows_q, rows_p = [], []
for k in keys:
    bb = C[k]["bb"]
    rows_q.append(f"{nm(k)} & {union_row(k, 'QDF_LR', M_ALL, M_GRID[bb])} & {grid_row(k, 'REPLAY_K', K_GRID)} \\\\")
    rows_p.append(f"{nm(k)} & {union_row(k, 'PLUGIN_A', A_ALL, A_GRID[bb])} \\\\")
for fn, rows in (("v8_dose_qdf_rows.tex", rows_q), ("v8_dose_plug_rows.tex", rows_p)):
    open(os.path.join(OUT, fn), "w", encoding="utf-8").write(
        "% Auto-generated by paper/scripts/v8_numbers.py -- do not edit by hand.\n" + neg("\n".join(rows)) + "\n")

# ---- supplement: shape of learned Sigma, 49 earlier runs and 32 v8 pairs ----
rows = []
for e in rep["extras"]:
    name = f"{e['bb'].replace('iTransformer', 'iT').replace('DLinear', 'DL').replace('TQNet', 'TQ')} & {e['ds']} & {e['T']}"
    r2 = lambda v: f"{v[0]:.3f}" if f"{v[0]:.3f}" == f"{v[1]:.3f}" else f"{v[0]:.3f}--{v[1]:.3f}"
    rows.append(f"{name} & {e['n']} & {r2(e['dev'])} & {r2(e['fo'])} & {r2(e['sq'])} & {r2(e['rp'])} \\\\")
open(os.path.join(OUT, "extras_rows.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v8_numbers.py -- do not edit by hand.\n" + "\n".join(rows) + "\n")
rows = []
for k in keys:
    sh = [x for x in shape if x["cell"] == k and not x.get("missing")]
    e = [x["err"] for x in sh]
    dl = [x["dev_learned"] for x in sh]
    dr = [x["dev_replay"] for x in sh]
    rr = lambda v: f"{min(v):.3f}" if f"{min(v):.3f}" == f"{max(v):.3f}" else f"{min(v):.3f}--{max(v):.3f}"
    rows.append(f"{nm(k)} & {len(sh)} & {rr(dl)} & {rr(dr)} & {rr(e)} & {R(k, 'REPLAY_K1', 'QDF_VAL'):.2f} \\\\")
open(os.path.join(OUT, "v8_shape_rows.tex"), "w", encoding="utf-8").write(
    "% Auto-generated by paper/scripts/v8_numbers.py -- do not edit by hand.\n" + "\n".join(rows) + "\n")

# ---- figure data: R of I + a S0n against a (denominator QDF_VAL) ------------
for i, k in enumerate(keys, 1):
    bb = C[k]["bb"]
    with open(os.path.join(OUT, f"v8dose_{i}.dat"), "w", encoding="utf-8") as f:
        f.write("a R\n")
        for v in A_GRID[bb]:
            f.write(f"{v} {R(k, f'PLUGIN_A{v}', 'QDF_VAL'):.4f}\n")
fig_min = min(R(k, f"PLUGIN_A{v}", "QDF_VAL") for k in keys for v in A_GRID[C[k]["bb"]])

# ---- macros ----------------------------------------------------------------
m = {}
m["EightRuns"] = agg["n_runs"]
m["EightOK"] = agg["n_ok"]
m["EightRecPlug"] = agg["rec_PLUGIN_VAL_QDF_VAL"]
m["EightRecPlugOne"] = agg["rec_PLUGIN_VAL_QDF_LR1"]
m["EightRecDiag"] = agg["rec_PDIAG_VAL_QDF_VAL"]
m["EightRecRep"] = agg["rec_REPLAY_K1_QDF_VAL"]
m["EightRecRepOne"] = agg["rec_REPLAY_K1_QDF_LR1"]
m["EightRecRepVal"] = agg["rec_REPLAY_VAL_QDF_VAL"]
for tag, arm, den in (("PlugVal", "PLUGIN_VAL", "QDF_VAL"), ("PlugOne", "PLUGIN_VAL", "QDF_LR1"),
                      ("RepVal", "REPLAY_K1", "QDF_VAL"), ("RepOne", "REPLAY_K1", "QDF_LR1")):
    x = agg[f"diff_{arm}_{den}"]
    m[f"EightD{tag}"] = pct(x["mean"])
    m[f"EightD{tag}Neg"] = x["neg"]
    m[f"EightD{tag}P"] = f"{x['p']:.3f}"
x = agg["diff_QDF_VAL_QDF_LR1"]
m["EightDTune"], m["EightDTuneNeg"], m["EightDTuneP"] = pct(x["mean"]), x["neg"], f"{x['p']:.2f}"
m["EightSelOne"] = agg["qdf_val_selected_m1"]
m["EightSelN"] = agg["qdf_val_n"]
m["EightGOneMin"] = pct(min(mean(k, "QDF_LR1") for k in keys))
m["EightGOneMax"] = pct(max(mean(k, "QDF_LR1") for k in keys))
for v, tag in (("Q8.1", "One"), ("Q8.2", "Two"), ("Q8.3_K1", "ThreeA"), ("Q8.3_VAL", "ThreeB")):
    m["EightV" + tag] = "\\textsc{" + S8["verdicts"][v].lower().replace("_", "\\_") + "}"
h2 = [k for k in keys if C[k]["ds"] == "ETTh2"]
m["EightRepHTwoMin"] = f"{min(R(k, 'REPLAY_K1', 'QDF_VAL') for k in h2):.2f}"
m["EightRepHTwoMax"] = f"{max(R(k, 'REPLAY_K1', 'QDF_VAL') for k in h2):.2f}"
m["EightRepKTwoHTwoMin"] = f"{min(R(k, 'REPLAY_K2', 'QDF_VAL') for k in h2):.2f}"
m["EightRepKTwoHTwoMax"] = f"{max(R(k, 'REPLAY_K2', 'QDF_VAL') for k in h2):.2f}"
m["EightRepSecMin"], m["EightRepSecMax"] = f"{rep['replay_seconds'][0]:.2f}", f"{rep['replay_seconds'][1]:.1f}"
m["EightSZeroSecMax"] = f"{rep['s0_seconds'][1]:.1f}"
m["EightMetaSecMin"], m["EightMetaSecMax"] = f"{rep['qdf_meta_phase_seconds'][0]:.0f}", f"{rep['qdf_meta_phase_seconds'][1]:.0f}"
ok_shape = [x for x in shape if not x.get("missing")]
m["EightShapeN"] = len(ok_shape)
m["EightShapeMin"] = f"{min(x['err'] for x in ok_shape):.2f}"
m["EightShapeMax"] = f"{max(x['err'] for x in ok_shape):.2f}"
nonh2 = [x for x in ok_shape if not x["cell"].startswith("ETTh2")]
m["EightShapeOtherMax"] = f"{max(x['err'] for x in nonh2):.2f}"
far = [x for x in ok_shape if x["cell"].startswith(("ETTh2", "ETTm2"))]
near = [x for x in ok_shape if not x["cell"].startswith(("ETTh2", "ETTm2"))]
m["EightShapeNearMin"], m["EightShapeNearMax"] = f"{min(x['err'] for x in near):.2f}", f"{max(x['err'] for x in near):.2f}"
m["EightShapeFarMin"], m["EightShapeFarMax"] = f"{min(x['err'] for x in far):.2f}", f"{max(x['err'] for x in far):.2f}"
frac = [R8[rid("QDF_LR1", bb, ds, T, s)]["meta_phase_seconds"] / R8[rid("QDF_LR1", bb, ds, T, s)]["seconds"]
        for bb, ds, T in CELLS for s in rep["cells"][ckey(bb, ds, T)]["seeds"]]
m["EightMetaFracMin"], m["EightMetaFracMax"] = f"{100 * min(frac):.0f}", f"{100 * max(frac):.0f}"
m["ExN"] = len(EX)
m["ExGroups"] = len(rep["extras"])
m["ExAllBest"] = "yes" if all(e["replay_best"] for e in rep["extras"]) else "no"
m["ExRpMin"] = f"{min(e['rp'][0] for e in rep['extras']):.3f}"
m["ExRpMax"] = f"{max(e['rp'][1] for e in rep['extras']):.2f}"
m["ExFoMin"] = f"{min(e['fo'][0] for e in rep['extras']):.2f}"
m["ExFoMax"] = f"{max(e['fo'][1] for e in rep['extras']):.2f}"
m["ExSqMin"] = f"{min(e['sq'][0] for e in rep['extras']):.2f}"
m["ExSqMax"] = f"{max(e['sq'][1] for e in rep['extras']):.2f}"
rs96 = json.load(open(os.path.join(OUT, "resim_meta_T96.json"), encoding="utf-8"))
m["AllShapeN"] = len(rs96) + len(EX)
m["AllRpMin"] = f"{min([r['resim']['err'] for r in rs96] + [r['replay']['err'] for r in EX]):.3f}"
m["AllRpMax"] = f"{max([r['resim']['err'] for r in rs96] + [r['replay']['err'] for r in EX]):.2f}"
m["AllFoMin"] = f"{min([r['first_order']['err'] for r in rs96] + [r['first_order']['err'] for r in EX]):.2f}"
m["AllFoMax"] = f"{max([r['first_order']['err'] for r in rs96] + [r['first_order']['err'] for r in EX]):.2f}"
m["AllSqMin"] = f"{min([r['sqrt_family']['err'] for r in rs96] + [r['sqrt_family']['err'] for r in EX]):.2f}"
m["AllSqMax"] = f"{max([r['sqrt_family']['err'] for r in rs96] + [r['sqrt_family']['err'] for r in EX]):.2f}"
m["AllRpBest"] = "yes" if (all(r["resim"]["err"] < min(r["first_order"]["err"], r["sqrt_family"]["err"]) for r in rs96)
                           and m["ExAllBest"] == "yes") else "no"
m["ExRatioMin"] = f"{min(e['ratio'][0] for e in rep['extras']):.2f}"
m["ExRatioMax"] = f"{max(e['ratio'][1] for e in rep['extras']):.2f}"
dm = agg["diag_minus_plain"]
m["EightDiagPosCells"] = sum(v["mean"] > 0 for v in dm.values())
m["EightDiagP"] = f"{agg['diag_minus_plain_wilcoxon_p']:.2f}"
m["EightFigMin"] = f"{fig_min:.1f}"
for k, tag in (("ETTm2-96", "Mtwo"), ("ETTh2-96", "Htwo")):
    a_or, g_or = agg["oracle_PLUGIN_A"][k]
    m[f"Eight{tag}OrA"] = a_or
    m[f"Eight{tag}OrR"] = f"{g_or / mean(k, 'QDF_VAL'):.2f}"
    m[f"Eight{tag}RVal"] = f"{R(k, 'PLUGIN_VAL', 'QDF_VAL'):.2f}"
qgrid = [(k, v, mean(k, f"QDF_LR{v}")) for k in keys for v in M_GRID[C[k]["bb"]]]
pgrid = [(k, v, mean(k, f"PLUGIN_A{v}")) for k in keys for v in A_GRID[C[k]["bb"]]]
m["EightQGridN"] = len(qgrid)
m["EightQGridNeg"] = sum(x[2] < 0 for x in qgrid)
m["EightQGridMin"] = pct(min(x[2] for x in qgrid))
m["EightPGridNeg"] = sum(x[2] < 0 for x in pgrid)
m["EightPGridN"] = len(pgrid)
m["EightPGridMin"] = pct(min(x[2] for x in pgrid))
m["EightPGridMinCell"] = min(pgrid, key=lambda x: x[2])[0]
m["EightPGridMinA"] = min(pgrid, key=lambda x: x[2])[1]
qo = agg["oracle_QDF_LR"]
m["EightHtwoLongQOr"] = pct(qo["ETTh2-720"][1])
m["EightHtwoLongQOrM"] = qo["ETTh2-720"][0]
m["EightNQOrAbove"] = sum(qo[k][0] != "1" for k in keys)
m["EightOrEighty"] = sum(agg["oracle_PLUGIN_A"][k][1] / mean(k, "QDF_VAL") >= 0.8 for k in keys)
xpg = rep["cross_platform"].get("gpu703")
xpc = rep["cross_platform"].get("cpu")
m["XpGpuN"], m["XpGpuMax"] = xpg["n"], pct(xpg["max_abs"])
m["XpCpuN"], m["XpCpuMax"] = xpc["n"], pct(xpc["max_abs"])
with open(os.path.join(OUT, "v8.tex"), "w", encoding="utf-8") as f:
    f.write("% Auto-generated by paper/scripts/v8_numbers.py -- do not edit by hand.\n")
    for k, v in m.items():
        f.write(f"\\newcommand{{\\num{k}}}{{{neg(str(v))}}}\n")
print(json.dumps(m, indent=0, default=str))
print("cross_platform", json.dumps(rep["cross_platform"], default=float))
print("oracle plugin", {k: (v[0], round(v[1] / mean(k, 'QDF_VAL'), 2)) for k, v in agg["oracle_PLUGIN_A"].items()})
print("oracle qdf", {k: (v[0], round(100 * v[1], 2)) for k, v in agg["oracle_QDF_LR"].items()})
print("diag-plain", dm)
for x in ok_shape:
    print(x["cell"], x["seed"], round(x["err"], 3), round(x["dev_learned"], 3), round(x["dev_replay"], 3))
