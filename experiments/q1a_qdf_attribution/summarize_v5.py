"""Apply protocol_v5.json (stage A: analysis cells and selected a; stage B: Q5.1-Q5.3 verdicts). GPU runs only."""
import json
import math
import os
import statistics as st
import sys

import run_grid_v3 as g3

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = json.load(open(os.path.join(HERE, "protocol_v5.json"), encoding="utf-8"))
SEEDS = PROTO["grid"]["seeds"]
TAGS = [g3.a_tag(a) for a in PROTO["a_grid"]]
STAGE_A_ARMS = ["QDF", "SIGMA_I", "SIGMA_I_REP"] + [f"PLUGIN_A{t}" for t in TAGS]
STAGE_B_ARMS = [f"COVY_A{t}" for t in TAGS] + ["SPECTRUM_RANDOM"]


def load(path=os.path.join(HERE, "results", "runs_v5.jsonl")):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["returncode"] == 0 and r["test_mse"] is not None:
                out[r["run_id"]] = r
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def band(R):
    return None if R is None else ("RECOVERS" if R >= 0.8 else "FAILS" if R < 0.5 else "PARTIAL")


def split(cell):
    ds, T = cell.rsplit("-", 1)
    return ds, int(T)


def rid(arm, cell, s):
    ds, T = split(cell)
    return f"{arm}_{ds}_{T}_{s}"


def val_pick(src, cell, prefix):
    picks = {}
    for s in SEEDS:
        cand = [(src[rid(prefix + t, cell, s)]["best_val_mse"], float(t), src[rid(prefix + t, cell, s)]["test_mse"])
                for t in TAGS if rid(prefix + t, cell, s) in src and src[rid(prefix + t, cell, s)]["best_val_mse"] is not None]
        if len(cand) == len(TAGS):
            best = min(cand, key=lambda c: (c[0], c[1]))
            oracle = min(cand, key=lambda c: (c[2], c[1]))
            picks[s] = dict(a=best[1], test_mse=best[2], oracle_a=oracle[1])
    return picks


def stage_a(src):
    """Return (complete, {cell: row}) with the analysis-cell decision and per-seed selected a."""
    rows, complete = {}, True
    for cell in PROTO["cells"]:
        have = all(rid(arm, cell, s) in src for arm in STAGE_A_ARMS for s in SEEDS)
        complete &= have
        mse = lambda arm: {s: src[rid(arm, cell, s)]["test_mse"] for s in SEEDS if rid(arm, cell, s) in src}
        base, rep = mse("SIGMA_I"), mse("SIGMA_I_REP")
        gq = {s: (base[s] - m) / base[s] for s, m in mse("QDF").items() if s in base}
        nu = mean(abs(rep[s] - base[s]) / base[s] for s in SEEDS if s in rep and s in base)
        gQ = mean(gq.values()) if len(gq) == len(SEEDS) else None
        pv = val_pick(src, cell, "PLUGIN_A")
        analysis = (have and gQ is not None and sum(g > 0 for g in gq.values()) == len(SEEDS)
                    and gQ >= max(0.005, 3 * (nu or 0.0)))
        rows[cell] = dict(cell=cell, stage_a_complete=have, analysis=analysis, g_QDF_mean=gQ,
                          g_QDF_per_seed=[gq.get(s) for s in SEEDS], g_QDF_pos=sum(g > 0 for g in gq.values()), nu=nu,
                          nu_per_seed=[abs(rep[s] - base[s]) / base[s] if s in rep and s in base else None for s in SEEDS],
                          selected_a={str(s): pv[s]["a"] for s in pv})
    return complete, rows


def main():
    src = load()
    complete_a, rows_a = stage_a(src)
    an_cells = [c for c, r in rows_a.items() if r["analysis"]] if complete_a else []
    n = len(an_cells)
    k = math.ceil(2 * n / 3)
    cells, complete_b = [], complete_a
    for cell in PROTO["cells"]:
        row = dict(rows_a[cell], arms={})
        mse = lambda arm: {s: src[rid(arm, cell, s)]["test_mse"] for s in SEEDS if rid(arm, cell, s) in src}
        base = mse("SIGMA_I")
        gain = lambda m: {s: (base[s] - m[s]) / base[s] for s in m if s in base}
        gq = gain(mse("QDF"))
        gQ = row["g_QDF_mean"]

        def stats(g):
            m = mean(g.values())
            R = m / gQ if (len(g) == len(SEEDS) and gQ) else None
            return dict(n=len(g), g_mean=m, g_per_seed=[g.get(s) for s in SEEDS], pos=sum(x > 0 for x in g.values()),
                        R=R, band=band(R))

        pv, cv = val_pick(src, cell, "PLUGIN_A"), val_pick(src, cell, "COVY_A")
        g_pv, g_cv = gain({s: p["test_mse"] for s, p in pv.items()}), gain({s: p["test_mse"] for s, p in cv.items()})
        row["arms"]["PLUGIN_VAL"] = dict(stats(g_pv), selected_a=[pv.get(s, {}).get("a") for s in SEEDS],
                                         oracle_a=[pv.get(s, {}).get("oracle_a") for s in SEEDS])
        row["arms"]["COVY_VAL"] = dict(stats(g_cv), selected_a=[cv.get(s, {}).get("a") for s in SEEDS],
                                       oracle_a=[cv.get(s, {}).get("oracle_a") for s in SEEDS])
        for arm in [f"PLUGIN_A{t}" for t in TAGS] + STAGE_B_ARMS + ["QDF", "SIGMA_I_REP"]:
            row["arms"][arm] = dict(stats(gain(mse(arm))), mse_mean=mean(mse(arm).values()))
        row["arms"]["SIGMA_I"] = dict(mse_mean=mean(base.values()))
        D_label = {s: g_pv[s] - g_cv[s] for s in SEEDS if s in g_pv and s in g_cv}
        g_sr = gain(mse("SPECTRUM_RANDOM"))
        matched = {}
        for s in SEEDS:
            a = pv.get(s, {}).get("a")
            r = rid(f"PLUGIN_A{g3.a_tag(a)}", cell, s) if a is not None else None
            if r in src and s in base:
                matched[s] = (base[s] - src[r]["test_mse"]) / base[s]
        D_struct = {s: matched[s] - g_sr[s] for s in SEEDS if s in matched and s in g_sr}
        row["D_label"] = dict(mean=mean(D_label.values()), per_seed=[D_label.get(s) for s in SEEDS],
                              pos=sum(x > 0 for x in D_label.values()), neg=sum(x < 0 for x in D_label.values()))
        row["D_struct"] = dict(mean=mean(D_struct.values()), per_seed=[D_struct.get(s) for s in SEEDS],
                               pos=sum(x > 0 for x in D_struct.values()))
        st_info = [src[rid(arm, cell, s)]["sigma_target"] for arm in STAGE_B_ARMS + [f"PLUGIN_A{t}" for t in TAGS]
                   for s in SEEDS if rid(arm, cell, s) in src and src[rid(arm, cell, s)].get("sigma_target")]
        row["descriptive"] = dict(
            cos_S0_Cy=mean(x.get("cos_S0_Cy") for x in st_info),
            max_inject_err=max((x["inject_max_abs_err"] for x in st_info), default=None),
            max_spectrum_rel_err=max((x.get("spectrum_max_rel_err", 0) for x in st_info), default=None),
            seconds_mean={arm: mean(src[rid(arm, cell, s)]["seconds"] for s in SEEDS if rid(arm, cell, s) in src)
                          for arm in ("QDF", "SIGMA_I", "PLUGIN_A0.25")})
        if row["analysis"]:
            complete_b &= (all(rid(arm, cell, s) in src for arm in STAGE_B_ARMS for s in SEEDS)
                           and len(D_label) == len(SEEDS) and len(D_struct) == len(SEEDS))
        cells.append(row)

    an = [c for c in cells if c["analysis"]]
    verdicts = {"Q5.0": dict(n_analysis_cells=n, analysis_cells=an_cells, k=k) if complete_a else "INCOMPLETE"}
    if not complete_a:
        verdicts.update({q: "INCOMPLETE" for q in ("Q5.1", "Q5.2", "Q5.3")})
    elif n < 2:
        verdicts.update({q: "INSUFFICIENT_CELLS" for q in ("Q5.1", "Q5.2", "Q5.3")})
    else:
        S = len(SEEDS)
        rec = sum(c["arms"]["PLUGIN_VAL"]["R"] >= 0.8 and c["arms"]["PLUGIN_VAL"]["pos"] == S for c in an)
        add = sum(c["arms"]["PLUGIN_VAL"]["R"] < 0.5 for c in an)
        verdicts["Q5.1"] = ("META_LEARNING_UNNECESSARY_GENERALISES" if rec >= k
                            else "META_LEARNING_ADDS_VALUE" if add >= k else "MIXED")
        if not complete_b:
            verdicts.update({"Q5.2": "INCOMPLETE", "Q5.3": "INCOMPLETE"})
        else:
            thr = lambda c: 0.25 * c["g_QDF_mean"]
            eq = sum(c["arms"]["COVY_VAL"]["R"] >= 0.8 and abs(c["D_label"]["mean"]) <= thr(c) for c in an)
            res = sum(c["D_label"]["mean"] > thr(c) and c["D_label"]["pos"] == S for c in an)
            lab = sum(c["D_label"]["mean"] < -thr(c) and c["D_label"]["neg"] == S for c in an)
            verdicts["Q5.2"] = ("LABEL_COV_EQUIVALENT" if eq >= k else "RESIDUAL_BETTER" if res >= k
                                else "LABEL_BETTER" if lab >= k else "MIXED")
            sm = sum(c["arms"]["SPECTRUM_RANDOM"]["R"] < 0.5 and c["D_struct"]["pos"] == S for c in an)
            ss = sum(c["arms"]["SPECTRUM_RANDOM"]["R"] >= 0.8 for c in an)
            verdicts["Q5.3"] = "STRUCTURE_MATTERS" if sm >= k else "SPECTRUM_SUFFICES" if ss >= k else "PARTIAL"
    out = dict(protocol_id=PROTO["protocol_id"], verdicts=verdicts, stage_a_complete=complete_a,
               stage_b_complete=complete_b, n_runs=len(src), cells=cells)
    json.dump(out, open(os.path.join(HERE, "results", "summary_v5.json"), "w", encoding="utf-8"), indent=1)
    if complete_a:
        json.dump(dict(analysis_cells=an_cells, rows=rows_a), open(os.path.join(HERE, "results", "stage_a_v5.json"), "w",
                                                                   encoding="utf-8"), indent=1)
    fmt = lambda x, p="+.4f": "None" if x is None else format(x, p)
    for c in cells:
        print(f"== {c['cell']}{'' if c['analysis'] else ' (descriptive)'}  g_QDF {fmt(c['g_QDF_mean'])} "
              f"({c['g_QDF_pos']}/{len(SEEDS)})  nu {fmt(c['nu'], '.2e')}  sec {c['descriptive']['seconds_mean']}")
        for arm in ["PLUGIN_VAL", "COVY_VAL", "SPECTRUM_RANDOM", "SIGMA_I_REP"] + [f"PLUGIN_A{t}" for t in TAGS] \
                + [f"COVY_A{t}" for t in TAGS]:
            a = c["arms"][arm]
            if a["n"]:
                print(f"  {arm:16s} g {fmt(a['g_mean'])}  R {fmt(a['R'], '+.2f')}  {a['pos']}/{a['n']} {a['band'] or ''}"
                      + (f"  sel a {a['selected_a']} oracle {a['oracle_a']}" if "selected_a" in a else ""))
        print(f"  D_label {fmt(c['D_label']['mean'])} pos {c['D_label']['pos']} neg {c['D_label']['neg']}; "
              f"D_struct {fmt(c['D_struct']['mean'])} pos {c['D_struct']['pos']}")
        print("  descriptive:", json.dumps({k: v for k, v in c["descriptive"].items() if k != "seconds_mean"}))
    print("verdicts:", json.dumps(verdicts), f"({len(src)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
