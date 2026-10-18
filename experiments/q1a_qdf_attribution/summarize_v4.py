"""Apply protocol_v4.json decision rules (AutoDL runs only; local T=720 runs only for the cross-platform check)."""
import json
import os
import statistics as st

import run_grid_v3 as g3

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = ("ETTh1-96", "ETTh2-96", "ETTh2-720")


def load(name):
    out = {}
    path = os.path.join(HERE, "results", name)
    if not os.path.exists(path):
        return out
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


def selected_a_v3():
    s = json.load(open(os.path.join(HERE, "results", "summary_v3_amend1.json"), encoding="utf-8"))
    return {(c["cell"], e["seed"]): e["selected_a"] for c in s["cells"] for e in c["selection"]}, \
           {c["cell"]: c["arms"]["PLUGIN_VAL"]["R"] for c in s["cells"]}


def main():
    proto = json.load(open(os.path.join(HERE, "protocol_v4.json"), encoding="utf-8"))
    v3a, v4 = load("runs_v3_autodl.jsonl"), load("runs_v4_autodl.jsonl")
    local = dict(load("runs.jsonl"), **load("runs_v3.jsonl"))
    sel_v3, R_pv_v3 = selected_a_v3()
    seeds = proto["grid"]["seeds"]
    tags = [g3.a_tag(a) for a in proto["a_grid"]]
    cells, complete = [], True
    for ds in proto["grid"]["datasets"]:
        for T in proto["grid"]["pred_len"]:
            key = f"{ds}-{T}"
            src = dict(v3a, **v4) if T == 96 else v4
            rid = lambda arm, s: f"{arm}_{ds}_{T}_{s}"
            mse = lambda arm: {s: src[rid(arm, s)]["test_mse"] for s in seeds if rid(arm, s) in src}
            base = mse("SIGMA_I")
            gain = lambda m: {s: (base[s] - m[s]) / base[s] for s in m if s in base}
            gq = gain(mse("QDF"))
            gQ = mean(gq.values()) if len(gq) == len(seeds) else None

            def stats(g):
                full = len(g) == len(seeds) and gQ
                m = mean(g.values())
                R = m / gQ if full else None
                return dict(n=len(g), g_mean=m, g_per_seed=[g.get(s) for s in seeds], pos=sum(x > 0 for x in g.values()),
                            R=R, band=band(R))

            def val_pick(prefix):
                picks = {}
                for s in seeds:
                    cand = [(src[rid(prefix + t, s)]["best_val_mse"], float(t), src[rid(prefix + t, s)]["test_mse"])
                            for t in tags if rid(prefix + t, s) in src and src[rid(prefix + t, s)]["best_val_mse"] is not None]
                    if len(cand) == len(tags):
                        best = min(cand, key=lambda c: (c[0], c[1]))
                        oracle = min(cand, key=lambda c: (c[2], c[1]))
                        picks[s] = dict(a=best[1], test_mse=best[2], oracle_a=oracle[1])
                return picks

            row = dict(cell=key, analysis=key in ANALYSIS, g_QDF_mean=gQ, g_QDF_pos=sum(x > 0 for x in gq.values()), arms={})
            pv, cv = val_pick("PLUGIN_A"), val_pick("COVY_A")
            g_pv, g_cv = gain({s: p["test_mse"] for s, p in pv.items()}), gain({s: p["test_mse"] for s, p in cv.items()})
            row["arms"]["PLUGIN_VAL"] = dict(stats(g_pv), selected_a=[pv.get(s, {}).get("a") for s in seeds],
                                             oracle_a=[pv.get(s, {}).get("oracle_a") for s in seeds])
            row["arms"]["COVY_VAL"] = dict(stats(g_cv), selected_a=[cv.get(s, {}).get("a") for s in seeds],
                                           oracle_a=[cv.get(s, {}).get("oracle_a") for s in seeds])
            for arm in [f"PLUGIN_A{t}" for t in tags] + [f"COVY_A{t}" for t in tags] + ["LW", "SPECTRUM_RANDOM"]:
                row["arms"][arm] = dict(stats(gain(mse(arm))), mse_mean=mean(mse(arm).values()))
            row["arms"]["QDF"] = dict(stats(gq), mse_mean=mean(mse("QDF").values()))
            row["arms"]["SIGMA_I"] = dict(mse_mean=mean(base.values()))

            D_label = {s: g_pv[s] - g_cv[s] for s in seeds if s in g_pv and s in g_cv}
            g_sr = gain(mse("SPECTRUM_RANDOM"))
            matched = {}
            for s in seeds:
                a = sel_v3.get((key, s))
                r = rid(f"PLUGIN_A{g3.a_tag(a)}", s) if a is not None else None
                if r in src and s in base:
                    matched[s] = (base[s] - src[r]["test_mse"]) / base[s]
            D_struct = {s: matched[s] - g_sr[s] for s in seeds if s in matched and s in g_sr}
            row["D_label"] = dict(mean=mean(D_label.values()), per_seed=[D_label.get(s) for s in seeds],
                                  pos=sum(x > 0 for x in D_label.values()), neg=sum(x < 0 for x in D_label.values()))
            row["D_struct"] = dict(mean=mean(D_struct.values()), per_seed=[D_struct.get(s) for s in seeds],
                                   pos=sum(x > 0 for x in D_struct.values()), matched_a=[sel_v3.get((key, s)) for s in seeds])

            st_info = [src[rid(arm, s)]["sigma_target"] for arm in ["LW", "SPECTRUM_RANDOM"] + [f"COVY_A{t}" for t in tags]
                       for s in seeds if rid(arm, s) in src and src[rid(arm, s)].get("sigma_target")]
            lw = [src[rid("LW", s)]["sigma_target"] for s in seeds if rid("LW", s) in src and src[rid("LW", s)].get("sigma_target")]
            row["descriptive"] = dict(
                cos_S0_Cy=mean(x.get("cos_S0_Cy") for x in st_info), cos_S0_My=mean(x.get("cos_S0_My") for x in st_info),
                relfro_S0_minus_My=mean(x.get("relfro_S0_minus_My") for x in st_info),
                lw_delta=[x.get("lw_delta") for x in lw], lw_a_equiv=[x.get("a_equiv") for x in lw],
                lw_fallback=[x.get("lw_fallback") for x in lw],
                max_inject_err=max((x["inject_max_abs_err"] for x in st_info), default=None),
                max_spectrum_rel_err=max((x.get("spectrum_max_rel_err", 0) for x in st_info), default=None))
            if T == 720:
                pairs = [(r, (src[r]["test_mse"] - local[r]["test_mse"]) / local[r]["test_mse"])
                         for r in sorted(src) if r in local and r.endswith(f"_{ds}_{T}_{r.split('_')[-1]}")]
                lb = {s: local[rid("SIGMA_I", s)]["test_mse"] for s in seeds if rid("SIGMA_I", s) in local}
                lq = {s: (lb[s] - local[rid("QDF", s)]["test_mse"]) / lb[s] for s in seeds if rid("QDF", s) in local and s in lb}
                row["cross_platform"] = dict(n_pairs=len(pairs), max_abs_rel_diff=max((abs(d) for _, d in pairs), default=None),
                                             g_QDF_local=mean(lq.values()), g_QDF_autodl=gQ,
                                             R_PLUGIN_VAL_local=R_pv_v3.get(key), R_PLUGIN_VAL_autodl=row["arms"]["PLUGIN_VAL"]["R"])
            if row["analysis"]:
                need = [row["arms"][a]["R"] for a in ("PLUGIN_VAL", "COVY_VAL", "LW", "SPECTRUM_RANDOM")]
                complete &= all(x is not None for x in need) and len(D_label) == len(seeds) and len(D_struct) == len(seeds)
            cells.append(row)

    an = [c for c in cells if c["analysis"]]
    verdicts = {}
    if not complete:
        verdicts = {"Q4.1": "INCOMPLETE", "Q4.2": "INCOMPLETE", "Q4.3": "INCOMPLETE"}
    else:
        thr = lambda c: 0.25 * c["g_QDF_mean"]
        eq = sum(c["arms"]["COVY_VAL"]["R"] >= 0.8 and abs(c["D_label"]["mean"]) <= thr(c) for c in an)
        res = sum(c["D_label"]["mean"] > thr(c) and c["D_label"]["pos"] >= 4 for c in an)
        lab = sum(c["D_label"]["mean"] < -thr(c) and c["D_label"]["neg"] >= 4 for c in an)
        verdicts["Q4.1"] = ("LABEL_COV_EQUIVALENT" if eq >= 2 else "RESIDUAL_BETTER" if res >= 2
                            else "LABEL_BETTER" if lab >= 2 else "MIXED")
        sm = sum(c["arms"]["SPECTRUM_RANDOM"]["R"] < 0.5 and c["D_struct"]["pos"] >= 4 for c in an)
        ss = sum(c["arms"]["SPECTRUM_RANDOM"]["R"] >= 0.8 for c in an)
        verdicts["Q4.2"] = "STRUCTURE_MATTERS" if sm >= 2 else "SPECTRUM_SUFFICES" if ss >= 2 else "PARTIAL"
        w = sum(c["arms"]["LW"]["R"] >= 0.8 and c["arms"]["LW"]["pos"] >= 4 for c in an)
        f = sum(c["arms"]["LW"]["R"] < 0.5 for c in an)
        verdicts["Q4.3"] = "AUTO_SHRINK_WORKS" if w >= 2 else "AUTO_SHRINK_FAILS" if f >= 2 else "PARTIAL"
    out = dict(protocol_id=proto["protocol_id"], verdicts=verdicts, n_v4_runs=len(v4), cells=cells)
    json.dump(out, open(os.path.join(HERE, "results", "summary_v4.json"), "w", encoding="utf-8"), indent=1)
    fmt = lambda x, p="+.4f": "None" if x is None else format(x, p)
    for c in cells:
        print(f"== {c['cell']}{'' if c['analysis'] else ' (descriptive)'}  g_QDF {fmt(c['g_QDF_mean'])} ({c['g_QDF_pos']}/5)")
        for arm in ["PLUGIN_VAL", "COVY_VAL", "LW", "SPECTRUM_RANDOM"] + [f"COVY_A{t}" for t in tags] + [f"PLUGIN_A{t}" for t in tags]:
            a = c["arms"][arm]
            print(f"  {arm:16s} g {fmt(a['g_mean'])}  R {fmt(a['R'], '+.2f')}  {a['pos']}/{a['n']} {a['band'] or ''}"
                  + (f"  sel a {a['selected_a']} oracle {a['oracle_a']}" if "selected_a" in a else ""))
        print(f"  D_label {fmt(c['D_label']['mean'])} pos {c['D_label']['pos']} neg {c['D_label']['neg']}; "
              f"D_struct {fmt(c['D_struct']['mean'])} pos {c['D_struct']['pos']} (matched a {c['D_struct']['matched_a']})")
        print("  descriptive:", json.dumps(c["descriptive"]))
        if "cross_platform" in c:
            print("  cross-platform:", json.dumps(c["cross_platform"]))
    print("verdicts:", verdicts, f"({len(v4)} v4 runs)")


if __name__ == "__main__":
    main()
