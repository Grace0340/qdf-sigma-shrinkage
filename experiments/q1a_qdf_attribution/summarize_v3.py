"""Apply protocol_v3.json decision rules to results/runs.jsonl (v1 QDF, SIGMA_I) and results/runs_v3.jsonl."""
import json
import os
import statistics as st
import sys

import run_grid_v3 as g3

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = ("ETTh1-96", "ETTh2-96", "ETTh2-720")


def load(name):
    out = {}
    for line in open(os.path.join(HERE, "results", name), encoding="utf-8"):
        r = json.loads(line)
        if r["returncode"] == 0 and r["test_mse"] is not None:
            out[r["run_id"]] = r
    return out


def band(R):
    return None if R is None else ("RECOVERS" if R >= 0.8 else "FAILS" if R < 0.5 else "PARTIAL")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def v1_timing(run_id, runs_dir="runs"):
    path = os.path.join(HERE, runs_dir, run_id, "stdout.txt")
    return g3.parse_stdout(open(path, encoding="utf-8", errors="replace").read()) if os.path.exists(path) else {}


def arm_stats(base, mse, n_seeds, gq_mean):
    g = {s: (base[s] - mse[s]) / base[s] for s in mse if s in base}
    m = mean(g.values())
    R = m / gq_mean if len(g) == n_seeds and gq_mean else None
    return dict(n=len(g), g_mean=m, g_per_seed=[g[s] for s in sorted(g)], pos=sum(x > 0 for x in g.values()), R=R,
                band=band(R), mse_mean=mean(mse.values()), mse_sd=st.stdev(mse.values()) if len(mse) > 1 else None)


def main(amend1=False):
    proto = json.load(open(os.path.join(HERE, "protocol_v3.json"), encoding="utf-8"))
    v1_local, v3_local = load("runs.jsonl"), load("runs_v3.jsonl")
    autodl = load("runs_v3_autodl.jsonl") if amend1 else {}
    seeds = proto["grid"]["seeds"]
    a_arms = [f"PLUGIN_A{g3.a_tag(a)}" for a in proto["a_grid"]]
    cells = []
    for ds in proto["grid"]["datasets"]:
        for T in proto["grid"]["pred_len"]:
            if amend1 and T == 96:
                v1, v3, runs_dir, plat = autodl, autodl, "runs_autodl", "autodl"
            else:
                v1, v3, runs_dir, plat = v1_local, v3_local, "runs", "local"
            key = f"{ds}-{T}"
            rid = lambda arm, s: f"{arm}_{ds}_{T}_{s}"
            base = {s: v1[rid("SIGMA_I", s)]["test_mse"] for s in seeds if rid("SIGMA_I", s) in v1}
            qdf = {s: v1[rid("QDF", s)]["test_mse"] for s in seeds if rid("QDF", s) in v1}
            gq = arm_stats(base, qdf, len(seeds), None)
            row = dict(cell=key, platform=plat, analysis=key in ANALYSIS, g_QDF_mean=gq["g_mean"], arms={}, selection=[])
            row["arms"]["SIGMA_I"] = dict(mse_mean=mean(base.values()), mse_sd=st.stdev(base.values()) if len(base) > 1 else None)
            row["arms"]["QDF"] = dict(gq, R=1.0 if gq["g_mean"] else None, band=None)
            sel_mse = {}
            for s in seeds:
                cand = [(v3[rid(a, s)]["best_val_mse"], float(a[len("PLUGIN_A"):]), v3[rid(a, s)]["test_mse"])
                        for a in a_arms if rid(a, s) in v3 and v3[rid(a, s)]["best_val_mse"] is not None]
                entry = dict(seed=s, n_candidates=len(cand))
                if len(cand) == len(a_arms):
                    best = min(cand, key=lambda c: (c[0], c[1]))
                    oracle = min(cand, key=lambda c: (c[2], c[1]))
                    sel_mse[s] = best[2]
                    entry.update(selected_a=best[1], selected_val_mse=best[0], selected_test_mse=best[2],
                                 oracle_a_by_test=oracle[1], oracle_test_mse=oracle[2],
                                 val_mse_by_a={c[1]: c[0] for c in cand}, test_mse_by_a={c[1]: c[2] for c in cand})
                row["selection"].append(entry)
            row["arms"]["PLUGIN_VAL"] = arm_stats(base, sel_mse, len(seeds), gq["g_mean"])
            for arm in a_arms + ["FGLS_FULL"]:
                mse = {s: v3[rid(arm, s)]["test_mse"] for s in seeds if rid(arm, s) in v3}
                row["arms"][arm] = arm_stats(base, mse, len(seeds), gq["g_mean"])
            # cost (wall seconds, 14 single-thread jobs in parallel)
            qt = [v1_timing(rid("QDF", s), runs_dir) for s in seeds]
            cost = dict(
                QDF_meta_phase_s=mean(t.get("meta_phase_seconds") for t in qt),
                QDF_meta_test_s=mean(t.get("meta_test_seconds") for t in qt),
                QDF_total_s=mean(v1[rid("QDF", s)]["seconds"] for s in seeds if rid("QDF", s) in v1),
                SIGMA_I_total_s=mean(v1[rid("SIGMA_I", s)]["seconds"] for s in seeds if rid("SIGMA_I", s) in v1),
                S0_s=mean(v3[rid(arm, s)]["sigma_target"]["s0_seconds"] for arm in a_arms + ["FGLS_FULL"] for s in seeds
                          if rid(arm, s) in v3 and v3[rid(arm, s)].get("sigma_target")),
                v3_inert_meta_phase_s=mean(v3[rid(arm, s)]["meta_phase_seconds"] for arm in a_arms + ["FGLS_FULL"]
                                           for s in seeds if rid(arm, s) in v3),
                total_s_by_arm={arm: mean(v3[rid(arm, s)]["seconds"] for s in seeds if rid(arm, s) in v3)
                                for arm in a_arms + ["FGLS_FULL"]},
                meta_test_s_by_arm={arm: mean(v3[rid(arm, s)]["meta_test_seconds"] for s in seeds if rid(arm, s) in v3)
                                    for arm in a_arms + ["FGLS_FULL"]},
            )
            mt = [cost["meta_test_s_by_arm"][a] for a in a_arms]
            cost["meta_free_procedure_s"] = (cost["S0_s"] or 0) + sum(mt) if all(x is not None for x in mt) else None
            cost["QDF_procedure_s"] = ((cost["QDF_meta_phase_s"] or 0) + cost["QDF_meta_test_s"]
                                       if cost["QDF_meta_test_s"] is not None else None)
            row["cost"] = cost
            row["injection"] = dict(
                max_inject_err=max((v3[rid(arm, s)]["sigma_target"]["inject_max_abs_err"] for arm in a_arms + ["FGLS_FULL"]
                                    for s in seeds if rid(arm, s) in v3 and v3[rid(arm, s)].get("sigma_target")), default=None),
                S0_trace_over_T=sorted({round(v3[rid(arm, s)]["sigma_target"]["S0_trace_over_T"], 6) for arm in a_arms + ["FGLS_FULL"]
                                        for s in seeds if rid(arm, s) in v3 and v3[rid(arm, s)].get("sigma_target")}),
                stub_runtime_use=sorted({json.dumps(v3[rid(arm, s)]["stub_runtime_use"], sort_keys=True)
                                         for arm in a_arms + ["FGLS_FULL"] for s in seeds if rid(arm, s) in v3}),
                test_windows=sorted({v3[rid(arm, s)]["test_windows"] for arm in a_arms + ["FGLS_FULL"] for s in seeds
                                     if rid(arm, s) in v3}))
            cells.append(row)
    an = [c for c in cells if c["analysis"]]
    if any(c["arms"]["PLUGIN_VAL"]["R"] is None for c in an):
        verdict = "INCOMPLETE"
    else:
        ok = sum(c["arms"]["PLUGIN_VAL"]["R"] >= 0.8 and c["arms"]["PLUGIN_VAL"]["pos"] >= 4 for c in an)
        bad = sum(c["arms"]["PLUGIN_VAL"]["R"] < 0.5 for c in an)
        verdict = "META_LEARNING_UNNECESSARY" if ok >= 2 else "META_LEARNING_ADDS_VALUE" if bad >= 2 else "PARTIAL"
    n_runs = len(v3_local) + len(autodl)
    out = dict(protocol_id=proto["protocol_id"], verdict=verdict, n_successful_v3_runs=n_runs, cells=cells)
    name = "summary_v3.json"
    if amend1:
        out["amendment"] = "Q1A_V3_AMEND1_PLATFORM"
        local = dict(v1_local, **v3_local)
        out["cross_platform"] = [dict(run_id=r, mse_autodl=autodl[r]["test_mse"], mse_local=local[r]["test_mse"],
                                      rel_diff=(autodl[r]["test_mse"] - local[r]["test_mse"]) / local[r]["test_mse"])
                                 for r in sorted(autodl) if r in local]
        name = "summary_v3_amend1.json"
    json.dump(out, open(os.path.join(HERE, "results", name), "w", encoding="utf-8"), indent=1)
    for c in cells:
        print(f"== {c['cell']}{'' if c['analysis'] else ' (descriptive)'}  g_QDF {c['g_QDF_mean']:+.4f}")
        for arm, a in c["arms"].items():
            R = a.get("R")
            if a["mse_mean"] is None:
                print(f"  {arm:14s} n=0")
                continue
            print(f"  {arm:14s} mse {a['mse_mean']:.5f}" + (f"  g {a['g_mean']:+.4f}  R {R:+.2f}  {a['pos']}/5 {a['band'] or ''}"
                                                           if R is not None and a.get("g_mean") is not None
                                                           else f"  n={a.get('n')}"))
        print("  selected a:", [e.get("selected_a") for e in c["selection"]], " oracle a:",
              [e.get("oracle_a_by_test") for e in c["selection"]])
        k = c["cost"]
        print(f"  cost: QDF meta {k['QDF_meta_phase_s']}s + train {k['QDF_meta_test_s']}s; S0 {k['S0_s']}s; "
              f"meta-free (S0 + 7 trainings) {k['meta_free_procedure_s']}s; injection {c['injection']}")
    if amend1:
        xs = [abs(x["rel_diff"]) for x in out["cross_platform"]]
        print(f"cross-platform: {len(xs)} paired runs, max |rel diff| {max(xs) if xs else None}")
    print("verdict:", verdict, f"({n_runs} v3 runs)")


if __name__ == "__main__":
    main(amend1="--amend1" in sys.argv)
