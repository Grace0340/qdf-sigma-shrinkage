"""Apply protocol_v6.json decision rules (v6 runs + v5 QDF/SIGMA_I comparators, GPU only)."""
import json
import os
import sys

import summarize_v5 as s5

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    proto = json.load(open(os.path.join(HERE, "protocol_v6.json"), encoding="utf-8"))
    v5, v6 = s5.load(), s5.load(os.path.join(HERE, "results", "runs_v6.jsonl"))
    src = dict(v5, **v6)
    seeds, arms = proto["seeds"], list(proto["arms"])
    cells, complete, rel_learned, closer, n_shape = [], True, [], 0, 0
    for cell in proto["cells"]:
        mse = lambda arm: {s: src[s5.rid(arm, cell, s)]["test_mse"] for s in seeds if s5.rid(arm, cell, s) in src}
        base, q = mse("SIGMA_I"), mse("QDF")
        gain = lambda m: {s: (base[s] - m[s]) / base[s] for s in m if s in base}
        gq = gain(q)
        gQ = s5.mean(gq.values())
        row = dict(cell=cell, g_QDF_mean=gQ, g_QDF_per_seed=[gq.get(s) for s in seeds], arms={})
        for arm in arms:
            g = gain(mse(arm))
            complete &= len(g) == len(seeds)
            m = s5.mean(g.values())
            R = m / gQ if len(g) == len(seeds) and gQ else None
            row["arms"][arm] = dict(g_mean=m, g_per_seed=[g.get(s) for s in seeds], pos=sum(x > 0 for x in g.values()),
                                    R=R, band=s5.band(R))
        ml = mse("LEARNED")
        row["rel_learned"] = [abs(ml[s] - q[s]) / q[s] if s in ml and s in q else None for s in seeds]
        rel_learned += [x for x in row["rel_learned"] if x is not None]
        shape = []
        for s in seeds:
            r = src.get(s5.rid("PLUGIN_DIAG_MATCHED", cell, s), {}).get("sigma_target") or {}
            r0 = src.get(s5.rid("PLUGIN_MATCHED", cell, s), {}).get("sigma_target") or {}
            if "cos_learned_S0" in r:
                n_shape += 1
                closer += r["cos_learned_S0diag"] > r["cos_learned_S0"]
            shape.append(dict(seed=s, cos_S0=r.get("cos_learned_S0"), cos_S0diag=r.get("cos_learned_S0diag"),
                              relfro_plugin=r0.get("relfro_learned_minus_target"),
                              relfro_plugin_diag=r.get("relfro_learned_minus_target"), c_diag=r.get("c"), c=r0.get("c")))
        row["shape"] = shape
        cells.append(row)
    verdicts = {}
    if not complete:
        verdicts = {q: "INCOMPLETE" for q in ("Q6.0", "Q6.1", "Q6.2")}
    else:
        verdicts["Q6.0"] = "PIPELINE_OK" if max(rel_learned) <= 0.005 else "PIPELINE_DEVIATES"
        for q, arm in (("Q6.1", "PLUGIN_MATCHED"), ("Q6.2", "PLUGIN_DIAG_MATCHED")):
            rec = sum(c["arms"][arm]["R"] >= 0.8 and c["arms"][arm]["pos"] == len(seeds) for c in cells)
            fail = sum(c["arms"][arm]["R"] < 0.5 for c in cells)
            verdicts[q] = "RECOVERS" if rec >= 3 else "FAILS" if fail >= 3 else "PARTIAL"
        verdicts["Q6.2_shape_closer"] = f"{closer}/{n_shape}"
    out = dict(protocol_id=proto["protocol_id"], verdicts=verdicts, max_rel_learned=max(rel_learned, default=None),
               n_v6_runs=len(v6), cells=cells)
    json.dump(out, open(os.path.join(HERE, "results", "summary_v6.json"), "w", encoding="utf-8"), indent=1)
    fmt = lambda x, p="+.4f": "None" if x is None else format(x, p)
    for c in cells:
        print(f"== {c['cell']}  g_QDF {fmt(c['g_QDF_mean'])}  rel_learned {[fmt(x, '.1e') for x in c['rel_learned']]}")
        for arm, a in c["arms"].items():
            print(f"  {arm:20s} g {fmt(a['g_mean'])}  R {fmt(a['R'], '+.2f')}  {a['pos']}/{len(seeds)} {a['band'] or ''}")
        for s in c["shape"]:
            print("  shape", json.dumps(s))
    print("verdicts:", json.dumps(verdicts), f"({len(v6)} v6 runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
