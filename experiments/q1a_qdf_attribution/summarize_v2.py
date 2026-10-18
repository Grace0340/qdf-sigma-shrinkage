"""Apply protocol_v2.json decision rules to results/runs.jsonl (v1 QDF, SIGMA_I) and results/runs_v2.jsonl."""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("PLUGIN", "SCALAR", "DIAG", "TOPK8")
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


def main():
    proto = json.load(open(os.path.join(HERE, "protocol_v2.json"), encoding="utf-8"))
    v1, v2 = load("runs.jsonl"), load("runs_v2.jsonl")
    seeds = proto["grid"]["seeds"]
    cells, checks = [], []
    for ds in proto["grid"]["datasets"]:
        for T in proto["grid"]["pred_len"]:
            key = f"{ds}-{T}"
            r = v2.get(f"REINJECT_{ds}_{T}_2021")
            q = v1.get(f"QDF_{ds}_{T}_2021")
            if r and q:
                rel = abs(r["test_mse"] - q["test_mse"]) / q["test_mse"]
                checks.append(dict(cell=key, reinject=r["test_mse"], qdf_v1=q["test_mse"], rel_diff=rel, ok=rel <= 0.005))
            base = {s: v1[f"SIGMA_I_{ds}_{T}_{s}"]["test_mse"] for s in seeds if f"SIGMA_I_{ds}_{T}_{s}" in v1}
            g = lambda arm, src: {s: (base[s] - src[f"{arm}_{ds}_{T}_{s}"]["test_mse"]) / base[s]
                                  for s in seeds if s in base and f"{arm}_{ds}_{T}_{s}" in src}
            gq = g("QDF", v1)
            row = dict(cell=key, analysis=key in ANALYSIS, g_QDF_mean=st.mean(gq.values()) if gq else None, arms={})
            for arm in ARMS:
                ga = g(arm, v2)
                complete = len(ga) == len(seeds)
                m = st.mean(ga.values()) if ga else None
                R = m / row["g_QDF_mean"] if complete and row["g_QDF_mean"] else None
                mse = [v2[f"{arm}_{ds}_{T}_{s}"]["test_mse"] for s in seeds if f"{arm}_{ds}_{T}_{s}" in v2]
                row["arms"][arm] = dict(n=len(ga), g_mean=m, g_per_seed=[ga[s] for s in sorted(ga)],
                                        pos=sum(x > 0 for x in ga.values()), R=R, band=band(R),
                                        mse_mean=st.mean(mse) if mse else None)
            cells.append(row)
    an = [c for c in cells if c["analysis"]]
    if any(c["arms"]["PLUGIN"]["R"] is None for c in an) or len(checks) < 4:
        verdict = "INCOMPLETE"
    elif not all(c["ok"] for c in checks):
        verdict = "VOID_INJECTION_CHECK_FAILED"
    else:
        suff = sum(c["arms"]["PLUGIN"]["R"] >= 0.8 and c["arms"]["PLUGIN"]["pos"] >= 4 for c in an)
        fail = sum(c["arms"]["PLUGIN"]["R"] < 0.5 for c in an)
        verdict = "PLUGIN_SUFFICES" if suff >= 2 else "PLUGIN_FAILS" if fail >= 2 else "PARTIAL"
    out = dict(protocol_id=proto["protocol_id"], verdict=verdict, injection_checks=checks, cells=cells)
    json.dump(out, open(os.path.join(HERE, "results", "summary_v2.json"), "w", encoding="utf-8"), indent=1)
    for c in checks:
        print(f"check {c['cell']}: reinject {c['reinject']:.5f} vs v1 QDF {c['qdf_v1']:.5f} ({c['rel_diff']:.2%}) {'ok' if c['ok'] else 'FAIL'}")
    for c in cells:
        line = f"{c['cell']:10s}{'' if c['analysis'] else ' (descr.)'} g_QDF {c['g_QDF_mean']:+.4f}"
        for arm, a in c["arms"].items():
            line += f" | {arm} g {a['g_mean']:+.4f} R {a['R']:+.2f} {a['pos']}/5" if a["R"] is not None else f" | {arm} n={a['n']}"
        print(line)
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
