"""Apply the frozen decision rules in protocol.json to results/runs.jsonl."""
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED_STUB_USE = {"cupy.random.seed", "cupy.random.seed()"}


def load_runs():
    best = {}
    for line in open(os.path.join(HERE, "results", "runs.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["returncode"] == 0 and r["test_mse"] is not None:
            best[r["run_id"]] = r
    return best


def sigma_stats(run_id, T):
    path = None
    for root, _, files in os.walk(os.path.join(HERE, "runs", run_id, "ck")):
        if "A.pth" in files:
            path = os.path.join(root, "A.pth")
    if path is None:
        return None
    sys.path.insert(0, HERE)
    import cpu_launch
    import torch
    if not any(isinstance(f, cpu_launch._StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, cpu_launch._StubFinder())
    qdf = os.path.join(HERE, "QDF")
    if qdf not in sys.path:
        sys.path.insert(0, qdf)
    A = torch.load(path, weights_only=False, map_location="cpu")
    with torch.no_grad():
        S = A().double()
    I = torch.eye(T, dtype=S.dtype)
    ev = torch.linalg.eigvalsh(S)
    d = torch.sqrt(torch.diagonal(S))
    C = S / (d[:, None] * d[None, :])
    off = C[~torch.eye(T, dtype=torch.bool)]
    return dict(rel_dev=float(torch.linalg.norm(S - I) / math.sqrt(T)), eig_min=float(ev[0]),
                eig_max=float(ev[-1]), mean_offdiag_corr=float(off.mean()), trace_over_T=float(S.trace() / T))


def main():
    proto = json.load(open(os.path.join(HERE, "protocol.json"), encoding="utf-8"))
    ref = proto["reported_reference"]
    runs = load_runs()
    g = proto["grid"]
    cells, verdicts = [], {}
    for ds in g["datasets"]:
        for T in g["pred_len"]:
            key = f"{ds}-{T}"
            rep_q, rep_b = ref["QDF_reported"][ds][str(T)], ref["TQNet_in_QDF_table"][ds][str(T)]
            g_rep = (rep_b - rep_q) / rep_b
            arms = {a: {} for a in ("QDF", "SIGMA_I", "TQNET_OFFICIAL")}
            issues = []
            for a in arms:
                for s in g["seeds"]:
                    r = runs.get(f"{a}_{ds}_{T}_{s}")
                    if r is None:
                        continue
                    arms[a][s] = r["test_mse"]
                    extra = set((r.get("stub_runtime_use") or {})) - ALLOWED_STUB_USE
                    if extra:
                        issues.append(f"{r['run_id']} used stubs {sorted(extra)}")
            win = {runs[k]["test_windows"] for k in runs if runs[k]["dataset"] == ds and runs[k]["pred_len"] == T}
            if len(win) > 1:
                issues.append(f"test window counts differ across runs: {sorted(win)}")
            paired = [s for s in g["seeds"] if s in arms["QDF"] and s in arms["SIGMA_I"]]
            g_sig = [(arms["SIGMA_I"][s] - arms["QDF"][s]) / arms["SIGMA_I"][s] for s in paired]
            paired_c = [s for s in g["seeds"] if s in arms["SIGMA_I"] and s in arms["TQNET_OFFICIAL"]]
            g_cfg = [(arms["TQNET_OFFICIAL"][s] - arms["SIGMA_I"][s]) / arms["TQNET_OFFICIAL"][s] for s in paired_c]
            complete = len(paired) == len(g["seeds"])
            if not complete:
                verdict = "INCOMPLETE"
            else:
                m, pos = st.mean(g_sig), sum(x > 0 for x in g_sig)
                if m >= 0.5 * g_rep and pos >= 4:
                    verdict = "SIGMA_EFFECT"
                elif abs(m) < 0.25 * g_rep or (pos < 4 and m < 0.5 * g_rep):
                    verdict = "NO_SIGMA_EFFECT"
                else:
                    verdict = "PARTIAL"
            verdicts[key] = verdict
            mean = lambda d: st.mean(d.values()) if d else None
            sd = lambda d: st.stdev(d.values()) if len(d) > 1 else None
            q_mean = mean(arms["QDF"])
            sig = [sigma_stats(f"QDF_{ds}_{T}_{s}", T) for s in sorted(arms["QDF"])]
            cells.append(dict(
                cell=key, g_rep=g_rep, verdict=verdict,
                mse_mean={a: mean(v) for a, v in arms.items()}, mse_sd={a: sd(v) for a, v in arms.items()},
                n_seeds={a: len(v) for a, v in arms.items()},
                g_sigma=g_sig, g_sigma_mean=st.mean(g_sig) if g_sig else None,
                g_cfg=g_cfg, g_cfg_mean=st.mean(g_cfg) if g_cfg else None,
                reproduction_gap=(abs(q_mean - rep_q) / rep_q > 0.03) if q_mean else None,
                qdf_reported=rep_q, tqnet_in_qdf_table=rep_b,
                tqnet_official_result_txt=ref["TQNet_official_result_txt"][ds][str(T)],
                sigma_stats=[x for x in sig if x], issues=issues))
    v = list(verdicts.values())
    if "INCOMPLETE" in v:
        overall = "INCOMPLETE"
    elif v.count("NO_SIGMA_EFFECT") >= 3 and verdicts.get("ETTh1-720") == "NO_SIGMA_EFFECT":
        overall = "CONTINUE_Q1_DIAGNOSTIC"
    elif v.count("SIGMA_EFFECT") >= 2 and verdicts.get("ETTh1-720") == "SIGMA_EFFECT":
        overall = "STOP_Q1_DIAGNOSTIC"
    else:
        overall = "INCONCLUSIVE"
    out = dict(protocol_id=proto["protocol_id"], overall=overall, cells=cells)
    json.dump(out, open(os.path.join(HERE, "results", "summary.json"), "w", encoding="utf-8"), indent=1)
    for c in cells:
        mm = c["mse_mean"]
        f = lambda x: f"{x:.4f}" if x is not None else "  -   "
        print(f"{c['cell']:10s} QDF {f(mm['QDF'])} SIGMA_I {f(mm['SIGMA_I'])} OFFICIAL {f(mm['TQNET_OFFICIAL'])} "
              f"g_rep {c['g_rep']:.3f} g_sigma {f(c['g_sigma_mean'])} g_cfg {f(c['g_cfg_mean'])} -> {c['verdict']}"
              + (f"  issues: {c['issues']}" if c["issues"] else ""))
    print("overall:", overall)


if __name__ == "__main__":
    main()
