"""Descriptive extras of protocol_v8 (post hoc, no forecaster trained): shape errors of the learned Sigma of
earlier QDF runs against the replay, the first-order formula and the square-root family, each with one fitted
scalar, as in paper/scripts/resim_meta.py. S0n comes from the PLUGIN_A1 run of the same seed (Sigma = I + S0n).
-> results/extras_v8.json
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.optimize import minimize_scalar

import fixed_sigma_patch as fp
import run_grid as g1
import run_grid_v5 as g5

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SETS = [("runs_v4", "TQNet", ds, 720, s) for ds in ("ETTh1", "ETTh2") for s in range(2021, 2026)]
SETS += [("runs_v5", "TQNet", ds, 720, s) for ds in ("ETTm1", "ETTm2", "Weather") for s in range(2021, 2024)]
SETS += [("runs_v7", bb, ds, T, s) for bb in ("DLinear", "iTransformer") for ds, T in (("ECL", 96), ("Weather", 96),
                                                                                       ("Weather", 720))
         for s in range(2021, 2026)]


def learned_sigma(path):
    class Holder:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state)

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith(("torch", "collections", "builtins", "_codecs", "numpy")):
                return super().find_class(mod, name)
            return Holder

    shim = type("shim", (), {"Unpickler": Unpickler, "load": pickle.load})
    A = torch.load(path, weights_only=False, map_location="cpu", pickle_module=shim)
    P = A._parameters["L_param"].detach().double().numpy()
    L = np.tril(P) + EPS * np.eye(P.shape[0])
    return L @ L.T


def released(ds, T):
    if ds.startswith("ETTh"):
        return g1.QDF_HP[(ds, T)]["lr_meta"], 300
    _, lr, warm, _, _ = g5.HP[(ds, T)]
    return lr, warm


def nshape(M):
    return M * (M.shape[0] / np.trace(M))


def err(Sl, M):
    I = np.eye(Sl.shape[0])
    return float(np.linalg.norm(nshape(Sl) - nshape(M)) / np.linalg.norm(nshape(Sl) - I))


def best(f, lo, hi):
    r = minimize_scalar(lambda u: f(np.exp(u)), bounds=(np.log(lo), np.log(hi)), method="bounded",
                        options={"xatol": 1e-3})
    return float(np.exp(r.x)), float(r.fun)


def one(runs, bb, ds, T, s):
    rid = f"{ds}_{T}_{s}" if bb == "TQNet" else f"{bb}_{ds}_{T}_{s}"
    a = glob.glob(os.path.join(HERE, runs, f"QDF_{rid}", "ck", "*", "A.pth"))
    p = glob.glob(os.path.join(HERE, runs, f"PLUGIN_A1_{rid}", "res", "*", "sigma_target.npy"))
    if not a or not p:
        return dict(run=f"{runs}/{rid}", missing=dict(A=bool(a), plugin=bool(p)))
    Sl = learned_sigma(a[0])
    I = np.eye(T)
    S0n = np.load(p[0]).astype(np.float64) - I
    lr, steps = released(ds, T)
    B1 = S0n + np.diag(np.diag(S0n))
    w, V = np.linalg.eigh(I + S0n)
    S0t = torch.tensor(S0n, device=DEV)
    sim = lambda k: fp._replay(k * S0t, lr, steps).cpu().numpy()
    c1, e1 = best(lambda c: err(Sl, I + c * B1), 1e-4, 1e3)
    cs, es = best(lambda c: err(Sl, (V * np.sqrt(np.clip(1 + c * (w - 1), 0, None))) @ V.T), 1e-4, 1e4)
    k, ek = best(lambda k: err(Sl, sim(k)), 1e-4, 1e3)
    dl = Sl - I
    off = ~np.eye(T, dtype=bool)
    slope = lambda m: float((dl[m] * S0n[m]).sum() / (S0n[m] ** 2).sum())
    return dict(run=f"{runs}/{rid}", backbone=bb, dataset=ds, T=T, seed=s, meta_lr=lr, steps=steps,
                dev_from_I=float(np.linalg.norm(nshape(Sl) - I) / np.sqrt(T)),
                trace_over_T=float(np.trace(Sl) / T), diag_over_off_slope=slope(np.eye(T, dtype=bool)) / slope(off),
                first_order=dict(c=c1, err=e1, spec=float(np.linalg.norm(c1 * B1, 2))),
                sqrt_family=dict(c=cs, err=es), replay=dict(kappa=k, err=ek))


def main():
    torch.set_num_threads(2)
    rows = []
    for job in SETS:
        r = one(*job)
        rows.append(r)
        if "missing" in r:
            print(r, flush=True)
        else:
            print(f"{r['run']:34s} dev {r['dev_from_I']:.3f} tr/T {r['trace_over_T']:.3f} | 1st {r['first_order']['err']:.3f}"
                  f" (|term| {r['first_order']['spec']:.1f}) | sqrt {r['sqrt_family']['err']:.3f}"
                  f" | replay {r['replay']['err']:.3f} (kappa {r['replay']['kappa']:.3g})", flush=True)
    json.dump(rows, open(os.path.join(HERE, "results", "extras_v8.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
