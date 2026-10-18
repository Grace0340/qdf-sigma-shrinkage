# Replays the released first-order meta phase (plain SGD on P, L = tril(P) + eps*I, P0 = I,
# meta learning rate halved every 100 steps) with a constant second moment kappa*S0n, and
# compares the resulting shape with the learned Sigma of each QDF run. kappa is the only free
# scalar; the first-order formula I + c(S0n + Diag S0n) and the square-root family
# (I + c S0n)^(1/2) get one free scalar each as well. Shape error: Frobenius distance between
# trace-normalised matrices, relative to the distance of the learned shape from I.
# No forecaster is trained or evaluated.
#   python paper/scripts/resim_meta.py [tags...]
import glob
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
EXP = os.path.join(PAPER, "..", "experiments", "q1a_qdf_attribution")
SHAPE = os.path.join(EXP, "remote", "shape_inputs")
OUT = os.path.join(PAPER, "generated")
sys.path.insert(0, EXP)

META = {("ETTh1", 96): 0.2, ("ETTh1", 720): 0.01, ("ETTh2", 96): 0.2, ("ETTh2", 720): 0.2}
STEPS, EPS = 300, 1e-6


def learned_sigma(path):
    # A.pth pickles the whole loss module; only its L_param is needed, so every QDF class is
    # replaced by a plain state holder instead of importing the QDF package.
    import pickle
    import torch

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
    T = P.shape[0]
    L = np.tril(P) + EPS * np.eye(T)
    return L @ L.T


def simulate(S, meta_lr, T):
    P = np.eye(T)
    for t in range(STEPS):
        eta = meta_lr * 0.5 ** (t // 100)
        L = np.tril(P) + EPS * np.eye(T)
        Li = solve_triangular(L, np.eye(T), lower=True)
        G = -(2.0 / T) * Li.T @ (Li @ S @ Li.T)
        P = P - eta * np.tril(G)
    L = np.tril(P) + EPS * np.eye(T)
    return L @ L.T


def nshape(M):
    return M * (M.shape[0] / np.trace(M))


def err(Sl, M):
    I = np.eye(Sl.shape[0])
    return float(np.linalg.norm(nshape(Sl) - nshape(M)) / np.linalg.norm(nshape(Sl) - I))


def best(f, lo, hi):
    r = minimize_scalar(lambda u: f(np.exp(u)), bounds=(np.log(lo), np.log(hi)), method="bounded",
                        options={"xatol": 1e-3})
    return float(np.exp(r.x)), float(r.fun)


def main(tags):
    rows = []
    for f in sorted(glob.glob(os.path.join(SHAPE, "QDF_*.A.pth"))):
        tag = os.path.basename(f)[4:-6]
        if tags and tag not in tags:
            continue
        ds, T, seed = tag.split("_")
        T = int(T)
        Sl = learned_sigma(f)
        S0n = np.load(os.path.join(SHAPE, f"S0src_{tag}.npy")).astype(np.float64) - np.eye(T)
        I = np.eye(T)
        w, V = np.linalg.eigh(I + S0n)
        lr = META[(ds, T)]
        B1 = S0n + np.diag(np.diag(S0n))
        c1, e1 = best(lambda c: err(Sl, I + c * B1), 1e-4, 1e3)
        cs, es = best(lambda c: err(Sl, (V * np.sqrt(np.clip(1 + c * (w - 1), 0, None))) @ V.T), 1e-4, 1e4)
        k, ek = best(lambda k: err(Sl, simulate(k * S0n, lr, T)), 1e-4, 1e3)
        rows.append(dict(tag=tag, T=T, dev_from_I=float(np.linalg.norm(nshape(Sl) - I) / np.sqrt(T)),
                         first_order=dict(c=c1, err=e1, spec=float(np.linalg.norm(c1 * B1, 2))),
                         sqrt_family=dict(c=cs, err=es), resim=dict(kappa=k, err=ek)))
        r = rows[-1]
        print(f"{tag:16s} dev {r['dev_from_I']:.3f} | 1st-order err {e1:.3f} (|term|_2 {r['first_order']['spec']:.2f})"
              f" | sqrt err {es:.3f} | resim err {ek:.3f} (kappa {k:.3g})", flush=True)
    name = "resim_meta.json" if not tags else "resim_meta_" + "_".join(tags) + ".json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
