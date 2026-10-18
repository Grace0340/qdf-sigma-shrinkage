"""Replace QDF's learned Sigma by a fixed target before the meta-test phase.

Activated by cpu_launch.py when Q1_SIGMA_MODE is set. The meta phase still runs
(callers pass --meta_lr 0), so model init and batch order match the QDF run with
the same seed. Afterwards L is set to chol(Sigma_target) and A.pth is rewritten.

Q1_SIGMA_MODE:
  learned   Sigma = learned Sigma from Q1_LEARNED_A (re-injection check)
  scalar    Sigma = (tr(S_l)/T) I
  diag      Sigma = diag(S_l)
  topk      Sigma = I + V_k (Lambda_k - 1) V_k^T from S_l, k = Q1_TOPK (default 8)
  plugin    Sigma = I + c S0, S0 = residual second moment of the initial model on the
            training set (eval mode, sequential); c = (tr(S_l) - T) / tr(S0)
  plugin_diag  Sigma = I + c (S0 + Diag S0), c = (tr(S_l) - T) / (2 tr(S0)) (one-step meta-gradient shape)
  plugin_abs  Sigma = I + a S0n, S0n = S0 T / tr(S0), a = Q1_PLUGIN_A
  fgls      Sigma = S0n + r I, r = Q1_FGLS_RIDGE (default 1e-3)
  covy_abs  Sigma = I + a Cyn, Cy = centered label covariance on the training set, Cyn = Cy T / tr(Cy),
            a = Q1_PLUGIN_A
  lw        Ledoit-Wolf shrinkage of S0 towards (tr(S0)/T) I with data-driven delta:
            Sigma = delta I + (1 - delta) S0n (the LW estimate divided by tr(S0)/T)
  spectrum_random  eigenvalues of I + a S0n (a = Q1_PLUGIN_A) with Haar-random eigenvectors
            (generator seeded by Q1_ROT_SEED, global RNG untouched)
  plugin_diag_abs  Sigma = I + a D0n, D0n = (S0n + Diag S0n) / 2 (trace T), a = Q1_PLUGIN_A
  replay    Sigma = the released first-order meta phase replayed without a forecaster: plain SGD on P from
            P0 = I, L = tril(P) + 1e-6 I, gradient tril(-(2/T) L^-T L^-1 (k S0) L^-T), Q1_REPLAY_STEPS steps,
            learning rate Q1_REPLAY_META_LR halved every 100 steps, k = Q1_REPLAY_K

Only learned/scalar/diag/topk/plugin/plugin_diag read Q1_LEARNED_A.
"""
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import default_collate


def _residual_moment(exp, T):
    ds, _ = exp._get_data(flag="train")
    was_training = exp.model.training
    exp.model.eval()
    acc, n = torch.zeros(T, T, dtype=torch.float64), 0
    with torch.no_grad():
        for i in range(0, len(ds), 256):
            bx, by, bxm, bym, bc = default_collate([ds[j] for j in range(i, min(i + 256, len(ds)))])
            out, y, _ = exp.forward_step(bx, by, bxm, bym, bc)
            e = (out - y).permute(0, 2, 1).reshape(-1, T).double()
            acc += (e.T @ e).cpu()
            n += e.shape[0]
    exp.model.train(was_training)
    return acc / n


def _moments(exp, T):
    """Same pass as _residual_moment, also accumulating label moments and sum ||e||^4."""
    ds, _ = exp._get_data(flag="train")
    was_training = exp.model.training
    exp.model.eval()
    ee, yy = torch.zeros(T, T, dtype=torch.float64), torch.zeros(T, T, dtype=torch.float64)
    ys, e4, n = torch.zeros(T, dtype=torch.float64), 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ds), 256):
            bx, by, bxm, bym, bc = default_collate([ds[j] for j in range(i, min(i + 256, len(ds)))])
            out, y, _ = exp.forward_step(bx, by, bxm, bym, bc)
            e = (out - y).permute(0, 2, 1).reshape(-1, T).double()
            yv = y.permute(0, 2, 1).reshape(-1, T).double()
            ee += (e.T @ e).cpu()
            yy += (yv.T @ yv).cpu()
            ys += yv.sum(0).cpu()
            e4 += float((e * e).sum(1).pow(2).sum())
            n += e.shape[0]
    exp.model.train(was_training)
    mu = ys / n
    return ee / n, yy / n - torch.outer(mu, mu), yy / n, e4, n


def _cos(A, B):
    return float((A * B).sum() / (A.norm() * B.norm()))


def _replay(S, meta_lr, steps, eps=1e-6):
    """Plain-SGD meta phase of the released code on a constant moment S (float64, S's device)."""
    T = S.shape[0]
    I = torch.eye(T, dtype=S.dtype, device=S.device)
    P = I.clone()
    for t in range(steps):
        eta = meta_lr * 0.5 ** (t // 100)
        L = torch.tril(P) + eps * I
        Li = torch.linalg.solve_triangular(L, I, upper=False)
        G = -(2.0 / T) * Li.T @ (Li @ S @ Li.T)
        P = P - eta * torch.tril(G)
    L = torch.tril(P) + eps * I
    return L @ L.T


NEEDS_LEARNED = {"learned", "scalar", "diag", "topk", "plugin", "plugin_diag"}
V4_MODES = {"covy_abs", "lw", "spectrum_random"}


def _timed_moment(exp, T, info):
    t0 = time.time()
    S0 = _residual_moment(exp, T)
    info["s0_seconds"] = round(time.time() - t0, 2)
    info["S0_trace_over_T"] = float(S0.trace() / T)
    return S0


def _target(exp, mode, T):
    I = torch.eye(T, dtype=torch.float64)
    info = {"mode": mode}
    if mode in NEEDS_LEARNED:
        A = torch.load(os.environ["Q1_LEARNED_A"], weights_only=False, map_location="cpu")
        with torch.no_grad():
            S_l = A().double()
        info["learned_trace_over_T"] = float(S_l.trace() / T)
    if mode in V4_MODES:
        t0 = time.time()
        S0, Cy, My, e4, n = _moments(exp, T)
        info["s0_seconds"] = round(time.time() - t0, 2)
        S0n = S0 * T / S0.trace()
        info.update(S0_trace_over_T=float(S0.trace() / T), Cy_trace_over_T=float(Cy.trace() / T),
                    My_trace_over_T=float(My.trace() / T), cos_S0_Cy=_cos(S0, Cy), cos_S0_My=_cos(S0, My),
                    relfro_S0_minus_My=float((S0 - My).norm() / My.norm()), n_vectors=n)
    if mode == "covy_abs":
        a = float(os.environ["Q1_PLUGIN_A"])
        S = I + a * (Cy * T / Cy.trace())
        info["a"] = a
    elif mode == "lw":
        mu = S0.trace() / T
        d2 = float((S0 - mu * I).pow(2).sum())
        b2 = max((e4 - n * float(S0.pow(2).sum())) / n ** 2, 0.0)
        delta = min(1.0, b2 / d2)
        info.update(lw_delta=delta, lw_b2=b2, lw_d2=d2)
        if delta < 1e-6:
            S = S0n + 1e-3 * I
            info["lw_fallback"] = "fgls_ridge_1e-3"
        else:
            info["a_equiv"] = (1 - delta) / delta
            S = delta * I + (1 - delta) * S0n
    elif mode == "spectrum_random":
        a = float(os.environ["Q1_PLUGIN_A"])
        w = torch.linalg.eigvalsh(I + a * S0n)
        g = torch.Generator().manual_seed(int(os.environ["Q1_ROT_SEED"]))
        Q, R = torch.linalg.qr(torch.randn(T, T, generator=g, dtype=torch.float64))
        Q = Q * torch.sign(torch.diagonal(R))
        S = Q @ torch.diag(w) @ Q.T
        w2 = torch.linalg.eigvalsh((S + S.T) / 2)
        info.update(a=a, rot_seed=int(os.environ["Q1_ROT_SEED"]),
                    spectrum_max_rel_err=float((w2 - w).abs().max() / w.abs().max()))
    elif mode == "plugin_abs":
        a = float(os.environ["Q1_PLUGIN_A"])
        S0 = _timed_moment(exp, T, info)
        S = I + a * (S0 * T / S0.trace())
        info["a"] = a
    elif mode == "plugin_diag_abs":
        a = float(os.environ["Q1_PLUGIN_A"])
        S0 = _timed_moment(exp, T, info)
        S0n = S0 * T / S0.trace()
        S = I + a * (S0n + torch.diag(torch.diagonal(S0n))) / 2
        info["a"] = a
    elif mode == "replay":
        k = float(os.environ["Q1_REPLAY_K"])
        meta_lr, steps = float(os.environ["Q1_REPLAY_META_LR"]), int(os.environ["Q1_REPLAY_STEPS"])
        S0 = _timed_moment(exp, T, info)
        t0 = time.time()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        S = _replay((k * S0).to(dev), meta_lr, steps).cpu()
        info.update(k=k, meta_lr=meta_lr, steps=steps, replay_seconds=round(time.time() - t0, 2),
                    replay_device=dev, replay_finite=bool(torch.isfinite(S).all()))
    elif mode == "fgls":
        r = float(os.environ.get("Q1_FGLS_RIDGE", "1e-3"))
        S0 = _timed_moment(exp, T, info)
        S = S0 * T / S0.trace() + r * I
        info["ridge"] = r
    elif mode == "learned":
        S = S_l
    elif mode == "scalar":
        S = (S_l.trace() / T) * I
    elif mode == "diag":
        S = torch.diag(torch.diagonal(S_l))
    elif mode == "topk":
        k = int(os.environ.get("Q1_TOPK", "8"))
        w, V = torch.linalg.eigh(S_l)
        w, V = w[-k:], V[:, -k:]
        S = I + V @ torch.diag(w - 1) @ V.T
        info["k"] = k
    elif mode in ("plugin", "plugin_diag"):
        S0 = _timed_moment(exp, T, info)
        D = S0 + torch.diag(torch.diagonal(S0)) if mode == "plugin_diag" else S0
        c = float((S_l.trace() - T) / D.trace())
        S = I + c * D
        info["c"] = c
        dl = S_l - I
        info.update(cos_learned_S0=_cos(dl, S0), cos_learned_S0diag=_cos(dl, S0 + torch.diag(torch.diagonal(S0))),
                    relfro_learned_minus_target=float((S_l - S).norm() / dl.norm()))
    else:
        raise ValueError(mode)
    S = (S + S.T) / 2
    ev = torch.linalg.eigvalsh(S)
    info.update(target_trace_over_T=float(S.trace() / T), target_eig_min=float(ev[0]), target_eig_max=float(ev[-1]))
    return S, info


def install():
    from exp import exp_long_term_forecasting_meta_ml3 as m

    orig = m.Exp_Long_Term_Forecast_META_ML3.meta_train

    def meta_train(self, support_loader_list, query_loader_list, path, res_path):
        orig(self, support_loader_list, query_loader_list, path, res_path)
        mode = os.environ["Q1_SIGMA_MODE"]
        rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        S, info = _target(self, mode, self.pred_len)
        torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
        L = torch.linalg.cholesky(S).float()
        with torch.no_grad():
            self.A.L_param.copy_(L)
            got = self.A().double().cpu()
        info["inject_max_abs_err"] = float((got - S).abs().max())
        torch.save(self.A, os.path.join(path, "A.pth"))
        np.save(os.path.join(res_path, "sigma_target.npy"), S.numpy())
        with open(os.path.join(res_path, "sigma_target.json"), "w") as f:
            json.dump(info, f, indent=1)
        print("fixed sigma injected:", json.dumps(info))

    m.Exp_Long_Term_Forecast_META_ML3.meta_train = meta_train
