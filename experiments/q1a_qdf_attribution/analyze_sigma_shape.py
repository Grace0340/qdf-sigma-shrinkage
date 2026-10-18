"""Post hoc descriptive check of the meta-gradient derivation (no training).

For each (cell, seed): Sigma_l = learned Sigma of the v1 QDF run (A.pth); S0n recovered from the PLUGIN_A1
sigma_target.npy (Sigma = I + S0n). Reports how well Sigma_l - I is explained, with one free scalar, by
S0n, S0n + Diag(S0n) (one-step SGD prediction) and (I + S0n)^{1/2} - I (gradient-flow shape), and the
separate diagonal / off-diagonal slopes (derivation predicts diag/off ~ 2 for small steps).
Usage: python analyze_sigma_shape.py <dir with QDF_<cell>_<seed>.A.pth and S0src_<cell>_<seed>.npy>
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpu_launch  # noqa: E402
import torch  # noqa: E402

sys.meta_path.insert(0, cpu_launch._StubFinder())
sys.path.insert(0, os.path.join(HERE, "QDF"))


def fit(Y, B):
    c = float((Y * B).sum() / (B * B).sum())
    return c, float((Y - c * B).norm() / Y.norm())


def slope(y, x):
    return float((x * y).sum() / (x * x).sum())


def main(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "QDF_*.A.pth"))):
        tag = os.path.basename(f)[len("QDF_"):-len(".A.pth")]
        with torch.no_grad():
            S_l = torch.load(f, weights_only=False, map_location="cpu")().double()
        T = S_l.shape[0]
        I = torch.eye(T, dtype=torch.float64)
        S0n = torch.from_numpy(__import__("numpy").load(os.path.join(d, f"S0src_{tag}.npy"))).double() - I
        Y = S_l - I
        w, V = torch.linalg.eigh(I + S0n)
        sqrt_shape = V @ torch.diag(w.clamp_min(0).sqrt()) @ V.T - I
        off = ~torch.eye(T, dtype=torch.bool)
        r = dict(tag=tag, T=T, rel_dev_from_I=float(Y.norm() / T ** 0.5))
        for name, B in [("S0n", S0n), ("S0n_plus_diag", S0n + torch.diag(torch.diagonal(S0n))), ("sqrt_shape", sqrt_shape)]:
            c, e = fit(Y, B)
            r[f"fit_{name}"] = dict(c=c, rel_resid=e)
        c_d, c_o = slope(torch.diagonal(Y), torch.diagonal(S0n)), slope(Y[off], S0n[off])
        r.update(slope_diag=c_d, slope_off=c_o, diag_over_off=c_d / c_o if c_o else None)
        rows.append(r)
        print(f"{tag:16s} |S-I|/sqrtT {r['rel_dev_from_I']:.3f}  resid S0n {r['fit_S0n']['rel_resid']:.3f}  "
              f"S0n+Diag {r['fit_S0n_plus_diag']['rel_resid']:.3f}  sqrt {r['fit_sqrt_shape']['rel_resid']:.3f}  "
              f"diag/off {r['diag_over_off'] if r['diag_over_off'] is None else round(r['diag_over_off'], 2)}")
    json.dump(rows, open(os.path.join(d, "sigma_shape.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
