"""Toy sanity checks for paper/derivations/qdf_meta_gradient.md.

Re-implements (does not import) the L-parametrisation and loss of
QDF exp/exp_long_term_forecasting_meta_ml3.py @ eb0693a9
(CovarianceMatrix._get_L lines 40-55, get_loss lines 94-129 with meta_type='all',
auxi_loss='MSE', reg_lambda=0) and the first-order inner loop (lines 208-237).
CPU, single thread, float64, T=8. Runs in a few seconds.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

T = 8
EPS = 1e-6
t0 = time.time()


def get_L(P):
    L = torch.tril(P)
    diag = torch.diag_embed(torch.diagonal(L) + EPS)
    return L - torch.diag_embed(torch.diagonal(L)) + diag


def qdf_loss(P, E):
    """E: [N, T] residual rows (N = B*D). Returns mean(x**2), x = L^{-1} e."""
    L = get_L(P)
    x = torch.linalg.solve_triangular(L, E.T, upper=False).T
    return torch.mean(x ** 2)


def formula_grad(P, E):
    L = get_L(P).detach()
    S = E.T @ E / E.shape[0]
    Linv = torch.linalg.inv(L)
    Sigma_inv = Linv.T @ Linv
    G = -(2.0 / T) * Sigma_inv @ S @ Linv.T
    return torch.tril(G), S


def report(name, val, tol):
    ok = val < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {val:.3e} (tol {tol:.0e})")
    return ok


all_ok = True

# A. gradient at L = I and at a generic lower-triangular L
N = 500
A = torch.randn(T, T)
E = torch.randn(N, T) @ A.T  # correlated residuals
P = torch.eye(T).requires_grad_(True)
qdf_loss(P, E).backward()
G_formula, S = formula_grad(P, E)
rel = lambda X_, Y_: ((X_ - Y_).norm() / Y_.norm()).item()
all_ok &= report("A0 grad at L=I+eps*I vs tril(-(2/T) Sigma^-1 S L^-T)",
                 (P.grad - G_formula).abs().max().item(), 1e-10)
all_ok &= report("A1 grad at L=I vs -(2/T) tril(S) (rel.; differs only by eps=1e-6)",
                 rel(P.grad, -(2.0 / T) * torch.tril(S)), 1e-5)
all_ok &= report("A1' upper-triangular grad entries are zero",
                 torch.triu(P.grad, 1).abs().max().item(), 1e-15)

P2 = (torch.eye(T) * 1.5 + 0.3 * torch.tril(torch.randn(T, T), -1)).requires_grad_(True)
qdf_loss(P2, E).backward()
G2, _ = formula_grad(P2, E)
all_ok &= report("A2 grad at generic L vs tril(-(2/T) Sigma^-1 S L^-T)",
                 (P2.grad - G2).abs().max().item(), 1e-10)

# G. Euler identity <P, grad> = -2 J (degree -2 homogeneity; eps-level error)
with torch.no_grad():
    J2 = qdf_loss(P2, E).item()
all_ok &= report("G  <L, grad_P J> + 2J (Euler identity)",
                 abs((get_L(P2.detach()) * P2.grad).sum().item() + 2 * J2), 1e-10)

# B. one plain-SGD step from P = I (optim.SGD, momentum 0) and the exact Sigma_1
eta = 0.2
P_sgd = torch.eye(T).requires_grad_(True)
opt = torch.optim.SGD([P_sgd], lr=eta)
opt.zero_grad()
qdf_loss(P_sgd, E).backward()
opt.step()
with torch.no_grad():
    L1 = get_L(P_sgd)
    Sigma1 = L1 @ L1.T
    a = 2 * eta / T
    trilS = torch.tril(S)
    Sigma1_pred = torch.eye(T) + a * (S + torch.diag(torch.diag(S))) + a ** 2 * trilS @ trilS.T
    all_ok &= report("B1 Sigma_1 = I + a(S + Diag S) + a^2 tril(S)tril(S)^T (rel.)",
                     rel(Sigma1, Sigma1_pred), 1e-5)
    DS = torch.diag(torch.diag(S))
    print(f"     info: ||Diag S||_F / ||S||_F = {(DS.norm() / S.norm()).item():.3f}"
          " (size of the extra diagonal term relative to the S term)")

# D. Adam (not used by released scripts) first step is lr*sign(g) up to eps
P_adam = torch.eye(T).requires_grad_(True)
opt = torch.optim.Adam([P_adam], lr=eta)
opt.zero_grad()
qdf_loss(P_adam, E).backward()
g = P_adam.grad.clone()
opt.step()
with torch.no_grad():
    step = P_adam - torch.eye(T)
    mask = torch.tril(torch.ones(T, T)).bool()
    all_ok &= report("D  Adam first step = -lr*sign(g) on lower triangle",
                     (step[mask] - (-eta * torch.sign(g[mask]))).abs().max().item(), 1e-6)

# C. first-order inner loop: L-gradient flows only through the explicit query loss
d_in, H = 6, 16


def mlp(params, X):
    h = torch.tanh(X @ params["W1"].T + params["b1"])
    return h @ params["W2"].T + params["b2"]


def clip_grads(grads, max_norm):
    total = torch.norm(torch.stack([gg.norm() for gg in grads]))
    if total > max_norm:
        s = max_norm / (total + 1e-6)
        return [gg * s for gg in grads]
    return grads


theta0 = {"W1": torch.randn(H, d_in) * 0.5, "b1": torch.zeros(H),
          "W2": torch.randn(T, H) * 0.5, "b2": torch.zeros(T)}
for v in theta0.values():
    v.requires_grad_(True)
Xs, Ys = torch.randn(64, d_in), torch.randn(64, T)
Xq, Yq = torch.randn(64, d_in), torch.randn(64, T)
P_meta0 = (torch.eye(T) + 0.1 * torch.tril(torch.randn(T, T))).detach()


def meta_grad(first_order, n_inner=3, inner_lr=0.05, max_norm=5.0):
    Pm = P_meta0.clone().requires_grad_(True)
    fast = {k: v.clone() for k, v in theta0.items()}
    for _ in range(n_inner):
        loss = qdf_loss(Pm, mlp(fast, Xs) - Ys)
        grads = torch.autograd.grad(loss, list(fast.values()),
                                    create_graph=not first_order, allow_unused=True)
        grads = clip_grads(list(grads), max_norm)
        fast = {k: v - inner_lr * gg for (k, v), gg in zip(fast.items(), grads)}
    e_q = mlp(fast, Xq) - Yq
    qdf_loss(Pm, e_q).backward()
    return Pm.grad.clone(), e_q.detach(), fast


g_fo, e_fo, fast_fo = meta_grad(first_order=True)
g_ref, _ = formula_grad(P_meta0, e_fo)
all_ok &= report("C1 first-order meta-grad == explicit formula with adapted residual",
                 (g_fo - g_ref).abs().max().item(), 1e-10)
g_so, _, _ = meta_grad(first_order=False)
print(f"     info: |second-order - first-order| max = {(g_so - g_fo).abs().max().item():.3e}"
      " (nonzero: the extra path exists only when create_graph=True)")
with torch.no_grad():
    e0 = mlp(theta0, Xq) - Yq
    dtheta = torch.sqrt(sum(((fast_fo[k] - theta0[k]) ** 2).sum() for k in theta0)).item()
    print(f"     info: ||theta_K - theta_0|| = {dtheta:.4f} <= K*inner_lr*max_norm = {3 * 0.05 * 5.0:.4f};"
          f" ||e_K - e_0||_F/||e_0||_F = {((e_fo - e0).norm() / e0.norm()).item():.4f}")
all_ok &= report("C2 ||theta_K - theta_0|| - K*inner_lr*max_norm <= 0",
                 max(0.0, dtheta - 3 * 0.05 * 5.0), 1e-12)

# E. Kruskal/Zellner: fixed-W weighted LS with common regressors equals OLS
p, n = 5, 200
X = torch.randn(p, n)
Btrue = torch.randn(T, p)
Y = Btrue @ X + A @ torch.randn(T, n)
B_ols = Y @ X.T @ torch.linalg.inv(X @ X.T)
M = torch.randn(T, T)
W = M @ M.T + 0.1 * torch.eye(T)
Wh = torch.linalg.cholesky(W)
# GLS in vec form: minimise ||(I_n kron Wh^T) (vec Y - (X^T kron I_T) vec B)||^2
Z = torch.kron(X.T.contiguous(), torch.eye(T))
Om_half = torch.kron(torch.eye(n), Wh.T)
vecY = Y.T.reshape(-1)
vecB = torch.linalg.lstsq(Om_half @ Z, Om_half @ vecY).solution
B_gls = vecB.reshape(p, T).T
all_ok &= report("E  W-weighted LS (vec/GLS form) == OLS for random PD W",
                 (B_gls - B_ols).abs().max().item(), 1e-9)

# F. diagonal case, small-step SGD ~ gradient flow: Sigma_ii(t) = sqrt(1 + 8 s_i t / T)
s = torch.linspace(0.2, 3.0, T)
Ed = torch.randn(4000, T)
Ed = Ed / Ed.std(0, keepdim=True) * torch.sqrt(s)
Sd = Ed.T @ Ed / Ed.shape[0]
sdiag = torch.diag(Sd)
Pd = torch.eye(T)
h, steps = 0.05, 2000
for _ in range(steps):
    Ld = torch.diag(torch.diagonal(Pd))
    Pd = Pd + h * (2.0 / T) * torch.diag(sdiag / torch.diagonal(Ld) ** 3)
t_end = h * steps
pred = torch.sqrt(1 + 8 * sdiag * t_end / T)
all_ok &= report("F  diagonal flow Sigma_ii(t) vs sqrt(1+8 s_i t/T) (rel.)",
                 ((torch.diagonal(Pd) ** 2 - pred) / pred).abs().max().item(), 1e-3)
shape = torch.diagonal(Pd) ** 2
corr_sqrt = torch.corrcoef(torch.stack([shape / shape.mean(), torch.sqrt(sdiag)]))[0, 1].item()
print(f"     info: at t={t_end}, corr(Sigma_ii, sqrt(s_i)) = {corr_sqrt:.4f}")

# I. second-order expansion of Sigma(tau) under gradient flow with constant S,
#    tau = (2/T) * cumulative step size:  Sigma = I + tau(S+D) + tau^2 R + O(tau^3)
S1 = S / torch.linalg.matrix_norm(S, 2)
U = torch.tril(S1)
D1 = torch.diag(torch.diag(S1))
Mm = (S1 + D1) @ S1 + S1 @ U.T
R = U @ U.T - 0.5 * (torch.tril(Mm) + torch.tril(Mm).T)


def flow_sigma(tau, h=1e-4):
    Pf = torch.eye(T)
    for _ in range(int(round(tau / h))):
        Lf = torch.tril(Pf)
        Li = torch.linalg.inv(Lf)
        Pf = Pf + h * torch.tril(Li.T @ Li @ S1 @ Li.T)
    Lf = torch.tril(Pf)
    return Lf @ Lf.T


errs = []
for tau in (0.1, 0.05):
    Sg = flow_sigma(tau)
    e1 = (Sg - (torch.eye(T) + tau * (S1 + D1))).norm().item()
    e2 = (Sg - (torch.eye(T) + tau * (S1 + D1) + tau ** 2 * R)).norm().item()
    errs.append(e2)
    print(f"     info: tau={tau}: ||Sigma - 1st-order||_F = {e1:.2e}, ||Sigma - 2nd-order||_F = {e2:.2e}")
all_ok &= report("I  2nd-order remainder ratio err(0.1)/err(0.05) ~ 8 (O(tau^3)); |ratio-8|",
                 abs(errs[0] / errs[1] - 8.0), 1.0)

# H. Adam loss-scale invariance (meta-test optimiser): one step with loss*c vs loss
w0 = torch.randn(10)
outs = []
for c in (1.0, 37.0):
    w = w0.clone().requires_grad_(True)
    o = torch.optim.Adam([w], lr=1e-2)
    for _ in range(5):
        o.zero_grad()
        (c * ((w - 1.0) ** 2 * torch.arange(1, 11)).sum()).backward()
        o.step()
    outs.append(w.detach())
all_ok &= report("H  Adam trajectory invariant to loss scale c=37 (5 steps)",
                 (outs[0] - outs[1]).abs().max().item(), 1e-6)

# eta_tot implied by meta_train's type1 schedule (lr halves after every 100 steps)
print("\neta_tot = sum_t eta_t for the released schedules:")
for steps_ in (200, 300, 500, 700):
    tot = sum(0.5 ** ((t - 1) // 100) for t in range(1, steps_ + 1))
    print(f"  warmup_steps={steps_}: eta_tot = {tot:.4f} * meta_lr")

print(f"\nALL PASS: {bool(all_ok)}   wall time {time.time() - t0:.2f}s")
