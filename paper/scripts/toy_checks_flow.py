# Toy checks (T = 8, float64) of the trace increment in Proposition 1 and of the gradient-flow
# statements in Proposition 3. Small-step SGD stands in for the flow. No forecaster is involved.
#   python paper/scripts/toy_checks_flow.py
import numpy as np
from scipy.linalg import solve_triangular, sqrtm

rng = np.random.default_rng(0)
T = 8
A = rng.standard_normal((T, 3 * T))
S = A @ A.T / (3 * T) + 0.05 * np.eye(T)


def J_and_grad(L):
    Li = solve_triangular(L, np.eye(T), lower=True)
    C = Li @ S @ Li.T
    return np.trace(C) / T, np.tril(-(2.0 / T) * Li.T @ C)


# J: one SGD step increases tr(L L^T) by exactly 4 eta J + eta^2 ||grad||_F^2
L = np.tril(rng.standard_normal((T, T))) + 3 * np.eye(T)
J, G = J_and_grad(L)
eta = 0.37
lhs = np.linalg.norm(L - eta * G) ** 2 - np.linalg.norm(L) ** 2
rhs = 4 * eta * J + eta ** 2 * np.linalg.norm(G) ** 2
print(f"[J] trace increment identity, relative error {abs(lhs - rhs) / rhs:.1e}")

# K: F = J tr(Sigma) is non-increasing along small-step descent from L = I, and the
#    normalised shape approaches S^{1/2} / tr S^{1/2}
L = np.eye(T)
target = np.real(sqrtm(S))
target /= np.trace(target)
F_prev, worst, eta = np.inf, -np.inf, 1e-2
for k in range(200000):
    J, G = J_and_grad(L)
    F = J * np.linalg.norm(L) ** 2
    worst = max(worst, F - F_prev)
    F_prev = F
    L = L - eta * G
Sig = L @ L.T
Fmin = np.trace(np.real(sqrtm(S))) ** 2 / T
print(f"[K1] largest increase of F along the path {worst:.1e} (non-positive up to step error)")
print(f"[K2] F at end / lower bound (tr S^1/2)^2/T = {F_prev / Fmin:.6f}")
print(f"[K3] ||Sigma/tr Sigma - S^1/2/tr S^1/2||_F / ||S^1/2/tr S^1/2||_F = "
      f"{np.linalg.norm(Sig / np.trace(Sig) - target) / np.linalg.norm(target):.1e}")
