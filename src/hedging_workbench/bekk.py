"""GARCH-BEKK(1,1) 2x2 fit and time-varying hedge ratio (ticket 10-10).

H_t = C C' + A eps_{t-1} eps_{t-1}' A' + B H_{t-1} B'

Estimation uses VARIANCE TARGETING (standard for BEKK): the intercept CC'
is pinned analytically to the sample covariance given (A, B),
    vec(CC') = (I - A(x)A - B(x)B)^-1 vec(S),
so only A and B (8 params) are optimised. Free-C fits were numerically
unstable here (likelihood rough, 11-dim numerical gradient) — targeting
removes the intercept misspecification and anchors the unconditional
covariance at the sample value. Documented choice.

The recursion/log-likelihood use explicit 2x2 SCALAR algebra: L-BFGS-B
needs ~9 likelihood evaluations per iteration for an 8-dim numerical
gradient, and numpy per-call overhead dominated runtime.

Optimal hedge ratio (literature standard, e.g. Kroner-Sultan):
    h_t = H12_t / H22_t   (units of futures per unit of the hedged item)

Choices documented: BEKK hand-rolled because `arch` has no BEKK; constant
means removed before fitting; stationarity CHECKED (spectral radius of
A(x)A + B(x)B < 1), enforced during search, reported on the fit. First
BURN_IN observations of h_t are H_0 burn-in artefacts and are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BURN_IN = 20
BIG = 1e10


@dataclass
class BekkFit:
    C: np.ndarray  # lower-triangular 2x2 intercept (from targeting)
    A: np.ndarray  # 2x2
    B: np.ndarray  # 2x2
    loglikelihood: float
    stationary: bool  # spectral radius of A(x)A + B(x)B < 1
    H: np.ndarray  # (T, 2, 2) conditional covariances

    def hedge_ratio(self) -> pd.Series:
        """h_t = H12 / H22 per period, burn-in observations removed."""
        h = pd.Series(self.H[:, 0, 1] / self.H[:, 1, 1])
        return h.iloc[BURN_IN:].reset_index(drop=True)


def _target_C(A: np.ndarray, B: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Variance-targeting intercept: C with CC' solving the unconditional
    covariance equation S = CC' + A S A' + B S B', i.e.
    vec(CC') = (I - A(x)A - B(x)B) vec(S). None if not positive definite."""
    k = np.eye(4) - np.kron(A, A) - np.kron(B, B)
    vec = k @ S.reshape(4, order="F")
    m = np.array([[vec[0], vec[2]], [vec[1], vec[3]]])
    m = 0.5 * (m + m.T)
    try:
        return np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        return None


def _recurse(r: np.ndarray, c00, c01, c11, a11, a12, a21, a22, b11, b12, b21, b22):
    """Scalar 2x2 BEKK recursion. Returns (T, 2, 2) array of H_t."""
    T = len(r)
    H = np.empty((T, 2, 2))
    h00, h01, h11 = c00 * c00, c00 * c01, c01 * c01 + c11 * c11
    for t in range(T):
        e0, e1 = r[t - 1] if t > 0 else (0.0, 0.0)
        u0 = a11 * e0 + a12 * e1
        u1 = a21 * e0 + a22 * e1
        m00 = b11 * h00 + b12 * h01
        m01 = b11 * h01 + b12 * h11
        m10 = b21 * h00 + b22 * h01
        m11 = b21 * h01 + b22 * h11
        h00 = c00 + u0 * u0 + m00 * b11 + m01 * b12
        h01 = c01 + u0 * u1 + m00 * b21 + m01 * b22
        h11 = c11 + u1 * u1 + m10 * b21 + m11 * b22
        H[t, 0, 0], H[t, 0, 1], H[t, 1, 1] = h00, h01, h11
        H[t, 1, 0] = h01
    return H


def _neg_loglik(theta8: np.ndarray, r: np.ndarray, S: np.ndarray) -> float:
    """Multivariate-normal BEKK loglik, constants dropped, scalar algebra."""
    A = theta8[:4].reshape(2, 2)
    B = theta8[4:].reshape(2, 2)
    spectral = np.max(np.abs(np.linalg.eigvals(np.kron(A, A) + np.kron(B, B))))
    if spectral >= 1.0:
        return BIG
    C = _target_C(A, B, S)
    if C is None:
        return BIG
    c00, c01, c11 = C[0, 0], C[1, 0], C[1, 1]
    a11, a12, a21, a22 = A.flat
    b11, b12, b21, b22 = B.flat
    h00, h01, h11 = c00 * c00, c00 * c01, c01 * c01 + c11 * c11
    total = 0.0
    for t in range(len(r)):
        e0, e1 = r[t - 1] if t > 0 else (0.0, 0.0)
        u0 = a11 * e0 + a12 * e1
        u1 = a21 * e0 + a22 * e1
        m00 = b11 * h00 + b12 * h01
        m01 = b11 * h01 + b12 * h11
        m10 = b21 * h00 + b22 * h01
        m11 = b21 * h01 + b22 * h11
        h00 = c00 + u0 * u0 + m00 * b11 + m01 * b12
        h01 = c01 + u0 * u1 + m00 * b21 + m01 * b22
        h11 = c11 + u1 * u1 + m10 * b21 + m11 * b22
        det = h00 * h11 - h01 * h01
        if det <= 0.0:
            return BIG
        i00, i01, i11 = h11 / det, -h01 / det, h00 / det
        quad = i00 * e0 * e0 + 2.0 * i01 * e0 * e1 + i11 * e1 * e1
        total += 0.5 * (np.log(det) + quad)
    return total


def simulate_bekk(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, T: int, seed: int = 0
) -> np.ndarray:
    """Simulate returns from a BEKK(1,1) with the given matrices."""
    rng = np.random.default_rng(seed)
    r = np.empty((T, 2))
    CCt = C @ C.T
    h = CCt.copy()
    for t in range(T):
        eps = rng.multivariate_normal(np.zeros(2), h + 1e-10 * np.eye(2))
        r[t] = eps
        h = CCt + A @ np.outer(eps, eps) @ A.T + B @ h @ B.T
    return r


def fit_bekk(returns: pd.DataFrame) -> BekkFit:
    """Variance-targeting MLE fit on (T, 2) returns. Column 0 = hedged
    item, column 1 = hedging futures (hedge-ratio convention)."""
    r = np.asarray(returns, dtype=float)
    r = r - r.mean(axis=0)
    S = np.cov(r.T)
    theta0 = np.array([0.08, 0.0, 0.0, 0.08, 0.80, 0.0, 0.0, 0.80])
    res = minimize(
        _neg_loglik,
        theta0,
        args=(r, S),
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-8},
    )
    A = res.x[:4].reshape(2, 2)
    B = res.x[4:].reshape(2, 2)
    C = _target_C(A, B, S)
    H = _recurse(r, C[0, 0], C[1, 0], C[1, 1], *A.flat, *B.flat)
    spec = np.max(np.abs(np.linalg.eigvals(np.kron(A, A) + np.kron(B, B))))
    return BekkFit(
        C=C, A=A, B=B, loglikelihood=-res.fun, stationary=bool(spec < 1), H=H
    )
