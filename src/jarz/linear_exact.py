"""Exact work statistics for the Gaussian chain (b = 0) under the *discrete* BAOAB pulling scheme.

For b = 0 all forces are linear, so (q, p, W) along x is a linear-Gaussian Markov chain: every sub-step
is an affine map plus Gaussian noise. Propagating mean and covariance gives the exact mean and variance
of W for the simulated algorithm (W is Gaussian). The y-direction does not contribute to W.
"""
import math

import numpy as np

from .dynamics import protocol
from .params import Chain


def hessian(P: Chain):
    K = np.diag(np.full(P.N, 2 * P.k)) - P.k * (np.eye(P.N, k=1) + np.eye(P.N, k=-1))
    K[-1, -1] = P.k + P.kt
    return K


def work_moments(P: Chain, tau, dt=0.01, lam=None, return_path=False):
    """Exact <W> and Var W of the BAOAB pulling algorithm for the b = 0 chain."""
    assert P.b == 0
    if lam is None:
        lam = protocol(P, max(1, int(round(tau / dt))))
    N, h, n = P.N, dt, 2 * P.N + 1
    K0 = hessian(P)
    eN = np.zeros(N)
    eN[-1] = 1.0
    iq, ip, iW = slice(0, N), slice(N, 2 * N), 2 * N

    mu = np.zeros(n)
    S = np.zeros((n, n))
    Kinv = np.linalg.inv(K0)
    mu[iq] = Kinv @ (P.kt * lam[0] * eN)
    S[iq, iq] = P.kT * Kinv
    S[ip, ip] = P.m * P.kT * np.eye(N)

    def lin(A, a=None):
        nonlocal mu, S
        mu = A @ mu + (0 if a is None else a)
        S = A @ S @ A.T

    I = np.eye(n)
    B = I.copy()
    B[ip, iq] = -0.5 * h * K0
    A = I.copy()
    A[iq, ip] = 0.5 * h / P.m * np.eye(N)
    c = math.exp(-P.gamma * h)
    O = I.copy()
    O[ip, ip] = c * np.eye(N)
    noise = np.zeros((n, n))
    noise[ip, ip] = (1 - c * c) * P.m * P.kT * np.eye(N)

    path = []
    for k in range(len(lam) - 1):
        lo, ln = lam[k], lam[k + 1]
        Sw = I.copy()
        Sw[iW, N - 1] = P.kt * (lo - ln)
        aw = np.zeros(n)
        aw[iW] = 0.5 * P.kt * (ln**2 - lo**2)
        lin(Sw, aw)
        ab = np.zeros(n)
        ab[ip] = 0.5 * h * P.kt * ln * eN
        lin(B, ab)
        lin(A)
        lin(O)
        S = S + noise
        lin(A)
        lin(B, ab)
        if return_path:
            path.append((mu[iW], S[iW, iW]))
    if return_path:
        return mu[iW], S[iW, iW], np.array(path)
    return mu[iW], S[iW, iW]
