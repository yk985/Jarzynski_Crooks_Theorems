import numpy as np
import pytest
from scipy.special import j0

from jarz.params import Chain
from jarz.reference import dF_entropic_spring, logZ_curve


def hankel_logZ(P, lams):
    """Fourier route (fine at small N / small Delta F where there is no cancellation)."""
    r = np.linspace(max(0, P.b - 1), P.b + 1, 2001)
    g = r * np.exp(-0.5 * P.k * (r - P.b) ** 2 / P.kT)
    q = np.linspace(0, 40, 8001)
    gh = np.trapezoid(g[None] * j0(np.outer(q, r)), r, axis=1) / np.trapezoid(g, r)
    base = gh**P.N * np.exp(-q**2 * P.kT / (2 * P.kt)) * q
    return np.log([np.trapezoid(base * j0(q * l), q) for l in lams])


@pytest.mark.parametrize("N", [3, 12])
def test_logspace_convolution_matches_hankel(N):
    P = Chain(N=N, b=1.0)
    R, lz = logZ_curve(P)
    lams = [0.0, 0.45, 0.9, 1.8]
    a = np.interp(lams, R, lz)
    b = hankel_logZ(P, lams)
    assert np.allclose(a - a[0], b - b[0], atol=2e-3)


def test_entropic_spring_error_shrinks_like_one_over_N():
    """The Gaussian approximation misses O(1/N) non-Gaussian corrections, even at small extension."""
    err = []
    for N in (12, 24):
        P = Chain(N=N, b=1.0, frac0=0.0, frac1=0.15)
        R, lz = logZ_curve(P)
        exact = -P.kT * (np.interp(P.x1, R, lz) - np.interp(P.x0, R, lz))
        err.append(abs(exact / dF_entropic_spring(P) - 1))
    assert err[1] < 0.6 * err[0]
