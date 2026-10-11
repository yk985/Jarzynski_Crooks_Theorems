import numpy as np
import pytest

from jarz.dynamics import run
from jarz.estimators import jarzynski
from jarz.linear_exact import hessian, work_moments
from jarz.params import Chain
from jarz.reference import dF_gaussian_chain


@pytest.mark.parametrize("b", [0.0, 1.0])
def test_first_law_per_trajectory(b):
    P = Chain(N=5, b=b)
    out = run(P, tau=2.0, M=50, dt=0.01, seed=3, track_heat=True)
    assert np.allclose(out["dH"], out["W"] + out["Q"], atol=1e-9)


def test_equilibrium_preserved_at_fixed_trap():
    P = Chain(N=6, b=0.0)
    out = run(P, tau=None, M=20_000, dt=0.01, seed=4, lam=np.full(2001, P.x0), n_record=20_000,
              record_every=2000)
    x = out["rec_r"][-1][..., 0]
    K = hessian(P)
    cov = P.kT * np.linalg.inv(K)
    mean = np.linalg.solve(K, P.kt * P.x0 * np.eye(P.N)[-1])
    assert np.allclose(x.mean(0), mean, atol=5 * np.sqrt(cov.max() / 20_000))
    assert np.allclose(np.cov(x.T), cov, atol=0.05 * cov.max())
    assert np.all(out["W"] == 0)


def test_simulated_work_matches_exact_linear_theory():
    P = Chain(N=6, b=0.0)
    tau = 1.0
    W = run(P, tau=tau, M=20_000, dt=0.01, seed=5)["W"]
    mu, var = work_moments(P, tau, dt=0.01)
    assert abs(W.mean() - mu) < 5 * np.sqrt(var / len(W))
    assert abs(W.var() / var - 1) < 0.05
    assert abs(jarzynski(W, P.kT) - dF_gaussian_chain(P)) < 0.1


@pytest.mark.parametrize("tau", [0.05, 0.5, 5.0])
def test_jarzynski_exact_for_gaussian_chain(tau):
    """Gaussian W: Jarzynski <=> <W> - Var W / 2kT = Delta F, checked with exact moments of the algorithm."""
    P = Chain(N=10, b=0.0)
    mu, var = work_moments(P, tau, dt=0.01)
    assert mu - dF_gaussian_chain(P) > 0
    assert abs(mu - var / (2 * P.kT) - dF_gaussian_chain(P)) < 1e-3 * dF_gaussian_chain(P)


def test_crooks_exact_for_gaussian_chain():
    """Reverse protocol: exact moments satisfy mu_F - dF = mu_R + dF = var / 2kT (Crooks for Gaussian work)."""
    from jarz.dynamics import protocol
    P = Chain(N=10, b=0.0)
    lam = protocol(P, 50)
    muF, vF = work_moments(P, None, lam=lam)
    muR, vR = work_moments(P, None, lam=lam[::-1])
    dF = dF_gaussian_chain(P)
    assert abs(vF - vR) < 1e-3 * vF
    assert abs((muF - dF) - vF / (2 * P.kT)) < 1e-3 * vF
    assert abs((muR + dF) - vR / (2 * P.kT)) < 1e-3 * vR
