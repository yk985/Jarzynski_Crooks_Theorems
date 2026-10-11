import numpy as np

from jarz.linear_exact import hessian
from jarz.params import Chain, mean_sq_bond
from jarz.reference import free_energy_curve
from jarz.sampling import sample_positions


def test_gaussian_chain_moments():
    P = Chain(N=6, b=0.0)
    x = sample_positions(P, P.x0, 40_000, np.random.default_rng(1))
    K = hessian(P)
    eN = np.eye(P.N)[-1]
    mean = np.linalg.solve(K, P.kt * P.x0 * eN)
    cov = P.kT * np.linalg.inv(K)
    assert np.allclose(x[..., 0].mean(0), mean, atol=4 * np.sqrt(np.diag(cov) / 40_000).max())
    assert np.allclose(np.cov(x[..., 0].T), cov, atol=0.05 * cov.max())
    assert np.allclose(np.cov(x[..., 1].T), cov, atol=0.05 * cov.max())


def test_bond_length_and_mean_trap_force():
    """<d_lambda H> = -k_t <x_N - lambda> must equal dF/dlambda from the exact reference."""
    P = Chain(N=4, b=1.0)
    free = sample_positions(P.with_(kt=1e-9), 0.0, 40_000, np.random.default_rng(2))
    u = np.diff(np.concatenate([np.zeros((len(free), 1, 2)), free], 1), axis=1)
    assert abs((u**2).sum(-1).mean() - mean_sq_bond(P)) < 2e-3
    x = sample_positions(P, P.x1, 40_000, np.random.default_rng(2))
    R, F = free_energy_curve(P)
    dFdl = np.gradient(F, R)[np.searchsorted(R, P.x1)]
    f = -P.kt * (x[:, -1, 0] - P.x1)
    assert abs(f.mean() - dFdl) < 4 * f.std() / np.sqrt(len(f))


def test_tilted_and_plain_samplers_agree():
    """The tilt changes only the proposal; both samplers must give the same equilibrium ensemble."""
    P = Chain(N=6, b=1.0)
    a = sample_positions(P, 3.0, 30_000, np.random.default_rng(5), tilt=True)
    b = sample_positions(P, 3.0, 30_000, np.random.default_rng(6), tilt=False)
    for fa, fb in [(a[:, -1, 0], b[:, -1, 0]), (a[:, 2, 0], b[:, 2, 0]), (a[:, 2, 1] ** 2, b[:, 2, 1] ** 2)]:
        se = np.sqrt(fa.var() / len(fa) + fb.var() / len(fb))
        assert abs(fa.mean() - fb.mean()) < 4 * se


def test_tilted_sampler_gaussian_chain_stretched():
    P = Chain(N=6, b=0.0)
    x = sample_positions(P, P.x1, 40_000, np.random.default_rng(7))
    K = hessian(P)
    mean = np.linalg.solve(K, P.kt * P.x1 * np.eye(P.N)[-1])
    cov = P.kT * np.linalg.inv(K)
    assert np.allclose(x[..., 0].mean(0), mean, atol=4 * np.sqrt(cov.max() / 40_000))
    assert np.allclose(np.cov(x[..., 0].T), cov, atol=0.05 * cov.max())
