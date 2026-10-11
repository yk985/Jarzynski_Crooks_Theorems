"""Exact sampling of the equilibrium ensemble rho_eq(x; lambda) of the phantom chain.

Positions: rho(u_1..u_N) ~ prod_i g(u_i) * w(lambda - R), R = sum_i u_i, with g the bond Boltzmann factor and
w(s) = exp(-beta k_t |s|^2 / 2).

Tilted rejection sampling. Propose bonds independently from g(u) exp(beta f u_x) (a constant pulling force f
along x). The target/proposal ratio is w(lambda - R) exp(-beta f R_x) ~ exp(-beta k_t |R - lambda'|^2 / 2) with
lambda' = lambda - (f / k_t) x_hat, which is <= 1, so accepting with that probability is exact for ANY f.
f = 0 is plain rejection sampling; f chosen so that the proposal's mean end position equals lambda' keeps the
acceptance rate high even for strongly stretched chains (needed for reverse pulls starting at 0.7 L).
Momenta are Maxwellian.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e

from .params import Chain


def _tilted_radial_grid(P: Chain, f, n=4001):
    """Radial grid and log of the tilted marginal density r g(r) I0(beta f r)."""
    s = np.sqrt(P.kT / P.k)
    r = np.linspace(max(0.0, P.b - 10 * s), P.b + 10 * s + 3 * abs(f) / P.k, n)
    a = abs(f) / P.kT
    logd = np.log(np.maximum(r, 1e-300)) - 0.5 * P.k * (r - P.b) ** 2 / P.kT + np.log(i0e(a * r)) + a * r
    return r, logd


def mean_tilted_bond_x(P: Chain, f):
    """<u_x> of one bond under the tilt exp(beta f u_x)."""
    r, logd = _tilted_radial_grid(P, f)
    d = np.exp(logd - logd.max())
    a = f / P.kT
    ratio = np.sign(a) * i1e(abs(a) * r) / i0e(abs(a) * r)
    return np.trapezoid(d * r * ratio, r) / np.trapezoid(d, r)


def optimal_tilt(P: Chain, lam_x):
    """Force f solving N <u_x>_f = lam_x - f / k_t (proposal centred on the acceptance peak)."""
    h = lambda f: P.N * mean_tilted_bond_x(P, f) + f / P.kt - lam_x
    if abs(lam_x) < 1e-12:
        return 0.0
    hi = 1.0
    while h(hi * np.sign(lam_x)) * np.sign(lam_x) < 0:
        hi *= 2
    lo, hi = sorted([0.0, hi * np.sign(lam_x)])
    return brentq(h, lo, hi, xtol=1e-10)


def sample_bonds(P: Chain, f, size, rng):
    """(size..., 2) bond vectors from g(u) exp(beta f u_x)."""
    r, logd = _tilted_radial_grid(P, f)
    d = np.exp(logd - logd.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(r))])
    length = np.interp(rng.random(size), cdf / cdf[-1], r)
    kappa = abs(f) * length / P.kT
    th = rng.vonmises(0.0 if f >= 0 else np.pi, kappa) if f != 0 else rng.uniform(-np.pi, np.pi, size)
    return np.stack([length * np.cos(th), length * np.sin(th)], -1)


def sample_positions(P: Chain, lam_x, M, rng, tilt=True, batch=200_000, max_batches=100_000, info=None):
    """Return (M, N, 2) positions drawn exactly from the equilibrium ensemble at trap position (lam_x, 0)."""
    f = optimal_tilt(P, lam_x) if tilt else 0.0
    lam_p = lam_x - f / P.kt
    out, n, tried = [], 0, 0
    for _ in range(max_batches):
        B = batch if not n else min(batch, max(4 * (M - n), 10_000))
        u = sample_bonds(P, f, (B, P.N), rng)
        R = u.sum(1)
        logacc = -0.5 * P.kt * ((R[:, 0] - lam_p) ** 2 + R[:, 1] ** 2) / P.kT
        keep = np.log(rng.random(B)) < logacc
        out.append(np.cumsum(u[keep], axis=1))
        n += keep.sum()
        tried += B
        if n >= M:
            if info is not None:
                info.update(f=f, acceptance=n / tried)
            return np.concatenate(out)[:M]
    raise RuntimeError("rejection sampler did not collect enough samples")


def sample_momenta(P: Chain, M, rng):
    return rng.normal(0.0, np.sqrt(P.m * P.kT), (M, P.N, 2))
