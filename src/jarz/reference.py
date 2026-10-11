"""Exact free energies F(lambda) = -kT ln Z(lambda) of the phantom chain in a trap.

Z(lambda) ~ (g * g * ... * g * w)(lambda) (N bond factors, one trap factor), all radial in 2D.
- b = 0: closed form (springs in series).
- b > 0: iterated radial convolution evaluated in log space (no cancellation, unlike Fourier/Hankel).
"""
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .params import Chain, mean_sq_bond


def dF_gaussian_chain(P: Chain, xa=None, xb=None):
    """Exact Delta F for b = 0: k_eff/2 (xb^2 - xa^2), 1/k_eff = N/k + 1/k_t."""
    xa = P.x0 if xa is None else xa
    xb = P.x1 if xb is None else xb
    keff = 1.0 / (P.N / P.k + 1.0 / P.kt)
    return 0.5 * keff * (xb**2 - xa**2)


def dF_entropic_spring(P: Chain, xa=None, xb=None):
    """Gaussian (entropic-spring) approximation: kT (xb^2 - xa^2) / (N <u^2> + 2 kT / k_t)."""
    xa = P.x0 if xa is None else xa
    xb = P.x1 if xb is None else xb
    return P.kT * (xb**2 - xa**2) / (P.N * mean_sq_bond(P) + 2 * P.kT / P.kt)


def _radial_conv(logh, Rg, logg, rg, nth=256, chunk=64):
    """log of the 2D convolution (h * g)(R) of radial functions; h tabulated on Rg, g on rg."""
    th = (np.arange(nth) + 0.5) * np.pi / nth
    lw = logg + np.log(rg * np.gradient(rg) * 2 * np.pi / nth)    # theta in [0, pi], doubled
    cos = np.cos(th)
    out = np.empty_like(Rg)
    for s in range(0, len(Rg), chunk):
        R = Rg[s:s + chunk, None, None]
        d = np.sqrt(np.maximum(R**2 + rg[None, :, None] ** 2 - 2 * R * rg[None, :, None] * cos, 0.0))
        vals = np.interp(d, Rg, logh, right=-np.inf)
        out[s:s + chunk] = logsumexp(lw[None, :, None] + vals, axis=(1, 2))
    return out


def logZ_curve(P: Chain, dR=0.01, nr=161):
    """Return grid R and ln Z(|lambda| = R) up to an additive constant."""
    if P.b == 0:
        raise ValueError("use the closed form for b = 0")
    s = np.sqrt(P.kT / P.k)
    Rg = np.arange(0, P.N * P.b + 10 * s * np.sqrt(P.N) + 8 * np.sqrt(P.kT / P.kt) + 1.0, dR)
    rg = np.linspace(max(1e-6, P.b - 8 * s), P.b + 8 * s, nr)
    logg_fn = lambda r: -0.5 * P.k * (r - P.b) ** 2 / P.kT
    logh = logg_fn(Rg)
    for _ in range(P.N - 1):
        logh = _radial_conv(logh, Rg, logg_fn(rg), rg)
    st = np.sqrt(P.kT / P.kt)
    rt = np.linspace(1e-6, 8 * st, nr)
    return Rg, _radial_conv(logh, Rg, -0.5 * P.kt * rt**2 / P.kT, rt)


DATA = Path(__file__).resolve().parents[2] / "data" / "reference"


def free_energy_curve(P: Chain):
    """Cached (x, F(x) - F(x0)) along the pulling axis for the stiff-bond chain."""
    DATA.mkdir(parents=True, exist_ok=True)
    f = DATA / f"{P.key()}.npz"
    if f.exists():
        z = np.load(f)
        return z["R"], z["F"]
    R, lz = logZ_curve(P)
    F = -P.kT * (lz - np.interp(P.x0, R, lz))
    np.savez(f, R=R, F=F)
    return R, F


def delta_F(P: Chain):
    """Exact Delta F for the protocol x0 -> x1."""
    if P.b == 0:
        return dF_gaussian_chain(P)
    R, F = free_energy_curve(P)
    return float(np.interp(P.x1, R, F))
