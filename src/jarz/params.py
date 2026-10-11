"""Parameters of the anchored bead-spring chain and its pulling protocol (reduced units)."""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Chain:
    N: int = 10          # mobile beads (= number of bonds)
    b: float = 1.0       # bond rest length (0 -> Gaussian chain)
    k: float = 100.0     # bond stiffness
    kt: float = 10.0     # trap stiffness
    kT: float = 1.0      # bath temperature
    m: float = 1.0       # bead mass
    gamma: float = 1.0   # friction rate, zeta = m * gamma
    frac0: float = 0.2   # initial trap position / L
    frac1: float = 0.7   # final trap position / L
    ell: float = 0.3     # nominal bond length used for L when b = 0

    @property
    def beta(self):
        return 1.0 / self.kT

    @property
    def zeta(self):
        return self.m * self.gamma

    @property
    def L(self):
        """Length scale of the protocol: contour length N*b (or N*ell for b = 0)."""
        return self.N * (self.b if self.b > 0 else self.ell)

    @property
    def x0(self):
        return self.frac0 * self.L

    @property
    def x1(self):
        return self.frac1 * self.L

    def with_(self, **kw):
        return replace(self, **kw)

    def key(self):
        return f"b{self.b:g}_N{self.N}_kT{self.kT:g}_k{self.k:g}_kt{self.kt:g}"


def bond_radial_grid(P: Chain, n=4001):
    """Grid r and unnormalised radial density r*exp(-beta U_b(r)) of one bond length."""
    s = np.sqrt(P.kT / P.k)
    r = np.linspace(max(0.0, P.b - 10 * s), P.b + 10 * s, n)
    dens = r * np.exp(-0.5 * P.k * (r - P.b) ** 2 / P.kT)
    return r, dens


def mean_sq_bond(P: Chain):
    """<u^2> of one bond in 2D (exact by quadrature)."""
    r, dens = bond_radial_grid(P)
    return np.trapezoid(dens * r**2, r) / np.trapezoid(dens, r)


def rouse_matrix(P: Chain):
    """Stiffness matrix of the linearised chain: springs kappa in series, bead 0 fixed, trap k_t on bead N.

    kappa = k for the Gaussian chain, and the 2D entropic spring 2 kT/<u^2> for the stiff-bond chain.
    """
    kappa = P.k if P.b == 0 else 2 * P.kT / mean_sq_bond(P)
    K = np.diag(np.full(P.N, 2 * kappa)) - kappa * (np.eye(P.N, k=1) + np.eye(P.N, k=-1))
    K[-1, -1] = kappa + P.kt
    return K


def tau1(P: Chain):
    """Slowest overdamped relaxation time zeta / lambda_min(K)."""
    return P.zeta / np.linalg.eigvalsh(rouse_matrix(P))[0]
