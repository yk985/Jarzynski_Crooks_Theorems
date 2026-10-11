"""Energy and forces of the anchored chain in a harmonic trap. Works on torch tensors of shape (M, N, 2)."""
import torch

from .params import Chain

EPS = 1e-12


def bond_vectors(r):
    """u_i = r_i - r_{i-1} with r_0 = 0."""
    u = r.clone()
    u[:, 1:] -= r[:, :-1]
    return u


def bond_forces(r, P: Chain):
    """Forces on beads 1..N from the bonds only."""
    u = bond_vectors(r)
    if P.b == 0:
        fb = -P.k * u
    else:
        d = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        fb = -P.k * (1.0 - P.b / (d + EPS)) * u
    F = fb.clone()
    F[:, :-1] -= fb[:, 1:]
    return F


def trap_force(rN, lam_x, P: Chain):
    """Force on bead N from the trap centred at (lam_x, 0); rN has shape (M, 2)."""
    f = -P.kt * rN.clone()
    f[:, 0] += P.kt * lam_x
    return f


def forces(r, lam_x, P: Chain):
    F = bond_forces(r, P)
    F[:, -1] += trap_force(r[:, -1], lam_x, P)
    return F


def bond_energy(r, P: Chain):
    d = torch.linalg.vector_norm(bond_vectors(r), dim=-1)
    return 0.5 * P.k * ((d - P.b) ** 2).sum(-1)


def trap_energy(rN, lam_x, P: Chain):
    return 0.5 * P.kt * ((rN[:, 0] - lam_x) ** 2 + rN[:, 1] ** 2)


def potential(r, lam_x, P: Chain):
    return bond_energy(r, P) + trap_energy(r[:, -1], lam_x, P)


def hamiltonian(r, p, lam_x, P: Chain):
    return potential(r, lam_x, P) + 0.5 * (p**2).sum((-1, -2)) / P.m
