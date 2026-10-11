"""Underdamped Langevin dynamics (BAOAB) with a moving trap and work bookkeeping.

Step k (k = 0..K-1):
  1. switch lambda_k -> lambda_{k+1} at frozen x, W += H(x; lambda_{k+1}) - H(x; lambda_k)
  2. one BAOAB step at lambda_{k+1}
This is the switch/relax scheme of the discrete-time Jarzynski proof.
"""
import math

import numpy as np
import torch

from .model import bond_forces, hamiltonian, trap_force
from .params import Chain
from .sampling import sample_momenta, sample_positions


def protocol(P: Chain, K):
    """Trap x-positions lambda_0..lambda_K (linear ramp)."""
    return P.x0 + (P.x1 - P.x0) * np.arange(K + 1) / K


def initial_state(P: Chain, M, seed, lam_x=None):
    rng = np.random.default_rng(seed)
    lam_x = P.x0 if lam_x is None else lam_x
    return sample_positions(P, lam_x, M, rng), sample_momenta(P, M, rng)


@torch.no_grad()
def run(P: Chain, tau, M, dt=0.01, seed=0, device="cpu", lam=None, x_init=None,
        n_record=0, record_every=1, track_heat=False):
    """Pull M independent replicas from equilibrium at lambda_0 over a time tau.

    lam: optional explicit array lambda_0..lambda_K (default: linear ramp x0 -> x1 in tau/dt steps).
    Returns dict with W (M,), and optionally recorded positions / work of the first n_record replicas,
    and per-replica heat Q and energy change dH when track_heat is True.
    """
    if lam is None:
        K = max(1, int(round(tau / dt)))
        lam = protocol(P, K)
    K = len(lam) - 1
    dev = torch.device(device)
    dtype = torch.float64
    if x_init is None:
        x_init = initial_state(P, M, seed, lam_x=lam[0])
    r = torch.as_tensor(x_init[0], dtype=dtype, device=dev).clone()
    p = torch.as_tensor(x_init[1], dtype=dtype, device=dev).clone()
    gen = torch.Generator(device=dev).manual_seed(int(seed) + 12345)

    h = dt
    c = math.exp(-P.gamma * h)
    sig = math.sqrt((1 - c * c) * P.m * P.kT)
    W = torch.zeros(M, dtype=dtype, device=dev)
    Q = torch.zeros(M, dtype=dtype, device=dev) if track_heat else None
    H0 = hamiltonian(r, p, lam[0], P) if track_heat else None
    Fb = bond_forces(r, P)

    rec_r, rec_W, rec_t = [], [], []

    def record(k):
        if n_record and k % record_every == 0:
            rec_r.append(r[:n_record].cpu().numpy().copy())
            rec_W.append(W[:n_record].cpu().numpy().copy())
            rec_t.append(k * h)

    record(0)
    for k in range(K):
        lo, ln = float(lam[k]), float(lam[k + 1])
        xN = r[:, -1, 0]
        W += 0.5 * P.kt * ((xN - ln) ** 2 - (xN - lo) ** 2)
        if track_heat:
            Hs = hamiltonian(r, p, ln, P)
        # B
        F = Fb  # Fb is recomputed below, so it can be modified in place here
        F[:, -1] += trap_force(r[:, -1], ln, P)
        p.add_(F, alpha=0.5 * h)
        # A
        r.add_(p, alpha=0.5 * h / P.m)
        # O
        p.mul_(c).add_(torch.randn(p.shape, generator=gen, dtype=dtype, device=dev), alpha=sig)
        # A
        r.add_(p, alpha=0.5 * h / P.m)
        # B
        Fb = bond_forces(r, P)
        F = Fb.clone()
        F[:, -1] += trap_force(r[:, -1], ln, P)
        p.add_(F, alpha=0.5 * h)
        if track_heat:
            Q += hamiltonian(r, p, ln, P) - Hs
        record(k + 1)

    out = {"W": W.cpu().numpy(), "lam": np.asarray(lam), "dt": dt}
    if n_record:
        out.update(rec_r=np.array(rec_r), rec_W=np.array(rec_W), rec_t=np.array(rec_t))
    if track_heat:
        out.update(Q=Q.cpu().numpy(), dH=(hamiltonian(r, p, lam[-1], P) - H0).cpu().numpy())
    return out


def best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
