"""Exact Delta F references (stiff-bond chain) for every parameter set used in the sweeps."""
from multiprocessing import Pool

from _common import N_SWEEP, T_SWEEP
from jarz.params import Chain
from jarz.reference import delta_F, dF_entropic_spring

CASES = [Chain(N=N, b=1.0) for N in N_SWEEP] + [Chain(N=10, b=1.0, kT=t) for t in T_SWEEP if t != 1.0]


def work(P):
    return P.key(), delta_F(P), dF_entropic_spring(P)


if __name__ == "__main__":
    with Pool(len(CASES)) as pool:
        for key, dF, gauss in pool.imap(work, sorted(CASES, key=lambda P: -P.N)):
            print(f"{key:35s} dF={dF:9.4f}  entropic-spring approx={gauss:9.4f}", flush=True)
