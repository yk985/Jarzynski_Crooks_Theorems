"""Diagnostics (not a report figure): time-step halving and seed-to-seed scatter of the Jarzynski estimate."""
import json

from _common import DATA, np, simulate
from jarz.estimators import bootstrap, jarzynski
from jarz.params import Chain, tau1
from jarz.reference import delta_F

M = 100_000
out = []
for kT, c in [(2.0, 4), (1.0, 4)]:
    P = Chain(N=10, b=1.0, kT=kT)
    dF, t1 = delta_F(P), tau1(P)
    for dt in (0.01, 0.005):
        for seed in (601, 602, 603):
            W = simulate(f"S6_kT{kT:g}_c{c}_dt{dt:g}_s{seed}", P, c * t1, M, seed=seed, dt=dt, dF=dF)
            out.append(dict(kT=kT, c=c, dt=dt, seed=seed, dF=dF, W_mean=W.mean(),
                            bias_kT=(jarzynski(W, kT) - dF) / kT, boot_kT=bootstrap(W, kT, B=300) / kT))
for r in out:
    print({k: round(float(v), 4) for k, v in r.items()})
(DATA / "S6_checks.json").write_text(json.dumps(out, indent=1, default=float))
