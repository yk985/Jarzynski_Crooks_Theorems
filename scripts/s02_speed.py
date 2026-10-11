"""F2: effect of the pulling time on P(W) for the stiff-bond chain (b = 1, N = 10, kT = 1)."""
import json

import matplotlib.ticker
from scipy.stats import skew

from _common import DATA, RAMP3, expw_panel, load_run, np, plt, save, simulate
from jarz.estimators import block_curve, cumulant2, ess, jarzynski
from jarz.params import Chain, tau1
from jarz.reference import delta_F

P = Chain(N=10, b=1.0)
dF = delta_F(P)
t1 = tau1(P)
CS = [1, 3, 30]
M = 100_000
print(f"b=1 N=10: dF={dF:.4f} tau1={t1:.3f}")

fig, axd = plt.subplot_mosaic([["a"] * 3 + ["b"] * 3, ["c"] * 2 + ["d"] * 2 + ["e"] * 2], figsize=(7.4, 5.6),
                              gridspec_kw=dict(hspace=0.6, wspace=1.2))
ax = [axd["a"], axd["b"]]
res = {}
for i, c in enumerate(CS):
    W = simulate(f"F2_b1_N10_c{c}", P, c * t1, M, seed=20 + i, dF=dF)
    res[c] = dict(tau=c * t1, W_mean=W.mean(), W_std=W.std(ddof=1), skew=skew(W), dF_J=jarzynski(W, P.kT),
                  dF_c2=cumulant2(W, P.kT), ess=ess(W, P.kT), beta_Wdiss=(W.mean() - dF) / P.kT,
                  affine_estimate=np.pi**2 * P.N / (6 * c), frac_below=np.mean(W < dF))
    lab = rf"$\tau/\tau_1={c}$"
    ax[0].hist(W, bins=np.linspace(-2, 70, 181), density=True, color=RAMP3[i], alpha=0.75, label=lab)
    n, m, s = block_curve(W, P.kT, np.unique(np.logspace(0, 5, 26).astype(int)))
    ax[1].fill_between(n, m - s, m + s, color=RAMP3[i], alpha=0.2, lw=0)
    ax[1].plot(n, m, color=RAMP3[i], label=lab)
    a = axd["cde"[i]]
    expw_panel(a, W, P.kT, dF, RAMP3[i], WR=load_run(f"R_b1_N10_c{c}"))
    a.set(title=f"({'cde'[i]}) " + r"$e^{-\beta W}$, " + lab, ylabel="fraction per bin" if i == 0 else None)
ax[0].axvline(dF, color="k", ls="--", lw=1.2, label=r"exact $\Delta F$")
ax[0].set(xlabel=r"work $W$ [$\varepsilon$]", ylabel=r"$P(W)$", title="(a) work distributions", xlim=(-2, 60))
ax[0].legend(loc="upper right")
ax[1].axhline(dF, color="k", ls="--", lw=1.2, label=r"exact $\Delta F$")
ax[1].set(xscale="log", yscale="log", xlabel="pulls used per estimate $n$",
          ylabel=r"$\widehat{\Delta F}_J$ [$\varepsilon$]", title="(b) convergence of the Jarzynski estimate")
ax[1].legend(loc="upper right")
ax[1].set_yticks([4, 5, 7, 10, 20, 30])
ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
h, l = axd["e"].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
save(fig, "F2_speed")
(DATA / "F2_summary.json").write_text(json.dumps({"dF": dF, "tau1": t1, **{str(k): v for k, v in res.items()}},
                                                 indent=1, default=float))
for c, r in res.items():
    print(c, {k: round(float(v), 3) for k, v in r.items()})
