"""F4: number of beads at fixed relative extension and fixed tau/tau1 (stiff-bond chain, kT = 1)."""
import json

from _common import CAT, DATA, N_SWEEP, RAMP4, expw_panel, load_run, np, plt, save, simulate
from jarz.estimators import bootstrap, block_curve, ess, jarzynski
from jarz.params import Chain, tau1
from jarz.reference import delta_F, dF_entropic_spring

M = 100_000
C = 4
recs = []
fig, axd = plt.subplot_mosaic([["a"] * 4 + ["b"] * 4 + ["c"] * 4, ["d"] * 3 + ["e"] * 3 + ["f"] * 3 + ["g"] * 3],
                              figsize=(7.4, 5.4), gridspec_kw=dict(hspace=0.6, wspace=2.2))
ax = [axd["a"], axd["b"], axd["c"]]
for i, N in enumerate(N_SWEEP):
    P = Chain(N=N, b=1.0)
    dF, t1 = delta_F(P), tau1(P)
    W = simulate(f"F4_b1_N{N}_c{C}", P, C * t1, M, seed=400 + i, dF=dF)
    n, m, s = block_curve(W, P.kT, [100, 1000, 10_000])
    recs.append(dict(N=N, tau1=t1, tau=C * t1, dF=dF, dF_gauss=dF_entropic_spring(P), W_mean=W.mean(),
                     W_std=W.std(ddof=1), dF_J=jarzynski(W, P.kT), dF_J_err=bootstrap(W, P.kT, B=300),
                     ess=ess(W, P.kT), beta_Wdiss=(W.mean() - dF) / P.kT,
                     affine_estimate=np.pi**2 * N / (6 * C), bias_n100=m[0] - dF, bias_n1000=m[1] - dF,
                     bias_n10000=m[2] - dF))
    ax[0].hist(W, bins=150, density=True, color=RAMP4[i], alpha=0.8, label=rf"$N={N}$")
    ax[0].axvline(dF, color=RAMP4[i], ls="--", lw=1)
    a = axd["defg"[i]]
    expw_panel(a, W, P.kT, dF, RAMP4[i], WR=load_run(f"R_b1_N{N}_c{C}"), nb=50)
    a.set(title=f"({'defg'[i]}) " + r"$e^{-\beta W}$, " + f"$N={N}$", ylabel="fraction per bin" if i == 0 else None)
    a.set_xlabel(r"$\log_{10} e^{-\beta(W-\Delta F)}$", fontsize=7)
ax[0].set(xlabel=r"work $W$ [$\varepsilon$]", ylabel=r"$P(W)$", title=r"(a) $P(W)$, $\tau=4\tau_1(N)$")
ax[0].legend()

Ns = np.array(N_SWEEP)
ax[1].plot(Ns, [r["dF"] for r in recs], "k--", lw=1.2, label=r"exact $\Delta F$")
ax[1].plot(Ns, [r["dF_gauss"] for r in recs], ":", color="#6b6a64", lw=1.2, label="entropic spring")
ax[1].plot(Ns, [r["W_mean"] for r in recs], "s", ms=4, color=CAT[1], label=r"$\langle W\rangle$")
ax[1].errorbar(Ns, [r["dF_J"] for r in recs], yerr=[r["dF_J_err"] for r in recs], fmt="o", ms=4,
               color=CAT[0], capsize=2, lw=1, label=r"$\widehat{\Delta F}_J$ ($M=10^5$)")
ax[1].set(xscale="log", yscale="log", xlabel="beads $N$", ylabel=r"[$\varepsilon$]", title=r"(b) free energy and work")
ax[1].legend(fontsize=6.5)
ax[2].plot(Ns, [r["beta_Wdiss"] for r in recs], "o-", ms=4, color=CAT[1], label=r"$\beta W_{\rm diss}$")
for k, (key, lab) in enumerate([("bias_n100", "$n=10^2$"), ("bias_n10000", "$n=10^4$")]):
    ax[2].plot(Ns, [r[key] for r in recs], "o-", ms=4, color=[CAT[0], "#0d366b"][k],
               label=rf"bias of $\beta\widehat{{\Delta F}}_J$, {lab}")
ax[2].plot(Ns, np.pi**2 * Ns / (6 * C), "--", color=CAT[1], lw=1, label=r"affine estimate $\pi^2N/24$")
ax[2].axhline(np.log(M), color="#6b6a64", ls=":", lw=1.2)
ax[2].text(5.2, np.log(M) * 1.12, r"$\ln M$", color="#6b6a64", fontsize=8)
ax[2].set(xscale="log", yscale="log", xlabel="beads $N$", ylabel=r"[$k_BT$]", title="(c) dissipation and bias")
ax[2].legend(fontsize=6, loc="lower right")
for a in ax[1:]:
    a.set_xticks(N_SWEEP, [str(n) for n in N_SWEEP])
    a.xaxis.set_minor_formatter(plt.NullFormatter())
ax[1].set_yticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
ax[1].yaxis.set_minor_formatter(plt.NullFormatter())
h, l = axd["g"].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
save(fig, "F4_size")
(DATA / "F4_summary.json").write_text(json.dumps(recs, indent=1, default=float))
for r in recs:
    print({k: round(float(v), 4) for k, v in r.items()})
