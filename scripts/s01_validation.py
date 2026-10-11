"""F1: validation on the Gaussian chain (b = 0), where Delta F and P(W) are known exactly."""
import json

from _common import CAT, DATA, expw_panel, np, plt, save, simulate
from jarz.estimators import block_curve, jarzynski
from jarz.linear_exact import work_moments
from jarz.params import Chain, tau1
from jarz.reference import dF_gaussian_chain

P = Chain(N=10, b=0.0)
dF = dF_gaussian_chain(P)
TAUS = [0.2, 2.0]
M = 100_000
print(f"b=0, N=10: Delta F = {dF:.4f}, tau1 = {tau1(P):.3f}")

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.6))
ax = axs[0]
res = {}
for i, tau in enumerate(TAUS):
    W = simulate(f"F1_b0_N10_tau{tau:g}", P, tau, M, seed=10 + i, dF=dF)
    mu, var = work_moments(P, tau)
    res[tau] = dict(mu_exact=mu, var_exact=var, mu_sim=W.mean(), var_sim=W.var(ddof=1),
                    dF_J=jarzynski(W, P.kT), J_exact_moments=mu - var / (2 * P.kT))
    lab = rf"$\tau={tau:g}$ ($\tau/\tau_1={tau / tau1(P):.1f}$)"
    ax[0].hist(W, bins=120, density=True, color=CAT[i], alpha=0.45, label=lab)
    w = np.linspace(W.min(), W.max(), 400)
    ax[0].plot(w, np.exp(-(w - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var), color=CAT[i], lw=1.5)
    n, m, s = block_curve(W, P.kT, np.unique(np.logspace(0, 5, 26).astype(int)))
    ax[1].fill_between(n, m - s, m + s, color=CAT[i], alpha=0.18, lw=0)
    ax[1].plot(n, m, color=CAT[i], label=lab)
    ax[1].axhline(mu, color=CAT[i], lw=0.8, ls=":")
    expw_panel(axs[1, i], W, P.kT, dF, CAT[i], gauss=(mu, var))
    axs[1, i].set(ylabel="fraction per bin", title=f"({'cd'[i]}) " + r"$e^{-\beta W}$, " + lab)
ax[0].axvline(dF, color="k", ls="--", lw=1.2, label=r"exact $\Delta F$")
ax[0].set(xlabel=r"work $W$ [$\varepsilon$]", ylabel=r"$P(W)$", title="(a) work distributions")
ax[0].legend(loc="upper left", fontsize=7)
ax[1].axhline(dF, color="k", ls="--", lw=1.2)
ax[1].set(xscale="log", xlabel="pulls used per estimate $n$", ylabel=r"$\widehat{\Delta F}_J$ [$\varepsilon$]",
          title="(b) convergence of the Jarzynski estimate")
ax[1].legend(loc="upper right", fontsize=7)
h, l = axs[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
save(fig, "F1_validation")
(DATA / "F1_summary.json").write_text(json.dumps({"dF": dF, **{str(k): v for k, v in res.items()}}, indent=1,
                                                 default=float))
print(json.dumps(res, indent=1, default=float))
