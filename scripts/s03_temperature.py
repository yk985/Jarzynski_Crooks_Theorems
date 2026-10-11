"""F3: temperature sweep. Stiff-bond chain (entropic) at fixed tau/tau1 and at fixed tau; Gaussian chain for contrast."""
import json

from _common import CAT, DATA, RAMP4, T_SWEEP, np, plt, save, simulate
from jarz.estimators import bootstrap, ess, jarzynski
from jarz.linear_exact import work_moments
from jarz.params import Chain, tau1
from jarz.reference import delta_F, dF_gaussian_chain

M = 100_000
C = 4
TAU_FIXED = C * tau1(Chain(N=10, b=1.0, kT=1.0))
TAU_B0 = 2.0
series = {"b1_c4": [], "b1_fixtau": [], "b0": []}

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.42))
for i, kT in enumerate(T_SWEEP):
    P1 = Chain(N=10, b=1.0, kT=kT)
    P0 = Chain(N=10, b=0.0, kT=kT)
    dF1, dF0 = delta_F(P1), dF_gaussian_chain(P0)
    runs = {"b1_c4": (P1, C * tau1(P1), dF1), "b1_fixtau": (P1, TAU_FIXED, dF1), "b0": (P0, TAU_B0, dF0)}
    for j, (name, (P, tau, dF)) in enumerate(runs.items()):
        W = simulate(f"F3_{name}_kT{kT:g}", P, tau, M, seed=300 + 10 * i + j, dF=dF)
        rec = dict(kT=kT, tau=tau, dF=dF, W_mean=W.mean(), W_std=W.std(ddof=1), dF_J=jarzynski(W, kT),
                   dF_J_err=bootstrap(W, kT, B=300), ess=ess(W, kT), beta_Wdiss=(W.mean() - dF) / kT)
        if name == "b0":
            mu, var = work_moments(P, tau)
            rec.update(beta_Wdiss_exact=(mu - dF) / kT, W_std_exact=np.sqrt(var))
        series[name].append(rec)
        if name == "b1_c4":
            ax[0].hist(W / kT, bins=np.linspace(-3, 20, 116), density=True, color=RAMP4[i], histtype="step",
                       lw=1.5, label=rf"$k_BT={kT:g}$")
            ax[0].axvline(dF / kT, color=RAMP4[i], ls="--", lw=1)

ax[0].set(xlabel=r"$\beta W$", ylabel=r"$P(\beta W)$", title=r"(a) $P(\beta W)$, $b=1$, $\tau=4\tau_1(T)$", xlim=(-3, 20))
ax[0].legend(loc="upper right", fontsize=6.5, handlelength=1.0, borderaxespad=0.1)
ax[0].set_ylim(0, 0.19)

T = np.array(T_SWEEP)
labels = {"b1_c4": r"$b=1$, $\tau=4\tau_1(T)$", "b1_fixtau": r"$b=1$, $\tau$ fixed", "b0": r"$b=0$, $\tau$ fixed"}
for j, (name, recs) in enumerate(series.items()):
    dFJ = np.array([r["dF_J"] for r in recs])
    err = np.array([r["dF_J_err"] for r in recs])
    off = (j - 1) * 0.035
    ax[1].errorbar(T * (1 + off), dFJ, yerr=err, fmt="o", ms=4, color=CAT[j], label=labels[name], capsize=2, lw=1)
    ax[2].plot(T, [r["beta_Wdiss"] for r in recs], "o-", ms=4, color=CAT[j], label=labels[name])
ax[1].plot(T, [r["dF"] for r in series["b1_c4"]], "k--", lw=1.2, label=r"exact $\Delta F$")
ax[1].plot(T, [r["dF"] for r in series["b0"]], "k--", lw=1.2)
ax[1].set(xscale="log", yscale="log", xlabel=r"$k_BT$ [$\varepsilon$]", ylabel=r"$\Delta F$ [$\varepsilon$]",
          title=r"(b) $\widehat{\Delta F}_J$ vs exact")
ax[1].legend(fontsize=6.5, loc="lower right")
ax[1].set_ylim(0.8, 25)
ax[1].set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
ax[1].yaxis.set_minor_formatter(plt.NullFormatter())

ax[2].axhline(np.log(M), color="#6b6a64", ls=":", lw=1.2)
ax[2].text(0.26, np.log(M) * 1.08, r"$\ln M$", color="#6b6a64", fontsize=8)
ax[2].set(xscale="log", yscale="log", xlabel=r"$k_BT$ [$\varepsilon$]", ylabel=r"$\beta W_{\rm diss}$",
          title="(c) dissipated work")
for a in ax[1:]:
    a.set_xticks(T_SWEEP, [f"{t:g}" for t in T_SWEEP])
    a.xaxis.set_minor_formatter(plt.NullFormatter())
ax[2].set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
ax[2].yaxis.set_minor_formatter(plt.NullFormatter())
ax[2].legend(fontsize=6.5, loc="lower left")
save(fig, "F3_temperature")
(DATA / "F3_summary.json").write_text(json.dumps(dict(tau_fixed=TAU_FIXED, tau_b0=TAU_B0, **series), indent=1,
                                                 default=float))
for name, recs in series.items():
    for r in recs:
        print(name, {k: round(float(v), 4) for k, v in r.items()})
