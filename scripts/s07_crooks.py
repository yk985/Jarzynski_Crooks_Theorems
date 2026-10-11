"""F5: Crooks fluctuation theorem with reverse pulls; two-sided (BAR) vs one-sided (Jarzynski) estimates."""
import json
import sys

from scipy.stats import kurtosis, skew

from _common import CAT, DATA, INK, N_SWEEP, RAMP3, np, plt, save, simulate
from jarz.estimators import bar, crooks_log_ratio, jarzynski, jarzynski_reverse
from jarz.params import Chain, tau1
from jarz.reference import delta_F

M = 100_000
SPEEDS = [1, 3, 30]


def runs():
    """Forward runs are the cached F2/F4 runs; reverse runs use new seeds."""
    out = {}
    P = Chain(N=10, b=1.0)
    dF, t1 = delta_F(P), tau1(P)
    for i, c in enumerate(SPEEDS):
        WF = simulate(f"F2_b1_N10_c{c}", P, c * t1, M, seed=20 + i, dF=dF)
        WR = simulate(f"R_b1_N10_c{c}", P, c * t1, M, seed=700 + i, dF=-dF, reverse=True)
        out[("speed", c)] = (P, dF, WF, WR)
    for i, N in enumerate(N_SWEEP):
        P = Chain(N=N, b=1.0)
        dF, t1 = delta_F(P), tau1(P)
        WF = simulate(f"F4_b1_N{N}_c4", P, 4 * t1, M, seed=400 + i, dF=dF)
        WR = simulate(f"R_b1_N{N}_c4", P, 4 * t1, M, seed=710 + i, dF=-dF, reverse=True)
        out[("size", N)] = (P, dF, WF, WR)
    return out


def repeat_spread(est, WF, WR, kT, n_rep=10):
    """Spread of an estimator over n_rep disjoint sub-samples (honest error bar, scaled to the full sample)."""
    vals = [est(a, b, kT) for a, b in zip(np.array_split(WF, n_rep), np.array_split(WR, n_rep))]
    return np.std(vals) / np.sqrt(n_rep)


def gauss(x, w):
    return np.exp(-(x - w.mean()) ** 2 / (2 * w.var())) / np.sqrt(2 * np.pi * w.var())


if __name__ == "__main__":
    data = runs()
    if "--runs-only" in sys.argv:
        sys.exit()
    fig, axd = plt.subplot_mosaic([["a", "b", "c"], ["d", "d", "e"]], figsize=(7.4, 5.4),
                                  gridspec_kw=dict(hspace=0.55, wspace=0.35))
    res = {}
    for i, c in enumerate(SPEEDS):
        P, dF, WF, WR = data[("speed", c)]
        a = axd["abc"[i]]
        lo = np.percentile(-WR, 0.02)
        hi = np.percentile(WF, 99.98)
        bins = np.linspace(lo, hi, 70)
        x = np.linspace(lo, hi, 400)
        a.hist(WF, bins, density=True, histtype="stepfilled", color=CAT[0], alpha=0.35, lw=0,
               label=r"forward $P_F(W)$")
        a.hist(-WR, bins, density=True, histtype="stepfilled", color=CAT[1], alpha=0.35, lw=0,
               label=r"reverse, mirrored $P_R(-W)$")
        a.plot(x, gauss(x, WF), color=CAT[0], lw=1)
        a.plot(x, gauss(x, -WR), color=CAT[1], lw=1)
        a.plot([], [], color=INK, lw=1, label="Gaussian, same mean and variance")
        a.axvline(dF, color="k", ls="--", lw=1.2, label=r"exact $\Delta F$")
        a.set(xlabel=r"$W$ [$\varepsilon$]", ylabel="density" if i == 0 else None,
              title=f"({'abc'[i]}) " + rf"$\tau={c}\,\tau_1$")
        cb, lr, err = crooks_log_ratio(WF, WR, P.kT, np.linspace(dF - 8, dF + 8, 33))
        ok = np.isfinite(lr) & (err < 0.5)
        axd["d"].errorbar((cb[ok] - dF) / P.kT, lr[ok], yerr=err[ok], fmt="o", ms=3.5, color=RAMP3[i], lw=0.8,
                          capsize=0, label=rf"$\tau={c}\,\tau_1$")
        sl, ic = np.polyfit(cb[ok] / P.kT, lr[ok], 1, w=1 / err[ok]) if ok.sum() > 2 else (np.nan, np.nan)
        res[f"speed_c{c}"] = dict(dF=dF, J_F=jarzynski(WF, P.kT), J_R=jarzynski_reverse(WR, P.kT),
                                  BAR=bar(WF, WR, P.kT), BAR_err=repeat_spread(bar, WF, WR, P.kT),
                                  WF_mean=WF.mean(), WR_mean=WR.mean(), WF_std=WF.std(), WR_std=WR.std(),
                                  WF_skew=skew(WF), WR_skew=skew(WR), WF_kurt=kurtosis(WF), WR_kurt=kurtosis(WR),
                                  fdt_F=WF.var() / (2 * P.kT * (WF.mean() - dF)),
                                  fdt_R=WR.var() / (2 * P.kT * (WR.mean() + dF)),
                                  crooks_slope=sl, crooks_bins=int(ok.sum()))
    h, l = axd["a"].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    x = np.linspace(-7, 7, 2)
    axd["d"].plot(x, x, "k--", lw=1.2, label="Crooks: slope 1")
    axd["d"].set(xlabel=r"$\beta(W-\Delta F)$", ylabel=r"$\ln\,[P_F(W)/P_R(-W)]$",
                 title=r"(d) Crooks test, $N=10$")
    axd["d"].legend(fontsize=7, loc="upper left")

    Ns = np.array(N_SWEEP)
    for N in N_SWEEP:
        P, dF, WF, WR = data[("size", N)]
        res[f"size_N{N}"] = dict(dF=dF, J_F=jarzynski(WF, P.kT), J_R=jarzynski_reverse(WR, P.kT),
                                 BAR=bar(WF, WR, P.kT), BAR_err=repeat_spread(bar, WF, WR, P.kT),
                                 WF_mean=WF.mean(), WR_mean=WR.mean())
    for j, (key, lab) in enumerate([("J_F", "Jarzynski, forward"), ("J_R", "Jarzynski, reverse"), ("BAR", "BAR")]):
        y = [(res[f"size_N{N}"][key] - res[f"size_N{N}"]["dF"]) for N in N_SWEEP]
        axd["e"].plot(Ns, y, "o-", ms=4, color=CAT[j], label=lab)
    axd["e"].axhline(0, color="k", ls="--", lw=1)
    axd["e"].set(xscale="log", xlabel="beads $N$", ylabel=r"$\beta(\widehat{\Delta F}-\Delta F)$",
                 title=r"(e) estimator error, $\tau=4\tau_1$")
    axd["e"].set_xticks(N_SWEEP, [str(n) for n in N_SWEEP])
    axd["e"].xaxis.set_minor_formatter(plt.NullFormatter())
    axd["e"].legend(fontsize=7, loc="upper left")
    fig.subplots_adjust(top=0.88)
    save(fig, "F5_crooks")
    (DATA / "F5_summary.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res.items():
        print(k, {a: round(float(b), 4) for a, b in v.items()})
