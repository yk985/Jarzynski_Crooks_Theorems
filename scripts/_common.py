import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
FIG = ROOT / "figures"
ANIM = ROOT / "animations"
for d in (DATA, FIG, ANIM):
    d.mkdir(exist_ok=True)

T_SWEEP = [0.25, 0.5, 1.0, 2.0]
N_SWEEP = [5, 10, 20, 40]

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jarz.dynamics import best_device, initial_state, protocol, run
from jarz.estimators import summary

RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # ordinal (sweeps), light -> dark
RAMP3 = ["#86b6ef", "#2a78d6", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical, fixed order
INK, MUTED = "#1f1f1e", "#6b6a64"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def simulate(tag, P, tau, M, seed=0, dt=0.01, dF=None, reverse=False):
    """Run (or load cached) M pulls; store work values and a JSON summary.

    reverse=True runs the time-reversed protocol x1 -> x0 from equilibrium at x1 (pass dF = -Delta F)."""
    f = DATA / "runs" / f"{tag}.npz"
    f.parent.mkdir(exist_ok=True)
    if f.exists():
        return np.load(f)["W"]
    lam = protocol(P, max(1, int(round(tau / dt))))
    W = run(P, tau, M, dt=dt, seed=seed, device=best_device(), lam=lam[::-1] if reverse else lam)["W"]
    np.savez(f, W=W, tau=tau, dt=dt)
    s = summary(W, P.kT, dF)
    s.update(tag=tag, params=P.__dict__, tau=tau)
    (DATA / "runs" / f"{tag}.json").write_text(json.dumps(s, indent=1))
    print(tag, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}, flush=True)
    return W


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=110)
    plt.close(fig)


def load_run(tag):
    f = DATA / "runs" / f"{tag}.npz"
    return np.load(f)["W"] if f.exists() else None


def expw_panel(ax, W, kT, dF, color, WR=None, gauss=None, nb=60, legend=False):
    """Distribution of x = exp(-beta (W - dF)), whose exact mean is 1 (Jarzynski).

    filled: fraction of pulls per bin; solid: share of the sample mean contributed by each bin;
    dashed: exact share, from reverse pulls via Crooks (P_F(W) e^{-beta(W-dF)} = P_R(-W)) or from the
    exact Gaussian (gauss = (mean, var) of W). Vertical lines: exact mean 1, sample mean."""
    from scipy.stats import norm
    lx = -(np.asarray(W) - dF) / kT / np.log(10)                      # log10 x
    lo, hi = lx.min(), max(lx.max(), 0.5)
    if WR is not None:
        lxr = (np.asarray(WR) + dF) / kT / np.log(10)                 # reverse pulls mapped to W = -W_R
        lo, hi = min(lo, np.percentile(lxr, 0.1)), max(hi, np.percentile(lxr, 99.9))
    if gauss is not None:
        mu, var = gauss
        m, s = -(mu - dF) / kT / np.log(10), np.sqrt(var) / kT / np.log(10)
        ms = m + var / kT**2 / np.log(10)                                # share density: shifted by beta^2 var
        lo, hi = min(lo, m - 4.5 * s), max(hi, ms + 4.5 * s)
    edges = np.linspace(lo, hi, nb + 1)
    x = 10.0**lx
    ax.hist(lx, edges, weights=np.full(len(lx), 1 / len(lx)), histtype="stepfilled", color=color, alpha=0.3,
            label="pulls")
    ax.hist(lx, edges, weights=x / x.sum(), histtype="step", color=color, lw=1.5, label="share of sample mean")
    if WR is not None:
        ax.hist(lxr, edges, weights=np.full(len(lxr), 1 / len(lxr)), histtype="step", color=INK, lw=1, ls="--",
                label="exact share (reverse pulls)")
    if gauss is not None:
        c = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(c, np.diff(norm.cdf(edges, m, s)), color=INK, lw=0.8, ls=":", label="exact (Gaussian)")
        ax.plot(c, np.diff(norm.cdf(edges, ms, s)), color=INK, lw=1, ls="--", label="exact share")
    ax.axvline(0, color="k", lw=1.2, label=r"exact mean $=1$")
    ax.axvline(np.log10(x.mean()), color=color, ls=":", lw=1.5, label="sample mean")
    ax.set_xlabel(r"$\log_{10}\, e^{-\beta(W-\Delta F)}$")
    if legend:
        ax.legend(fontsize=6, loc="upper left")
