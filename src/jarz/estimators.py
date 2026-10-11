"""Free-energy estimators from a sample of work values."""
import numpy as np
from scipy.special import logsumexp


def jarzynski(W, kT):
    """-kT ln( mean exp(-W/kT) ), computed stably."""
    W = np.asarray(W)
    return -kT * (logsumexp(-W / kT) - np.log(len(W)))


def cumulant2(W, kT):
    """Second-cumulant estimate <W> - Var(W) / (2 kT)."""
    return np.mean(W) - np.var(W, ddof=1) / (2 * kT)


def ess(W, kT):
    """Effective sample size of the weights exp(-W/kT)."""
    lw = -np.asarray(W) / kT
    w = np.exp(lw - lw.max())
    return w.sum() ** 2 / (w**2).sum()


def bootstrap(W, kT, est=jarzynski, B=1000, seed=0):
    """Bootstrap standard error of an estimator."""
    rng = np.random.default_rng(seed)
    W = np.asarray(W)
    vals = [est(W[rng.integers(0, len(W), len(W))], kT) for _ in range(B)]
    return float(np.std(vals))


def block_curve(W, kT, sizes):
    """Jarzynski estimate using n pulls: mean and spread over disjoint blocks of size n.

    Returns arrays (n, mean, std). The mean over blocks exposes the finite-sample bias.
    """
    W = np.asarray(W)
    out = []
    for n in sizes:
        nb = len(W) // n
        if nb < 1:
            continue
        est = [jarzynski(blk, kT) for blk in W[: nb * n].reshape(nb, n)]
        out.append((n, np.mean(est), np.std(est) if nb > 1 else np.nan))
    return np.array(out).T


def summary(W, kT, dF=None):
    W = np.asarray(W)
    s = dict(M=len(W), W_mean=float(W.mean()), W_std=float(W.std(ddof=1)),
             dF_J=float(jarzynski(W, kT)), dF_J_err=bootstrap(W, kT, B=300),
             dF_c2=float(cumulant2(W, kT)), ess=float(ess(W, kT)),
             frac_below=float(np.mean(W < dF)) if dF is not None else None)
    if dF is not None:
        s.update(dF=float(dF), beta_Wdiss=float((W.mean() - dF) / kT))
    return s


# ---------------------------------------------------------------- two-sided (Crooks) estimators
# Convention: W_F are works of the forward protocol lambda_0 -> lambda_1, W_R works of the time-reversed
# protocol lambda_1 -> lambda_0 started in equilibrium at lambda_1. Crooks: P_F(W) / P_R(-W) = exp(beta (W - dF)).


def jarzynski_reverse(W_R, kT):
    """Delta F from reverse pulls: <exp(-beta W_R)> = exp(+beta Delta F)."""
    return -jarzynski(W_R, kT)


def bar(W_F, W_R, kT):
    """Bennett acceptance ratio: root of sum_F fermi(beta(W_F - dF) + ln q) = sum_R fermi(beta(W_R + dF) - ln q),
    fermi(x) = 1 / (1 + e^x), q = n_F / n_R. Minimum-variance combination of forward and reverse works."""
    from scipy.optimize import brentq
    from scipy.special import expit

    W_F, W_R = np.asarray(W_F), np.asarray(W_R)
    lnq = np.log(len(W_F) / len(W_R))

    def g(dF):
        return expit(-(W_F - dF) / kT - lnq).sum() - expit(-(W_R + dF) / kT + lnq).sum()

    lo = min(W_F.min(), -W_R.max()) - 50 * kT
    hi = max(W_F.max(), -W_R.min()) + 50 * kT
    return brentq(g, lo, hi, xtol=1e-12)


def crooks_log_ratio(W_F, W_R, kT, bins):
    """ln[P_F(W) / P_R(-W)] on histogram bins (NaN where either is empty). Crooks: = beta (W - dF)."""
    pF, _ = np.histogram(W_F, bins, density=True)
    pR, _ = np.histogram(-np.asarray(W_R), bins, density=True)
    c = 0.5 * (bins[1:] + bins[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where((pF > 0) & (pR > 0), np.log(pF / pR), np.nan)
    nF, _ = np.histogram(W_F, bins)
    nR, _ = np.histogram(-np.asarray(W_R), bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.sqrt(1.0 / nF + 1.0 / nR)          # Poisson error of the log ratio
    return c, lr, err
