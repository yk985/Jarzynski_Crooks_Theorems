import numpy as np

from jarz.estimators import block_curve, ess, jarzynski


def test_jarzynski_gaussian_work():
    rng = np.random.default_rng(0)
    mu, sig, kT = 5.0, 1.0, 1.0
    W = rng.normal(mu, sig, 400_000)
    assert abs(jarzynski(W, kT) - (mu - sig**2 / (2 * kT))) < 0.01


def test_ess_and_blocks():
    W = np.full(1000, 3.0)
    assert np.isclose(ess(W, 1.0), 1000)
    n, m, s = block_curve(W, 1.0, [10, 100])
    assert np.allclose(m, 3.0) and np.allclose(s, 0.0)


def test_bar_and_reverse_on_gaussian_crooks_pair():
    """Gaussian works consistent with Crooks: mu_F = dF + s^2/2kT, mu_R = -dF + s^2/2kT."""
    from jarz.estimators import bar, crooks_log_ratio, jarzynski_reverse
    rng = np.random.default_rng(1)
    kT, dF, s = 1.0, 3.0, 2.0
    WF = rng.normal(dF + s**2 / (2 * kT), s, 200_000)
    WR = rng.normal(-dF + s**2 / (2 * kT), s, 200_000)
    assert abs(bar(WF, WR, kT) - dF) < 0.02
    assert abs(jarzynski_reverse(WR, kT) - dF) < 0.05
    c, lr, err = crooks_log_ratio(WF, WR, kT, np.linspace(1, 5, 9))
    assert np.allclose(lr, (c - dF) / kT, atol=5 * np.nanmax(err))
