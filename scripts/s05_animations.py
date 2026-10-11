"""A1: one chain pulled slowly vs fast. A2: work histogram and running Jarzynski estimate pull after pull."""
import matplotlib.animation as animation

from _common import ANIM, CAT, DATA, FIG, INK, MUTED, RAMP3, np, plt, run
from jarz.estimators import jarzynski
from jarz.params import Chain, tau1
from jarz.reference import delta_F

P = Chain(N=10, b=1.0)
dF, t1 = delta_F(P), tau1(P)
FRAMES = 160
WRITER = dict(fps=24, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])


def typical_pull(c, seed):
    """Simulate 64 pulls with recording; keep the one whose W is closest to the median."""
    K = int(round(c * t1 / 0.01))
    out = run(P, c * t1, 64, seed=seed, device="cuda", n_record=64, record_every=max(1, K // FRAMES))
    i = int(np.argmin(np.abs(out["W"] - np.median(out["W"]))))
    lam_t = np.interp(out["rec_t"], np.arange(len(out["lam"])) * 0.01, out["lam"])
    f = np.linspace(0, len(out["rec_t"]) - 1, FRAMES + 1).round().astype(int)   # same frames for every pull
    return dict(r=out["rec_r"][f, i], W=out["rec_W"][f, i], s=out["rec_t"][f] / (c * t1), lam=lam_t[f], c=c)


def anim_chain():
    pulls = [typical_pull(30, 501), typical_pull(1, 502)]
    nf = min(len(p["s"]) for p in pulls)
    fig = plt.figure(figsize=(8, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.45, wspace=0.25)
    axc = [fig.add_subplot(gs[0, j]) for j in range(2)]
    axw = fig.add_subplot(gs[1, :])
    arts = []
    for j, (a, p) in enumerate(zip(axc, pulls)):
        a.set(xlim=(-0.8, 8.3), ylim=(-3, 3), aspect="equal", title=rf"$\tau = {p['c']}\,\tau_1$"
              + (" (slow)" if p["c"] > 1 else " (fast)"))
        a.grid(False)
        a.axvspan(-0.8, 0, color="#d9d8d2", lw=0)
        line, = a.plot([], [], "-", color=MUTED, lw=1.2, zorder=1)
        beads = a.scatter([], [], s=36, color=CAT[0], zorder=3, edgecolor="white", linewidth=0.8)
        spring, = a.plot([], [], ":", color=CAT[1], lw=1.5, zorder=2)
        trap = a.scatter([], [], s=140, facecolor="none", edgecolor=CAT[1], linewidth=2, zorder=4)
        txt = a.text(0.02, 0.04, "", transform=a.transAxes, fontsize=8, color=INK)
        arts.append((line, beads, spring, trap, txt))
    for j, p in enumerate(pulls):
        axw.plot(p["s"][:nf], p["W"][:nf], color=RAMP3[[0, 2][j]], lw=0.8, alpha=0.3)
    dots = [axw.plot([], [], color=RAMP3[[0, 2][j]], lw=2, label=rf"$\tau={p['c']}\tau_1$")[0]
            for j, p in enumerate(pulls)]
    axw.axhline(dF, color="k", ls="--", lw=1, label=r"$\Delta F$")
    axw.set(xlabel=r"$t/\tau$", ylabel=r"work $W(t)$ [$\varepsilon$]", xlim=(0, 1))
    axw.legend(loc="upper left", fontsize=8)

    def update(f):
        for (line, beads, spring, trap, txt), p, d in zip(arts, pulls, dots):
            r = np.vstack([[0, 0], p["r"][f]])
            line.set_data(r[:, 0], r[:, 1])
            beads.set_offsets(r[1:])
            spring.set_data([r[-1, 0], p["lam"][f]], [r[-1, 1], 0])
            trap.set_offsets([[p["lam"][f], 0]])
            txt.set_text(rf"$W = {p['W'][f]:5.1f}\,\varepsilon$")
            d.set_data(p["s"][: f + 1], p["W"][: f + 1])
        return []

    ani = animation.FuncAnimation(fig, update, frames=nf, blit=False)
    ani.save(ANIM / "A1_chain_pull.mp4", writer=animation.FFMpegWriter(**WRITER), dpi=120)
    update(int(0.7 * nf))
    fig.savefig(FIG / "A1_still.pdf", bbox_inches="tight")
    plt.close(fig)


def anim_histogram():
    cases = [(3, np.load(DATA / "runs" / "F2_b1_N10_c3.npz")["W"]), (1, np.load(DATA / "runs" / "F2_b1_N10_c1.npz")["W"])]
    ns = np.unique(np.logspace(0, 5, FRAMES).astype(int))
    fig, ax = plt.subplots(2, 2, figsize=(8, 4.8), gridspec_kw=dict(hspace=0.55, wspace=0.3))
    arts = []
    for row, (c, W) in enumerate(cases):
        bins = np.linspace(min(W.min(), dF) - 1, np.percentile(W, 99.9), 80)
        run_J = np.array([jarzynski(W[:n], P.kT) for n in ns])
        a, b = ax[row]
        a.set(xlim=(bins[0], bins[-1]), xlabel=r"$W$ [$\varepsilon$]", ylabel="pulls",
              title=rf"$\tau={c}\,\tau_1$:  work histogram")
        a.axvline(dF, color="k", ls="--", lw=1)
        b.axhline(dF, color="k", ls="--", lw=1, label=r"exact $\Delta F$")
        b.axhline(W.mean(), color=CAT[1], ls=":", lw=1.2, label=r"$\langle W\rangle$")
        b.set(xscale="log", xlim=(1, 1e5), ylim=(min(run_J.min(), dF) - 1, W.mean() + 2), xlabel="pulls $n$",
              ylabel=r"$\widehat{\Delta F}_J$ [$\varepsilon$]", title="running Jarzynski estimate")
        b.legend(fontsize=7, loc="center right")
        Jl, = b.plot([], [], color=CAT[0])
        Jv = a.axvline(np.nan, color=CAT[0], lw=1.5)
        txt = a.text(0.98, 0.9, "", transform=a.transAxes, ha="right", fontsize=8)
        arts.append((a, W, bins, run_J, Jl, Jv, txt))

    def update(f):
        n = ns[f]
        for a, W, bins, run_J, Jl, Jv, txt in arts:
            for pt in list(a.patches):
                pt.remove()
            a.hist(W[:n], bins=bins, color=RAMP3[1], alpha=0.75)
            Jl.set_data(ns[: f + 1], run_J[: f + 1])
            Jv.set_xdata([run_J[f]])
            txt.set_text(f"n = {n:,}")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(ns), blit=False)
    ani.save(ANIM / "A2_work_histogram.mp4", writer=animation.FFMpegWriter(**WRITER), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    anim_chain()
    anim_histogram()
    print("animations written to", ANIM)
