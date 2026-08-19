"""
Validation of eta_pca against the analytic eta.

eta_pca replaces the true tangent frames of Definition (approximate atlas)(4)
by local-PCA frames.  The question that matters for soundness is not how
accurately it approximates eta, but whether it can produce a *false
certification*, i.e. whether

        eta_pca <= 1 < eta_true

can occur -- the estimator passing the binding hypothesis when the true
quantity fails it.

This script pools every run in which both quantities were recorded (the
codimension sweeps and the three synthetic manifolds with analytic
parametrisations), reports the agreement statistics, and draws the scatter
plot with the eta = 1 crosshairs, so that the dangerous quadrant is visible.

Usage:
    python eta_validation.py                    # statistics
    python eta_validation.py --fig OUT.png      # figure
    python eta_validation.py --latex            # numbers for the paper
"""

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)

# (label, glob) -- every source that records both eta_true and eta_pca
SOURCES = [
    ("codim sweep, $S^2$",     "E1_eta_codim/results/*_S2_*/results.json"),
    ("codim sweep, Klein",     "E1_eta_codim/results/*_Klein_*/results.json"),
    ("$S^2$",                  "results/results_paper_*/S2.json"),
    ("M\\\"obius band",        "results/results_paper_*/Mobius.json"),
    ("Klein bottle ($\\R^4$)", "results/results_paper_*/Klein.json"),
]


def load():
    """-> list of dicts with eta_true, eta_pca, N, label."""
    out, seen = [], set()
    for label, pat in SOURCES:
        for path in sorted(glob.glob(os.path.join(EXP, pat))):
            recs = json.load(open(path))
            recs = recs if isinstance(recs, list) else [recs]
            for r in recs:
                et, ep = r.get("eta_true"), r.get("eta_pca")
                if not et or not ep:
                    continue
                key = (round(et, 9), round(ep, 9))
                if key in seen:          # RESULTS/ holds duplicate copies
                    continue
                seen.add(key)
                out.append(dict(eta_true=float(et), eta_pca=float(ep),
                                N=r.get("N", 3), label=label))
    return out


def stats(rows):
    et = np.array([r["eta_true"] for r in rows])
    ep = np.array([r["eta_pca"] for r in rows])
    rel = 100 * (ep - et) / et
    unsound = int(((ep <= 1) & (et > 1)).sum())
    conserv = int(((ep > 1) & (et <= 1)).sum())
    return dict(
        n=len(rows), rel=rel, et=et, ep=ep,
        corr=float(np.corrcoef(np.log(et), np.log(ep))[0, 1]),
        median=float(np.median(rel)), median_abs=float(np.median(abs(rel))),
        p90=float(np.percentile(abs(rel), 90)),
        lo=float(rel.min()), hi=float(rel.max()),
        over=int((rel > 0).sum()),
        agree=int(((et <= 1) == (ep <= 1)).sum()),
        unsound=unsound, conservative=conserv,
        eta_min=float(et.min()), eta_max=float(et.max()),
        codim_min=int(min(r["N"] for r in rows) - 2),
        codim_max=int(max(r["N"] for r in rows) - 2),
    )


def report(s):
    print(f"\npaired trials: {s['n']}   "
          f"eta_true range [{s['eta_min']:.2f}, {s['eta_max']:.1f}]   "
          f"codimension {s['codim_min']}-{s['codim_max']}")
    print(f"log-log correlation           : {s['corr']:.4f}")
    print(f"relative error (pca - true)   : median {s['median']:+.1f}%  "
          f"|median| {s['median_abs']:.1f}%  p90 {s['p90']:.1f}%  "
          f"range [{s['lo']:+.1f}%, {s['hi']:+.1f}%]")
    print(f"over-estimates                : {s['over']}/{s['n']}")
    print(f"agree on the eta<1 test       : {s['agree']}/{s['n']}")
    print(f"conservative disagreements    : {s['conservative']}  "
          f"(eta_pca>1 >= eta_true)")
    print(f"FALSE CERTIFICATIONS          : {s['unsound']}  "
          f"(eta_pca<=1 < eta_true)")
    print("\nrelative error by magnitude of eta_true:")
    for lo, hi in [(0, 0.5), (0.5, 2), (2, 10), (10, np.inf)]:
        m = (s["et"] >= lo) & (s["et"] < hi)
        if m.sum():
            r = s["rel"][m]
            hs = "inf" if hi == np.inf else f"{hi}"
            print(f"   [{lo}, {hs}) : n={m.sum():3d}  median {np.median(r):+6.1f}%"
                  f"  max |rel| {abs(r).max():6.1f}%")


def figure(rows, s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    codim = np.array([r["N"] - 2 for r in rows])
    sc = ax.scatter(s["et"], s["ep"], c=codim, cmap="viridis",
                    s=34, alpha=.85, edgecolor="k", linewidth=.35, zorder=3,
                    norm=matplotlib.colors.LogNorm())

    lim = [0.12, 60]
    ax.plot(lim, lim, "k-", lw=.9, alpha=.55, zorder=1)
    for f, ls in ((1.25, "--"), (0.8, "--")):
        ax.plot(lim, [f * x for x in lim], ls, color="k", lw=.7, alpha=.3, zorder=1)

    # the eta = 1 decision crosshairs
    ax.axhline(1, color="crimson", lw=1.1, zorder=2)
    ax.axvline(1, color="crimson", lw=1.1, zorder=2)
    # the quadrant that would break soundness: eta_pca <= 1 < eta_true
    ax.add_patch(plt.Rectangle((1, lim[0]), lim[1] - 1, 1 - lim[0],
                               facecolor="crimson", alpha=.13, zorder=0))
    ax.text(6.5, 0.30, "false certification\n$\\eta_{\\rm pca}\\leq1<\\eta$\n"
                       f"({s['unsound']} of {s['n']} trials)",
            ha="center", va="center", fontsize=8.4, color="crimson")
    ax.text(0.30, 6.5, "conservative\n$\\eta_{\\rm pca}>1\\geq\\eta$\n"
                       f"({s['conservative']} trials)",
            ha="center", va="center", fontsize=8.4, color="dimgray")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"analytic $\eta$ (true tangent frames)")
    ax.set_ylabel(r"$\eta_{\mathrm{pca}}$ (local-PCA frames)")
    ax.set_title(f"$n={s['n']}$ trials, codimension "
                 f"{s['codim_min']}--{s['codim_max']}\n"
                 f"log--log correlation {s['corr']:.3f}, "
                 f"agree on $\\eta<1$ in {s['agree']}/{s['n']}", fontsize=9.5)
    cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.03)
    cb.set_label("codimension $N-d$", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    rows = load()
    s = stats(rows)
    report(s)
    if a.fig:
        figure(rows, s, a.fig)
    if a.latex:
        print(f"\n%% n={s['n']}, corr={s['corr']:.3f}, agree={s['agree']}/{s['n']}, "
              f"conservative={s['conservative']}, unsound={s['unsound']}, "
              f"median={s['median']:+.1f}%%, codim {s['codim_min']}-{s['codim_max']}")
