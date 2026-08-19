"""
Figure: why the same manifold certifies in one embedding and not another.

Reproduces cyclo_fig/rp2_embedding.png, which had no generator in the repo.

(a) chart-size distribution of the landmark cover on raw versus contrast-
    normalised RP^2 line patches -- the raw cloud has a strongly non-uniform
    density, so the same construction yields a few enormous charts;
(b) the resulting per-chart eta_pca, which is the condition the raw embedding
    fails (it meets the epsilon and delta thresholds).

Panel (b) needs trained atlases and is read from the recorded runs rather than
retrained, so run the RP2 experiments first (see REGENERATE.md).

Usage:
    python make_rp2_embedding_fig.py --out ../Fcom__Atlas_Autoencoders/cyclo_fig/rp2_embedding.png
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def cover_sizes(seeds=(42, 43, 44, 45, 46)):
    import paper_experiments as PE
    out = {}
    for norm in (False, True):
        sizes = []
        for s in seeds:
            pts, _ = PE.sample_RP2(5625, s, sphere_normalize=norm)
            sizes.append(np.array([len(a) for a in PE.cover_RP2(pts, s)]))
        out["normalised" if norm else "raw"] = sizes
    return out


def per_chart_eta():
    """eta_pca from the recorded RP2 runs.

    Returns (values, n_charts, n_outliers, per_chart) where per_chart says
    whether the values are genuinely per-chart or only the per-seed maximum
    (the RP2 runs record eta_per_chart only for some configurations).
    """
    out = {}
    for key, pat in (("raw", "results_paper_*/RP2_raw.json"),
                     ("normalised", "results_paper_*/RP2.json")):
        hits = sorted(glob.glob(os.path.join(HERE, pat)), key=os.path.getmtime)
        if not hits:
            continue
        recs = json.load(open(hits[-1]))
        vals, per_chart = [], True
        for r in recs:
            pc = r.get("eta_per_chart")
            if not pc:
                vals.append(r["eta_pca"])
                per_chart = False
            elif isinstance(pc[0], dict):
                vals.extend(c["eta_pca"] for c in pc)
            else:
                vals.extend(pc)
        out[key] = dict(
            vals=np.asarray(vals, dtype=float),
            per_chart=per_chart,
            n_charts=int(np.mean([r.get("n_charts", 0) for r in recs])),
            n_out=float(np.mean([r.get("n_eta_outliers", 0) for r in recs])),
            n_seeds=len(recs),
        )
    return out


def main(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes, etas = cover_sizes(), per_chart_eta()
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.8))
    cols = {"raw": "#c0392b", "normalised": "#2471a3"}

    for k in ("raw", "normalised"):
        s = np.concatenate(sizes[k])
        ax[0].hist(s, bins=np.linspace(0, 3000, 31), alpha=.62,
                   color=cols[k], label=f"{k} (largest {s.max()}, "
                                        f"{s.max()/s.mean():.1f}$\\times$ mean)")
    ax[0].set_xlabel("chart size (points, of 5625)")
    ax[0].set_ylabel("charts")
    ax[0].set_title("(a) cover produced by the same construction", fontsize=10)
    ax[0].legend(fontsize=8)

    if etas:
        pos, lbl = [], []
        any_per_chart = any(e["per_chart"] for e in etas.values())
        for n, k in enumerate(("raw", "normalised")):
            if k not in etas:
                continue
            e = etas[k]
            ax[1].scatter(np.full(len(e["vals"]), n)
                          + np.random.default_rng(0).normal(0, .045, len(e["vals"])),
                          e["vals"], s=34, color=cols[k], alpha=.8,
                          edgecolor="k", linewidth=.35, zorder=3)
            pos.append(n)
            lbl.append(f"{k}\n{e['n_out']:.1f}/{e['n_charts']} charts $>1$")
        ax[1].axhline(1, color="crimson", lw=1.2, zorder=2)
        ax[1].text(.02, 1.06, r"$\eta_{\rm pca}=1$", color="crimson",
                   fontsize=8, transform=ax[1].get_yaxis_transform())
        ax[1].set_yscale("log")
        ax[1].set_xlim(-.6, 1.6)
        ax[1].set_xticks(pos)
        ax[1].set_xticklabels(lbl, fontsize=8.5)
        ax[1].set_ylabel(r"$\eta_{\rm pca}$" if any_per_chart
                         else r"$\eta_{\rm pca}$ (max over charts, per seed)")
        ax[1].set_title("(b) the condition the raw embedding fails", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "Fcom__Atlas_Autoencoders", "cyclo_fig", "rp2_embedding.png"))
    main(ap.parse_args().out)
