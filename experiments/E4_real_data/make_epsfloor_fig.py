"""
Figure: the reconstruction error on real image patches is a floor, not undertraining.

The van Hateren patch cloud sits a fixed distance from any 2-manifold. The
evidence is that epsilon does not respond to the two levers that would fix an
undertrained model -- optimisation budget and sample size -- while an exact
model of the same topology, swept identically, improves by a factor of three.
That contrast is the argument, and it reads far better as a plot than as a table.

Reads the sweeps produced by eps_floor_sweep.py; no retraining.

Usage:
    python make_epsfloor_fig.py \
        --out ../../Fcom__Atlas_Autoencoders/epsfloor_fig/epsfloor.png
"""

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    hits = sorted(glob.glob(os.path.join(HERE, f"results_epsfloor_{tag}_*/results.json")),
                  key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"no {tag} sweep found; run eps_floor_sweep.py first")
    return json.load(open(hits[-1]))


def series(rows, axis):
    r = sorted([x for x in rows if x["axis"] == axis], key=lambda x: x["value"])
    return ([x["value"] for x in r],
            np.array([x["epsilon"] for x in r]),
            np.array([x["epsilon_sd"] for x in r]))


def main(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real, model = load("real"), load("synthetic")
    axes = [("epochs", "optimisation budget (epochs)"),
            ("core", "sample size (points)")]

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    C = {"real": "#c0392b", "model": "#2471a3"}

    for k, (axis, xlabel) in enumerate(axes):
        a = ax[k]
        for rows, key, lab in ((real, "real", "van Hateren patches"),
                               (model, "model", "analytic Klein model")):
            x, y, sd = series(rows, axis)
            a.errorbar(x, y, yerr=sd, marker="o", ms=5, lw=1.8, capsize=3,
                       color=C[key], label=lab if k == 0 else None)
        a.set_xscale("log")
        a.set_xticks(series(real, axis)[0])
        a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.set_xlabel(xlabel)
        a.grid(alpha=.25, lw=.6)

    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"reconstruction error $\varepsilon$")
    ax[0].legend(fontsize=9, frameon=False, loc="center left")

    # annotate the contrast on the budget axis, where it is cleanest
    xr, yr, _ = series(real, "epochs")
    xm, ym, _ = series(model, "epochs")
    ax[0].annotate(f"flat: {yr[0]:.3f} $\\to$ {yr[-1]:.3f}",
                   xy=(xr[-1], yr[-1]), xytext=(-8, 16),
                   textcoords="offset points", fontsize=8.5,
                   color=C["real"], ha="right")
    ax[0].annotate(f"falls {ym[0]/ym[-1]:.1f}$\\times$",
                   xy=(xm[-1], ym[-1]), xytext=(-8, -20),
                   textcoords="offset points", fontsize=8.5,
                   color=C["model"], ha="right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    print(f"  real  epochs {yr[0]:.4f} -> {yr[-1]:.4f}")
    print(f"  model epochs {ym[0]:.4f} -> {ym[-1]:.4f}  ({ym[0]/ym[-1]:.1f}x)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "..", "Fcom__Atlas_Autoencoders", "epsfloor_fig", "epsfloor.png"))
    main(ap.parse_args().out)
