"""
Per-chart differential error vs chart size on the saved RP2 atlases.

Tests, at chart level, the mechanism proposed in `Why the embedding governs
certification' (main.tex, sec:exp-summary): that in the raw embedding the
charts failing eta_i < 1 are the oversized ones produced by the fixed-radius
cover, whereas after contrast normalisation no chart fails.  The recorded
RP2 runs store only the per-seed max eta, so this recomputes eta_pca per
chart from the saved atlases (a few forward passes; no retraining).

Also prints, as a control, the same correlation for the low-curvature patch
model, where it is expected to be NEGATIVE (larger charts simply train on
more data) -- the paper's claim is about the joint effect of size and
curvature, not size alone.

Usage:
    python rp2_perchart_eta.py
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

RUNS = [
    ("RP2 raw",        os.path.join(HERE, "results", "results_paper_20260804_110243", "atlases")),
    ("RP2 normalised", os.path.join(HERE, "results", "results_paper_20260804_095231", "atlases")),
]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    from atlasae import load_atlas, pca_tangent_frames, compute_eta_true_pointwise

    for tag, adir in RUNS:
        print(f"\n== {tag} ==")
        rows = []
        for path in sorted(glob.glob(os.path.join(adir, "RP2_seed*"))):
            system = load_atlas(path)
            points = np.load(os.path.join(path, "points.npy"))
            meta = json.load(open(os.path.join(path, "meta.json")))
            assignments = [np.load(os.path.join(path, "assignments", f"chart_{i}.npy"))
                           for i in range(meta["n_charts"])]
            frames = pca_tangent_frames(points, d=meta["latent_dim"], k=25)
            sizes, etas = [], []
            for i, idx in enumerate(assignments):
                e = float(compute_eta_true_pointwise(
                    system.autoencoders[i], points[idx], frames[idx]).max())
                sizes.append(len(idx))
                etas.append(e)
            sizes, etas = np.array(sizes), np.array(etas)
            rho = spearman(sizes, etas)
            out = np.where(etas > 1)[0]
            size_rank = len(sizes) - np.argsort(np.argsort(sizes))  # 1 = largest
            seed = os.path.basename(path).split("seed")[-1]
            print(f"  seed {seed}: spearman(size, eta) = {rho:+.2f}   "
                  f"eta>1 on {len(out)}/{len(sizes)} charts, "
                  f"size-ranks {sorted(int(size_rank[k]) for k in out)}")
            rows.append(dict(seed=int(seed), spearman=rho,
                             sizes=sizes.tolist(), etas=etas.tolist()))
        with open(os.path.join(HERE, f"rp2_perchart_eta_{tag.split()[-1]}.json"), "w") as f:
            json.dump(rows, f, indent=1)

    # control: recorded per-chart data of the low-curvature patch model
    pm = sorted(glob.glob(os.path.join(HERE, "E4_real_data",
                                       "results_patches_*", "results.json")),
                key=os.path.getmtime)
    for f in pm:
        r = json.load(open(f))
        if not r or not r[0].get("eta_per_chart"):
            continue
        rhos = [spearman(np.array([c["n_points"] for c in e["eta_per_chart"]]),
                         np.array([c["eta_pca"] for c in e["eta_per_chart"]]))
                for e in r]
        print(f"\ncontrol {os.path.basename(os.path.dirname(f))}: "
              f"spearman(size, eta) per seed = {[round(x, 2) for x in rhos]}")


if __name__ == "__main__":
    main()
