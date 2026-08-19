"""
Is the van Hateren reconstruction error a floor, or just undertraining?

The paper interprets the large epsilon on real photographic patches as an
intrinsic *thickness* of the empirical cloud -- the data is not close to any
exact 2-manifold -- rather than as a training failure.  That interpretation is
what licenses the conclusion that no cover or embedding could make eta < 1
attainable there, so it needs evidence: if epsilon fell steadily with more
epochs, smaller charts or more data, the honest reading would be undertraining.

This sweep varies, one axis at a time from a common baseline,

    epochs      2000 / 4000 / 8000       (optimisation budget)
    core_size   2000 / 3500 / 5000       (sample size)
    n_charts    via target chart size    (cover granularity)

and reports epsilon for each.  A floor shows up as epsilon flat in all three.
The analytic patch model is run alongside as a positive control: it *should*
improve with budget, since it is an exact manifold.

Usage:
    python eps_floor_sweep.py --images /path/to/vanhateren_iml
    python eps_floor_sweep.py --synthetic            # control only
    python eps_floor_sweep.py --images ... --quick   # 1 seed, short
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import natural_patches as NP


def one(core, patches, n_charts_target, epochs, seed, pca_k=25,
        width=(32, 16)):
    """Train an atlas on a fixed core and return the sup reconstruction error."""
    import tensorflow as tf
    from atlasae import AtlasAutoencoder, fast_fit

    np.random.seed(seed)
    tf.random.set_seed(seed)
    n_charts = max(8, round(len(core) / n_charts_target))
    from codim_sweep import knn_landmark_cover
    assignments = knn_landmark_cover(core, n_charts, k_landmark=3, seed=seed)

    system = AtlasAutoencoder(data=core, n_charts=len(assignments),
                              subset_assignments=assignments,
                              latent_dim=2, hidden_dims=list(width))
    fast_fit(system, epochs=epochs, batch_size=64,
             lambda_jac=0.0, lambda_diff=0.01, verbose=False)
    eps = max(float(system.compute_varepsilon(
        i, tf.constant(core[assignments[i]], dtype=tf.float32)).numpy())
        for i in range(system.n_charts))
    return eps, len(assignments)


def build_core(images, synthetic, max_images, n_patches, density_frac, core_size):
    D, B = NP.dct_sphere_basis()
    if synthetic:
        return NP.make_synthetic(n=core_size)
    imgs = NP.load_images(images, max_images=max_images)
    patches = NP.sample_patches(imgs, n_patches=n_patches)
    return NP.to_klein_core(patches, D, B, density_frac=density_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--n-patches", type=int, default=200000)
    ap.add_argument("--density-frac", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not a.synthetic and not a.images:
        ap.error("give --images FOLDER or --synthetic")

    epochs_axis = [2000, 4000, 8000]
    core_axis = [2000, 3500, 5000]
    chart_axis = [120, 180, 260]          # target points per chart
    width_axis = [(16, 8), (32, 16), (64, 32), (128, 64)]   # encoder widths
    base = dict(epochs=4000, core=3500, chart=180, width=(32, 16))
    if a.quick:
        epochs_axis, core_axis, chart_axis = [500, 1000], [1500], [180]
        width_axis = [(32, 16), (64, 32)]
        base = dict(epochs=1000, core=1500, chart=180, width=(32, 16))
        a.seeds = 1

    full_core, full_patches = build_core(
        a.images, a.synthetic, a.max_images, a.n_patches,
        a.density_frac, max(core_axis))
    print(f"[sweep] core pool: {len(full_core)} points "
          f"({'synthetic' if a.synthetic else 'van Hateren'})")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "synthetic" if a.synthetic else "real"
    outdir = os.path.join(HERE, f"results_epsfloor_{tag}_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "results.json")

    rows = []

    def sweep(axis_name, values):
        for v in values:
            cfg = dict(base)
            cfg[axis_name] = v
            eps_list = []
            for s in range(a.seeds):
                seed = 42 + s
                rng = np.random.default_rng(seed)
                n = min(cfg["core"], len(full_core))
                idx = rng.choice(len(full_core), n, replace=False)
                e, nc = one(full_core[idx], full_patches[idx],
                            cfg["chart"], cfg["epochs"], seed,
                            width=cfg["width"])
                eps_list.append(e)
            m, sd = float(np.mean(eps_list)), float(np.std(eps_list))
            rec = dict(cfg)
            rec["width"] = "x".join(str(w) for w in rec["width"])
            rows.append(dict(axis=axis_name,
                             value=(v if axis_name != "width"
                                    else "x".join(str(w) for w in v)),
                             epsilon=m, epsilon_sd=sd,
                             n_charts=nc, seeds=a.seeds,
                             source="synthetic" if a.synthetic else "van Hateren",
                             **rec))
            json.dump(rows, open(outfile, "w"), indent=2)   # save as we go
            label = "x".join(map(str, v)) if isinstance(v, tuple) else str(v)
            print(f"  {axis_name}={label:<6} eps = {m:.4f} +- {sd:.4f}  "
                  f"({nc} charts)")

    print("\nvarying optimisation budget:")
    sweep("epochs", epochs_axis)
    print("varying sample size:")
    sweep("core", core_axis)
    print("varying cover granularity (target points per chart):")
    sweep("chart", chart_axis)
    print("varying encoder width (decoder mirrored):")
    sweep("width", width_axis)

    e = np.array([r["epsilon"] for r in rows])
    spread = (e.max() - e.min()) / e.mean()
    print(f"\nepsilon over the whole sweep: {e.min():.4f} to {e.max():.4f} "
          f"(relative spread {spread:.1%})")
    print("  -> a floor (thickness) if flat; undertraining if it falls with "
          "epochs or sample size")

    json.dump(rows, open(outfile, "w"), indent=2)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main()
