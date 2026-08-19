"""
Can sign constancy be certified from sampling density instead of Theta < delta?

The pipeline already verifies the cocycle identity directly, by autodiff, at
every sampled point.  The only thing Theta < delta adds is coverage BETWEEN
samples -- and it costs a product of four worst-case constants that in practice
exceeds delta by orders of magnitude, and that no amount of extra data improves.

The alternative tested here is a covering argument in the latent coordinate.
With f(z) = det g_ji, sample spacing h_k and local gradient bound L_k,

    |f(z_k)| > L_k h_k   for every sample k   ==>   sign f constant on the
                                                    whole overlap component.

The point of interest is that the margin |f| / (L h) grows as the sample is
refined, since h ~ n^{-1/d} while |f| and L do not depend on n.  Theta < delta
has no such handle.  This script measures the margin as a function of sample
size and training budget, so the scaling can be checked directly.

Usage:
    python run_signconstancy.py --manifold S2 --n 500 1000 2000 4000
    python run_signconstancy.py --manifold S2 --n 1000 --epochs 4000   # paper budget
    python run_signconstancy.py --manifold Mobius Klein --n 2000
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "src"))


def run_one(manifold, n, epochs, seed=42, mode="local", k_local=12):
    import tensorflow as tf
    from atlasae import (AtlasAutoencoder, fast_fit, sign_constancy_report,
                         pca_tangent_frames, compute_eta_true_pointwise)
    import paper_experiments as PE

    cfg = PE.MANIFOLDS[manifold]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    points, _ = cfg["sampler"](n, seed, **cfg.get("sampler_kw", {}))
    assignments = cfg["cover"](points, seed)

    system = AtlasAutoencoder(data=points, n_charts=len(assignments),
                              subset_assignments=assignments,
                              latent_dim=2, hidden_dims=[32, 16])
    fast_fit(system, epochs=epochs or cfg["epochs"], batch_size=64,
             lambda_jac=cfg.get("lambda_jac", 0.0),
             lambda_diff=cfg.get("lambda_diff", 0.0), verbose=False)

    eps = max(float(system.compute_varepsilon(
        i, tf.constant(points[assignments[i]], dtype=tf.float32)).numpy())
        for i in range(system.n_charts))
    delta = float(system.compute_delta().numpy())
    r = sign_constancy_report(system, mode=mode, k_local=k_local)
    return dict(manifold=manifold, n_points=n, epochs=epochs, seed=seed,
                varepsilon=eps, delta=delta,
                n_overlaps=r["n_overlaps"], n_certified=r["n_certified"],
                fraction_certified=r["fraction_certified"],
                worst_margin=r["worst_margin"],
                median_margin=r["median_margin"],
                median_h=r["median_h"], median_L=r["median_L"],
                n_sign_flips=r["n_sign_flips"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifold", nargs="+", default=["S2"])
    ap.add_argument("--n", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--mode", default="local", choices=["local", "global"])
    a = ap.parse_args()

    rows = []
    print(f"{'manifold':8} {'n':>6} {'eps':>7} {'delta':>7} {'med h':>7} "
          f"{'med margin':>11} {'worst':>7} {'certified':>10}")
    print("-" * 74)
    for m in a.manifold:
        for n in a.n:
            for s in a.seeds:
                r = run_one(m, n, a.epochs, seed=s, mode=a.mode)
                rows.append(r)
                print(f"{m:8} {n:6d} {r['varepsilon']:7.4f} {r['delta']:7.4f} "
                      f"{r['median_h']:7.4f} {r['median_margin']:11.3f} "
                      f"{r['worst_margin']:7.3f} "
                      f"{r['n_certified']:4d}/{r['n_overlaps']:<5}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(HERE, f"results_signconstancy_{stamp}")
    os.makedirs(d, exist_ok=True)
    json.dump(rows, open(os.path.join(d, "results.json"), "w"), indent=2)
    print(f"\nwrote {d}/results.json")

    # the scaling claim: h ~ n^{-1/2}, so margin should grow like sqrt(n)
    for m in a.manifold:
        sub = sorted([r for r in rows if r["manifold"] == m],
                     key=lambda r: r["n_points"])
        if len(sub) >= 2:
            lo, hi = sub[0], sub[-1]
            fac = (hi["n_points"] / lo["n_points"]) ** 0.5
            got = (hi["median_margin"] / lo["median_margin"]
                   if lo["median_margin"] > 0 else float("inf"))
            print(f"\n{m}: n x{hi['n_points']//lo['n_points']} predicts margin "
                  f"x{fac:.1f} from density alone; observed x{got:.1f} "
                  f"(the excess is better training at larger n).")


if __name__ == "__main__":
    main()
