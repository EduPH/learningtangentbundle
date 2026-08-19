"""
Does volume regularisation raise the ceiling on delta?

Diagnosis (atlasae.gauge, Proposition "gauge covariance"):  under any latent
rescaling the attainable non-degeneracy gap is capped at

    ceiling  =  min_{i,j} sqrt( inf_x |det g_ji(x)| / sup_x |det g_ji(x)| ),

so what limits delta is not the scaling of the latent coordinates but the
*variation* of the transition Jacobian determinant across a single overlap.
Measured ceilings of 0.13 (S^2), 0.50 (Mobius) and 0.09 (RP^2) correspond to
|det g| spanning roughly 57x, 4x and 132x within some overlap.

Fix under test: penalise the variance of log|det dE_i| over each chart
(`lambda_vol` in fast_fit).  If each encoder distorts volume uniformly, then
det g_ji = det(dE_j) det(dD_i) is nearly constant on overlaps, the ceiling
rises toward 1, and -- combined with eta < 0.14 -- Theta < delta comes into
range for the first time.

This script trains each manifold with and without the term and reports
epsilon, eta, delta, the ceiling, and whether Theta < delta closes.

Usage:
    python run_volreg.py --manifolds S2 Mobius --lambda-vol 0.01 0.1 1.0
    python run_volreg.py --manifolds S2 --lambda-vol 0.1 --epochs 800   # quick
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


def theta_lower_bound(eta, d=2):
    return np.inf if eta >= 1 else d * ((3.0 - eta) * eta) ** d


def run_one(name, seed, lambda_vol, epochs=None, pca_k=25):
    import tensorflow as tf
    from atlasae import (AtlasAutoencoder, fast_fit, pca_tangent_frames,
                         compute_eta_true_pointwise, gauge_report)
    import paper_experiments as PE

    cfg = PE.MANIFOLDS[name]
    np.random.seed(seed)
    tf.random.set_seed(seed)

    points, _ = cfg["sampler"](cfg["n"], seed, **cfg.get("sampler_kw", {}))
    assignments = cfg["cover"](points, seed)
    frames = pca_tangent_frames(points, d=2, k=pca_k)

    system = AtlasAutoencoder(data=points, n_charts=len(assignments),
                              subset_assignments=assignments,
                              latent_dim=2, hidden_dims=[32, 16])
    fast_fit(system, epochs=epochs or cfg["epochs"], batch_size=64,
             lambda_jac=cfg.get("lambda_jac", 0.0),
             lambda_diff=cfg.get("lambda_diff", 0.0),
             lambda_vol=lambda_vol, verbose=False)

    import tensorflow as tf2
    eps = max(float(system.compute_varepsilon(
        i, tf2.constant(points[assignments[i]], dtype=tf2.float32)).numpy())
        for i in range(system.n_charts))
    etas = [float(compute_eta_true_pointwise(
        system.autoencoders[i], points[assignments[i]],
        frames[assignments[i]]).max()) for i in range(system.n_charts)]
    eta = max(etas)
    g = gauge_report(system, d=2)
    lb = theta_lower_bound(eta)
    return dict(manifold=name, seed=seed, lambda_vol=lambda_vol,
                varepsilon=eps, eta_pca=eta,
                n_eta_outliers=int(sum(e > 1 for e in etas)),
                theta_lower_bound=float(lb),
                closes_after=bool(lb < g["delta_after"]),
                closes_at_ceiling=bool(lb < g["ceiling"]),
                **{k: v for k, v in g.items() if k != "lambdas"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifolds", nargs="+", default=["S2", "Mobius"])
    ap.add_argument("--lambda-vol", type=float, nargs="+",
                    default=[0.0, 0.01, 0.1, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    rows = []
    print(f"{'manifold':8} {'lam_vol':>8} {'eps':>7} {'eta':>7} {'ThetaLB':>8} "
          f"{'delta':>7} {'after':>7} {'ceiling':>8}  closes?")
    print("-" * 78)
    for m in a.manifolds:
        for lv in a.lambda_vol:
            for s in a.seeds:
                r = run_one(m, s, lv, epochs=a.epochs)
                rows.append(r)
                print(f"{m:8} {lv:8.3g} {r['varepsilon']:7.4f} {r['eta_pca']:7.3f} "
                      f"{r['theta_lower_bound']:8.3f} {r['delta_before']:7.4f} "
                      f"{r['delta_after']:7.4f} {r['ceiling']:8.4f}  "
                      f"{'YES' if r['closes_after'] else 'no'}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(HERE, f"results_volreg_{stamp}")
    os.makedirs(d, exist_ok=True)
    json.dump(rows, open(os.path.join(d, "results.json"), "w"), indent=2)
    print(f"\nwrote {d}/results.json")
    print("\nread: ceiling should RISE with lambda_vol; watch that eps and eta "
          "do not degrade in exchange.")


if __name__ == "__main__":
    main()
