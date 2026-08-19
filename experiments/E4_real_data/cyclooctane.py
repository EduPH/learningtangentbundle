"""
E4 — Cyclo-octane conformation space (real chemistry data).

The conformation space of cyclo-octane (C8H16) is a *non-manifold*: a
2-sphere and a Klein bottle glued along two disjoint circles (Martin,
Thompson, Coutsias & Watson 2010). 6040 points in R^24. This is the
flagship real-data experiment: a genuinely non-orientable stratum (the
Klein bottle) inside real molecular data, with an orientable stratum
(the sphere) as a built-in control.

Pipeline:
  1. download pointsCycloOctane.mat from javaPlex (runs on the user's
     machine; the sandbox has no GitHub egress).
  2. estimate local intrinsic dimension by PCA; the singular set (the two
     intersection circles) is where local dimension exceeds 2. Remove it.
  3. HDBSCAN the regular part into strata (Lupo et al. 2022: 1 Klein
     cluster + 3 sphere clusters).
  4. per stratum: geodesic-landmark cover -> atlas autoencoder ->
     eta_pca certificate -> orientability verdict.
     Expected: Klein stratum non-orientable (w1 != 0), sphere strata
     orientable (w1 = 0).

Usage:
  python cyclooctane.py                 # download (if needed), run all strata
  python cyclooctane.py --mat PATH.mat  # use a local copy
  python cyclooctane.py --synthetic     # sandbox self-test on a stand-in
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime

import numpy as np
import tensorflow as tf

# package + sibling driver (geodesic cover, certificate plumbing)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlasae import (AtlasAutoencoder, fast_fit, check_orientability,
                     pca_tangent_frames, compare_eta)
from codim_sweep import geodesic_landmark_cover

JAVAPLEX_URL = ("https://raw.githubusercontent.com/appliedtopology/javaplex/"
                "master/src/matlab/for_distribution/tutorial_examples/"
                "pointsCycloOctane.mat")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAT = os.path.join(DATA_DIR, "pointsCycloOctane.mat")


# ============================================================
# Data loading
# ============================================================

def download_cyclooctane(dest: str = DEFAULT_MAT) -> str:
    """Fetch pointsCycloOctane.mat from javaPlex if not already present."""
    if os.path.exists(dest):
        print(f"[data] using cached {dest}")
        return dest
    print(f"[data] downloading cyclo-octane from javaPlex ...")
    urllib.request.urlretrieve(JAVAPLEX_URL, dest)
    print(f"[data] saved to {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def load_cyclooctane(mat_path: str) -> np.ndarray:
    """Load the 6040 x 24 point cloud from the MATLAB file."""
    from scipy.io import loadmat
    md = loadmat(mat_path)
    # variable name is 'pointsCycloOctane'; fall back to first 2-D array
    if "pointsCycloOctane" in md:
        X = md["pointsCycloOctane"]
    else:
        arrays = [v for k, v in md.items()
                  if not k.startswith("__") and getattr(v, "ndim", 0) == 2]
        X = max(arrays, key=lambda a: a.size)
    X = np.asarray(X, dtype=float)
    print(f"[data] point cloud {X.shape}")
    return X


def make_synthetic(n_sphere: int = 1500, n_klein: int = 1500,
                   N: int = 24, seed: int = 0) -> np.ndarray:
    """
    Sandbox stand-in: an S^2 and a Klein bottle placed in R^24 (disjoint,
    not glued — enough to exercise stratification + per-stratum pipeline).
    """
    rng = np.random.default_rng(seed)
    # sphere in first 3 coords, offset
    s = rng.normal(size=(n_sphere, 3)); s /= np.linalg.norm(s, axis=1, keepdims=True)
    sph = np.zeros((n_sphere, N)); sph[:, :3] = s * 2.0
    # klein in coords 5..8, offset far away
    u = rng.uniform(0, 2*np.pi, n_klein); v = rng.uniform(0, 2*np.pi, n_klein)
    kb = np.stack([(4+2*np.cos(v))*np.cos(u), (4+2*np.cos(v))*np.sin(u),
                   2*np.sin(v)*np.cos(u/2), 2*np.sin(v)*np.sin(u/2)], axis=1)
    kln = np.zeros((n_klein, N)); kln[:, 4:8] = kb; kln[:, 4] += 40.0
    return np.vstack([sph, kln])


# ============================================================
# Singularity removal + stratification
# ============================================================

def local_intrinsic_dimension(X: np.ndarray, k: int = 30,
                              var_thresh: float = 0.9) -> np.ndarray:
    """
    Per-point local dimension = number of local-PCA components needed to
    capture `var_thresh` of neighbourhood variance. On a clean 2-manifold
    this is ~2; near the intersection circles it rises.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    _, idx = tree.query(X, k=k + 1)
    dims = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        nb = X[idx[i]] - X[idx[i]].mean(0)
        s = np.linalg.svd(nb, compute_uv=False) ** 2
        c = np.cumsum(s) / s.sum()
        dims[i] = int(np.searchsorted(c, var_thresh) + 1)
    return dims


def remove_singularities(X: np.ndarray, k: int = 30, var_thresh: float = 0.9,
                         max_dim: int = 2, dilate: float = 0.3):
    """
    Drop the singular set — the two circles where the sphere and Klein bottle
    meet — plus a band of width `dilate` around it. Flagging alone only thins
    the intersection circles; the strata stay connected through the residue,
    so we dilate the removed set to actually cut the necks (cf. Stolz et al.
    2020, who remove singular points before taking connected components).

    `dilate` is in the raw data units (cyclo-octane points have norm ~5);
    0.3 disconnects the regular set into 4 components matching Lupo et al.
    (2022): 1 Klein + 3 sphere. Returns (X_regular, keep_mask).
    """
    from scipy.spatial import cKDTree
    dims = local_intrinsic_dimension(X, k=k, var_thresh=var_thresh)
    sing = dims > max_dim
    remove = sing.copy()
    if dilate > 0 and sing.any():
        tree = cKDTree(X)
        for lst in tree.query_ball_point(X[sing], dilate):
            remove[lst] = True
    keep = ~remove
    print(f"[strat] singular removed: {sing.sum()} flagged + "
          f"{remove.sum() - sing.sum()} dilated = {remove.sum()} / {len(X)}")
    return X[keep], keep


def stratify(X: np.ndarray, min_cluster_size: int = 150, k: int = 15):
    """
    Split the regular part into strata as CONNECTED COMPONENTS of the kNN
    graph. Removing the intersection circles disconnects the conformation
    space: sphere-minus-two-circles = 3 pieces, Klein-minus-two-circles = 1
    piece, matching Lupo et al. (2022)'s 1 Klein + 3 sphere strata. This is
    more faithful than density clustering, which fragments the glued surface.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components
    g = kneighbors_graph(X, k, mode="connectivity")
    g = g + g.T  # symmetrise
    n_comp, labels = connected_components(g, directed=False)
    strata = [np.where(labels == c)[0] for c in range(n_comp)]
    strata = [s for s in strata if len(s) >= min_cluster_size]
    strata.sort(key=len, reverse=True)
    print(f"[strat] {n_comp} raw components -> {len(strata)} strata "
          f"(>= {min_cluster_size}), sizes {[len(s) for s in strata]}")
    return strata


# ============================================================
# Per-stratum orientability
# ============================================================

def analyse_stratum(points: np.ndarray, name: str, n_charts: int = None,
                    atlas_dir: str = None,
                    percentile: float = None, epochs: int = 4000,
                    min_points: int = 5,
                    lambda_jac: float = 0.01, lambda_diff: float = 0.01,
                    pca_k: int = 25, seed: int = 42, fig_dir: str = None) -> dict:
    np.random.seed(seed); tf.random.set_seed(seed)
    # centre + scale the stratum: tanh autoencoders need O(1) inputs, and
    # w1 is scale/translation invariant so this changes nothing topological.
    points = points - points.mean(0)
    scale = float(np.sqrt((points ** 2).sum(1).mean()))
    if scale > 0:
        points = points / scale
    # Adaptive cover: chart *size* (not count) drives per-chart eta — a single
    # oversized chart cannot be flattened and both blows up eta and breaks
    # cocycle verification. Target ~250 pts/chart via many small charts
    # (more landmarks + smaller geodesic radius), which keeps every chart
    # inside the stability regime. Small strata keep the original 8/20 recipe.
    n = len(points)
    if n_charts is None:
        n_charts = max(8, round(n / 180))
    if percentile is None:
        percentile = 20 if n <= 600 else 8
    assignments = geodesic_landmark_cover(points, n_charts,
                                          percentile=percentile, seed=seed)

    system = AtlasAutoencoder(data=points, n_charts=len(assignments),
                              subset_assignments=assignments,
                              latent_dim=2, hidden_dims=[32, 16])
    fast_fit(system, epochs=epochs, lambda_jac=lambda_jac,
             lambda_diff=lambda_diff)

    frames = pca_tangent_frames(points, d=2, k=pca_k)
    eta = compare_eta(system, points, assignments, frames_pca=frames)
    orient = check_orientability(system, points, assignments,
                                 eps_cluster=1.0, min_points=min_points,
                                 verbose=False)

    if fig_dir is not None:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
        from natural_patches import plot_nerve_signs
        _o.makedirs(fig_dir, exist_ok=True)
        plot_nerve_signs(orient, _o.path.join(fig_dir, f"nerve_{name}.png"))

    eps_sup = max(float(system.compute_varepsilon(
        i, tf.constant(points[assignments[i]], dtype=tf.float32)).numpy())
        for i in range(system.n_charts))
    n_out = sum(c['eta_pca'] > 1 for c in eta['per_chart'])
    delta = float(system.compute_delta().numpy())

    # --- nerve-truncation diagnostic ----------------------------------------
    # Overlaps with fewer than `min_points` samples are excluded from the Cech
    # computation. This matters: removing an edge can only destroy the odd
    # cycle that carries the twist, never create one, so an aggressively
    # truncated nerve biases the verdict towards "orientable" (with
    # min_points=20 this produced incorrect certified verdicts on this
    # stratum). We therefore keep min_points small and *report* how much was
    # dropped: on real data the discarded overlaps are one- to three-point
    # slivers at chart boundaries, negligible against charts of hundreds of
    # points, whereas genuine intersection regions are retained.
    n_dropped, max_dropped = 0, 0
    for a in range(len(assignments)):
        for b in range(a + 1, len(assignments)):
            n_ov = len(np.intersect1d(assignments[a], assignments[b]))
            if 0 < n_ov < min_points:
                n_dropped += 1
                max_dropped = max(max_dropped, n_ov)
    smallest_chart = min(len(a) for a in assignments)
    sliver_frac = max_dropped / max(smallest_chart, 1)

    certified = bool(eps_sup <= 0.15 and n_out <= 1 and delta > 0.005)

    verdict = ('undetermined' if orient['is_orientable'] is None
               else ('orientable' if orient['is_orientable'] else 'non-orientable'))
    print(f"[{name}] n={len(points)} eps={eps_sup:.3f} eta_pca={eta['eta_pca']:.2f} "
          f"(#>1={n_out}) delta={delta:.4f} certified={certified} "
          f"verdict={verdict}")
    if atlas_dir is not None:
        from atlasae import save_atlas
        import os as _o
        _ap = _o.path.join(atlas_dir, f'{name}_seed{seed}')
        save_atlas(system, _ap, note=f'cyclo-octane {name} seed {seed}',
                   extra=dict(seed=int(seed), stratum=name))
    return {'stratum': name, 'n_points': int(len(points)),
            'n_charts': int(system.n_charts),
            'varepsilon': eps_sup, 'eta_pca': eta['eta_pca'],
            'n_eta_outliers': n_out, 'delta': delta, 'certified': certified,
            'n_dropped_overlaps': int(n_dropped),
            'max_dropped_overlap': int(max_dropped),
            'sliver_frac': float(sliver_frac), 'min_points': int(min_points),
            'n_overlaps_used': int(len(orient.get('signs', {}))),
            'verdict': verdict, 'cocycle_verified': bool(orient['cocycle_verified']),
            'eta_per_chart': eta['per_chart']}


# ============================================================
# Driver
# ============================================================

def run(mat_path=None, synthetic=False, out_dir=None, dilate=None, seeds=1,
        **kw):
    """
    Analyse every stratum, repeating each with `seeds` independent trials
    (different network initialisation and cover landmarks per seed). The
    stratification is deterministic and computed once.
    """
    if synthetic:
        X = make_synthetic()
    else:
        X = load_cyclooctane(mat_path or download_cyclooctane())

    if dilate is None:
        dilate = 0.0 if synthetic else 0.3
    Xr, _ = remove_singularities(X, dilate=dilate)
    strata = stratify(Xr, min_cluster_size=(150 if not synthetic else 100))

    out_dir = out_dir or os.path.join(
        DATA_DIR, f"results_cyclooctane_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    _atlas_dir = os.path.join(out_dir, 'atlases')
    os.makedirs(_atlas_dir, exist_ok=True)
    results = []
    for i, idx in enumerate(strata):
        for s in range(seeds):
            seed = 42 + s
            r = analyse_stratum(Xr[idx], name=f"stratum_{i}", seed=seed,
                                atlas_dir=_atlas_dir,
                                fig_dir=(out_dir if s == 0 else None), **kw)
            r['seed'] = seed
            r['stratum_index'] = i
            results.append(r)
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=1)

    print("\n=== Summary ===")
    print("Expected: exactly one stratum non-orientable (Klein), rest orientable.")
    for i in sorted({r['stratum_index'] for r in results}):
        g = [r for r in results if r['stratum_index'] == i]
        cert = sum(r['certified'] for r in g)
        verdicts = {}
        for r in g:
            verdicts[r['verdict']] = verdicts.get(r['verdict'], 0) + 1
        vs = ', '.join(f"{v}: {n}/{len(g)}" for v, n in sorted(verdicts.items()))
        eps = np.mean([r['varepsilon'] for r in g])
        print(f"  stratum_{i}: n={g[0]['n_points']:5d}  certified {cert}/{len(g)}"
              f"  mean eps {eps:.3f}  [{vs}]")
    print(f"\nSaved to {out_dir}/results.json")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default=None, help="path to pointsCycloOctane.mat")
    p.add_argument("--synthetic", action="store_true",
                   help="sandbox self-test on a Klein + S^2 stand-in")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lambda-diff", type=float, default=0.01)
    p.add_argument("--dilate", type=float, default=None,
                   help="singular-set dilation radius (raw units; default 0.3 "
                        "gives the 4 literature strata). Tune if #strata != 4.")
    p.add_argument("--min-points", type=int, default=5,
                   help="minimum samples for an overlap to enter the nerve; "
                        "non-empty overlaps below this are dropped, which "
                        "truncates the nerve and blocks certification")
    p.add_argument("--seeds", type=int, default=1,
                   help="independent trials per stratum (different seeds)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    run(mat_path=args.mat, synthetic=args.synthetic, out_dir=args.out,
        epochs=args.epochs, lambda_diff=args.lambda_diff, dilate=args.dilate,
        seeds=args.seeds, min_points=args.min_points)
