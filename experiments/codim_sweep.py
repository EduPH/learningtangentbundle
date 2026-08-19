"""
Codimension sweep (E1.3): validate the eta_lat proxy against the true
tangent-restricted eta across ambient dimension.

For S² and the Klein bottle, embed the manifold isometrically into R^N for
increasing N (zero-padding followed by a random orthogonal map), train the
same atlas autoencoder at every N, and measure:

    eta_true  (ground-truth tangent frames, the quantity in the theorems)
    eta_pca   (local-PCA tangent frames — the practical instrument)
    eta_lat   (latent proxy currently reported in the paper)
    delta, orientability verdict

Prediction (paper Sec 7.3.2 narrative): eta_true stays controlled while
eta_lat inflates with codimension.

Usage:
    python codim_sweep.py --manifold S2 --dims 3 10 25 50 100 --seeds 5
    python codim_sweep.py --manifold Klein --dims 4 10 25 50 100 --seeds 5
    python codim_sweep.py --plot results_codim_S2_<stamp>/results.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf

# Silence tf.function retracing warnings: batch_jacobian rebuilds its pfor
# tf.function on every eager call (varying batch shapes), which is noisy but
# harmless. TODO (post-sweep, changes timing not results): wrap train_step in
# @tf.function with fixed signatures / batch with drop_remainder=True.
tf.get_logger().setLevel('ERROR')

from atlasae.atlasautoencoder import AtlasAutoencoder
from atlasae.orientability import check_orientability
from atlasae.sphere_good_cover import tetrahedral_cover_S2
from atlasae.eta_true import (
    sphere_tangent_frames,
    klein_param,
    klein_tangent_frames,
    pca_tangent_frames,
    frame_alignment,
    compare_eta,
    summarize_comparison,
    compute_eta_true_pointwise,
)


# ============================================================
# Samplers returning (points, ground-truth frames)
# ============================================================

def sample_S2(n: int, seed: int):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts, sphere_tangent_frames(pts)


def sample_klein(n: int, seed: int, m: float = 4.0, r: float = 2.0):
    # r=2.0 matches the paper's CODE (dreimac klein_bottle_4d(n, 4, 2)),
    # which differs from the r=1 formula printed in the paper text.
    rng = np.random.default_rng(seed)
    uv = np.stack([rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)], axis=1)
    return klein_param(uv, m, r), klein_tangent_frames(uv, m, r)


# ============================================================
# Random isometric embedding R^n0 -> R^N
# ============================================================

def random_isometric_embedding(points, frames, N: int, seed: int):
    """Zero-pad to R^N, then apply a random orthogonal Q (Haar via QR)."""
    n, n0 = points.shape
    if N == n0:
        return points.copy(), frames.copy()
    if N < n0:
        raise ValueError("target dimension below ambient dimension")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
    pad_p = np.concatenate([points, np.zeros((n, N - n0))], axis=1)
    pad_f = np.concatenate([frames, np.zeros((n, N - n0, frames.shape[2]))], axis=1)
    return pad_p @ Q.T, np.einsum('ab,nbd->nad', Q, pad_f)


# ============================================================
# Self-contained geodesic landmark cover (no dreimac, no reordering)
# ============================================================

def geodesic_landmark_cover(points, n_charts: int, percentile: float = 20,
                            n_neighbors: int = 100, seed: int = 0):
    """FPS landmarks on the kNN geodesic graph; charts = geodesic balls."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra

    graph = kneighbors_graph(points, n_neighbors, mode='distance')
    graph = 0.5 * (graph + graph.T)  # symmetrize

    rng = np.random.default_rng(seed)
    landmarks = [int(rng.integers(len(points)))]
    dist_to_set = dijkstra(graph, indices=landmarks[0])
    for _ in range(n_charts - 1):
        landmarks.append(int(np.argmax(dist_to_set)))
        dist_to_set = np.minimum(dist_to_set, dijkstra(graph, indices=landmarks[-1]))

    dist_mat = dijkstra(graph, indices=landmarks)          # (n_charts, n)
    finite = dist_mat[np.isfinite(dist_mat)]
    eps = np.percentile(finite, percentile)
    return [np.nonzero(dist_mat[i] < eps)[0] for i in range(n_charts)]


def knn_landmark_cover(points, n_charts: int, k_landmark: int = 3,
                       n_neighbors: int = 60, seed: int = 0, verbose: bool = False):
    """
    Density-robust cover: FPS landmarks, then each point joins the charts of
    its `k_landmark` nearest landmarks (geodesic). Chart sizes are balanced by
    construction (~k_landmark * n / n_charts each) regardless of how uneven
    the sampling density is, and adjacent charts always overlap (shared
    boundary points), so the sign cocycle is well posed. This fixes the
    oversized-chart pathology that the fixed-radius percentile cover suffers
    on real (non-uniform) data.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra

    n = len(points)
    g = kneighbors_graph(points, min(n_neighbors, n - 1), mode='distance')
    g = 0.5 * (g + g.T)
    rng = np.random.default_rng(seed)
    L = [int(rng.integers(n))]
    dts = dijkstra(g, indices=L[0])
    for _ in range(n_charts - 1):
        L.append(int(np.argmax(dts)))
        dts = np.minimum(dts, dijkstra(g, indices=L[-1]))
    D = dijkstra(g, indices=L)                 # (n_charts, n) geodesic dists
    # each point -> its k nearest landmarks
    nearest_k = np.argsort(D, axis=0)[:k_landmark]   # (k, n)
    charts = [np.nonzero((nearest_k == i).any(axis=0))[0] for i in range(n_charts)]
    charts = [c for c in charts if len(c) >= 20]
    if verbose:
        s = sorted([len(c) for c in charts], reverse=True)
        print(f"[knn cover] {len(charts)} charts, sizes {s}")
    return charts


def balanced_geodesic_cover(points, target_chart_size: int = 250,
                            overlap_frac: float = 0.35, max_size_ratio: float = 1.6,
                            n_neighbors: int = 100, seed: int = 0,
                            verbose: bool = False):
    """
    Geodesic-landmark cover that (a) targets a chart *size* rather than a
    fixed radius, and (b) enforces an overlap floor between adjacent charts.

    Rationale (see the eta-vs-overlap tension): eta is small when each chart
    is small enough to be flat, but the sign cocycle needs the overlaps to
    stay large. So we pick the number of charts from the target size and set
    the radius to a percentile that yields ~target_chart_size points, then
    split any chart exceeding max_size_ratio * target by adding a landmark,
    and grow the radius if adjacent charts fail the overlap floor.

    Returns a list of index arrays. Falls back gracefully on small clouds.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra

    n = len(points)
    n_charts = max(4, int(round(n / target_chart_size)))
    graph = kneighbors_graph(points, min(n_neighbors, n - 1), mode='distance')
    graph = 0.5 * (graph + graph.T)

    rng = np.random.default_rng(seed)
    landmarks = [int(rng.integers(n))]
    dist_to_set = dijkstra(graph, indices=landmarks[0])
    # over-sample landmarks so we can split crowded regions
    for _ in range(2 * n_charts - 1):
        landmarks.append(int(np.argmax(dist_to_set)))
        dist_to_set = np.minimum(dist_to_set, dijkstra(graph, indices=landmarks[-1]))
    dist_mat = dijkstra(graph, indices=landmarks)

    # radius = smallest percentile giving ~target_chart_size in a typical ball
    per_lm = np.sort(dist_mat, axis=1)
    radius = float(np.median(per_lm[:, min(target_chart_size, n - 1)]))

    def build(r):
        return [np.nonzero(dist_mat[i] < r)[0] for i in range(len(landmarks))]

    charts = build(radius)
    # grow radius until adjacent charts overlap enough (overlap floor)
    for _ in range(6):
        ok = True
        for i in range(len(charts)):
            si = set(charts[i].tolist())
            # nearest other landmark
            j = int(np.argsort([np.linalg.norm(points[landmarks[i]] -
                                               points[landmarks[k]])
                                if k != i else np.inf
                                for k in range(len(landmarks))])[0])
            inter = len(si & set(charts[j].tolist()))
            if inter < overlap_frac * min(len(charts[i]), len(charts[j])):
                ok = False; break
        if ok:
            break
        radius *= 1.15
        charts = build(radius)

    # drop empty/duplicate charts, keep those with enough points
    seen, out = set(), []
    for c in charts:
        if len(c) < 20:
            continue
        key = tuple(c[:10].tolist())
        if key in seen:
            continue
        seen.add(key); out.append(c)
    if verbose:
        sizes = sorted([len(c) for c in out], reverse=True)
        print(f"[balanced cover] {len(out)} charts, sizes {sizes}")
    return out


# ============================================================
# Single run
# ============================================================

MANIFOLDS = {
    'S2': dict(sampler=sample_S2, base_dim=3, orientable=True,
               n_points=1000, cover='tetrahedral', n_charts=4,
               eps_cluster=1.0, min_points=5),
    'Klein': dict(sampler=sample_klein, base_dim=4, orientable=False,
                  n_points=1000, cover='geodesic', n_charts=8,
                  eps_cluster=1.0, min_points=5,
                  # paper Klein config (Klein_bottle.py): without Jacobian
                  # regularisation the paper itself reports 0% convergence
                  lambda_jac=0.01, min_epochs=5000,
                  # differential reconstruction loss ON by default: the
                  # lambda_diff=0 arm (results_codim_Klein_20260714_230528)
                  # is the recorded baseline/ablation
                  lambda_diff=0.01,
                  # paper retry protocol: restart with fresh init until
                  # certified, up to max_retries attempts
                  recon_threshold=0.15, max_retries=3),
}


def run_one(manifold: str, N: int, seed: int, epochs: int,
            hidden_dims=(32, 16), lambda_jac: Optional[float] = None,
            pca_k: int = 25, verbose: bool = False):
    cfg = MANIFOLDS[manifold]
    if lambda_jac is None:
        lambda_jac = cfg.get('lambda_jac', 0.0)
    epochs = max(epochs, cfg.get('min_epochs', 0))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Sample intrinsically, build the cover BEFORE embedding
    # (assignments are index sets; an isometric embedding does not change them)
    points0, frames0 = cfg['sampler'](cfg['n_points'], seed)
    if cfg['cover'] == 'tetrahedral':
        assignments = tetrahedral_cover_S2(epsilon=0.3).get_assignments(points0)
    else:
        assignments = geodesic_landmark_cover(points0, cfg['n_charts'], seed=seed)

    points, frames = random_isometric_embedding(points0, frames0, N, seed + 10_000)

    # PCA tangent frames from the embedded cloud (real-data instrument)
    frames_pca = pca_tangent_frames(points, d=2, k=pca_k)
    align = frame_alignment(frames, frames_pca)

    # Certificate-aware training protocol.
    #
    # The binding constraints in practice are eta and delta, not epsilon:
    # runs routinely converge in eps while several charts keep eta > 1.
    # We therefore retry until the full (ground-truth-free) certificate
    # passes: eps <= threshold, at most one chart with eta_lat > 1
    # (re-indexing remark), delta > delta_min. Per attempt: main fit ->
    # optional +2000-epoch extension (paper protocol: extend promising
    # runs before discarding progress) -> low-LR polish phase.
    threshold = cfg.get('recon_threshold', np.inf)
    max_retries = cfg.get('max_retries', 0)
    delta_min = cfg.get('delta_min', 0.005)
    lambda_diff = cfg.get('lambda_diff', 0.0)
    extension_epochs = cfg.get('extension_epochs', 2000)
    polish_epochs = cfg.get('polish_epochs', 500)

    def sup_eps(sys_):
        return max(
            float(sys_.compute_varepsilon(
                i, tf.constant(points[assignments[i]], dtype=tf.float32)).numpy())
            for i in range(sys_.n_charts)
        )

    def certificate(sys_):
        """(eps_ok, cert_ok, eps, n_eta_outliers, delta) — no ground truth.

        Uses eta measured with PCA tangent frames (the paper's instrument):
        computable from the point cloud alone, and — unlike the latent
        proxy eta_lat — validated against ground-truth tangents. The
        eta_lat-based certificate produced a wrong certified verdict
        (Klein N=4 seed=43); the PCA certificate rejects that run.
        """
        eps_now = sup_eps(sys_)
        etas = [float(compute_eta_true_pointwise(
            sys_.autoencoders[i], points[assignments[i]],
            frames_pca[assignments[i]]).max())
            for i in range(sys_.n_charts)]
        n_out = sum(e > 1.0 for e in etas)
        delta_now = float(sys_.compute_delta().numpy())
        eps_ok = eps_now <= threshold
        cert_ok = eps_ok and n_out <= 1 and delta_now > delta_min
        return eps_ok, cert_ok, eps_now, n_out, delta_now

    from atlasae.fast_training import train_until_certified

    best = None  # (cert_ok, -n_out, -eps) ranking
    n_attempts = 0
    run_trajectory = []
    total_budget = epochs + extension_epochs  # same budget as the old protocol
    for attempt in range(max_retries + 1):
        n_attempts = attempt + 1
        tf.random.set_seed(seed + 1000 * attempt)
        system = AtlasAutoencoder(
            data=points,
            n_charts=len(assignments),
            subset_assignments=assignments,
            latent_dim=2,
            hidden_dims=list(hidden_dims),
        )
        out = train_until_certified(
            system,
            certificate=lambda s=system: certificate(s),
            sup_eps=lambda s=system: sup_eps(s),
            total_epochs=total_budget,
            scout_epochs=500,
            scout_bar=cfg.get('scout_bar', float('inf')),
            block=1000,
            polish_epochs=polish_epochs,
            batch_size=64,
            lambda_jac=lambda_jac,
            lambda_diff=lambda_diff,
            verbose=verbose,
        )
        if out['hopeless']:
            print(f"  attempt {attempt + 1}: abandoned at scout "
                  f"({out['epochs_used']} epochs) — restarting")
            if best is None:
                best = ((False, -99, -np.inf), system, np.inf, False, False)
            continue
        run_trajectory = out.get('trajectory', [])
        eps_ok, cert_ok, eps_now, n_out, delta_now = out['cert']

        rank = (cert_ok, -n_out, -eps_now)
        if best is None or rank > best[0]:
            best = (rank, system, eps_now, cert_ok, eps_ok)
        if cert_ok:
            print(f"  attempt {attempt + 1}: CERTIFIED after "
                  f"{out['epochs_used']} epochs")
            break
        if attempt < max_retries:
            print(f"  attempt {attempt + 1}: eps={eps_now:.3f} "
                  f"eta-outliers={n_out} delta={delta_now:.4f} — restarting")

    _, system, best_eps, certified, converged = best
    if not np.isfinite(best_eps):  # all attempts abandoned at scout
        best_eps = sup_eps(system)

    eta = compare_eta(system, points, assignments,
                      frames_true=frames, frames_pca=frames_pca)
    if verbose:
        print(summarize_comparison(eta))

    orient = check_orientability(system, points, assignments,
                                 eps_cluster=cfg['eps_cluster'],
                                 min_points=cfg['min_points'], verbose=False)

    eps_sup = best_eps

    # Exact hypothesis of the LOCAL stability theorem (logged, not gated):
    # every nerve triangle contains at most ONE chart with eta_pca > 1
    # (that chart goes in the encoder-only gamma-slot via re-indexing).
    # Strictly weaker than the global "<= 1 outlier chart overall" gate,
    # so it can legitimately certify runs the conservative gate rejects.
    outliers = {c['chart'] for c in eta['per_chart'] if c['eta_pca'] > 1.0}
    triangles = list({tuple(sorted(k)) for k in system.triple_overlaps})
    local_thm_ok = all(len(outliers & set(t)) <= 1 for t in triangles)
    certified_local = bool(converged and local_thm_ok
                           and float(system.compute_delta().numpy()) >
                           cfg.get('delta_min', 0.005))

    return {
        'manifold': manifold, 'N': N, 'seed': seed, 'epochs': epochs,
        'converged': bool(converged), 'certified': bool(certified),
        'certified_local': certified_local,
        'n_eta_outlier_charts': len(outliers),
        'n_triangles': len(triangles),
        'n_attempts': n_attempts, 'lambda_diff': lambda_diff,
        'trajectory': run_trajectory,
        'codimension': N - 2,
        'eta_true': eta['eta_true'],
        'eta_pca': eta['eta_pca'],
        'eta_lat': eta['eta_lat'],
        'eta_per_chart': eta['per_chart'],
        'pca_alignment_max_rad': float(align.max()),
        'pca_alignment_mean_rad': float(align.mean()),
        'varepsilon': eps_sup,
        # paper-table metrics (mean recon, encoder sigma_min, pairwise
        # compatibility/cocycle error, delta_mean)
        'varepsilon_mean': float(np.mean([
            float(system.compute_varepsilon_mean(
                i, tf.constant(points[assignments[i]], dtype=tf.float32)).numpy())
            for i in range(system.n_charts)])),
        'sigma_min_enc': float(min(
            float(system.compute_encoder_sigma_min(
                i, tf.constant(points[assignments[i]], dtype=tf.float32)).numpy())
            for i in range(system.n_charts))),
        'cocycle_error': float(system.compute_cocycle_error().numpy()),
        'delta_mean': float(system.compute_delta_mean().numpy()),
        'delta': float(system.compute_delta().numpy()),
        # is_orientable is True / False / None (None = cocycle not verified,
        # verdict undetermined — NOT the same as a wrong verdict)
        'verdict': ('undetermined' if orient['is_orientable'] is None
                    else ('orientable' if orient['is_orientable']
                          else 'non-orientable')),
        'cocycle_verified': bool(orient['cocycle_verified']),
        'detected_orientable': orient['is_orientable'],
        'correct': (None if orient['is_orientable'] is None
                    else bool(orient['is_orientable'] == cfg['orientable'])),
        # pointwise arrays for the eta_true-vs-eta_lat scatter (subsampled)
        'pointwise': [
            {'chart': pw['chart'],
             'eta_true': pw['eta_true'][::5],
             'eta_lat': pw['eta_lat'][::5],
             'eta_pca': pw['eta_pca'][::5]}
            for pw in eta['pointwise']
        ],
    }


# ============================================================
# Sweep driver
# ============================================================

def run_sweep(manifold: str, dims, seeds: int, epochs: int, out_dir: str,
              seed_base: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for N in dims:
        for s in range(seeds):
            seed = seed_base + s
            print(f"\n=== {manifold}  N={N}  seed={seed} ===")
            r = run_one(manifold, N, seed, epochs)
            print(f"  eta_true={r['eta_true']:.3f}  eta_pca={r['eta_pca']:.3f}  "
                  f"eta_lat={r['eta_lat']:.3f}  delta={r['delta']:.4f}  "
                  f"verdict={r['verdict']}  correct={r['correct']}")
            results.append(r)
            with open(os.path.join(out_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=1)
    print(f"\nSaved {len(results)} runs to {out_dir}/results.json")
    return results


# ============================================================
# Plots
# ============================================================

def make_plots(results_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        results = json.load(f)
    out_dir = os.path.dirname(results_path) or '.'
    manifold = results[0]['manifold']
    dims = sorted({r['N'] for r in results})

    # --- eta vs ambient dimension ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, style in [('eta_true', 'o-'), ('eta_pca', 's--'), ('eta_lat', '^:')]:
        mean = [np.mean([r[key] for r in results if r['N'] == N]) for N in dims]
        std = [np.std([r[key] for r in results if r['N'] == N]) for N in dims]
        ax.errorbar(dims, mean, yerr=std, fmt=style, capsize=3,
                    label={'eta_true': r'$\eta$ (true, GT tangents)',
                           'eta_pca': r'$\eta$ (PCA tangents)',
                           'eta_lat': r'$\eta_{\mathrm{lat}}$ (latent proxy)'}[key])
    ax.axhline(1.0, color='gray', lw=0.8, ls='-')
    ax.text(dims[-1], 1.02, r'$\eta=1$ (theorem threshold)',
            ha='right', va='bottom', fontsize=8, color='gray')
    ax.set_xlabel(r'ambient dimension $N$')
    ax.set_ylabel(r'differential error (sup over charts)')
    ax.set_yscale('log')
    ax.set_title(f'{manifold}: tangent-restricted error vs latent proxy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'eta_vs_codim_{manifold}.png'), dpi=200)

    # --- pointwise scatter eta_true vs eta_lat, colored by N ---
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap('viridis')
    for k, N in enumerate(dims):
        xs, ys = [], []
        for r in results:
            if r['N'] != N:
                continue
            for pw in r['pointwise']:
                xs.extend(pw['eta_lat'])
                ys.extend(pw['eta_true'])
        ax.scatter(xs, ys, s=3, alpha=0.25, color=cmap(k / max(len(dims) - 1, 1)),
                   label=f'N={N}')
    lim = [1e-3, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, 'k-', lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\eta_{\mathrm{lat}}(x)$ (proxy)')
    ax.set_ylabel(r'$\eta(x)$ (true, tangent-restricted)')
    ax.set_title(f'{manifold}: pointwise proxy vs true error')
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'eta_scatter_{manifold}.png'), dpi=200)

    print(f"Plots written to {out_dir}/")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifold', choices=list(MANIFOLDS), default='S2')
    p.add_argument('--dims', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--plot', type=str, default=None,
                   help='path to results.json — make plots and exit')
    p.add_argument('--lambda-diff', type=float, default=None,
                   help='weight of differential reconstruction loss '
                        '||d(E∘D)-I||² (ablation; try 0.01)')
    args = p.parse_args()

    if args.lambda_diff is not None:
        MANIFOLDS[args.manifold]['lambda_diff'] = args.lambda_diff

    if args.plot:
        make_plots(args.plot)
    else:
        dims = args.dims or ([3, 10, 25, 50, 100] if args.manifold == 'S2'
                             else [4, 10, 25, 50, 100])
        out = args.out or (
            f"results_codim_{args.manifold}_{datetime.now():%Y%m%d_%H%M%S}")
        results = run_sweep(args.manifold, dims, args.seeds, args.epochs, out)
        make_plots(os.path.join(out, 'results.json'))
