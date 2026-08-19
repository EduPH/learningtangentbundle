"""
Paper tables driver — runs S2, Mobius, Klein, RP2 at their native settings
(the per-manifold experiments of Section 7), with the final protocol:
fast_fit + certificate + eta measured directly (analytic + local-PCA frames).

Produces, per manifold: eps, eps_mean, eta_pca, eta_true (where analytic
frames exist), delta, delta_mean, compatibility error, sigma_min(dE),
certificate, orientability verdict; aggregated as mean +/- std over seeds.
Also writes LaTeX rows.

Usage:
    python paper_experiments.py --manifold S2 --seeds 5
    python paper_experiments.py --manifold Mobius --seeds 5
    python paper_experiments.py --manifold Klein --seeds 5
    python paper_experiments.py --manifold RP2 --seeds 5        # needs [tda]
    python paper_experiments.py --all --seeds 5
    python paper_experiments.py --latex <results_dir>           # emit table rows
"""

import argparse
import json
import os
import os as _os
from datetime import datetime

import numpy as np
import tensorflow as tf

from atlasae import (AtlasAutoencoder, fast_fit, check_orientability,
                     pca_tangent_frames, compare_eta, tetrahedral_cover_S2,
                     sphere_tangent_frames, mobius_param, mobius_tangent_frames,
                     klein_param, klein_tangent_frames, numeric_tangent_frames)
from codim_sweep import geodesic_landmark_cover

tf.get_logger().setLevel('ERROR')


# ============================================================
# Per-manifold sampling + cover (native paper settings)
# ============================================================

def sample_S2(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p, sphere_tangent_frames(p)

def cover_S2(points, seed):
    return tetrahedral_cover_S2(epsilon=0.3).get_assignments(points)


def sample_Mobius(n, seed):
    rng = np.random.default_rng(seed)
    uv = np.stack([rng.uniform(0, 2*np.pi, n), rng.uniform(-1, 1, n)], axis=1)
    return mobius_param(uv), mobius_tangent_frames(uv)

def cover_Mobius(points, seed, threshold=0.3):
    # 2-chart cover by y-coordinate; overlap |y|<threshold has TWO components
    # (front/back of the band) — the mechanism that exposes non-orientability.
    y = points[:, 1]
    return [np.where(y > -threshold)[0], np.where(y < threshold)[0]]


def sample_Klein(n, seed, m=4.0, r=2.0):
    rng = np.random.default_rng(seed)
    uv = np.stack([rng.uniform(0, 2*np.pi, n), rng.uniform(0, 2*np.pi, n)], axis=1)
    return klein_param(uv, m, r), klein_tangent_frames(uv, m, r)

def cover_Klein(points, seed):
    # native paper setting: 8-chart geodesic cover at the 20th percentile.
    # (A smaller radius shrinks the overlaps that carry the Mobius-twist
    # signal and makes the coboundary test miss non-orientability.)
    return geodesic_landmark_cover(points, n_charts=8, percentile=20, seed=seed)


def sample_RP2(n, seed, patch_dim=10, sigma=0.25,
               sphere_normalize=False, pca_dim=None):
    """
    Line-patch images (RP2 topology) via DREiMaC's generator. No analytic
    tangent frames (the generator is not a clean parametrisation), so eta is
    reported with local-PCA frames only.

    sphere_normalize: mean-centre each patch and divide by its norm, placing
        the cloud on a unit sphere (the contrast normalisation that makes the
        natural-image-patch experiment well conditioned). Bounds the
        embedding and evens out density.
    pca_dim: after normalisation, project onto the top-`pca_dim` principal
        directions and re-normalise onto the sphere in that subspace. Drops
        the codimension from 98 to pca_dim-2 with negligible loss of the
        (2-dimensional) topology, moving RP2 into the low-codimension regime
        where certification is feasible.
    """
    from dreimac import GeometryExamples
    n_angles = int(np.sqrt(n)) + 1
    X = GeometryExamples.line_patches(dim=patch_dim, n_angles=n_angles,
                                      n_offsets=n_angles, sigma=sigma)
    X = np.asarray(X, dtype=float)
    if sphere_normalize:
        X = X - X.mean(axis=1, keepdims=True)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if pca_dim:
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ Vt[:pca_dim].T
        if sphere_normalize:
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, None

def cover_RP2(points, seed):
    # adaptive: ~200 pts/chart with a small radius. On the sphere-normalised,
    # PCA-reduced cloud this yields balanced charts (max ~2x mean); on the raw
    # R^100 cloud it cannot fully balance the uneven density (see discussion).
    n_charts = max(10, round(len(points) / 200))
    return geodesic_landmark_cover(points, n_charts=n_charts, percentile=12,
                                   seed=seed)


# ---- d = 3 -----------------------------------------------------------------
#
# Two synthetic 3-manifolds, mirroring the d=2 design: one non-orientable
# (Mobius band x S^1, "the Mobius band one dimension up") and one orientable
# control (the flat 3-torus). Analytic tangent frames exist for both, so they
# also extend the eta validation to d=3.
#
# Sign-constancy caveat: the margin scales as n^(1/d) (sec:exp-signconstancy),
# so d=3 needs a substantially denser sample for the same coverage. The
# non-orientable verdict is protected by the odd-cycle asymmetry (one certified
# cycle suffices); the orientable verdict on T3 needs every overlap component
# and may plausibly end correct-but-uncertified at these sample sizes.

# Three overlapping arcs covering S^1: centres 0, 120, 240 degrees, half-width
# 75 degrees. Pairwise intersections are single 30-degree arcs and there are
# no triple overlaps, so this is a good cover of the circle.
_ARC_CENTERS = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
_ARC_HALF = np.pi / 3 + np.pi / 12


def _arc_mask(theta, center, half=_ARC_HALF):
    d = np.angle(np.exp(1j * (theta - center)))
    return np.abs(d) < half


def sample_MobiusS1(n, seed):
    """Mobius band x S^1 in R^5 = R^3 x R^2: a non-orientable 3-manifold
    (w1 pulled back from the Mobius factor). Frames are block-diagonal:
    the analytic Mobius 2-frame plus the circle tangent, mutually orthogonal
    by construction."""
    rng = np.random.default_rng(seed)
    uv = np.stack([rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n)], axis=1)
    t = rng.uniform(0, 2 * np.pi, n)
    pts = np.concatenate(
        [mobius_param(uv), np.stack([np.cos(t), np.sin(t)], axis=1)], axis=1)
    f2 = mobius_tangent_frames(uv)              # (n, 3, 2), orthonormal
    frames = np.zeros((n, 5, 3))
    frames[:, :3, :2] = f2
    frames[:, 3, 2] = -np.sin(t)
    frames[:, 4, 2] = np.cos(t)
    return pts, frames


def cover_MobiusS1(points, seed, threshold=0.3):
    """Product cover: the paper's 2-chart Mobius cover (by y-coordinate;
    the |y|<threshold overlap has TWO components, front/back of the band)
    times 3 arcs on the S^1 factor -> 6 charts. Products of contractible
    sets stay contractible, so goodness is inherited from the factors."""
    y = points[:, 1]
    theta = np.arctan2(points[:, 4], points[:, 3])
    band = [y > -threshold, y < threshold]
    return [np.where(b & _arc_mask(theta, c))[0]
            for b in band for c in _ARC_CENTERS]


def sample_T3(n, seed):
    """Flat 3-torus in R^6, the product of three unit circles: the orientable
    d=3 control (the role S^2 plays at d=2)."""
    rng = np.random.default_rng(seed)
    abc = rng.uniform(0, 2 * np.pi, size=(n, 3))
    pts = np.empty((n, 6))
    frames = np.zeros((n, 6, 3))
    for k in range(3):
        pts[:, 2 * k] = np.cos(abc[:, k])
        pts[:, 2 * k + 1] = np.sin(abc[:, k])
        frames[:, 2 * k, k] = -np.sin(abc[:, k])
        frames[:, 2 * k + 1, k] = np.cos(abc[:, k])
    return pts, frames


def cover_T3(points, seed):
    """Product cover: 3 arcs per circle factor -> 27 box charts. Every
    pairwise chart intersection is a product of arcs (single component),
    and the arc system has no triple overlaps per factor."""
    thetas = [np.arctan2(points[:, 2 * k + 1], points[:, 2 * k])
              for k in range(3)]
    charts = []
    for c0 in _ARC_CENTERS:
        for c1 in _ARC_CENTERS:
            for c2 in _ARC_CENTERS:
                m = (_arc_mask(thetas[0], c0) & _arc_mask(thetas[1], c1)
                     & _arc_mask(thetas[2], c2))
                charts.append(np.where(m)[0])
    return charts


MANIFOLDS = {
    'S2':     dict(sampler=sample_S2, cover=cover_S2, n=1000, orientable=True,
                   epochs=4000, lambda_jac=0.0, has_analytic=True),
    'Mobius': dict(sampler=sample_Mobius, cover=cover_Mobius, n=1500,
                   orientable=False, epochs=4000, lambda_jac=0.0,
                   has_analytic=True),
    'Klein':  dict(sampler=sample_Klein, cover=cover_Klein, n=1000,
                   orientable=False, epochs=5000, lambda_jac=0.01,
                   lambda_diff=0.01, has_analytic=True, max_retries=3),
    'RP2':    dict(sampler=sample_RP2, cover=cover_RP2, n=5625,
                   orientable=False, epochs=5000, extension_epochs=2000,
                   lambda_jac=0.01, lambda_diff=0.01, has_analytic=False,
                   max_retries=3,
                   # contrast normalisation onto the sphere is the recipe that
                   # makes RP2 certifiable (raw R^100 line patches: 0/5); it is
                   # the default. Use --raw-patches for the raw baseline.
                   sampler_kw=dict(sphere_normalize=True)),
    # d = 3 (see the block comment above cover definitions). Sample sizes are
    # the d=3 analogue of the d=2 densities: the margin scales as n^(1/3).
    'MobiusS1': dict(sampler=sample_MobiusS1, cover=cover_MobiusS1, n=9000,
                     orientable=False, epochs=5000, lambda_jac=0.01,
                     lambda_diff=0.01, has_analytic=True, max_retries=3, d=3),
    'T3':       dict(sampler=sample_T3, cover=cover_T3, n=12000,
                     orientable=True, epochs=5000, lambda_jac=0.01,
                     lambda_diff=0.01, has_analytic=True, max_retries=3, d=3),
}


# ============================================================
# One run
# ============================================================

def run_one(name, seed, lambda_diff=None, pca_k=25, max_retries=None,
            diff_hinge=0.0, diff_warmup=0.0, balanced_cover=False,
            knn_cover=False, min_points=5, save_dir=None, epochs=None,
            n_points=None, hidden_dims=None, verbose=False):
    cfg = MANIFOLDS[name]
    np.random.seed(seed); tf.random.set_seed(seed)
    if lambda_diff is None:
        lambda_diff = cfg.get('lambda_diff', 0.0)
    if max_retries is None:
        max_retries = cfg.get('max_retries', 0)

    points, frames_true = cfg['sampler'](n_points or cfg['n'], seed,
                                        **cfg.get('sampler_kw', {}))
    if knn_cover:
        from codim_sweep import knn_landmark_cover
        nc = max(12, round(3 * len(points) / 180))
        assignments = knn_landmark_cover(points, nc, k_landmark=3, seed=seed)
    elif balanced_cover:
        from codim_sweep import balanced_geodesic_cover
        assignments = balanced_geodesic_cover(points, target_chart_size=200,
                                              seed=seed)
    else:
        assignments = cfg['cover'](points, seed)

    d_int = cfg.get('d', 2)   # intrinsic (= latent) dimension
    frames_pca_full = pca_tangent_frames(points, d=d_int, k=pca_k)

    def sup_eps(sys_):
        return max(float(sys_.compute_varepsilon(
            i, tf.constant(points[assignments[i]], dtype=tf.float32)).numpy())
            for i in range(sys_.n_charts))

    def certificate(sys_):
        """(eps_ok, cert_ok, eps, n_eta_outliers, delta) — no ground truth."""
        e = sup_eps(sys_)
        from atlasae import compute_eta_true_pointwise
        etas = [float(compute_eta_true_pointwise(
            sys_.autoencoders[i], points[assignments[i]],
            frames_pca_full[assignments[i]]).max())
            for i in range(sys_.n_charts)]
        n_out = sum(x > 1.0 for x in etas)
        d = float(sys_.compute_delta().numpy())
        eps_ok = e <= 0.15
        return eps_ok, (eps_ok and n_out <= 1 and d > 0.005), e, n_out, d

    # Proven protocol (train_until_certified): scout -> main blocks with
    # early stop on certificate -> polish; retry with fresh init.
    from atlasae import train_until_certified
    best, best_rank = None, None
    for attempt in range(max_retries + 1):
        tf.random.set_seed(seed + 1000 * attempt)
        system = AtlasAutoencoder(data=points, n_charts=len(assignments),
                                  subset_assignments=assignments,
                                  latent_dim=d_int,
                                  hidden_dims=list(hidden_dims or [32, 16]))
        out = train_until_certified(
            system, certificate=lambda s=system: certificate(s),
            sup_eps=lambda s=system: sup_eps(s),
            total_epochs=(epochs if epochs else
                          cfg['epochs'] + cfg.get('extension_epochs', 2000)),
            scout_epochs=(min(500, max(1, (epochs or 5000) // 4))),
            block=(min(1000, max(1, (epochs or 5000) // 2))),
            polish_epochs=(min(300, max(1, (epochs or 5000) // 8))),
            lambda_jac=cfg['lambda_jac'], lambda_diff=lambda_diff,
            diff_hinge=diff_hinge, diff_warmup=diff_warmup,
            verbose=verbose)
        eps_ok, cert_ok, e, n_out, d = certificate(system)
        rank = (cert_ok, -n_out, -e)
        if best is None or rank > best_rank:
            best, best_rank = system, rank
        if cert_ok:
            break
        if attempt < max_retries:
            print(f"    {name} seed={seed} attempt {attempt+1}: "
                  f"eps={e:.3f} outliers={n_out} delta={d:.4f} — retry")
    system = best

    frames_pca = frames_pca_full
    eta = compare_eta(system, points, assignments,
                      frames_true=frames_true if cfg['has_analytic'] else None,
                      frames_pca=frames_pca)

    tf32 = lambda idx: tf.constant(points[idx], dtype=tf.float32)
    eps = max(float(system.compute_varepsilon(i, tf32(assignments[i])).numpy())
              for i in range(system.n_charts))
    eps_mean = float(np.mean([
        float(system.compute_varepsilon_mean(i, tf32(assignments[i])).numpy())
        for i in range(system.n_charts)]))
    sigma_min = float(min(
        float(system.compute_encoder_sigma_min(i, tf32(assignments[i])).numpy())
        for i in range(system.n_charts)))
    delta = float(system.compute_delta().numpy())
    delta_mean = float(system.compute_delta_mean().numpy())
    cocycle_err = float(system.compute_cocycle_error().numpy())
    n_out = sum(c['eta_pca'] > 1 for c in eta['per_chart'])
    certified = bool(eps <= 0.15 and n_out <= 1 and delta > 0.005)

    orient = check_orientability(system, points, assignments,
                                 eps_cluster=1.0, min_points=min_points,
                                 verbose=False)
    verdict = ('undetermined' if orient['is_orientable'] is None
               else ('orientable' if orient['is_orientable'] else 'non-orientable'))
    correct = (None if orient['is_orientable'] is None
               else bool(orient['is_orientable'] == cfg['orientable']))

    skw = cfg.get('sampler_kw', {})
    r = dict(manifold=name, seed=seed, min_points=min_points,
             n_points=int(len(points)), d=d_int,
             sphere_normalize=bool(skw.get('sphere_normalize', False)),
             pca_dim=skw.get('pca_dim'),
             varepsilon=eps, varepsilon_mean=eps_mean,
             eta_pca=eta['eta_pca'], eta_true=eta.get('eta_true'),
             sigma_min_enc=sigma_min, delta=delta, delta_mean=delta_mean,
             hidden_dims=list(hidden_dims or [32, 16]),
             cocycle_error=cocycle_err, n_eta_outliers=n_out,
             certified=certified, verdict=verdict, correct=correct,
             n_charts=len(assignments))
    if save_dir is not None:
        # keep the trained atlas: every post-hoc diagnostic (Theta, gauge,
        # sign constancy) needs the networks, not just these scalars, and a
        # full set costs under 2 MB.
        from atlasae import save_atlas
        ap = _os.path.join(save_dir, f'{name}_seed{seed}')
        save_atlas(system, ap, note=f'{name} seed {seed}',
                   extra=dict(seed=int(seed), manifold=name,
                              min_points=int(min_points)))
        r['atlas_path'] = _os.path.relpath(ap, save_dir)
    print(f"  {name} seed={seed}: eps={eps:.3f} eta_pca={eta['eta_pca']:.2f} "
          f"delta={delta:.4f} cert={certified} verdict={verdict} correct={correct}")
    return r


# ============================================================
# Sweep + LaTeX
# ============================================================

def run_manifold(name, seeds, out_dir, seed_base=42, **run_kw):
    os.makedirs(out_dir, exist_ok=True)
    # RP2 is run in two embeddings; keep them in separate files
    tag = name
    if name == 'RP2' and not MANIFOLDS['RP2'].get('sampler_kw', {}).get(
            'sphere_normalize', True):
        tag = 'RP2_raw'
    # ablation runs with a non-default architecture get their own filename,
    # so the newest-file lookup of make_master_table.py (which backs the
    # paper tables and the audit) never mistakes them for the headline run
    hd = run_kw.get('hidden_dims')
    if hd and list(hd) != [32, 16]:
        tag += '_h' + 'x'.join(str(w) for w in hd)
    atlas_dir = os.path.join(out_dir, 'atlases')
    os.makedirs(atlas_dir, exist_ok=True)
    results = []
    for s in range(seeds):
        results.append(run_one(name, seed_base + s, save_dir=atlas_dir, **run_kw))
        with open(os.path.join(out_dir, f'{tag}.json'), 'w') as f:
            json.dump(results, f, indent=1)
    return results


def latex_rows(results_dir):
    def ms(vals):
        vals = [v for v in vals if v is not None]
        return (np.mean(vals), np.std(vals)) if vals else (float('nan'), 0.0)
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith('.json'):
            continue
        rs = json.load(open(os.path.join(results_dir, fn)))
        name = rs[0]['manifold']
        cert = sum(r['certified'] for r in rs)
        det = [r for r in rs if r['verdict'] != 'undetermined']
        acc = sum(1 for r in det if r['correct'])
        e, es = ms([r['varepsilon'] for r in rs])
        et, ets = ms([r['eta_pca'] for r in rs])
        d, ds = ms([r['delta'] for r in rs])
        print(f"{name:8s} & {e:.3f}$\\pm${es:.3f} & {et:.2f}$\\pm${ets:.2f} & "
              f"{d:.3f}$\\pm${ds:.3f} & {cert}/{len(rs)} & {acc}/{len(det)} \\\\")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--manifold', choices=list(MANIFOLDS))
    p.add_argument('--all', action='store_true')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--lambda-diff', type=float, default=None)
    p.add_argument('--diff-hinge', type=float, default=0.0,
                   help='hinge margin tau for the differential loss '
                        '(penalise only eta above tau; 0 = plain Frobenius)')
    p.add_argument('--diff-warmup', type=float, default=0.0,
                   help='fraction of the first block over which lambda_diff '
                        'ramps 0->full (0 = full weight immediately)')
    p.add_argument('--balanced-cover', action='store_true',
                   help='use the size-balanced cover with an overlap floor')
    p.add_argument('--knn-cover', action='store_true',
                   help='density-robust k-nearest-landmark cover (balanced '
                        'chart sizes; rescues oversized-chart failures)')
    p.add_argument('--raw-patches', action='store_true',
                   help='RP2: raw line patches, no contrast normalisation '
                        '(the R^100 baseline; normalisation is on by default)')
    p.add_argument('--pca-dim', type=int, default=None,
                   help='RP2: also project onto top-k PCA dims (optional; '
                        'normalisation alone already certifies)')
    p.add_argument('--n-points', type=int, default=None,
                   help='override the sample size (the sign-constancy margin \n                        scales as sqrt(n); see sec:exp-signconstancy)')
    p.add_argument('--epochs', type=int, default=None,
                   help='override the per-manifold training budget '
                        '(for smoke tests; the paper uses the defaults)')
    p.add_argument('--hidden', default=None,
                   help='override the encoder hidden widths, e.g. "64,32" '
                        '(decoder mirrored; the paper uses 32,16). For the '
                        'width ablation on raw RP2.')
    p.add_argument('--out', default=None)
    p.add_argument('--latex', default=None, help='results dir -> print LaTeX rows')
    args = p.parse_args()

    if args.latex:
        latex_rows(args.latex)
    else:
        # Runs land in experiments/results/, which is where make_master_table.py
        # and eta_validation.py look; superseded runs are archived to
        # experiments/_superseded/ and are not picked up by either.
        out = args.out or os.path.join(
            "results", f"results_paper_{datetime.now():%Y%m%d_%H%M%S}")
        MANIFOLDS['RP2']['sampler_kw'] = dict(
            sphere_normalize=not args.raw_patches, pca_dim=args.pca_dim)
        names = list(MANIFOLDS) if args.all else [args.manifold]
        for nm in names:
            run_manifold(nm, args.seeds, out, lambda_diff=args.lambda_diff,
                         diff_hinge=args.diff_hinge, diff_warmup=args.diff_warmup,
                         balanced_cover=args.balanced_cover,
                         knn_cover=args.knn_cover, epochs=args.epochs,
                         n_points=args.n_points,
                         hidden_dims=([int(w) for w in args.hidden.split(',')]
                                      if args.hidden else None))
        print(f"\nSaved to {out}/ — LaTeX rows:")
        latex_rows(out)
