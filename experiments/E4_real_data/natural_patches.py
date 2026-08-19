"""
E4 — Natural image patches (Klein bottle) : real vision/neuroscience data.

High-contrast 3x3 patches from natural photographs, after contrast
normalisation and density filtering, concentrate on a 2-manifold with the
topology of a KLEIN BOTTLE (non-orientable) --- the classic result of
Carlsson, Ishkhanov, de Silva & Zomorodian (2008), building on Lee,
Pedersen & Mumford (2003). This is a second real-data, non-trivial-
characteristic-class experiment, complementing cyclo-octane (chemistry)
with a vision example, and complementing the synthetic RP2 line patches
with genuine photographs.

Pipeline (Carlsson et al.):
  1. sample 3x3 patches from natural images (any grayscale photos).
  2. keep the top-contrast patches by the D-norm (neighbour-difference norm).
  3. mean-centre and D-normalise -> points on a 7-sphere in the DCT basis.
  4. density-filter to the densest core (k-NN density) -> Klein bottle.
  5. atlas autoencoder per the certified pipeline -> expect w1 != 0.

Usage:
  python natural_patches.py --images /path/to/photos   # any .png/.jpg/.tif
  python natural_patches.py --synthetic                # Klein-patch stand-in
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlasae import (AtlasAutoencoder, fast_fit, check_orientability,
                     pca_tangent_frames, compare_eta)
from codim_sweep import geodesic_landmark_cover, knn_landmark_cover

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# D-norm and DCT basis for 3x3 patches (Carlsson et al.)
# ============================================================

def dnorm_matrix():
    """
    8-neighbour difference operator D on 3x3 patches (9-dim), as a 9x9 PSD
    matrix: x^T D x = sum over adjacent pixel pairs (x_i - x_j)^2. Constant
    patches are in its kernel, so after mean-removal D is positive-definite.
    """
    idx = np.arange(9).reshape(3, 3)
    D = np.zeros((9, 9))
    for a in range(3):
        for b in range(3):
            for da, db in [(0, 1), (1, 0)]:  # right and down neighbours
                a2, b2 = a + da, b + db
                if a2 < 3 and b2 < 3:
                    i, j = idx[a, b], idx[a2, b2]
                    D[i, i] += 1; D[j, j] += 1
                    D[i, j] -= 1; D[j, i] -= 1
    return D


def dct_sphere_basis():
    """
    Orthonormal basis (w.r.t. the D-norm) of the 8-dim mean-zero patch space,
    so that D-normalised patches map to the unit sphere S^7 in R^8.
    """
    D = dnorm_matrix()
    # mean-zero subspace: project out the constant vector
    ones = np.ones(9) / 3.0
    P = np.eye(9) - np.outer(ones, ones)
    Dp = P.T @ D @ P
    w, V = np.linalg.eigh(Dp)
    keep = w > 1e-8              # drop the constant direction
    V = V[:, keep]; w = w[keep]
    # whiten so that x^T D x = ||coords||^2
    B = V * (1.0 / np.sqrt(w))  # 9 x 8 : patch -> (coords = B^T D patch)
    return D, B


# ============================================================
# Patch extraction + normalisation + density core
# ============================================================

def read_iml(path, shape=(1024, 1536)):
    """
    Read a raw van Hateren .iml image: 16-bit big-endian unsigned ints,
    no header, 1024 x 1536 (the linear-intensity 'iml' format). '.imc'
    (deblurred) files share the same layout.
    """
    a = np.fromfile(path, dtype='>u2')
    if a.size != shape[0] * shape[1]:
        # some files are transposed; infer if it matches the other orientation
        if a.size == shape[1] * shape[0]:
            shape = shape[::-1]
        else:
            raise ValueError(f"{path}: unexpected size {a.size}")
    return a.reshape(shape).astype(float)


def load_images(folder, max_images=400, seed=0):
    """
    Load up to `max_images` images from a folder (a random subset if there
    are more — the full van Hateren set is thousands of 1.5 MP images and
    would exhaust memory). Supports standard formats and van Hateren
    .iml/.imc raw files.
    """
    from PIL import Image
    exts = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "pgm")
    std = sum([glob.glob(os.path.join(folder, f"*.{e}")) for e in exts], [])
    iml = (glob.glob(os.path.join(folder, "*.iml"))
           + glob.glob(os.path.join(folder, "*.imc")))
    # sort: glob order is filesystem-dependent, so without this the selection
    # below is not reproducible across machines or after the folder changes.
    files = sorted([("std", f) for f in std] + [("iml", f) for f in iml],
                   key=lambda kf: os.path.basename(kf[1]))
    if not files:
        raise FileNotFoundError(
            f"no images in {folder} (looked for {exts} and .iml/.imc)")
    n_available = len(files)
    if n_available > max_images:
        # take the first max_images by filename, not a random subset: rng.choice
        # depends on n_available, so adding images to the folder silently
        # changed which ones were used and moved the reported errors.  Taking a
        # sorted prefix is stable when the folder later grows.
        files = files[:max_images]
    load_images.last_selection = dict(
        n_available=n_available, n_used=len(files),
        first=os.path.basename(files[0][1]),
        last=os.path.basename(files[-1][1]))
    imgs, n_std, n_iml = [], 0, 0
    for kind, f in files:
        im = (np.asarray(Image.open(f).convert("L"), dtype=float) if kind == "std"
              else read_iml(f))
        imgs.append(np.log(im + 1.0))          # log intensity (linear images)
        n_std += kind == "std"; n_iml += kind == "iml"
    print(f"[patches] loaded {len(imgs)} images ({n_std} standard, "
          f"{n_iml} van Hateren .iml/.imc)")
    return imgs


def sample_patches(imgs, n_patches=200000, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty((n_patches, 9))
    per = n_patches // len(imgs) + 1
    k = 0
    for im in imgs:
        H, W = im.shape
        if H < 3 or W < 3:
            continue
        ys = rng.integers(0, H - 2, per); xs = rng.integers(0, W - 2, per)
        for y, x in zip(ys, xs):
            if k >= n_patches:
                break
            out[k] = im[y:y+3, x:x+3].ravel(); k += 1
    return out[:k]


def to_klein_core(patches, D, B, contrast_frac=0.2, density_frac=0.2,
                  k_density=15):
    """Contrast filter -> mean-centre + D-normalise -> densest core."""
    # contrast = D-norm
    dn = np.sqrt(np.einsum('ni,ij,nj->n', patches, D, patches) + 1e-12)
    thr = np.quantile(dn, 1 - contrast_frac)
    hi = patches[dn >= thr]
    print(f"[patches] high-contrast: {len(hi)} / {len(patches)}")
    # mean-centre, D-normalise, map to S^7 in DCT basis
    hi = hi - hi.mean(axis=1, keepdims=True)
    coords = hi @ (D @ B)                      # n x 8
    coords /= (np.linalg.norm(coords, axis=1, keepdims=True) + 1e-12)
    # density core: small kth-NN distance = dense
    from scipy.spatial import cKDTree
    d, _ = cKDTree(coords).query(coords, k=k_density + 1)
    kth = d[:, -1]
    keep = kth <= np.quantile(kth, density_frac)
    core = coords[keep]
    core_patches = hi[keep]              # raw mean-centred 3x3 patches (n x 9)
    print(f"[patches] dense core (Klein bottle): {len(core)} points in R^8")
    return core, core_patches


def make_synthetic(n=3000, seed=0):
    """
    Klein-bottle-of-patches model (Perea; Carlsson et al.): a patch is a
    linear+quadratic profile along direction theta, with the Klein
    identification (theta, phi) ~ (theta + pi, -phi). Produces a genuine
    Klein bottle in R^8 patch-coordinate space for pipeline validation.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, n); phi = rng.uniform(0, 2*np.pi, n)
    ax = np.array([-1., 0., 1.])
    gx, gy = np.meshgrid(ax, ax)
    D, B = dct_sphere_basis()
    P = np.empty((n, 9))
    for m in range(n):
        t = gx * np.cos(theta[m]) + gy * np.sin(theta[m])
        prof = np.cos(phi[m]) * t + np.sin(phi[m]) * (2*t**2 - 1)  # linear+quad
        P[m] = prof.ravel()
    P = P - P.mean(axis=1, keepdims=True)
    C = P @ (D @ B)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C, P


# ============================================================
# Plotting
# ============================================================

def plot_sample_patches(patches, out_path, n=64, title=None, seed=0):
    """Grid of random 3x3 patches (patches: n x 9, mean-centred)."""
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(patches), size=min(n, len(patches)), replace=False)
    side = int(np.ceil(np.sqrt(len(idx))))
    vmax = np.percentile(np.abs(patches[idx]), 99) + 1e-9
    fig, axes = plt.subplots(side, side, figsize=(side * 0.6, side * 0.6))
    for k, ax in enumerate(np.array(axes).ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if k < len(idx):
            ax.imshow(patches[idx[k]].reshape(3, 3), cmap='gray',
                      vmin=-vmax, vmax=vmax, interpolation='nearest')
        else:
            ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot] {out_path}")


def plot_chart_latents(system, points, assignments, out_path):
    """Grid of per-chart latent embeddings E_i(U_i) in R^2."""
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    nc = system.n_charts
    side = int(np.ceil(np.sqrt(nc)))
    fig, axes = plt.subplots(side, side, figsize=(side * 2, side * 2))
    for k, ax in enumerate(np.array(axes).ravel()):
        if k < nc:
            z = system.encode(points[assignments[k]], k)
            ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.5)
            ax.set_title(f"chart {k}  (n={len(assignments[k])})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis('off')
    fig.suptitle("Per-chart latent embeddings  $E_i(U_i)\\subset\\mathbb{R}^2$",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot] {out_path}")


def plot_nerve_signs(orient, out_path, title_prefix=""):
    """
    Nerve graph with overlap edges coloured by the sign of det g_ji
    (blue = orientation-preserving +1, red = reversing -1).

    We attempt a 2-colouring nu: charts -> {+-1} with omega_ji = nu_j nu_i by
    breadth-first propagation. If one exists the bundle is orientable and the
    nodes are filled with their nu value; if propagation meets a contradiction
    (an odd number of -1 edges around some cycle) no colouring exists and that
    obstruction is exactly w_1 != 0.
    """
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    signs = orient.get('signs', {})
    nodes, edges = set(), []
    for key, s in signs.items():
        ci, cj = (key[0], key[1]), (key[2], key[3])
        nodes.add(ci); nodes.add(cj); edges.append((ci, cj, s))
    nodes = sorted(nodes)
    if not nodes:
        print("[plot] no sign data for nerve figure"); return
    # --- attempt 2-colouring ---
    adj = {n: [] for n in nodes}
    for ci, cj, s in edges:
        adj[ci].append((cj, s)); adj[cj].append((ci, s))
    nu, consistent = {}, True
    for root in nodes:
        if root in nu:
            continue
        nu[root] = +1; stack = [root]
        while stack:
            u = stack.pop()
            for v, s in adj[u]:
                want = nu[u] * s
                if v not in nu:
                    nu[v] = want; stack.append(v)
                elif nu[v] != want:
                    consistent = False
    ang = {n: 2*np.pi*i/len(nodes) for i, n in enumerate(nodes)}
    pos = {n: (np.cos(a), np.sin(a)) for n, a in ang.items()}
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for ci, cj, s in edges:
        ax.plot([pos[ci][0], pos[cj][0]], [pos[ci][1], pos[cj][1]],
                color=('tab:blue' if s > 0 else 'tab:red'), lw=1.8,
                alpha=0.85, zorder=1)
    for n, (x, y) in pos.items():
        face = 'white'
        if consistent:
            face = '#c6dbef' if nu.get(n, 1) > 0 else '#fdd0a2'
        ax.scatter([x], [y], s=300, color=face, edgecolor='k', zorder=2)
        lab = f"{n[0]}" + (f"$^{{{'+' if nu.get(n,1)>0 else '-'}}}$" if consistent else "")
        ax.text(x, y, lab, ha='center', va='center', fontsize=8, zorder=3)
    n_rev = sum(1 for _, _, s in edges if s < 0)
    if consistent:
        verdict = ("a consistent $2$-colouring exists $\\Rightarrow w_1=0$ "
                   "(orientable)")
    else:
        verdict = ("no consistent $2$-colouring $\\Rightarrow w_1\\neq0$ "
                   "(non-orientable)")
    ax.set_title(f"{title_prefix}{n_rev} of {len(edges)} overlaps orientation-reversing\n"
                 + verdict, fontsize=9)
    ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout(); fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] {out_path} (colouring consistent: {consistent})")


# ============================================================
# Atlas orientability on the core
# ============================================================

def analyse(core, n_charts=None, epochs=4000, lambda_jac=0.01, atlas_dir=None,
            lambda_diff=0.01, pca_k=25, seed=42, out_dir=None,
            min_points=5, verbose=False):
    np.random.seed(seed); tf.random.set_seed(seed)
    points = core - core.mean(0)
    s = float(np.sqrt((points**2).sum(1).mean()))
    if s > 0:
        points = points / s
    # density-robust balanced cover (real cores have very uneven density; the
    # fixed-radius percentile cover produced 800-point charts that could not
    # be flattened, inflating eta and breaking cocycle verification). Each
    # point joins its 3 nearest landmarks, so with n_charts = 3*n/target the
    # mean chart size is ~target regardless of density.
    k_landmark = 3
    target = 180
    if n_charts is None:
        n_charts = max(12, round(k_landmark * len(points) / target))
    assignments = knn_landmark_cover(points, n_charts, k_landmark=k_landmark,
                                     seed=seed, verbose=True)
    system = AtlasAutoencoder(data=points, n_charts=len(assignments),
                              subset_assignments=assignments,
                              latent_dim=2, hidden_dims=[32, 16])
    fast_fit(system, epochs=epochs, lambda_jac=lambda_jac,
             lambda_diff=lambda_diff, verbose=verbose)
    frames = pca_tangent_frames(points, d=2, k=pca_k)
    eta = compare_eta(system, points, assignments, frames_pca=frames)
    orient = check_orientability(system, points, assignments,
                                 eps_cluster=1.0, min_points=min_points,
                                 verbose=False)
    eps = max(float(system.compute_varepsilon(
        i, tf.constant(points[assignments[i]], dtype=tf.float32)).numpy())
        for i in range(system.n_charts))
    n_out = sum(c['eta_pca'] > 1 for c in eta['per_chart'])
    delta = float(system.compute_delta().numpy())
    certified = bool(eps <= 0.15 and n_out <= 1 and delta > 0.005)
    verdict = ('undetermined' if orient['is_orientable'] is None
               else ('orientable' if orient['is_orientable'] else 'non-orientable'))
    if out_dir is not None:
        plot_chart_latents(system, points, assignments,
                           os.path.join(out_dir, "chart_latents.png"))
        plot_nerve_signs(orient, os.path.join(out_dir, "nerve_signs.png"))
    print(f"[patches] n={len(points)} charts={system.n_charts} eps={eps:.3f} "
          f"eta_pca={eta['eta_pca']:.2f} (#>1={n_out}) delta={delta:.4f} "
          f"certified={certified} verdict={verdict}")
    if atlas_dir is not None:
        from atlasae import save_atlas
        import os as _o
        save_atlas(system, _o.path.join(atlas_dir, f'seed{seed}'),
                   note=f'patches seed {seed}', extra=dict(seed=int(seed)))
    return dict(n_points=int(len(points)), n_charts=int(system.n_charts),
                min_points=int(min_points),
                varepsilon=eps, eta_pca=eta['eta_pca'], n_eta_outliers=n_out,
                delta=delta, certified=certified, verdict=verdict,
                cocycle_verified=bool(orient['cocycle_verified']),
                eta_per_chart=eta['per_chart'])


def run(images=None, synthetic=False, out_dir=None, max_images=400,
        n_patches=200000, core_size=3500, density_frac=0.2, seeds=1, **kw):
    """
    Extract the dense core ONCE (the expensive image/patch step), then run
    `seeds` independent trials that each subsample the core, build a cover,
    and train with a different seed. Saves per-seed results + aggregate.
    """
    D, B = dct_sphere_basis()
    if synthetic:
        full_core, full_patches = make_synthetic()
        print(f"[patches] synthetic Klein core: {full_core.shape}")
    else:
        imgs = load_images(images, max_images=max_images)
        patches = sample_patches(imgs, n_patches=n_patches)
        full_core, full_patches = to_klein_core(patches, D, B,
                                                density_frac=density_frac)
    out_dir = out_dir or os.path.join(
        DATA_DIR, f"results_patches_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    _atlas_dir = os.path.join(out_dir, 'atlases')
    os.makedirs(_atlas_dir, exist_ok=True)
    results = []
    for s in range(seeds):
        seed = 42 + s
        core, core_patches = full_core, full_patches
        if len(full_core) > core_size:      # different subsample per seed
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(full_core), core_size, replace=False)
            core, core_patches = full_core[idx], full_patches[idx]
        if s == 0:                          # figures from the first seed only
            plot_sample_patches(
                core_patches, os.path.join(out_dir, "core_patches.png"),
                title="Dense-core high-contrast patches"
                      + ("" if synthetic else " (natural images)"))
        r = analyse(core, out_dir=(out_dir if s == 0 else None), seed=seed,
                    atlas_dir=_atlas_dir, **kw)
        r['seed'] = seed
        r['synthetic'] = bool(synthetic)
        r['source'] = 'analytic patch model' if synthetic else 'van Hateren photographs'
        # provenance: the reported errors depend on which photographs were used,
        # so record the selection rather than leaving it implicit
        if not synthetic:
            r['image_selection'] = getattr(load_images, 'last_selection', None)
            r['n_patches_drawn'] = n_patches
            r['density_frac'] = density_frac
            r['core_size'] = core_size
        results.append(r)
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"),
                  indent=1)

    cert = sum(x['certified'] for x in results)
    nonor = sum(1 for x in results if x['verdict'] == 'non-orientable')
    eps = np.mean([x['varepsilon'] for x in results])
    print(f"\n=== {seeds} seeds: certified {cert}/{seeds}, "
          f"non-orientable {nonor}/{seeds}, mean eps {eps:.3f} ===")
    print(f"Saved to {out_dir}/results.json")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=None, help="folder of natural photos")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max-images", type=int, default=400,
                   help="cap on images loaded (random subset if more)")
    p.add_argument("--n-patches", type=int, default=200000)
    p.add_argument("--core-size", type=int, default=3500,
                   help="subsample the dense core to this many points")
    p.add_argument("--density-frac", type=float, default=0.2,
                   help="keep this densest fraction as the core (smaller = "
                        "tighter, cleaner Klein bottle, easier to certify)")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lambda-diff", type=float, default=0.01)
    p.add_argument("--seeds", type=int, default=1,
                   help="number of independent trials (different seeds)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not args.synthetic and not args.images:
        p.error("provide --images FOLDER or --synthetic")
    run(images=args.images, synthetic=args.synthetic, out_dir=args.out,
        epochs=args.epochs, lambda_diff=args.lambda_diff,
        max_images=args.max_images, n_patches=args.n_patches,
        core_size=args.core_size, density_frac=args.density_frac,
        seeds=args.seeds)
