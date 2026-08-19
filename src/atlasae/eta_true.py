"""
Direct measurement of the tangent-restricted differential error η (E1).

The stability theorems use
    η_i = sup_{x in U_i} || d(D_i ∘ E_i)_x |_{T_xM} - Id_{T_xM} ||_op,
while the paper's experiments so far report the latent-side proxy
    η_lat,i = sup_x || d(E_i ∘ D_i)_{E_i(x)} - I_d ||_op
(`AtlasAutoencoder.compute_eta_tangent`).

This module computes η directly:
  * with ground-truth tangent frames (analytic parametrizations), and
  * with tangent frames estimated by local PCA (usable on real data),
so the proxy claim can be validated empirically (reviewer criticism:
"not established by theorem or sufficient experimentation").

Key identity used: for an orthonormal frame T_x ∈ R^{N×d} of T_xM,
    || (J - I_N)|_{T_xM} ||_op = || (J - I_N) T_x ||_op = || J T_x - T_x ||_op,
and J T_x is computed column-by-column with forward-mode JVPs
(d forward passes instead of an N×N Jacobian).
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional


# ============================================================
# Ground-truth tangent frames for the paper's manifolds
# ============================================================

def orthonormalize(frames: np.ndarray) -> np.ndarray:
    """Orthonormalize (batch, N, d) frames column-wise via QR."""
    q, _ = np.linalg.qr(frames)
    return q


def sphere_tangent_frames(points: np.ndarray) -> np.ndarray:
    """T_x S² = x^⊥ ⊂ R³. points: (n, 3) unit vectors → (n, 3, 2)."""
    n = len(points)
    frames = np.zeros((n, 3, 2))
    for i, x in enumerate(points):
        # pick coordinate axis least aligned with x
        a = np.eye(3)[np.argmin(np.abs(x))]
        t1 = np.cross(x, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(x, t1)
        frames[i] = np.stack([t1, t2], axis=1)
    return frames


def mobius_param(uv: np.ndarray) -> np.ndarray:
    """Standard Möbius immersion R² ⊃ [0,2π)×[-1,1] → R³ (paper Sec 7.2)."""
    u, v = uv[:, 0], uv[:, 1]
    r = 1 + 0.5 * v * np.cos(0.5 * u)
    return np.stack([r * np.cos(u), r * np.sin(u), 0.5 * v * np.sin(0.5 * u)], axis=1)


def mobius_tangent_frames(uv: np.ndarray) -> np.ndarray:
    """Analytic ∂/∂u, ∂/∂v of the Möbius immersion, orthonormalized."""
    u, v = uv[:, 0], uv[:, 1]
    r = 1 + 0.5 * v * np.cos(0.5 * u)
    dr_du = -0.25 * v * np.sin(0.5 * u)
    du = np.stack([
        dr_du * np.cos(u) - r * np.sin(u),
        dr_du * np.sin(u) + r * np.cos(u),
        0.25 * v * np.cos(0.5 * u),
    ], axis=1)
    dv = np.stack([
        0.5 * np.cos(0.5 * u) * np.cos(u),
        0.5 * np.cos(0.5 * u) * np.sin(u),
        0.5 * np.sin(0.5 * u),
    ], axis=1)
    return orthonormalize(np.stack([du, dv], axis=2))


def klein_param(uv: np.ndarray, m: float = 4.0, r: float = 1.0) -> np.ndarray:
    """
    Klein bottle immersion in R⁴ with outer radius m and inner radius r.
    NOTE: the paper text (Sec 7.3.1) prints r=1, but the paper's code
    (Klein_bottle.py → dreimac klein_bottle_4d(n, 4, 2)) uses r=2.
    Pass r=2.0 to reproduce the paper's experiments.
    """
    u, v = uv[:, 0], uv[:, 1]
    return np.stack([
        (m + r * np.cos(v)) * np.cos(u),
        (m + r * np.cos(v)) * np.sin(u),
        r * np.sin(v) * np.cos(0.5 * u),
        r * np.sin(v) * np.sin(0.5 * u),
    ], axis=1)


def klein_tangent_frames(uv: np.ndarray, m: float = 4.0, r: float = 1.0) -> np.ndarray:
    """Analytic ∂/∂u, ∂/∂v of the Klein immersion, orthonormalized."""
    u, v = uv[:, 0], uv[:, 1]
    du = np.stack([
        -(m + r * np.cos(v)) * np.sin(u),
        (m + r * np.cos(v)) * np.cos(u),
        -0.5 * r * np.sin(v) * np.sin(0.5 * u),
        0.5 * r * np.sin(v) * np.cos(0.5 * u),
    ], axis=1)
    dv = np.stack([
        -r * np.sin(v) * np.cos(u),
        -r * np.sin(v) * np.sin(u),
        r * np.cos(v) * np.cos(0.5 * u),
        r * np.cos(v) * np.sin(0.5 * u),
    ], axis=1)
    return orthonormalize(np.stack([du, dv], axis=2))


def numeric_tangent_frames(param_fn, uv: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Central finite-difference tangent frames for any parametrization
    param_fn: (n, 2) → (n, N). Used e.g. for the RP² line-patch generator,
    where the analytic derivative of the blurred-line map is awkward.
    """
    cols = []
    for k in range(uv.shape[1]):
        e = np.zeros_like(uv)
        e[:, k] = h
        cols.append((param_fn(uv + e) - param_fn(uv - e)) / (2 * h))
    return orthonormalize(np.stack(cols, axis=2))


# ============================================================
# PCA tangent frames (no ground truth needed — for real data)
# ============================================================

def pca_tangent_frames(points: np.ndarray, d: int = 2, k: int = 25) -> np.ndarray:
    """
    Estimate T_xM at every point by local PCA on the k nearest neighbours.
    Returns (n, N, d) orthonormal frames. This is the practical instrument
    that makes the stability hypotheses checkable from samples alone.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    n, N = points.shape
    frames = np.zeros((n, N, d))
    for i in range(n):
        nbrs = points[idx[i]]
        nbrs = nbrs - nbrs.mean(axis=0)
        # top-d right singular vectors = principal directions
        _, _, vt = np.linalg.svd(nbrs, full_matrices=False)
        frames[i] = vt[:d].T
    return frames


def frame_alignment(frames_a: np.ndarray, frames_b: np.ndarray) -> np.ndarray:
    """
    Largest principal angle (in radians) between the subspaces spanned by
    two frame fields, per point. 0 = identical tangent planes.
    """
    n = len(frames_a)
    angles = np.zeros(n)
    for i in range(n):
        s = np.linalg.svd(frames_a[i].T @ frames_b[i], compute_uv=False)
        angles[i] = np.arccos(np.clip(s.min(), -1.0, 1.0))
    return angles


# ============================================================
# True η via forward-mode autodiff
# ============================================================

def compute_eta_true_pointwise(
    autoencoder,
    x: np.ndarray,
    frames: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Pointwise η(x) = || (d(D∘E)_x - I_N) T_x ||_op for one chart.

    Args:
        autoencoder: LocalAutoencoder with .encode/.decode
        x: (n, N) points in the chart domain
        frames: (n, N, d) orthonormal tangent frames at x
    Returns:
        (n,) array of pointwise tangent-restricted errors.
    """
    d = frames.shape[2]
    out = []
    for s in range(0, len(x), batch_size):
        xb = tf.constant(x[s:s + batch_size], dtype=tf.float32)
        fb = frames[s:s + batch_size]
        cols = []
        for kcol in range(d):
            tangent = tf.constant(fb[:, :, kcol], dtype=tf.float32)
            with tf.autodiff.ForwardAccumulator(primals=xb, tangents=tangent) as acc:
                y = autoencoder.decode(autoencoder.encode(xb))
            jvp = acc.jvp(y)              # J T_k, shape (b, N)
            cols.append(jvp - tangent)    # (J - I) T_k
        M = tf.stack(cols, axis=2)        # (b, N, d)
        svals = tf.linalg.svd(M, compute_uv=False)
        out.append(svals[:, 0].numpy())
    return np.concatenate(out)


def compute_eta_lat_pointwise(
    autoencoder,
    x: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Pointwise latent proxy η_lat(x) = || d(E∘D)_{E(x)} - I_d ||_op
    (pointwise version of AtlasAutoencoder.compute_eta_tangent, for
    scatter plots against the true η)."""
    out = []
    for s in range(0, len(x), batch_size):
        xb = tf.constant(x[s:s + batch_size], dtype=tf.float32)
        z = autoencoder.encode(xb)
        with tf.GradientTape() as tape:
            tape.watch(z)
            z_round = autoencoder.encode(autoencoder.decode(z))
        J = tape.batch_jacobian(z_round, z)
        I_d = tf.eye(J.shape[-1], batch_shape=[tf.shape(z)[0]])
        svals = tf.linalg.svd(J - I_d, compute_uv=False)
        out.append(svals[:, 0].numpy())
    return np.concatenate(out)


# ============================================================
# Atlas-level comparison
# ============================================================

def compare_eta(
    system,
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    frames_true: Optional[np.ndarray] = None,
    frames_pca: Optional[np.ndarray] = None,
) -> Dict:
    """
    For every chart of a trained AtlasAutoencoder, compute:
      eta_true      sup and pointwise, ground-truth tangent frames
      eta_pca       sup and pointwise, PCA-estimated frames (if given)
      eta_lat       sup and pointwise, latent proxy
    Returns a dict ready for json.dump (pointwise arrays as lists).
    """
    res = {'per_chart': [], 'pointwise': []}
    for i in range(system.n_charts):
        idx = subset_assignments[i]
        x = points[idx]
        chart = {'chart': i, 'n_points': len(idx)}
        pw = {'chart': i}

        eta_lat = compute_eta_lat_pointwise(system.autoencoders[i], x)
        chart['eta_lat'] = float(eta_lat.max())
        pw['eta_lat'] = eta_lat.tolist()

        if frames_true is not None:
            eta_t = compute_eta_true_pointwise(system.autoencoders[i], x, frames_true[idx])
            chart['eta_true'] = float(eta_t.max())
            pw['eta_true'] = eta_t.tolist()

        if frames_pca is not None:
            eta_p = compute_eta_true_pointwise(system.autoencoders[i], x, frames_pca[idx])
            chart['eta_pca'] = float(eta_p.max())
            pw['eta_pca'] = eta_p.tolist()

        res['per_chart'].append(chart)
        res['pointwise'].append(pw)

    # Atlas-level sups (the quantities entering the theorems)
    res['eta_lat'] = max(c['eta_lat'] for c in res['per_chart'])
    if frames_true is not None:
        res['eta_true'] = max(c['eta_true'] for c in res['per_chart'])
    if frames_pca is not None:
        res['eta_pca'] = max(c['eta_pca'] for c in res['per_chart'])
    return res


def summarize_comparison(res: Dict) -> str:
    """Human-readable per-chart table of eta_true / eta_pca / eta_lat."""
    lines = ["chart |     n | eta_true | eta_pca | eta_lat",
             "------+-------+----------+---------+--------"]
    for c in res['per_chart']:
        lines.append(
            f"{c['chart']:5d} | {c['n_points']:5d} | "
            f"{c.get('eta_true', float('nan')):8.4f} | "
            f"{c.get('eta_pca', float('nan')):7.4f} | "
            f"{c['eta_lat']:7.4f}"
        )
    tail = [f"atlas sup: eta_true={res.get('eta_true', float('nan')):.4f}  "
            f"eta_pca={res.get('eta_pca', float('nan')):.4f}  "
            f"eta_lat={res['eta_lat']:.4f}"]
    return "\n".join(lines + tail)
