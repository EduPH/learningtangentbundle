"""
Figure: a transition map on an orientation-PRESERVING vs an
orientation-REVERSING overlap of the learned cyclo-octane atlas.

Both panels show the same set of overlap points twice: their latent
coordinates in chart i (left of each panel, blue->yellow colour scale by
angle) and their images under T_ji = E_j o D_i in chart j. Where
det g_ji > 0 the colour cycle keeps its handedness; where det g_ji < 0 it
is mirrored -- that flip is the sign the cocycle records.
"""
import os, sys
import numpy as np, tensorflow as tf, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cyclooctane as co
from atlasae import AtlasAutoencoder, fast_fit
from codim_sweep import geodesic_landmark_cover


def build(epochs=450, seed=42):
    X = co.load_cyclooctane('pointsCycloOctane.mat')
    Xr, _ = co.remove_singularities(X, dilate=0.3)
    st = co.stratify(Xr, min_cluster_size=150)
    P = Xr[st[0]]                      # Klein stratum
    P = P - P.mean(0); P = P / np.sqrt((P**2).sum(1).mean())
    np.random.seed(seed); tf.random.set_seed(seed)
    n_charts = max(8, round(len(P)/180))
    A = geodesic_landmark_cover(P, n_charts, percentile=8, seed=seed)
    s = AtlasAutoencoder(data=P, n_charts=len(A), subset_assignments=A,
                         latent_dim=2, hidden_dims=[32, 16])
    fast_fit(s, epochs=epochs, lambda_jac=0.01, lambda_diff=0.01)
    return s, P, A


def det_sign(system, P, idx, i, j):
    z = tf.constant(P[idx], dtype=tf.float32)
    zi = system.autoencoders[i].encode(z)
    with tf.GradientTape() as t:
        t.watch(zi)
        zj = system.autoencoders[j].encode(system.autoencoders[i].decode(zi))
    J = t.batch_jacobian(zj, zi)
    d = tf.linalg.det(J).numpy()
    return d, zi.numpy(), zj.numpy()


def _loop_in_overlap(zi, n=80, shrink=0.55):
    """A small oriented circle inside the overlap's latent footprint."""
    c = zi.mean(0); r = shrink * np.median(np.linalg.norm(zi - c, axis=1))
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return c + r * np.stack([np.cos(t), np.sin(t)], 1), t


def _push(system, i, j, z):
    z = tf.constant(z, dtype=tf.float32)
    return system.autoencoders[j].encode(system.autoencoders[i].decode(z)).numpy()


def _draw(ax, pts, t, title):
    ax.scatter(pts[:, 0], pts[:, 1], c=t, cmap='twilight', s=10, zorder=2)
    ax.plot(np.r_[pts[:, 0], pts[0, 0]], np.r_[pts[:, 1], pts[0, 1]],
            color='0.6', lw=0.8, zorder=1)
    k = len(pts)//12
    for m in (0, k):
        ax.annotate('', xy=pts[m+1], xytext=pts[m],
                    arrowprops=dict(arrowstyle='-|>', lw=1.6, color='k'))
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=55, facecolor='none',
               edgecolor='k', lw=1.2, zorder=3)
    ax.set_title(title, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect(1)


def main(out='cyclo_fig/transitions.png', epochs=450):
    """Pick ONE chart that has both a preserving and a reversing neighbour, so a
    single source loop can be pushed through both transitions."""
    s, P, A = build(epochs=epochs)
    by_i = {}
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j: continue
            ov = np.intersect1d(A[i], A[j])
            if len(ov) < 40: continue
            d, zi, _ = det_sign(s, P, ov, i, j)
            frac = (d > 0).mean()
            sgn = '+' if frac > 0.98 else ('-' if frac < 0.02 else None)
            if sgn: by_i.setdefault(i, {'zi': zi, '+': [], '-': []})[sgn].append((j, len(ov)))
    best = None
    for i, rec in by_i.items():
        if rec['+'] and rec['-']:
            score = max(n for _, n in rec['+']) + max(n for _, n in rec['-'])
            if best is None or score > best[0]:
                best = (score, i, max(rec['+'], key=lambda t: t[1])[0],
                        max(rec['-'], key=lambda t: t[1])[0], rec['zi'])
    if best is None:
        print('no chart has both signs'); return
    _, i, jp, jn, zi = best
    loop, t = _loop_in_overlap(zi)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    _draw(axes[0], loop, t, f'oriented loop in chart {i}')
    _draw(axes[1], _push(s, i, jp, loop), t,
          f'$T_{{{jp}{i}}}$:  $\\det g_{{{jp}{i}}}>0$,  preserved')
    _draw(axes[2], _push(s, i, jn, loop), t,
          f'$T_{{{jn}{i}}}$:  $\\det g_{{{jn}{i}}}<0$,  reversed')
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'saved {out}  (chart {i} -> {jp} preserving, {i} -> {jn} reversing)')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=450)
    p.add_argument('--out', default='cyclo_fig/transitions.png')
    a = p.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    main(a.out, a.epochs)
