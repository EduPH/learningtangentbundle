"""
Static atlas figures from saved atlases (pure numpy + matplotlib; no TF).

Two figure styles, sharing all computation with make_atlas_viz.py:

  --panel     data | latent charts + signed nerve | reconstruction
              (one per experiment; used in the supplement's per-experiment
              sections)
  --pipeline  pipeline-summary figure with the latent charts drawn as
              planes embedded in 3D between the data and its reconstruction
              (used in the main text; intended for the Mobius band atlas)

Usage:
    python make_atlas_figs.py <atlas_dir> --panel    --out fig.png
    python make_atlas_figs.py <atlas_dir> --pipeline --out fig.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from make_atlas_viz import (load_atlas, fwd, transition_signs, to3d, to2d,
                            PALETTE)

EDGE_COL = {1: "#1f77b4", -1: "#d62728", 0: "#e07b00"}


def compute(atlas):
    pts, asg, nets, meta = load_atlas(atlas)
    edges = transition_signs(pts, asg, nets,
                             min_points=meta.get("min_points", 5))
    P3, basis, mean = to3d(pts)
    lat, recon = [], np.full_like(pts, np.nan)
    for i, a in enumerate(asg):
        z = fwd(nets[i][0], pts[a])
        lat.append(z)
        recon[a] = fwd(nets[i][1], z)
    R3 = ((recon - mean) @ basis.T) if basis is not None else to3d(recon)[0]
    return pts, asg, meta, edges, P3, R3, lat


def _scatter3(ax, P3, asg, title, s=1.0):
    for i, a in enumerate(asg):
        ax.scatter(P3[a, 0], P3[a, 1], P3[a, 2], s=s,
                   c=PALETTE[i % len(PALETTE)], linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    lim = np.nanmax(np.abs(P3 - np.nanmean(P3, 0)))
    m = np.nanmean(P3, 0)
    ax.set_xlim(m[0] - lim, m[0] + lim)
    ax.set_ylim(m[1] - lim, m[1] + lim)
    ax.set_zlim(m[2] - lim, m[2] + lim)


def _edge_style(e):
    return (EDGE_COL[0], "--") if e["mixed"] else (EDGE_COL[e["sign"]], "-")


# ------------------------------------------------------------ panel figure

def fig_panel(atlas, out, elev=18, azim=-60):
    pts, asg, meta, edges, P3, R3, lat = compute(atlas)
    nc = meta["n_charts"]
    cols = int(np.ceil(np.sqrt(nc)))
    rows = int(np.ceil(nc / cols))

    fig = plt.figure(figsize=(11.5, 3.9))
    axd = fig.add_subplot(131, projection="3d")
    axm = fig.add_subplot(132)
    axr = fig.add_subplot(133, projection="3d")
    axd.view_init(elev, azim)
    axr.view_init(elev, azim)
    _scatter3(axd, P3, asg, "data")
    _scatter3(axr, R3, asg, "reconstruction")

    dense = nc > 12          # hairball guard: draw a sign matrix instead
    # atlas plane: cells with latents, nerve edges coloured by sign
    pad = 0.14
    cell = {i: (i % cols, rows - 1 - i // cols) for i in range(nc)}
    for i in range(nc):
        Z = to2d(lat[i])
        Z = (Z - Z.min(0)) / (Z.max(0) - Z.min(0) + 1e-12)
        cx, cy = cell[i]
        axm.scatter(cx + pad + (1 - 2 * pad) * Z[:, 0],
                    cy + pad + (1 - 2 * pad) * Z[:, 1],
                    s=1.2, c=PALETTE[i % len(PALETTE)], linewidths=0,
                    rasterized=True, zorder=2)
        axm.add_patch(plt.Rectangle((cx + .03, cy + .03), .94, .94, fill=False,
                                    ec="#bbbbbb", lw=.7, zorder=1))
        axm.text(cx + .08, cy + .84, str(i), fontsize=7, zorder=3,
                 color=PALETTE[i % len(PALETTE)], fontweight="bold")
    seen = {}
    for e in (() if dense else edges):
        c1, c2 = np.array(cell[e["i"]]) + .5, np.array(cell[e["j"]]) + .5
        k = seen.get((e["i"], e["j"]), 0)
        seen[(e["i"], e["j"])] = k + 1
        d = c2 - c1
        nrm = np.array([-d[1], d[0]])
        nrm /= np.linalg.norm(nrm) + 1e-12
        mid = (c1 + c2) / 2 + nrm * 0.13 * ((k + 1) // 2) * (-1) ** k
        colr, ls = _edge_style(e)
        t = np.linspace(0, 1, 24)[:, None]
        curve = ((1 - t) ** 2 * c1 + 2 * t * (1 - t) * mid + t ** 2 * c2)
        axm.plot(curve[:, 0], curve[:, 1], color=colr, ls=ls, lw=1.4,
                 alpha=.85, zorder=4)
    axm.set_xlim(-.1, cols + .1)
    axm.set_ylim(-.1, rows + .1)
    axm.set_aspect("equal")
    axm.set_axis_off()
    axm.set_title("latent charts" if dense
                  else "latent charts and signed nerve", fontsize=10)

    if dense:
        # nerve too dense to draw as edges: show the sign adjacency instead
        M = np.full((nc, nc), np.nan)
        for e in edges:
            v = 0.0 if e["mixed"] else float(e["sign"])
            for a, b in ((e["i"], e["j"]), (e["j"], e["i"])):
                M[a, b] = v if np.isnan(M[a, b]) or M[a, b] == v else 0.0
        axs = axm.inset_axes([1.06, 0.18, 0.52, 0.64])
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(["#d62728", "#e07b00", "#1f77b4"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
        axs.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm,
                   origin="upper", interpolation="nearest")
        axs.set_title("sign of $\\det dT_{ji}$ per pair", fontsize=8)
        axs.set_xticks([]); axs.set_yticks([])
        for sp in axs.spines.values():
            sp.set_color("#bbbbbb")

    fig.tight_layout()
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- pipeline figure

def fig_pipeline(atlas, out, elev=16, azim=-58):
    """Data -> charts as planes embedded in 3D -> reconstruction."""
    pts, asg, meta, edges, P3, R3, lat = compute(atlas)
    nc = meta["n_charts"]

    fig = plt.figure(figsize=(12.5, 4.1))
    axd = fig.add_subplot(131, projection="3d")
    axc = fig.add_subplot(132, projection="3d")
    axr = fig.add_subplot(133, projection="3d")
    for ax in (axd, axr):
        ax.view_init(elev, azim)
    axc.view_init(22, -55)
    _scatter3(axd, P3, asg, "point cloud $X\\subset\\mathbb{R}^N$")
    _scatter3(axr, R3, asg, "reconstruction $D_i(E_i(x))$")

    # charts as tilted planes stacked in 3D
    centers = [np.array([0, 0, (nc - 1) / 2 - i]) * 1.35 for i in range(nc)]
    tilt = np.deg2rad(18)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    latpos = []
    for i in range(nc):
        Z = to2d(lat[i])
        Z = (Z - Z.min(0)) / (Z.max(0) - Z.min(0) + 1e-12) - 0.5
        p = centers[i][None, :] + Z[:, :1] * e1 * 1.9 + Z[:, 1:2] * e2 * 1.15
        latpos.append(p)
        axc.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.3,
                    c=PALETTE[i % len(PALETTE)], linewidths=0, rasterized=True)
        corners = np.array([[-.55, -.55], [.55, -.55], [.55, .55],
                            [-.55, .55], [-.55, -.55]])
        cp = centers[i][None, :] + corners[:, :1] * e1 * 1.9 \
            + corners[:, 1:2] * e2 * 1.15
        axc.plot(cp[:, 0], cp[:, 1], cp[:, 2], color="#999999", lw=.8)
        axc.text(centers[i][0] - 1.25, centers[i][1] - .4, centers[i][2] + .35,
                 f"$E_{{{i}}}(U_{{{i}}})$", fontsize=9,
                 color=PALETTE[i % len(PALETTE)])

    # overlap components: connect latent images of shared points across planes
    rng = np.random.default_rng(0)
    index = {i: {int(g): k for k, g in enumerate(asg[i])} for i in range(nc)}
    for e in edges:
        i, j = e["i"], e["j"]
        shared = np.intersect1d(asg[i], asg[j])
        from make_atlas_viz import components
        comp = components(pts[shared])
        idx = shared[comp == e["comp"]]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(14, len(idx)), replace=False)
        colr, ls = _edge_style(e)
        segs = [(latpos[i][index[i][int(k)]], latpos[j][index[j][int(k)]])
                for k in pick]
        lc = Line3DCollection(segs, colors=colr, linewidths=.7, alpha=.55,
                              linestyles=ls)
        axc.add_collection3d(lc)
    axc.set_title("charts, and the sign of $\\det dT_{ji}$ on each overlap"
                  "\ncomponent (blue $+1$, red $-1$)", fontsize=10)
    axc.set_axis_off()
    axc.set_box_aspect((1.5, 1, 1.1))

    # flow arrows between panels
    for x0, x1, label in ((0.315, 0.365, "$E_i$"),
                          (0.645, 0.695, "$D_i$")):
        fig.patches.append(matplotlib.patches.FancyArrowPatch(
            (x0, 0.5), (x1, 0.5), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=22, color="#444444", lw=1.6))
        fig.text((x0 + x1) / 2, 0.545, label, ha="center", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("atlas")
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--elev", type=float, default=18)
    ap.add_argument("--azim", type=float, default=-60)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.pipeline:
        fig_pipeline(args.atlas, args.out, args.elev, args.azim)
    else:
        fig_panel(args.atlas, args.out, args.elev, args.azim)
    print(args.out)
