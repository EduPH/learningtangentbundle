"""
Interactive HTML visualization of a saved autoencoder atlas.

Reads any atlas saved by the pipeline (points.npy, assignments/, weights.npz,
meta.json) and produces a single self-contained HTML page with three linked
panels:

    [ original data, 3D ]   [ atlas plane: latent charts + nerve ]   [ reconstruction, 3D ]

* Left: the point cloud, coloured by chart (PCA-projected to 3D if the
  ambient dimension exceeds 3).
* Middle: every chart's latent embedding drawn in its own cell, with one
  line per overlap component connecting the cells; blue = orientation-
  preserving (det > 0), red = orientation-reversing (det < 0), dashed
  orange = mixed signs within one component (a diagnostic, see the paper).
* Right: the decoded reconstruction, same colours and projection.

Everything is recomputed in pure numpy from the saved weights (no
TensorFlow needed): forward passes, transition Jacobians (analytic, tanh),
overlap components (union-find on a radius graph).

Usage:
    python make_atlas_viz.py results/results_paper_*/atlases/Mobius_seed42
    python make_atlas_viz.py <atlas_dir> --out viz/mobius.html
"""

import argparse
import json
import os

import numpy as np

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
           "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
           "#dbdb8d", "#9edae5"]


# ---------------------------------------------------------------- atlas I/O

def load_atlas(path):
    z = np.load(os.path.join(path, "weights.npz"))
    meta = json.load(open(os.path.join(path, "meta.json")))
    pts = np.load(os.path.join(path, "points.npy"))
    asg = [np.load(os.path.join(path, "assignments", f"chart_{i}.npy"))
           for i in range(meta["n_charts"])]
    nets = []
    for i in range(meta["n_charts"]):
        def stack(tag, i=i):
            ws, k = [], 0
            while f"chart_{i}_{tag}_{k}" in z.files:
                ws.append(z[f"chart_{i}_{tag}_{k}"]); k += 1
            return [(ws[j], ws[j + 1]) for j in range(0, len(ws), 2)]
        nets.append((stack("enc"), stack("dec")))
    return pts, asg, nets, meta


def fwd(layers, x):
    h = x
    for j, (W, b) in enumerate(layers):
        h = h @ W + b
        if j < len(layers) - 1:
            h = np.tanh(h)
    return h


def jac(layers, x):
    """Analytic Jacobian of a dense tanh stack at a single point x."""
    h = x
    J = np.eye(len(x))
    for j, (W, b) in enumerate(layers):
        pre = h @ W + b
        J = W.T @ J
        if j < len(layers) - 1:
            h = np.tanh(pre)
            J = (1 - h ** 2)[:, None] * J
        else:
            h = pre
    return J, h


# ------------------------------------------------------- overlap components

def components(X, r_factor=4.0):
    """Union-find components of a radius graph on X."""
    n = len(X)
    if n == 1:
        return np.zeros(1, int)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    nn = np.partition(D + np.eye(n) * 1e9, 0, axis=1)[:, 0]
    r = r_factor * np.median(nn)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.where(D < r)):
        if a < b:
            parent[find(a)] = find(b)
    roots = {find(a) for a in range(n)}
    remap = {rt: k for k, rt in enumerate(sorted(roots))}
    return np.array([remap[find(a)] for a in range(n)])


def transition_signs(pts, asg, nets, min_points=5, max_eval=60, seed=0):
    """One record per overlap component: (i, j, sign, mixed, n_points)."""
    rng = np.random.default_rng(seed)
    out = []
    nc = len(asg)
    for i in range(nc):
        for j in range(i + 1, nc):
            shared = np.intersect1d(asg[i], asg[j])
            if len(shared) < min_points:
                continue
            comp = components(pts[shared])
            for c in range(comp.max() + 1):
                idx = shared[comp == c]
                if len(idx) < min_points:
                    continue
                sample = rng.choice(idx, min(max_eval, len(idx)), replace=False)
                dets = []
                for k in sample:
                    zi = fwd(nets[i][0], pts[k][None])[0]
                    Jd, y = jac(nets[i][1], zi)
                    Je, _ = jac(nets[j][0], y)
                    dets.append(np.linalg.det((Je @ Jd)[: len(zi), : len(zi)]))
                dets = np.array(dets)
                pos, neg = (dets > 0).mean(), (dets < 0).mean()
                mixed = min(pos, neg) > 0.1
                out.append(dict(i=i, j=j, comp=int(c), n=int(len(idx)),
                                sign=int(np.sign(np.median(dets))),
                                mixed=bool(mixed)))
    return out


# --------------------------------------------------------------- projection

def to3d(X, basis=None, mean=None):
    if X.shape[1] <= 3:
        Y = np.zeros((len(X), 3))
        Y[:, : X.shape[1]] = X
        return Y, None, None
    if basis is None:
        mean = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        basis = Vt[:3]
    return (X - mean) @ basis.T, basis, mean


def to2d(Z):
    if Z.shape[1] == 2:
        return Z
    m = Z.mean(0)
    _, _, Vt = np.linalg.svd(Z - m, full_matrices=False)
    return (Z - m) @ Vt[:2].T


# --------------------------------------------------------------------- html

def build_figure(pts, asg, nets, meta, edges):
    nc = meta["n_charts"]
    P3, basis, mean = to3d(pts)
    recon = np.full_like(pts, np.nan)
    lat = []
    for i, a in enumerate(asg):
        zi = fwd(nets[i][0], pts[a])
        lat.append(zi)
        recon[a] = fwd(nets[i][1], zi)
    R3 = ((recon - mean) @ basis.T) if basis is not None else to3d(recon)[0]

    # grid of chart cells for the atlas plane
    cols = int(np.ceil(np.sqrt(nc)))
    rows = int(np.ceil(nc / cols))
    cell = {}
    for i in range(nc):
        r, c = divmod(i, cols)
        cell[i] = (c, rows - 1 - r)        # (x, y) cell coordinates

    traces, shapes, annotations = [], [], []
    col = lambda i: PALETTE[i % len(PALETTE)]

    # left: original data (scene), right: reconstruction (scene2)
    for i, a in enumerate(asg):
        for scene, M, nm in (("scene", P3[a], "data"), ("scene2", R3[a], "recon")):
            traces.append(dict(
                type="scatter3d", mode="markers", scene=scene,
                x=M[:, 0].tolist(), y=M[:, 1].tolist(), z=M[:, 2].tolist(),
                marker=dict(size=1.6, color=col(i)),
                name=f"chart {i}", legendgroup=f"c{i}",
                showlegend=(scene == "scene"),
                hovertext=f"chart {i} ({nm})", hoverinfo="text"))

    # middle: latent cells
    pad = 0.12
    for i in range(nc):
        Z = to2d(lat[i])
        Z = (Z - Z.min(0)) / (Z.max(0) - Z.min(0) + 1e-12)
        cx, cy = cell[i]
        traces.append(dict(
            type="scatter", mode="markers", xaxis="x", yaxis="y",
            x=(cx + pad + (1 - 2 * pad) * Z[:, 0]).tolist(),
            y=(cy + pad + (1 - 2 * pad) * Z[:, 1]).tolist(),
            marker=dict(size=2.2, color=col(i)),
            name=f"chart {i}", legendgroup=f"c{i}", showlegend=False,
            hovertext=f"chart {i} latent", hoverinfo="text"))
        shapes.append(dict(type="rect", xref="x", yref="y",
                           x0=cx + 0.03, y0=cy + 0.03, x1=cx + 0.97, y1=cy + 0.97,
                           line=dict(color="#bbbbbb", width=1)))
        annotations.append(dict(x=cx + 0.10, y=cy + 0.94, xref="x", yref="y",
                                text=f"<b>{i}</b>", showarrow=False,
                                font=dict(size=11, color=col(i))))

    # nerve edges between cell centres, one per overlap component
    seen = {}
    for e in edges:
        (cx1, cy1), (cx2, cy2) = cell[e["i"]], cell[e["j"]]
        p1 = np.array([cx1 + 0.5, cy1 + 0.5])
        p2 = np.array([cx2 + 0.5, cy2 + 0.5])
        k = seen.get((e["i"], e["j"]), 0)
        seen[(e["i"], e["j"])] = k + 1
        d = p2 - p1
        nrm = np.array([-d[1], d[0]])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
        mid = (p1 + p2) / 2 + nrm * 0.10 * (k - 0.5 * (k > 0))
        colr = "#e07b00" if e["mixed"] else ("#1f77b4" if e["sign"] > 0 else "#d62728")
        label = ("mixed" if e["mixed"] else
                 ("+1 (preserves)" if e["sign"] > 0 else "-1 (reverses)"))
        traces.append(dict(
            type="scatter", mode="lines", xaxis="x", yaxis="y",
            x=[p1[0], mid[0], p2[0]], y=[p1[1], mid[1], p2[1]],
            line=dict(color=colr, width=2.5,
                      dash="dash" if e["mixed"] else "solid"),
            opacity=0.85, showlegend=False, hoverinfo="text",
            hovertext=(f"T({e['j']}&#8592;{e['i']}) component {e['comp']}: "
                       f"sign {label}, {e['n']} pts")))

    note = ("" if pts.shape[1] <= 3 else
            f"  (PCA projection of ambient R^{pts.shape[1]})")
    layout = dict(
        title=dict(text=(f"Autoencoder atlas: {meta.get('note', '')} "
                         f"&#8212; {nc} charts, d={meta['latent_dim']}{note}"),
                   x=0.5, font=dict(size=15)),
        margin=dict(l=0, r=0, t=50, b=0), height=640,
        paper_bgcolor="white", legend=dict(orientation="h", y=-0.02),
        scene=dict(domain=dict(x=[0.0, 0.30], y=[0, 1]),
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False),
                   annotations=[]),
        scene2=dict(domain=dict(x=[0.70, 1.0], y=[0, 1]),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    zaxis=dict(visible=False)),
        xaxis=dict(domain=[0.33, 0.67], visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        shapes=shapes,
        annotations=annotations + [
            dict(x=0.15, y=1.04, xref="paper", yref="paper", showarrow=False,
                 text="<b>data</b>", font=dict(size=13)),
            dict(x=0.50, y=1.04, xref="paper", yref="paper", showarrow=False,
                 text=("<b>charts &amp; nerve</b>  "
                       "(<span style='color:#1f77b4'>+1</span> / "
                       "<span style='color:#d62728'>&#8722;1</span> / "
                       "<span style='color:#e07b00'>mixed</span>)"),
                 font=dict(size=13)),
            dict(x=0.85, y=1.04, xref="paper", yref="paper", showarrow=False,
                 text="<b>reconstruction</b>", font=dict(size=13)),
        ])
    return dict(data=traces, layout=layout)


TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Helvetica,Arial,sans-serif;margin:12px}}</style>
</head><body>
<div id="fig"></div>
<p style="color:#555;font-size:13px;max-width:900px">
Left: sampled point cloud, coloured by chart (click legend entries to
toggle charts). Middle: each chart's latent embedding in its own cell;
lines are overlap components of the nerve, <span style="color:#1f77b4">
blue</span> where the transition preserves orientation
(det&nbsp;&gt;&nbsp;0), <span style="color:#d62728">red</span> where it
reverses (det&nbsp;&lt;&nbsp;0); the verdict is the two-colourability of
this graph. Right: the atlas's reconstruction, decoded chart by chart.
Hover any element for details; 3D panels rotate.</p>
<script>
var fig = {figjson};
Plotly.newPlot("fig", fig.data, fig.layout, {{responsive: true}});
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("atlas", help="saved atlas directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pts, asg, nets, meta = load_atlas(args.atlas)
    edges = transition_signs(pts, asg, nets,
                             min_points=meta.get("min_points", 5))
    fig = build_figure(pts, asg, nets, meta, edges)
    out = args.out or os.path.join(
        "viz", os.path.basename(os.path.normpath(args.atlas)) + ".html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    title = f"Atlas viz: {meta.get('note', os.path.basename(args.atlas))}"
    with open(out, "w") as f:
        f.write(TEMPLATE.format(title=title, figjson=json.dumps(fig)))
    n_neg = sum(1 for e in edges if e["sign"] < 0 and not e["mixed"])
    print(f"{out}: {meta['n_charts']} charts, {len(edges)} overlap components "
          f"({n_neg} orientation-reversing), "
          f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
