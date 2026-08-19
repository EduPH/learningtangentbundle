"""Figures of a learned atlas: latent charts and the signed nerve.

The picture the method deserves is the one that shows *why* a verdict came
out the way it did: each chart's latent coordinates drawn in its own cell,
and one line per overlap component joining the cells it relates, coloured by
the sign of det dT_ji on that component.  A non-orientable atlas is the one
whose coloured lines admit no consistent two-colouring, and on the Mobius
band you can see it directly: two charts whose overlap carries both a blue
component and a red one, which no assignment of +-1 to the charts can match.

    from atlasae import plot_atlas, plot_chart_transitions

    fig = plot_atlas(system)                 # data | charts + nerve | recon
    fig = plot_chart_transitions(system)     # the middle panel alone
    fig.savefig("atlas.png", dpi=200, bbox_inches="tight")

Colours follow the paper: blue = +1 (orientation preserving), red = -1
(orientation reversing), dashed orange = both signs seen on one component (a
diagnostic: the component is either mis-split or under-resolved).  Edges whose
sign is *certified* constant between the sample points (margin > 1,
prop:sign-constancy) are drawn solid and opaque; uncertified edges are drawn
thin and pale, so a glance separates what is proved from what is merely
observed.

Signs come from `sign_constancy_report`, i.e. from the same computation the
certificate uses; nothing here re-derives them.  That call differentiates the
transition maps, so it is the expensive part.  Pass `rows=` to reuse a report
you already have:

    rep = sign_constancy_report(system)
    fig = plot_atlas(system, rows=rep["rows"])
"""

import numpy as np

__all__ = ["plot_atlas", "plot_chart_transitions", "PALETTE"]

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
           "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
           "#dbdb8d", "#9edae5"]

SIGN_COLOUR = {1: "#1f77b4", -1: "#d62728", 0: "#e07b00"}
_DENSE_NERVE = 12          # above this many charts, draw a sign matrix instead


# ----------------------------------------------------------------- helpers

def _project(X, k, basis=None, mean=None):
    """PCA-project X to k dimensions (pad with zeros if already smaller)."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] <= k:
        Y = np.zeros((len(X), k))
        Y[:, : X.shape[1]] = X
        return Y, None, None
    if basis is None:
        mean = np.nanmean(X, axis=0)
        finite = X[np.isfinite(X).all(axis=1)]
        _, _, Vt = np.linalg.svd(finite - mean, full_matrices=False)
        basis = Vt[:k]
    return (X - mean) @ basis.T, basis, mean


def _assignments(system):
    asg = getattr(system, "subset_assignments", None)
    if asg is None:
        raise AttributeError(
            "system has no subset_assignments; pass an AtlasAutoencoder built "
            "with a cover, or one restored by load_atlas()")
    return [np.asarray(a) for a in asg]


def _latents_and_reconstruction(system):
    """Per-chart latent coordinates, the reconstruction, and the atlas's eps.

    A point in several charts is reconstructed by each of them; the drawn
    reconstruction keeps one of those (charts overlap, so the panel would
    otherwise show duplicates), but eps is the supremum over every
    chart-point pair, as in def:approximate-atlas -- taking it from the
    overwritten array would silently report the last chart's error only.
    """
    data = np.asarray(system.data, dtype=float)
    asg = _assignments(system)
    latents, recon, eps = [], np.full_like(data, np.nan), 0.0
    for i, a in enumerate(asg):
        ae = system.autoencoders[i]
        z = np.asarray(ae.encode(data[a]))
        latents.append(z)
        r = np.asarray(ae.decode(z))
        recon[a] = r
        if len(a):
            eps = max(eps, float(np.linalg.norm(r - data[a], axis=1).max()))
    return data, asg, latents, recon, eps


def _edges(system, rows, min_points):
    """Normalise sign-constancy rows into drawable edge records."""
    if rows is None:
        from atlasae.sign_constancy import sign_constancy_report
        rows = sign_constancy_report(system, min_points=min_points)["rows"]
    out = []
    for r in rows:
        j, i = r["pair"]                      # rows record (target, source)
        out.append(dict(
            i=int(min(i, j)), j=int(max(i, j)),
            comp=int(r.get("comp", 0)),
            sign=0 if r.get("sign_flip_observed") else int(r.get("omega", 1)),
            certified=bool(r.get("certified", False)),
            margin=float(r.get("margin", np.nan)),
            n=int(r.get("n", 0)),
        ))
    return out


def _edge_style(e):
    colour = SIGN_COLOUR[e["sign"]]
    if e["sign"] == 0:                        # mixed signs on one component
        return colour, "--", 1.4, 0.9
    if e["certified"]:
        return colour, "-", 1.5, 0.9
    return colour, "-", 0.8, 0.4               # observed but not certified


def _cell_grid(n_charts):
    cols = int(np.ceil(np.sqrt(n_charts)))
    rows_ = int(np.ceil(n_charts / cols))
    cell = {i: (i % cols, rows_ - 1 - i // cols) for i in range(n_charts)}
    return cols, rows_, cell


def _cube(P3):
    """Centre and half-width of a cubic bounding box around the finite points."""
    finite = P3[np.isfinite(P3).all(axis=1)]
    if not len(finite):
        return np.zeros(3), 1.0
    m = finite.mean(0)
    return m, (np.abs(finite - m).max() or 1.0)


def _scatter3(ax, P3, asg, title, s=1.0, cube=None):
    for i, a in enumerate(asg):
        ax.scatter(P3[a, 0], P3[a, 1], P3[a, 2], s=s,
                   c=PALETTE[i % len(PALETTE)], linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    try:                                       # zoom= needs matplotlib >= 3.6
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    # Shared limits, not per-panel autoscale: the data and its reconstruction
    # must be drawn to the same scale, or a shrunken reconstruction is silently
    # stretched back to fill its frame and looks faithful when it is not.
    m, lim = _cube(P3) if cube is None else cube
    ax.set_xlim(m[0] - lim, m[0] + lim)
    ax.set_ylim(m[1] - lim, m[1] + lim)
    ax.set_zlim(m[2] - lim, m[2] + lim)


def _draw_charts_and_nerve(ax, latents, edges, n_charts, draw_nerve=True):
    """The atlas plane: one cell per chart, one curve per overlap component."""
    from matplotlib.patches import Rectangle

    cols, rows_, cell = _cell_grid(n_charts)
    pad = 0.14
    for i in range(n_charts):
        Z, _, _ = _project(latents[i], 2)
        span = Z.max(0) - Z.min(0)
        Z = (Z - Z.min(0)) / np.where(span > 0, span, 1.0)
        cx, cy = cell[i]
        ax.scatter(cx + pad + (1 - 2 * pad) * Z[:, 0],
                   cy + pad + (1 - 2 * pad) * Z[:, 1],
                   s=1.2, c=PALETTE[i % len(PALETTE)], linewidths=0,
                   rasterized=True, zorder=2)
        ax.add_patch(Rectangle((cx + .03, cy + .03), .94, .94, fill=False,
                               ec="#bbbbbb", lw=.7, zorder=1))
        ax.text(cx + .08, cy + .84, str(i), fontsize=7, zorder=3,
                color=PALETTE[i % len(PALETTE)], fontweight="bold")

    if draw_nerve:
        seen = {}
        for e in edges:
            c1 = np.array(cell[e["i"]], dtype=float) + .5
            c2 = np.array(cell[e["j"]], dtype=float) + .5
            k = seen.get((e["i"], e["j"]), 0)
            seen[(e["i"], e["j"])] = k + 1
            d = c2 - c1
            u = d / (np.linalg.norm(d) + 1e-12)
            # start clear of the latent point cloud (which fills +-0.36 of the
            # cell) rather than at its centre, so edges do not cross the charts
            # they connect; diagonal neighbours need a longer reach
            reach = 0.38 / max(abs(u[0]), abs(u[1]), 1e-12)
            c1, c2 = c1 + u * reach, c2 - u * reach
            d = c2 - c1
            nrm = np.array([-d[1], d[0]])
            nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
            # fan parallel components apart so both Mobius signs stay visible
            mid = (c1 + c2) / 2 + nrm * 0.13 * ((k + 1) // 2) * (-1) ** k
            colour, ls, lw, alpha = _edge_style(e)
            t = np.linspace(0, 1, 24)[:, None]
            curve = (1 - t) ** 2 * c1 + 2 * t * (1 - t) * mid + t ** 2 * c2
            ax.plot(curve[:, 0], curve[:, 1], color=colour, ls=ls, lw=lw,
                    alpha=alpha, zorder=4)

    ax.set_xlim(-.1, cols + .1)
    ax.set_ylim(-.1, rows_ + .1)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _sign_matrix(ax, edges, n_charts):
    """Fallback for a nerve too dense to draw: sign adjacency as an image."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    M = np.full((n_charts, n_charts), np.nan)
    for e in edges:
        v = float(e["sign"])
        for a, b in ((e["i"], e["j"]), (e["j"], e["i"])):
            M[a, b] = v if (np.isnan(M[a, b]) or M[a, b] == v) else 0.0
    ax.imshow(np.ma.masked_invalid(M),
              cmap=ListedColormap(["#d62728", "#e07b00", "#1f77b4"]),
              norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3),
              origin="upper", interpolation="nearest")
    ax.set_title(r"sign of $\det dT_{ji}$ per pair", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")


def _legend(fig, edges):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=SIGN_COLOUR[1], lw=1.5,
               label=r"$\omega_{ji}=+1$ (certified)"),
        Line2D([], [], color=SIGN_COLOUR[-1], lw=1.5,
               label=r"$\omega_{ji}=-1$ (certified)"),
        Line2D([], [], color="#777777", lw=0.8, alpha=.5,
               label="uncertified (margin $\\leq 1$)"),
    ]
    if any(e["sign"] == 0 for e in edges):
        handles.append(Line2D([], [], color=SIGN_COLOUR[0], lw=1.4, ls="--",
                              label="mixed signs on one component"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))


# ------------------------------------------------------------------- public

def plot_chart_transitions(system, rows=None, min_points=5, figsize=None,
                           title="latent charts and signed nerve", ax=None):
    """Draw the latent charts and the signed nerve of the cover.

    Each chart's latent coordinates occupy one cell (PCA-projected to the
    plane when the latent dimension exceeds 2); each overlap component is one
    curve between cells, coloured by the sign of det dT_ji and drawn solid
    when that sign is certified constant between the sample points.

    Parameters
    ----------
    system : AtlasAutoencoder
        A trained atlas carrying `subset_assignments`.
    rows : list of dict, optional
        `sign_constancy_report(system)["rows"]`. Recomputed if omitted.
    min_points : int
        Minimum sample points for an overlap component to be drawn; matches
        the pipeline's `m_0` (see the note on pruning in the README).
    figsize : tuple, optional
        Defaults to a size matching the chart grid.
    ax : matplotlib Axes, optional
        Draw into an existing axes instead of creating a figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    _, _, latents, _, _ = _latents_and_reconstruction(system)
    edges = _edges(system, rows, min_points)
    n_charts = system.n_charts

    if ax is None:
        if figsize is None:                    # match the chart grid's aspect
            cols, rows_, _ = _cell_grid(n_charts)
            figsize = (2.7 * cols + (2.2 if n_charts > _DENSE_NERVE else 0),
                       2.7 * rows_ + 0.8)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    dense = n_charts > _DENSE_NERVE
    _draw_charts_and_nerve(ax, latents, edges, n_charts, draw_nerve=not dense)
    ax.set_title(("latent charts" if dense else title), fontsize=10)
    if dense:
        _sign_matrix(ax.inset_axes([1.06, 0.18, 0.52, 0.64]), edges, n_charts)
    else:
        _legend(fig, edges)
    fig.tight_layout()
    return fig


def plot_atlas(system, rows=None, min_points=5, figsize=(11.5, 3.9),
               elev=18, azim=-60, point_size=1.0):
    """Three panels: the data, the latent charts with the signed nerve, and
    the reconstruction.

    The outer panels are PCA-projected to three dimensions when the ambient
    dimension exceeds three, sharing one projection so that the data and its
    reconstruction are directly comparable. Colours identify charts.

    Parameters as in `plot_chart_transitions`; `elev`/`azim` set the 3-D view.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    data, asg, latents, recon, eps = _latents_and_reconstruction(system)
    edges = _edges(system, rows, min_points)
    n_charts = system.n_charts

    P3, basis, mean = _project(data, 3)
    R3, _, _ = _project(recon, 3, basis, mean)

    fig = plt.figure(figsize=figsize)
    ax_data = fig.add_subplot(131, projection="3d")
    ax_mid = fig.add_subplot(132)
    ax_recon = fig.add_subplot(133, projection="3d")
    for ax in (ax_data, ax_recon):
        ax.view_init(elev, azim)

    # one projection, one view, one set of limits for both panels, so that a
    # faithful reconstruction overlays the data and a poor one visibly departs
    cube = _cube(P3)
    _scatter3(ax_data, P3, asg, "data", s=point_size, cube=cube)
    _scatter3(ax_recon, R3, asg,
              r"reconstruction $D_i(E_i(x))$"
              f"   ($\\varepsilon={eps:.3g}$)",
              s=point_size, cube=cube)

    dense = n_charts > _DENSE_NERVE
    _draw_charts_and_nerve(ax_mid, latents, edges, n_charts,
                           draw_nerve=not dense)
    ax_mid.set_title("latent charts" if dense
                     else "latent charts and signed nerve", fontsize=10)
    if dense:
        _sign_matrix(ax_mid.inset_axes([1.06, 0.18, 0.52, 0.64]),
                     edges, n_charts)
    else:
        _legend(fig, edges)

    fig.tight_layout()
    return fig
