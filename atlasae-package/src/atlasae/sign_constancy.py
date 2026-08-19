"""
Sign constancy from sampling density, as an alternative to Theta < delta.

The pipeline already verifies the cocycle identity omega_ki = omega_kj * omega_ji
*directly*, by autodiff, at every sampled point of every triple overlap.  What
direct verification cannot see is what happens BETWEEN samples: det g_ji could
dip through zero in a gap and flip the sign there unobserved.  Closing that gap
is the only job the condition Theta < delta of the global stability theorem
actually does.

Theta buys it very expensively.  It is a product of four regularity constants
through three worst-case stages (eta -> eta_eff -> Gamma -> Theta), taken
uniformly over the whole atlas, and in practice exceeds delta by orders of
magnitude (see the supplement).

A far cheaper argument gives the same conclusion.  Fix an overlap component and
work in the latent coordinate z = E_i(x), in which f(z) := det g_ji is a scalar
C^1 function.  If

    min_z |f(z)|  >  L * h,        L := sup_z ||grad f(z)||,
                                   h := covering radius of the samples,

then f cannot reach zero anywhere in the component: any point is within h of a
sample, where |f| >= min|f|, and moving distance h changes f by at most L*h.
By the intermediate value theorem the sign of f is therefore constant on the
whole component, not merely at the sampled points.

This needs ONE constant, directly measurable by autodiff, instead of four; it
carries no (L_E L_D)^d amplification; and -- unlike Theta < delta -- it can be
made to hold simply by sampling more densely, since h shrinks while min|f| and
L do not.

The estimates of L and h are themselves sampled, so this is a sharper and
cheaper sufficient condition, not a removal of all sampling assumptions.  What
it removes is the constant cascade.

Usage:
    from atlasae.sign_constancy import sign_constancy_report
    r = sign_constancy_report(system)
    r['fraction_certified']      # overlaps whose sign is provably constant
    r['worst_margin']            # min |f| / (L h) over overlaps; >1 is good
"""

from itertools import combinations

import numpy as np

__all__ = ["overlap_sign_constancy", "sign_constancy_report"]


def _covering_radius(z):
    """Estimate the covering radius of a point set by its largest
    nearest-neighbour distance.  Cheap and standard; mildly optimistic."""
    if len(z) < 2:
        return np.inf
    from scipy.spatial import cKDTree
    d, _ = cKDTree(z).query(z, k=2)
    return float(d[:, 1].max())


def overlap_sign_constancy(system, i, j, idx, k_local=12, mode="local"):
    """Sign-constancy test for det g_ji on the overlap points `idx`.

    mode="global"  min|f| > (max L)(max h)   -- simple but very pessimistic:
                   it charges the largest gradient anywhere against the
                   sparsest region anywhere, while only the neighbourhood of
                   the minimum of |f| actually binds.

    mode="local"   a covering argument.  Each sample z_k protects the ball of
                   radius h_k around it, where h_k is the local spacing; the
                   sign is constant provided

                       |f(z_k)| > L_k h_k     for every k,

                   with L_k the largest gradient among the k_local nearest
                   neighbours of z_k.  Points where |f| is large carry plenty
                   of slack, so only the near-zero region has to work hard.

    Returns the per-point worst margin min_k |f(z_k)| / (L_k h_k).
    """
    import tensorflow as tf

    x = tf.constant(system.data[idx], dtype=tf.float32)
    ae_i, ae_j = system.autoencoders[i], system.autoencoders[j]
    z = tf.constant(ae_i.encode(x).numpy(), dtype=tf.float32)

    with tf.GradientTape() as t1:
        t1.watch(z)
        with tf.GradientTape() as t2:
            t2.watch(z)
            zz = ae_j.encode(ae_i.decode(z))
        J = t2.batch_jacobian(zz, z)          # (n, d, d) = g_ji
        det = tf.linalg.det(J)                # (n,)
    grad = t1.gradient(det, z)                # (n, d)

    det = np.asarray(det)
    gn = np.linalg.norm(np.asarray(grad), axis=1)
    zn = np.asarray(z)
    n = len(zn)

    if mode == "global" or n < max(3, k_local):
        L = float(gn.max())
        h = _covering_radius(zn)
        margin = (float(np.abs(det).min()) / (L * h)
                  if (L > 0 and np.isfinite(h) and h > 0) else np.inf)
        L_rep, h_rep = L, h
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(zn)
        kk = min(k_local + 1, n)
        dists, nbrs = tree.query(zn, k=kk)
        h_k = dists[:, 1]                      # local spacing at each sample
        L_k = gn[nbrs].max(axis=1)             # local gradient bound
        denom = L_k * h_k
        with np.errstate(divide="ignore", invalid="ignore"):
            per_point = np.where(denom > 0, np.abs(det) / denom, np.inf)
        margin = float(per_point.min())
        L_rep, h_rep = float(np.median(L_k)), float(np.median(h_k))

    # The sign of the overlap, needed to assemble the cocycle.  When the margin
    # exceeds 1 the sign is provably constant on the component, so any sampled
    # value is *the* value; we take the majority regardless so that the field is
    # populated for uncertified overlaps too (where it is only indicative).
    n_pos = int((det > 0).sum())
    omega = 1 if n_pos * 2 >= n else -1

    return dict(pair=(j, i), n=n, min_abs_det=float(np.abs(det).min()),
                L=L_rep, h=h_rep, margin=float(margin),
                certified=bool(margin > 1.0),
                omega=omega, n_positive=n_pos,
                sign_flip_observed=bool(det.min() < 0 < det.max()))


def overlap_components(system, idx, min_points=5, scale=3.0):
    """Split an overlap into connected components.

    prop:sign-constancy is a statement about a connected component of
    U_i \\cap U_j: a non-vanishing continuous function has constant sign on a
    *connected* set.  An overlap that falls into several pieces must therefore
    be tested piecewise -- and the pieces may genuinely carry different signs.
    The Mobius band is the whole point: two charts, one overlap with two
    components, one of which reverses orientation.  Testing the union would
    average the two away.

    Components are the connected components of the graph joining points closer
    than `scale` times the median nearest-neighbour distance of the overlap.

    The threshold is adaptive here, whereas chart domains are split at the fixed
    absolute radius the pipeline uses (see sign_constancy_report).  That is
    deliberate: charts are *designed* to be of comparable size, the cover
    targeting a fixed number of points each, so a single absolute radius suits
    them; overlaps are whatever the intersection of two charts happens to
    contain, and their size and density vary widely within one atlas, so a rule
    scaled to the local spacing avoids fragmenting the sparse ones and merging
    the dense ones.  Verdicts are unchanged if the fixed radius is used here
    too -- only the component counts differ.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix

    pts = system.data[idx]
    n = len(pts)
    if n < 2:
        return [idx]
    d, _ = cKDTree(pts).query(pts, k=2)
    eps = scale * float(np.median(d[:, 1]))
    if not np.isfinite(eps) or eps <= 0:
        return [idx]
    pairs = np.array(list(cKDTree(pts).query_pairs(eps)))
    if len(pairs) == 0:
        return [idx]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(n, n))
    n_comp, lab = connected_components(g, directed=False)
    out = [np.asarray(idx)[lab == c] for c in range(n_comp)]
    return [c for c in out if len(c) >= min_points]


def sign_constancy_report(system, min_points=5, mode="local", k_local=12,
                          split_components=True, eps_cluster=1.0):
    """Run the test on every overlap component of the atlas.

    'certified' components have provably constant sign on the whole component,
    not merely at the sampled points.

    Each unordered pair {i,j} is visited once: omega_ij = omega_ji in {+1,-1},
    so visiting both orderings would double-count every overlap.
    """
    # Vertices of the nerve are *chart components*, not charts.  The relation
    # omega_ji = nu_j nu_i assigns one sign per connected component of a chart
    # domain; identifying the components of a disconnected chart would force
    # nu equal across pieces that the cocycle treats separately, adding
    # constraints that can manufacture an odd cycle that is not there.  Around
    # half the charts in our atlases are disconnected, so this is not a corner
    # case.
    # Chart components use the pipeline's own criterion -- DBSCAN at
    # eps_cluster, the same call check_orientability makes -- so that the graph
    # certified here is the graph the verdict was read off.  A finer split
    # would fragment charts and destroy genuine odd cycles; a coarser one would
    # merge components and invent them.
    chart_comp = {}
    if split_components:
        from sklearn.cluster import DBSCAN
        for i in range(system.n_charts):
            idx = np.asarray(getattr(system, "subset_assignments",
                                     [[]] * (i + 1))[i])
            if len(idx) == 0:
                continue
            lab = DBSCAN(eps=eps_cluster, min_samples=5).fit(
                system.data[idx]).labels_
            for p, l in zip(idx, lab):
                chart_comp[(i, int(p))] = int(l)

    def vertex(i, part):
        """The component of chart i that this overlap component sits in."""
        if not split_components:
            return (i, 0)
        labs = [chart_comp.get((i, int(p))) for p in part]
        labs = [l for l in labs if l is not None]
        if not labs:
            return (i, 0)
        return (i, max(set(labs), key=labs.count))

    rows = []
    for i, j in combinations(range(system.n_charts), 2):
        ov = getattr(system, "pairwise_overlaps", {}).get((i, j))
        if ov is None or len(ov) < min_points:
            continue
        parts = (overlap_components(system, ov, min_points=min_points)
                 if split_components else [ov])
        for c, part in enumerate(parts):
            r = overlap_sign_constancy(system, i, j, part,
                                       k_local=k_local, mode=mode)
            r["comp"] = c
            r["n_comp"] = len(parts)
            r["vi"] = list(vertex(i, part))
            r["vj"] = list(vertex(j, part))
            rows.append(r)
    if not rows:
        return dict(n_overlaps=0, fraction_certified=0.0, worst_margin=0.0,
                    rows=[])
    marg = np.array([r["margin"] for r in rows])
    return dict(
        n_overlaps=len(rows),
        n_certified=int(sum(r["certified"] for r in rows)),
        fraction_certified=float(np.mean([r["certified"] for r in rows])),
        worst_margin=float(marg.min()),
        median_margin=float(np.median(marg)),
        n_sign_flips=int(sum(r["sign_flip_observed"] for r in rows)),
        median_h=float(np.median([r["h"] for r in rows])),
        median_L=float(np.median([r["L"] for r in rows])),
        rows=rows,
    )
