"""
Post-hoc evaluations for the three review TODOs (M2b, M5b, M6a).

Runs on the SAVED atlases -- no retraining, only forward/backward passes.

    python eval_review_todos.py --task discarded              # M2b
    python eval_review_todos.py --task fixed-radius           # M5b
    python eval_review_todos.py --task margins                # M6a (local-max)
    python eval_review_todos.py --task margins --h-mode mult --h-mult 2.0
    python eval_review_todos.py --task all

Each task prints a per-atlas table and writes
results_review_todos/<task>.json next to this script.

    discarded     M2b: count overlap components dropped by the m0 pruning
                  (overlap_components keeps only components with >= m0 sample
                  points and drops the rest without flagging).  Output feeds
                  the per-trial "discarded" column promised in tab:pertrial.

    fixed-radius  M5b: re-split every overlap at the FIXED absolute radius the
                  pipeline uses for chart domains (eps_cluster, default 1.0)
                  instead of the adaptive 3x-median-NN rule, re-certify, and
                  compare component counts and verdicts under both rules.

    margins       M6a: recompute every margin with a conservative h_k in place
                  of the nearest-neighbour distance, and report how the
                  per-trial C3 status (and hence the certified count) moves.
                  --h-mode local-max   h_k = max NN distance among the k_local
                                       neighbours of z_k (local covering proxy)
                  --h-mode mult        h_k = --h-mult x NN distance
                  --h-mode nn          paper baseline (sanity check)

    adjacency     N1: classify every discarded component as a FRAGMENT
                  (kNN-adjacent, within the overlap's sample, to a retained
                  component of the same overlap -- harmless for the nerve) or
                  as part of a DROPPED EDGE (an overlap retaining nothing --
                  a genuinely deleted nerve edge, which no sample-based check
                  can restore).  Needs no TensorFlow.
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "..", "src"))

from eval_saved import find_atlases, label  # noqa: E402


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------

def _chart_components(system, eps_cluster=1.0):
    """(chart, point) -> component label, exactly as sign_constancy_report."""
    from sklearn.cluster import DBSCAN
    chart_comp = {}
    for i in range(system.n_charts):
        idx = np.asarray(getattr(system, "subset_assignments",
                                 [[]] * (i + 1))[i])
        if len(idx) == 0:
            continue
        lab = DBSCAN(eps=eps_cluster, min_samples=5).fit(
            system.data[idx]).labels_
        for p, l in zip(idx, lab):
            chart_comp[(i, int(p))] = int(l)
    return chart_comp


def _vertex(chart_comp, i, part):
    labs = [chart_comp.get((i, int(p))) for p in part]
    labs = [l for l in labs if l is not None]
    return (i, max(set(labs), key=labs.count) if labs else 0)


def _split_fixed(system, idx, eps, min_points):
    """Connected components at a FIXED absolute radius (M5b comparator)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pts = system.data[idx]
    n = len(pts)
    if n < 2:
        return [np.asarray(idx)]
    pairs = np.array(list(cKDTree(pts).query_pairs(eps)))
    if len(pairs) == 0:
        out = [np.asarray([p]) for p in idx]
    else:
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
        _, lab = connected_components(g, directed=False)
        out = [np.asarray(idx)[lab == c] for c in range(lab.max() + 1)]
    return [c for c in out if len(c) >= min_points]


def _margin(system, i, j, idx, k_local=12, h_mode="nn", h_mult=1.0):
    """overlap_sign_constancy with a configurable h_k (M6a)."""
    import tensorflow as tf
    from atlasae.sign_constancy import _covering_radius

    x = tf.constant(system.data[idx], dtype=tf.float32)
    ae_i, ae_j = system.autoencoders[i], system.autoencoders[j]
    z = tf.constant(ae_i.encode(x).numpy(), dtype=tf.float32)
    with tf.GradientTape() as t1:
        t1.watch(z)
        with tf.GradientTape() as t2:
            t2.watch(z)
            zz = ae_j.encode(ae_i.decode(z))
        J = t2.batch_jacobian(zz, z)
        det = tf.linalg.det(J)
    grad = t1.gradient(det, z)

    det = np.asarray(det)
    gn = np.linalg.norm(np.asarray(grad), axis=1)
    zn = np.asarray(z)
    n = len(zn)

    if n < max(3, k_local):
        L = float(gn.max())
        h = _covering_radius(zn)
        margin = (float(np.abs(det).min()) / (L * h)
                  if (L > 0 and np.isfinite(h) and h > 0) else np.inf)
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(zn)
        kk = min(k_local + 1, n)
        dists, nbrs = tree.query(zn, k=kk)
        d1 = dists[:, 1]                       # NN distance per point
        if h_mode == "nn":
            h_k = d1
        elif h_mode == "mult":
            h_k = h_mult * d1
        elif h_mode == "local-max":
            h_k = d1[nbrs].max(axis=1)         # largest NN dist in the nbhd
        else:
            raise ValueError(h_mode)
        L_k = gn[nbrs].max(axis=1)
        denom = L_k * h_k
        with np.errstate(divide="ignore", invalid="ignore"):
            per_point = np.where(denom > 0, np.abs(det) / denom, np.inf)
        margin = float(per_point.min())

    n_pos = int((det > 0).sum())
    return dict(pair=(j, i), n=n, margin=float(margin),
                certified=bool(margin > 1.0),
                omega=1 if n_pos * 2 >= n else -1,
                sign_flip_observed=bool(det.min() < 0 < det.max()))


def _report(system, splitter, min_points=5, k_local=12,
            h_mode="nn", h_mult=1.0, eps_cluster=1.0):
    """sign_constancy_report re-run with a custom splitter and h_k rule."""
    chart_comp = _chart_components(system, eps_cluster)
    rows = []
    for i, j in combinations(range(system.n_charts), 2):
        ov = getattr(system, "pairwise_overlaps", {}).get((i, j))
        if ov is None or len(ov) < min_points:
            continue
        for c, part in enumerate(splitter(system, ov)):
            r = _margin(system, i, j, part, k_local, h_mode, h_mult)
            r["comp"] = c
            r["vi"] = list(_vertex(chart_comp, i, part))
            r["vj"] = list(_vertex(chart_comp, j, part))
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------------

def task_discarded(paths, m0):
    """M2b: components dropped by the m0 pruning, per atlas."""
    from atlasae import load_atlas
    from atlasae.sign_constancy import overlap_components
    out = []
    print(f"{'experiment':26} {'seed':>5} {'retained':>9} {'discarded':>10} "
          f"{'pts dropped':>12}")
    print("-" * 68)
    for p in paths:
        meta = json.load(open(os.path.join(p, "meta.json")))
        system = load_atlas(p)
        n_ret = n_disc = pts_disc = 0
        for i, j in combinations(range(system.n_charts), 2):
            ov = getattr(system, "pairwise_overlaps", {}).get((i, j))
            if ov is None or len(ov) == 0:
                continue
            # min_points=1 keeps every component; then apply the m0 filter
            parts = overlap_components(system, ov, min_points=1)
            for part in parts:
                if len(part) >= m0:
                    n_ret += 1
                else:
                    n_disc += 1
                    pts_disc += len(part)
        out.append(dict(experiment=label(p), seed=meta.get("seed"),
                        path=os.path.relpath(p, EXP), m0=m0,
                        retained=n_ret, discarded=n_disc,
                        points_discarded=pts_disc))
        print(f"{label(p):26} {str(meta.get('seed')):>5} {n_ret:>9} "
              f"{n_disc:>10} {pts_disc:>12}")
    return out


def task_fixed_radius(paths, m0, eps_fixed):
    """M5b: adaptive vs fixed-radius overlap splitting."""
    from atlasae import load_atlas, certify_verdict
    from atlasae.sign_constancy import overlap_components
    out = []
    print(f"{'experiment':26} {'seed':>5} {'comp(adapt)':>12} "
          f"{'comp(fixed)':>12} {'verdict(adapt)':>15} {'verdict(fixed)':>15} "
          f"{'same':>5}")
    print("-" * 100)
    for p in paths:
        meta = json.load(open(os.path.join(p, "meta.json")))
        system = load_atlas(p)
        rows_a = _report(system, lambda s, ov: overlap_components(
            s, ov, min_points=m0), min_points=m0)
        rows_f = _report(system, lambda s, ov: _split_fixed(
            s, ov, eps_fixed, m0), min_points=m0)
        ca, cf = certify_verdict(rows_a), certify_verdict(rows_f)
        same = (ca["verdict"] == cf["verdict"]
                and ca["certified"] == cf["certified"])
        out.append(dict(experiment=label(p), seed=meta.get("seed"),
                        path=os.path.relpath(p, EXP), eps_fixed=eps_fixed,
                        n_comp_adaptive=len(rows_a), n_comp_fixed=len(rows_f),
                        verdict_adaptive=ca["verdict"],
                        certified_adaptive=ca["certified"],
                        verdict_fixed=cf["verdict"],
                        certified_fixed=cf["certified"], agree=same))
        print(f"{label(p):26} {str(meta.get('seed')):>5} {len(rows_a):>12} "
              f"{len(rows_f):>12} {str(ca['verdict']):>15} "
              f"{str(cf['verdict']):>15} {'yes' if same else 'NO':>5}")
    return out


def task_margins(paths, m0, h_mode, h_mult):
    """M6a: margins with a conservative h_k; C3 status old vs new."""
    from atlasae import load_atlas, certify_verdict
    from atlasae.sign_constancy import overlap_components
    out = []
    print(f"h_mode={h_mode}" + (f" h_mult={h_mult}" if h_mode == "mult" else ""))
    print(f"{'experiment':26} {'seed':>5} {'cert(nn)':>9} {'cert(new)':>10} "
          f"{'C3(nn)':>7} {'C3(new)':>8}")
    print("-" * 72)
    for p in paths:
        meta = json.load(open(os.path.join(p, "meta.json")))
        system = load_atlas(p)
        split = lambda s, ov: overlap_components(s, ov, min_points=m0)
        rows_nn = _report(system, split, min_points=m0, h_mode="nn")
        rows_new = _report(system, split, min_points=m0,
                           h_mode=h_mode, h_mult=h_mult)
        c_nn, c_new = certify_verdict(rows_nn), certify_verdict(rows_new)
        k_nn = sum(r["certified"] for r in rows_nn)
        k_new = sum(r["certified"] for r in rows_new)
        out.append(dict(experiment=label(p), seed=meta.get("seed"),
                        path=os.path.relpath(p, EXP),
                        h_mode=h_mode, h_mult=h_mult,
                        n_comp=len(rows_nn),
                        n_cert_nn=k_nn, n_cert_new=k_new,
                        C3_nn=c_nn["certified"], C3_new=c_new["certified"],
                        verdict_nn=c_nn["verdict"],
                        verdict_new=c_new["verdict"]))
        print(f"{label(p):26} {str(meta.get('seed')):>5} "
              f"{k_nn:>4}/{len(rows_nn):<4} {k_new:>4}/{len(rows_new):<5} "
              f"{str(c_nn['certified']):>7} {str(c_new['certified']):>8}")
    print("\nNOTE: C3 above is the margin condition only.  A trial's overall "
          "certificate also needs C1 (eps <= eps*) and C2 (eta outliers <= 1) "
          "from tab:pertrial; trials currently failing those stay uncertified "
          "regardless of C3.")
    return out


# ----------------------------------------------------------------------------

def task_adjacency(paths, m0, knn=10):
    """N1: fragments (adjacent to a retained component) vs dropped edges.

    Loads only points.npy and assignments/ -- no TensorFlow required."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    def split_adaptive(pts, idx, scale=3.0):
        n = len(idx)
        if n < 2:
            return [np.asarray(idx)]
        P = pts[idx]
        d, _ = cKDTree(P).query(P, k=2)
        eps = scale * float(np.median(d[:, 1]))
        if not np.isfinite(eps) or eps <= 0:
            return [np.asarray(idx)]
        pairs = np.array(list(cKDTree(P).query_pairs(eps)))
        if len(pairs) == 0:
            return [np.asarray([p]) for p in idx]
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
        _, lab = connected_components(g, directed=False)
        return [np.asarray(idx)[lab == c] for c in range(lab.max() + 1)]

    def knn_components(pts, idx, k):
        n = len(idx)
        kk = min(k + 1, n)
        _, nb = cKDTree(pts[idx]).query(pts[idx], k=kk)
        rows = np.repeat(np.arange(n), kk - 1)
        g = coo_matrix((np.ones(len(rows)), (rows, nb[:, 1:].ravel())),
                       shape=(n, n))
        _, lab = connected_components(g, directed=False)
        return lab

    out = []
    print(f"{'experiment':26} {'seed':>5} {'frags':>6} {'adjacent':>9} "
          f"{'dropped edges':>14}")
    print("-" * 68)
    for p in paths:
        meta = json.load(open(os.path.join(p, "meta.json")))
        pts = np.load(os.path.join(p, "points.npy"))
        asg = [np.load(os.path.join(p, "assignments", f"chart_{i}.npy"))
               for i in range(meta["n_charts"])]
        n_frag = n_adj = n_edge = n_edge_pts = 0
        for i, j in combinations(range(meta["n_charts"]), 2):
            ov = np.intersect1d(asg[i], asg[j])
            if len(ov) == 0:
                continue
            comps = split_adaptive(pts, ov)
            ret = [c for c in comps if len(c) >= m0]
            disc = [c for c in comps if len(c) < m0]
            if not ret:
                n_edge += 1
                n_edge_pts += len(ov)
                continue
            if not disc:
                continue
            lab = knn_components(pts, ov, knn)
            pos = {int(q): t for t, q in enumerate(ov)}
            ret_labs = {lab[pos[int(q)]] for c in ret for q in c}
            for c in disc:
                n_frag += 1
                if any(lab[pos[int(q)]] in ret_labs for q in c):
                    n_adj += 1
        out.append(dict(experiment=label(p), seed=meta.get("seed"),
                        path=os.path.relpath(p, EXP), m0=m0, knn=knn,
                        fragments=n_frag, fragments_adjacent=n_adj,
                        dropped_edges=n_edge, dropped_edge_points=n_edge_pts))
        print(f"{label(p):26} {str(meta.get('seed')):>5} {n_frag:>6} "
              f"{n_adj:>9} {n_edge:>14}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["discarded", "fixed-radius", "margins",
                             "adjacency", "all"])
    ap.add_argument("--glob", default="*")
    ap.add_argument("--m0", type=int, default=5)
    ap.add_argument("--eps-fixed", type=float, default=1.0,
                    help="fixed splitting radius (pipeline eps_cluster)")
    ap.add_argument("--h-mode", default="local-max",
                    choices=["nn", "local-max", "mult"])
    ap.add_argument("--h-mult", type=float, default=2.0)
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings of run paths to skip")
    ap.add_argument("--canonical",
                    default="results_review_todos/canonical_paths.json",
                    help="take atlas paths from this json (the 65 trials of "
                         "tab:pertrial, resolved the same way "
                         "make_master_table.py resolves them and validated "
                         "against the table's eps column); pass '' to glob "
                         "every saved atlas instead")
    a = ap.parse_args()

    canon = os.path.join(HERE, a.canonical) if a.canonical else None
    if canon and os.path.isfile(canon):
        rel = [r["path"] for r in json.load(open(canon))]
        paths = [os.path.join(EXP, r) for r in rel]
        paths = [p for p in paths if os.path.isdir(p)]
        missing = len(rel) - len(paths)
        if missing:
            print(f"warning: {missing} canonical atlas dirs not found on disk")
        if a.glob != "*":
            import fnmatch
            paths = [p for p in paths
                     if fnmatch.fnmatch(os.path.relpath(p, EXP), a.glob)
                     or fnmatch.fnmatch(os.path.basename(p), a.glob)]
    else:
        paths = find_atlases(a.glob)
    if a.exclude:
        bad = [x for x in a.exclude.split(",") if x]
        paths = [p for p in paths if not any(b in p for b in bad)]
    if not paths:
        print("no saved atlases found (trained atlases are gitignored).")
        return

    os.makedirs(os.path.join(HERE, "results_review_todos"), exist_ok=True)
    tasks = (["discarded", "fixed-radius", "margins", "adjacency"]
             if a.task == "all" else [a.task])
    for t in tasks:
        if t == "discarded":
            rows = task_discarded(paths, a.m0)
        elif t == "fixed-radius":
            rows = task_fixed_radius(paths, a.m0, a.eps_fixed)
        elif t == "adjacency":
            rows = task_adjacency(paths, a.m0)
        else:
            rows = task_margins(paths, a.m0, a.h_mode, a.h_mult)
        f = os.path.join(HERE, "results_review_todos", f"{t}.json")
        json.dump(rows, open(f, "w"), indent=1)
        print(f"\nwrote {os.path.relpath(f, EXP)}\n")


if __name__ == "__main__":
    main()
