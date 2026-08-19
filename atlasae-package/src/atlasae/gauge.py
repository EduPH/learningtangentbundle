"""
Latent gauge fixing: raising the non-degeneracy gap without touching the invariant.

Rescaling the latent coordinates of chart i by lambda_i > 0,

    E_i -> lambda_i E_i,        D_i -> D_i( . / lambda_i),

leaves the reconstruction map Phi_i = D_i o E_i -- and therefore epsilon and eta --
exactly unchanged, but acts on the transition Jacobians by

    g_ji -> (lambda_j / lambda_i) g_ji,
    det g_ji -> (lambda_j / lambda_i)^d det g_ji.

Since lambda_j / lambda_i > 0 the sign cocycle omega_ji = sign det g_ji is invariant,
so the orientability verdict is unaffected.  The non-degeneracy gap
delta = min_{i,j} inf_x |det g_ji(x)| is NOT invariant, and can be optimised.

Writing a_ji = log inf_x |det g_ji(x)| and u_i = d log lambda_i, the problem is

    maximise   t
    subject to a_ji + u_j - u_i >= t   for every directed nerve edge (i,j),

a maximum-of-minimum problem on the nerve graph whose optimum, by Karp's theorem,
is the minimum mean weight over directed cycles.

Two consequences are worth recording.

1.  delta <= 1 always.  The two-cycle (i,j),(j,i) has a_ji + a_ij = 0 exactly
    (Proposition: transition properties), so its mean is 0 and t* <= 0.

2.  For an *exact* atlas, delta can be gauge-fixed to exactly 1.  The cocycle
    condition g_ki = g_kj g_ji makes a a coboundary, a_ji = v_j - v_i with
    v_i = -log|det ...|, so every cycle mean vanishes and t* = 0.  In the
    approximate setting the cocycle holds only to O(epsilon), so the attainable
    ceiling is 1 - O(epsilon), the deficit being the holonomy of a around the
    nerve.

So a small measured delta is largely an artefact of how the latent coordinates
happen to be scaled by training, not an intrinsic property of the atlas.

Usage:
    from atlasae.gauge import gauge_report
    r = gauge_report(system)          # delta before/after, holonomy, lambdas
"""

from itertools import combinations, permutations

import numpy as np

__all__ = ["pairwise_logdet", "solve_gauge", "gauge_report", "apply_gauge_to_delta"]


def pairwise_logdet(system, min_points=5):
    """{(j,i): log inf_x |det g_ji(x)|} over every non-empty overlap.

    Uses the system's own overlap bookkeeping, so it sees exactly the overlaps
    the certificate sees.  Overlaps with fewer than min_points samples are
    skipped, matching the m_0 truncation of the pipeline.
    """
    out = {}
    for i, j in permutations(range(system.n_charts), 2):
        if (i, j) not in getattr(system, "pairwise_overlaps", {}):
            continue
        if len(system.pairwise_overlaps[(i, j)]) < min_points:
            continue
        _, jac = system.compute_transition_jacobians(i, j)
        if jac is None or len(jac) == 0:
            continue
        det = np.abs(np.linalg.det(np.asarray(jac)))
        m = float(det.min())
        out[(j, i)] = float(np.log(m)) if m > 0 else -np.inf
    return out


def _cycle_means(a, n, max_len=3):
    """Mean weights of directed cycles up to max_len (Karp optimum for small nerves)."""
    means = []
    for i, j in combinations(range(n), 2):
        if (j, i) in a and (i, j) in a:
            means.append((a[(j, i)] + a[(i, j)]) / 2.0)
    if max_len >= 3:
        for i, j, k in permutations(range(n), 3):
            if (j, i) in a and (k, j) in a and (i, k) in a:
                means.append((a[(j, i)] + a[(k, j)] + a[(i, k)]) / 3.0)
    return means


def solve_gauge(a, n, method="lp"):
    """Optimal u maximising min_{(i,j)} (a_ji + u_j - u_i).

    method="lp"  exact max-min via linear programming (needs scipy)
    method="ls"  least-squares/Laplacian fit, faster and near-optimal when the
                 holonomy is small (which is the near-exact regime)

    Returns (u, t_star) with u a dict chart -> u_i.
    """
    edges = [(j, i) for (j, i) in a if np.isfinite(a[(j, i)])]
    if not edges:
        return {i: 0.0 for i in range(n)}, -np.inf

    if method == "ls":
        # minimise sum (a_ji + u_j - u_i)^2  -- a graph Laplacian least squares
        A = np.zeros((len(edges), n))
        b = np.zeros(len(edges))
        for r, (j, i) in enumerate(edges):
            A[r, j] += 1.0
            A[r, i] -= 1.0
            b[r] = -a[(j, i)]
        u, *_ = np.linalg.lstsq(A, b, rcond=None)
        u -= u.mean()
        t = min(a[(j, i)] + u[j] - u[i] for (j, i) in edges)
        return {i: float(u[i]) for i in range(n)}, float(t)

    from scipy.optimize import linprog
    # variables [u_0..u_{n-1}, t];  maximise t  =>  minimise -t
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub, b_ub = [], []
    for (j, i) in edges:
        row = np.zeros(n + 1)
        row[j] -= 1.0
        row[i] += 1.0
        row[-1] += 1.0            # t - u_j + u_i <= a_ji
        A_ub.append(row)
        b_ub.append(a[(j, i)])
    A_eq = np.zeros((1, n + 1))   # gauge freedom: fix sum u_i = 0
    A_eq[0, :n] = 1.0
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=[0.0], bounds=[(None, None)] * (n + 1),
                  method="highs")
    if not res.success:
        return solve_gauge(a, n, method="ls")
    u = res.x[:n]
    return {i: float(u[i]) for i in range(n)}, float(res.x[-1])


def apply_gauge_to_delta(a, u):
    """delta after applying the gauge u."""
    vals = [a[(j, i)] + u[j] - u[i] for (j, i) in a if np.isfinite(a[(j, i)])]
    return float(np.exp(min(vals))) if vals else 0.0


def gauge_report(system, min_points=5, d=2, method="lp"):
    """delta before and after optimal latent rescaling, plus the holonomy.

    epsilon and eta are unchanged by the gauge, and so is the verdict; only
    delta moves.  'ceiling' is the best delta any gauge can achieve, and
    1 - ceiling measures how far the linearised cocycle is from exact.
    """
    a = pairwise_logdet(system, min_points=min_points)
    n = system.n_charts
    finite = [v for v in a.values() if np.isfinite(v)]
    if not finite:
        return dict(delta_before=0.0, delta_after=0.0, ceiling=0.0,
                    holonomy=np.inf, n_overlaps=0)
    before = float(np.exp(min(finite)))
    u, t = solve_gauge(a, n, method=method)
    after = apply_gauge_to_delta(a, u)
    means = _cycle_means(a, n)
    ceiling = float(np.exp(min(means))) if means else 1.0
    return dict(
        delta_before=before,
        delta_after=after,
        ceiling=ceiling,
        holonomy=float(-min(means)) if means else 0.0,
        gain=after / before if before > 0 else np.inf,
        lambdas={i: float(np.exp(u[i] / d)) for i in u},
        n_overlaps=len(a),
    )
