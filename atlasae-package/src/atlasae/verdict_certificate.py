"""
Certify an orientability verdict from the sign-constancy margins.

The margin of prop:sign-constancy certifies one overlap at a time: it proves
that omega_ji is constant on that overlap, so the value read at a sample point
is the value everywhere.  Turning a set of certified overlaps into a certified
*verdict* is not symmetric in the two answers, and that asymmetry is what this
module implements (prop:odd-cycle in the supplement).

    non-orientable   needs ONE certified odd cycle.  If the overlaps of a cycle
                     are all certified and their signs multiply to -1, then no
                     assignment nu_i with omega_ji = nu_j nu_i can exist on that
                     cycle -- each nu appears twice, so the product would be +1.
                     The other overlaps are irrelevant: they cannot repair a
                     contradiction.

    orientable       needs EVERY overlap certified.  A single uncertified
                     overlap could hide a sign flip, and one hidden flip turns
                     an even cycle odd, so a 2-colouring of the certified
                     subgraph proves nothing about the whole nerve.

The consequence is that the strict "every overlap" reading of the certificate
is necessary for one verdict and needlessly strong for the other -- which
matters, because every experiment that falls short of full coverage happens to
be non-orientable.

Usage:
    from atlasae import certify_verdict
    c = certify_verdict(rows)          # rows from sign_constancy_report
    c['verdict']                       # 'non-orientable' | 'orientable' | None
    c['certified']                     # bool
    c['witness']                       # the odd cycle, when there is one
"""

import collections

__all__ = ["certify_verdict"]


def _edges(rows, certified_only):
    """Signed edges [(i, j, omega)], one per overlap *component*.

    This is a multigraph, not a simple graph: two charts may meet in several
    components carrying different signs.  That is not a corner case but the
    smallest non-orientable example there is -- the Mobius band has two charts
    whose single overlap falls into two components, one of which reverses
    orientation.  Keying edges by chart pair would collapse the two and lose
    the flip entirely.
    """
    out = []
    for r in rows:
        if certified_only and not r["certified"]:
            continue
        if "vi" in r and "vj" in r:
            u, v = tuple(r["vi"]), tuple(r["vj"])
        else:                                   # pre-component rows
            j, i = r["pair"]
            u, v = (min(i, j), 0), (max(i, j), 0)
        out.append((min(u, v), max(u, v), int(r["omega"])))
    return out, 0


def _find_odd_cycle(edges):
    """A cycle whose signs multiply to -1, or None.

    Two-colour each component by BFS, propagating nu_j = omega_ji * nu_i.  An
    edge joining two already-coloured vertices with the wrong parity closes an
    odd cycle; the cycle itself is recovered from the BFS tree paths.  A
    parallel edge with the opposite sign is the degenerate case, a 2-cycle
    between the same two charts, and is picked up by the same test.
    """
    adj = collections.defaultdict(list)
    for e, (i, j, w) in enumerate(edges):
        adj[i].append((j, w, e))
        adj[j].append((i, w, e))

    colour, parent = {}, {}
    for src in list(adj):
        if src in colour:
            continue
        colour[src] = 1
        parent[src] = (None, None)
        queue = collections.deque([src])
        while queue:
            u = queue.popleft()
            for v, w, e in adj[u]:
                if v not in colour:
                    colour[v] = colour[u] * w
                    parent[v] = (u, e)
                    queue.append(v)
                elif e != parent[u][1] and colour[v] != colour[u] * w:
                    # u--v closes an odd cycle; walk both to their common root
                    def path(x):
                        out = []
                        while x is not None:
                            out.append(x)
                            x = parent[x][0]
                        return out
                    pu, pv = path(u), path(v)
                    sv = set(pv)
                    meet = next(x for x in pu if x in sv)
                    return pu[:pu.index(meet) + 1] + pv[:pv.index(meet)][::-1]
    return None


def certify_verdict(rows):
    """Decide, and certify, the orientability verdict of one atlas.

    `rows` is sign_constancy_report(...)['rows'].
    """
    all_edges, _ = _edges(rows, certified_only=False)
    cert_edges, dropped = _edges(rows, certified_only=True)

    n_total = len(all_edges)
    n_cert = len(cert_edges)

    # 1. a certified odd cycle proves non-orientability outright
    cycle = _find_odd_cycle(cert_edges)
    if cycle is not None:
        return dict(verdict="non-orientable", certified=True,
                    reason="certified odd cycle",
                    witness=cycle, cycle_length=len(cycle),
                    n_overlaps=n_total, n_certified=n_cert,
                    n_sign_conflicts=dropped)

    # 2. otherwise an orientable verdict needs every overlap certified
    full = n_cert == n_total and n_total > 0
    uncertified_cycle = _find_odd_cycle(all_edges)
    verdict = ("non-orientable" if uncertified_cycle is not None
               else "orientable" if n_total else None)

    if verdict == "orientable" and full:
        return dict(verdict="orientable", certified=True,
                    reason="every overlap certified, 2-colouring exists",
                    witness=None, n_overlaps=n_total, n_certified=n_cert,
                    n_sign_conflicts=dropped)

    reason = ("odd cycle exists but is not fully certified"
              if uncertified_cycle is not None else
              f"{n_total - n_cert} of {n_total} overlaps uncertified")
    return dict(verdict=verdict, certified=False, reason=reason,
                witness=uncertified_cycle,
                n_overlaps=n_total, n_certified=n_cert,
                n_sign_conflicts=dropped)
