"""Train a two-chart atlas on the Mobius band and draw its signed nerve.

The smallest non-orientable example there is, and the one where the figure
explains the method by itself: two charts, one overlap falling into two
components, one of which reverses orientation. No assignment of +-1 to the
two charts can match both signs -- that failure *is* the non-orientability,
and it is what the red and blue edges in the middle panel show.

    python plot_mobius_atlas.py            # writes mobius_atlas.png

Takes a couple of minutes on a laptop CPU.
"""

import numpy as np

from atlasae import (AtlasAutoencoder, fast_fit, check_orientability,
                     sign_constancy_report, certify_verdict,
                     plot_atlas, plot_chart_transitions)


def mobius_band(n_theta=400, n_width=8, half_width=0.3, seed=0):
    """Points on a Mobius band embedded in R^3."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_theta * n_width)
    w = rng.uniform(-half_width, half_width, n_theta * n_width)
    x = (1 + w * np.cos(theta / 2)) * np.cos(theta)
    y = (1 + w * np.cos(theta / 2)) * np.sin(theta)
    z = w * np.sin(theta / 2)
    return np.stack([x, y, z], axis=1), theta


def two_arc_cover(theta, overlap=0.35):
    """Two overlapping arcs in the theta circle; their overlap has two pieces."""
    a = np.where((theta < np.pi + overlap) | (theta > 2 * np.pi - overlap))[0]
    b = np.where((theta > np.pi - overlap) | (theta < overlap))[0]
    return [a, b]


def main():
    pts, theta = mobius_band()
    assign = two_arc_cover(theta)
    print(f"{len(pts)} points, charts of size "
          f"{[len(a) for a in assign]}, overlap "
          f"{len(np.intersect1d(*assign))} points")

    system = AtlasAutoencoder(data=pts, n_charts=2, subset_assignments=assign,
                              latent_dim=2, hidden_dims=[32, 16])
    fast_fit(system, epochs=3000, lambda_jac=0.01, lambda_diff=0.01)

    # the signs, and whether they are certified constant between the samples
    report = sign_constancy_report(system, min_points=5)
    verdict = certify_verdict(report["rows"])
    print(f"verdict: {verdict['verdict']} "
          f"({'certified' if verdict['certified'] else 'uncertified'}: "
          f"{verdict['reason']})")
    for r in report["rows"]:
        j, i = r["pair"]
        print(f"  overlap component {r['comp']} of charts {i}-{j}: "
              f"omega = {r['omega']:+d}, margin = {r['margin']:.2f}, "
              f"{r['n']} points")

    # reuse the report so the figure does not recompute the Jacobians
    fig = plot_atlas(system, rows=report["rows"])
    fig.savefig("mobius_atlas.png", dpi=200, bbox_inches="tight")
    print("wrote mobius_atlas.png")

    fig = plot_chart_transitions(system, rows=report["rows"])
    fig.savefig("mobius_nerve.png", dpi=200, bbox_inches="tight")
    print("wrote mobius_nerve.png")

    # the same conclusion through the pipeline's own entry point
    result = check_orientability(system, pts, assign, eps_cluster=1.0,
                                 min_points=5)
    print(f"check_orientability: is_orientable = {result['is_orientable']}")


if __name__ == "__main__":
    main()
