"""Minimal end-to-end example: certify the orientability of S^2 from a sample.

Trains a four-chart autoencoder atlas on a point cloud drawn from the unit
sphere, measures the tangent-restricted differential error entering the
stability theory, and runs the sign-cocycle orientability test.

Expected output: is_orientable=True, with eta_pca well below 1.
Runtime: a few minutes on CPU.
"""

import numpy as np

from atlasae import (
    AtlasAutoencoder,
    fast_fit,
    check_orientability,
    pca_tangent_frames,
    compare_eta,
    tetrahedral_cover_S2,
)

rng = np.random.default_rng(42)

# a point cloud on the 2-sphere, covered by four overlapping caps
pts = rng.standard_normal((1000, 3))
pts /= np.linalg.norm(pts, axis=1, keepdims=True)
assign = tetrahedral_cover_S2(epsilon=0.3).get_assignments(pts)

system = AtlasAutoencoder(data=pts, n_charts=4, subset_assignments=assign,
                          latent_dim=2, hidden_dims=[32, 16])
fast_fit(system, epochs=2000, lambda_jac=0.01, lambda_diff=0.01)

# hypotheses of the stability theory, measured with local-PCA tangent frames
frames = pca_tangent_frames(pts, d=2, k=25)
eta = compare_eta(system, pts, assign, frames_pca=frames)

# sign cocycle + coboundary (two-colouring) test
result = check_orientability(system, pts, assign, eps_cluster=1.0, min_points=5)

print(f"eta_pca       = {eta['eta_pca']:.3f}   (binding hypothesis: < 1)")
print(f"is_orientable = {result['is_orientable']}")
