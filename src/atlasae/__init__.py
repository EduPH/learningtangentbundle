"""
atlasae — autoencoder atlases on data manifolds.

Core objects:
    AtlasAutoencoder        multi-chart autoencoder with theoretical metrics
    fast_fit / train_until_certified   compiled training + certificate protocol
    check_orientability     sign-cocycle w1 detection (coboundary test)
    compare_eta / pca_tangent_frames   direct tangent-restricted eta measurement
    plot_atlas / plot_chart_transitions   latent charts and the signed nerve
"""

from atlasae.atlasautoencoder import (
    AtlasAutoencoder,
    LocalAutoencoder,
    plot_all_transitions,
)
from atlasae.fast_training import fast_fit, train_until_certified
from atlasae.orientability import check_orientability, verify_cocycle_condition
from atlasae.eta_true import (
    compare_eta,
    compute_eta_true_pointwise,
    compute_eta_lat_pointwise,
    pca_tangent_frames,
    frame_alignment,
    sphere_tangent_frames,
    mobius_param,
    mobius_tangent_frames,
    klein_param,
    klein_tangent_frames,
    numeric_tangent_frames,
)
from atlasae.sphere_good_cover import tetrahedral_cover_S2

__version__ = "0.1.0"

__all__ = [
    "AtlasAutoencoder", "LocalAutoencoder", "plot_all_transitions",
    "fast_fit", "train_until_certified",
    "check_orientability", "verify_cocycle_condition",
    "compare_eta", "compute_eta_true_pointwise", "compute_eta_lat_pointwise",
    "pca_tangent_frames", "frame_alignment",
    "sphere_tangent_frames", "mobius_param", "mobius_tangent_frames",
    "klein_param", "klein_tangent_frames", "numeric_tangent_frames",
    "tetrahedral_cover_S2",
    "gauge_report", "pairwise_logdet", "solve_gauge",
    "sign_constancy_report", "overlap_sign_constancy", "certify_verdict",
    "save_atlas", "load_atlas",
    "plot_atlas", "plot_chart_transitions",
]

from atlasae.gauge import gauge_report, pairwise_logdet, solve_gauge
from atlasae.sign_constancy import sign_constancy_report, overlap_sign_constancy
from atlasae.verdict_certificate import certify_verdict
from atlasae.persistence import save_atlas, load_atlas
from atlasae.viz import plot_atlas, plot_chart_transitions
