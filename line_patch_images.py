"""
Line Patch Images - ℝP² Orientability Experiment
=================================================

This script runs multiple trials of the atlas autoencoder on line patch images
and collects metrics for inclusion in the paper. The line patch space is
homeomorphic to ℝP² (real projective plane), which is non-orientable.

Mathematical background:
- Line patches are small image patches containing line segments at various angles
- The space of lines through the origin in ℝ² is ℝP¹ ≅ S¹
- Adding offsets gives a space homeomorphic to ℝP²
- ℝP² is non-orientable: w₁(TℝP²) ≠ 0
- H₂(ℝP²; ℤ/2) = ℤ/2 (detected by persistent homology with ℤ/2 coefficients)

This is a key experiment because:
1. It demonstrates detection on REAL IMAGE DATA (not synthetic manifolds)
2. ℝP² cannot be embedded in ℝ³, so we must work in higher dimensions
3. The non-orientability is subtle and requires careful cover construction

Outputs:
- JSON file with all metrics from all trials
- Summary statistics (mean ± std) for paper tables
- Figures saved to disk
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from atlasautoencoder import AtlasAutoencoder, plot_all_transitions
from orientability import check_orientability

# DREiMaC for line patch generation and projective coordinate analysis
from dreimac import (
    ProjectiveCoords, GeometryUtils, GeometryExamples,
    PlotUtils, ProjectiveMapUtils
)

# For persistent homology verification
from ripser import ripser
from persim import plot_diagrams


# ============================================================
# Configuration
# ============================================================

EXPERIMENT_CONFIG = {
    'manifold': 'RP2',
    'manifold_name': 'ℝP² (Line Patches)',
    'true_orientable': False,  # ℝP² is NON-orientable
    'intrinsic_dim': 2,
    'ambient_dim': 100,  # 10x10 image patches = 100 dimensions
    
    # Line patch generation
    'patch_dim': 10,        # 10x10 pixel patches
    'n_angles': 75,         # Number of angle samples
    'n_offsets': 75,        # Number of offset samples
    'sigma': 0.25,          # Line width parameter
    
    # Cover: landmark-based with projective coordinates
    'cover_type': 'landmark_projective',
    'n_charts': 10,
    'n_landmarks': 300,
    'n_neighbors_geodesic': 20,
    'threshold_percentile': 20,
    
    # Architecture
    'latent_dim': 2,
    'hidden_dims': [64, 32],
    
    # Training
    'epochs': 5000,
    'batch_size': 64,
    'lambda_jac': 0.0,
    'lambda_cocycle': 0.0,
    
    # Orientability detection
    'eps_cluster': 1.0,
    'min_points': 100,
    'eps_det': 1e-6,
    
    # Experiment
    'n_trials': 5,
    'random_seed_base': 42,
}


# ============================================================
# Data generation
# ============================================================

def generate_line_patches(
    patch_dim: int = 10,
    n_angles: int = 75,
    n_offsets: int = 75,
    sigma: float = 0.25,
    seed: int = None
) -> np.ndarray:
    """
    Generate line patch images.
    
    Each patch is a small grayscale image containing a line segment.
    The space of all such patches (parameterized by angle and offset)
    is homeomorphic to ℝP² because:
    - Angles θ and θ+π give the same line (antipodal identification)
    - This creates the ℝP² topology
    
    Args:
        patch_dim: Size of each patch (patch_dim × patch_dim pixels)
        n_angles: Number of angle samples
        n_offsets: Number of offset samples  
        sigma: Line width (Gaussian blur parameter)
        seed: Random seed
        
    Returns:
        points: Array of shape (n_angles * n_offsets, patch_dim²)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use DREiMaC's line patch generator
    points = GeometryExamples.line_patches(
        dim=patch_dim,
        n_angles=n_angles,
        n_offsets=n_offsets,
        sigma=sigma
    )
    
    return points


def create_projective_cover(
    X: np.ndarray,
    n_charts: int,
    n_landmarks: int = 300,
    n_neighbors: int = 20,
    threshold_percentile: float = 20
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a cover suitable for detecting ℝP² non-orientability.
    
    This uses DREiMaC's projective coordinate machinery to:
    1. Compute geodesic distances via landmark approximation
    2. Extract projective coordinates that respect the ℝP² topology
    3. Create charts as geodesic balls around landmarks
    
    Args:
        X: Data points (line patches)
        n_charts: Number of charts
        n_landmarks: Number of landmarks for geodesic computation
        n_neighbors: k for k-NN graph in geodesic approximation
        threshold_percentile: Percentile of distances for chart radius
        
    Returns:
        subset_assignments: List of index arrays for each chart
        X_reordered: Points reordered by landmark computation
        proj_coords: Projective coordinates (for visualization)
        pointcloud_permutation: Permutation from original to reordered
        dist_mat: Geodesic distance matrix to landmarks
    """
    # Compute geodesic distances to landmarks
    dist_mat, pointcloud_permutation = GeometryUtils.landmark_geodesic_distance(
        X, n_landmarks, n_neighbors
    )
    
    # Reorder data according to landmark permutation
    X_reordered = X[pointcloud_permutation]
    
    # Compute projective coordinates for visualization
    pc = ProjectiveCoords(dist_mat, n_landmarks=n_landmarks, distance_matrix=True)
    proj_coords = pc.get_coordinates(proj_dim=2, perc=0.8, cocycle_idx=0)
    
    # Create charts from first n_charts landmarks
    # Use percentile of distances as epsilon
    epsilon = np.percentile(dist_mat[1:n_charts+1], threshold_percentile)
    
    subset_assignments = []
    for i in range(n_charts):
        # Points within epsilon of landmark i+1 (skip landmark 0)
        cluster_indices = np.nonzero(dist_mat[i + 1] < epsilon)[0]
        subset_assignments.append(cluster_indices)
    
    return subset_assignments, X_reordered, proj_coords, pointcloud_permutation, dist_mat


def verify_rp2_topology(X: np.ndarray, save_path: str = None):
    """
    Verify that the line patch data has ℝP² topology using persistent homology.
    
    ℝP² has distinctive homology:
    - H₀ = ℤ (connected)
    - H₁ = ℤ/2 (non-orientable)
    - H₂ = 0 with ℤ coefficients, but ℤ/2 with ℤ/2 coefficients
    
    The key signature is H₂ with ℤ/2 coefficients (torsion).
    """
    print("\nVerifying ℝP² topology via persistent homology...")
    
    # Subsample for computational efficiency
    n_sample = min(500, len(X))
    indices = np.random.choice(len(X), size=n_sample, replace=False)
    X_sample = X[indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, prime in enumerate([2, 3]):
        pd = ripser(X_sample, coeff=prime, maxdim=2)["dgms"]
        
        plt.sca(axes[i])
        plot_diagrams(pd)
        axes[i].set_title(f"$\\mathbb{{Z}}/{prime}\\mathbb{{Z}}$ coefficients")
    
    plt.suptitle("Persistent Homology (ℝP² has H₂ with ℤ/2 coefficients only)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================
# Single trial runner
# ============================================================

def run_single_trial(
    config: Dict,
    trial_idx: int,
    save_dir: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single trial of the experiment.
    """
    seed = config['random_seed_base'] + trial_idx
    np.random.seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRIAL {trial_idx + 1}/{config['n_trials']} (seed={seed})")
        print(f"{'='*60}")
    
    # Generate data
    X = generate_line_patches(
        patch_dim=config['patch_dim'],
        n_angles=config['n_angles'],
        n_offsets=config['n_offsets'],
        sigma=config['sigma'],
        seed=seed
    )
    
    if verbose:
        print(f"\nGenerated {len(X)} line patches")
        print(f"Patch dimension: {config['patch_dim']}×{config['patch_dim']} = {X.shape[1]}")
    
    # Create cover
    if verbose:
        print("\nCreating landmark-based projective cover...")
    
    subset_assignments, X_ordered, proj_coords, perm, dist_mat = create_projective_cover(
        X,
        n_charts=config['n_charts'],
        n_landmarks=config['n_landmarks'],
        n_neighbors=config['n_neighbors_geodesic'],
        threshold_percentile=config['threshold_percentile']
    )
    n_charts = len(subset_assignments)
    
    # Report cover statistics
    if verbose:
        print(f"Cover statistics:")
        for i, indices in enumerate(subset_assignments):
            print(f"  Chart {i}: {len(indices)} points")
        
        # Count overlaps
        n_overlaps = 0
        for i in range(n_charts):
            for j in range(i + 1, n_charts):
                overlap = len(np.intersect1d(subset_assignments[i], subset_assignments[j]))
                if overlap > 0:
                    n_overlaps += 1
        print(f"  Pairwise overlaps: {n_overlaps}")
    
    # Build atlas autoencoder
    import tensorflow as tf
    tf.random.set_seed(seed)
    
    system = AtlasAutoencoder(
        data=X_ordered,
        n_charts=n_charts,
        subset_assignments=subset_assignments,
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims']
    )
    
    # Train
    system.fit(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lambda_jac=config['lambda_jac'],
        lambda_cocycle=config['lambda_cocycle'],
        verbose=verbose
    )
    
    # Collect metrics
    metrics = collect_metrics(system, X_ordered, subset_assignments, config, trial_idx)
    
    # Run orientability detection
    orient_result = check_orientability(
        system=system,
        points=X_ordered,
        subset_assignments=subset_assignments,
        eps_cluster=config['eps_cluster'],
        min_points=config['min_points'],
        eps_det=config.get('eps_det', 1e-6),
        verbose=verbose
    )
    
    # Add orientability results to metrics
    metrics['orientability'] = {
        'detected_orientable': orient_result['is_orientable'],
        'correct_detection': orient_result['is_orientable'] == config['true_orientable'],
        'cocycle_verified': orient_result['cocycle_verified'],
        'coboundary_passed': orient_result['coboundary_passed'],
        'has_mixed_within': orient_result.get('has_mixed_within', False),
        'has_different_across': orient_result.get('has_different_across', False),
        'n_chart_components': len(orient_result['chart_components']),
        'n_overlaps': len(orient_result['signs']),
    }
    
    # Record sign statistics
    if orient_result['signs']:
        metrics['orientability']['sign_summary'] = summarize_signs(orient_result['signs'])
    
    # Save figures if requested
    if save_dir is not None:
        save_trial_figures(system, X_ordered, subset_assignments, proj_coords, 
                          perm, trial_idx, save_dir)
    
    return metrics


def summarize_signs(signs: Dict) -> Dict:
    """Summarize the sign distribution across overlaps."""
    n_positive = sum(1 for s in signs.values() if s == 1)
    n_negative = sum(1 for s in signs.values() if s == -1)
    
    # Check for chart pairs with multiple overlap components
    chart_pairs = {}
    for key, sign in signs.items():
        chart_i, ci, chart_j, cj, _ = key
        pair_key = ((chart_i, ci), (chart_j, cj))
        if pair_key not in chart_pairs:
            chart_pairs[pair_key] = []
        chart_pairs[pair_key].append(sign)
    
    n_inconsistent_pairs = sum(
        1 for signs_list in chart_pairs.values() 
        if len(set(signs_list)) > 1
    )
    
    return {
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_total': len(signs),
        'n_inconsistent_pairs': n_inconsistent_pairs,
    }


def collect_metrics(
    system: AtlasAutoencoder,
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    config: Dict,
    trial_idx: int
) -> Dict[str, Any]:
    """
    Collect all theoretical metrics from a trained system.
    """
    import tensorflow as tf
    
    metrics = {
        'trial_idx': trial_idx,
        'seed': config['random_seed_base'] + trial_idx,
    }
    
    # Per-chart metrics
    varepsilon_list = []
    varepsilon_mean_list = []
    eta_list = []
    sigma_min_list = []
    
    for i in range(system.n_charts):
        x = tf.constant(points[subset_assignments[i]], dtype=tf.float32)
        
        # ε: reconstruction error
        varepsilon_list.append(float(system.compute_varepsilon(i, x).numpy()))
        varepsilon_mean_list.append(float(system.compute_varepsilon_mean(i, x).numpy()))
        
        # η: differential error
        eta_list.append(float(system.compute_eta_tangent(i, x).numpy()))
        
        # σ_min: encoder regularity
        sigma_min_list.append(float(system.compute_encoder_sigma_min(i, x).numpy()))
    
    metrics['per_chart'] = {
        'varepsilon': varepsilon_list,
        'varepsilon_mean': varepsilon_mean_list,
        'eta': eta_list,
        'sigma_min_enc': sigma_min_list,
    }
    
    # Global metrics
    metrics['theoretical'] = {
        'varepsilon': max(varepsilon_list),
        'varepsilon_mean': np.mean(varepsilon_mean_list),
        'eta': max(eta_list),
        'delta': float(system.compute_delta().numpy()),
        'delta_mean': float(system.compute_delta_mean().numpy()),
        'cocycle_error': float(system.compute_cocycle_error().numpy()),
        'sigma_min_enc': min(sigma_min_list),
    }
    
    # Cover statistics
    metrics['cover'] = {
        'n_charts': system.n_charts,
        'chart_sizes': [len(a) for a in subset_assignments],
        'n_pairwise_overlaps': len([k for k in system.pairwise_overlaps.keys() if k[0] < k[1]]),
        'n_triple_overlaps': len(set(tuple(sorted(k)) for k in system.triple_overlaps.keys())),
    }
    
    return metrics


def save_trial_figures(
    system: AtlasAutoencoder,
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    proj_coords: np.ndarray,
    perm: np.ndarray,
    trial_idx: int,
    save_dir: str
):
    """Save visualization figures for a trial."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize some line patches
    fig = plt.figure(figsize=(10, 10))
    n_show = min(64, len(points))
    indices = np.random.choice(len(points), size=n_show, replace=False)
    
    patch_dim = int(np.sqrt(points.shape[1]))
    for i, idx in enumerate(indices[:64]):
        ax = fig.add_subplot(8, 8, i + 1)
        ax.imshow(points[idx].reshape(patch_dim, patch_dim), cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Sample Line Patches')
    fig.savefig(os.path.join(save_dir, f'patches_trial_{trial_idx}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Projective coordinates visualization (stereographic projection)
    if proj_coords is not None and len(proj_coords) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get stereographic projection
        try:
            subsample_idx = GeometryUtils.get_greedy_perm_pc(proj_coords, 
                                                             min(300, len(proj_coords)))['perm']
            stereo = ProjectiveMapUtils.get_stereo_proj_codim1(proj_coords[subsample_idx])
            
            ax.scatter(stereo[:, 0], stereo[:, 1], s=5, alpha=0.6)
            ax.set_aspect('equal')
            ax.set_title('Projective Coordinates (Stereographic Projection)')
            
            # Draw projective plane boundary
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Projection failed: {e}', ha='center', va='center',
                   transform=ax.transAxes)
        
        fig.savefig(os.path.join(save_dir, f'projcoords_trial_{trial_idx}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Transition maps
    fig, _ = plot_all_transitions(system)
    fig.savefig(os.path.join(save_dir, f'transitions_trial_{trial_idx}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Multi-trial experiment runner
# ============================================================

def run_experiment(
    config: Dict = None,
    save_dir: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run multiple trials and compute summary statistics.
    """
    if config is None:
        config = EXPERIMENT_CONFIG.copy()
    
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"results_{config['manifold']}_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT: {config['manifold_name']} Orientability Detection")
    print(f"# Trials: {config['n_trials']}")
    print(f"# Ground truth: {'Orientable' if config['true_orientable'] else 'Non-orientable'}")
    print(f"# Output directory: {save_dir}")
    print(f"{'#'*60}")
    
    # Verify topology on first trial's data
    print("\nVerifying ℝP² topology...")
    X_verify = generate_line_patches(
        patch_dim=config['patch_dim'],
        n_angles=config['n_angles'],
        n_offsets=config['n_offsets'],
        sigma=config['sigma'],
        seed=config['random_seed_base']
    )
    verify_rp2_topology(X_verify, save_path=os.path.join(save_dir, 'topology_verification.png'))
    
    # Run trials
    all_trials = []
    for trial_idx in range(config['n_trials']):
        trial_metrics = run_single_trial(
            config=config,
            trial_idx=trial_idx,
            save_dir=save_dir,
            verbose=verbose
        )
        all_trials.append(trial_metrics)
    
    # Compute summary statistics
    summary = compute_summary_statistics(all_trials, config)
    
    # Compile results
    results = {
        'config': config,
        'trials': all_trials,
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save results
    results_file = os.path.join(save_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print_summary(summary, config)
    
    # Save LaTeX table
    latex_file = os.path.join(save_dir, 'table.tex')
    save_latex_table(summary, config, latex_file)
    
    return results


def compute_summary_statistics(
    trials: List[Dict],
    config: Dict
) -> Dict[str, Any]:
    """
    Compute mean and standard deviation across trials.
    """
    summary = {}
    
    # Theoretical metrics
    metric_names = ['varepsilon', 'varepsilon_mean', 'eta', 'delta', 'delta_mean',
                    'cocycle_error', 'sigma_min_enc']
    
    summary['theoretical'] = {}
    for name in metric_names:
        values = [t['theoretical'][name] for t in trials]
        summary['theoretical'][name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values,
        }
    
    # Orientability detection accuracy
    correct_detections = [t['orientability']['correct_detection'] for t in trials]
    detected_values = [t['orientability']['detected_orientable'] for t in trials]
    
    summary['orientability'] = {
        'accuracy': float(np.mean(correct_detections)),
        'n_correct': int(np.sum(correct_detections)),
        'n_trials': len(trials),
        'all_cocycle_verified': all(t['orientability']['cocycle_verified'] for t in trials),
        'detected_orientable': detected_values,
        'has_different_across': [t['orientability'].get('has_different_across', False) for t in trials],
        'coboundary_failed': [not t['orientability'].get('coboundary_passed', True) for t in trials],
    }
    
    # Track detection method
    n_coboundary_failed = sum(summary['orientability']['coboundary_failed'])
    n_different_across = sum(summary['orientability']['has_different_across'])
    summary['orientability']['n_coboundary_failed'] = n_coboundary_failed
    summary['orientability']['n_different_across'] = n_different_across
    
    return summary


def print_summary(summary: Dict, config: Dict):
    """Print formatted summary to console."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {config['manifold_name']} ({config['n_trials']} trials)")
    print(f"{'='*60}")
    
    print(f"\n--- Theoretical Metrics (mean ± std) ---")
    for name, stats in summary['theoretical'].items():
        print(f"  {name:20s}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    
    print(f"\n--- Orientability Detection ---")
    print(f"  Ground truth:           {'Orientable' if config['true_orientable'] else 'Non-orientable'}")
    print(f"  Accuracy:               {summary['orientability']['accuracy']*100:.1f}% "
          f"({summary['orientability']['n_correct']}/{summary['orientability']['n_trials']})")
    print(f"  All cocycle verified:   {summary['orientability']['all_cocycle_verified']}")
    print(f"  Detection per trial:    {summary['orientability']['detected_orientable']}")
    
    # Detection method breakdown
    print(f"\n  Detection method breakdown:")
    print(f"    Coboundary test failed:    {summary['orientability']['n_coboundary_failed']}/{config['n_trials']}")
    print(f"    Opposite signs in overlap: {summary['orientability']['n_different_across']}/{config['n_trials']}")
    
    print(f"\n  Note: ℝP² non-orientability may be detected via either method")
    print(f"        depending on the cover structure.")
    
    print(f"\n{'='*60}\n")


def save_latex_table(summary: Dict, config: Dict, filename: str):
    """
    Save a LaTeX table for paper inclusion.
    """
    with open(filename, 'w') as f:
        f.write("% Auto-generated LaTeX table for paper\n")
        f.write(f"% Manifold: {config['manifold_name']}\n")
        f.write(f"% Trials: {config['n_trials']}\n")
        f.write(f"% Ground truth: {'Orientable' if config['true_orientable'] else 'Non-orientable'}\n")
        f.write(f"% Generated: {datetime.now().isoformat()}\n\n")
        
        # Metrics table
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Theoretical metrics for $\\mathbb{{RP}}^2$ (line patches) "
                f"(mean $\\pm$ std over {config['n_trials']} trials)}}\n")
        f.write("\\label{tab:rp2_metrics}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Metric & Value & Paper Definition \\\\\n")
        f.write("\\midrule\n")
        
        metric_latex = {
            'varepsilon': ('$\\varepsilon$', '$\\sup_x \\|D_i(E_i(x)) - x\\|$'),
            'eta': ('$\\eta$', '$\\sup_x \\|d(D_i \\circ E_i)_x - I\\|_{\\text{op}}$'),
            'delta': ('$\\delta$', '$\\min_{i,j,x} |\\det g_{ji}(x)|$'),
            'cocycle_error': ('Cocycle err', '$\\|T_{ji}(E_i(x)) - E_j(x)\\|$'),
            'sigma_min_enc': ('$\\sigma_{\\min}(dE)$', '$\\min \\sigma_{\\min}(dE_i)$'),
        }
        
        for name, (latex_name, latex_def) in metric_latex.items():
            if name in summary['theoretical']:
                stats = summary['theoretical'][name]
                f.write(f"{latex_name} & ${stats['mean']:.4f} \\pm {stats['std']:.4f}$ & {latex_def} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Detection result
        acc = summary['orientability']['accuracy'] * 100
        n_correct = summary['orientability']['n_correct']
        n_trials = summary['orientability']['n_trials']
        
        f.write(f"% Orientability detection accuracy: {acc:.1f}\\% ({n_correct}/{n_trials})\n")
        f.write(f"% Ground truth: Non-orientable (RP2)\n")
        f.write(f"% Coboundary failed: {summary['orientability']['n_coboundary_failed']}/{n_trials}\n")
        f.write(f"% Opposite signs: {summary['orientability']['n_different_across']}/{n_trials}\n")


# ============================================================
# Convenience function for quick single visualization
# ============================================================

def run_single_visualization(config: Dict = None, seed: int = 42):
    """
    Run a single trial with full visualization (for debugging/presentation).
    """
    if config is None:
        config = EXPERIMENT_CONFIG.copy()
    
    np.random.seed(seed)
    
    print("="*60)
    print("ℝP² (LINE PATCHES) SINGLE VISUALIZATION")
    print("="*60)
    
    # Generate data
    print("\n1. GENERATING LINE PATCHES")
    print("-"*40)
    
    X = generate_line_patches(
        patch_dim=config['patch_dim'],
        n_angles=config['n_angles'],
        n_offsets=config['n_offsets'],
        sigma=config['sigma'],
        seed=seed
    )
    
    print(f"Generated {len(X)} line patches")
    print(f"Dimension: {X.shape[1]} (={config['patch_dim']}×{config['patch_dim']})")
    
    # Visualize patches
    fig = plt.figure(figsize=(8, 8))
    PlotUtils.plot_patches(X[:256], zoom=2)
    plt.gca().set_facecolor((0.7, 0.7, 0.7))
    plt.title("Line Patches (ℝP² topology)")
    plt.show()
    
    # Verify topology
    print("\n2. VERIFYING ℝP² TOPOLOGY")
    print("-"*40)
    verify_rp2_topology(X)
    
    # Create cover
    print("\n3. CREATING COVER")
    print("-"*40)
    
    subset_assignments, X_ordered, proj_coords, perm, dist_mat = create_projective_cover(
        X,
        n_charts=config['n_charts'],
        n_landmarks=config['n_landmarks'],
        n_neighbors=config['n_neighbors_geodesic'],
        threshold_percentile=config['threshold_percentile']
    )
    n_charts = len(subset_assignments)
    
    print(f"Cover statistics:")
    for i, indices in enumerate(subset_assignments):
        print(f"  Chart {i}: {len(indices)} points")
    
    # Visualize projective coordinates
    print("\nVisualizing projective coordinates...")
    try:
        subsample_idx = GeometryUtils.get_greedy_perm_pc(proj_coords, 300)['perm']
        stereo = ProjectiveMapUtils.get_stereo_proj_codim1(proj_coords[subsample_idx])
        
        plt.figure(figsize=(8, 8))
        PlotUtils.imscatter(stereo, X[perm][subsample_idx], 10)
        PlotUtils.plot_proj_boundary()
        plt.title("Line patches in projective coordinates")
        plt.show()
    except Exception as e:
        print(f"  Projective visualization failed: {e}")
    
    # Train
    print("\n4. TRAINING ATLAS AUTOENCODER")
    print("-"*40)
    
    import tensorflow as tf
    tf.random.set_seed(seed)
    
    system = AtlasAutoencoder(
        data=X_ordered,
        n_charts=n_charts,
        subset_assignments=subset_assignments,
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims']
    )
    
    system.fit(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lambda_jac=config['lambda_jac'],
        lambda_cocycle=config['lambda_cocycle'],
        verbose=True
    )
    
    # Visualize transitions
    print("\n5. VISUALIZING TRANSITIONS")
    print("-"*40)
    plot_all_transitions(system)
    plt.show()
    
    # Run orientability detection
    print("\n6. ORIENTABILITY DETECTION")
    print("-"*40)
    
    result = check_orientability(
        system=system,
        points=X_ordered,
        subset_assignments=subset_assignments,
        eps_cluster=config['eps_cluster'],
        min_points=config['min_points'],
        eps_det=config.get('eps_det', 1e-6),
        verbose=True
    )
    
    # Print final metrics
    system.print_metrics_summary()
    
    return system, result


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ℝP² (line patches) orientability experiment')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--charts', type=int, default=10, help='Number of charts')
    parser.add_argument('--single', action='store_true', help='Run single visualization only')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    config = EXPERIMENT_CONFIG.copy()
    config['n_trials'] = args.trials
    config['epochs'] = args.epochs
    config['n_charts'] = args.charts
    
    if args.single:
        # Single run with visualization
        system, result = run_single_visualization(config)
    else:
        # Full experiment
        results = run_experiment(config, save_dir=args.output)
        
        print("\nExperiment complete!")
        print(f"Results saved to: {args.output or 'results_RP2_<timestamp>'}")