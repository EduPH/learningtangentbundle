"""
Klein Bottle Orientability Experiment
=====================================

This script runs multiple trials of the atlas autoencoder on the Klein bottle
and collects metrics for inclusion in the paper. The Klein bottle is non-orientable,
so we expect the algorithm to correctly identify wâ‚(TK) â‰  0.

The Klein bottle is a non-orientable closed surface that cannot be embedded in â„Â³
without self-intersection. We use the 4D embedding from DREiMaC which avoids
self-intersection.

Mathematical background:
- Klein bottle K = TÂ² with one SÂ¹ factor "twisted"
- Hâ‚(K; â„¤/2) = â„¤/2 âŠ• â„¤/2, Hâ‚‚(K; â„¤/2) = â„¤/2
- wâ‚(TK) â‰  0 (non-orientable)
- The orientation double cover of K is the torus TÂ²

Outputs:
- JSON file with all metrics from all trials
- Summary statistics (mean Â± std) for paper tables
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

# DREiMaC for Klein bottle generation and geodesic-based covers
from dreimac import GeometryUtils, GeometryExamples


# ============================================================
# Configuration
# ============================================================

EXPERIMENT_CONFIG = {
    'manifold': 'Klein',
    'manifold_name': 'Klein Bottle',
    'true_orientable': False,  # Klein bottle is NON-orientable
    'intrinsic_dim': 2,
    'ambient_dim': 4,  # 4D embedding to avoid self-intersection
    
    # Sampling (using DREiMaC)
    'n_points': 1000,
    'klein_m': 4,  # Parameter for Klein bottle sampling
    'klein_n': 2,  # Parameter for Klein bottle sampling
    
    # Cover: landmark-based geodesic cover
    'cover_type': 'landmark_geodesic',
    'n_charts': 8,
    'cover_percentile': 20,  # Percentile for epsilon selection
    'n_neighbors_geodesic': 100,  # For geodesic distance computation
    
    # Architecture
    'latent_dim': 2,
    'hidden_dims': [32, 16],
    
    # Training
    'epochs': 5000,
    'batch_size': 64,
    'lambda_smooth': 0,
    'lambda_jac': 0.01,
    'lambda_cocycle': 0.0,
    
    # Orientability detection
    'eps_cluster': 1.0,
    'min_points': 20,
    'eps_det': 1e-6,
    
    # Experiment
    'n_trials': 5,
    'random_seed_base': 42,
    
    # Retry policy (retrain if reconstruction quality is insufficient)
    # Motivation: the stability theorem (Theorem 4.6) requires ε small enough
    # that d·Γ·(L_E L_D + Γ)^{d-1} < δ. High ε invalidates sign cocycle
    # computation and leads to incorrect orientability detection.
    'max_retries': 3,               # Max retry attempts per trial
    'recon_threshold': 0.15,        # Max acceptable sup-ε across charts
    'extra_epochs_per_extension': 2000,  # Additional epochs when extending training
    'max_extensions_before_restart': 1,  # Extensions before trying a fresh seed
}


# ============================================================
# Data generation
# ============================================================

def generate_klein_bottle(
    n_points: int,
    m: int = 4,
    n: int = 2,
    seed: int = None
) -> np.ndarray:
    """
    Generate points on the Klein bottle in 4D.
    
    Uses the DREiMaC library's implementation which provides a proper
    embedding in â„â´ without self-intersection.
    
    Args:
        n_points: Number of points to sample
        m, n: Parameters controlling the Klein bottle shape
        seed: Random seed for reproducibility
        
    Returns:
        points: Array of shape (n_points, 4)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # DREiMaC's Klein bottle implementation
    points = GeometryExamples.klein_bottle_4d(n_points, m, n)
    
    return points


def create_landmark_cover(
    points: np.ndarray,
    n_charts: int,
    percentile: float = 20,
    n_neighbors: int = 100
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Create a landmark-based geodesic cover.
    
    This method:
    1. Selects n_charts+1 landmark points using farthest point sampling
    2. Computes approximate geodesic distances to each landmark
    3. Creates charts as balls around each landmark
    
    This produces a good cover that respects the manifold geometry,
    which is especially important for the Klein bottle's non-trivial topology.
    
    Args:
        points: Data points, shape (n_points, ambient_dim)
        n_charts: Number of charts to create
        percentile: Percentile of distances to use as chart radius
        n_neighbors: Number of neighbors for geodesic approximation
        
    Returns:
        subset_assignments: List of index arrays for each chart
        points_reordered: Points reordered by landmark computation
    """
    n_landmarks = n_charts + 1
    
    # Compute geodesic distances using DREiMaC
    dist_mat, pointcloud_permutation = GeometryUtils.landmark_geodesic_distance(
        points, n_landmarks, n_neighbors
    )
    
    # Reorder data according to landmark computation
    points_reordered = points[pointcloud_permutation]
    
    # Create charts from landmarks
    # Use percentile of distances as epsilon for good overlap
    epsilon = np.percentile(dist_mat[1:], percentile)
    
    subset_assignments = []
    for i in range(n_charts):
        # Points within epsilon of landmark i+1 (skip landmark 0)
        cluster_indices = np.nonzero(dist_mat[i + 1] < epsilon)[0]
        subset_assignments.append(cluster_indices)
    
    return subset_assignments, points_reordered


# ============================================================
# Reconstruction quality evaluation
# ============================================================

def evaluate_reconstruction_quality(
    system: AtlasAutoencoder,
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
) -> Dict[str, float]:
    """
    Quick evaluation of reconstruction quality after training.
    
    Computes ε = sup_x ||D_i(E_i(x)) - x|| for each chart and returns
    the maximum (worst-case) and mean values.
    
    This is used by the retry logic to determine whether the trained
    system satisfies the quality conditions required by the stability
    theorem. Specifically, the sign cocycle ω_{ji} = sign(det g_{ji})
    is a valid Čech 1-cocycle only when ε is small enough that
    
        d · Γ · (L_E L_D + Γ)^{d-1} < δ,
    
    where Γ depends on ε and η. High reconstruction error invalidates
    this bound and leads to unreliable orientability detection.
    
    Returns:
        Dictionary with 'max_varepsilon', 'mean_varepsilon', and
        per-chart 'varepsilon_list'.
    """
    import tensorflow as tf
    
    varepsilon_list = []
    for i in range(system.n_charts):
        x = tf.constant(points[subset_assignments[i]], dtype=tf.float32)
        varepsilon_list.append(float(system.compute_varepsilon(i, x).numpy()))
    
    return {
        'max_varepsilon': max(varepsilon_list),
        'mean_varepsilon': float(np.mean(varepsilon_list)),
        'varepsilon_list': varepsilon_list,
    }


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
    
    Returns:
        Dictionary containing all metrics for this trial
    """
    seed = config['random_seed_base'] + trial_idx
    np.random.seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRIAL {trial_idx + 1}/{config['n_trials']} (seed={seed})")
        print(f"{'='*60}")
    
    # Sample data
    points = generate_klein_bottle(
        config['n_points'],
        m=config['klein_m'],
        n=config['klein_n'],
        seed=seed
    )
    
    # Create cover
    if verbose:
        print("\nCreating landmark-based cover...")
    
    subset_assignments, points_reordered = create_landmark_cover(
        points,
        n_charts=config['n_charts'],
        percentile=config['cover_percentile'],
        n_neighbors=config['n_neighbors_geodesic']
    )
    n_charts = len(subset_assignments)
    
    # Report cover statistics
    if verbose:
        print(f"Cover statistics:")
        for i, indices in enumerate(subset_assignments):
            print(f"  Chart {i}: {len(indices)} points")
        
        # Count total overlaps
        total_overlap = 0
        for i in range(n_charts):
            for j in range(i + 1, n_charts):
                overlap = len(np.intersect1d(subset_assignments[i], subset_assignments[j]))
                if overlap > 0:
                    total_overlap += 1
        print(f"  Pairwise overlaps: {total_overlap}")
    
    # Build and train atlas autoencoder with retry logic
    import tensorflow as tf
    
    max_retries = config.get('max_retries', 3)
    recon_threshold = config.get('recon_threshold', 0.15)
    extra_epochs = config.get('extra_epochs_per_extension', 2000)
    max_extensions = config.get('max_extensions_before_restart', 1)
    
    best_system = None
    best_varepsilon = float('inf')
    retry_log = []
    
    attempt = 0
    while attempt <= max_retries:
        # Determine seed: original for first attempt, perturbed for restarts
        n_restarts = sum(1 for r in retry_log if r['action'] == 'restart')
        current_seed = seed + 1000 * n_restarts
        
        if attempt == 0 or retry_log[-1]['action'] == 'restart':
            # Fresh initialization
            tf.random.set_seed(current_seed)
            np.random.seed(current_seed)
            
            system = AtlasAutoencoder(
                data=points_reordered,
                n_charts=n_charts,
                subset_assignments=subset_assignments,
                latent_dim=config['latent_dim'],
                hidden_dims=config['hidden_dims']
            )
            
            epochs_this_round = config['epochs']
            extensions_so_far = 0
            
            if verbose and attempt > 0:
                print(f"\n  [Retry {attempt}/{max_retries}] Fresh restart with seed={current_seed}")
        else:
            # Extension: continue training the same model
            epochs_this_round = extra_epochs
            extensions_so_far += 1
            
            if verbose:
                print(f"\n  [Retry {attempt}/{max_retries}] Extending training by {extra_epochs} epochs")
        
        # Train
        system.fit(
            epochs=epochs_this_round,
            batch_size=config['batch_size'],
            lambda_smooth=config['lambda_smooth'],
            lambda_jac=config['lambda_jac'],
            lambda_cocycle=config['lambda_cocycle'],
            verbose=verbose
        )
        
        # Evaluate reconstruction quality
        quality = evaluate_reconstruction_quality(system, points_reordered, subset_assignments)
        current_varepsilon = quality['max_varepsilon']
        
        if verbose:
            print(f"\n  Reconstruction quality: max ε = {current_varepsilon:.4f} "
                  f"(threshold = {recon_threshold:.4f})")
        
        # Track the best system seen so far
        if current_varepsilon < best_varepsilon:
            best_varepsilon = current_varepsilon
            best_system = system
        
        # Check if quality is acceptable
        if current_varepsilon <= recon_threshold:
            if verbose and attempt > 0:
                print(f"  ✓ Reconstruction quality acceptable after {attempt} retries")
            break
        
        # Decide retry strategy
        attempt += 1
        if attempt > max_retries:
            if verbose:
                print(f"\n  ⚠ Max retries ({max_retries}) exhausted. "
                      f"Using best system (max ε = {best_varepsilon:.4f})")
            system = best_system
            break
        
        if extensions_so_far < max_extensions:
            action = 'extend'
        else:
            action = 'restart'
        
        retry_log.append({
            'attempt': attempt,
            'action': action,
            'varepsilon_before': current_varepsilon,
            'seed': current_seed,
        })
    
    # Collect metrics (using the best system)
    metrics = collect_metrics(system, points_reordered, subset_assignments, config, trial_idx)
    
    # Record retry information in metrics
    metrics['retry_info'] = {
        'n_retries': len(retry_log),
        'final_max_varepsilon': best_varepsilon,
        'threshold': recon_threshold,
        'log': retry_log,
    }
    
    # Run orientability detection
    orient_result = check_orientability(
        system=system,
        points=points_reordered,
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
    
    # Record sign information for analysis
    if orient_result['signs']:
        metrics['orientability']['sign_summary'] = summarize_signs(orient_result['signs'])
    
    # Save figures if requested
    if save_dir is not None:
        save_trial_figures(system, points_reordered, subset_assignments, trial_idx, save_dir)
    
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
        
        # Îµ: reconstruction error
        varepsilon_list.append(float(system.compute_varepsilon(i, x).numpy()))
        varepsilon_mean_list.append(float(system.compute_varepsilon_mean(i, x).numpy()))
        
        # Î·: differential error (tangent space version)
        eta_list.append(float(system.compute_eta_tangent(i, x).numpy()))
        
        # Ïƒ_min: encoder regularity
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
    trial_idx: int,
    save_dir: str
):
    """Save visualization figures for a trial."""
    os.makedirs(save_dir, exist_ok=True)
    
    # For 4D data, project to 3D for visualization
    # Use first 3 coordinates
    fig = plt.figure(figsize=(10, 5))
    
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 3], cmap='viridis', s=2, alpha=0.6)
    ax.set_title('Klein Bottle (projected to 3D)')
    ax.set_xlabel('xâ‚')
    ax.set_ylabel('xâ‚‚')
    ax.set_zlabel('xâ‚ƒ')
    
    # 2D projection
    ax2 = fig.add_subplot(122)
    sc = ax2.scatter(points[:, 0], points[:, 1],
                     c=points[:, 3], cmap='viridis', s=2, alpha=0.6)
    ax2.set_aspect('equal')
    ax2.set_xlabel('xâ‚')
    ax2.set_ylabel('xâ‚‚')
    plt.colorbar(sc, ax=ax2, label='xâ‚„')
    
    fig.savefig(os.path.join(save_dir, f'pointcloud_trial_{trial_idx}.png'),
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
    
    # Retry statistics
    retry_counts = [t.get('retry_info', {}).get('n_retries', 0) for t in trials]
    summary['retry'] = {
        'total_retries': sum(retry_counts),
        'trials_with_retries': sum(1 for r in retry_counts if r > 0),
        'retry_counts': retry_counts,
    }
    
    return summary


def print_summary(summary: Dict, config: Dict):
    """Print formatted summary to console."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {config['manifold_name']} ({config['n_trials']} trials)")
    print(f"{'='*60}")
    
    print(f"\n--- Theoretical Metrics (mean Â± std) ---")
    for name, stats in summary['theoretical'].items():
        print(f"  {name:20s}: {stats['mean']:.6f} Â± {stats['std']:.6f}")
    
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
    
    # Retry statistics
    if 'retry' in summary:
        print(f"\n--- Retry Statistics ---")
        print(f"  Total retries:        {summary['retry']['total_retries']}")
        print(f"  Trials with retries:  {summary['retry']['trials_with_retries']}/{config['n_trials']}")
        print(f"  Per-trial retries:    {summary['retry']['retry_counts']}")
    
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
        f.write(f"\\caption{{Theoretical metrics for {config['manifold_name']} "
                f"(mean $\\pm$ std over {config['n_trials']} trials)}}\n")
        f.write("\\label{tab:" + config['manifold'].lower() + "_metrics}\n")
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
        f.write(f"% Ground truth: {'Orientable' if config['true_orientable'] else 'Non-orientable'}\n")
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
    
    # Sample data
    print("Generating Klein bottle point cloud...")
    points = generate_klein_bottle(
        config['n_points'],
        m=config['klein_m'],
        n=config['klein_n'],
        seed=seed
    )
    
    print(f"  Points shape: {points.shape}")
    print(f"  Ambient dimension: {points.shape[1]}")
    
    # Visualize (project to 3D)
    fig = plt.figure(figsize=(12, 5))
    
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 3], cmap='viridis', s=2, alpha=0.6)
    ax.set_title('Klein Bottle (xâ‚, xâ‚‚, xâ‚ƒ)')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 3],
                c=points[:, 2], cmap='plasma', s=2, alpha=0.6)
    ax2.set_title('Klein Bottle (xâ‚, xâ‚‚, xâ‚„)')
    
    plt.tight_layout()
    plt.show()
    
    # Create cover
    print("\nCreating landmark-based cover...")
    subset_assignments, points_reordered = create_landmark_cover(
        points,
        n_charts=config['n_charts'],
        percentile=config['cover_percentile'],
        n_neighbors=config['n_neighbors_geodesic']
    )
    n_charts = len(subset_assignments)
    
    print(f"Cover statistics:")
    for i, indices in enumerate(subset_assignments):
        print(f"  Chart {i}: {len(indices)} points")
    
    # Train
    import tensorflow as tf
    tf.random.set_seed(seed)
    
    system = AtlasAutoencoder(
        data=points_reordered,
        n_charts=n_charts,
        subset_assignments=subset_assignments,
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims']
    )
    
    print("\nTraining Atlas Autoencoder...")
    system.fit(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lambda_smooth=config['lambda_smooth'],
        lambda_jac=config['lambda_jac'],
        lambda_cocycle=config['lambda_cocycle'],
        verbose=True
    )
    
    # Visualize transitions
    print("\nVisualizing transition maps...")
    plot_all_transitions(system)
    plt.show()
    
    # Run orientability detection
    print("\nRunning orientability detection...")
    result = check_orientability(
        system=system,
        points=points_reordered,
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
    
    parser = argparse.ArgumentParser(description='Klein bottle orientability experiment')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=4000, help='Training epochs')
    parser.add_argument('--charts', type=int, default=8, help='Number of charts')
    parser.add_argument('--single', action='store_true', help='Run single visualization only')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Max acceptable sup-ε for reconstruction quality')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum retry attempts per trial')
    
    args = parser.parse_args()
    
    config = EXPERIMENT_CONFIG.copy()
    config['n_trials'] = args.trials
    config['epochs'] = args.epochs
    config['n_charts'] = args.charts
    config['recon_threshold'] = args.threshold
    config['max_retries'] = args.max_retries
    
    if args.single:
        # Single run with visualization
        system, result = run_single_visualization(config)
    else:
        # Full experiment
        results = run_experiment(config, save_dir=args.output)
        
        print("\nExperiment complete!")
        print(f"Results saved to: {args.output or 'results_Klein_<timestamp>'}")