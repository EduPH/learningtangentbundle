"""
Sphere Orientability Experiment
===============================

This script runs multiple trials of the atlas autoencoder on S² and collects
metrics for inclusion in the paper. The sphere is orientable, so we expect
the algorithm to correctly identify w₁(TS²) = 0.

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
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from atlasautoencoder import AtlasAutoencoder, plot_all_transitions
from orientability import check_orientability
from sphere_good_cover import tetrahedral_cover_S2


# ============================================================
# Configuration
# ============================================================

EXPERIMENT_CONFIG = {
    'manifold': 'S2',
    'manifold_name': 'Sphere',
    'true_orientable': True,
    'intrinsic_dim': 2,
    'ambient_dim': 3,
    
    # Sampling
    'n_points': 1000,
    
    # Cover
    'cover_type': 'tetrahedral',
    'cover_epsilon': 0.3,
    
    # Architecture
    'latent_dim': 2,
    'hidden_dims': [32, 16],
    
    # Training
    'epochs': 1000,
    'batch_size': 64,
    'lambda_jac': 0.0,
    'lambda_cocycle': 0.0,
    
    # Orientability detection
    'eps_cluster': 1.0,
    'min_points': 20,
    
    # Experiment
    'n_trials': 5,
    'random_seed_base': 42,
}


# ============================================================
# Data generation
# ============================================================

def sample_sphere_uniform(n_samples: int, seed: int = None) -> np.ndarray:
    """Sample points uniformly on S²."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n_samples, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


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
    points = sample_sphere_uniform(config['n_points'], seed=seed)
    
    # Create cover
    cover = tetrahedral_cover_S2(epsilon=config['cover_epsilon'])
    subset_assignments = cover.get_assignments(points)
    n_charts = len(subset_assignments)
    
    # Build atlas autoencoder
    # Use a different seed for TF initialization
    import tensorflow as tf
    tf.random.set_seed(seed)
    
    system = AtlasAutoencoder(
        data=points,
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
    metrics = collect_metrics(system, points, subset_assignments, config, trial_idx)
    
    # Run orientability detection
    orient_result = check_orientability(
        system=system,
        points=points,
        subset_assignments=subset_assignments,
        eps_cluster=config['eps_cluster'],
        min_points=config['min_points'],
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
    
    # Save figures if requested
    if save_dir is not None:
        save_trial_figures(system, points, subset_assignments, trial_idx, save_dir)
    
    return metrics


def collect_metrics(
    system: AtlasAutoencoder,
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    config: Dict,
    trial_idx: int
) -> Dict[str, Any]:
    """
    Collect all theoretical metrics from a trained system.
    
    These are the metrics defined in the paper:
    - ε (varepsilon): pointwise reconstruction error
    - η (eta): differential reconstruction error  
    - δ (delta): non-degeneracy gap
    - cocycle_error: transition consistency
    - σ_min(dE): encoder regularity
    """
    import tensorflow as tf
    
    metrics = {
        'trial_idx': trial_idx,
        'seed': config['random_seed_base'] + trial_idx,
        'config': config,
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
        
        # η: differential error (tangent space version)
        eta_list.append(float(system.compute_eta_tangent(i, x).numpy()))
        
        # σ_min: encoder regularity
        sigma_min_list.append(float(system.compute_encoder_sigma_min(i, x).numpy()))
    
    metrics['per_chart'] = {
        'varepsilon': varepsilon_list,
        'varepsilon_mean': varepsilon_mean_list,
        'eta': eta_list,
        'sigma_min_enc': sigma_min_list,
    }
    
    # Global metrics (paper definitions use sup/min over charts)
    metrics['theoretical'] = {
        'varepsilon': max(varepsilon_list),      # sup over charts
        'varepsilon_mean': np.mean(varepsilon_mean_list),
        'eta': max(eta_list),                     # sup over charts
        'delta': float(system.compute_delta().numpy()),  # min over overlaps
        'delta_mean': float(system.compute_delta_mean().numpy()),
        'cocycle_error': float(system.compute_cocycle_error().numpy()),
        'sigma_min_enc': min(sigma_min_list),     # min over charts
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
    
    Returns:
        Dictionary with all trial results and summary statistics
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
    summary['orientability'] = {
        'accuracy': float(np.mean(correct_detections)),
        'n_correct': int(np.sum(correct_detections)),
        'n_trials': len(trials),
        'all_cocycle_verified': all(t['orientability']['cocycle_verified'] for t in trials),
        'all_coboundary_passed': all(t['orientability']['coboundary_passed'] for t in trials),
    }
    
    # Detection details
    summary['orientability']['detected_orientable'] = [
        t['orientability']['detected_orientable'] for t in trials
    ]
    
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
    print(f"  Ground truth:        {'Orientable' if config['true_orientable'] else 'Non-orientable'}")
    print(f"  Accuracy:            {summary['orientability']['accuracy']*100:.1f}% "
          f"({summary['orientability']['n_correct']}/{summary['orientability']['n_trials']})")
    print(f"  All cocycle verified: {summary['orientability']['all_cocycle_verified']}")
    print(f"  Detection per trial: {summary['orientability']['detected_orientable']}")
    
    print(f"\n{'='*60}\n")


def save_latex_table(summary: Dict, config: Dict, filename: str):
    """
    Save a LaTeX table for paper inclusion.
    """
    with open(filename, 'w') as f:
        f.write("% Auto-generated LaTeX table for paper\n")
        f.write(f"% Manifold: {config['manifold_name']}\n")
        f.write(f"% Trials: {config['n_trials']}\n")
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
        
        # Format each metric
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
        f.write(f"% Orientability detection accuracy: "
                f"{summary['orientability']['accuracy']*100:.1f}\\% "
                f"({summary['orientability']['n_correct']}/{summary['orientability']['n_trials']})\n")
        f.write(f"% Ground truth: {'Orientable' if config['true_orientable'] else 'Non-orientable'}\n")


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
    
    # Sample and visualize data
    points = sample_sphere_uniform(config['n_points'], seed=seed)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title("Uniform sampling on S²")
    plt.show()
    
    # Create cover
    cover = tetrahedral_cover_S2(epsilon=config['cover_epsilon'])
    print("\nVerifying good cover property:")
    cover.verify_good_cover(points)
    subset_assignments = cover.get_assignments(points)
    n_charts = len(subset_assignments)
    
    # Visualize charts
    fig = plt.figure(figsize=(4 * n_charts, 4))
    for i, mask in enumerate(subset_assignments):
        ax = fig.add_subplot(1, n_charts, i + 1, projection="3d")
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], s=2, alpha=0.6)
        ax.set_title(f"Chart {i}")
        ax.set_box_aspect([1, 1, 0.5])
    plt.tight_layout()
    plt.show()
    
    # Train
    import tensorflow as tf
    tf.random.set_seed(seed)
    
    system = AtlasAutoencoder(
        data=points,
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
    plot_all_transitions(system)
    plt.show()
    
    # Run orientability detection
    result = check_orientability(
        system=system,
        points=points,
        subset_assignments=subset_assignments,
        eps_cluster=config['eps_cluster'],
        min_points=config['min_points'],
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
    
    parser = argparse.ArgumentParser(description='Sphere orientability experiment')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--single', action='store_true', help='Run single visualization only')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    config = EXPERIMENT_CONFIG.copy()
    config['n_trials'] = args.trials
    config['epochs'] = args.epochs
    
    if args.single:
        # Single run with visualization
        system, result = run_single_visualization(config)
    else:
        # Full experiment
        results = run_experiment(config, save_dir=args.output)
        
        print("\nExperiment complete!")
        print(f"Results saved to: {args.output or 'results_S2_<timestamp>'}")