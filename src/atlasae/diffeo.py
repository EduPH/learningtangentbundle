"""
Verifying that Autoencoder Charts are Diffeomorphisms
=====================================================

For the learned bundle T_A to be isomorphic to the true tangent bundle TM,
we need each encoder E_i: U_i → R^d to be a diffeomorphism onto its image.

This requires:
1. Injectivity: E_i is one-to-one
2. Immersion: rank(DE_i(x)) = d for all x ∈ U_i  
3. Inverse exists: D_i ∘ E_i = Id (reconstruction condition)

Additionally, for the atlas to be compatible with the smooth structure:
4. Transition maps E_j ∘ D_i should be smooth diffeomorphisms

This script provides numerical tests for these conditions.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# ============================================================
# Test 1: Injectivity of Encoders
# ============================================================

def check_injectivity(
    system,  # AtlasAutoencoder
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    tolerance: float = 1e-3,
    verbose: bool = True
) -> Dict:
    """
    Check if each encoder E_i is injective on its chart domain U_i.
    
    A map is injective if distinct points map to distinct images:
        x ≠ y  ⟹  E_i(x) ≠ E_i(y)
    
    Numerically, we check:
    - No two points in U_i map to the same latent code (within tolerance)
    - The ratio ||E_i(x) - E_i(y)|| / ||x - y|| is bounded away from 0
    
    Returns:
        Dict with injectivity diagnostics per chart
    """
    results = {}
    
    if verbose:
        print("="*60)
        print("TEST 1: INJECTIVITY OF ENCODERS")
        print("="*60)
    
    for i, indices in enumerate(subset_assignments):
        chart_points = points[indices]
        n_points = len(chart_points)
        
        if n_points < 2:
            continue
        
        # Encode all points
        latent = system.encode(chart_points, chart=i)
        
        # Check for collisions in latent space
        # Two points collide if their latent codes are within tolerance
        latent_dists = squareform(pdist(latent))
        ambient_dists = squareform(pdist(chart_points))
        
        # Mask diagonal
        np.fill_diagonal(latent_dists, np.inf)
        np.fill_diagonal(ambient_dists, np.inf)
        
        # Find near-collisions in latent space
        collisions = latent_dists < tolerance
        n_collisions = np.sum(collisions) // 2  # Each pair counted twice
        
        # Compute distortion ratio: ||E(x) - E(y)|| / ||x - y||
        # Should be bounded away from 0 for injectivity
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = latent_dists / ambient_dists
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        
        min_ratio = np.min(ratios) if len(ratios) > 0 else 0
        mean_ratio = np.mean(ratios) if len(ratios) > 0 else 0
        
        # Bi-Lipschitz constant (lower bound)
        # For a diffeomorphism, this should be > 0
        bi_lipschitz_lower = min_ratio
        
        results[i] = {
            'n_points': n_points,
            'n_collisions': n_collisions,
            'min_ratio': min_ratio,
            'mean_ratio': mean_ratio,
            'bi_lipschitz_lower': bi_lipschitz_lower,
            'is_injective': n_collisions == 0 and min_ratio > tolerance
        }
        
        if verbose:
            status = "✓" if results[i]['is_injective'] else "✗"
            print(f"\nChart {i} ({n_points} points):")
            print(f"  Collisions (tol={tolerance}): {n_collisions}")
            print(f"  Min distance ratio ||E(x)-E(y)||/||x-y||: {min_ratio:.6f}")
            print(f"  Mean distance ratio: {mean_ratio:.4f}")
            print(f"  {status} Injective: {results[i]['is_injective']}")
    
    return results


# ============================================================
# Test 2: Full Rank Jacobian (Immersion Condition)
# ============================================================

def check_immersion(
    system,  # AtlasAutoencoder
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    ambient_dim: int = 3,
    latent_dim: int = 2,
    tolerance: float = 1e-6,
    verbose: bool = True
) -> Dict:
    """
    Check if each encoder E_i is an immersion: rank(DE_i(x)) = latent_dim.
    
    For E_i: R^n → R^d with n > d, the Jacobian DE_i(x) is a d×n matrix.
    For an immersion, this must have full row rank (= d) everywhere.
    
    Equivalently, the d×d matrix (DE_i)(DE_i)^T must be non-singular,
    or the smallest singular value of DE_i must be > 0.
    
    Returns:
        Dict with immersion diagnostics per chart
    """
    results = {}
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 2: IMMERSION CONDITION (FULL RANK JACOBIAN)")
        print("="*60)
    
    for i, indices in enumerate(subset_assignments):
        chart_points = points[indices]
        n_points = len(chart_points)
        
        if n_points < 1:
            continue
        
        # Compute Jacobian of encoder at each point
        x_tf = tf.constant(chart_points.astype(np.float32))
        
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            z = system.autoencoders[i].encode(x_tf)
        
        # Jacobian: (n_points, latent_dim, ambient_dim)
        jacobians = tape.batch_jacobian(z, x_tf).numpy()
        
        # Compute singular values for each Jacobian
        min_singular_values = []
        ranks = []
        condition_numbers = []
        
        for J in jacobians:
            # J is latent_dim × ambient_dim
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            min_sv = S[-1] if len(S) > 0 else 0
            max_sv = S[0] if len(S) > 0 else 1
            
            min_singular_values.append(min_sv)
            ranks.append(np.sum(S > tolerance))
            
            if min_sv > tolerance:
                condition_numbers.append(max_sv / min_sv)
            else:
                condition_numbers.append(np.inf)
        
        min_singular_values = np.array(min_singular_values)
        ranks = np.array(ranks)
        condition_numbers = np.array(condition_numbers)
        
        # Check if all Jacobians have full rank
        full_rank_count = np.sum(ranks == latent_dim)
        min_sv_overall = np.min(min_singular_values)
        mean_sv = np.mean(min_singular_values)
        max_condition = np.max(condition_numbers[np.isfinite(condition_numbers)])
        
        results[i] = {
            'n_points': n_points,
            'full_rank_count': full_rank_count,
            'full_rank_fraction': full_rank_count / n_points,
            'min_singular_value': min_sv_overall,
            'mean_min_singular_value': mean_sv,
            'max_condition_number': max_condition,
            'is_immersion': full_rank_count == n_points
        }
        
        if verbose:
            status = "✓" if results[i]['is_immersion'] else "✗"
            print(f"\nChart {i} ({n_points} points):")
            print(f"  Full rank Jacobians: {full_rank_count}/{n_points} ({100*full_rank_count/n_points:.1f}%)")
            print(f"  Min singular value (overall): {min_sv_overall:.6f}")
            print(f"  Mean of min singular values: {mean_sv:.6f}")
            print(f"  Max condition number: {max_condition:.2f}")
            print(f"  {status} Immersion: {results[i]['is_immersion']}")
    
    return results


# ============================================================
# Test 3: Reconstruction Quality (Inverse Condition)
# ============================================================

def check_reconstruction(
    system,  # AtlasAutoencoder
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    tolerance: float = 1e-3,
    verbose: bool = True
) -> Dict:
    """
    Check reconstruction quality: D_i ∘ E_i ≈ Id on U_i.
    
    For D_i to be the inverse of E_i, we need:
        ||D_i(E_i(x)) - x|| ≈ 0 for all x ∈ U_i
    
    Returns:
        Dict with reconstruction diagnostics per chart
    """
    results = {}
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 3: RECONSTRUCTION QUALITY (D_i ∘ E_i = Id)")
        print("="*60)
    
    for i, indices in enumerate(subset_assignments):
        chart_points = points[indices]
        n_points = len(chart_points)
        
        if n_points < 1:
            continue
        
        # Encode then decode
        latent = system.encode(chart_points, chart=i)
        reconstructed = system.decode(latent, chart=i)
        
        # Reconstruction errors
        errors = np.linalg.norm(reconstructed - chart_points, axis=1)
        
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        
        # Fraction of points with error below tolerance
        good_reconstructions = np.sum(errors < tolerance)
        
        results[i] = {
            'n_points': n_points,
            'max_error': max_error,
            'mean_error': mean_error,
            'median_error': median_error,
            'good_fraction': good_reconstructions / n_points,
            'is_inverse': max_error < tolerance
        }
        
        if verbose:
            status = "✓" if mean_error < tolerance else "⚠"
            print(f"\nChart {i} ({n_points} points):")
            print(f"  Max reconstruction error: {max_error:.6f}")
            print(f"  Mean reconstruction error: {mean_error:.6f}")
            print(f"  Median reconstruction error: {median_error:.6f}")
            print(f"  Points with error < {tolerance}: {good_reconstructions}/{n_points} ({100*good_reconstructions/n_points:.1f}%)")
            print(f"  {status} Approximate inverse: {mean_error < tolerance}")
    
    return results


# ============================================================
# Test 4: Transition Map Smoothness
# ============================================================

def check_transition_smoothness(
    system,  # AtlasAutoencoder
    points: np.ndarray,
    cover,  # GoodCover
    n_samples: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Check if transition maps T_ji = E_j ∘ D_i are smooth diffeomorphisms.
    
    For smoothness, we verify:
    1. T_ji has full rank Jacobian (local diffeomorphism)
    2. T_ji is approximately invertible: T_ij ∘ T_ji ≈ Id
    3. Jacobian varies smoothly (bounded derivatives)
    
    Returns:
        Dict with smoothness diagnostics per overlap
    """
    results = {}
    membership = cover.membership(points)
    
    if verbose:
        print("\n" + "="*60)
        print("TEST 4: TRANSITION MAP SMOOTHNESS")
        print("="*60)
    
    for i, j in combinations(range(cover.n_charts), 2):
        overlap_mask = membership[:, i] & membership[:, j]
        overlap_indices = np.where(overlap_mask)[0]
        
        if len(overlap_indices) < 10:
            continue
        
        # Sample points from overlap
        if len(overlap_indices) > n_samples:
            sample_idx = np.random.choice(len(overlap_indices), n_samples, replace=False)
            overlap_indices = overlap_indices[sample_idx]
        
        overlap_points = points[overlap_indices]
        
        # Encode in chart i
        z_i = system.encode(overlap_points, chart=i)
        z_i_tf = tf.constant(z_i.astype(np.float32))
        
        # Compute T_ji and its Jacobian
        with tf.GradientTape() as tape:
            tape.watch(z_i_tf)
            z_j = system.transition_map(z_i_tf, i, j)
        
        jacobians_ji = tape.batch_jacobian(z_j, z_i_tf).numpy()
        z_j = z_j.numpy()
        
        # Check determinants (should be non-zero)
        determinants = np.linalg.det(jacobians_ji)
        min_abs_det = np.min(np.abs(determinants))
        
        # Check invertibility: T_ij ∘ T_ji ≈ Id
        z_j_tf = tf.constant(z_j.astype(np.float32))
        z_i_back = system.transition_map(z_j_tf, j, i).numpy()
        
        roundtrip_errors = np.linalg.norm(z_i_back - z_i, axis=1)
        max_roundtrip = np.max(roundtrip_errors)
        mean_roundtrip = np.mean(roundtrip_errors)
        
        # Check Jacobian smoothness (variation across nearby points)
        # Use standard deviation of determinants as a proxy
        det_std = np.std(determinants)
        det_mean = np.mean(determinants)
        det_cv = det_std / np.abs(det_mean) if det_mean != 0 else np.inf
        
        results[(i, j)] = {
            'n_points': len(overlap_indices),
            'min_abs_det': min_abs_det,
            'det_mean': det_mean,
            'det_std': det_std,
            'det_coefficient_of_variation': det_cv,
            'max_roundtrip_error': max_roundtrip,
            'mean_roundtrip_error': mean_roundtrip,
            'is_local_diffeo': min_abs_det > 1e-6,
            'is_invertible': max_roundtrip < 0.1
        }
        
        if verbose:
            status = "✓" if results[(i,j)]['is_local_diffeo'] and results[(i,j)]['is_invertible'] else "⚠"
            print(f"\nTransition T_{j}{i} ({len(overlap_indices)} points):")
            print(f"  Min |det(Jacobian)|: {min_abs_det:.6f}")
            print(f"  Determinant mean ± std: {det_mean:.4f} ± {det_std:.4f}")
            print(f"  Roundtrip error (T_ij ∘ T_ji): max={max_roundtrip:.6f}, mean={mean_roundtrip:.6f}")
            print(f"  {status} Local diffeomorphism: {results[(i,j)]['is_local_diffeo']}")
            print(f"  {status} Invertible: {results[(i,j)]['is_invertible']}")
    
    return results


# ============================================================
# Combined Test: Is the Atlas a Diffeomorphism Atlas?
# ============================================================

def verify_diffeomorphism_atlas(
    system,  # AtlasAutoencoder
    points: np.ndarray,
    cover,  # GoodCover
    subset_assignments: List[np.ndarray],
    verbose: bool = True
) -> Dict:
    """
    Run all tests to verify if the learned atlas defines diffeomorphism charts.
    
    For T_A ≅ TM (Proposition 3.7), we need:
    1. Each E_i is injective on U_i
    2. Each E_i is an immersion (full rank Jacobian)
    3. Each D_i ∘ E_i = Id (reconstruction)
    4. Transition maps are smooth diffeomorphisms
    
    Returns:
        Dict with all test results and final verdict
    """
    if verbose:
        print("\n" + "="*70)
        print("  VERIFYING DIFFEOMORPHISM ATLAS PROPERTY")
        print("  (Required for T_A ≅ TM by Proposition 3.7)")
        print("="*70)
    
    results = {
        'injectivity': check_injectivity(system, points, subset_assignments, verbose=verbose),
        'immersion': check_immersion(system, points, subset_assignments, verbose=verbose),
        'reconstruction': check_reconstruction(system, points, subset_assignments, verbose=verbose),
        'transitions': check_transition_smoothness(system, points, cover, verbose=verbose)
    }
    
    # Aggregate results
    all_injective = all(r['is_injective'] for r in results['injectivity'].values())
    all_immersions = all(r['is_immersion'] for r in results['immersion'].values())
    good_reconstruction = all(r['mean_error'] < 0.01 for r in results['reconstruction'].values())
    good_transitions = all(r['is_local_diffeo'] and r['is_invertible'] 
                          for r in results['transitions'].values())
    
    is_diffeomorphism_atlas = all_injective and all_immersions and good_reconstruction and good_transitions
    
    results['summary'] = {
        'all_injective': all_injective,
        'all_immersions': all_immersions,
        'good_reconstruction': good_reconstruction,
        'good_transitions': good_transitions,
        'is_diffeomorphism_atlas': is_diffeomorphism_atlas
    }
    
    if verbose:
        print("\n" + "="*70)
        print("  SUMMARY")
        print("="*70)
        print(f"\n  Injectivity:      {'✓' if all_injective else '✗'} {all_injective}")
        print(f"  Immersion:        {'✓' if all_immersions else '✗'} {all_immersions}")
        print(f"  Reconstruction:   {'✓' if good_reconstruction else '✗'} {good_reconstruction}")
        print(f"  Transitions:      {'✓' if good_transitions else '✗'} {good_transitions}")
        print(f"\n  {'='*50}")
        if is_diffeomorphism_atlas:
            print(f"  ✓ ATLAS IS A DIFFEOMORPHISM ATLAS")
            print(f"  ⟹ By Proposition 3.7: T_A ≅ TM")
        else:
            print(f"  ⚠ ATLAS MAY NOT BE A DIFFEOMORPHISM ATLAS")
            print(f"  ⟹ T_A is still a well-defined vector bundle,")
            print(f"     but may not be isomorphic to TM")
        print(f"  {'='*50}")
    
    return results


# ============================================================
# Visualization: Encoder Quality
# ============================================================

def visualize_encoder_quality(
    system,
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    figsize: Tuple[int, int] = (16, 4)
):
    """
    Visualize encoder quality for each chart.
    
    Shows:
    - Latent space embeddings
    - Color-coded by reconstruction error
    - Singular value distribution of Jacobians
    """
    n_charts = len(subset_assignments)
    fig, axes = plt.subplots(2, n_charts, figsize=(4*n_charts, 8))
    
    if n_charts == 1:
        axes = axes.reshape(2, 1)
    
    for i, indices in enumerate(subset_assignments):
        chart_points = points[indices]
        
        # Encode
        latent = system.encode(chart_points, chart=i)
        reconstructed = system.decode(latent, chart=i)
        errors = np.linalg.norm(reconstructed - chart_points, axis=1)
        
        # Compute Jacobian singular values
        x_tf = tf.constant(chart_points.astype(np.float32))
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            z = system.autoencoders[i].encode(x_tf)
        jacobians = tape.batch_jacobian(z, x_tf).numpy()
        
        min_svs = []
        for J in jacobians:
            _, S, _ = np.linalg.svd(J, full_matrices=False)
            min_svs.append(S[-1])
        min_svs = np.array(min_svs)
        
        # Plot 1: Latent space colored by reconstruction error
        ax1 = axes[0, i]
        sc = ax1.scatter(latent[:, 0], latent[:, 1], c=errors, cmap='hot', s=10, alpha=0.7)
        plt.colorbar(sc, ax=ax1, label='Recon. error')
        ax1.set_title(f'Chart {i}: Latent space')
        ax1.set_xlabel('$z_1$')
        ax1.set_ylabel('$z_2$')
        ax1.set_aspect('equal')
        
        # Plot 2: Histogram of minimum singular values
        ax2 = axes[1, i]
        ax2.hist(min_svs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', label='Rank deficient')
        ax2.set_title(f'Chart {i}: Min singular values')
        ax2.set_xlabel('$\\sigma_{\\min}(DE_i)$')
        ax2.set_ylabel('Count')
        ax2.legend()
    
    plt.tight_layout()
    return fig


# ============================================================
# Main demonstration
# ============================================================

if __name__ == "__main__":
    print("This module provides tests for verifying diffeomorphism properties.")
    print("Import and use with a trained AtlasAutoencoder system.")
    print("\nExample usage:")
    print("  from diffeomorphism_check import verify_diffeomorphism_atlas")
    print("  results = verify_diffeomorphism_atlas(system, points, cover, subset_assignments)")