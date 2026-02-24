"""
Corrected Orientability Detection for Disconnected Chart Domains

This module implements the orientability detection algorithm from the paper,
handling the case where learned covers may produce charts with multiple
disconnected components.

Key theoretical points (matching paper notation):
- Transition map: T_ji = E_j ∘ D_i (maps from chart i to chart j)
- Linearized transition: g_ji(x) = d(T_ji)_{E_i(x)}
- Sign cocycle: ω_ji(x) = sign(det g_ji(x))
- Cocycle condition: ω_ki = ω_kj · ω_ji on triple overlaps
- Coboundary condition: ω_ji = ν_j · ν_i for some {ν_i} ∈ {±1}
- Orientable iff w_1(TM) = 0 iff sign cocycle is a coboundary
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional, Union
from collections import deque


def find_chart_components(
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    eps_cluster: float = 0.15,
    min_samples: int = 5,
    min_points: int = 20
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Find connected components of each chart's domain.
    
    Classical differential topology assumes chart domains are connected.
    Learned covers may violate this, so we decompose each chart into
    connected components and treat each as a separate chart.
    
    Args:
        points: Full dataset, shape (n_points, n_features)
        subset_assignments: List of index arrays for each chart
        eps_cluster: DBSCAN epsilon for clustering
        min_samples: DBSCAN min_samples parameter
        min_points: Minimum points for a component to be considered
        
    Returns:
        Dictionary {(chart_idx, component_id): point_indices}
    """
    print("\n" + "="*60)
    print("STEP 1: Finding Connected Components of Chart Domains")
    print("="*60)
    
    chart_components = {}
    
    for chart_idx in range(len(subset_assignments)):
        chart_indices = subset_assignments[chart_idx]
        chart_points = points[chart_indices]
        
        if len(chart_points) < min_points:
            print(f"\nChart {chart_idx}: Too few points ({len(chart_points)}), skipping")
            continue
        
        # Cluster chart domain into connected components
        clustering = DBSCAN(eps=eps_cluster, min_samples=min_samples).fit(chart_points)
        labels = clustering.labels_
        
        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        print(f"\nChart {chart_idx}:")
        print(f"  Total points: {len(chart_points)}")
        print(f"  Connected components: {len(unique_labels)}")
        
        for comp_id in unique_labels:
            comp_mask = (labels == comp_id)
            n_comp = np.sum(comp_mask)
            
            if n_comp < min_points:
                print(f"    Component {comp_id}: {n_comp} points (too small, skipping)")
                continue
            
            # Get original indices for this component
            comp_indices = chart_indices[comp_mask]
            chart_components[(chart_idx, comp_id)] = comp_indices
            
            print(f"    Component {comp_id}: {n_comp} points")
    
    print(f"\n  Total chart components: {len(chart_components)}")
    return chart_components


def _compute_jacobian_signs(
    system, 
    i: int, 
    j: int, 
    points: np.ndarray, 
    eps_det: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobian determinant signs for transition T_ji on given points.
    
    The transition map T_ji = E_j ∘ D_i maps from chart i to chart j.
    The linearized transition g_ji(x) = d(T_ji)_{E_i(x)} is a d×d matrix.
    
    Args:
        system: AtlasAutoencoder instance
        i: Source chart index
        j: Target chart index
        points: Points in ambient space (overlap points)
        eps_det: Threshold for considering determinant as zero
        
    Returns:
        (determinants, signs): Arrays of determinant values and their signs
    """
    import tensorflow as tf
    
    x_tf = tf.constant(points.astype(np.float32))
    
    # Encode with chart i
    z_i = system.autoencoders[i].encode(x_tf)
    
    # Compute Jacobian of transition T_ji = E_j ∘ D_i
    with tf.GradientTape() as tape:
        tape.watch(z_i)
        x_recon = system.autoencoders[i].decode(z_i)
        z_j = system.autoencoders[j].encode(x_recon)
    
    # Jacobian g_ji = d(T_ji), shape (batch, d, d)
    jacobian = tape.batch_jacobian(z_j, z_i)
    dets = tf.linalg.det(jacobian).numpy()
    
    # Compute signs, treating near-zero as undefined
    signs = np.zeros_like(dets)
    nonzero_mask = np.abs(dets) > eps_det
    signs[nonzero_mask] = np.sign(dets[nonzero_mask])
    
    return dets, signs


def _analyze_overlap_signs(
    dets: np.ndarray,
    signs: np.ndarray,
    eps_det: float = 1e-6
) -> Dict:
    """
    Analyze the distribution of signs within an overlap.
    
    For an orientable manifold with a good cover, the sign should be
    constant on each connected component of each overlap.
    
    Returns:
        Dictionary with:
        - 'dominant_sign': Most common sign (±1) or None if degenerate
        - 'is_constant': Whether all signs agree
        - 'has_mixed': Whether there are both +1 and -1 signs
        - 'n_positive': Count of positive determinants
        - 'n_negative': Count of negative determinants
        - 'n_degenerate': Count of near-zero determinants
        - 'mean_abs_det': Mean absolute determinant
    """
    nonzero_mask = np.abs(dets) > eps_det
    valid_signs = signs[nonzero_mask]
    
    n_positive = np.sum(valid_signs > 0)
    n_negative = np.sum(valid_signs < 0)
    n_degenerate = np.sum(~nonzero_mask)
    
    result = {
        'n_positive': int(n_positive),
        'n_negative': int(n_negative),
        'n_degenerate': int(n_degenerate),
        'n_total': len(dets),
        'mean_abs_det': float(np.mean(np.abs(dets[nonzero_mask]))) if np.any(nonzero_mask) else 0.0
    }
    
    if n_positive == 0 and n_negative == 0:
        result['dominant_sign'] = None
        result['is_constant'] = True  # Vacuously true
        result['has_mixed'] = False
    elif n_positive > 0 and n_negative > 0:
        result['dominant_sign'] = +1 if n_positive > n_negative else -1
        result['is_constant'] = False
        result['has_mixed'] = True
    elif n_positive > 0:
        result['dominant_sign'] = +1
        result['is_constant'] = True
        result['has_mixed'] = False
    else:
        result['dominant_sign'] = -1
        result['is_constant'] = True
        result['has_mixed'] = False
    
    return result


def compute_component_transition_signs(
    points: np.ndarray,
    chart_components: Dict[Tuple[int, int], np.ndarray],
    system,  # AtlasAutoencoder
    eps_cluster: float = 0.15,
    min_samples: int = 5,
    min_points: int = 20,
    eps_det: float = 1e-6
) -> Dict:
    """
    Compute transition signs between chart components.
    
    This treats each connected component of each chart as a separate chart,
    then computes the sign cocycle ω_ji on each overlap.
    
    The paper requires constant sign on each connected component of each
    overlap. We detect and report mixed signs as potential indicators of
    non-orientability or numerical issues.
    
    Returns:
        Dictionary with:
        - 'signs': {(chart_i, comp_i, chart_j, comp_j, overlap_comp): sign_info}
        - 'overlaps': Same keys mapping to point indices
        - 'chart_components': The input chart components
        - 'has_any_mixed': Whether any overlap has mixed signs
    """
    print("\n" + "="*60)
    print("STEP 2: Computing Transition Signs Between Chart Components")
    print("="*60)
    
    results = {
        'signs': {},
        'sign_info': {},  # Detailed sign analysis
        'overlaps': {},
        'chart_components': chart_components,
        'has_any_mixed': False
    }
    
    comp_list = list(chart_components.keys())
    
    for idx_i, (chart_i, comp_i) in enumerate(comp_list):
        for idx_j in range(idx_i + 1, len(comp_list)):
            chart_j, comp_j = comp_list[idx_j]
            
            # Get indices for these components
            indices_i = chart_components[(chart_i, comp_i)]
            indices_j = chart_components[(chart_j, comp_j)]
            
            # Find overlap
            overlap_indices = np.intersect1d(indices_i, indices_j)
            n_overlap = len(overlap_indices)
            
            if n_overlap < min_points:
                continue
            
            overlap_points = points[overlap_indices]
            
            # Cluster overlap into connected components
            overlap_clustering = DBSCAN(eps=eps_cluster, min_samples=min_samples).fit(overlap_points)
            overlap_labels = overlap_clustering.labels_
            
            unique_overlap_labels = [l for l in np.unique(overlap_labels) if l != -1]
            
            if len(unique_overlap_labels) == 0:
                continue
            
            print(f"\nChart {chart_i}.{comp_i} ↔ Chart {chart_j}.{comp_j}:")
            print(f"  Total overlap: {n_overlap} points")
            print(f"  Overlap components: {len(unique_overlap_labels)}")
            
            for overlap_comp_id in unique_overlap_labels:
                comp_mask = (overlap_labels == overlap_comp_id)
                n_comp = np.sum(comp_mask)
                
                if n_comp < min_points:
                    continue
                
                comp_overlap_points = overlap_points[comp_mask]
                comp_overlap_indices = overlap_indices[comp_mask]
                
                # Compute Jacobian signs for all points in this overlap component
                dets, signs = _compute_jacobian_signs(
                    system, chart_i, chart_j, comp_overlap_points, eps_det
                )
                
                # Analyze sign distribution
                sign_info = _analyze_overlap_signs(dets, signs, eps_det)
                
                if sign_info['dominant_sign'] is None:
                    print(f"    Overlap component {overlap_comp_id}: DEGENERATE "
                          f"(all determinants near zero), n = {n_comp}")
                    continue
                
                # Store results
                key = (chart_i, comp_i, chart_j, comp_j, overlap_comp_id)
                results['signs'][key] = sign_info['dominant_sign']
                results['sign_info'][key] = sign_info
                results['overlaps'][key] = comp_overlap_indices
                
                if sign_info['has_mixed']:
                    results['has_any_mixed'] = True
                    print(f"    Overlap component {overlap_comp_id}: sign = {sign_info['dominant_sign']:+d} "
                          f"⚠ MIXED ({sign_info['n_positive']}+/{sign_info['n_negative']}-), "
                          f"|det| = {sign_info['mean_abs_det']:.4f}, n = {n_comp}")
                else:
                    print(f"    Overlap component {overlap_comp_id}: sign = {sign_info['dominant_sign']:+d}, "
                          f"|det| = {sign_info['mean_abs_det']:.4f}, n = {n_comp}")
    
    return results


def verify_cocycle_condition(
    results: Dict,
    comp_list: List[Tuple[int, int]],
    verbose: bool = True
) -> Tuple[bool, List]:
    """
    Verify the cocycle condition: ω_ki = ω_kj · ω_ji on all triple overlaps.
    
    The cocycle condition is a necessary condition for the sign cocycle to
    be well-defined as a Čech 1-cocycle. Violations indicate numerical issues
    or training problems, not non-orientability.
    
    Args:
        results: Output from compute_component_transition_signs
        comp_list: List of chart components
        verbose: Whether to print details
        
    Returns:
        (cocycle_verified, violations): Boolean and list of violating triples
    """
    print("\n" + "="*60)
    print("STEP 3: Verifying Cocycle Condition")
    print("="*60)
    
    signs = results['signs']
    violations = []
    n_checked = 0
    
    n_comps = len(comp_list)
    
    for idx_i in range(n_comps):
        comp_i = comp_list[idx_i]
        chart_i, ci = comp_i
        
        for idx_j in range(idx_i + 1, n_comps):
            comp_j = comp_list[idx_j]
            chart_j, cj = comp_j
            
            for idx_k in range(idx_j + 1, n_comps):
                comp_k = comp_list[idx_k]
                chart_k, ck = comp_k
                
                # Check if all three pairwise overlaps exist
                # We need ω_ij, ω_jk, ω_ik
                sign_ij = _get_sign_for_pair(signs, chart_i, ci, chart_j, cj)
                sign_jk = _get_sign_for_pair(signs, chart_j, cj, chart_k, ck)
                sign_ik = _get_sign_for_pair(signs, chart_i, ci, chart_k, ck)
                
                if sign_ij is None or sign_jk is None or sign_ik is None:
                    continue
                
                n_checked += 1
                
                # Cocycle condition: ω_ik = ω_ij · ω_jk
                # (This is equivalent to ω_ki = ω_kj · ω_ji by symmetry and commutativity)
                expected = sign_ij * sign_jk
                
                if sign_ik == expected:
                    if verbose:
                        print(f"  ✓ ({chart_i}.{ci}, {chart_j}.{cj}, {chart_k}.{ck}): "
                              f"ω_ij·ω_jk = {sign_ij}·{sign_jk} = {expected} = ω_ik")
                else:
                    violations.append((comp_i, comp_j, comp_k))
                    if verbose:
                        print(f"  ✗ ({chart_i}.{ci}, {chart_j}.{cj}, {chart_k}.{ck}): "
                              f"ω_ij·ω_jk = {sign_ij}·{sign_jk} = {expected} ≠ {sign_ik} = ω_ik")
    
    cocycle_verified = len(violations) == 0
    
    if n_checked == 0:
        print("\n  No triple overlaps found to verify")
    elif cocycle_verified:
        print(f"\n✓ Cocycle condition VERIFIED ({n_checked} triples checked)")
    else:
        print(f"\n✗ Cocycle condition VIOLATED ({len(violations)}/{n_checked} triples)")
        print("  This indicates training error - atlas is not well-formed!")
    
    return cocycle_verified, violations


def _get_sign_for_pair(
    signs: Dict,
    chart_i: int, comp_i: int,
    chart_j: int, comp_j: int
) -> Optional[int]:
    """
    Get the sign for a pair of chart components.
    
    Handles the fact that signs are stored with chart_i < chart_j convention
    and that there may be multiple overlap components.
    """
    # Try direct order
    for key, sign in signs.items():
        if key[:4] == (chart_i, comp_i, chart_j, comp_j):
            return sign
    
    # Try reversed order (sign is symmetric: ω_ji = ω_ij for real bundles)
    for key, sign in signs.items():
        if key[:4] == (chart_j, comp_j, chart_i, comp_i):
            return sign
    
    return None


def test_coboundary_condition(
    results: Dict,
    comp_list: List[Tuple[int, int]],
    verbose: bool = True
) -> Tuple[bool, Optional[Dict[Tuple[int, int], int]]]:
    """
    Test the coboundary condition via graph coloring.
    
    The sign cocycle is a coboundary iff we can assign orientations
    ν_i ∈ {±1} to each chart component such that ω_ji = ν_j · ν_i.
    
    CRITICAL: When the overlap between two charts has multiple connected
    components with DIFFERENT signs, this immediately implies non-orientability.
    This is exactly the Möbius band case: the overlap has two components,
    one with sign +1 and one with sign -1.
    
    Returns:
        (is_coboundary, orientations): Boolean and orientation assignment if successful
    """
    print("\n" + "="*60)
    print("STEP 4: Testing Coboundary Condition")
    print("="*60)
    
    signs = results['signs']
    
    # CRITICAL CHECK: Look for chart pairs with multiple overlap components
    # having different signs. This is an immediate obstruction to orientability.
    chart_pair_signs = {}  # {(comp_i, comp_j): [list of signs from different overlap components]}
    
    for key, sign in signs.items():
        chart_i, ci, chart_j, cj, overlap_comp = key
        comp_i = (chart_i, ci)
        comp_j = (chart_j, cj)
        pair_key = (comp_i, comp_j) if comp_i < comp_j else (comp_j, comp_i)
        
        if pair_key not in chart_pair_signs:
            chart_pair_signs[pair_key] = []
        chart_pair_signs[pair_key].append((overlap_comp, sign))
    
    # Check for inconsistent signs between same chart pair
    inconsistent_pairs = []
    for pair_key, overlap_signs in chart_pair_signs.items():
        unique_signs = set(s for _, s in overlap_signs)
        if len(unique_signs) > 1:
            inconsistent_pairs.append((pair_key, overlap_signs))
    
    if inconsistent_pairs:
        print("\n✗ IMMEDIATE NON-ORIENTABILITY DETECTED")
        print("  Found chart pairs with different signs on different overlap components:")
        for pair_key, overlap_signs in inconsistent_pairs:
            comp_i, comp_j = pair_key
            ci_chart, ci_id = comp_i
            cj_chart, cj_id = comp_j
            print(f"\n  Chart {ci_chart}.{ci_id} ↔ Chart {cj_chart}.{cj_id}:")
            for oc, s in overlap_signs:
                print(f"    Overlap component {oc}: ω = {s:+d}")
            print(f"  → Cannot satisfy ω = ν_j · ν_i with constant ν_i, ν_j")
        
        print("\n  This is the classic Möbius band / Klein bottle signature:")
        print("  The overlap has multiple components with opposite transition signs.")
        return False, None
    
    # Build adjacency with signs (now we know each pair has consistent sign)
    edges = {}
    for pair_key, overlap_signs in chart_pair_signs.items():
        # All signs are the same, just take the first
        edges[pair_key] = overlap_signs[0][1]
    
    # BFS to assign orientations
    orientations = {}
    is_consistent = True
    inconsistent_edge = None
    
    for start_comp in comp_list:
        if start_comp in orientations:
            continue
        
        # Start new connected component with arbitrary orientation
        queue = deque([start_comp])
        orientations[start_comp] = +1
        
        while queue:
            comp_i = queue.popleft()
            
            # Find all neighbors
            for (c1, c2), sign in edges.items():
                if c1 == comp_i:
                    comp_j = c2
                elif c2 == comp_i:
                    comp_j = c1
                else:
                    continue
                
                # Required: ω_ij = ν_j · ν_i, so ν_j = ω_ij · ν_i
                required_orientation = sign * orientations[comp_i]
                
                if comp_j in orientations:
                    # Check consistency
                    if orientations[comp_j] != required_orientation:
                        is_consistent = False
                        inconsistent_edge = (comp_i, comp_j, sign)
                        if verbose:
                            ci_chart, ci_id = comp_i
                            cj_chart, cj_id = comp_j
                            print(f"  ✗ Inconsistency detected:")
                            print(f"      Chart {ci_chart}.{ci_id} (ν={orientations[comp_i]:+d}) "
                                  f"↔ Chart {cj_chart}.{cj_id} (ν={orientations[comp_j]:+d})")
                            print(f"      Edge sign ω = {sign:+d}")
                            print(f"      Required: ν_j = ω·ν_i = {sign}·{orientations[comp_i]} = {required_orientation:+d}")
                            print(f"      Actual: ν_j = {orientations[comp_j]:+d}")
                else:
                    orientations[comp_j] = required_orientation
                    queue.append(comp_j)
    
    if is_consistent:
        print("\n✓ Coboundary test PASSED")
        print("  Found consistent orientation assignment:")
        for comp in sorted(orientations.keys()):
            chart, comp_id = comp
            print(f"    Chart {chart}.{comp_id}: ν = {orientations[comp]:+d}")
        return True, orientations
    else:
        print("\n✗ Coboundary test FAILED")
        print("  Cannot assign consistent orientations → NON-ORIENTABLE")
        return False, None


def check_mixed_signs(
    results: Dict,
    verbose: bool = True
) -> Tuple[bool, List, List]:
    """
    Check for sign issues within overlaps.
    
    There are two types of "mixed signs":
    
    1. Mixed signs WITHIN a single connected overlap component:
       This indicates numerical issues or that the component isn't truly connected.
       
    2. Different signs ACROSS multiple components of the same overlap:
       This is the classic non-orientability signature (Möbius band case).
       The overlap U_i ∩ U_j has multiple connected components with different
       transition signs.
    
    Returns:
        (has_mixed_within, mixed_within_list, different_across_list)
    """
    print("\n" + "="*60)
    print("STEP 5: Checking for Mixed Signs Within Overlaps")
    print("="*60)
    
    sign_info = results.get('sign_info', {})
    signs = results.get('signs', {})
    
    # Type 1: Mixed signs within a single overlap component
    mixed_within = []
    for key, info in sign_info.items():
        if info.get('has_mixed', False):
            mixed_within.append((key, info))
    
    # Type 2: Different signs across components of the same chart pair overlap
    chart_pair_signs = {}
    for key, sign in signs.items():
        chart_i, ci, chart_j, cj, overlap_comp = key
        pair_key = ((chart_i, ci), (chart_j, cj))
        if pair_key not in chart_pair_signs:
            chart_pair_signs[pair_key] = []
        chart_pair_signs[pair_key].append((overlap_comp, sign))
    
    different_across = []
    for pair_key, overlap_signs in chart_pair_signs.items():
        unique_signs = set(s for _, s in overlap_signs)
        if len(unique_signs) > 1:
            different_across.append((pair_key, overlap_signs))
    
    # Report findings
    if mixed_within:
        print(f"\n⚠ Found {len(mixed_within)} overlap components with mixed signs WITHIN:")
        for key, info in mixed_within[:5]:  # Show first 5
            chart_i, ci, chart_j, cj, oc = key
            print(f"  Chart {chart_i}.{ci} ↔ Chart {chart_j}.{cj} (component {oc}):")
            print(f"    Positive: {info['n_positive']}, Negative: {info['n_negative']}")
        if verbose:
            print("\n  This may indicate:")
            print("    1. Numerical precision issues")
            print("    2. Overlap component not truly connected")
    else:
        print("\n✓ No mixed signs within individual overlap components")
    
    if different_across:
        print(f"\n✗ Found {len(different_across)} chart pairs with DIFFERENT signs across overlap components:")
        for pair_key, overlap_signs in different_across:
            comp_i, comp_j = pair_key
            ci_chart, ci_id = comp_i
            cj_chart, cj_id = comp_j
            print(f"\n  Chart {ci_chart}.{ci_id} ↔ Chart {cj_chart}.{cj_id}:")
            for oc, s in overlap_signs:
                print(f"    Overlap component {oc}: ω = {s:+d}")
        print("\n  *** This is the classic NON-ORIENTABILITY signature! ***")
        print("  (Like the Möbius band: overlap has two components with opposite signs)")
    else:
        print("\n✓ All chart pairs have consistent signs across their overlap components")
    
    has_mixed_within = len(mixed_within) > 0
    return has_mixed_within, mixed_within, different_across


def check_orientability(
    system,
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    eps_cluster: float = 0.15,
    min_points: int = 20,
    eps_det: float = 1e-6,
    verbose: bool = True
) -> Dict:
    """
    Complete orientability detection for learned atlas.
    
    This implements the algorithm from the paper:
    1. Find connected components of each chart domain
    2. Compute transition signs ω_ji between all component pairs
    3. Verify cocycle condition: ω_ki = ω_kj · ω_ji
    4. Test coboundary condition: find {ν_i} with ω_ji = ν_j · ν_i
    5. Check for mixed signs within overlaps
    
    The manifold is orientable iff:
    - Cocycle condition is satisfied (necessary for well-formed atlas)
    - Coboundary condition is satisfied (sign cocycle is trivial in H¹)
    - No mixed signs within connected overlap components
    
    Args:
        system: Trained AtlasAutoencoder
        points: Full dataset
        subset_assignments: Chart domain assignments
        eps_cluster: DBSCAN epsilon for finding connected components
        min_points: Minimum points for a component to be considered
        eps_det: Threshold for considering a determinant as zero
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with complete analysis results
    """
    # Step 1: Find chart components
    chart_components = find_chart_components(
        points, subset_assignments, eps_cluster=eps_cluster, min_points=min_points
    )
    
    comp_list = list(chart_components.keys())
    
    # Step 2: Compute transition signs
    results = compute_component_transition_signs(
        points, chart_components, system, 
        eps_cluster=eps_cluster, min_points=min_points, eps_det=eps_det
    )
    
    # Step 3: Verify cocycle condition
    cocycle_verified, cocycle_violations = verify_cocycle_condition(
        results, comp_list, verbose=verbose
    )
    
    # Step 4: Test coboundary condition
    is_coboundary, orientations = test_coboundary_condition(
        results, comp_list, verbose=verbose
    )
    
    # Step 5: Check for mixed signs
    has_mixed_within, mixed_within, different_across = check_mixed_signs(results, verbose=verbose)
    
    # Different signs across overlap components is definitive non-orientability
    has_different_across = len(different_across) > 0
    
    # Final determination
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    
    if not cocycle_verified:
        print("⚠️  COCYCLE CONDITION VIOLATED - ATLAS IS NOT WELL-FORMED")
        print(f"  Found {len(cocycle_violations)} cocycle violations")
        print(f"  This indicates training error, not non-orientability!")
        print(f"\n  Recommendations:")
        print(f"    1. Train longer (increase epochs)")
        print(f"    2. Increase reconstruction loss weight")
        print(f"    3. Check for degenerate Jacobians (δ should be > 0)")
        print(f"    4. Verify chart overlaps are sufficient")
        print(f"\n  Cannot reliably determine orientability until cocycle is verified.")
        
        return {
            'is_orientable': None,  # Cannot determine
            'cocycle_verified': False,
            'coboundary_passed': None,
            'has_mixed_within': has_mixed_within,
            'has_different_across': has_different_across,
            'orientations': None,
            'cocycle_violations': cocycle_violations,
            'mixed_within': mixed_within,
            'different_across': different_across,
            'chart_components': chart_components,
            'signs': results['signs'],
            'sign_info': results.get('sign_info', {}),
            'overlaps': results['overlaps']
        }
    
    # Cocycle verified - determine orientability
    # Non-orientable if: coboundary failed OR different signs across overlap components
    is_orientable = is_coboundary and not has_different_across
    
    if is_orientable:
        print("✓ MANIFOLD IS ORIENTABLE")
        print(f"  - Cocycle condition verified ✓")
        print(f"  - Coboundary test passed ✓")
        print(f"  - Consistent signs across overlap components ✓")
        if has_mixed_within:
            print(f"  - Note: {len(mixed_within)} overlap components have internal mixed signs")
            print(f"    (This may indicate numerical issues but doesn't affect orientability)")
        print(f"\n  Chart component orientations:")
        for comp in sorted(orientations.keys()):
            chart, comp_id = comp
            print(f"    Chart {chart}.{comp_id}: ν = {orientations[comp]:+d}")
    else:
        print("✗ MANIFOLD IS NON-ORIENTABLE")
        print(f"  - Cocycle condition verified ✓")
        
        reasons = []
        if not is_coboundary:
            reasons.append("Coboundary test failed")
            print(f"  - Coboundary test failed ✗")
            print(f"      Cannot find consistent orientation assignment")
            print(f"      This means w₁(TM) ≠ 0 in H¹(M; ℤ/2)")
        else:
            print(f"  - Coboundary test passed ✓")
            
        if has_different_across:
            reasons.append(f"Different signs across {len(different_across)} overlap(s)")
            print(f"  - Different signs across overlap components ✗")
            print(f"      Found {len(different_across)} chart pair(s) with inconsistent signs:")
            for pair_key, overlap_signs in different_across[:3]:
                comp_i, comp_j = pair_key
                ci_chart, ci_id = comp_i
                cj_chart, cj_id = comp_j
                signs_str = ", ".join(f"comp {oc}: {s:+d}" for oc, s in overlap_signs)
                print(f"        Chart {ci_chart}.{ci_id} ↔ Chart {cj_chart}.{cj_id}: {signs_str}")
            print(f"      This is the classic Möbius band / ℝP² signature!")
        else:
            print(f"  - Consistent signs across overlap components ✓")
        
        if has_mixed_within:
            print(f"  - Note: {len(mixed_within)} components have internal mixed signs")
        
        print(f"\n  w₁(TM) ≠ 0 detected via: {', '.join(reasons)}")
    
    return {
        'is_orientable': is_orientable,
        'cocycle_verified': cocycle_verified,
        'coboundary_passed': is_coboundary,
        'has_mixed_within': has_mixed_within,
        'has_different_across': has_different_across,
        'orientations': orientations,
        'cocycle_violations': cocycle_violations,
        'mixed_within': mixed_within,
        'different_across': different_across,
        'chart_components': chart_components,
        'signs': results['signs'],
        'sign_info': results.get('sign_info', {}),
        'overlaps': results['overlaps']
    }


# Convenience function with old name for backward compatibility
def check_orientability_corrected(*args, **kwargs):
    """Alias for check_orientability (backward compatibility)."""
    return check_orientability(*args, **kwargs)


if __name__ == "__main__":
    print("""
Orientability Detection Module
==============================

Usage:
------
    from orientability import check_orientability
    
    results = check_orientability(
        system=atlas,              # Trained AtlasAutoencoder
        points=X,                  # Dataset
        subset_assignments=covers, # Chart domain assignments
        eps_cluster=0.15,          # DBSCAN epsilon
        min_points=20,             # Minimum points per component
        eps_det=1e-6,              # Determinant threshold
        verbose=True
    )
    
    # Results dictionary contains:
    # - is_orientable: True/False/None (None if cocycle violated)
    # - cocycle_verified: Whether cocycle condition holds
    # - coboundary_passed: Whether w₁ = 0
    # - has_mixed_signs: Whether any overlap has mixed signs
    # - orientations: {(chart, comp): ±1} if orientable
    # - signs: {overlap_key: sign} for all overlaps
    # - sign_info: Detailed sign analysis per overlap

Mathematical Background:
------------------------
A manifold M is orientable iff the first Stiefel-Whitney class w₁(TM) = 0.

For an autoencoder atlas with transition maps T_ji = E_j ∘ D_i, we compute:
- Linearized transitions: g_ji(x) = d(T_ji)_{E_i(x)}
- Sign cocycle: ω_ji(x) = sign(det g_ji(x))

The sign cocycle satisfies:
- Cocycle condition: ω_ki = ω_kj · ω_ji (necessary for well-defined atlas)
- Coboundary iff ∃{ν_i} with ω_ji = ν_j · ν_i (equivalent to w₁ = 0)

This implementation:
1. Handles disconnected chart domains correctly
2. Detects mixed signs within overlaps
3. Provides detailed diagnostics for debugging
""")