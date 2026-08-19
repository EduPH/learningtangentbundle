"""
Good Cover Verification
=======================

A good cover (or Leray cover) of a topological space X is an open cover 
{U_i} such that all finite nonempty intersections are contractible.

For data sampled from a manifold, we verify the good cover property by checking:
1. Each chart U_i is connected (necessary for contractibility)
2. Each pairwise intersection U_i ∩ U_j is connected
3. Each triple intersection U_i ∩ U_j ∩ U_k is connected
4. Higher intersections are connected (or empty)

Contractibility is stronger than connectivity, but for "nice" subsets of manifolds
(geodesically convex, star-shaped, etc.), connectivity is a reasonable proxy.

For rigorous verification, we also check:
- Homology of each intersection (contractible ⟹ H_k = 0 for k > 0)
- This requires computing persistent homology on the point cloud

Author: For atlas autoencoder project
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from itertools import combinations
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import warnings


# ============================================================
# Connected Component Analysis
# ============================================================

def count_connected_components(
    points: np.ndarray,
    eps: float = None,
    min_samples: int = 5,
    metric: str = 'euclidean'
) -> Tuple[int, np.ndarray]:
    """
    Count connected components in a point cloud using DBSCAN.
    
    Args:
        points: Point cloud (N, d)
        eps: DBSCAN epsilon. If None, auto-computed from data.
        min_samples: DBSCAN min_samples parameter
        metric: Distance metric
        
    Returns:
        n_components: Number of connected components
        labels: Component labels for each point (-1 for noise)
    """
    if len(points) < min_samples:
        return 1 if len(points) > 0 else 0, np.zeros(len(points), dtype=int)
    
    # Auto-compute eps if not provided
    if eps is None:
        # Use k-th nearest neighbor distance as heuristic
        k = min(min_samples, len(points) - 1)
        dists = squareform(pdist(points, metric=metric))
        np.fill_diagonal(dists, np.inf)
        knn_dists = np.sort(dists, axis=1)[:, k-1]
        eps = np.percentile(knn_dists, 90)  # 90th percentile of k-NN distances
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = clustering.fit_predict(points)
    
    # Count components (excluding noise label -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_components = len(unique_labels)
    
    # If all points are noise, treat as 1 component
    if n_components == 0 and len(points) > 0:
        n_components = 1
        labels = np.zeros(len(points), dtype=int)
    
    return n_components, labels


# ============================================================
# Good Cover Verification
# ============================================================

class GoodCoverChecker:
    """
    Verify if a cover satisfies the good cover property.
    
    A good cover requires all finite nonempty intersections to be contractible.
    We check this via:
    1. Connectivity (necessary condition)
    2. Optionally, homology (sufficient condition via H_k = 0 for k > 0)
    """
    
    def __init__(
        self,
        points: np.ndarray,
        subset_assignments: List[np.ndarray],
        eps: float = None,
        min_samples: int = 5
    ):
        """
        Args:
            points: Full point cloud (N, d)
            subset_assignments: List of index arrays, one per chart
            eps: DBSCAN epsilon for connectivity check
            min_samples: DBSCAN min_samples
        """
        self.points = points
        self.subset_assignments = subset_assignments
        self.n_charts = len(subset_assignments)
        self.eps = eps
        self.min_samples = min_samples
        
        # Results storage
        self.chart_components = {}
        self.pairwise_components = {}
        self.triple_components = {}
        self.higher_components = {}
        
        self.is_good_cover = None
        self.failure_reasons = []
    
    def check_charts(self, verbose: bool = True) -> bool:
        """Check if each chart domain is connected."""
        all_connected = True
        
        if verbose:
            print("="*60)
            print("CHECKING CHART CONNECTIVITY")
            print("="*60)
        
        for i, indices in enumerate(self.subset_assignments):
            if len(indices) == 0:
                self.chart_components[i] = (0, np.array([]))
                if verbose:
                    print(f"  Chart {i}: EMPTY")
                continue
            
            chart_points = self.points[indices]
            n_comp, labels = count_connected_components(
                chart_points, eps=self.eps, min_samples=self.min_samples
            )
            self.chart_components[i] = (n_comp, labels)
            
            if n_comp != 1:
                all_connected = False
                self.failure_reasons.append(f"Chart {i} has {n_comp} components (should be 1)")
            
            if verbose:
                status = "✓" if n_comp == 1 else "✗"
                print(f"  {status} Chart {i}: {len(indices)} points, {n_comp} component(s)")
        
        return all_connected
    
    def check_pairwise_intersections(self, verbose: bool = True) -> bool:
        """Check if each pairwise intersection is connected."""
        all_connected = True
        
        if verbose:
            print("\n" + "="*60)
            print("CHECKING PAIRWISE INTERSECTION CONNECTIVITY")
            print("="*60)
        
        for i, j in combinations(range(self.n_charts), 2):
            # Compute intersection
            intersection = np.intersect1d(
                self.subset_assignments[i],
                self.subset_assignments[j]
            )
            
            if len(intersection) == 0:
                self.pairwise_components[(i, j)] = (0, np.array([]))
                if verbose:
                    print(f"  Chart {i} ∩ Chart {j}: EMPTY (OK)")
                continue
            
            intersection_points = self.points[intersection]
            n_comp, labels = count_connected_components(
                intersection_points, eps=self.eps, min_samples=self.min_samples
            )
            self.pairwise_components[(i, j)] = (n_comp, labels)
            
            if n_comp != 1:
                all_connected = False
                self.failure_reasons.append(
                    f"U_{i} ∩ U_{j} has {n_comp} components (should be 1)"
                )
            
            if verbose:
                status = "✓" if n_comp == 1 else "✗"
                print(f"  {status} U_{i} ∩ U_{j}: {len(intersection)} points, {n_comp} component(s)")
        
        return all_connected
    
    def check_triple_intersections(self, verbose: bool = True) -> bool:
        """Check if each triple intersection is connected (or empty)."""
        all_ok = True
        
        if verbose:
            print("\n" + "="*60)
            print("CHECKING TRIPLE INTERSECTION CONNECTIVITY")
            print("="*60)
        
        for i, j, k in combinations(range(self.n_charts), 3):
            # Compute intersection
            intersection = np.intersect1d(
                self.subset_assignments[i],
                np.intersect1d(
                    self.subset_assignments[j],
                    self.subset_assignments[k]
                )
            )
            
            if len(intersection) == 0:
                self.triple_components[(i, j, k)] = (0, np.array([]))
                if verbose:
                    print(f"  U_{i} ∩ U_{j} ∩ U_{k}: EMPTY (OK)")
                continue
            
            intersection_points = self.points[intersection]
            n_comp, labels = count_connected_components(
                intersection_points, eps=self.eps, min_samples=self.min_samples
            )
            self.triple_components[(i, j, k)] = (n_comp, labels)
            
            if n_comp != 1:
                all_ok = False
                self.failure_reasons.append(
                    f"U_{i} ∩ U_{j} ∩ U_{k} has {n_comp} components (should be 0 or 1)"
                )
            
            if verbose:
                status = "✓" if n_comp <= 1 else "✗"
                print(f"  {status} U_{i} ∩ U_{j} ∩ U_{k}: {len(intersection)} points, {n_comp} component(s)")
        
        return all_ok
    
    def check_higher_intersections(self, max_order: int = None, verbose: bool = True) -> bool:
        """Check higher-order intersections (4-fold, 5-fold, etc.)."""
        if max_order is None:
            max_order = self.n_charts
        
        all_ok = True
        
        if verbose and max_order > 3:
            print("\n" + "="*60)
            print("CHECKING HIGHER-ORDER INTERSECTIONS")
            print("="*60)
        
        for order in range(4, min(max_order + 1, self.n_charts + 1)):
            for indices in combinations(range(self.n_charts), order):
                # Compute intersection
                intersection = self.subset_assignments[indices[0]]
                for idx in indices[1:]:
                    intersection = np.intersect1d(intersection, self.subset_assignments[idx])
                
                if len(intersection) == 0:
                    self.higher_components[indices] = (0, np.array([]))
                    continue
                
                intersection_points = self.points[intersection]
                n_comp, labels = count_connected_components(
                    intersection_points, eps=self.eps, min_samples=self.min_samples
                )
                self.higher_components[indices] = (n_comp, labels)
                
                if n_comp > 1:
                    all_ok = False
                    indices_str = ', '.join(map(str, indices))
                    self.failure_reasons.append(
                        f"Intersection ({indices_str}) has {n_comp} components"
                    )
                
                if verbose and len(intersection) > 0:
                    indices_str = ' ∩ '.join([f'U_{i}' for i in indices])
                    status = "✓" if n_comp <= 1 else "✗"
                    print(f"  {status} {indices_str}: {len(intersection)} points, {n_comp} component(s)")
        
        return all_ok
    
    def check_all(self, verbose: bool = True) -> bool:
        """
        Run all good cover checks.
        
        Returns:
            True if the cover satisfies good cover property (all intersections connected)
        """
        self.failure_reasons = []
        
        if verbose:
            print("\n" + "="*70)
            print("  GOOD COVER VERIFICATION")
            print("="*70)
        
        charts_ok = self.check_charts(verbose=verbose)
        pairwise_ok = self.check_pairwise_intersections(verbose=verbose)
        triple_ok = self.check_triple_intersections(verbose=verbose)
        higher_ok = self.check_higher_intersections(verbose=verbose)
        
        self.is_good_cover = charts_ok and pairwise_ok and triple_ok and higher_ok
        
        if verbose:
            print("\n" + "="*70)
            print("  SUMMARY")
            print("="*70)
            print(f"\n  Charts connected:              {'✓' if charts_ok else '✗'}")
            print(f"  Pairwise intersections connected: {'✓' if pairwise_ok else '✗'}")
            print(f"  Triple intersections connected:   {'✓' if triple_ok else '✗'}")
            print(f"  Higher intersections connected:   {'✓' if higher_ok else '✗'}")
            
            print(f"\n  {'='*50}")
            if self.is_good_cover:
                print("  ✓ THIS IS A GOOD COVER")
                print("    All finite intersections are connected.")
            else:
                print("  ✗ THIS IS NOT A GOOD COVER")
                print("    Failure reasons:")
                for reason in self.failure_reasons:
                    print(f"      - {reason}")
            print(f"  {'='*50}")
        
        return self.is_good_cover
    
    def get_problematic_intersections(self) -> Dict:
        """Return all intersections with more than 1 connected component."""
        problematic = {}
        
        for key, (n_comp, labels) in self.chart_components.items():
            if n_comp > 1:
                problematic[('chart', key)] = n_comp
        
        for key, (n_comp, labels) in self.pairwise_components.items():
            if n_comp > 1:
                problematic[('pairwise', key)] = n_comp
        
        for key, (n_comp, labels) in self.triple_components.items():
            if n_comp > 1:
                problematic[('triple', key)] = n_comp
        
        for key, (n_comp, labels) in self.higher_components.items():
            if n_comp > 1:
                problematic[('higher', key)] = n_comp
        
        return problematic


# ============================================================
# Homology-based verification (optional, requires ripser)
# ============================================================

def check_contractibility_via_homology(
    points: np.ndarray,
    max_dim: int = 1,
    threshold: float = 0.1
) -> Tuple[bool, Dict]:
    """
    Check if a point cloud is approximately contractible using persistent homology.
    
    A contractible space has trivial homology: H_0 = Z, H_k = 0 for k > 0.
    
    We check:
    - H_0 has exactly 1 persistent component (connected)
    - H_k for k > 0 has no significant persistent features
    
    Args:
        points: Point cloud
        max_dim: Maximum homology dimension to compute
        threshold: Persistence threshold (features with persistence < threshold ignored)
        
    Returns:
        is_contractible: True if homology suggests contractibility
        homology_info: Dictionary with persistence diagram info
    """
    try:
        from ripser import ripser
    except ImportError:
        warnings.warn("ripser not installed. Skipping homology check.")
        return None, {}
    
    if len(points) < 3:
        return True, {'H0': 1, 'higher': 0}
    
    # Compute persistent homology
    result = ripser(points, maxdim=max_dim)
    diagrams = result['dgms']
    
    homology_info = {}
    is_contractible = True
    
    # Check H_0: should have exactly 1 component (1 point at infinity)
    h0 = diagrams[0]
    # Count components that persist significantly
    h0_persistent = np.sum((h0[:, 1] - h0[:, 0]) > threshold)
    # The one infinite bar represents the single connected component
    h0_infinite = np.sum(np.isinf(h0[:, 1]))
    homology_info['H0_components'] = h0_infinite
    
    if h0_infinite != 1:
        is_contractible = False
    
    # Check H_k for k > 0: should have no significant features
    for k in range(1, max_dim + 1):
        if k < len(diagrams):
            hk = diagrams[k]
            # Count features with persistence above threshold
            if len(hk) > 0:
                persistence = hk[:, 1] - hk[:, 0]
                significant = np.sum(persistence > threshold)
            else:
                significant = 0
            
            homology_info[f'H{k}_features'] = significant
            
            if significant > 0:
                is_contractible = False
    
    return is_contractible, homology_info


def verify_good_cover_with_homology(
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    max_dim: int = 1,
    threshold: float = 0.1,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Verify good cover property using persistent homology.
    
    This is more rigorous than connectivity checking, as it verifies
    that each intersection has trivial higher homology.
    
    Args:
        points: Full point cloud
        subset_assignments: Chart assignments
        max_dim: Maximum homology dimension
        threshold: Persistence threshold
        verbose: Print progress
        
    Returns:
        is_good_cover: True if all intersections are contractible
        results: Dictionary with homology info for each intersection
    """
    try:
        from ripser import ripser
    except ImportError:
        warnings.warn("ripser not installed. Cannot verify via homology.")
        return None, {}
    
    n_charts = len(subset_assignments)
    results = {}
    is_good_cover = True
    
    if verbose:
        print("\n" + "="*70)
        print("  GOOD COVER VERIFICATION VIA PERSISTENT HOMOLOGY")
        print("="*70)
    
    # Check charts
    if verbose:
        print("\nChecking charts:")
    for i, indices in enumerate(subset_assignments):
        if len(indices) < 3:
            results[f'chart_{i}'] = {'contractible': len(indices) > 0, 'reason': 'too few points'}
            continue
        
        chart_points = points[indices]
        contractible, info = check_contractibility_via_homology(
            chart_points, max_dim=max_dim, threshold=threshold
        )
        results[f'chart_{i}'] = {'contractible': contractible, 'homology': info}
        
        if not contractible:
            is_good_cover = False
        
        if verbose:
            status = "✓" if contractible else "✗"
            print(f"  {status} Chart {i}: {info}")
    
    # Check pairwise intersections
    if verbose:
        print("\nChecking pairwise intersections:")
    for i, j in combinations(range(n_charts), 2):
        intersection = np.intersect1d(subset_assignments[i], subset_assignments[j])
        
        if len(intersection) < 3:
            results[f'U{i}_U{j}'] = {'contractible': True, 'reason': 'empty or trivial'}
            if verbose:
                print(f"  ✓ U_{i} ∩ U_{j}: empty or trivial")
            continue
        
        intersection_points = points[intersection]
        contractible, info = check_contractibility_via_homology(
            intersection_points, max_dim=max_dim, threshold=threshold
        )
        results[f'U{i}_U{j}'] = {'contractible': contractible, 'homology': info}
        
        if not contractible:
            is_good_cover = False
        
        if verbose:
            status = "✓" if contractible else "✗"
            print(f"  {status} U_{i} ∩ U_{j}: {info}")
    
    # Check triple intersections
    if verbose:
        print("\nChecking triple intersections:")
    for i, j, k in combinations(range(n_charts), 3):
        intersection = np.intersect1d(
            subset_assignments[i],
            np.intersect1d(subset_assignments[j], subset_assignments[k])
        )
        
        if len(intersection) < 3:
            results[f'U{i}_U{j}_U{k}'] = {'contractible': True, 'reason': 'empty or trivial'}
            if verbose and len(intersection) > 0:
                print(f"  ✓ U_{i} ∩ U_{j} ∩ U_{k}: trivial ({len(intersection)} points)")
            continue
        
        intersection_points = points[intersection]
        contractible, info = check_contractibility_via_homology(
            intersection_points, max_dim=max_dim, threshold=threshold
        )
        results[f'U{i}_U{j}_U{k}'] = {'contractible': contractible, 'homology': info}
        
        if not contractible:
            is_good_cover = False
        
        if verbose:
            status = "✓" if contractible else "✗"
            print(f"  {status} U_{i} ∩ U_{j} ∩ U_{k}: {info}")
    
    if verbose:
        print(f"\n{'='*50}")
        if is_good_cover:
            print("✓ GOOD COVER VERIFIED (all intersections contractible)")
        else:
            print("✗ NOT A GOOD COVER (some intersections have non-trivial homology)")
        print(f"{'='*50}")
    
    return is_good_cover, results


# ============================================================
# Convenience function
# ============================================================

def is_good_cover(
    points: np.ndarray,
    subset_assignments: List[np.ndarray],
    method: str = 'connectivity',
    eps: float = None,
    verbose: bool = True
) -> bool:
    """
    Check if a cover is a good cover.
    
    Args:
        points: Full point cloud (N, d)
        subset_assignments: List of index arrays, one per chart
        method: 'connectivity' (fast) or 'homology' (rigorous, requires ripser)
        eps: DBSCAN epsilon for connectivity method
        verbose: Print progress
        
    Returns:
        True if the cover is a good cover
    """
    if method == 'connectivity':
        checker = GoodCoverChecker(points, subset_assignments, eps=eps)
        return checker.check_all(verbose=verbose)
    elif method == 'homology':
        is_good, _ = verify_good_cover_with_homology(
            points, subset_assignments, verbose=verbose
        )
        return is_good
    else:
        raise ValueError(f"Unknown method: {method}. Use 'connectivity' or 'homology'.")


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    print("Good Cover Verification Module")
    print("="*50)
    
    # Example: Create a simple cover of S^2
    print("\nExample: Tetrahedral cover of S^2")
    
    # Sample sphere
    np.random.seed(42)
    n_points = 1000
    points = np.random.randn(n_points, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Tetrahedral vertices
    vertices = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Create cover
    epsilon = 0.3
    subset_assignments = []
    for v in vertices:
        mask = points @ v > -epsilon
        subset_assignments.append(np.where(mask)[0])
    
    # Check if it's a good cover
    print("\nUsing connectivity method:")
    checker = GoodCoverChecker(points, subset_assignments, eps=0.3)
    is_good = checker.check_all(verbose=True)
    
    # Try homology method if ripser is available
    try:
        print("\n\nUsing homology method:")
        is_good_h, results = verify_good_cover_with_homology(
            points, subset_assignments, verbose=True
        )
    except Exception as e:
        print(f"\nHomology check skipped: {e}")