import numpy as np
from typing import List, Tuple
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dreimac import (
    ProjectiveCoords, GeometryUtils, GeometryExamples,
    PlotUtils, ProjectiveMapUtils
)


def create_cover(X, n_charts=6, threshold_percentile=70,n_landmarks = 150,n_neighbors = 20):
    """
    Create chart cover using Perea's projective coordinates method.
    
    This approach uses GEODESIC DISTANCE epsilon-balls for chart construction:
    1. Computes landmark geodesic distances (sparse k-NN graph)
    2. Extracts projective coordinates via persistent cohomology (for visualization)
    3. Selects chart centers from landmarks
    4. Creates epsilon-ball charts using GEODESIC distances (not Euclidean!)
    
    The key insight: Geodesic distance respects the manifold's intrinsic geometry,
    while Euclidean distance in projective space does not.
    
    Args:
        X: Data points (line patches)
        n_charts: Number of charts to create
        threshold_percentile: Percentile for epsilon (50-80 recommended)
                            Higher = larger charts = more overlap
        
    Returns:
        subset_assignments: List of index arrays for each chart
        X_reordered: Data reordered by landmark proximity
        proj_coords: Projective coordinates for visualization
        pointcloud_permutation: Permutation used for reordering
        dist_mat: Geodesic distance matrix (for reference)
    """
    print("\n" + "="*60)
    print("CONSTRUCTING PEREA PROJECTIVE COORDINATE COVER")
    print("="*60)
    
    # Step 1: Compute landmark geodesic distances (Perea's approach)
    print("\n1. Computing landmark geodesic distances...")
    
    
    
    dist_mat, pointcloud_permutation = GeometryUtils.landmark_geodesic_distance(
        X, n_landmarks, n_neighbors
    )
    print(f"   Distance matrix shape: {dist_mat.shape}")
    print(f"   Landmarks: {n_landmarks}")
    print(f"   Neighbors per landmark: {n_neighbors}")
    
    # Step 2: Extract projective coordinates via persistent cohomology
    print("\n2. Computing projective coordinates...")
    pc = ProjectiveCoords(dist_mat, n_landmarks=n_landmarks, distance_matrix=True)
    proj_coords = pc.get_coordinates(proj_dim=2, perc=0.8, cocycle_idx=0)
    print(f"   Projective coordinate shape: {proj_coords.shape}")
    
    # Step 3: Greedy permutation for landmark selection
    print("\n3. Selecting landmarks via greedy permutation...")
    subsample = GeometryUtils.get_greedy_perm_pc(proj_coords, 300)['perm']
    print(f"   Selected {len(subsample)} landmarks")
    
    # Visualize projective coordinates (Perea's stereographic projection)
    stereo_projection_dim_red_subsample = ProjectiveMapUtils.get_stereo_proj_codim1(
        proj_coords[subsample, :]
    )
    
    plt.figure(figsize=(8, 8))
    PlotUtils.imscatter(
        stereo_projection_dim_red_subsample, 
        X[pointcloud_permutation][subsample], 
        10
    )
    _ = PlotUtils.plot_proj_boundary()
    plt.title("Projective Coordinates (Perea's Method)")
    plt.show()
    
    # Step 4: Reorder data by landmark proximity
    X_reordered = X[pointcloud_permutation]
    
    # Step 5: Create charts using geodesic distance epsilon-balls
    print("\n4. Creating charts from geodesic distance epsilon-balls...")
    
    # The distance matrix dist_mat has shape (n_landmarks, n_points)
    # dist_mat[i, j] = geodesic distance from landmark i to point j
    
    # Select chart centers from landmarks
    # Use greedy permutation or just take first n_charts landmarks
    chart_center_indices = list(range(1, min(n_charts + 1, n_landmarks)))  # Skip landmark 0
    
    print(f"   Using landmarks {chart_center_indices} as chart centers")
    
    # For each chart center, create epsilon-ball
    # Epsilon is chosen based on percentile of distances
    
    # Strategy: For each landmark, look at distances to all points
    # Choose epsilon such that a certain percentile of points are included
    
    subset_assignments = []
    
    # Compute epsilon based on the distribution of distances
    # We want overlapping charts, so use a percentile that creates generous balls
    
    # Collect all distances from chart centers
    all_center_distances = []
    for center_idx in chart_center_indices:
        distances = dist_mat[center_idx, :]
        all_center_distances.extend(distances[distances > 0])  # Exclude self-distance
    
    # Choose epsilon as a percentile of these distances
    # Higher percentile = larger epsilon = bigger charts = more overlap
    epsilon = np.percentile(all_center_distances, threshold_percentile)
    
    print(f"   Epsilon (percentile {threshold_percentile}): {epsilon:.4f}")
    
    for i, center_idx in enumerate(chart_center_indices):
        # Get distances from this landmark to all points
        distances = dist_mat[center_idx, :]
        
        # Include all points within epsilon of this center
        in_chart = np.where(distances <= epsilon)[0]
        
        subset_assignments.append(in_chart)
        pct = 100 * len(in_chart) / len(X_reordered)
        print(f"   Chart {i} (landmark {center_idx}): {len(in_chart)} points ({pct:.1f}%)")
    
    # Verify coverage
    all_covered = set()
    for indices in subset_assignments:
        all_covered.update(indices)
    
    coverage = 100 * len(all_covered) / len(X_reordered)
    print(f"\n   Coverage: {len(all_covered)}/{len(X_reordered)} points ({coverage:.1f}%)")
    
    if coverage < 95:
        print(f"   ⚠️  Warning: Coverage < 95%")
        print(f"      Increase threshold_percentile or add more charts")
    
    # Check overlaps
    print("\n5. Verifying chart overlaps...")
    total_overlaps = 0
    for i in range(n_charts):
        for j in range(i + 1, n_charts):
            overlap = np.intersect1d(subset_assignments[i], subset_assignments[j])
            if len(overlap) > 0:
                pct = 100 * len(overlap) / len(X_reordered)
                print(f"   Charts {i} ∩ {j}: {len(overlap)} points ({pct:.1f}%)")
                total_overlaps += 1
    
    print(f"\n   Total pairwise overlaps: {total_overlaps}")
    
    # Check triple overlaps
    triple_count = 0
    for i in range(n_charts):
        for j in range(i + 1, n_charts):
            for k in range(j + 1, n_charts):
                triple = np.intersect1d(
                    subset_assignments[i],
                    np.intersect1d(subset_assignments[j], subset_assignments[k])
                )
                if len(triple) > 20:  # Minimum size for cocycle verification
                    triple_count += 1
    
    print(f"   Triple intersections (>20 points): {triple_count}")
    
    return subset_assignments, X_reordered, proj_coords, pointcloud_permutation, dist_mat

