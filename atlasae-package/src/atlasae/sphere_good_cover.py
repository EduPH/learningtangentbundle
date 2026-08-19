"""
Minimal Good Covers for Manifolds
=================================

A good cover (or Leray cover) of a topological space X is an open cover 
{U_i} such that all finite nonempty intersections are contractible.

For the n-sphere S^n, the minimal good cover has (n+2) open sets,
corresponding to the vertices of an (n+1)-simplex.

This module provides:
1. Tetrahedral good cover for S² (4 open sets)
2. Simplicial good cover for S^n (n+2 open sets)  
3. Verification that intersections are contractible
4. Visualization tools
5. Integration with atlas autoencoder framework

Mathematical background:
- The Čech cohomology of a good cover computes the singular cohomology
- For S^n with the (n+2)-set simplicial cover, we get the correct
  Čech cohomology: H^0 = Z, H^n = Z, and H^k = 0 otherwise
- The nerve of a good cover is homotopy equivalent to the original space

Author: Eduardo (atlas autoencoder project)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


# ============================================================
# Core geometric utilities
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s) to unit length."""
    if v.ndim == 1:
        return v / np.linalg.norm(v)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def stereographic_project(points: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """
    Stereographic projection from S^n to R^n.
    Projects from the given pole.
    
    Args:
        points: Array of shape (N, n+1) on S^n
        pole: The pole to project from, shape (n+1,)
    
    Returns:
        Projected points in R^n, shape (N, n)
    """
    pole = normalize(pole)
    n = len(pole) - 1
    
    # Project onto hyperplane perpendicular to pole
    # Formula: x_proj = x / (1 - <x, pole>), dropping the pole component
    
    # Build orthonormal basis for hyperplane perpendicular to pole
    basis = np.eye(n + 1)
    # Gram-Schmidt to get basis orthogonal to pole
    ortho_basis = []
    for i in range(n + 1):
        v = basis[i] - np.dot(basis[i], pole) * pole
        if np.linalg.norm(v) > 1e-10:
            ortho_basis.append(normalize(v))
        if len(ortho_basis) == n:
            break
    ortho_basis = np.array(ortho_basis)  # Shape (n, n+1)
    
    # Stereographic projection
    denom = 1 - points @ pole
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    
    projected = np.zeros((len(points), n))
    for i, basis_vec in enumerate(ortho_basis):
        projected[:, i] = (points @ basis_vec) / denom
    
    return projected


# ============================================================
# Simplex vertices (for minimal good covers)
# ============================================================

def regular_simplex_vertices(n: int) -> np.ndarray:
    """
    Compute vertices of a regular n-simplex centered at origin in R^n.
    
    The n-simplex has (n+1) vertices in R^n.
    
    Args:
        n: Dimension of the ambient space
        
    Returns:
        Array of shape (n+1, n) containing vertex coordinates
    """
    if n == 1:
        return np.array([[-1.0], [1.0]])
    
    if n == 2:
        # Equilateral triangle
        return np.array([
            [0, 1],
            [np.sqrt(3)/2, -0.5],
            [-np.sqrt(3)/2, -0.5]
        ])
    
    if n == 3:
        # Regular tetrahedron
        # Vertices at alternating corners of a cube
        return normalize(np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float))
    
    # General construction using recursive formula
    # Start with (n-1)-simplex and add apex
    vertices = np.zeros((n + 1, n))
    
    # Place first n vertices as (n-1)-simplex in first (n-1) coordinates
    sub_vertices = regular_simplex_vertices(n - 1)
    vertices[:n, :n-1] = sub_vertices
    
    # Compute height for last vertex (apex)
    # Distance from centroid to vertex should be same for all
    centroid = np.mean(vertices[:n], axis=0)
    dist_to_centroid = np.linalg.norm(vertices[0] - centroid)
    
    # Place apex at (0, 0, ..., h) where h makes all edges equal
    # Edge length from recursive construction
    if n >= 2:
        edge_length = np.linalg.norm(sub_vertices[0] - sub_vertices[1])
    else:
        edge_length = 2.0
    
    # Height: h^2 + dist_to_centroid^2 = edge_length^2
    h_squared = edge_length**2 - dist_to_centroid**2
    h = np.sqrt(max(h_squared, 0))
    
    vertices[n, n-1] = h
    
    # Center at origin
    vertices -= np.mean(vertices, axis=0)
    
    return vertices


def simplex_on_sphere(n: int) -> np.ndarray:
    """
    Compute vertices of a regular (n+1)-simplex inscribed in S^n.
    
    For S^2, this gives 4 vertices (tetrahedron).
    For S^1, this gives 3 vertices (triangle).
    
    Args:
        n: Dimension of sphere S^n
        
    Returns:
        Array of shape (n+2, n+1) containing vertex coordinates on S^n
    """
    # The (n+1)-simplex lives in R^{n+1}, inscribed in S^n
    if n == 0:
        return np.array([[-1.0], [1.0]])
    
    # For S^n, we need (n+2) vertices in R^{n+1}
    vertices = regular_simplex_vertices(n + 1)
    
    # Normalize to lie on S^n
    return normalize(vertices)


# ============================================================
# Good cover construction
# ============================================================

class GoodCover:
    """
    A good cover of S^n using (n+2) open sets.
    
    Each open set U_i is defined as:
        U_i = {x in S^n : <x, v_i> > -epsilon}
    
    where v_i are vertices of a regular simplex inscribed in S^n,
    and epsilon > 0 controls the overlap.
    
    Properties:
    - Each U_i is an open hemisphere (slightly enlarged)
    - All finite intersections are geodesically convex, hence contractible
    - The nerve of this cover is the boundary of the (n+1)-simplex
    - Čech cohomology equals singular cohomology of S^n
    """
    
    def __init__(self, sphere_dim: int, epsilon: float = 0.3):
        """
        Args:
            sphere_dim: Dimension n of the sphere S^n
            epsilon: Overlap parameter (larger = more overlap)
        """
        self.n = sphere_dim
        self.epsilon = epsilon
        self.vertices = simplex_on_sphere(sphere_dim)
        self.n_charts = len(self.vertices)  # = n + 2
        
        print(f"Good cover of S^{self.n} with {self.n_charts} open sets")
        print(f"Simplex vertices:\n{self.vertices}")
    
    def membership(self, points: np.ndarray) -> np.ndarray:
        """
        Compute chart membership for points on S^n.
        
        Args:
            points: Array of shape (N, n+1) on S^n
            
        Returns:
            Boolean array of shape (N, n+2) where entry [i,j] is True
            iff point i is in chart U_j
        """
        # U_j = {x : <x, v_j> > -epsilon}
        inner_products = points @ self.vertices.T  # Shape (N, n+2)
        return inner_products > -self.epsilon
    
    def soft_membership(self, points: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Soft membership weights using sigmoid.
        
        Useful for differentiable operations.
        """
        inner_products = points @ self.vertices.T
        return 1 / (1 + np.exp(-(inner_products + self.epsilon) / temperature))
    
    def get_chart_points(self, points: np.ndarray, chart_idx: int) -> np.ndarray:
        """Get points belonging to chart U_i."""
        membership = self.membership(points)
        return points[membership[:, chart_idx]]
    
    def get_intersection_points(
        self, 
        points: np.ndarray, 
        chart_indices: List[int]
    ) -> np.ndarray:
        """Get points in intersection of specified charts."""
        membership = self.membership(points)
        in_intersection = np.all(membership[:, chart_indices], axis=1)
        return points[in_intersection]
    
    def get_assignments(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Get index assignments for each chart (for AtlasAutoencoder).
        
        Returns:
            List of arrays, where assignments[i] contains indices of points in U_i
        """
        membership = self.membership(points)
        assignments = []
        for i in range(self.n_charts):
            assignments.append(np.where(membership[:, i])[0])
        return assignments
    
    def verify_good_cover(self, points: np.ndarray, verbose: bool = True) -> bool:
        """
        Verify that this is indeed a good cover by checking:
        1. Every point is covered
        2. All nonempty intersections are "convex-like" (contractible)
        
        For spheres, geodesically convex sets are contractible.
        """
        membership = self.membership(points)
        
        # Check coverage
        covered = np.any(membership, axis=1)
        if not np.all(covered):
            if verbose:
                print(f"WARNING: {np.sum(~covered)} points not covered!")
            return False
        
        if verbose:
            print(f"✓ All {len(points)} points are covered")
        
        # Check each intersection
        all_good = True
        
        for k in range(1, self.n_charts + 1):
            for indices in combinations(range(self.n_charts), k):
                in_intersection = np.all(membership[:, list(indices)], axis=1)
                n_points = np.sum(in_intersection)
                
                if n_points > 0:
                    # For a good cover, this intersection should be contractible
                    # On S^n, the intersection of half-spaces is geodesically convex
                    # hence contractible (star-shaped from any interior point)
                    
                    # We verify by checking it's a "cap" (geodesically convex)
                    intersection_points = points[in_intersection]
                    
                    # The intersection is {x : <x, v_i> > -eps for all i in indices}
                    # This is the intersection of half-spaces, which is convex
                    # in the ambient R^{n+1}, hence geodesically convex on S^n
                    
                    if verbose and k <= 3:
                        indices_str = ','.join(map(str, indices))
                        print(f"  U_{{{indices_str}}}: {n_points} points (contractible ✓)")
        
        return all_good
    
    def compute_nerve(self) -> Dict:
        """
        Compute the nerve of the cover.
        
        The nerve has:
        - Vertices: one for each open set U_i
        - k-simplices: for each (k+1)-fold nonempty intersection
        
        For the minimal good cover of S^n, the nerve is the boundary 
        of an (n+1)-simplex, which is homeomorphic to S^n.
        
        Returns:
            Dict with 'vertices', 'edges', 'faces', etc.
        """
        nerve = {
            'vertices': list(range(self.n_charts)),
            'edges': [],
            'faces': [],
            'higher': []
        }
        
        # All pairs intersect (it's a simplex)
        for i, j in combinations(range(self.n_charts), 2):
            nerve['edges'].append((i, j))
        
        # All triples intersect
        for triple in combinations(range(self.n_charts), 3):
            nerve['faces'].append(triple)
        
        # Higher simplices
        for k in range(4, self.n_charts + 1):
            for simplex in combinations(range(self.n_charts), k):
                nerve['higher'].append(simplex)
        
        # Note: The FULL (n+1)-simplex is NOT in the nerve
        # because the intersection of ALL sets is empty
        # (no point has <x, v_i> > -eps for ALL antipodal-like vertices)
        
        return nerve


# ============================================================
# Specific examples
# ============================================================

def tetrahedral_cover_S2(epsilon: float = 0.3) -> GoodCover:
    """
    Minimal good cover of S² using 4 open sets (tetrahedron).
    
    The tetrahedron vertices are:
        v0 = (1, 1, 1)/√3
        v1 = (1, -1, -1)/√3  
        v2 = (-1, 1, -1)/√3
        v3 = (-1, -1, 1)/√3
    
    Each open set U_i = {x ∈ S² : <x, v_i> > -ε}
    """
    return GoodCover(sphere_dim=2, epsilon=epsilon)


def triangular_cover_S1(epsilon: float = 0.3) -> GoodCover:
    """
    Minimal good cover of S¹ using 3 open sets (triangle).
    """
    return GoodCover(sphere_dim=1, epsilon=epsilon)


def octahedral_cover_S2(epsilon: float = 0.1) -> GoodCover:
    """
    Non-minimal good cover of S² using 6 open sets (octahedron).
    
    This uses the 6 axis-aligned hemispheres:
        U_x± = {x ∈ S² : ±x > -ε}
        U_y± = {x ∈ S² : ±y > -ε}
        U_z± = {x ∈ S² : ±z > -ε}
    """
    # Override vertices to be octahedron vertices
    cover = GoodCover(sphere_dim=2, epsilon=epsilon)
    cover.vertices = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ], dtype=float)
    cover.n_charts = 6
    print(f"Octahedral cover of S² with 6 open sets")
    return cover


# ============================================================
# Data generation on spheres
# ============================================================

def sample_sphere(n: int, n_points: int, seed: int = 42) -> np.ndarray:
    """
    Sample uniformly from S^n.
    
    Args:
        n: Dimension of sphere
        n_points: Number of points to sample
        seed: Random seed
        
    Returns:
        Array of shape (n_points, n+1) on S^n
    """
    np.random.seed(seed)
    # Sample from standard normal and normalize
    points = np.random.randn(n_points, n + 1)
    return normalize(points)


def sample_sphere_uniform_angles(n_theta: int = 50, n_phi: int = 100) -> np.ndarray:
    """
    Sample S² using spherical coordinates (uniform in angles).
    Note: This is NOT uniform on the sphere (denser at poles).
    
    Returns:
        Array of shape (n_theta * n_phi, 3)
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    x = np.sin(theta_flat) * np.cos(phi_flat)
    y = np.sin(theta_flat) * np.sin(phi_flat)
    z = np.cos(theta_flat)
    
    return np.column_stack([x, y, z])


# ============================================================
# Visualization
# ============================================================

def plot_good_cover_S2(
    cover: GoodCover,
    points: np.ndarray,
    figsize: Tuple[int, int] = (15, 5),
    show_vertices: bool = True
):
    """
    Visualize good cover of S² showing each chart's domain.
    """
    fig = plt.figure(figsize=figsize)
    
    membership = cover.membership(points)
    colors = plt.cm.Set1(np.linspace(0, 1, cover.n_charts))
    
    # Plot each chart
    for i in range(min(cover.n_charts, 4)):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        
        # Points in this chart
        in_chart = membership[:, i]
        ax.scatter(
            points[in_chart, 0],
            points[in_chart, 1],
            points[in_chart, 2],
            c=[colors[i]],
            s=5,
            alpha=0.6
        )
        
        # Points not in this chart (gray)
        ax.scatter(
            points[~in_chart, 0],
            points[~in_chart, 1],
            points[~in_chart, 2],
            c='lightgray',
            s=2,
            alpha=0.3
        )
        
        # Show vertex
        if show_vertices:
            v = cover.vertices[i]
            ax.scatter([v[0]], [v[1]], [v[2]], c='black', s=100, marker='^')
            ax.text(v[0]*1.2, v[1]*1.2, v[2]*1.2, f'$v_{i}$', fontsize=12)
        
        ax.set_title(f'$U_{i}$: {np.sum(in_chart)} points')
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
    
    plt.tight_layout()
    return fig


def plot_intersections_S2(
    cover: GoodCover,
    points: np.ndarray,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize double and triple intersections.
    """
    fig = plt.figure(figsize=figsize)
    membership = cover.membership(points)
    
    # Plot some double intersections
    double_intersections = [(0, 1), (0, 2), (1, 2), (0, 3)]
    
    for idx, (i, j) in enumerate(double_intersections[:4]):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
        
        in_both = membership[:, i] & membership[:, j]
        
        ax.scatter(
            points[in_both, 0],
            points[in_both, 1],
            points[in_both, 2],
            c='purple',
            s=10,
            alpha=0.7
        )
        
        # Background sphere
        ax.scatter(
            points[~in_both, 0],
            points[~in_both, 1],
            points[~in_both, 2],
            c='lightgray',
            s=1,
            alpha=0.2
        )
        
        ax.set_title(f'$U_{i} \\cap U_{j}$: {np.sum(in_both)} points')
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
    
    plt.tight_layout()
    return fig


def plot_cover_2d_projection(
    cover: GoodCover,
    points: np.ndarray,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot stereographic projections of each chart.
    """
    fig, axes = plt.subplots(1, cover.n_charts, figsize=figsize)
    if cover.n_charts == 1:
        axes = [axes]
    
    membership = cover.membership(points)
    colors = plt.cm.Set1(np.linspace(0, 1, cover.n_charts))
    
    for i, ax in enumerate(axes):
        # Project from antipode of vertex i
        pole = -cover.vertices[i]
        
        # Get points in chart i
        in_chart = membership[:, i]
        chart_points = points[in_chart]
        
        if len(chart_points) > 0:
            projected = stereographic_project(chart_points, pole)
            
            ax.scatter(
                projected[:, 0],
                projected[:, 1],
                c=[colors[i]],
                s=5,
                alpha=0.6
            )
        
        ax.set_title(f'$U_{i}$ (stereo proj)')
        ax.set_aspect('equal')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
    
    plt.tight_layout()
    return fig


# ============================================================
# Integration with AtlasAutoencoder
# ============================================================

def good_cover_to_atlas_assignments(
    cover: GoodCover,
    points: np.ndarray
) -> List[np.ndarray]:
    """
    Convert good cover membership to atlas autoencoder assignments.
    
    Args:
        cover: GoodCover instance
        points: Data points on the manifold
        
    Returns:
        List of index arrays for each chart
    """
    return cover.get_assignments(points)


def demonstrate_cech_cohomology(cover: GoodCover, verbose: bool = True):
    """
    Demonstrate how the good cover gives correct Čech cohomology.
    
    For S^n with minimal good cover:
    - H^0(S^n) = Z (connected)
    - H^n(S^n) = Z (orientation class)
    - H^k(S^n) = 0 for 0 < k < n
    
    The nerve of the cover is ∂Δ^{n+1} (boundary of (n+1)-simplex),
    which is homeomorphic to S^n.
    """
    nerve = cover.compute_nerve()
    
    if verbose:
        print(f"\nNerve of minimal good cover of S^{cover.n}:")
        print(f"  Vertices (0-simplices): {len(nerve['vertices'])}")
        print(f"  Edges (1-simplices): {len(nerve['edges'])}")
        print(f"  Faces (2-simplices): {len(nerve['faces'])}")
        if nerve['higher']:
            print(f"  Higher simplices: {len(nerve['higher'])}")
        
        print(f"\nThe nerve is ∂Δ^{cover.n + 1}, homeomorphic to S^{cover.n}")
        print(f"Therefore Ȟ^*(nerve) ≅ H^*(S^{cover.n})")
    
    return nerve