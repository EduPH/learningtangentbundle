#
# Atlas Autoencoder (Fixed Cover)
# ---------------------------------------------
# A principled implementation of an atlas of local autoencoders.
# The tangent bundle is constructed from the linearized transition maps.
#
# Core principles:
# - Reconstruction loss alone enforces cocycle consistency.
# - Transition maps are implicit: T_ji = E_j ∘ D_i.
# - Jacobian regularity ensures well-defined linearized structure.
# - Each chart sees ONLY its assigned data subset (faithful to atlas semantics).
#
# This implementation is faithful to the theoretical framework of
# atlas learning via local autoencoders.
#
# METRICS:
# - ε (varepsilon): pointwise reconstruction error sup_x ||D_i(E_i(x)) - x||
# - η (eta): differential reconstruction error sup_x ||d(D_i∘E_i)_x - Id||_op
# - δ (delta): non-degeneracy gap min_{i,j,x} |det g_ji(x)|
# - Cocycle error: ||T_ji(E_i(x)) - E_j(x)|| measuring transition consistency

import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


ACTIVATION = "tanh"


# ============================================================
# Local chart autoencoder
# ============================================================

class LocalAutoencoder(tf.keras.Model):
    """
    Single local autoencoder defining one chart.
    
    Represents a chart (U_i, φ_i) where:
    - U_i ⊂ X is the chart domain (subset of data)
    - E_i: U_i → R^d is the encoder (local coordinates)
    - D_i: R^d → X is the decoder (inverse map)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        name: str,
    ):
        super().__init__(name=name)

        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(h, activation=ACTIVATION) for h in hidden_dims]
            + [tf.keras.layers.Dense(latent_dim)],
            name=f"{name}_encoder",
        )

        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(h, activation=ACTIVATION) for h in reversed(hidden_dims)]
            + [tf.keras.layers.Dense(input_dim)],
            name=f"{name}_decoder",
        )

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode to local coordinates: E_i(x)."""
        return self.encoder(x)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode from local coordinates: D_i(z)."""
        return self.decoder(z)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Full reconstruction: D_i(E_i(x))."""
        return self.decode(self.encode(x))


# ============================================================
# Atlas autoencoder system
# ============================================================

class AtlasAutoencoder:
    """
    System of local autoencoders forming a learned atlas.
    
    Given a dataset X and a covering {U_i}, this class trains a collection
    of local autoencoders {(E_i, D_i)} such that:
    
    1. Each autoencoder (E_i, D_i) is trained ONLY on points in U_i
    2. Transition maps T_ji = E_j ∘ D_i are implicitly defined
    3. Cocycle consistency T_ki = T_kj ∘ T_ji emerges from reconstruction loss
    4. Linearized transition maps yield vector bundle structure
    
    The tangent bundle is constructed from Jacobians of transition maps,
    enabling computation of characteristic classes.
    """

    def __init__(
        self,
        data: np.ndarray,
        n_charts: int,
        subset_assignments: List[np.ndarray],
        latent_dim: int = 2,
        hidden_dims: List[int] = [64, 32],
    ):
        """
        Initialize the atlas autoencoder system.
        
        Args:
            data: Full dataset, shape (n_points, n_features)
            n_charts: Number of charts in the atlas
            subset_assignments: List of index arrays, one per chart.
                subset_assignments[i] contains indices of points in U_i.
                Charts may overlap (same index in multiple arrays).
            latent_dim: Dimension of local coordinates (manifold dimension)
            hidden_dims: Hidden layer dimensions for encoder/decoder networks
        
        Raises:
            ValueError: If subset_assignments length doesn't match n_charts
        """
        self.data = data.astype(np.float32)
        self.n_charts = n_charts
        self.latent_dim = latent_dim
        self.input_dim = data.shape[1]
        
        # Validate subset assignments
        if len(subset_assignments) != n_charts:
            raise ValueError(
                f"Number of subset_assignments ({len(subset_assignments)}) "
                f"must match n_charts ({n_charts})"
            )
        self.subset_assignments = subset_assignments

        # Create local charts
        self.autoencoders = [
            LocalAutoencoder(
                input_dim=data.shape[1],
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                name=f"chart_{i}",
            )
            for i in range(self.n_charts)
        ]

        self.optimizers = [
            tf.keras.optimizers.Adam(1e-3) for _ in range(self.n_charts)
        ]
        
        # Precompute chart overlaps for cocycle computation
        self._compute_overlaps()

    # --------------------------------------------------------
    # Overlap computation
    # --------------------------------------------------------
    
    def _compute_overlaps(self):
        """
        Precompute pairwise and triple intersections of chart domains.
        
        For charts U_i, U_j, we compute:
        - pairwise_overlaps[(i,j)]: indices of points in U_i ∩ U_j
        - triple_overlaps[(i,j,k)]: indices of points in U_i ∩ U_j ∩ U_k
        
        These are used for:
        - Transition map computation (pairwise overlaps)
        - Cocycle consistency verification (triple overlaps)
        """
        self.pairwise_overlaps = {}
        self.triple_overlaps = {}
        
        # Compute pairwise intersections
        for i in range(self.n_charts):
            for j in range(i + 1, self.n_charts):
                overlap_ij = np.intersect1d(
                    self.subset_assignments[i],
                    self.subset_assignments[j]
                )
                if len(overlap_ij) > 0:
                    self.pairwise_overlaps[(i, j)] = overlap_ij
                    self.pairwise_overlaps[(j, i)] = overlap_ij  # symmetric
        
        # Compute triple intersections
        for i in range(self.n_charts):
            for j in range(i + 1, self.n_charts):
                for k in range(j + 1, self.n_charts):
                    overlap_ijk = np.intersect1d(
                        self.subset_assignments[i],
                        np.intersect1d(
                            self.subset_assignments[j],
                            self.subset_assignments[k]
                        )
                    )
                    if len(overlap_ijk) > 0:
                        # Store all permutations for easy lookup
                        for perm in [(i,j,k), (i,k,j), (j,i,k), 
                                     (j,k,i), (k,i,j), (k,j,i)]:
                            self.triple_overlaps[perm] = overlap_ijk
    
    def print_overlap_statistics(self):
        """Print statistics about chart overlaps (useful for debugging)."""
        print("\n" + "="*60)
        print("ATLAS OVERLAP STATISTICS")
        print("="*60)
        
        print(f"\nCharts: {self.n_charts}")
        print(f"Data points: {len(self.data)}")
        
        # Chart sizes
        print("\nChart sizes:")
        for i, indices in enumerate(self.subset_assignments):
            pct = 100 * len(indices) / len(self.data)
            print(f"  Chart {i}: {len(indices)} points ({pct:.1f}%)")
        
        # Pairwise overlaps
        unique_pairs = {}
        for key, val in self.pairwise_overlaps.items():
            sorted_key = tuple(sorted(key))
            if sorted_key not in unique_pairs:
                unique_pairs[sorted_key] = val
        
        print(f"\nPairwise overlaps: {len(unique_pairs)}")
        if unique_pairs:
            sizes = [len(v) for v in unique_pairs.values()]
            print(f"  Min: {min(sizes)} points")
            print(f"  Max: {max(sizes)} points")
            print(f"  Avg: {np.mean(sizes):.1f} points")
        
        # Triple overlaps
        unique_triples = {}
        for key, val in self.triple_overlaps.items():
            sorted_key = tuple(sorted(key))
            if sorted_key not in unique_triples:
                unique_triples[sorted_key] = val
        
        print(f"\nTriple overlaps: {len(unique_triples)}")
        if unique_triples:
            sizes = [len(v) for v in unique_triples.values()]
            print(f"  Min: {min(sizes)} points")
            print(f"  Max: {max(sizes)} points")
            print(f"  Avg: {np.mean(sizes):.1f} points")
            
            if min(sizes) < 20:
                print(f"\n  ⚠ Warning: Some triple overlaps < 20 points")
        else:
            print("  ✗ NO TRIPLE OVERLAPS FOUND")
            print("  → Cocycle loss will be 0")
        
        print("="*60 + "\n")

    # --------------------------------------------------------
    # Transition maps
    # --------------------------------------------------------

    def transition_map(self, z: tf.Tensor, i: int, j: int) -> tf.Tensor:
        """
        Compute transition map T_ji = E_j ∘ D_i.
        
        Maps points from chart i's latent space to chart j's latent space.
        
        Args:
            z: Points in chart i's latent space, shape (batch, latent_dim)
            i: Source chart index
            j: Target chart index
            
        Returns:
            Points in chart j's latent space, shape (batch, latent_dim)
        """
        x = self.autoencoders[i].decode(z)
        return self.autoencoders[j].encode(x)

    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    def reconstruction_loss(
        self,
        autoencoder: LocalAutoencoder,
        x: tf.Tensor,
    ) -> tf.Tensor:
        """
        Reconstruction loss for a single chart.
        
        L_rec = (1/|U_i|) Σ_{x ∈ U_i} ||x - D_i(E_i(x))||²
        
        This loss implicitly enforces cocycle consistency:
        minimizing reconstruction error forces T_ii = id and
        makes transition maps approximately invertible.
        """
        recon = autoencoder(x)
        return tf.reduce_mean(tf.reduce_sum((x - recon) ** 2, axis=1))

    def jacobian_regularity_loss(
        self,
        encoder: tf.keras.Model,
        x: tf.Tensor,
        epsilon: float = 1e-3,
    ) -> tf.Tensor:
        """
        Jacobian regularity: ensure encoder is a submersion (full row rank).
        
        For encoder E: R^N → R^d where d < N, the Jacobian dE/dx
        should have full row rank d. This ensures the encoder
        locally preserves the manifold dimension.
        
        We penalize when the smallest singular value falls below epsilon.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = encoder(x)

        J = tape.batch_jacobian(z, x)  # Shape: (batch, d, N)
        
        # J @ J^T is a d×d matrix; its eigenvalues are squared singular values
        JJT = tf.matmul(J, J, transpose_b=True)  # Shape: (batch, d, d)
        
        eigvals = tf.linalg.eigvalsh(JJT)
        sigma_min = tf.sqrt(eigvals[:, 0] + 1e-8)  # Smallest singular value
        
        # Penalize when smallest singular value is below threshold
        return tf.reduce_mean(tf.maximum(0.0, epsilon - sigma_min))

    def cocycle_loss(self) -> tf.Tensor:
        """
        Cocycle consistency loss on triple intersections.
        
        For charts (i, j, k) with U_i ∩ U_j ∩ U_k ≠ ∅, the cocycle condition is:
            T_ki = T_kj ∘ T_ji
        
        Equivalently, in terms of determinants:
            det(dT_ki) = det(dT_kj) · det(dT_ji)
        
        This loss enforces that transition maps compose consistently,
        which is necessary for the collection to define a valid atlas.
        
        Note: The sign of det(dT_ji) determines orientability. On each
        connected component of U_i ∩ U_j, the sign should be constant.
        """
        if not self.triple_overlaps:
            return tf.constant(0.0)
        
        cocycle_errors = []
        
        # Iterate over unique triple intersections
        for i in range(self.n_charts):
            for j in range(i + 1, self.n_charts):
                for k in range(j + 1, self.n_charts):
                    if (i, j, k) not in self.triple_overlaps:
                        continue
                    
                    overlap_idx = self.triple_overlaps[(i, j, k)]
                    
                    # Sample points (Jacobian computation is expensive)
                    n_samples = min(len(overlap_idx), 100)
                    sample_idx = np.random.choice(overlap_idx, size=n_samples, replace=False)
                    x = tf.constant(self.data[sample_idx], dtype=tf.float32)
                    
                    # Encode to source charts
                    z_i = self.autoencoders[i].encode(x)
                    z_j = self.autoencoders[j].encode(x)
                    
                    # Compute determinants for the three transition paths
                    det_ki = self._compute_transition_determinant(z_i, i, k)
                    det_kj = self._compute_transition_determinant(z_j, j, k)
                    det_ji = self._compute_transition_determinant(z_i, i, j)
                    
                    # Cocycle condition: det(T_ki) = det(T_kj) · det(T_ji)
                    cocycle_product = det_kj * det_ji
                    
                    # Relative error (scale-invariant)
                    eps = 1e-6
                    scale = tf.maximum(tf.abs(det_ki), tf.abs(cocycle_product)) + eps
                    relative_error = ((det_ki - cocycle_product) / scale) ** 2
                    
                    error = tf.reduce_mean(relative_error)
                    cocycle_errors.append(error)
        
        if not cocycle_errors:
            return tf.constant(0.0)
        
        return tf.reduce_mean(tf.stack(cocycle_errors))
    
    def _compute_transition_determinant(
        self, 
        z_source: tf.Tensor, 
        i: int, 
        j: int
    ) -> tf.Tensor:
        """
        Compute det(dT_ji) where T_ji = E_j ∘ D_i.
        
        Args:
            z_source: Points in chart i's latent space
            i: Source chart index
            j: Target chart index
            
        Returns:
            Determinants of Jacobian matrices, shape (batch,)
        """
        with tf.GradientTape() as tape:
            tape.watch(z_source)
            x_recon = self.autoencoders[i].decode(z_source)
            z_target = self.autoencoders[j].encode(x_recon)
        
        # Jacobian: dz_target/dz_source, shape (batch, d, d)
        jacobian = tape.batch_jacobian(z_target, z_source)
        
        return tf.linalg.det(jacobian)

    # --------------------------------------------------------
    # Theoretical Metrics (matching paper definitions)
    # --------------------------------------------------------
    
    def compute_varepsilon(self, chart_idx: int, x: tf.Tensor) -> tf.Tensor:
        """
        Compute ε (varepsilon): pointwise reconstruction error.
        
        Paper Definition (Definition 4, condition 3):
            ε := sup_{x ∈ U_i} || D_i(E_i(x)) - x ||
        
        We compute both sup (for theoretical bound) and mean (for monitoring).
        
        Args:
            chart_idx: Chart index i
            x: Points in chart domain, shape (batch, N)
            
        Returns:
            Sup of reconstruction error over the batch
        """
        ae = self.autoencoders[chart_idx]
        recon = ae(x)
        errors = tf.norm(x - recon, axis=1)  # Shape: (batch,)
        return tf.reduce_max(errors)  # Sup over batch
    
    def compute_varepsilon_mean(self, chart_idx: int, x: tf.Tensor) -> tf.Tensor:
        """
        Compute mean reconstruction error (for monitoring during training).
        
        Args:
            chart_idx: Chart index i
            x: Points in chart domain, shape (batch, N)
            
        Returns:
            Mean reconstruction error
        """
        ae = self.autoencoders[chart_idx]
        recon = ae(x)
        errors = tf.norm(x - recon, axis=1)
        return tf.reduce_mean(errors)
    
    def compute_eta(self, chart_idx: int, x: tf.Tensor) -> tf.Tensor:
        """
        Compute η (eta): differential reconstruction error restricted to tangent space.
        
        Paper Definition (Definition 4, condition 4):
            η := sup_{x ∈ U_i} || d(D_i ∘ E_i)_x |_{T_x M} - Id_{T_x M} ||_op
        
        Since Φ = D ∘ E factors through R^d, the Jacobian dΦ_x has rank ≤ d.
        Let σ_1 ≥ ... ≥ σ_d > 0 be the d nonzero singular values of dΦ_x.
        
        For a well-trained autoencoder, T_x M ≈ span{v_1, ..., v_d} where v_j are
        the right singular vectors. The restriction dΦ|_{T_x M} acts as:
            v_j ↦ σ_j u_j ≈ σ_j v_j
        
        Therefore:
            || dΦ|_{T_x M} - Id ||_op ≈ max_j |σ_j - 1| = max(σ_1 - 1, 1 - σ_d)
        
        Args:
            chart_idx: Chart index i
            x: Points in chart domain, shape (batch, N)
            
        Returns:
            Sup of differential error over the batch
        """
        ae = self.autoencoders[chart_idx]
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            recon = ae(x)  # Φ(x) = D(E(x))
        
        # Jacobian of reconstruction map: dΦ/dx, shape (batch, N, N)
        J_phi = tape.batch_jacobian(recon, x)
        
        # Compute singular values of J_phi
        # We only need the top d singular values (the nonzero ones)
        s = tf.linalg.svd(J_phi, compute_uv=False)  # Shape: (batch, N), descending order
        
        # Extract top d singular values
        s_top_d = s[:, :self.latent_dim]  # Shape: (batch, d)
        
        # η ≈ max(σ_1 - 1, 1 - σ_d) for each sample
        sigma_max = s_top_d[:, 0]   # σ_1 (largest)
        sigma_min = s_top_d[:, -1]  # σ_d (smallest of top d)
        
        eta_per_sample = tf.maximum(sigma_max - 1.0, 1.0 - sigma_min)
        
        return tf.reduce_max(eta_per_sample)  # Sup over batch


    def compute_eta_mean(self, chart_idx: int, x: tf.Tensor) -> tf.Tensor:
        """
        Compute mean differential error (for monitoring during training).
        
        Uses the same tangent-space approximation as compute_eta.
        """
        ae = self.autoencoders[chart_idx]
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            recon = ae(x)
        
        J_phi = tape.batch_jacobian(recon, x)
        s = tf.linalg.svd(J_phi, compute_uv=False)
        
        s_top_d = s[:, :self.latent_dim]
        sigma_max = s_top_d[:, 0]
        sigma_min = s_top_d[:, -1]
        
        eta_per_sample = tf.maximum(sigma_max - 1.0, 1.0 - sigma_min)
        
        return tf.reduce_mean(eta_per_sample)
    
    def compute_eta_tangent(self, chart_idx: int, x: tf.Tensor) -> tf.Tensor:
        """
        Compute η restricted to latent space (d×d version).
        
        This computes || d(E ∘ D ∘ E)_z - I_d ||_op which measures
        how well the round-trip E → D → E preserves the latent coordinates.
        
        This is often more numerically stable than the N×N version.
        
        Args:
            chart_idx: Chart index i
            x: Points in chart domain, shape (batch, N)
            
        Returns:
            Sup of tangent-space differential error
        """
        ae = self.autoencoders[chart_idx]
        z = ae.encode(x)
        
        with tf.GradientTape() as tape:
            tape.watch(z)
            x_recon = ae.decode(z)
            z_round = ae.encode(x_recon)  # E(D(E(x)))
        
        # Jacobian of E∘D: dz_round/dz, shape (batch, d, d)
        J_ED = tape.batch_jacobian(z_round, z)
        
        I_d = tf.eye(self.latent_dim, batch_shape=[tf.shape(z)[0]])
        delta_J = J_ED - I_d
        
        s = tf.linalg.svd(delta_J, compute_uv=False)
        op_norms = s[:, 0]
        
        return tf.reduce_max(op_norms)
    
    def compute_delta(self) -> tf.Tensor:
        """
        Compute δ (delta): non-degeneracy gap.
        
        Paper Definition (Definition 5):
            δ(A) := min_{(i,j): U_i ∩ U_j ≠ ∅} inf_{x ∈ U_i ∩ U_j} |det g_ji(x)|
        
        where g_ji(x) = d(T_ji)_{E_i(x)} is the Jacobian of the transition map.
        
        This is the crucial quantity for stability: δ > 0 ensures the sign
        cocycle is well-defined and stable under perturbations.
        
        Returns:
            The non-degeneracy gap (minimum absolute determinant)
        """
        delta_min = tf.constant(float('inf'))
        
        for (i, j), overlap_idx in self.pairwise_overlaps.items():
            if i >= j:
                continue
            
            x_overlap = tf.constant(self.data[overlap_idx], dtype=tf.float32)
            z_i = self.autoencoders[i].encode(x_overlap)
            
            with tf.GradientTape() as tape:
                tape.watch(z_i)
                x_recon = self.autoencoders[i].decode(z_i)
                z_j = self.autoencoders[j].encode(x_recon)
            
            # Jacobian g_ji = d(T_ji) = d(E_j ∘ D_i), shape (batch, d, d)
            J_ji = tape.batch_jacobian(z_j, z_i)
            
            # Absolute determinants
            abs_dets = tf.abs(tf.linalg.det(J_ji))
            
            # Minimum over this overlap
            min_det_ij = tf.reduce_min(abs_dets)
            delta_min = tf.minimum(delta_min, min_det_ij)
        
        return delta_min
    
    def compute_delta_mean(self) -> tf.Tensor:
        """
        Compute mean absolute determinant (for monitoring).
        """
        det_list = []
        
        for (i, j), overlap_idx in self.pairwise_overlaps.items():
            if i >= j:
                continue
            
            # Sample for efficiency
            n_samples = min(len(overlap_idx), 200)
            sample_idx = np.random.choice(overlap_idx, size=n_samples, replace=False)
            
            x_overlap = tf.constant(self.data[sample_idx], dtype=tf.float32)
            z_i = self.autoencoders[i].encode(x_overlap)
            
            with tf.GradientTape() as tape:
                tape.watch(z_i)
                x_recon = self.autoencoders[i].decode(z_i)
                z_j = self.autoencoders[j].encode(x_recon)
            
            J_ji = tape.batch_jacobian(z_j, z_i)
            abs_dets = tf.abs(tf.linalg.det(J_ji))
            det_list.append(tf.reduce_mean(abs_dets))
        
        if det_list:
            return tf.reduce_mean(tf.stack(det_list))
        return tf.constant(0.0)
    
    def compute_cocycle_error(self) -> tf.Tensor:
        """
        Compute cocycle/transition consistency error.
        
        This measures: E[|| T_ji(E_i(x)) - E_j(x) ||]
        
        For an exact atlas, this is zero since T_ji(E_i(x)) = E_j(D_i(E_i(x))) = E_j(x).
        For approximate atlases, this quantifies how well transitions agree
        with direct encodings.
        
        Returns:
            Mean cocycle error over all overlaps
        """
        error_list = []
        
        for (i, j), overlap_idx in self.pairwise_overlaps.items():
            if i >= j:
                continue
            
            x_overlap = tf.constant(self.data[overlap_idx], dtype=tf.float32)
            
            # Direct encoding in chart j
            z_j_direct = self.autoencoders[j].encode(x_overlap)
            
            # Transition from chart i to chart j
            z_i = self.autoencoders[i].encode(x_overlap)
            z_j_via_i = self.transition_map(z_i, i, j)
            
            # Error: || T_ji(E_i(x)) - E_j(x) ||
            error_ij = tf.reduce_mean(tf.norm(z_j_via_i - z_j_direct, axis=1))
            error_list.append(error_ij)
        
        if error_list:
            return tf.reduce_mean(tf.stack(error_list))
        return tf.constant(0.0)
    
    def compute_encoder_sigma_min(self, chart_idx: int, x: tf.Tensor) -> tf.Tensor:
        """
        Compute minimum singular value of encoder Jacobian.
        
        This is s_E in Proposition 10 of the paper:
            σ_min(dE_i|_{T_x M}) ≥ s_E > 0
        
        Used to verify the encoder is a submersion (full row rank).
        
        Args:
            chart_idx: Chart index
            x: Points in chart domain
            
        Returns:
            Minimum singular value over the batch
        """
        ae = self.autoencoders[chart_idx]
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = ae.encode(x)
        
        J = tape.batch_jacobian(z, x)  # Shape: (batch, d, N)
        
        # Singular values of J (d values per sample since d ≤ N)
        JJT = tf.matmul(J, J, transpose_b=True)  # (batch, d, d)
        eigvals = tf.linalg.eigvalsh(JJT)  # Eigenvalues in ascending order
        sigma_min = tf.sqrt(eigvals[:, 0] + 1e-8)  # Smallest singular value
        
        return tf.reduce_min(sigma_min)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    @tf.function
    def train_step(
        self,
        batches: List[Optional[tf.Tensor]],
        lambda_jac: float,
        lambda_cocycle: float,
    ) -> Dict[str, tf.Tensor]:
        """
        Single training step with corrected theoretical metrics.
        
        Metrics computed (matching paper definitions):
        - varepsilon: pointwise reconstruction error sup ||x - D(E(x))||
        - varepsilon_mean: mean reconstruction error (for monitoring)
        - eta: differential error sup ||dΦ - I||_op (computed in tangent space)
        - delta: non-degeneracy gap min |det g_ji|
        - cocycle_error: transition consistency ||T_ji(E_i(x)) - E_j(x)||
        """
        
        losses = {}
        
        # Lists to accumulate metrics
        varepsilon_list = []
        varepsilon_mean_list = []
        eta_list = []
        sigma_min_list = []
        
        # -----------------------------
        # Local chart training
        # -----------------------------
        for i, (ae, opt) in enumerate(zip(self.autoencoders, self.optimizers)):
            x = batches[i]
            if x is None:
                continue

            with tf.GradientTape() as tape:
                # Standard losses
                L_rec = self.reconstruction_loss(ae, x)
                L_jac = self.jacobian_regularity_loss(ae.encoder, x)

                loss = L_rec + lambda_jac * L_jac

            grads = tape.gradient(loss, ae.trainable_variables)
            opt.apply_gradients(zip(grads, ae.trainable_variables))

            losses[f"recon_{i}"] = L_rec
            losses[f"jac_{i}"] = L_jac

            # -----------------------------
            # ε (varepsilon) — pointwise reconstruction error
            # Paper: sup_x ||D_i(E_i(x)) - x||
            # -----------------------------
            recon = ae(x)
            errors = tf.norm(x - recon, axis=1)
            varepsilon_list.append(tf.reduce_max(errors))
            varepsilon_mean_list.append(tf.reduce_mean(errors))

            # -----------------------------
            # η (eta) — differential reconstruction error
            # Paper: sup_x ||d(D∘E)_x|_{T_xM} - Id||_op
            # Approximated via top d singular values of d(D∘E)
            # -----------------------------
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                recon = ae(x)
            
            J_phi = tape2.batch_jacobian(recon, x)  # (batch, N, N)
            s = tf.linalg.svd(J_phi, compute_uv=False)  # (batch, N)
            
            # Top d singular values
            s_top_d = s[:, :self.latent_dim]  # (batch, d)
            sigma_max = s_top_d[:, 0]
            sigma_min = s_top_d[:, -1]
            
            # η ≈ max(σ_1 - 1, 1 - σ_d)
            eta_per_sample = tf.maximum(sigma_max - 1.0, 1.0 - sigma_min)
            eta_batch = tf.reduce_max(eta_per_sample)
            eta_list.append(eta_batch)
            
            # -----------------------------
            # σ_min(dE) — encoder regularity
            # For verifying submersion property
            # -----------------------------
            with tf.GradientTape() as tape3:
                tape3.watch(x)
                z_enc = ae.encoder(x)
            J_E = tape3.batch_jacobian(z_enc, x)  # (batch, d, N)
            JJT = tf.matmul(J_E, J_E, transpose_b=True)
            eigvals = tf.linalg.eigvalsh(JJT)
            sigma_min = tf.sqrt(eigvals[:, 0] + 1e-8)
            sigma_min_list.append(tf.reduce_min(sigma_min))

        # -----------------------------
        # Cocycle consistency loss
        # -----------------------------
        if lambda_cocycle > 0 and self.triple_overlaps:
            with tf.GradientTape(persistent=True) as tape:
                L_cocycle = self.cocycle_loss()

            losses["cocycle"] = L_cocycle

            for ae, opt in zip(self.autoencoders, self.optimizers):
                grads = tape.gradient(lambda_cocycle * L_cocycle, ae.trainable_variables)
                if grads is not None and any(g is not None for g in grads):
                    opt.apply_gradients(zip(grads, ae.trainable_variables))

            del tape
        else:
            L_cocycle = tf.constant(0.0)

        # -----------------------------
        # δ (delta) — non-degeneracy gap
        # Paper: min_{i,j,x} |det g_ji(x)|
        # Computed on overlaps (sampled for efficiency)
        # -----------------------------
        delta_min = tf.constant(float('inf'))
        det_mean_list = []
        
        for (i, j), overlap_idx in self.pairwise_overlaps.items():
            if i >= j:
                continue
            
            # Sample for efficiency during training
            n_samples = min(len(overlap_idx), 50)
            sample_idx = np.random.choice(overlap_idx, size=n_samples, replace=False)
            
            x_overlap = tf.constant(self.data[sample_idx], dtype=tf.float32)
            z_i = self.autoencoders[i].encode(x_overlap)
            
            with tf.GradientTape() as tape:
                tape.watch(z_i)
                x_recon = self.autoencoders[i].decode(z_i)
                z_j = self.autoencoders[j].encode(x_recon)
            
            J_ji = tape.batch_jacobian(z_j, z_i)
            abs_dets = tf.abs(tf.linalg.det(J_ji))
            
            delta_min = tf.minimum(delta_min, tf.reduce_min(abs_dets))
            det_mean_list.append(tf.reduce_mean(abs_dets))

        # -----------------------------
        # Cocycle error — transition consistency
        # Measures ||T_ji(E_i(x)) - E_j(x)||
        # -----------------------------
        cocycle_error_list = []
        
        for (i, j), overlap_idx in self.pairwise_overlaps.items():
            if i >= j:
                continue

            # Sample for efficiency
            n_samples = min(len(overlap_idx), 50)
            sample_idx = np.random.choice(overlap_idx, size=n_samples, replace=False)
            
            x_overlap = tf.constant(self.data[sample_idx], dtype=tf.float32)

            z_i = self.autoencoders[i].encode(x_overlap)
            z_j_from_i = self.transition_map(z_i, i, j)
            z_j_true = self.autoencoders[j].encode(x_overlap)

            error_ij = tf.reduce_mean(tf.norm(z_j_from_i - z_j_true, axis=1))
            cocycle_error_list.append(error_ij)

        # -----------------------------
        # Aggregate theoretical metrics
        # -----------------------------
        if varepsilon_list:
            varepsilon = tf.reduce_max(tf.stack(varepsilon_list))
            varepsilon_mean = tf.reduce_mean(tf.stack(varepsilon_mean_list))
            eta = tf.reduce_max(tf.stack(eta_list))
            sigma_min_enc = tf.reduce_min(tf.stack(sigma_min_list))
        else:
            varepsilon = tf.constant(0.0)
            varepsilon_mean = tf.constant(0.0)
            eta = tf.constant(0.0)
            sigma_min_enc = tf.constant(0.0)
        
        if det_mean_list:
            delta_mean = tf.reduce_mean(tf.stack(det_mean_list))
        else:
            delta_mean = tf.constant(0.0)
            delta_min = tf.constant(0.0)
        
        if cocycle_error_list:
            cocycle_error = tf.reduce_mean(tf.stack(cocycle_error_list))
        else:
            cocycle_error = tf.constant(0.0)

        # Store metrics with paper-consistent names
        losses["varepsilon"] = varepsilon           # sup reconstruction error
        losses["varepsilon_mean"] = varepsilon_mean # mean reconstruction error
        losses["eta"] = eta                          # differential error
        losses["delta"] = delta_min                  # non-degeneracy gap (min)
        losses["delta_mean"] = delta_mean            # mean |det g_ji|
        losses["cocycle_error"] = cocycle_error      # transition consistency
        losses["sigma_min_enc"] = sigma_min_enc      # encoder regularity

        return losses

    
    def fit(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        lambda_jac: float = 0.01,
        lambda_cocycle: float = 0.1,
        verbose: bool = True,
    ):
        """
        Train the atlas autoencoder.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            lambda_jac: Weight for Jacobian regularity loss
            lambda_cocycle: Weight for cocycle consistency loss
            verbose: Whether to print training progress
        """
        # Prepare per-chart datasets
        subset_datasets = []
        for i in range(self.n_charts):
            subset_data = self.data[self.subset_assignments[i]]
            if len(subset_data) > 0:
                ds = tf.data.Dataset.from_tensor_slices(subset_data)
                ds = ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                subset_datasets.append(ds)
            else:
                subset_datasets.append(None)

        for epoch in range(epochs):
            epoch_losses = []

            # Create fresh iterators
            iterators = [iter(ds) if ds is not None else None for ds in subset_datasets]

            # Count batches in largest dataset
            max_steps = max(
                sum(1 for _ in ds) if ds is not None else 0 
                for ds in subset_datasets
            )
            # Reset iterators after counting
            iterators = [iter(ds) if ds is not None else None for ds in subset_datasets]

            for _ in range(max_steps):
                batch_data = []
                for it in iterators:
                    if it is None:
                        batch_data.append(None)
                    else:
                        try:
                            batch_data.append(next(it))
                        except StopIteration:
                            batch_data.append(None)

                losses = self.train_step(
                    batch_data,
                    lambda_jac=lambda_jac,
                    lambda_cocycle=lambda_cocycle,
                )
                epoch_losses.append(losses)

            if verbose and epoch % 25 == 0:
                self._print_epoch_losses(epoch, epoch_losses)
    
    def _print_epoch_losses(self, epoch: int, epoch_losses: List[Dict[str, tf.Tensor]]):
        """Pretty-print averaged losses including theoretical metrics."""
    
        # Average losses over the epoch
        avg_losses = {}
    
        for losses in epoch_losses:
            for key, value in losses.items():
                if key not in avg_losses:
                    avg_losses[key] = []
                avg_losses[key].append(value.numpy())
    
        for key in avg_losses:
            avg_losses[key] = sum(avg_losses[key]) / len(avg_losses[key])
    
        print(f"Epoch {epoch:04d}")
    
        # --- Standard per-chart losses ---
        for key in sorted(avg_losses.keys()):
            if key.startswith("recon_") or key.startswith("jac_"):
                print(f"  {key:12s}: {avg_losses[key]:.6f}")
    
        # --- Cocycle loss ---
        if "cocycle" in avg_losses:
            print(f"  {'cocycle':12s}: {avg_losses['cocycle']:.6f}")
    
        # --- Theoretical metrics (paper-consistent) ---
        print("\n  --- Theoretical Metrics (Paper Definitions) ---")
        if "varepsilon" in avg_losses:
            print(f"  {'ε (sup)':16s}: {avg_losses['varepsilon']:.6f}  [pointwise recon error]")
        if "varepsilon_mean" in avg_losses:
            print(f"  {'ε (mean)':16s}: {avg_losses['varepsilon_mean']:.6f}")
        if "eta" in avg_losses:
            print(f"  {'η':16s}: {avg_losses['eta']:.6f}  [differential error ||dΦ-I||]")
        if "delta" in avg_losses:
            print(f"  {'δ (min)':16s}: {avg_losses['delta']:.6f}  [non-degeneracy gap]")
        if "delta_mean" in avg_losses:
            print(f"  {'δ (mean)':16s}: {avg_losses['delta_mean']:.6f}")
        if "cocycle_error" in avg_losses:
            print(f"  {'cocycle err':16s}: {avg_losses['cocycle_error']:.6f}  [||T_ji(E_i)-E_j||]")
        if "sigma_min_enc" in avg_losses:
            print(f"  {'σ_min(dE)':16s}: {avg_losses['sigma_min_enc']:.6f}  [encoder regularity]")
    
        print("-" * 60)


    # --------------------------------------------------------
    # Full metric computation (post-training)
    # --------------------------------------------------------
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all theoretical metrics on the full dataset.
        
        This is more accurate than the training metrics which use sampling.
        Call after training for final evaluation.
        
        Returns:
            Dictionary with all metrics matching paper definitions
        """
        metrics = {}
        
        # Per-chart metrics
        varepsilon_list = []
        eta_list = []
        sigma_min_list = []
        
        for i in range(self.n_charts):
            x = tf.constant(self.data[self.subset_assignments[i]], dtype=tf.float32)
            
            # ε: reconstruction error
            eps_i = self.compute_varepsilon(i, x).numpy()
            varepsilon_list.append(eps_i)
            
            # η: differential error
            eta_i = self.compute_eta_tangent(i, x).numpy()
            eta_list.append(eta_i)
            
            # σ_min: encoder regularity
            sigma_i = self.compute_encoder_sigma_min(i, x).numpy()
            sigma_min_list.append(sigma_i)
        
        metrics['varepsilon'] = max(varepsilon_list)
        metrics['varepsilon_per_chart'] = varepsilon_list
        metrics['eta'] = max(eta_list)
        metrics['eta_per_chart'] = eta_list
        metrics['sigma_min_enc'] = min(sigma_min_list)
        metrics['sigma_min_per_chart'] = sigma_min_list
        
        # δ: non-degeneracy gap (full computation)
        metrics['delta'] = self.compute_delta().numpy()
        metrics['delta_mean'] = self.compute_delta_mean().numpy()
        
        # Cocycle error
        metrics['cocycle_error'] = self.compute_cocycle_error().numpy()
        
        return metrics
    
    def print_metrics_summary(self):
        """Print a comprehensive summary of all metrics."""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*60)
        print("ATLAS METRICS SUMMARY (Paper Definitions)")
        print("="*60)
        
        print("\n[Pointwise Reconstruction Error]")
        print(f"  ε = sup_x ||D(E(x)) - x|| = {metrics['varepsilon']:.6f}")
        for i, eps_i in enumerate(metrics['varepsilon_per_chart']):
            print(f"    Chart {i}: {eps_i:.6f}")
        
        print("\n[Differential Reconstruction Error]")
        print(f"  η = sup_x ||d(D∘E) - I||_op = {metrics['eta']:.6f}")
        for i, eta_i in enumerate(metrics['eta_per_chart']):
            print(f"    Chart {i}: {eta_i:.6f}")
        
        print("\n[Non-degeneracy Gap]")
        print(f"  δ = min |det g_ji| = {metrics['delta']:.6f}")
        print(f"  δ_mean = {metrics['delta_mean']:.6f}")
        
        print("\n[Encoder Regularity]")
        print(f"  σ_min(dE) = {metrics['sigma_min_enc']:.6f}")
        for i, s_i in enumerate(metrics['sigma_min_per_chart']):
            print(f"    Chart {i}: {s_i:.6f}")
        
        print("\n[Cocycle/Transition Consistency]")
        print(f"  ||T_ji(E_i) - E_j|| = {metrics['cocycle_error']:.6f}")
        
        # Stability check
        print("\n[Stability Analysis]")
        if metrics['delta'] > 0:
            print(f"  ✓ δ > 0: Sign cocycle is well-defined")
        else:
            print(f"  ✗ δ = 0: Degenerate transition maps detected!")
        
        if metrics['eta'] < 1.0:
            print(f"  ✓ η < 1: Reconstruction differential is contractive")
        else:
            print(f"  ⚠ η ≥ 1: Large differential error")
        
        print("="*60 + "\n")
        
        return metrics


    # --------------------------------------------------------
    # Inference utilities
    # --------------------------------------------------------

    def encode(self, x: np.ndarray, chart: int) -> np.ndarray:
        """Encode data points using specified chart."""
        return self.autoencoders[chart].encode(x.astype(np.float32)).numpy()

    def decode(self, z: np.ndarray, chart: int) -> np.ndarray:
        """Decode latent points using specified chart."""
        return self.autoencoders[chart].decode(z.astype(np.float32)).numpy()
    
    def get_all_encodings(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get encodings for all charts.
        
        Returns:
            List of (indices, encodings) tuples, one per chart.
            indices[i] contains the data indices for chart i.
            encodings[i] contains the latent coordinates for those points.
        """
        results = []
        for i in range(self.n_charts):
            indices = self.subset_assignments[i]
            x = self.data[indices]
            z = self.encode(x, i)
            results.append((indices, z))
        return results

    # --------------------------------------------------------
    # Orientability detection
    # --------------------------------------------------------
    
    def compute_transition_jacobians(
        self, 
        i: int, 
        j: int,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobians of T_ji on the overlap U_i ∩ U_j.
        
        Args:
            i: Source chart index
            j: Target chart index  
            n_samples: Number of samples (None = all points in overlap)
            
        Returns:
            (points, jacobians): Data points and their Jacobian matrices
        """
        if (i, j) not in self.pairwise_overlaps:
            return np.array([]), np.array([])
        
        overlap_idx = self.pairwise_overlaps[(i, j)]
        
        if n_samples is not None and n_samples < len(overlap_idx):
            overlap_idx = np.random.choice(overlap_idx, size=n_samples, replace=False)
        
        x = self.data[overlap_idx]
        z_i = self.encode(x, i)
        
        z_tf = tf.constant(z_i, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(z_tf)
            x_recon = self.autoencoders[i].decode(z_tf)
            z_j = self.autoencoders[j].encode(x_recon)
        
        jacobians = tape.batch_jacobian(z_j, z_tf).numpy()
        
        return x, jacobians
    
    def compute_determinant_signs(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute signs of det(dT_ji) on the overlap.
        
        Returns:
            (points, signs): Data points and sign of determinant at each point
        """
        x, jacobians = self.compute_transition_jacobians(i, j)
        if len(jacobians) == 0:
            return np.array([]), np.array([])
        
        determinants = np.linalg.det(jacobians)
        signs = np.sign(determinants)
        
        return x, signs


# ============================================================
# Utility functions
# ============================================================

def create_cover_from_neighborhoods(
    data: np.ndarray,
    n_charts: int,
    overlap_ratio: float = 0.3,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Create a covering by selecting random centers and including nearby points.
    
    Args:
        data: Dataset, shape (n_points, n_features)
        n_charts: Number of charts
        overlap_ratio: Controls overlap between charts (higher = more overlap)
        seed: Random seed
        
    Returns:
        List of index arrays, one per chart
    """
    np.random.seed(seed)
    n_points = len(data)
    
    # Select random centers
    center_indices = np.random.choice(n_points, size=n_charts, replace=False)
    centers = data[center_indices]
    
    # Compute distances from each point to each center
    # Shape: (n_points, n_charts)
    distances = np.sqrt(
        np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )
    
    # Assign each point to nearest center
    nearest = np.argmin(distances, axis=1)
    
    # Compute radius for each chart to include overlap
    radii = []
    for i in range(n_charts):
        chart_points = np.where(nearest == i)[0]
        if len(chart_points) > 0:
            max_dist = np.max(distances[chart_points, i])
            radii.append(max_dist * (1 + overlap_ratio))
        else:
            radii.append(0)
    
    # Create assignments with overlap
    assignments = []
    for i in range(n_charts):
        in_chart = distances[:, i] <= radii[i]
        assignments.append(np.where(in_chart)[0])
    
    return assignments


def subsample_dataset(
    X: np.ndarray, 
    n_samples: int = 1000, 
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly subsample dataset.
    
    Args:
        X: Full dataset (n_points, n_features)
        n_samples: Number of points to keep
        random_seed: Random seed for reproducibility
        
    Returns:
        (X_sub, indices): Subsampled data and original indices
    """
    np.random.seed(random_seed)
    n_total = len(X)
    
    if n_samples >= n_total:
        return X, np.arange(n_total)
    
    indices = np.sort(np.random.choice(n_total, size=n_samples, replace=False))
    
    return X[indices], indices


# ============================================================
# Visualization
# ============================================================

def plot_atlas_embeddings(
    atlas: AtlasAutoencoder,
    figsize: Tuple[int, int] = (12, 4),
    point_size: int = 10,
    alpha: float = 0.6
):
    """
    Plot latent embeddings for each chart side by side.
    """
    n_charts = atlas.n_charts
    fig, axes = plt.subplots(1, n_charts, figsize=figsize)
    
    if n_charts == 1:
        axes = [axes]
    
    encodings = atlas.get_all_encodings()
    
    for i, (indices, z) in enumerate(encodings):
        ax = axes[i]
        ax.scatter(z[:, 0], z[:, 1], s=point_size, alpha=alpha)
        ax.set_title(f"Chart {i} ({len(indices)} pts)")
        ax.set_aspect('equal')
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
    
    plt.tight_layout()
    return fig, axes


def plot_atlas_with_transitions(
    atlas: AtlasAutoencoder,
    n_arrows: int = 30,
    arrow_alpha: float = 0.6,
    point_size: int = 15,
    figsize_per_chart: float = 4.0,
    seed: int = 42,
    colormap: str = 'tab10',
):
    """
    Plot all charts in a row, with arrows showing transition maps T_ji.
    
    For each chart pair (i, j) with overlap, arrows are drawn FROM chart i's
    subplot TO chart j's subplot, showing how the transition map T_ji = E_j ∘ D_i
    maps points from chart i's coordinates to chart j's coordinates.
    
    The key insight: for a point x in U_i ∩ U_j, we have two representations:
    - z_i = E_i(x) in chart i's latent space
    - z_j = E_j(x) in chart j's latent space
    
    The transition map satisfies: z_j = T_ji(z_i) = E_j(D_i(z_i)) ≈ E_j(x)
    
    Args:
        atlas: Trained AtlasAutoencoder
        n_arrows: Number of arrows to draw per chart pair
        arrow_alpha: Transparency of arrows
        point_size: Size of scatter points  
        figsize_per_chart: Size of each chart subplot
        seed: Random seed for arrow sampling
        colormap: Matplotlib colormap for chart colors
    """
    np.random.seed(seed)
    
    n_charts = atlas.n_charts
    figsize = (figsize_per_chart * n_charts, figsize_per_chart)
    
    fig, axes = plt.subplots(1, n_charts, figsize=figsize)
    if n_charts == 1:
        axes = [axes]
    
    # Get colormap
    cmap = plt.cm.get_cmap(colormap)
    colors = [cmap(i / max(n_charts - 1, 1)) for i in range(n_charts)]
    
    # Get all encodings
    encodings = atlas.get_all_encodings()
    
    # Plot each chart
    for i, (indices, z) in enumerate(encodings):
        ax = axes[i]
        ax.scatter(z[:, 0], z[:, 1], s=point_size, alpha=0.5, c=[colors[i]])
        ax.set_title(f"Chart {i} ($U_{i}$)", fontsize=11)
        ax.set_xlabel("$z_1$", fontsize=10)
        ax.set_ylabel("$z_2$", fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
    
    # We need to draw arrows across subplots using figure coordinates
    # For each overlapping pair (i, j), draw arrows from chart i to chart j
    
    for (i, j) in atlas.pairwise_overlaps.keys():
        if i >= j:
            continue
        
        overlap_idx = atlas.pairwise_overlaps[(i, j)]
        x_overlap = atlas.data[overlap_idx]
        
        # Get coordinates in both charts
        z_i = atlas.encode(x_overlap, i)  # Coordinates in chart i
        z_j = atlas.encode(x_overlap, j)  # Coordinates in chart j
        
        # Compute determinant signs for coloring
        _, signs = atlas.compute_determinant_signs(i, j)
        
        # Sample points for arrows
        n_sample = min(n_arrows, len(overlap_idx))
        sample_idx = np.random.choice(len(overlap_idx), size=n_sample, replace=False)
        
        for k in sample_idx:
            # Determine arrow color based on determinant sign
            if len(signs) > k:
                arrow_color = 'green' if signs[k] > 0 else 'red'
            else:
                arrow_color = 'gray'
            
            # Get data coordinates
            start_data = (z_i[k, 0], z_i[k, 1])
            end_data = (z_j[k, 0], z_j[k, 1])
            
            # Transform to figure coordinates
            start_disp = axes[i].transData.transform(start_data)
            end_disp = axes[j].transData.transform(end_data)
            
            start_fig = fig.transFigure.inverted().transform(start_disp)
            end_fig = fig.transFigure.inverted().transform(end_disp)
            
            # Draw arrow in figure coordinates
            arrow = plt.matplotlib.patches.FancyArrowPatch(
                start_fig, end_fig,
                transform=fig.transFigure,
                arrowstyle='->,head_width=3,head_length=4',
                color=arrow_color,
                alpha=arrow_alpha,
                connectionstyle='arc3,rad=0.2',
                linewidth=0.8,
            )
            fig.patches.append(arrow)
            
            # Mark the points in each chart
            axes[i].plot(z_i[k, 0], z_i[k, 1], 'o', markersize=3, 
                        color=arrow_color, alpha=0.8)
            axes[j].plot(z_j[k, 0], z_j[k, 1], 'o', markersize=3,
                        color=arrow_color, alpha=0.8)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', marker='>', linestyle='-',
               markersize=6, label='$\\det(dT_{ji}) > 0$'),
        Line2D([0], [0], color='red', marker='>', linestyle='-',
               markersize=6, label='$\\det(dT_{ji}) < 0$'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=10)
    
    plt.suptitle("Atlas with Transition Maps $T_{ji} = E_j \\circ D_i$", 
                 y=1.12, fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_single_transition(
    atlas: AtlasAutoencoder,
    i: int,
    j: int,
    n_arrows: int = 50,
    figsize: Tuple[int, int] = (12, 5),
    seed: int = 42,
):
    """
    Visualize a single transition map T_ji in detail.
    
    Shows three panels:
    1. Chart i with overlap points highlighted
    2. Chart j with overlap points highlighted  
    3. The transition map T_ji as a vector field in chart i's coordinates
    
    The third panel shows z_i → T_ji(z_i), visualizing how points move
    under the transition map, all within chart i's coordinate system.
    
    Args:
        atlas: Trained AtlasAutoencoder
        i: Source chart index
        j: Target chart index
        n_arrows: Number of arrows in the vector field
        figsize: Figure size
        seed: Random seed
    """
    np.random.seed(seed)
    
    if (i, j) not in atlas.pairwise_overlaps:
        print(f"No overlap between charts {i} and {j}")
        return None, None
    
    overlap_idx = atlas.pairwise_overlaps[(i, j)]
    x_overlap = atlas.data[overlap_idx]
    
    # Get full chart data
    x_i = atlas.data[atlas.subset_assignments[i]]
    x_j = atlas.data[atlas.subset_assignments[j]]
    
    z_i_full = atlas.encode(x_i, i)
    z_j_full = atlas.encode(x_j, j)
    
    # Get overlap encodings
    z_i_overlap = atlas.encode(x_overlap, i)
    z_j_overlap = atlas.encode(x_overlap, j)
    
    # Compute T_ji(z_i) for overlap points
    z_i_tf = tf.constant(z_i_overlap, dtype=tf.float32)
    x_recon = atlas.autoencoders[i].decode(z_i_tf)
    z_j_from_i = atlas.autoencoders[j].encode(x_recon).numpy()
    
    # Compute determinant signs
    _, signs = atlas.compute_determinant_signs(i, j)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Chart i
    ax = axes[0]
    ax.scatter(z_i_full[:, 0], z_i_full[:, 1], s=10, alpha=0.3, c='gray', label='All of $U_i$')
    ax.scatter(z_i_overlap[:, 0], z_i_overlap[:, 1], s=15, alpha=0.7, c='blue', label='$U_i \\cap U_j$')
    ax.set_title(f"Chart {i}: $E_{i}(U_{i})$", fontsize=11)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Chart j
    ax = axes[1]
    ax.scatter(z_j_full[:, 0], z_j_full[:, 1], s=10, alpha=0.3, c='gray', label='All of $U_j$')
    ax.scatter(z_j_overlap[:, 0], z_j_overlap[:, 1], s=15, alpha=0.7, c='red', label='$U_i \\cap U_j$')
    ax.set_title(f"Chart {j}: $E_{j}(U_{j})$", fontsize=11)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Transition map as vector field
    ax = axes[2]
    
    # Plot overlap points in chart i coordinates
    ax.scatter(z_i_overlap[:, 0], z_i_overlap[:, 1], s=10, alpha=0.3, c='blue', label='$z_i = E_i(x)$')
    
    # Plot where they map to under T_ji (still in chart i coords for reference)
    # Actually, T_ji maps TO chart j coords, so we show arrows
    
    # Sample for arrows
    n_sample = min(n_arrows, len(overlap_idx))
    sample_idx = np.random.choice(len(overlap_idx), size=n_sample, replace=False)
    
    for k in sample_idx:
        if len(signs) > k:
            color = 'blue' if signs[k] > 0 else 'red'
        else:
            color = 'gray'
        
        # Arrow from z_i to T_ji(z_i) = z_j_from_i
        # But these are in different coordinate systems!
        # Instead, show the displacement in a normalized way
        
        start = z_i_overlap[k]
        end = z_j_from_i[k]
        
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(arrowstyle='->', color=color, alpha=0.5, linewidth=1)
        )
    
    ax.scatter(z_j_from_i[:, 0], z_j_from_i[:, 1], s=10, alpha=0.3, c='red', 
               label='$T_{ji}(z_i) = E_j(D_i(z_i))$')
    
    ax.set_title(f"Transition $T_{{{j}{i}}}$: $z_i \\mapsto E_{j}(D_{i}(z_i))$", fontsize=11)
    ax.set_xlabel("Coordinates")
    ax.set_ylabel("")
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    
    # Add stats
    if len(signs) > 0:
        pos_frac = np.mean(signs > 0)
        stats_str = f"$|U_{i} \\cap U_{j}|$ = {len(overlap_idx)}\ndet>0: {pos_frac:.1%}"
    else:
        stats_str = f"$|U_{i} \\cap U_{j}|$ = {len(overlap_idx)}"
    
    ax.annotate(stats_str, xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    return fig, axes


def plot_all_transitions(
    atlas: AtlasAutoencoder,
    n_arrows: int = 30,
    figsize_per_panel: float = 4.0,
    seed: int = 42,
):
    """
    Plot all pairwise transition maps.
    
    For each overlapping pair (i, j), shows the transition map T_ji
    as arrows from z_i to T_ji(z_i).
    
    Args:
        atlas: Trained AtlasAutoencoder
        n_arrows: Arrows per transition
        figsize_per_panel: Size per subplot
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Collect pairs
    pairs = [(i, j) for (i, j) in atlas.pairwise_overlaps.keys() if i < j]
    
    if not pairs:
        print("No overlapping chart pairs.")
        return None, None
    
    n_pairs = len(pairs)
    n_cols = min(n_pairs, 3)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    figsize = (figsize_per_panel * n_cols, figsize_per_panel * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        
        overlap_idx = atlas.pairwise_overlaps[(i, j)]
        x_overlap = atlas.data[overlap_idx]
        
        z_i = atlas.encode(x_overlap, i)
        
        # Compute T_ji(z_i)
        z_i_tf = tf.constant(z_i, dtype=tf.float32)
        x_recon = atlas.autoencoders[i].decode(z_i_tf)
        z_j_from_i = atlas.autoencoders[j].encode(x_recon).numpy()
        
        # Determinant signs
        _, signs = atlas.compute_determinant_signs(i, j)
        
        # Plot source points
        ax.scatter(z_i[:, 0], z_i[:, 1], s=12, alpha=0.4, c='blue', label='$z_i$')
        ax.scatter(z_j_from_i[:, 0], z_j_from_i[:, 1], s=12, alpha=0.4, c='red', label='$T_{ji}(z_i)$')
        
        # Sample arrows
        n_sample = min(n_arrows, len(overlap_idx))
        sample_idx = np.random.choice(len(overlap_idx), size=n_sample, replace=False)
        
        for k in sample_idx:
            color = 'green' if (len(signs) > k and signs[k] > 0) else 'red'
            ax.annotate(
                '', xy=z_j_from_i[k], xytext=z_i[k],
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, linewidth=0.8)
            )
        
        # Stats
        if len(signs) > 0:
            pos_frac = np.mean(signs > 0)
            title = f"$T_{{{j}{i}}}$: det>0 = {pos_frac:.0%}"
        else:
            title = f"$T_{{{j}{i}}}$"
        
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')
    
    # Hide unused
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Transition Maps: $z_i \\mapsto T_{ji}(z_i) = E_j(D_i(z_i))$", 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig, axes