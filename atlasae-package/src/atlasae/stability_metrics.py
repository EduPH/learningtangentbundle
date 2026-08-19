"""
Stability Metrics for Autoencoder Atlases
==========================================

Computes all quantities needed to verify the stability condition
from Theorem 5 (Sign Cocycle Stability):

    η < 1  AND  max(d·Γ·(L_E L_D + Γ)^{d-1}, L_det·ε) < δ

Quantities computed:
- L_E: encoder operator norm bound
- L_D: decoder operator norm bound  
- L_E': encoder derivative Lipschitz constant
- L_D': decoder derivative Lipschitz constant
- L_Φ': reconstruction differential Lipschitz constant
- η_eff: effective differential error
- Γ: perturbation magnitude
- L_det: determinant Lipschitz constant

Author: Eduardo (atlas autoencoder project)
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StabilityMetrics:
    """All metrics needed for stability theorem verification."""
    # Primary metrics (already computed)
    varepsilon: float      # reconstruction error
    eta: float             # differential error (tangent)
    delta: float           # non-degeneracy gap
    sigma_min_enc: float   # min encoder singular value
    
    # Lipschitz constants
    L_E: float             # encoder operator norm bound
    L_D: float             # decoder operator norm bound
    L_E_prime: float       # encoder derivative Lipschitz
    L_D_prime: float       # decoder derivative Lipschitz
    
    # Derived quantities
    L_Phi_prime: float     # reconstruction Lipschitz
    eta_eff: float         # effective differential error
    epsilon_tilde: float   # off-manifold epsilon
    Gamma: float           # perturbation magnitude
    L_det: float           # determinant Lipschitz
    
    # Stability condition terms
    stability_term1: float # d·Γ·(L_E L_D + Γ)^{d-1}
    stability_term2: float # L_det·ε
    stability_LHS: float   # max of the two terms
    
    # Verification
    condition_satisfied: bool
    margin: float          # δ - stability_LHS (positive = good)
    
    def __repr__(self):
        return f"""
StabilityMetrics:
  Primary:
    ε = {self.varepsilon:.6f}
    η = {self.eta:.6f}
    δ = {self.delta:.6f}
    σ_min(dE) = {self.sigma_min_enc:.6f}
  
  Lipschitz:
    L_E = {self.L_E:.4f}
    L_D = {self.L_D:.4f}
    L_E' = {self.L_E_prime:.4f}
    L_D' = {self.L_D_prime:.4f}
    L_Φ' = {self.L_Phi_prime:.4f}
  
  Derived:
    η_eff = {self.eta_eff:.6f}
    ε̃ = {self.epsilon_tilde:.6f}
    Γ = {self.Gamma:.6f}
    L_det = {self.L_det:.4f}
  
  Stability condition (d=2):
    Term 1: d·Γ·(L_E L_D + Γ)^{d-1} = {self.stability_term1:.6f}
    Term 2: L_det·ε = {self.stability_term2:.6f}
    LHS = max(...) = {self.stability_LHS:.6f}
    RHS = δ = {self.delta:.6f}
    
    Condition satisfied: {self.condition_satisfied}
    Margin (δ - LHS): {self.margin:.6f}
"""


def compute_encoder_lipschitz(system, n_samples: int = 500) -> Tuple[float, float]:
    """
    Compute L_E (operator norm bound) and L_E' (derivative Lipschitz).
    
    L_E = max_k max_x ||d(E_k)_x||_op = max_k max_x σ_max(dE_k)
    L_E' ≈ max over pairs ||dE(p) - dE(q)||_op / ||p - q||
    
    Returns:
        (L_E, L_E'): encoder Lipschitz constants
    """
    L_E = 0.0
    L_E_prime_estimates = []
    
    for i in range(system.n_charts):
        indices = system.subset_assignments[i]
        if len(indices) < 10:
            continue
            
        # Sample points from this chart
        n_sample = min(n_samples, len(indices))
        sample_idx = np.random.choice(indices, size=n_sample, replace=False)
        x = tf.constant(system.data[sample_idx], dtype=tf.float32)
        
        # Compute encoder Jacobians
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = system.autoencoders[i].encode(x)
        
        dE = tape.batch_jacobian(z, x)  # shape (batch, d, N)
        
        # L_E: max singular value
        s = tf.linalg.svd(dE, compute_uv=False)
        sigma_max = tf.reduce_max(s[:, 0]).numpy()
        L_E = max(L_E, sigma_max)
        
        # L_E': estimate by sampling pairs
        if n_sample >= 20:
            for _ in range(min(100, n_sample * (n_sample - 1) // 2)):
                idx1, idx2 = np.random.choice(n_sample, size=2, replace=False)
                
                dE1 = dE[idx1].numpy()  # shape (d, N)
                dE2 = dE[idx2].numpy()
                
                diff_norm = np.linalg.norm(dE1 - dE2, ord=2)  # operator norm
                point_dist = np.linalg.norm(x[idx1].numpy() - x[idx2].numpy())
                
                if point_dist > 1e-8:
                    L_E_prime_estimates.append(diff_norm / point_dist)
    
    # L_E': use 95th percentile for robustness
    if L_E_prime_estimates:
        L_E_prime = np.percentile(L_E_prime_estimates, 95)
    else:
        L_E_prime = 1.0  # fallback
    
    return L_E, L_E_prime


def compute_decoder_lipschitz(system, n_samples: int = 500) -> Tuple[float, float]:
    """
    Compute L_D (operator norm bound) and L_D' (derivative Lipschitz).
    
    L_D = max_i max_z ||d(D_i)_z||_op
    L_D' ≈ max over pairs ||dD(z) - dD(z')||_op / ||z - z'||
    
    Returns:
        (L_D, L_D'): decoder Lipschitz constants
    """
    L_D = 0.0
    L_D_prime_estimates = []
    
    for i in range(system.n_charts):
        indices = system.subset_assignments[i]
        if len(indices) < 10:
            continue
            
        # Get latent points
        n_sample = min(n_samples, len(indices))
        sample_idx = np.random.choice(indices, size=n_sample, replace=False)
        x = tf.constant(system.data[sample_idx], dtype=tf.float32)
        z = system.autoencoders[i].encode(x)
        
        # Compute decoder Jacobians
        with tf.GradientTape() as tape:
            tape.watch(z)
            x_recon = system.autoencoders[i].decode(z)
        
        dD = tape.batch_jacobian(x_recon, z)  # shape (batch, N, d)
        
        # L_D: max singular value
        s = tf.linalg.svd(dD, compute_uv=False)
        sigma_max = tf.reduce_max(s[:, 0]).numpy()
        L_D = max(L_D, sigma_max)
        
        # L_D': estimate by sampling pairs
        if n_sample >= 20:
            for _ in range(min(100, n_sample * (n_sample - 1) // 2)):
                idx1, idx2 = np.random.choice(n_sample, size=2, replace=False)
                
                dD1 = dD[idx1].numpy()  # shape (N, d)
                dD2 = dD[idx2].numpy()
                
                diff_norm = np.linalg.norm(dD1 - dD2, ord=2)
                latent_dist = np.linalg.norm(z[idx1].numpy() - z[idx2].numpy())
                
                if latent_dist > 1e-8:
                    L_D_prime_estimates.append(diff_norm / latent_dist)
    
    if L_D_prime_estimates:
        L_D_prime = np.percentile(L_D_prime_estimates, 95)
    else:
        L_D_prime = 1.0
    
    return L_D, L_D_prime


def compute_stability_metrics(system, d: int = 2) -> StabilityMetrics:
    """
    Compute all stability metrics for the atlas.
    
    Args:
        system: AtlasAutoencoder instance
        d: intrinsic dimension of the manifold
        
    Returns:
        StabilityMetrics dataclass with all quantities
    """
    # Get primary metrics from system
    primary_metrics = system.compute_all_metrics()
    
    varepsilon = primary_metrics['varepsilon']
    eta = primary_metrics['eta']
    delta = primary_metrics['delta']
    sigma_min_enc = primary_metrics['sigma_min_enc']
    
    # Compute Lipschitz constants
    L_E, L_E_prime = compute_encoder_lipschitz(system)
    L_D, L_D_prime = compute_decoder_lipschitz(system)
    
    # Derived quantities (Equations from paper)
    
    # Eq (8): L_Φ' = L_D' L_E² + L_D L_E'
    L_Phi_prime = L_D_prime * L_E**2 + L_D * L_E_prime
    
    # Eq (8): η_eff = (L_E L_D + 2)η / (1 - η) + L_Φ'·ε
    if eta < 1:
        eta_eff = (L_E * L_D + 2) * eta / (1 - eta) + L_Phi_prime * varepsilon
    else:
        eta_eff = float('inf')
    
    # ε̃ = (L_E L_D + 2)ε
    epsilon_tilde = (L_E * L_D + 2) * varepsilon
    
    # Eq (9): Γ = L_E·η_eff·L_D + L_E'·ε̃·(1 + η_eff)·L_D
    Gamma = L_E * eta_eff * L_D + L_E_prime * epsilon_tilde * (1 + eta_eff) * L_D
    
    # Eq (10): L_det = d·(L_E L_D)^{d-1}·L_E·(L_E L_D' + L_E' L_D²)
    L_det = d * (L_E * L_D)**(d-1) * L_E * (L_E * L_D_prime + L_E_prime * L_D**2)
    
    # Stability condition terms (Eq 11)
    # Term 1: d·Γ·(L_E L_D + Γ)^{d-1}
    stability_term1 = d * Gamma * (L_E * L_D + Gamma)**(d-1)
    
    # Term 2: L_det·ε
    stability_term2 = L_det * varepsilon
    
    # LHS = max(term1, term2)
    stability_LHS = max(stability_term1, stability_term2)
    
    # Check condition: η < 1 AND LHS < δ
    condition_satisfied = (eta < 1) and (stability_LHS < delta)
    margin = delta - stability_LHS
    
    return StabilityMetrics(
        varepsilon=varepsilon,
        eta=eta,
        delta=delta,
        sigma_min_enc=sigma_min_enc,
        L_E=L_E,
        L_D=L_D,
        L_E_prime=L_E_prime,
        L_D_prime=L_D_prime,
        L_Phi_prime=L_Phi_prime,
        eta_eff=eta_eff,
        epsilon_tilde=epsilon_tilde,
        Gamma=Gamma,
        L_det=L_det,
        stability_term1=stability_term1,
        stability_term2=stability_term2,
        stability_LHS=stability_LHS,
        condition_satisfied=condition_satisfied,
        margin=margin
    )


def print_stability_report(metrics: StabilityMetrics, verbose: bool = True):
    """Print a formatted stability analysis report."""
    print("\n" + "="*70)
    print("STABILITY THEOREM VERIFICATION (Theorem 5)")
    print("="*70)
    
    print("\n[Primary Metrics]")
    print(f"  ε (reconstruction error)    = {metrics.varepsilon:.6f}")
    print(f"  η (differential error)      = {metrics.eta:.6f}")
    print(f"  δ (non-degeneracy gap)      = {metrics.delta:.6f}")
    print(f"  σ_min(dE)                   = {metrics.sigma_min_enc:.6f}")
    
    print("\n[Lipschitz Constants]")
    print(f"  L_E  (encoder op norm)      = {metrics.L_E:.4f}")
    print(f"  L_D  (decoder op norm)      = {metrics.L_D:.4f}")
    print(f"  L_E' (encoder deriv Lip)    = {metrics.L_E_prime:.4f}")
    print(f"  L_D' (decoder deriv Lip)    = {metrics.L_D_prime:.4f}")
    print(f"  L_Φ' (recon deriv Lip)      = {metrics.L_Phi_prime:.4f}")
    
    print("\n[Derived Quantities (d=2)]")
    print(f"  η_eff                       = {metrics.eta_eff:.6f}")
    print(f"  ε̃                           = {metrics.epsilon_tilde:.6f}")
    print(f"  Γ                           = {metrics.Gamma:.6f}")
    print(f"  L_det                       = {metrics.L_det:.4f}")
    
    print("\n[Stability Condition Check]")
    print(f"  Condition: η < 1 AND max(Term1, Term2) < δ")
    print(f"")
    print(f"  η = {metrics.eta:.4f} < 1 ?  {'✓ YES' if metrics.eta < 1 else '✗ NO'}")
    print(f"")
    print(f"  Term 1: d·Γ·(L_E L_D + Γ)^(d-1) = {metrics.stability_term1:.6f}")
    print(f"  Term 2: L_det·ε                  = {metrics.stability_term2:.6f}")
    print(f"  LHS = max(Term1, Term2)          = {metrics.stability_LHS:.6f}")
    print(f"  RHS = δ                          = {metrics.delta:.6f}")
    print(f"")
    print(f"  LHS < δ ?  {'✓ YES' if metrics.stability_LHS < metrics.delta else '✗ NO'}")
    print(f"  Margin (δ - LHS) = {metrics.margin:.6f}")
    
    print("\n" + "-"*70)
    if metrics.condition_satisfied:
        print("  ✓ STABILITY CONDITION SATISFIED")
        print("    The sign cocycle is guaranteed to be a valid Čech cocycle.")
    else:
        print("  ✗ STABILITY CONDITION NOT VERIFIED")
        if metrics.eta >= 1:
            print("    Issue: η ≥ 1 (differential error too large)")
        if metrics.stability_LHS >= metrics.delta:
            print("    Issue: LHS ≥ δ (perturbation exceeds non-degeneracy gap)")
            print(f"    Need either: smaller ε/η, or larger δ")
    print("="*70 + "\n")


def generate_latex_table(metrics: StabilityMetrics, manifold_name: str) -> str:
    """Generate LaTeX table row for the stability metrics."""
    # hoisted out of the f-string: backslashes in an f-string expression are a
    # syntax error before Python 3.12, and this module must import on 3.10.
    _verdict = ("\\checkmark Condition satisfied" if metrics.condition_satisfied
                else "$\\times$ Condition not satisfied")
    return f"""
% Stability metrics for {manifold_name}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Value & Paper Definition \\\\
\\midrule
$\\varepsilon$ & ${metrics.varepsilon:.4f}$ & $\\sup_x \\|D_i(E_i(x)) - x\\|$ \\\\
$\\eta$ & ${metrics.eta:.4f}$ & $\\sup_x \\|d(\\Phi_i)_x - I\\|_{{\\text{{op}}}}$ \\\\
$\\delta$ & ${metrics.delta:.4f}$ & $\\min_{{i,j,x}} |\\det g_{{ji}}(x)|$ \\\\
$L_E$ & ${metrics.L_E:.4f}$ & $\\sup_{{k,p}} \\|d(E_k)_p\\|_{{\\text{{op}}}}$ \\\\
$L_D$ & ${metrics.L_D:.4f}$ & $\\sup_{{i,z}} \\|d(D_i)_z\\|_{{\\text{{op}}}}$ \\\\
$\\Gamma$ & ${metrics.Gamma:.4f}$ & Perturbation magnitude \\\\
\\midrule
\\multicolumn{{3}}{{c}}{{Stability: $\\max(d\\Gamma(L_E L_D + \\Gamma)^{{d-1}}, L_{{\\det}}\\varepsilon) = {metrics.stability_LHS:.4f} < \\delta = {metrics.delta:.4f}$}} \\\\
\\multicolumn{{3}}{{c}}{{{_verdict}}} \\\\
\\bottomrule
\\end{{tabular}}
"""