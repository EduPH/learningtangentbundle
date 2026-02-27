# Autoencoder Atlases: Learning Tangent Bundles and Characteristic Classes

Code for the paper *"Learning Tangent Bundles and Characteristic Classes with Autoencoder Atlases"* by Eduardo Paluzo-Hidalgo and Yuichi Ike.

## Overview

This repository implements **autoencoder atlases**: collections of locally trained encoder, decoder pairs that form a learned atlas on a data manifold. Rather than computing a single global embedding, we train local autoencoders on overlapping chart domains and extract differential, topological invariants from the transition maps between charts.

The key theoretical insight is that reconstruction-consistent autoencoders automatically satisfy the cocycle condition, and linearizing their transition maps yields a vector bundle isomorphic to the tangent bundle. This gives direct access to **characteristic classes**, in particular the first Stiefel–Whitney class $w_1$, whose vanishing characterizes orientability.

### What this code does

Given a point cloud sampled from a manifold $M$:

1. **Learns an atlas** $\mathcal{A} = \{(U_i, E_i, D_i)\}$ of local autoencoders on an open cover $\{U_i\}$.
2. **Extracts transition maps** $T_{ji} = E_j \circ D_i$ between overlapping charts.
3. **Computes the Jacobian sign cocycle** $\omega_{ji}(x) = sign(\det\, d(T_{ji})_{E_i(x)})$.
4. **Tests orientability** by checking whether $\omega$ is a coboundary in $C^1(\mathcal{U}; \mathbb{Z}/2)$.

## Repository Structure

```
.
├── atlasautoencoder.py        # Core library: LocalAutoencoder, AtlasAutoencoder, training, metrics, visualisation
├── orientability.py           # Orientability detection: sign cocycle, coboundary test, connected component handling
├── sphere_good_cover.py       # Good cover construction for S² (tetrahedral cover, stereographic projection)
├── Sphere.py                  # Experiment: S² (orientable, expected w₁ = 0)
├── mobius.py                  # Experiment: Möbius band (non-orientable, expected w₁ ≠ 0)
├── Klein_bottle.py            # Experiment: Klein bottle in ℝ⁴ (non-orientable, expected w₁ ≠ 0)
├── line_patch_images.py       # Experiment: ℝP² via line patch images (non-orientable, expected w₁ ≠ 0)
└── README.md
```

## Mathematical Background

### Autoencoder Atlas

An **autoencoder atlas** on a manifold $M$ is a collection $\mathcal{A} = \{(U_i, E_i, D_i)\}_{i \in I}$ where:

- $\{U_i\}$ is an open cover of $M$
- $E_i: U_i \to Z_i \subset \mathbb{R}^d$ is the encoder (local coordinates)
- $D_i: Z_i \to M$ is the decoder (inverse chart map)
- $D_i \circ E_i \approx Id_{U_i}$ (reconstruction consistency)

### Transition Maps and Cocycle Condition

For overlapping charts $U_i \cap U_j \neq \emptyset$, the **transition map** is

$$T_{ji} = E_j \circ D_i : E_i(U_i \cap U_j) \to E_j(U_i \cap U_j).$$

When reconstruction is exact ($D_i \circ E_i = Id$), these satisfy the **cocycle condition** on triple overlaps:

$$T_{ki} = T_{kj} \circ T_{ji} \quad \text{on } U_i \cap U_j \cap U_k.$$

This is not imposed as a regularisation term — it emerges algebraically from reconstruction consistency alone.

### Linearised Transition Maps and Tangent Bundle

The **linearised transition maps** $g_{ji}(x) = d(T_{ji})_{E_i(x)} \in GL_d(\mathbb{R})$ define a $GL_d(\mathbb{R})$-cocycle. When $d = \dim M$ and the atlas is compatible with the smooth structure, the resulting vector bundle $\mathcal{T}_{\mathcal{A}}$ is isomorphic to the tangent bundle $TM$.

### Orientability via the Sign Cocycle

The **sign cocycle** $\omega_{ji}(x) = sign(\det\, g_{ji}(x)) \in \{+1, -1\}$ defines a Čech 1-cocycle in $C^1(\mathcal{U}; \mathbb{Z}/2)$. The manifold $M$ is **orientable** if and only if $[\omega] = 0$ in $H^1(M; \mathbb{Z}/2)$, which is equivalent to $\omega$ being a coboundary: there exist signs $\nu_i \in \{+1, -1\}$ such that

$$\omega_{ji} = \nu_j \cdot \nu_i \quad \text{on each connected component of } U_i \cap U_j.$$

This is the **first Stiefel–Whitney class**: $w_1(TM) = [\omega]$.

### Theoretical Metrics

The code tracks three key quantities from the paper:

| Symbol | Name | Definition | Role |
|--------|------|------------|------|
| $\varepsilon$ | Reconstruction error | $\sup_x \\|D_i(E_i(x)) - x\\|$ | Approximation quality |
| $\eta$ | Differential error | $\sup_x \\|d(D_i \circ E_i)_x\|_{T_xM} - Id\\|\_{\mathrm{op}}$ | Tangent map fidelity |
| $\delta$ | Non-degeneracy gap | $\min_{i,j,x} \|\det\, g_{ji}(x)\|$ | Sign cocycle stability |

When $\delta > 0$ and $\eta$ is sufficiently small, the sign cocycle is stable under perturbations and correctly detects orientability.

## Installation

### Core dependencies (S², Möbius band)

```bash
pip install tensorflow numpy matplotlib scipy scikit-learn
```

### Additional dependencies (Klein bottle, ℝP²)

```bash
pip install dreimac ripser persim
```

### Full installation

```bash
pip install -r requirements.txt
```

**Python version:** 3.9+ recommended. **TensorFlow:** 2.10+ required.

## Usage

### Running experiments

Each experiment script is self-contained and runs multiple trials with configurable random seeds:

```bash
# Orientable manifolds
python Sphere.py           # S² — expects orientable (w₁ = 0)

# Non-orientable manifolds
python mobius.py            # Möbius band — expects non-orientable (w₁ ≠ 0)
python Klein_bottle.py      # Klein bottle — expects non-orientable (w₁ ≠ 0), requires dreimac
python line_patch_images.py # ℝP² line patches — expects non-orientable (w₁ ≠ 0), requires dreimac, ripser
```

Each script outputs:
- A JSON file with full metrics from all trials (reconstruction error, differential error, non-degeneracy gap, orientability verdict, etc.)
- Summary statistics suitable for paper tables
- Diagnostic figures saved to disk

### Using the library directly

```python
import numpy as np
from atlasautoencoder import AtlasAutoencoder, create_cover_from_neighborhoods
from orientability import check_orientability

# 1. Sample data from a manifold
theta = np.random.uniform(0, 2*np.pi, 1000)
phi = np.random.uniform(0, np.pi, 1000)
X = np.column_stack([
    np.sin(phi)*np.cos(theta),
    np.sin(phi)*np.sin(theta),
    np.cos(phi)
])

# 2. Create an open cover
cover = create_cover_from_neighborhoods(X, n_charts=4, overlap_ratio=0.3)

# 3. Build and train the atlas
atlas = AtlasAutoencoder(
    data=X,
    n_charts=4,
    subset_assignments=cover,
    latent_dim=2,
    hidden_dims=[32, 16]
)
atlas.fit(epochs=200,  lambda_jac=0.01)

# 4. Inspect theoretical metrics
metrics = atlas.print_metrics_summary()
# Prints ε, η, δ, cocycle error, σ_min(dE)

# 5. Test orientability
result = check_orientability(atlas, X, cover)
print(f"Orientable: {result['orientable']}")
```

### Core API

**`AtlasAutoencoder`** — the central class.

| Method | Description |
|--------|-------------|
| `fit(epochs, batch_size, lambda_jac, lambda_cocycle)` | Train all local autoencoders |
| `encode(x, chart)` / `decode(z, chart)` | Encode/decode with a specific chart |
| `transition_map(z, i, j)` | Compute $T_{ji}(z) = E_j(D_i(z))$ |
| `compute_transition_jacobians(i, j)` | Jacobian matrices of $T_{ji}$ on overlap |
| `compute_determinant_signs(i, j)` | Signs of $\det(dT_{ji})$ on overlap |
| `compute_all_metrics()` | Full $\varepsilon$, $\eta$, $\delta$, cocycle error |
| `print_metrics_summary()` | Formatted summary with stability diagnostics |

**`check_orientability`** — the main orientability detection function.

Takes a trained `AtlasAutoencoder`, the data, and the cover. Handles disconnected chart domains via DBSCAN clustering, computes the sign cocycle on each connected component of each pairwise overlap, verifies the cocycle condition on triple overlaps, and tests whether the cocycle is a coboundary.

### Loss function

The training loss for each chart combines three terms:

$$\mathcal{L}_i = \underbrace{\frac{1}{|U_i|}\sum_{x \in U_i} \|x - D_i(E_i(x))\|^2}_{\text{reconstruction}} + \lambda_J \underbrace{\max(0,\, \epsilon_J - \sigma_{\min}(dE_i))}_{\text{Jacobian regularity}}$$

An optional cocycle loss on triple overlaps can be added but is not required — cocycle consistency emerges from reconstruction alone.

## Experiments

### Test manifolds

| Manifold | Orientable | $\dim$ | Ambient $\dim$ | Cover type | Key challenge |
|----------|-----------|--------|----------------|------------|---------------|
| $S^2$ | ✓ | 2 | 3 | Tetrahedral (4 charts) | Good cover with contractible intersections |
| Möbius band | ✗ | 2 | 3 | 2-chart y-split | Overlap has 2 components with opposite signs |
| Klein bottle | ✗ | 2 | 4 | Geodesic landmark | Cannot embed in $\mathbb{R}^3$ without self-intersection |
| $\mathbb{R}P^2$ (line patches) | ✗ | 2 | 100 | Geodesic landmark | High-dimensional image data; non-embeddable in $\mathbb{R}^3$ |

### Cover construction

The quality of the open cover is often the limiting factor. The repository provides:

- **`sphere_good_cover.py`**: Tetrahedral good cover for $S^2$ based on inscribed tetrahedron vertices, with stereographic projection for each chart. All intersections are contractible by construction.
- **`create_cover_from_neighborhoods`** (in `atlasautoencoder.py`): General-purpose cover from random landmark points with geodesic balls.
- **Geodesic landmark covers** (in Klein bottle / ℝP² scripts): Uses DREiMaC's `GeometryUtils` for approximate geodesic distances.

## Diagnostics

A trial is considered **converged** when the following conditions hold simultaneously:

- **$\delta > 0$**: The non-degeneracy gap is positive, meaning all transition map Jacobians are non-singular. This ensures the sign cocycle is well-defined.
- **Sign consistency**: On each connected component of each overlap $U_i \cap U_j$, all sampled determinant signs agree (either all $+1$ or all $-1$).
- **Cocycle condition**: On triple overlaps, $\omega_{ki} = \omega_{kj} \cdot \omega_{ji}$ holds.

When these conditions fail, the trial has not converged to a valid atlas and its orientability verdict should be discarded. The experiment scripts report both raw accuracy (all trials) and converged accuracy (trials meeting the above criteria).

## Citation

If you use this code, please cite:

```bibtex
@misc{paluzohidalgo2026learningtangentbundlescharacteristic,
      title={Learning Tangent Bundles and Characteristic Classes with Autoencoder Atlases}, 
      author={Eduardo Paluzo-Hidalgo and Yuichi Ike},
      year={2026},
      eprint={2602.22873},
      archivePrefix={arXiv},
      primaryClass={math.AT},
      url={https://arxiv.org/abs/2602.22873}, 
}
```

## Use of AI

Claude was used extensively for code and repository documentation.
