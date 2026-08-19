# atlasae — Learning Tangent Bundles and Characteristic Classes with Autoencoder Atlases

Code for the paper *"Learning Tangent Bundles and Characteristic Classes with
Autoencoder Atlases"* (E. Paluzo-Hidalgo, Y. Ike). A collection of locally
trained autoencoders is treated as a learned atlas on a data manifold; the
linearised transition maps define a vector bundle whose first Stiefel–Whitney
class detects orientability, with a stability theory whose hypotheses are
checked directly on the trained networks.

## Install

```bash
pip install -e .            # core (numpy/scipy/sklearn/matplotlib/tensorflow)
pip install -e ".[tda]"     # + dreimac/ripser/persim (some covers, COIL-20)
```

## Layout

```
src/atlasae/            the package (mirrored standalone in atlasae-package/)
  atlasautoencoder.py     AtlasAutoencoder (charts, losses, metrics)
  fast_training.py        compiled trainer + certificate-aware protocol
  eta_true.py             direct tangent-restricted eta (PCA / analytic frames)
  orientability.py        sign cocycle, cocycle verification, coboundary test
  sign_constancy.py       per-overlap sign-constancy margins (prop:sign-constancy)
  verdict_certificate.py  certified verdict from the margins (odd-cycle asymmetry)
  gauge.py                latent gauge fixing and the delta ceiling (prop:gauge)
  persistence.py          save_atlas / load_atlas (bit-identical reload)
  stability_metrics.py    epsilon, eta, delta, cocycle error on a trained atlas
  viz.py                  latent charts + signed nerve (plot_atlas)
  *_cover*.py, cover.py   cover constructions
experiments/            drivers and results — see experiments/README.md for the
                        map from each run to the table or figure it backs
  paper_experiments.py    synthetic manifolds (S2, Mobius, Klein, RP2, 3-manifolds)
  make_master_table.py    tab:summary_all, tab:master, tab:pertrial
  audit_paper.py          recomputes every number quoted in prose and captions
  codim_sweep.py          codimension sweeps feeding the eta validation
  E1_eta_codim/           eta_pca vs analytic eta
  E4_real_data/           cyclo-octane and natural image patches
  E5_theta/ E6_gauge/     the Theta gap, gauge fixing
  E7_signconstancy/       margins; eval_saved.py (re-evaluate saved atlases),
                          eval_review_todos.py (discards, splitting-rule and
                          margin robustness; results_review_todos/)
  results/                every run the paper cites
  _superseded/            kept for provenance, cited nowhere (gitignored)
atlasae-package/        standalone copy of the package for Zenodo/community use
docs/                   experiment plan and notes
tools/sync_package.py   mirror src/atlasae/ into atlasae-package/ (--check to verify)
```

The LaTeX sources of the paper are not part of this repository; `REGENERATE.md`
describes where the figure-producing scripts write them locally.

The package is published separately, as its own repository and on Zenodo:

> **<https://github.com/EduPH/atlasae>** — `pip install`-able, paper-independent.

Install it from there if you only want the method; this repository is the
paper's experiments, results and reproduction scripts.

`atlasae-package/src/atlasae/` is a byte-identical copy of `src/atlasae/` and
is the staging area for that repository, so the two can drift. Before tagging a
release or uploading to Zenodo, run

```bash
python tools/sync_package.py --check    # exit 1 and a file list if out of sync
python tools/sync_package.py            # mirror src/ -> package/
```

Reproduce every number in the paper with `./run_all.sh` (roughly 6–9 h; see
`REGENERATE.md`). To check that a change has not moved any reported value, see
the three commands under "Checking that a change did not move the numbers" in
`experiments/README.md`.

## Quick start

```python
import numpy as np
from atlasae import (AtlasAutoencoder, fast_fit, check_orientability,
                     pca_tangent_frames, compare_eta, tetrahedral_cover_S2)

pts = np.random.randn(1000, 3); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
assign = tetrahedral_cover_S2(epsilon=0.3).get_assignments(pts)

system = AtlasAutoencoder(data=pts, n_charts=4, subset_assignments=assign,
                          latent_dim=2, hidden_dims=[32, 16])
fast_fit(system, epochs=2000, lambda_jac=0.01, lambda_diff=0.01)

frames = pca_tangent_frames(pts, d=2, k=25)
eta = compare_eta(system, pts, assign, frames_pca=frames)   # theorem hypotheses
result = check_orientability(system, pts, assign, eps_cluster=1.0, min_points=5)
print(eta['eta_pca'], result['is_orientable'])
```

Sweep driver:

```bash
cd experiments
python codim_sweep.py --manifold Klein --dims 4 25 100 --seeds 5
python codim_sweep.py --plot <results_dir>/results.json
```

## Citation

If you use this code, please cite the paper:

> E. Paluzo-Hidalgo and Y. Ike, *Learning Tangent Bundles and Characteristic
> Classes with Autoencoder Atlases*, 2026.

If you use the package itself rather than these experiments, cite the software
too: it is released separately at <https://github.com/EduPH/atlasae>, with a
Zenodo DOI and `CITATION.cff` metadata. The copy staged here lives in
[`atlasae-package/`](atlasae-package/).

```
@software{paluzo_hidalgo_2026_22005040,
  author       = {Paluzo-Hidalgo, Eduardo},
  title        = {atlasae: autoencoder atlases, tangent bundles, and
                   Stiefel-Whitney classes from data
                  },
  month        = aug,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.22005040},
  url          = {https://doi.org/10.5281/zenodo.22005040},
}
```
