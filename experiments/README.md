# Experiments

Every number in the paper is produced by the code in this directory from the
runs stored under `results/`. This file records which run backs which table and
figure, so that a claim can be traced to a file without rerunning anything.

Run everything with `../run_all.sh` (see `../REGENERATE.md` for options).

## Layout

```
experiments/
  paper_experiments.py      synthetic manifolds (S2, Mobius, Klein, RP2)
  make_master_table.py      tab:summary_all (--summary) and tab:master (--latex)
  make_rp2_embedding_fig.py fig:rp2-embedding
  codim_sweep.py            codimension sweep feeding the eta validation

  E1_eta_codim/             eta_pca vs analytic eta        -> app:eta-validation
  E4_real_data/             cyclo-octane, image patches    -> sec:exp-cyclooctane, sec:exp-patches
  E5_theta/                 the Theta < delta gap          -> tab:theta
  E6_gauge/                 latent gauge fixing            -> tab:gauge
  E7_signconstancy/         sign-constancy margins and the -> Margin column, def:certificate,
                            verdict certificate               sec:exp-signconstancy

  results/                  every run the paper cites (see below)
  _superseded/              runs and code kept for provenance, cited nowhere
```

`results/` is where `paper_experiments.py` writes and where
`make_master_table.py` and `eta_validation.py` look. `_superseded/` is
gitignored and is skipped by `E7_signconstancy/eval_saved.py`, so archived runs
can never leak into a reported total.

Trained atlases live in `<run>/atlases/` and are **gitignored** (~50 MB, 68
atlases). They are what `E7_signconstancy/eval_saved.py` reads, so margins can
be recomputed without retraining — but a *fresh clone has none*. To recompute
the certificate on a new machine you must first retrain with `../run_all.sh`,
or copy the `atlases/` directories across.

## Which run backs which result

`make_master_table.py` selects the **newest** run matching each pattern; the
table below is that selection as of 2026-08-05.

| Result | Run |
|---|---|
| `tab:summary_all`, `tab:master` — $S^2$ | `results/results_paper_20260805_101332` (n=4000) |
| — Möbius band | `results/results_paper_20260804_091501` |
| — Klein bottle ($\R^4$) | `results/results_paper_20260805_101547` (n=4000) |
| — $\mathbb{R}P^2$ raw | `results/results_paper_20260804_110243` |
| — $\mathbb{R}P^2$ normalised | `results/results_paper_20260804_095231` |
| — image patches (model) | `E4_real_data/results_patches_20260804_195745` |
| — image patches (van Hateren) | `E4_real_data/results_patches_20260804_202645` |
| `tab:cyclooctane` | `E4_real_data/results_cyclooctane_20260804_193749` |
| Certificate (third condition), Margin column, all experiments | `E7_signconstancy/results_certificate` |
| `fig:epsfloor`, `app:epsfloor` | `E4_real_data/results_epsfloor_{real_20260804_225654, synthetic_20260805_004929}` |
| `tab:gauge` | `E6_gauge/results_gauge_20260805_021727` |
| `tab:theta` | computed from the runs above by `E5_theta/theta_certificate.py` |
| `app:eta-validation` (116 trials) | **18 directories** — see below |

### The eta validation draws on more than the latest runs

`E1_eta_codim/eta_validation.py` globs *every* run that records both
`eta_true` and `eta_pca`, not just the newest. The reported 116 paired trials
come from 18 directories:

- `E1_eta_codim/results/results_codim_Klein_{20260714_230528, 20260716_090907, 20260716_093831}` — 22, 5, 6 trials
- `E1_eta_codim/results/results_codim_S2_20260716_104014` — 25 trials
- `results/results_paper_{20260730_132703, 20260730_133224, 20260730_133554}` — 5 each
- `results/results_paper_{20260803_110721, 20260803_111141, 20260803_111507}` — 5 each
- `results/results_paper_{20260804_090508, 20260804_090520, 20260804_090528}` — 1 each
- `results/results_paper_{20260804_091100, 20260804_091501, 20260804_091734}` — 5 each
- `results/results_paper_{20260805_101332, 20260805_101547}` — 5 each

**Do not archive any of these**, including the small and old ones. They look
superseded and are not: deleting them silently lowers the 116/110 counts in
`app:eta-validation`.

Two of them are cited a second time, as the n=1000 baselines of the sampling
claim in `sec:exp-signconstancy`:

- `results/results_paper_20260804_091100` — $S^2$ at n=1000, the 35/55 margin
- `results/results_paper_20260804_091734` — Klein at n=1000, the 60/158 margin

## Checking that a change did not move the numbers

```bash
cd experiments
python make_master_table.py            # 55 trials, 36 certified, 52/52 correct, 3 undetermined
python E1_eta_codim/eta_validation.py  # 116 paired trials, 110/116 agree, 0 false certifications
python E5_theta/theta_certificate.py   # Theta < delta: no, in every experiment
```

The certificate's third condition comes from `E7_signconstancy/results_certificate`,
which holds the per-overlap-component margins and the verdict certificate for
all 65 saved atlases. Regenerate it with

```bash
python E7_signconstancy/eval_saved.py --into results_certificate \
    --exclude 20260804_090508,20260804_090520,20260804_090528
```

`--into` **resumes**: it skips atlases already present in the directory, so
running the command above against the existing `results_certificate` reports
"nothing to do". To genuinely recompute, delete the directory first or pass a
different `--into`. Add `--budget N` to stop after N seconds and continue on
the next invocation (the full 65 take roughly 15 minutes, more than a single
shell call may allow).

The exclusions are the three one-seed runs kept only for the eta validation.
Their atlases are undertrained — one of them is a Mobius atlas with
eps = 0.54 that gets the verdict wrong — and including them would mix that into
the margin totals. They are correctly refused by the full certificate, but
there is no reason to evaluate them at all.

Diff the output against a run from before the change. That is how this
directory was reorganised without moving any reported value.

## Input data (gitignored, downloadable)

- `E4_real_data/vanhateren_iml/` — van Hateren natural image database
- `E4_real_data/pointsCycloOctane.mat` — cyclo-octane conformations
- `E4_real_data/teaser_cache.npz` — cached Isomap projection for `fig:teaser`

## `_superseded/`

Kept on disk, gitignored, cited nowhere: earlier sweeps, the pre-restructure
`paper_v1/` scripts, `E3_cover_robustness/`, the top-level codimension sweeps,
the single-atlas margin probes, and the volume-regularisation runs (a dead end,
recorded in `../REGENERATE.md` §5b). Safe to delete once you are satisfied the
three commands above still reproduce.
