"""
Optimized training loop for AtlasAutoencoder.

Speedups over AtlasAutoencoder.fit (identical losses, different plumbing):
  1. Whole train step compiled in @tf.function with a FIXED input shape
     per chart -> batch_jacobian's pfor function is traced once, not per call.
  2. No per-step metrics: .fit() computes eps, an NxN Jacobian + SVD (eta)
     and sigma_min on EVERY batch purely for logging. Here metrics are
     computed only when requested (between stages).
  3. drop_remainder batching -> constant shapes, no retracing.
  4. Cosine learning-rate decay (1e-3 -> 1e-4) instead of constant 1e-3.

Staged protocol for convergence rate (train_until_certified):
  scout:  short run; abandon clearly hopeless initialisations early
          instead of spending the full budget on them.
  main:   blocks of `block` epochs; after each block past `check_from`,
          evaluate the (ground-truth-free) certificate; stop when it passes.
  polish: short low-LR phase.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')


# ============================================================
# Compiled per-chart step
# ============================================================

def _make_step(ae, opt, lambda_jac: float, lambda_diff_var, batch_size: int,
               dim: int, jac_eps: float = 1e-3, diff_hinge: float = 0.0,
               lambda_vol: float = 0.0):
    """
    Build a compiled train step for one chart (fixed batch shape).

    lambda_diff_var is a tf.Variable so its value can be annealed (warmup)
    between epochs without retracing the tf.function.

    diff_hinge > 0 switches the differential loss to a HINGE form:
    penalise only the excess of the per-point round-trip Jacobian error
    over the margin `diff_hinge` (i.e. max(0, ||d(E o D) - I||_F - tau)^2).
    This leaves already-good charts (eta well below 1) untouched and
    concentrates gradient on the charts that violate the hypothesis,
    avoiding the destabilising eta^2 gradients of the plain Frobenius form.
    """

    @tf.function(input_signature=[tf.TensorSpec([batch_size, dim], tf.float32)])
    def step(xb):
        with tf.GradientTape() as tape:
            z = ae.encoder(xb)
            recon = ae.decoder(z)
            loss = tf.reduce_mean(tf.reduce_sum((xb - recon) ** 2, axis=1))

            if lambda_jac > 0 or lambda_vol > 0:
                with tf.GradientTape() as t2:
                    t2.watch(xb)
                    z2 = ae.encoder(xb)
                J = t2.batch_jacobian(z2, xb)
                JJT = tf.matmul(J, J, transpose_b=True)
                eig = tf.linalg.eigvalsh(JJT)
                if lambda_jac > 0:
                    sigma_min = tf.sqrt(eig[:, 0] + 1e-8)
                    loss += lambda_jac * tf.reduce_mean(
                        tf.maximum(0.0, jac_eps - sigma_min))
                if lambda_vol > 0:
                    # Volume uniformity.  delta's attainable ceiling under any
                    # latent rescaling is min_ij sqrt(inf|det g_ji|/sup|det g_ji|)
                    # (gauge.py), so it is bounded by how much the volume
                    # distortion of the encoders varies WITHIN a chart.  Making
                    # log|det dE_i| constant over U_i drives that ratio to 1.
                    # log sqrt(det J J^T) = 0.5 * sum log eig
                    logvol = 0.5 * tf.reduce_sum(
                        tf.math.log(eig + 1e-8), axis=1)
                    loss += lambda_vol * tf.math.reduce_variance(logvol)

            zc = tf.stop_gradient(ae.encoder(xb))
            with tf.GradientTape() as t3:
                t3.watch(zc)
                z_round = ae.encoder(ae.decoder(zc))
            Jr = t3.batch_jacobian(z_round, zc)
            I_d = tf.eye(tf.shape(zc)[1], batch_shape=[tf.shape(zc)[0]])
            if diff_hinge > 0:
                # per-point Frobenius error, hinged at the margin
                fro = tf.sqrt(tf.reduce_sum((Jr - I_d) ** 2, axis=[1, 2]) + 1e-12)
                diff_term = tf.reduce_mean(tf.maximum(0.0, fro - diff_hinge) ** 2)
            else:
                diff_term = tf.reduce_mean(
                    tf.reduce_sum((Jr - I_d) ** 2, axis=[1, 2]))
            loss += lambda_diff_var * diff_term

        grads = tape.gradient(loss, ae.trainable_variables)
        opt.apply_gradients(zip(grads, ae.trainable_variables))
        return loss

    return step


# ============================================================
# Fast fit
# ============================================================

def fast_fit(system, epochs: int, batch_size: int = 64,
             lambda_jac: float = 0.0, lambda_diff: float = 0.0,
             lr_start: float = 1e-3, lr_end: float = 1e-4,
             lr_schedule: bool = True, diff_hinge: float = 0.0,
             diff_warmup: float = 0.0, lambda_vol: float = 0.0,
             verbose: bool = False):
    """
    Drop-in replacement for AtlasAutoencoder.fit (reconstruction + jac +
    diff losses; smoothness assumed 0 as in all sweep configs).
    Reuses system.optimizers, so repeated calls continue training.

    diff_hinge: margin tau for the hinge differential loss (0 = plain
        Frobenius, original behaviour).
    diff_warmup: fraction of epochs over which lambda_diff ramps linearly
        from 0 to its full value (0 = full weight from the start). Lets
        reconstruction converge before the differential term engages.
    lambda_vol: weight on the volume-uniformity term, which penalises the
        variance of log|det dE_i| over each chart.  This targets the
        non-degeneracy gap delta: its attainable ceiling under any latent
        rescaling is min_ij sqrt(inf|det g_ji| / sup|det g_ji|), so a chart
        whose volume distortion varies widely caps delta far below 1 no
        matter how the latent coordinates are scaled (see atlasae.gauge).
    """
    dim = system.data.shape[1]
    charts = []
    for i, ae in enumerate(system.autoencoders):
        subset = system.data[system.subset_assignments[i]]
        if len(subset) == 0:
            continue
        bs = min(batch_size, len(subset))
        ds = (tf.data.Dataset.from_tensor_slices(subset.astype(np.float32))
              .shuffle(len(subset), reshuffle_each_iteration=True)
              .batch(bs, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
        ld_var = tf.Variable(lambda_diff, dtype=tf.float32, trainable=False)
        step = _make_step(ae, system.optimizers[i], lambda_jac, ld_var,
                          bs, dim, diff_hinge=diff_hinge,
                          lambda_vol=lambda_vol)
        charts.append((i, ds, step, system.optimizers[i], ld_var))

    warm_epochs = int(diff_warmup * epochs)
    for epoch in range(epochs):
        if lr_schedule and epochs > 1:
            t = epoch / (epochs - 1)
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * t))
            for c in charts:
                c[3].learning_rate.assign(lr)
        if warm_epochs > 0:
            frac = min(1.0, (epoch + 1) / warm_epochs)
            for c in charts:
                c[4].assign(lambda_diff * frac)
        for _, ds, step, _, _ in charts:
            for xb in ds:
                step(xb)
        if verbose and epoch % 200 == 0:
            print(f"    epoch {epoch}")


# ============================================================
# Staged, certificate-aware training
# ============================================================

def train_until_certified(
    system,
    certificate: Callable[[], tuple],  # () -> (eps_ok, cert_ok, eps, n_out, delta)
    sup_eps: Callable[[], float],
    total_epochs: int = 5000,
    scout_epochs: int = 500,
    # DISABLED by default (inf): on Klein-scale data, eps sits at ~2.5-3.5
    # after 500 epochs for good AND bad seeds alike — an early absolute bar
    # cannot discriminate and aborts everything. Re-enable only with a
    # trajectory-calibrated bar.
    scout_bar: float = float('inf'),
    block: int = 1000,
    check_from: float = 0.4,       # start certificate checks after this fraction
    polish_epochs: int = 300,
    batch_size: int = 64,
    lambda_jac: float = 0.0,
    lambda_diff: float = 0.0,
    diff_hinge: float = 0.0,
    diff_warmup: float = 0.0,
    verbose: bool = False,
) -> Dict:
    """
    Scout -> main blocks with early stop on certificate -> polish.
    Returns dict with 'hopeless', 'epochs_used', 'cert' (last certificate).

    diff_hinge / diff_warmup are passed to fast_fit (see there). The scout
    phase uses no differential term (warmup deferred to the main blocks),
    so reconstruction has a chance to settle first.
    """
    dkw = dict(diff_hinge=diff_hinge)
    # ---- scout (reconstruction + jac only; differential term warms up later)
    fast_fit(system, scout_epochs, batch_size, lambda_jac, 0.0,
             lr_start=1e-3, lr_end=1e-3, lr_schedule=False, **dkw,
             verbose=verbose)
    epochs_used = scout_epochs
    eps_now = sup_eps()
    if eps_now > scout_bar:
        if verbose:
            print(f"    scout: sup-eps {eps_now:.3f} > {scout_bar} — abandon")
        return {'hopeless': True, 'epochs_used': epochs_used, 'cert': None}

    # ---- main blocks with cosine decay over the remaining budget ----
    remaining = max(total_epochs - scout_epochs, 0)
    n_blocks = max(1, int(np.ceil(remaining / block)))
    cert = None
    trajectory = [{'epochs': epochs_used, 'eps': float(eps_now)}]
    for b in range(n_blocks):
        e = min(block, remaining - b * block)
        if e <= 0:
            break
        # Constant 1e-3 throughout the main blocks: trajectory data showed
        # the cosine decay throttled convergence (runs still descending at
        # budget end), while the baseline's constant 1e-3 converged further.
        # Low-LR refinement happens only in the polish phase.
        # warm up lambda_diff over the first main block only
        wu = diff_warmup if b == 0 else 0.0
        fast_fit(system, e, batch_size, lambda_jac, lambda_diff,
                 lr_start=1e-3, lr_end=1e-3, lr_schedule=False,
                 diff_hinge=diff_hinge, diff_warmup=wu, verbose=verbose)
        epochs_used += e
        if (scout_epochs + (b + 1) * block) >= check_from * total_epochs:
            cert = certificate()
            trajectory.append({'epochs': epochs_used, 'eps': float(cert[2]),
                               'n_eta_outliers': int(cert[3]),
                               'delta': float(cert[4])})
            if verbose:
                print(f"    block {b}: eps={cert[2]:.3f} outliers={cert[3]} "
                      f"delta={cert[4]:.4f} cert={cert[1]}")
            if cert[1]:
                break

    # ---- polish ----
    if polish_epochs > 0:
        fast_fit(system, polish_epochs, batch_size, lambda_jac, lambda_diff,
                 lr_start=1e-4, lr_end=5e-5, diff_hinge=diff_hinge,
                 verbose=verbose)
        epochs_used += polish_epochs
        cert = certificate()

    return {'hopeless': False, 'epochs_used': epochs_used, 'cert': cert,
            'trajectory': trajectory}
