"""
Saving and reloading a trained atlas.

Every diagnostic added after the fact -- the explicit threshold Theta, the
latent gauge, volume regularity, sign constancy -- has needed the trained
networks, not just the recorded scalars, and so has cost a full retrain.  A
complete set of trained atlases for every experiment in the paper is under
2 MB, so there is no reason to discard them.

Format: one directory per atlas, containing

    meta.json      latent_dim, hidden_dims, input_dim, n_charts, provenance
    points.npy     the point cloud the atlas was trained on
    assignments/   chart_<i>.npy, the index array of each chart
    weights.npz    every layer weight, keyed chart_<i>_{enc,dec}_<k>

Weights are stored as raw arrays rather than in a Keras format so that
reloading does not depend on the TensorFlow version.

Usage:
    from atlasae.persistence import save_atlas, load_atlas
    save_atlas(system, "runs/S2_seed42", note="paper settings, 4000 epochs")
    system = load_atlas("runs/S2_seed42")
"""

import json
import os
from datetime import datetime

import numpy as np

__all__ = ["save_atlas", "load_atlas"]


def save_atlas(system, path, note="", extra=None):
    """Write a trained atlas to `path` (created if absent)."""
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "assignments"), exist_ok=True)

    np.save(os.path.join(path, "points.npy"), np.asarray(system.data))
    for i, idx in enumerate(system.subset_assignments):
        np.save(os.path.join(path, "assignments", f"chart_{i}.npy"),
                np.asarray(idx))

    arrays = {}
    for i, ae in enumerate(system.autoencoders):
        for tag, net in (("enc", ae.encoder), ("dec", ae.decoder)):
            for k, w in enumerate(net.get_weights()):
                arrays[f"chart_{i}_{tag}_{k}"] = w
    np.savez_compressed(os.path.join(path, "weights.npz"), **arrays)

    ae0 = system.autoencoders[0]
    hidden = [l.units for l in ae0.encoder.layers[:-1]]
    meta = dict(
        n_charts=int(system.n_charts),
        latent_dim=int(ae0.encoder.layers[-1].units),
        hidden_dims=hidden,
        input_dim=int(system.data.shape[1]),
        n_points=int(system.data.shape[0]),
        saved=datetime.now().isoformat(timespec="seconds"),
        note=note,
    )
    if extra:
        meta.update(extra)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def load_atlas(path):
    """Reconstruct the atlas saved at `path`.  Returns an AtlasAutoencoder."""
    from atlasae.atlasautoencoder import AtlasAutoencoder

    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    points = np.load(os.path.join(path, "points.npy"))
    assignments = [np.load(os.path.join(path, "assignments", f"chart_{i}.npy"))
                   for i in range(meta["n_charts"])]

    system = AtlasAutoencoder(data=points, n_charts=meta["n_charts"],
                              subset_assignments=assignments,
                              latent_dim=meta["latent_dim"],
                              hidden_dims=meta["hidden_dims"])

    # build the layers before assigning weights
    import tensorflow as tf
    probe = tf.constant(points[:1], dtype=tf.float32)
    for ae in system.autoencoders:
        ae.decode(ae.encode(probe))

    z = np.load(os.path.join(path, "weights.npz"))
    for i, ae in enumerate(system.autoencoders):
        for tag, net in (("enc", ae.encoder), ("dec", ae.decoder)):
            ws, k = [], 0
            while f"chart_{i}_{tag}_{k}" in z:
                ws.append(z[f"chart_{i}_{tag}_{k}"])
                k += 1
            net.set_weights(ws)
    return system
