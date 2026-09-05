"""Class prototypes built from real footage rather than from written prompts.

Hand-written text prompts reach 23.2% on held-out training videos; the mean
embedding of each class's actual training clips reaches 61.2% on the same
split. Drone and CCTV footage sits far enough from SigLIP's web-image
distribution that describing a class in words is much weaker than showing it.

These are cheap: one matrix of class means, and a dot product at inference.
"""
from __future__ import annotations

import argparse

import numpy as np

from .config import CACHE, CLASS_TO_IDX, NUM_CLASSES
from .train_head import load_train_intervals


def video_mean_embeddings(npz_path=None):
    """-> (E (N,D) unit-norm clip means, Y (N,) labels, vids)."""
    data = np.load(npz_path or (CACHE / "train_emb.npz"))
    iv = load_train_intervals()
    vids = [v for v in sorted({k.split("__")[0] for k in data.files})
            if len(data[f"{v}__emb"])]
    E, Y = [], []
    for v in vids:
        e = np.asarray(data[f"{v}__emb"], np.float32)
        ts = data[f"{v}__ts"]
        evs = iv.get(v, [])
        if evs:
            s, t, cname = evs[0]
            sel = (ts >= s) & (ts <= t)
            if sel.sum() >= 2:
                e = e[sel]                 # average over the labelled event only
            label = CLASS_TO_IDX[cname]
        else:
            label = 0
        m = e.mean(0)
        E.append(m / (np.linalg.norm(m) + 1e-9))
        Y.append(label)
    return np.stack(E), np.asarray(Y), vids


def build(out=None, npz_path=None):
    E, Y, _ = video_mean_embeddings(npz_path)
    C = np.zeros((NUM_CLASSES, E.shape[1]), np.float32)
    for c in range(NUM_CLASSES):
        m = Y == c
        if m.sum():
            v = E[m].mean(0)
            C[c] = v / (np.linalg.norm(v) + 1e-9)
    path = out or (CACHE / "visual_centroids.npy")
    np.save(path, C)

    # honest read of their quality: held-out by video, centroids from train only
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Y))
    n = int(len(Y) * 0.2)
    va, tr = idx[:n], idx[n:]
    Ct = np.stack([E[tr][Y[tr] == c].mean(0) if (Y[tr] == c).any()
                   else np.zeros(E.shape[1], np.float32) for c in range(NUM_CLASSES)])
    Ct /= (np.linalg.norm(Ct, axis=1, keepdims=True) + 1e-9)
    acc = ((E[va] @ Ct.T).argmax(1) == Y[va]).mean()
    print(f"visual centroids -> {path}")
    print(f"  held-out accuracy {acc:.4f} over {len(va)} videos "
          f"(hand-written text prompts: 0.232)")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
