"""Clip-level classifier for Level 1.

Level 1 asks for one label per clip, but the window head answers a different
question ("what is happening in these 4 seconds") and its votes then have to be
pooled. Training directly on the clip-level task removes that mismatch.

Training clips are short and labelled at clip level, which is exactly this
task. To avoid overfitting 3,173 samples with a 3,075-dim input, each clip is
sampled several times as a random temporal crop — cheap augmentation that also
teaches the model to survive seeing only part of an event.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import CACHE, CLASS_TO_IDX, NUM_CLASSES, RUNS
from .features import window_feature
from .train_head import Head, load_train_intervals, train


def clip_feature(emb: np.ndarray) -> np.ndarray:
    """Same summary statistics as a window, taken over the whole clip."""
    return window_feature(emb)


def build_clip_dataset(crops: int = 4, seed: int = 0):
    data = np.load(CACHE / "train_emb.npz")
    intervals = load_train_intervals()
    vids = sorted({k.split("__")[0] for k in data.files})
    rng = np.random.default_rng(seed)

    X, y, groups = [], [], []
    for vid in vids:
        emb = np.asarray(data[f"{vid}__emb"], dtype=np.float32)
        if len(emb) == 0:
            continue
        evs = intervals.get(vid, [])
        label = CLASS_TO_IDX[evs[0][2]] if evs else 0

        X.append(clip_feature(emb))
        y.append(label)
        groups.append(vid)

        # random temporal crops covering 50-100% of the clip
        for _ in range(crops):
            if len(emb) < 4:
                break
            frac = rng.uniform(0.5, 1.0)
            n = max(2, int(len(emb) * frac))
            s = rng.integers(0, len(emb) - n + 1)
            X.append(clip_feature(emb[s : s + n]))
            y.append(label)
            groups.append(vid)          # same group, so crops cannot leak
    return np.stack(X), np.asarray(y), np.asarray(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3,
                    help="ensemble size; each seed gets its own video-level split")
    ap.add_argument("--out", default=str(RUNS / "clip_head.pt"))
    args = ap.parse_args()

    print("building clips...")
    X, y, g = build_clip_dataset(crops=args.crops)
    print(f"  {len(X)} samples from {len(np.unique(g))} clips, dim {X.shape[1]}")
    print("  label counts:", np.bincount(y, minlength=NUM_CLASSES).tolist())

    members, accs = [], []
    for s in range(args.seeds):
        print(f"seed {s}:")
        model, mu, sd, acc = train(X, y, g, epochs=args.epochs, seed=s)
        members.append({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "mu": mu, "sd": sd, "val_acc": acc})
        accs.append(acc)
        print(f"  seed {s} val clip acc {acc:.4f}")

    torch.save({"members": members, "in_dim": X.shape[1],
                # kept so a single-model loader still works
                "state_dict": members[0]["state_dict"],
                "mu": members[0]["mu"], "sd": members[0]["sd"],
                "val_acc": float(np.mean(accs))}, args.out)
    print(f"saved {args.out}  ({len(members)} members, mean val acc {np.mean(accs):.4f})")


if __name__ == "__main__":
    main()
