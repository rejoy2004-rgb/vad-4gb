"""Diagnose the anomaly signal itself, independent of the decoder.

Ranking AUC per video answers the question the score cannot: does the anomaly
score put event windows above non-event windows at all? If AUC is near 0.5 no
threshold can help, and tuning the decoder is wasted effort - which is exactly
what we found for T032 and T034.

Cheap enough to use for model selection, where a full decoder tune is not.
"""
from __future__ import annotations

import argparse

import numpy as np

from .config import CACHE
from .features import video_windows
from .score import load_ground_truth


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC. NaN when a video is all-positive or all-negative."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def window_labels(spans, events):
    lab = np.zeros(len(spans), bool)
    for g in events:
        if g["start"] is None:
            continue
        for i, (a, b) in enumerate(spans):
            if b >= g["start"] and a <= g["end"]:
                lab[i] = True
    return lab


def report(cache_dir=None, head=None, quiet=False):
    """-> dict of per-video AUC plus saturation stats, for levels 2 and 3."""
    from .infer import load_head, window_probs

    cache = np.load((cache_dir or CACHE) / "test_emb.npz")
    members, _, _, device = load_head(head)
    gt, levels = load_ground_truth()

    rows, aucs = [], []
    for vid in sorted(v for v in gt if levels[v] > 1):
        key = f"{vid}__emb"
        if key not in cache:
            continue
        E = np.asarray(cache[key], np.float32)
        ts = cache[f"{vid}__ts"]
        if len(E) == 0:
            continue
        feats, spans = video_windows(E, ts)
        P = window_probs(members, None, None, device, feats)
        anom = 1.0 - P[:, 0]
        lab = window_labels(spans, gt[vid])
        a = auc(anom, lab)
        rows.append((vid, levels[vid], len(spans), float(anom.min()),
                     float(anom.max()), a))
        if not np.isnan(a):
            aucs.append(a)

    if not quiet:
        print(f"{'vid':6s} {'L':>2s} {'win':>4s} {'min':>6s} {'max':>6s} {'AUC':>6s}")
        for vid, lvl, n, lo, hi, a in rows:
            flag = "  <- saturated" if lo > 0.9 else ""
            print(f"{vid:6s} {lvl:2d} {n:4d} {lo:6.3f} {hi:6.3f} {a:6.3f}{flag}")
        print(f"mean AUC {np.mean(aucs):.4f} over {len(aucs)} videos")
        sat = sum(1 for r in rows if r[3] > 0.9)
        print(f"saturated videos (min anomaly score > 0.9): {sat} of {len(rows)}")
    return {"rows": rows, "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "saturated": sum(1 for r in rows if r[3] > 0.9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None)
    args = ap.parse_args()
    report(head=args.head)


if __name__ == "__main__":
    main()
