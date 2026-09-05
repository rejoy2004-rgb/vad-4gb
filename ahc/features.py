"""Window features built on top of cached frame embeddings.

A single frame cannot tell a stalled car from a parked one, so every decision
is made on a short window. Each window is summarised by statistics that carry
both appearance (mean/max) and change over time (std, adjacent-frame drift) —
the temporal half is what separates "congestion" from "busy road", and
"loitering" from "someone walking past".
"""
from __future__ import annotations

import numpy as np

WIN = 8      # frames per window -> 4 s at 2 fps
STRIDE = 4   # 2 s hop


def window_bounds(n_frames: int, win: int = WIN, stride: int = STRIDE):
    """Yield (start_idx, end_idx) covering the clip, always at least one."""
    if n_frames <= 0:
        return
    if n_frames <= win:
        yield 0, n_frames
        return
    i = 0
    while i + win <= n_frames:
        yield i, i + win
        i += stride
    if i < n_frames and (n_frames - i) >= win // 2:
        yield n_frames - win, n_frames


def window_feature(emb: np.ndarray) -> np.ndarray:
    """(W, D) frame embeddings -> a single feature vector."""
    e = emb.astype(np.float32)
    mean = e.mean(axis=0)
    mx = e.max(axis=0)
    std = e.std(axis=0)
    if len(e) > 1:
        d = np.abs(np.diff(e, axis=0))
        drift = d.mean(axis=0)
        # scalar summary of how much the scene moves inside the window
        motion = np.array([d.sum(axis=1).mean(), d.sum(axis=1).max(),
                           float(np.linalg.norm(e[-1] - e[0]))], dtype=np.float32)
    else:
        drift = np.zeros_like(mean)
        motion = np.zeros(3, dtype=np.float32)
    return np.concatenate([mean, mx, std, drift, motion]).astype(np.float32)


def feature_dim(emb_dim: int) -> int:
    return emb_dim * 4 + 3


def video_windows(emb: np.ndarray, ts: np.ndarray):
    """-> (features (N,F), spans (N,2) in seconds)."""
    feats, spans = [], []
    for a, b in window_bounds(len(emb)):
        feats.append(window_feature(emb[a:b]))
        t0 = float(ts[a]) if len(ts) else 0.0
        t1 = float(ts[b - 1]) if len(ts) else 0.0
        spans.append((t0, t1))
    if not feats:
        return np.zeros((0, feature_dim(emb.shape[1] if emb.ndim == 2 else 768)),
                        np.float32), np.zeros((0, 2), np.float32)
    return np.stack(feats), np.asarray(spans, dtype=np.float32)
