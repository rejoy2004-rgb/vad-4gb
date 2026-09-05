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


def feature_dim(emb_dim: int, contrast: bool = False) -> int:
    return emb_dim * 4 + 3 + (emb_dim + 2 if contrast else 0)


def video_baseline(emb: np.ndarray) -> np.ndarray:
    """The video's own modal appearance: a per-dimension median over frames."""
    if len(emb) == 0:
        return np.zeros(emb.shape[1] if emb.ndim == 2 else 768, np.float32)
    m = np.median(emb, axis=0)
    n = np.linalg.norm(m)
    return (m / n if n > 0 else m).astype(np.float32)


def contrast_feature(win: np.ndarray, base: np.ndarray) -> np.ndarray:
    """How far this window departs from its own video's baseline.

    MEASURED AND REJECTED (kept for the record, off by default): held-out
    training accuracy rose to 0.869, but ranking AUC on the long test videos
    collapsed - T026 0.663 -> 0.268, T027 0.742 -> 0.245, T034 0.857 -> 0.532.
    In a short training clip the window IS the whole video, so contrast is
    near-zero there and the head learned to lean on a cue that does not exist
    at test time.
    """
    m = win.mean(axis=0)
    n = np.linalg.norm(m)
    mn = m / n if n > 0 else m
    return np.concatenate([
        (mn - base),
        np.array([1.0 - float(mn @ base), float(np.linalg.norm(mn - base))],
                 dtype=np.float32),
    ]).astype(np.float32)


def video_windows(emb: np.ndarray, ts: np.ndarray, contrast: bool = False):
    """-> (features (N,F), spans (N,2) in seconds)."""
    feats, spans = [], []
    base = video_baseline(emb) if contrast else None
    for a, b in window_bounds(len(emb)):
        f = window_feature(emb[a:b])
        if contrast:
            f = np.concatenate([f, contrast_feature(emb[a:b], base)])
        feats.append(f)
        t0 = float(ts[a]) if len(ts) else 0.0
        t1 = float(ts[b - 1]) if len(ts) else 0.0
        spans.append((t0, t1))
    if not feats:
        d = emb.shape[1] if emb.ndim == 2 else 768
        return (np.zeros((0, feature_dim(d, contrast)), np.float32),
                np.zeros((0, 2), np.float32))
    return np.stack(feats), np.asarray(spans, dtype=np.float32)
