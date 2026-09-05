"""Turn per-window class probabilities into submitted events.

This is where most of the score actually lives. The metric punishes fragments
(only the best-overlapping prediction can match) and punishes any prediction at
all on a normal video, so the decoder is deliberately conservative: hysteresis
to avoid chopping one event into three, a merge pass, a minimum duration, and a
global gate that stays silent unless some window is confidently anomalous.
"""
from __future__ import annotations

import numpy as np

from .config import CLASSES


class DecodeParams:
    def __init__(self, hi=0.55, lo=0.35, gate=0.55, merge_gap=3.0, min_dur=1.5,
                 pad=0.5, smooth=3, l1_bias=0.0, topk_frac=0.25,
                 relative=True, q_base=0.5, max_events=0):
        self.hi = hi              # start a segment
        self.lo = lo              # extend a segment
        self.gate = gate          # nothing at all below this peak (absolute)
        self.merge_gap = merge_gap
        self.min_dur = min_dur
        self.pad = pad            # widen each side; windows lag the true onset
        self.smooth = smooth
        self.l1_bias = l1_bias    # added to p(normal) at level 1
        self.topk_frac = topk_frac
        self.relative = relative  # threshold against the video's own baseline
        self.q_base = q_base      # quantile taken as that baseline
        self.max_events = max_events  # keep only the N strongest (0 = all)

    def as_dict(self):
        return dict(vars(self))

    def __repr__(self):
        return "DecodeParams(" + ", ".join(f"{k}={v}" for k, v in vars(self).items()) + ")"


def smooth_probs(P: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(P) < 2:
        return P
    k = min(k, len(P))
    kernel = np.ones(k, dtype=np.float32) / k
    out = np.empty_like(P)
    for c in range(P.shape[1]):
        out[:, c] = np.convolve(P[:, c], kernel, mode="same")
    return out / out.sum(axis=1, keepdims=True)


def topk_mean(x: np.ndarray, frac: float) -> float:
    k = max(1, int(round(len(x) * frac)))
    return float(np.sort(x)[-k:].mean())


def decode_level1(P: np.ndarray, p: DecodeParams) -> list[dict]:
    """One label for the whole clip, or [] for normal."""
    if len(P) == 0:
        return []
    P = smooth_probs(P, p.smooth)
    agg = np.array([topk_mean(P[:, c], p.topk_frac) for c in range(P.shape[1])])
    agg[0] += p.l1_bias
    idx = int(agg.argmax())
    if idx == 0:
        return []
    return [{"class_name": CLASSES[idx], "start_time_sec": None, "end_time_sec": None}]


def decode_temporal(P: np.ndarray, spans: np.ndarray, p: DecodeParams,
                    duration: float | None = None, classify=None) -> list[dict]:
    """Hysteresis segmentation over the anomaly score."""
    if len(P) == 0:
        return []
    P = smooth_probs(P, p.smooth)
    raw = 1.0 - P[:, 0]

    # Absolute gate first: this is the only thing standing between us and a
    # false alarm on a genuinely normal video, and those score zero.
    if raw.max() < p.gate:
        return []

    if p.relative:
        # The head was trained on short clips where the event fills the frame,
        # so on long footage it reports a high, roughly constant pedestal for
        # anomaly-ish scenery. What localises an event is how far the score
        # rises above that video's own baseline, not its absolute value.
        base = float(np.quantile(raw, p.q_base))
        spread = max(float(raw.max()) - base, 1e-6)
        anom = np.clip((raw - base) / spread, 0.0, 1.0)
    else:
        anom = raw

    segs, i, n = [], 0, len(anom)
    while i < n:
        if anom[i] >= p.hi:
            a = i
            while a > 0 and anom[a - 1] >= p.lo:
                a -= 1
            b = i
            while b + 1 < n and anom[b + 1] >= p.lo:
                b += 1
            segs.append((a, b))
            i = b + 1
        else:
            i += 1
    if not segs:
        return []

    # collapse overlaps produced by the backward walk
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    events = []
    for a, b in merged:
        if classify is not None:
            # The window head is good at *where* something happens; the
            # clip head is trained on *what* a span contains. Use each for
            # the question it was trained on.
            cls_idx = classify(float(spans[a][0]), float(spans[b][1]))
        if classify is None or cls_idx is None or cls_idx == 0:
            cls_scores = P[a : b + 1, 1:].sum(axis=0)
            cls_idx = int(cls_scores.argmax()) + 1
        t0 = float(spans[a][0]) - p.pad
        t1 = float(spans[b][1]) + p.pad
        events.append({"class_name": CLASSES[cls_idx], "start_time_sec": t0,
                       "end_time_sec": t1, "_score": float(1.0 - P[a : b + 1, 0].min())})

    # merge neighbours of the same class: fragments cannot both match
    events.sort(key=lambda e: e["start_time_sec"])
    out = [events[0]]
    for e in events[1:]:
        prev = out[-1]
        if (e["class_name"] == prev["class_name"]
                and e["start_time_sec"] - prev["end_time_sec"] <= p.merge_gap):
            prev["end_time_sec"] = max(prev["end_time_sec"], e["end_time_sec"])
            prev["_score"] = max(prev["_score"], e["_score"])
        else:
            out.append(e)

    final = []
    for e in out:
        s = max(0.0, e["start_time_sec"])
        # end must stay inside the duration or the submission is rejected
        t = e["end_time_sec"] if duration is None else min(e["end_time_sec"], duration)
        if t - s >= p.min_dur:
            e["start_time_sec"], e["end_time_sec"] = round(s, 2), round(t, 2)
            final.append(e)

    # Unmatched predictions count against precision, so when the decoder is
    # unsure it is better to keep the strongest few than to submit everything.
    if p.max_events and len(final) > p.max_events:
        final = sorted(final, key=lambda e: -e["_score"])[: p.max_events]
        final.sort(key=lambda e: e["start_time_sec"])
    return final
