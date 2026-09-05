# AHC Visual Intelligence Hackathon — near-real-time video anomaly detection

Detects the 11 anomaly classes in drone / CCTV / dashcam footage, in real time,
on a **4 GB laptop GPU** (RTX 3050 Laptop).

## Approach

The hard constraint here was 4 GB of VRAM. That rules out fine-tuning a 2B-class
VLM locally, so instead of training a VLM we **freeze one and train only what
sits on top of it**:

```
video ──► frame sampling ──► SigLIP-2 image encoder ──► window features ──► MLP head ──► event decoder ──► JSON
          (2 fps, OpenCV)     (frozen, fp16, ~1 GB)     (mean/max/std/drift)  (12-way)    (hysteresis)
```

1. **Sampling** — 2 fps. An accident lasts about a second, so this is roughly the
   floor before events are stepped over entirely.
2. **Encoder** — `google/siglip2-base-patch16-224`, frozen, fp16. Open-vocabulary
   semantics without an open-vocabulary price tag: **78 fps, 1.03 GB VRAM**.
   Because it is frozen, every video is encoded **once** into a cache and every
   later experiment reads embeddings instead of pixels.
3. **Window features** — one decision per 4 s window (2 s hop). Each window is
   summarised by mean / max / std / adjacent-frame drift plus scalar motion
   terms. The temporal half of that is what distinguishes congestion from a busy
   road, and loitering from someone walking past — a single frame cannot.
4. **Head** — a 3-layer MLP over window features, class-balanced. Trains in
   under a minute, which is what makes same-day iteration possible.
5. **Decoder** — hysteresis segmentation over the anomaly score, then merge,
   minimum duration, and a global silence gate. Tuned against a local
   re-implementation of the arena metric.

**Explanations** are produced by nearest-neighbour retrieval over training
`description_summary` text in the same embedding space — one dot product per
event, no generative model, no extra VRAM.

## Why not a VLM at runtime

A small VLM was the obvious starting point and is the wrong tool for the
always-on stage: it costs orders of magnitude more per frame for a decision that
is, in the end, a 12-way classification. The frozen-encoder split keeps the
open-vocabulary semantics of a VLM in the part that runs on every frame, and
puts the learned, task-specific part in a head small enough to retrain in
seconds. The design leaves room for a heavier verification stage on triggered
windows only; the retrieval explainer occupies that slot today.

## Layout

| file | role |
| --- | --- |
| `ahc/config.py` | paths, label set, zero-shot prompt bank |
| `ahc/video.py` | frame sampling (sequential grab/retrieve, no seeking) |
| `ahc/encoder.py` | frozen SigLIP wrapper, fp16, batched |
| `ahc/extract.py` | one-pass embedding cache (threaded decode → GPU) |
| `ahc/features.py` | window construction and summary statistics |
| `ahc/train_head.py` | window classifier, split by video to avoid leakage |
| `ahc/decode_events.py` | probabilities → events |
| `ahc/tune.py` | grid search over the decoder |
| `ahc/infer.py` | end-to-end run, writes submission JSON |
| `ahc/score.py` | local re-implementation of the arena metric |

## Running it

```bash
python -m ahc.extract --split both --workers 6   # once, ~cache/
python -m ahc.build_desc_bank
python -m ahc.train_head --epochs 30
python -m ahc.tune
python -m ahc.infer --params runs/params.json --explain --live --out runs/submission.json
python -m ahc.score runs/submission.json --per-video
```

`--live` decodes and encodes for real and reports true wall-clock timings in
`runtime_metadata`. Drop it to reuse the cache while tuning.

## Known limitations

- The decoder is tuned on 6 Level-2 and 4 Level-3 public videos. That is a thin
  basis for threshold selection; the search is coarse and ties break toward the
  least aggressive setting for that reason.
- The local scorer matches the published structure of the metric, but the arena's
  exact weighting of the Level-2/3 mix is not published. It is reliable for
  ranking configurations, not for predicting the leaderboard number.
- Classes are decided per window; a class that only makes sense over a long
  horizon (slow-building congestion) is handled by merging, not by a model that
  sees the whole horizon at once.
