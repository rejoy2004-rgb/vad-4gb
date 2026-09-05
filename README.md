# AHC Visual Intelligence Hackathon — near-real-time video anomaly detection

Detects the 11 anomaly classes in drone / CCTV / dashcam footage, in real time,
on a **4 GB laptop GPU** (RTX 3050 Laptop).

| | D1 (25 marks) | D2 (35 marks) | D3 (40 marks) | Total |
| --- | --- | --- | --- | --- |
| Correct | 63.7% | 70.3% | 50.4% | |
| Marks | 15.9 | 24.6 | 20.2 | **60.7 / 100** |

Arena-scored, public test set. An oracle search over ~15,000 decoder
configurations, taking each video at its own best setting, tops out near 61 -
the decoder is at its ceiling and the encoder is the binding constraint.

3,391 s of video processed in 167 s end to end — **20.3x real time**, decode
included — at 1.03 GB peak VRAM. For scale, predicting `normal` everywhere
scores 0.167 and calling every clip anomalous scores 0.293.

Architecture write-up: <https://claude.ai/code/artifact/dd560d94-f263-488f-a9e5-2e1cfe1ab9c4>

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
4. **Two heads, two questions** — a *window head* (3-layer MLP over window
   features) answers **where** something happens; a *clip head*, trained on whole
   clips with random temporal crops, answers **what** it is. The clip head both
   decides Level 1 and classifies each span the window head flags. Each trains in
   under a minute, which is what makes same-day iteration possible.
5. **Decoder** — hysteresis segmentation over the anomaly score, scored
   **relative to each video's own baseline** rather than a global threshold, then
   merge, minimum duration, and an absolute silence gate. Tuned against a local
   re-implementation of the arena metric.

The relative-scoring step is the single biggest win (L2 +0.12, L3 +0.10). The
head is trained on short clips where the event fills the clip, so on long footage
it reports a high, roughly constant pedestal — baselines ranged 0.55 to 0.88
across videos, and no absolute threshold transfers across that. What localises an
event is the rise above the video's own median. The absolute gate is kept in
front of it so genuinely normal videos stay silent, since a false alarm there
scores zero.

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
| `ahc/train_head.py` | window classifier ("where"), split by video to avoid leakage |
| `ahc/train_clip_head.py` | clip classifier ("what"), temporal-crop augmented, seed ensemble |
| `ahc/decode_events.py` | probabilities → events |
| `ahc/tune.py` | grid search over the decoder |
| `ahc/infer.py` | end-to-end run, writes submission JSON |
| `ahc/score.py` | local re-implementation of the arena metric |
| `ahc/validate.py` | checks a submission against the arena's field rules before upload |
| `make_deck.py` | builds the 2-slide submission deck and its charts |

## Running it

```bash
python -m ahc.extract --split both --workers 6      # once, populates cache/
python -m ahc.build_desc_bank
python -m ahc.train_head --epochs 30                # window head: where
python -m ahc.train_clip_head --epochs 40 --seeds 3 # clip head: what
python -m ahc.tune                                  # -> runs/params.json
python -m ahc.infer --params runs/params.json --explain --live --out runs/submission.json
python -m ahc.validate runs/submission.json         # before uploading
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
- Class confusion, not localisation, is the dominant remaining error. On T025 the
  intervals land almost exactly on the six true accidents and still score 0.300,
  because both heads independently call the footage `wrong_way_driving`. The
  Level-1 losses are similarly adjacent pairs: smoke/fire, fighting/loitering.
- Tried and dropped: larger SigLIP variants (segfault on load at 4 GB), zero-shot
  prompt fusion (0.292 alone, no reliable lift), one shared decoder for all tiers.
