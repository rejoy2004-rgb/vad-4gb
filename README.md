# Video anomaly detection on a 4 GB laptop GPU

AHC Visual Intelligence Hackathon. Detects the 11 anomaly classes in drone,
CCTV and dashcam footage in real time, on an RTX 3050 Laptop with **4 GB of
VRAM**.

## Results

**Private evaluation set** (28 videos: D1 20, D2 4, D3 4):

| | D1 (25 marks) | D2 (35 marks) | D3 (40 marks) | Total |
| --- | --- | --- | --- | --- |
| Correct | 56.2% | 51.0% | 57.0% | |
| Marks | 14.1 | 17.9 | 22.8 | **54.8 / 100** |

**Speed**: 2,840 s of evaluation video processed in 136 s — **20.9x real time**
end to end, decoding included — at 1.03 GB peak VRAM. On the practice pack,
3,391 s in 167 s (20.3x).

The practice pack (34 videos) scored 60.7/100 with a decoder tuned against it.
That same decoder scored **39.9** on the private set. The gap, and closing most
of it, is the most useful thing in this repo — see *What actually moved the
score*.

Architecture write-up: [`architecture.html`](architecture.html) — pipeline
diagram, per-frame vs per-window vs per-event cost, and the timeline
comparisons behind the findings below.

## Approach

4 GB rules out fine-tuning a 2B-parameter VLM locally, so instead of training a
VLM we **froze one and trained only what sits on top of it**:

```
video -> sample 2 fps -> SigLIP-2 base -> 4 s windows -> two MLP heads -> decoder -> JSON
         downsize on     FROZEN, fp16    mean/max/std   window: WHERE    hysteresis
         decode          78 fps, 1.03 GB /drift         clip:   WHAT     + span rules
```

1. **Sampling** — 2 fps. An accident lasts about a second, so this is roughly
   the floor before events are stepped over entirely. Frames are downsized
   during decode: a 10-minute 1080p video at 2 fps is ~9 GB of full-resolution
   frames but only ~180 MB at 224x224.
2. **Encoder** — `google/siglip2-base-patch16-224`, frozen, fp16. Because it is
   frozen, all 3,207 videos are embedded **once** into a cache, so every later
   experiment reads embeddings instead of pixels and a head retrains in under a
   minute. That is what made same-day iteration possible.
3. **Two heads, two questions** — a *window head* answers **where** something
   happens; a *clip head*, trained on whole clips with random temporal crops,
   answers **what** it is. The clip head decides Difficulty 1 outright and
   classifies each span the window head flags. Both are seed ensembles.
4. **Decoder** — hysteresis segmentation over the anomaly score, thresholded
   **relative to each video's own baseline**, then merged into the span shape
   each difficulty tier actually expects.

**Explanations** come from nearest-neighbour retrieval over training
`description_summary` text in the same embedding space — one dot product per
event, no generative model, no extra VRAM.

## What actually moved the score

**Score against each video's own baseline.** Trained on short clips where the
event fills the frame, the head reports a high, roughly constant pedestal on
long footage — baselines ranged 0.55 to 0.88 across videos, so no absolute
threshold transfers. Thresholding on the *rise above each video's own median*
found the events. An absolute gate is kept in front of it so genuinely normal
videos stay silent, since a false alarm there scores zero. *(L2 +0.12, L3 +0.10)*

**Use each head for the question it was trained on.** Letting the clip head
classify each detected span, instead of pooling window votes, fixed whole
videos at once. *(L3 0.296 -> 0.414)*

**Annotation granularity beat every tuned threshold.** Overlap is scored
against the **whole** ground-truth span. Several short windows inside one long
annotated event each fall under the 0.5 IoU gate and score nothing. Collapsing
a video to a single span lifted D3 from 8.0 to 22.8 marks and the total from
39.9 to 54.8 — after ~15,000 tuned decoder configurations had barely moved it.

The span shape is per tier, because the tiers differ:

| Tier | Description | Span rule |
| --- | --- | --- |
| D1 | short clips, timing not scored | one label, `null` timestamps |
| D2 | event is a portion, ordinary activity either side, 15 s tolerance | one interval on the strongest detection |
| D3 | long footage, anomaly recurs throughout | one span covering the video |

Applying the D3 rule to D2 *cost* marks (17.9 -> 14.0), which is why it is per
tier rather than global.

## Calibrating the metric

The arena returned 62.0% on D1 where an even anomaly/class split predicts
75.0%. Refitting against that showed **class accuracy carries roughly four
times the weight of anomaly accuracy** there — we had been tuning a detector
that was already right 23 times out of 24 while ignoring the class errors that
actually cost marks. After recalibration `ahc/score.py` reproduced the arena
within half a mark on every tier of the practice pack.

## Layout

| file | role |
| --- | --- |
| `ahc/config.py` | paths, label set, prompt bank, env overrides |
| `ahc/labels.py` | training annotations (torch-free) |
| `ahc/video.py` | frame sampling, downsize-on-decode, optional tiling |
| `ahc/encoder.py` | frozen SigLIP wrapper, fp16, batched, optional tiled views |
| `ahc/extract.py` | one-pass embedding cache, checkpointed every 250 videos |
| `ahc/extract_eval.py` | same for the private evaluation pack |
| `ahc/features.py` | window construction and summary statistics |
| `ahc/centroids.py` | class prototypes from real footage (61% vs 23% for prompts) |
| `ahc/train_head.py` | window classifier — *where* |
| `ahc/train_clip_head.py` | clip classifier — *what*, temporal-crop augmented |
| `ahc/pairwise.py` | binary discriminators for measured confusable pairs |
| `ahc/decode_events.py` | probabilities to events, span rules |
| `ahc/tune.py` | grid search over the decoder |
| `ahc/diagnose.py` | per-video ranking AUC and saturation |
| `ahc/infer.py` | end-to-end run on the practice pack |
| `ahc/run_eval.py` | end-to-end run on the private evaluation pack |
| `ahc/score.py` | local re-implementation of the arena metric |
| `ahc/validate.py` | checks a submission against every field rule |
| `ahc/explain_vlm.py` | optional SmolVLM explanations, triggered spans only |
| `make_deck.py` | builds the 2-slide submission deck and its charts |
| `run_experiments.sh` | unattended experiment queue |

## Running it

```bash
python -m ahc.extract --split both --workers 6      # once, populates cache/
python -m ahc.centroids                             # class prototypes
python -m ahc.build_desc_bank
python -m ahc.train_head --epochs 30 --seeds 5      # window head: where
python -m ahc.train_clip_head --epochs 40 --seeds 3 # clip head: what
python -m ahc.pairwise --epochs 60                  # confusable-pair heads
python -m ahc.tune                                  # -> runs/params.json

# practice pack
python -m ahc.infer --params runs/params_fa.json --explain --live \
    --manifest manifest.json --out runs/submission.json
python -m ahc.score runs/submission.json --per-video

# private evaluation pack
python -m ahc.extract_eval
python -m ahc.run_eval --params runs/params_best.json --explain \
    --out runs/eval_submission.json
python -m ahc.validate runs/eval_submission.json --manifest eval/manifest_eval.json
```

`--live` decodes and encodes for real and reports true wall-clock timings in
`runtime_metadata`. Drop it to reuse the cache while tuning.

### Variants

Environment variables switch configuration without disturbing a working setup:

```bash
AHC_ENCODER=google/siglip-large-patch16-256 AHC_CACHE=cache_large \
AHC_RUNS=runs_large python -m ahc.extract --split both
AHC_TILES=2 AHC_CACHE=cache_tiled python -m ahc.extract --split both
AHC_FPS=4  AHC_CACHE=cache_4fps  python -m ahc.extract --split test
```

`AHC_TILES=2` encodes the full frame plus a 2x2 grid of crops and pools across
views with max and mean. Max-pooling is the point: a tile holding only the
debris scores high even when the whole frame does not.

## Measured and rejected

Kept here because the negative results were expensive to obtain:

- **A larger encoder.** `siglip-large-patch16-256` raised ranking AUC (0.73 vs
  0.68) but *lost* marks: L2 collapsed 0.677 -> 0.519, estimate 49.1 against
  base 59.0. Higher AUC did not convert.
- **Contrast-to-baseline features.** Held-out training accuracy rose to 0.869,
  but ranking AUC on long test videos collapsed (T026 0.663 -> 0.268, T034
  0.857 -> 0.532). In a short training clip the window *is* the whole video, so
  the cue does not exist at test time. Kept in `features.py`, off by default.
- **In-scene hard negatives.** Monotonically negative: weight 1.0 -> AUC
  0.7334, 3.0 -> 0.7300, 10.0 -> 0.6922.
- **Logit adjustment for rare classes**, **test-time augmentation**, and
  **zero-shot prompt fusion** (23% alone, no reliable lift): no gain.
- **Emitting more D3 intervals.** The opposite of what the rules prescribe;
  precision fell 32% -> 23% for nothing.

## Known limitations

- **Class accuracy is the binding constraint.** `stalled_or_broken_down_vehicle`,
  `road_spill_or_debris`, `vehicle_blocking_traffic` and `wrong_way_driving` are
  all at 0% found on the evaluation set. No timing change fixes a wrong label.
- **Saturation on long footage.** On several videos `p(normal)` never exceeds
  0.02 for the entire clip, so the anomaly score carries no temporal
  information and localisation is luck. The root cause is training on short
  clips where the event fills the frame; the fix is training on long footage
  with in-scene negatives, which is a rebuild rather than a tweak.
- **Thresholds tuned on few videos do not transfer.** 60.7 on the practice pack
  became 39.9 on the private set.
- The local scorer matches the published structure of the metric; the exact
  within-tier weighting is not published, so it ranks configurations rather
  than predicting the leaderboard.
