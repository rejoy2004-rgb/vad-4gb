"""Run the pipeline over the private evaluation set and write a submission.

The evaluation pack has a different shape to the practice pack: levels come
from the L1/L2/L3 directory a video sits in rather than from a manifest, and
there is no ground truth, so nothing here can be scored locally. That makes
validation the only safety net - every field rule is checked before the file
is handed over.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .config import CACHE, RUNS
from .infer import (Explainer, load_clip_head, load_head, load_pairwise,
                    load_params, predict_video)

EVAL = Path("D:/AHC/eval")


def eval_videos(root: Path = EVAL):
    """-> [(video_id, level, path)] across L1/L2/L3."""
    out = []
    for lvl in (1, 2, 3):
        d = root / f"L{lvl}"
        csv_path = d / "videos.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.append((r["video_id"], lvl, str(d / r["filename"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(RUNS / "params_fa.json"))
    ap.add_argument("--out", default=str(RUNS / "eval_submission.json"))
    ap.add_argument("--root", default=str(EVAL))
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--emb-cache", default=None,
                    help="npz of pre-computed eval embeddings; encodes live if absent")
    args = ap.parse_args()

    items = eval_videos(Path(args.root))
    print(f"{len(items)} evaluation videos "
          f"(L1 {sum(1 for i in items if i[1]==1)}, "
          f"L2 {sum(1 for i in items if i[1]==2)}, "
          f"L3 {sum(1 for i in items if i[1]==3)})")

    members, _, _, device = load_head()
    clip_head = load_clip_head(device=device)
    pairwise = load_pairwise(device=device)
    params = load_params(args.params)
    explainer = Explainer() if args.explain else None

    cf, pf = CACHE / "visual_centroids.npy", CACHE / "text_prototypes.npy"
    protos = np.load(cf) if cf.exists() else (np.load(pf) if pf.exists() else None)
    proto_scale = 12.0 if cf.exists() else 60.0

    cache = np.load(args.emb_cache) if args.emb_cache else None
    encoder = None
    if cache is None:
        from .encoder import Encoder
        encoder = Encoder()

    preds, total_ms, total_sec = [], 0.0, 0.0
    wall0 = time.perf_counter()
    for vid, lvl, path in items:
        cached = None
        if cache is not None and f"{vid}__emb" in cache:
            cached = (cache[f"{vid}__emb"], cache[f"{vid}__ts"])
        pred, st = predict_video(vid, path, lvl, members, None, None, device,
                                 params[lvl], encoder, cached, explainer,
                                 clip_head, protos, proto_scale, pairwise)
        preds.append(pred)
        total_ms += st["total_ms"]
        total_sec += st["duration"]
        n_ev = len(pred["events"])
        print(f"  {vid} L{lvl}  {st['duration']:6.1f}s  "
              f"{st['total_ms']/1000:6.2f}s  {n_ev} event(s)"
              + (f"  [{pred['events'][0]['class_name']}]" if n_ev else "  [normal]"),
              flush=True)

    wall = (time.perf_counter() - wall0) * 1000
    sub = {
        "schema_version": "1.0",
        "submission_id": "ahc-vad-4gb-eval",
        "model_name": "frozen SigLIP + window head (where) + clip head (what)",
        "run_metadata": {
            "total_wall_time_ms": round(wall, 1),
            "max_parallel_videos": 1,
            "hardware": "1x RTX 3050 Laptop 4GB",
        },
        "predictions": preds,
    }
    json.dump(sub, open(args.out, "w"), indent=1)
    n_anom = sum(1 for p in preds if p["events"])
    print(f"\nwrote {args.out}")
    print(f"  {n_anom} of {len(preds)} videos flagged anomalous")
    print(f"  {total_sec:.0f}s of video in {total_ms/1000:.1f}s "
          f"-> {total_sec*1000/max(total_ms,1e-9):.1f}x real time")


if __name__ == "__main__":
    main()
