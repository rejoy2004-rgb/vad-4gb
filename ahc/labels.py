"""Training annotations. Deliberately torch-free so label-only tooling can run
alongside a long extraction without loading a second copy of torch."""
from __future__ import annotations

import csv

from .config import TRAIN


def load_train_intervals() -> dict[str, list[tuple[float, float, str]]]:
    """video_id -> list of (start, end, class). Normal videos map to []."""
    out: dict[str, list] = {}
    for cls_dir in sorted(p for p in TRAIN.iterdir() if p.is_dir()):
        gt = cls_dir / "ground_truth.csv"
        if not gt.exists():
            continue
        with open(gt, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vid = row["video_id"]
                out.setdefault(vid, [])
                if row["class_name"] == "normal":
                    continue
                s, e = row["start_time_sec"], row["end_time_sec"]
                if s and e:
                    out[vid].append((float(s), float(e), row["class_name"]))
                else:
                    out[vid].append((0.0, 1e9, row["class_name"]))
    return out
