"""Local re-implementation of the arena metric.

The brief specifies the structure exactly (L1 = half anomaly accuracy + half
class accuracy; L2/L3 = per-video mix of alert / matched events / timing, with
timing weighted higher at L3) but not the numeric weights of the L2/L3 mix.
The weights below are our reading of it; they are the one part of this file
that may not match the arena to the decimal. Ranking behaviour is what we use
it for, and that is stable under reasonable choices.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

from .config import DATA

IOU_GATE = 0.5
# (alert, match, timing) per level
WEIGHTS = {2: (0.30, 0.40, 0.30), 3: (0.20, 0.35, 0.45)}


def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(path=None):
    """-> (events_by_video, level_by_video). Normal videos map to []."""
    path = path or (DATA / "test" / "ground_truth.csv")
    events, levels = defaultdict(list), {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_id"]
            levels[vid] = int(row["level"])
            events.setdefault(vid, [])
            if row["class_name"] != "normal" and row["is_anomaly"].lower() == "true":
                events[vid].append({
                    "class_name": row["class_name"],
                    "start": float(row["start_time_sec"]) if row["start_time_sec"] else None,
                    "end": float(row["end_time_sec"]) if row["end_time_sec"] else None,
                })
    return dict(events), levels


def _match(preds, gts):
    """Greedy one-to-one matching on IoU, class must agree. -> (n, ious)."""
    pairs = []
    for pi, p in enumerate(preds):
        ps, pe = p.get("start_time_sec"), p.get("end_time_sec")
        if ps is None or pe is None or pe <= ps:
            continue          # an untimestamped event cannot match at L2/L3
        for gi, g in enumerate(gts):
            if p["class_name"] != g["class_name"]:
                continue
            v = iou((ps, pe), (g["start"], g["end"]))
            if v >= IOU_GATE:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        ious.append(v)
    return len(ious), ious


def score_video_temporal(preds, gts, level) -> float:
    if not gts:                       # normal video: any prediction is fatal
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    w_alert, w_match, w_time = WEIGHTS[level]
    n, ious = _match(preds, gts)
    recall = n / len(gts)
    precision = n / len(preds)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    timing = sum(ious) / len(gts) if ious else 0.0
    return w_alert * 1.0 + w_match * f1 + w_time * timing


def score(submission: dict, gt_path=None) -> dict:
    events_gt, levels = load_ground_truth(gt_path)
    preds = {p["video_id"]: p.get("events", []) for p in submission.get("predictions", [])}

    l1_anom_hits = l1_class_hits = l1_n = 0
    per_level = defaultdict(list)

    for vid, gts in events_gt.items():
        lvl = levels[vid]
        pv = preds.get(vid, [])          # unanswered video is scored as normal
        if lvl == 1:
            l1_n += 1
            pred_anom = bool(pv)
            true_anom = bool(gts)
            l1_anom_hits += int(pred_anom == true_anom)
            true_cls = gts[0]["class_name"] if gts else "normal"
            pred_cls = pv[0]["class_name"] if pv else "normal"
            l1_class_hits += int(pred_cls == true_cls)
        else:
            per_level[lvl].append(score_video_temporal(pv, gts, lvl))

    l1 = (0.5 * l1_anom_hits / l1_n + 0.5 * l1_class_hits / l1_n) if l1_n else 0.0
    out = {
        "level1": l1,
        "level1_anomaly_acc": l1_anom_hits / l1_n if l1_n else 0.0,
        "level1_class_acc": l1_class_hits / l1_n if l1_n else 0.0,
        "level2": sum(per_level[2]) / len(per_level[2]) if per_level[2] else 0.0,
        "level3": sum(per_level[3]) / len(per_level[3]) if per_level[3] else 0.0,
    }
    out["overall"] = (out["level1"] + out["level2"] + out["level3"]) / 3
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--per-video", action="store_true")
    args = ap.parse_args()

    sub = json.load(open(args.submission, encoding="utf-8"))
    res = score(sub, args.gt)
    for k, v in res.items():
        print(f"{k:22s} {v:.4f}")

    if args.per_video:
        events_gt, levels = load_ground_truth(args.gt)
        preds = {p["video_id"]: p.get("events", []) for p in sub["predictions"]}
        print("\nper-video (levels 2-3):")
        for vid, gts in sorted(events_gt.items()):
            if levels[vid] == 1:
                continue
            s = score_video_temporal(preds.get(vid, []), gts, levels[vid])
            print(f"  {vid} L{levels[vid]} {s:.3f}  gt={len(gts)} pred={len(preds.get(vid, []))}")


if __name__ == "__main__":
    main()
