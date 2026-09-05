"""Build the retrieval bank used for event explanations.

For every training clip that carries a description_summary, store the mean
embedding over its labelled interval alongside the text. At inference an event
looks up the nearest clip of the same class and reuses its wording — a
generative model's output without a generative model's latency.
"""
from __future__ import annotations

import csv

import numpy as np

from .config import CACHE, TRAIN


def main():
    data = np.load(CACHE / "train_emb.npz")
    have = {k.split("__")[0] for k in data.files}

    embs, texts, classes = [], [], []
    for cls_dir in sorted(p for p in TRAIN.iterdir() if p.is_dir()):
        if cls_dir.name == "normal":
            continue
        gt = cls_dir / "ground_truth.csv"
        if not gt.exists():
            continue
        with open(gt, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vid, desc = row["video_id"], (row["description_summary"] or "").strip()
                if vid not in have or not 20 <= len(desc) <= 500:
                    continue
                emb, ts = data[f"{vid}__emb"], data[f"{vid}__ts"]
                if len(emb) == 0:
                    continue
                s, e = row["start_time_sec"], row["end_time_sec"]
                sel = np.ones(len(ts), bool)
                if s and e:
                    m = (ts >= float(s)) & (ts <= float(e))
                    if m.any():
                        sel = m
                v = np.asarray(emb, np.float32)[sel].mean(0)
                embs.append(v / (np.linalg.norm(v) + 1e-9))
                texts.append(desc)
                classes.append(row["class_name"])

    np.savez(CACHE / "desc_bank.npz", emb=np.stack(embs),
             text=np.array(texts, dtype=object), cls=np.array(classes, dtype=object))
    print(f"desc bank: {len(texts)} descriptions over "
          f"{len(set(classes))} classes -> {CACHE / 'desc_bank.npz'}")


if __name__ == "__main__":
    main()
