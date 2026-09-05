"""Train the window classifier on cached embeddings.

The encoder stays frozen, so this is a small MLP over pre-computed features:
it trains in well under a minute on the GPU, which is what makes it possible
to iterate on windowing and thresholds during a one-day build.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
import torch.nn as nn

from .config import CACHE, CLASS_TO_IDX, NUM_CLASSES, RUNS, TRAIN
from .features import video_windows


class Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 768, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


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


def build_dataset(npz_path=None):
    """-> X (N,F), y (N,), groups (N,), hard (N,) for every training window.

    `hard` marks a normal window that came from an *anomalous* video. These are
    the only examples that teach "this scene, but not this moment"; every other
    negative comes from a different scene entirely. Without them the head learns
    scene appearance and saturates on long footage - on several test videos
    p(normal) never exceeds 0.02 for the whole video.
    """
    data = np.load(npz_path or (CACHE / "train_emb.npz"))
    intervals = load_train_intervals()
    vids = sorted({k.split("__")[0] for k in data.files})

    X, y, groups, hard = [], [], [], []
    for vid in vids:
        emb = data[f"{vid}__emb"]
        ts = data[f"{vid}__ts"]
        if len(emb) == 0:
            continue
        feats, spans = video_windows(emb, ts)
        evs = intervals.get(vid, [])
        for f, (t0, t1) in zip(feats, spans):
            label = 0
            best = 0.0
            for s, e, cname in evs:
                ov = max(0.0, min(t1, e) - max(t0, s))
                span = max(t1 - t0, 1e-6)
                # a window counts as the event if the event fills half of it,
                # or if the window sits inside a longer event
                frac = ov / span
                if frac > best and frac >= 0.5:
                    best, label = frac, CLASS_TO_IDX[cname]
            X.append(f)
            y.append(label)
            groups.append(vid)
            hard.append(label == 0 and bool(evs))
    return (np.stack(X), np.asarray(y), np.asarray(groups),
            np.asarray(hard, dtype=bool))


def train(X, y, groups, epochs=30, val_frac=0.15, seed=0, device=None,
          hard=None, hard_weight=1.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    # split by video, never by window, or windows from one clip leak across
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_ids = set(uniq[:n_val].tolist())
    va = np.array([g in val_ids for g in groups])
    tr = ~va

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd

    Xtr = torch.tensor(Xn[tr], device=device)
    ytr = torch.tensor(y[tr], device=device, dtype=torch.long)
    Xva = torch.tensor(Xn[va], device=device)
    yva = torch.tensor(y[va], device=device, dtype=torch.long)

    counts = np.bincount(y[tr], minlength=NUM_CLASSES).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = w / w[w > 0].mean()
    weight = torch.tensor(w, device=device, dtype=torch.float32)

    # per-sample weighting so in-scene negatives can be emphasised
    sw = np.ones(len(y), np.float32)
    if hard is not None and hard_weight != 1.0:
        sw[hard] = hard_weight
    swt = torch.tensor(sw[tr], device=device, dtype=torch.float32)

    model = Head(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05,
                                reduction="none")

    best_acc, best_state = -1.0, None
    bs = 512
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            per = lossf(model(Xtr[idx]), ytr[idx])
            loss = (per * swt[idx]).sum() / swt[idx].sum()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1)
            acc = (pred == yva).float().mean().item()
            anom = ((pred > 0) == (yva > 0)).float().mean().item()
            if hard is not None and hard[va].any():
                hv = torch.tensor(hard[va], device=device)
                hard_acc = (pred[hv] == 0).float().mean().item()
            else:
                hard_acc = float("nan")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep+1:02d}  loss {tot/len(Xtr):.4f}  val_cls {acc:.4f}  "
              f"val_anom {anom:.4f}  in-scene-neg {hard_acc:.4f}")

    model.load_state_dict(best_state)
    return model, mu, sd, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3,
                    help="ensemble size; averaging removes single-seed variance")
    ap.add_argument("--hard-weight", type=float, default=1.0,
                    help="loss weight on normal windows inside anomalous videos")
    ap.add_argument("--out", default=str(RUNS / "head.pt"))
    args = ap.parse_args()

    print("building windows...")
    X, y, g, hard = build_dataset()
    print(f"  {len(X)} windows, dim {X.shape[1]}, {len(np.unique(g))} videos")
    print(f"  in-scene negatives: {int(hard.sum())} of {int((y == 0).sum())} normals")
    print("  label counts:", np.bincount(y, minlength=NUM_CLASSES).tolist())

    members, accs = [], []
    for s in range(args.seeds):
        print(f"seed {s}:")
        model, mu, sd, acc = train(X, y, g, epochs=args.epochs, seed=s,
                                   hard=hard, hard_weight=args.hard_weight)
        members.append({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "mu": mu, "sd": sd, "val_acc": acc})
        accs.append(acc)

    torch.save({"members": members, "in_dim": X.shape[1],
                "state_dict": members[0]["state_dict"],
                "mu": members[0]["mu"], "sd": members[0]["sd"],
                "val_acc": float(np.mean(accs))}, args.out)
    json.dump({"val_cls_acc": float(np.mean(accs)), "seeds": args.seeds,
               "windows": int(len(X))},
              open(RUNS / "head_meta.json", "w"), indent=2)
    print(f"saved {args.out}  ({len(members)} members, mean val acc {np.mean(accs):.4f})")


if __name__ == "__main__":
    main()
