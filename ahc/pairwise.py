"""Second-stage discriminators for the class pairs that actually get confused.

The clip head is ~86% accurate overall but its errors concentrate in a few
semantically adjacent pairs: fire against smoke (smoke reaches 11.8% under a
nearest-centroid probe), and fighting against loitering. A 12-way head spends
its capacity separating everything from everything; a binary head trained only
on one pair sees far more of its decision boundary.

At inference: if the clip head's top two classes are a known pair and it is not
confident, the pair's own discriminator breaks the tie.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from .config import CACHE, CLASS_TO_IDX, CLASSES, RUNS
from .centroids import video_mean_embeddings

# pairs chosen from measured confusions, not from intuition
PAIRS = [
    ("fire", "smoke"),
    ("fighting_or_violence", "loitering_or_suspicious_presence"),
    ("traffic_accident", "wrong_way_driving"),
    ("road_spill_or_debris", "stalled_or_broken_down_vehicle"),
    ("stalled_or_broken_down_vehicle", "vehicle_blocking_traffic"),
    ("traffic_accident", "traffic_congestion"),
]


class Binary(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def train_pair(E, Y, a, b, epochs=60, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ia, ib = CLASS_TO_IDX[a], CLASS_TO_IDX[b]
    m = (Y == ia) | (Y == ib)
    if m.sum() < 40:
        return None, 0.0
    X = E[m]
    t = (Y[m] == ib).astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = max(4, int(len(X) * 0.2))
    va, tr = idx[:n], idx[n:]

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(t, device=device)

    w = torch.tensor([1.0 / max((t[tr] == 0).sum(), 1),
                      1.0 / max((t[tr] == 1).sum(), 1)],
                     dtype=torch.float32, device=device)
    w = w / w.mean()

    model = Binary(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        lossf(model(Xt[tr]), yt[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xt[va]).argmax(1) == yt[va]).float().mean().item()
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "mu": mu, "sd": sd, "a": a, "b": b, "in_dim": X.shape[1],
            "val_acc": best, "n": int(m.sum())}, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    E, Y, _ = video_mean_embeddings()
    print(f"{len(E)} clip embeddings, dim {E.shape[1]}")

    saved = {}
    for a, b in PAIRS:
        rec, acc = train_pair(E, Y, a, b, epochs=args.epochs)
        na = int((Y == CLASS_TO_IDX[a]).sum())
        nb = int((Y == CLASS_TO_IDX[b]).sum())
        base = max(na, nb) / max(na + nb, 1)      # majority-class baseline
        if rec is None:
            print(f"  {a} vs {b}: too few samples")
            continue
        keep = acc > base + 0.02
        print(f"  {a[:22]:22s} vs {b[:22]:22s} n={na+nb:4d} "
              f"val {acc:.3f} (majority {base:.3f}) {'KEEP' if keep else 'drop'}")
        if keep:
            saved[f"{a}|{b}"] = rec

    path = args.out or (RUNS / "pairwise.pt")
    torch.save(saved, path)
    print(f"saved {len(saved)} discriminators -> {path}")


if __name__ == "__main__":
    main()
