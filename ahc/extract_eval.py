"""Cache embeddings for the private evaluation videos.

Same encoder and sampling as the practice set; separate file so decoder work on
the evaluation pack does not require re-encoding 47 minutes of video each time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import CACHE
from .encoder import Encoder
from .extract import extract
from .run_eval import EVAL, eval_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(EVAL))
    ap.add_argument("--out", default=str(CACHE / "eval_emb.npz"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    items = [(vid, f"L{lvl}", path) for vid, lvl, path in eval_videos(Path(args.root))]
    print(f"{len(items)} evaluation videos")
    extract(items, Path(args.out), Encoder(), args.workers, args.batch_size)


if __name__ == "__main__":
    main()
