"""Encode every video once into cached frame embeddings.

Decoding is the bottleneck, not the GPU, so decoding runs in worker threads
(OpenCV releases the GIL) and feeds a single-consumer encode loop.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .config import CACHE, FPS, MAX_FRAMES, TEST, TRAIN
from .encoder import Encoder
from .video import sample_frames


def list_train_videos() -> list[tuple[str, str, str]]:
    """(video_id, class_name, path) for every training video."""
    out = []
    for cls_dir in sorted(p for p in TRAIN.iterdir() if p.is_dir()):
        for v in sorted((cls_dir / "videos").glob("*.mp4")):
            out.append((v.stem, cls_dir.name, str(v)))
    return out


def list_test_videos() -> list[tuple[str, str, str]]:
    return [(v.stem, "unknown", str(v)) for v in sorted((TEST / "videos").glob("*.mp4"))]


def _decode(path: str, out_size: int, tiles: int = 1):
    ts, frames = [], []
    for t, f in sample_frames(path, FPS, MAX_FRAMES, out_size=out_size, tiles=tiles):
        ts.append(t)
        frames.append(f)
    return np.asarray(ts, dtype=np.float32), frames


def extract(items, out_path, encoder: Encoder, workers: int = 4, batch_size: int = 64):
    """Write one .npz holding {vid}__emb and {vid}__ts for every item."""
    q: queue.Queue = queue.Queue(maxsize=workers * 2)
    todo = list(items)

    def _producer(shard):
        for vid, _cls, path in shard:
            try:
                q.put((vid, *_decode(path, encoder.size, encoder.tiles)))
            except Exception as exc:  # a corrupt file must not kill the run
                print(f"  decode failed {vid}: {exc}")
                q.put((vid, np.zeros(0, np.float32), []))

    # Checkpoint periodically: a crash an hour into a run must not cost the run.
    store, t0, done, nframes = {}, time.time(), 0, 0
    ckpt = Path(str(out_path) + ".partial.npz")
    if ckpt.exists():
        prev = np.load(ckpt)
        store = {k: prev[k] for k in prev.files}
        have = {k.split("__")[0] for k in store}
        todo = [t for t in todo if t[0] not in have]
        print(f"  resuming: {len(have)} already cached, {len(todo)} to go")
        if not todo:
            np.savez(out_path, **store)
            print(f"wrote {out_path} (from checkpoint)")
            return

    threads = [
        threading.Thread(target=_producer, args=(todo[i::workers],), daemon=True)
        for i in range(workers)
    ]
    for t in threads:
        t.start()

    while done < len(todo):
        vid, ts, frames = q.get()
        emb = encoder.encode_frames(frames, batch_size=batch_size) if frames else \
            np.zeros((0, encoder.dim), np.float32)
        store[f"{vid}__emb"] = emb.astype(np.float16)
        store[f"{vid}__ts"] = ts
        done += 1
        nframes += len(frames)
        if done % 100 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  {nframes} frames  {el:.0f}s  "
                  f"({nframes/max(el,1e-6):.0f} fps)", flush=True)
        if done % 250 == 0:
            np.savez(ckpt, **store)

    np.savez(out_path, **store)
    if ckpt.exists():
        ckpt.unlink()
    print(f"wrote {out_path}  ({len(store)//2} videos, {nframes} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    enc = Encoder()
    np.save(CACHE / "text_prototypes.npy", enc.class_prototypes())

    for split in (["train", "test"] if args.split == "both" else [args.split]):
        items = list_train_videos() if split == "train" else list_test_videos()
        if args.limit:
            items = items[: args.limit]
        print(f"{split}: {len(items)} videos")
        extract(items, CACHE / f"{split}_emb.npz", enc, args.workers, args.batch_size)


if __name__ == "__main__":
    main()
