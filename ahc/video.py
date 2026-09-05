"""Frame sampling. Sequential grab/retrieve rather than seeking, because
seeking on long H.264 files costs more than decoding straight through."""
from __future__ import annotations

import cv2
import numpy as np


def probe(path) -> tuple[float, int, float]:
    """Return (fps, frame_count, duration_sec)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or fps > 240:
        fps = 25.0
    return fps, n, (n / fps if n else 0.0)


def sample_frames(path, target_fps: float = 2.0, max_frames: int = 2048,
                  out_size: int | None = None):
    """Yield (timestamp_sec, RGB uint8 frame) at approximately target_fps.

    If the video is long enough that target_fps would exceed max_frames, the
    stride is widened so the whole video is still covered uniformly.

    `out_size` downsizes each frame as it is decoded. Do not skip it when
    collecting frames into a list: a 10-minute 1080p video at 2 fps is ~9 GB
    of full-resolution frames, but only ~180 MB at 224x224.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0 or src_fps > 240:
        src_fps = 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    stride = max(1, int(round(src_fps / target_fps)))
    if n and n / stride > max_frames:
        stride = int(np.ceil(n / max_frames))

    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if out_size is not None:
                    frame = cv2.resize(frame, (out_size, out_size),
                                       interpolation=cv2.INTER_AREA)
                yield idx / src_fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()
