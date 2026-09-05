"""Frozen SigLIP image/text encoder, fp16 on GPU.

The encoder is the only thing that touches every frame, so everything here is
about keeping it cheap: fp16, no grad, batched, and preprocessing done with
plain numpy rather than the (slow) PIL path in the HF processor.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import ENCODER_ID, CLASSES, PROMPTS, TILES, VIEW_AVG


def _as_tensor(out):
    """transformers 5.x may hand back a ModelOutput instead of a tensor."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v if v.ndim == 2 else v.mean(dim=1)
    return out[0]


class Encoder:
    def __init__(self, model_id: str = ENCODER_ID, device: str | None = None,
                 tiles: int = TILES, view_avg: int = VIEW_AVG):
        from transformers import AutoModel, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModel.from_pretrained(model_id, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        ip = self.processor.image_processor
        self.size = int(getattr(ip.size, "height", None) or ip.size["height"])
        self.mean = np.array(ip.image_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(ip.image_std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.tiles = max(1, tiles)
        self.view_avg = max(1, view_avg)
        base = self.model.config.text_config.hidden_size
        # tiled mode concatenates max- and mean-pooled views
        self.dim = base * 2 if self.tiles > 1 else base
        self.base_dim = base

    # ------------------------------------------------------------------ images
    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        import cv2

        s = self.size
        out = np.empty((len(frames), s, s, 3), dtype=np.uint8)
        for i, f in enumerate(frames):
            # already at target size when the decoder downsized for us
            out[i] = f if f.shape[0] == s and f.shape[1] == s else \
                cv2.resize(f, (s, s), interpolation=cv2.INTER_AREA)
        x = out.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).to(self.device, self.dtype)

    def _views(self, frame: np.ndarray) -> list[np.ndarray]:
        """Full frame plus a t x t grid of crops, each at the model's size."""
        import cv2

        t, s = self.tiles, self.size
        out = [cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA)]
        h, w = frame.shape[:2]
        ch, cw = h // t, w // t
        for r in range(t):
            for c in range(t):
                crop = frame[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                if crop.shape[0] != s or crop.shape[1] != s:
                    crop = cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA)
                out.append(crop)
        return out

    @torch.no_grad()
    def encode_frames(self, frames: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """L2-normalised image embeddings, float32 numpy of shape (N, dim)."""
        if not frames:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.tiles > 1:
            return self._encode_tiled(frames, batch_size)
        if self.view_avg > 1:
            return self._encode_view_avg(frames, batch_size)
        chunks = []
        for i in range(0, len(frames), batch_size):
            px = self._preprocess(frames[i : i + batch_size])
            feats = _as_tensor(self.model.get_image_features(pixel_values=px))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def _encode_view_avg(self, frames, batch_size):
        """Average embeddings over the full frame plus centre/quadrant crops.

        Dimension is unchanged, so heads trained on single-view embeddings still
        apply. view_avg=2 adds a centre crop; 5 adds the four quadrants.
        """
        import cv2

        s = self.size
        views = []
        for f in frames:
            h, w = f.shape[:2]
            out = [cv2.resize(f, (s, s), interpolation=cv2.INTER_AREA)]
            cy, cx = h // 4, w // 4
            out.append(cv2.resize(f[cy:h - cy, cx:w - cx], (s, s),
                                  interpolation=cv2.INTER_AREA))
            if self.view_avg >= 5:
                for r in range(2):
                    for c in range(2):
                        crop = f[r * h // 2:(r + 1) * h // 2,
                                 c * w // 2:(c + 1) * w // 2]
                        out.append(cv2.resize(crop, (s, s),
                                              interpolation=cv2.INTER_AREA))
            views.append(out[: self.view_avg])

        n_views = len(views[0])
        flat = [v for vs in views for v in vs]
        embs = []
        for i in range(0, len(flat), batch_size):
            px = self._preprocess(flat[i : i + batch_size])
            fe = _as_tensor(self.model.get_image_features(pixel_values=px))
            fe = fe / fe.norm(dim=-1, keepdim=True)
            embs.append(fe.float().cpu().numpy())
        E = np.concatenate(embs, axis=0).reshape(len(frames), n_views, self.base_dim)
        m = E.mean(axis=1)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return m.astype(np.float32)

    @torch.no_grad()
    def _encode_tiled(self, frames, batch_size):
        """Encode every view, then pool across views per frame.

        Max-pooling is what recovers a small object: a tile containing only the
        debris scores high on that dimension even though the full frame does
        not. The mean is kept alongside so global context is not lost.
        """
        n_views = 1 + self.tiles * self.tiles
        flat = [v for f in frames for v in self._views(f)]
        embs = []
        for i in range(0, len(flat), batch_size):
            px = self._preprocess(flat[i : i + batch_size])
            feats = _as_tensor(self.model.get_image_features(pixel_values=px))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.float().cpu().numpy())
        E = np.concatenate(embs, axis=0).reshape(len(frames), n_views, self.base_dim)
        mx, mn = E.max(axis=1), E.mean(axis=1)
        mn /= np.linalg.norm(mn, axis=1, keepdims=True) + 1e-9
        mx /= np.linalg.norm(mx, axis=1, keepdims=True) + 1e-9
        return np.concatenate([mx, mn], axis=1).astype(np.float32)

    # ------------------------------------------------------------------- text
    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        tok = self.processor.tokenizer(
            texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to(self.device)
        feats = _as_tensor(self.model.get_text_features(**tok))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def class_prototypes(self) -> np.ndarray:
        """(num_classes, dim) mean prompt embedding per class, re-normalised."""
        protos = []
        for c in CLASSES:
            e = self.encode_text(PROMPTS[c]).mean(axis=0)
            protos.append(e / np.linalg.norm(e))
        return np.stack(protos).astype(np.float32)
