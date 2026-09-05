"""Frozen SigLIP image/text encoder, fp16 on GPU.

The encoder is the only thing that touches every frame, so everything here is
about keeping it cheap: fp16, no grad, batched, and preprocessing done with
plain numpy rather than the (slow) PIL path in the HF processor.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import ENCODER_ID, CLASSES, PROMPTS


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
    def __init__(self, model_id: str = ENCODER_ID, device: str | None = None):
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
        self.dim = self.model.config.text_config.hidden_size

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

    @torch.no_grad()
    def encode_frames(self, frames: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """L2-normalised image embeddings, float32 numpy of shape (N, dim)."""
        chunks = []
        for i in range(0, len(frames), batch_size):
            px = self._preprocess(frames[i : i + batch_size])
            feats = _as_tensor(self.model.get_image_features(pixel_values=px))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.float().cpu().numpy())
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

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
