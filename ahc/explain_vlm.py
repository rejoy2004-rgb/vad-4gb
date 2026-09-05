"""Generate an explanation per detected event with a small VLM.

This is the one place a generative model earns its cost: it runs on the handful
of spans that already survived detection, never per frame. On the public test
set that is roughly a dozen calls for the whole Difficulty-3 set, against
~6,800 encoder calls - so it barely moves the latency figure.

Falls back to retrieval (ahc.infer.Explainer) whenever the model is missing or
the generated text fails the 20-500 character rule.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

PROMPT = {
    "traffic_accident": "Describe the vehicle collision and its aftermath in this scene.",
    "traffic_congestion": "Describe the traffic density and how the vehicles are moving.",
    "stalled_or_broken_down_vehicle": "Describe the stopped vehicle and where it is standing.",
    "vehicle_blocking_traffic": "Describe the vehicle obstructing the road and what it blocks.",
    "wrong_way_driving": "Describe the vehicle travelling against the flow of traffic.",
    "road_spill_or_debris": "Describe the debris or spilled material on the road.",
    "waterlogging_or_flood": "Describe the flooding and what it covers.",
    "fire": "Describe the fire and what is burning.",
    "smoke": "Describe the smoke and where it is coming from.",
    "fighting_or_violence": "Describe the physical altercation between the people.",
    "loitering_or_suspicious_presence": "Describe the person and how long they remain in place.",
}


class VLMExplainer:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.ok = False
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = (AutoModelForVision2Seq
                          .from_pretrained(model_id, dtype=dtype,
                                           low_cpu_mem_usage=True)
                          .to(self.device).eval())
            self.ok = True
        except Exception as exc:
            print(f"  VLM explainer unavailable ({type(exc).__name__}), "
                  f"falling back to retrieval")

    @torch.no_grad()
    def explain(self, frames: list[np.ndarray], class_name: str,
                start=None, end=None) -> str | None:
        """`frames` are RGB uint8 arrays sampled from inside the event."""
        if not self.ok or not frames:
            return None
        from PIL import Image

        # a few frames spread across the span, not the whole span
        pick = frames[:: max(1, len(frames) // 3)][:3]
        images = [Image.fromarray(f) for f in pick]
        q = PROMPT.get(class_name, "Describe what is happening in this scene.")
        msg = [{"role": "user", "content":
                [{"type": "image"} for _ in images] + [{"type": "text", "text": q}]}]
        try:
            text = self.processor.apply_chat_template(msg, add_generation_prompt=True)
            inputs = self.processor(text=text, images=images,
                                    return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=70, do_sample=False)
            gen = self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        except Exception:
            return None

        gen = " ".join(gen.split()).strip()
        if start is not None and end is not None and end > start:
            obs = f" Observed for {end - start:.0f} s from {start:.0f} s."
            if len(gen) + len(obs) <= 500:
                gen += obs
        if not 20 <= len(gen) <= 500:
            return None
        return gen
