"""
Zero-shot brand classification using OpenAI CLIP (ViT-B/32).

Design decisions:
- CLIP chosen because it requires no fine-tuning; brand prompts can be updated in
  config.py without retraining, making it practical for new brand roll-outs.
- ViT-B/32 selected over ViT-L/14 for faster inference with acceptable accuracy.
- Model weights loaded in float16 to halve RSS (~300 MB vs ~600 MB on CPU).
- Text embeddings for all brand prompts are pre-computed once at init time so
  repeated classify() calls only run the image encoder (much cheaper).
- Uses text_model + text_projection directly for compatibility with
  transformers >= 5.x where get_text_features() returns a model output object
  rather than a plain tensor.
- Crops smaller than MIN_CROP_PX are returned as "Other" to skip border noise.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import BRAND_PROMPTS, BRANDS, CLIP_MODEL
from pipeline.detector import Detection

MIN_CROP_PX = 10  # minimum edge length in pixels for a valid crop


class BrandClassifier:
    def __init__(self, model_name: str = CLIP_MODEL) -> None:
        self._model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()
        self._text_prompts: List[str] = [BRAND_PROMPTS[b] for b in BRANDS]
        # Pre-compute normalised text features once — reused for every crop
        self._text_features: torch.Tensor = self._encode_texts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image: np.ndarray, detection: Detection) -> str:
        """Return the best-matching brand name for a single detection crop."""
        x1, y1, x2, y2 = detection.bbox
        crop = Image.fromarray(image[y1:y2, x1:x2])
        if crop.width < MIN_CROP_PX or crop.height < MIN_CROP_PX:
            return "Other"
        return self._run_clip(crop)

    def classify_batch(
        self, image: np.ndarray, detections: List[Detection]
    ) -> Dict[int, str]:
        """Return a mapping of {detection_index: brand_name}."""
        return {i: self.classify(image, det) for i, det in enumerate(detections)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_texts(self) -> torch.Tensor:
        """Encode all brand prompts into normalised feature vectors."""
        inputs = self._processor(
            text=self._text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Cast any float tensors to match model dtype (float16)
        inputs = {
            k: v.to(torch.float16) if v.is_floating_point() else v
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            text_out = self._model.text_model(**inputs)
            features = self._model.text_projection(text_out.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
        return features  # [num_brands, embed_dim]

    def _run_clip(self, crop: Image.Image) -> str:
        """Run image encoding and return the closest brand."""
        inputs = self._processor(images=crop, return_tensors="pt")
        inputs = {
            k: v.to(torch.float16) if v.is_floating_point() else v
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            vision_out = self._model.vision_model(**inputs)
            img_features = self._model.visual_projection(vision_out.pooler_output)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            # cosine similarity scaled to logit range
            logits = (img_features @ self._text_features.T) * 100.0
            probs = logits.softmax(dim=-1)[0]

        return BRANDS[int(probs.argmax().item())]
