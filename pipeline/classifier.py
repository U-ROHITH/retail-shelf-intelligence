"""
Zero-shot brand classification using OpenAI CLIP (ViT-B/32).

Design decisions:
- CLIP chosen because it requires no fine-tuning; brand prompts can be updated in
  config.py without retraining, making it practical for new brand roll-outs.
- ViT-B/32 selected over ViT-L/14 for faster inference with acceptable accuracy;
  runs on CPU at ~200 ms/crop which is acceptable for offline batch analysis.
- Each detected crop is encoded independently; batch processing per image would
  be faster but complicates the API without significant benefit at this scale.
- Crops smaller than MIN_CROP_PX are returned as "Other" to avoid noise from
  tiny partial detections at image borders.
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
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()
        self._text_prompts: List[str] = [BRAND_PROMPTS[b] for b in BRANDS]

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

    def _run_clip(self, crop: Image.Image) -> str:
        inputs = self._processor(
            text=self._text_prompts,
            images=crop,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        brand_idx = int(probs.argmax().item())
        return BRANDS[brand_idx]
