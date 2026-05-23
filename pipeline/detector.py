"""
Product detection using YOLOv8n pretrained on COCO.

Design decisions:
- YOLOv8n chosen for its speed/accuracy balance and easy deployment (single .pt file).
- All 80 COCO classes are considered; irrelevant detections are filtered by bounding-box
  size rather than class ID because snack bags and dairy cartons have no COCO equivalent.
- Grid-based region proposals are generated as a fallback when YOLO returns fewer
  products than expected (e.g. dense snack shelves that COCO classes don't cover).
- NMS is handled internally by YOLO; no duplicate proposals for the YOLO path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from config import (
    DETECTION_CONFIDENCE,
    GRID_COLS,
    GRID_FALLBACK_THRESHOLD,
    GRID_ROWS,
    GRID_VARIANCE_THRESHOLD,
    MAX_BBOX_AREA_RATIO,
    MAX_BBOX_HEIGHT_RATIO,
    MAX_BBOX_WIDTH_RATIO,
    MIN_BBOX_AREA_RATIO,
    YOLO_MODEL,
)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def center_y(self) -> int:
        return (self.bbox[1] + self.bbox[3]) // 2


def _is_valid_bbox(
    x1: int, y1: int, x2: int, y2: int, img_h: int, img_w: int
) -> bool:
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    area_ratio = (w * h) / (img_h * img_w)
    if area_ratio < MIN_BBOX_AREA_RATIO or area_ratio > MAX_BBOX_AREA_RATIO:
        return False
    if w / img_w > MAX_BBOX_WIDTH_RATIO:
        return False
    if h / img_h > MAX_BBOX_HEIGHT_RATIO:
        return False
    return True


def _grid_proposals(image: np.ndarray) -> list[Detection]:
    """Return non-overlapping grid cells that contain visible content."""
    h, w = image.shape[:2]
    proposals: list[Detection] = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y1 = r * h // GRID_ROWS
            y2 = (r + 1) * h // GRID_ROWS
            x1 = c * w // GRID_COLS
            x2 = (c + 1) * w // GRID_COLS
            cell = image[y1:y2, x1:x2]
            if np.var(cell) < GRID_VARIANCE_THRESHOLD:
                continue
            proposals.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=0.50,
                    class_id=-1,
                    class_name="grid_proposal",
                )
            )
    return proposals


class ShelfDetector:
    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        confidence: float = DETECTION_CONFIDENCE,
    ) -> None:
        self._model = YOLO(model_name)
        self._confidence = confidence

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on an RGB image array and return validated detections."""
        img_h, img_w = image.shape[:2]
        results = self._model(image, conf=self._confidence, verbose=False)

        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if not _is_valid_bbox(x1, y1, x2, y2, img_h, img_w):
                    continue
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                        class_name=result.names[int(box.cls[0])],
                    )
                )

        if len(detections) < GRID_FALLBACK_THRESHOLD:
            detections = _grid_proposals(image)

        return detections
