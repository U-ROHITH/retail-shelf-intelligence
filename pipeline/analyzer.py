"""
Main pipeline orchestrator.

Wires together detection → classification → OCR → segmentation → metrics.
Each stage is independently replaceable; the analyzer only holds references
to the four component objects.

Memory strategy: models are loaded lazily on first use and held for the
lifetime of the ShelfAnalyzer so they are reused across multiple images.
On memory-constrained hosts each component can be instantiated and deleted
within analyze(); the segmenter is stateless and cheap to recreate.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from pipeline.classifier import BrandClassifier
from pipeline.detector import Detection, ShelfDetector
from pipeline.ocr_extractor import OCRExtractor
from pipeline.segmenter import ShelfSegmenter
from utils.visualizer import annotate_image


class ShelfAnalyzer:
    """
    Orchestrates the four-stage shelf analysis pipeline.

    Models are loaded one at a time and released before the next stage
    to keep peak RSS below ~900 MB on CPU-only hosts.
    """

    def analyze(self, image_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Run the full pipeline on *image_path*.

        Returns
        -------
        result : dict
            JSON-serialisable metrics dict matching the assignment output spec.
        annotated : np.ndarray
            RGB image with bounding boxes, brand labels, and row overlays.
        """
        path = Path(image_path)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. Detect product regions — load, run, release
        detector = ShelfDetector()
        detections: List[Detection] = detector.detect(rgb)
        del detector
        gc.collect()

        # 2. Classify each detected region into a brand — load, run, release
        classifier = BrandClassifier()
        brand_map = classifier.classify_batch(rgb, detections)
        brands: List[str] = [brand_map.get(i, "Other") for i in range(len(detections))]
        del classifier
        gc.collect()

        # 3. Extract shelf labels and price tags — load, run, release
        ocr = OCRExtractor()
        ocr_labels = ocr.extract(rgb)
        del ocr
        gc.collect()

        # 4. Segment shelf rows (stateless, no model)
        row_map = ShelfSegmenter().segment_rows(detections)

        # 5. Aggregate metrics
        brand_counts: Dict[str, int] = {}
        for brand in brands:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1

        shelf_share = ShelfSegmenter().shelf_share(detections, brands)

        result: Dict[str, Any] = {
            "image_name": path.name,
            "total_products": len(detections),
            "brands": brand_counts,
            "ocr_labels": ocr_labels,
            "shelf_share": shelf_share,
        }

        # 6. Build annotated image
        annotated = annotate_image(rgb.copy(), detections, brands, row_map)

        return result, annotated
