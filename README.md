# Retail Shelf Intelligence Pipeline

End-to-end ML inference pipeline that analyses retail shelf images and produces
brand-level presence metrics, OCR-extracted price tags, shelf-row segmentation,
and annotated output images.

---

## Pipeline Overview

```
Input Image
   │
   ├── [1] YOLOv8n Object Detection  ──→  product bounding boxes
   │
   ├── [2] CLIP Zero-Shot Classification  ──→  brand per box
   │
   ├── [3] EasyOCR Extraction  ──→  price tags & shelf labels
   │
   ├── [4] DBSCAN Shelf-Row Segmentation  ──→  row ID per product
   │
   └── [5] Metrics Aggregation + Visualisation
              │                │
              ▼                ▼
        JSON output     Annotated image
```

See **`architecture.png`** for the full visual diagram (generate it with
`python generate_architecture.py`).

---

## Quickstart

### 1. Clone and install

```bash
git clone <repo-url>
cd retail-shelf-intelligence

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU**: If a CUDA-capable GPU is present, PyTorch will use it automatically
> for both YOLO and CLIP inference.  All models also run on CPU.

### 2. Run on the provided test images

```bash
python main.py "images/img_1 (1).jpg" "images/img_2 (1).jpg" "images/img_3 (1).jpg"
```

Outputs are written to `outputs/`:
- `<stem>_annotated.jpg` — bounding boxes + brand labels + row overlays
- `<stem>_result.json`   — metrics dict

### 3. Run on any custom image

```bash
python main.py path/to/shelf.jpg --output-dir results/
```

### 4. Generate architecture diagram

```bash
python generate_architecture.py
# creates architecture.png in the project root
```

### 5. Run tests

```bash
pytest tests/ -v --cov=pipeline --cov=utils --cov-report=term-missing
```

---

## Output Format

For each image the pipeline returns:

```json
{
  "image_name": "img_1 (1).jpg",
  "total_products": 32,
  "brands": {
    "Coca-Cola": 12,
    "Pepsi":      6,
    "Amul":       0,
    "Other":      14
  },
  "ocr_labels": ["₹125", "₹50", "₹99", "₹20"],
  "shelf_share": {
    "Coca-Cola": 0.4231,
    "Pepsi":     0.1875,
    "Other":     0.3894
  }
}
```

---

## Model Selection Justification

### Detection — YOLOv8n (Ultralytics)

| Criterion | Choice |
|-----------|--------|
| Speed | YOLOv8n runs ~10 ms/image on GPU, ~150 ms on CPU — practical for batch jobs |
| Accuracy | Nano model trades ~5% mAP vs YOLOv8x; acceptable for a detection front-end |
| Deployment | Single `.pt` file, auto-downloaded; no Docker or compilation needed |
| Limitation | COCO classes don't include snack bags or dairy cartons; a grid-based region-proposal fallback compensates for dense shelves (img_2, img_3) |

**Alternative considered**: Grounding DINO (open-vocabulary) would detect arbitrary
product descriptions but requires heavier dependencies and is slower on CPU.
YOLOv8n's simpler setup is justified for a prototype.

### Classification — CLIP ViT-B/32 (OpenAI / Hugging Face)

| Criterion | Choice |
|-----------|--------|
| Zero-shot | Brand prompts live in `config.py`; adding a new brand requires no retraining |
| Accuracy | CLIP identifies visual brand cues (logo colours, typography) well enough for prototype-level classification |
| Speed | ~200 ms/crop on CPU; ~20 ms on GPU |
| Deployment | Downloadable via `transformers`; single model file |
| Limitation | Confuses visually similar brands (e.g. Sprite vs 7UP — both green bottles) |

**Alternative considered**: Fine-tuned ResNet-50 would be faster and more accurate
for fixed brand sets, but requires labelled training data that is not available here.

### OCR — EasyOCR

| Criterion | Choice |
|-----------|--------|
| Ease of use | Pure Python; no system Tesseract binary required |
| Accuracy | Better than Tesseract on natural scene text (shelf labels, price tags) |
| Speed | ~1–2 s per image on CPU |
| Limitation | Struggles with very small or rotated text |

**Alternative considered**: PaddleOCR achieves higher accuracy but is heavier to
install and overkill for the price-tag extraction task here.

### Segmentation — DBSCAN (scikit-learn)

| Criterion | Choice |
|-----------|--------|
| No prior | DBSCAN does not require knowing the number of shelf rows in advance |
| Robustness | Handles outlier products that don't sit neatly on a row |
| Speed | Negligible — runs on bounding-box centroids, not pixels |
| Limitation | `eps` parameter (55 px) may need tuning for very high-resolution images |

### CPU vs GPU

All four models run on CPU without modification.  Expected processing times:

| Model | CPU | GPU (T4) |
|-------|-----|----------|
| YOLOv8n | ~150 ms/img | ~10 ms/img |
| CLIP ViT-B/32 per crop | ~200 ms | ~15 ms |
| EasyOCR | ~1 500 ms/img | ~200 ms/img |

For offline batch analysis of 100s of images, a GPU reduces total runtime by ~10×.
For a real-time kiosk scenario, the CLIP step should be batched or replaced with a
lighter MobileNet classifier.

---

## Project Structure

```
retail-shelf-intelligence/
├── config.py                  # Central configuration (models, thresholds, brands)
├── main.py                    # CLI entry point
├── generate_architecture.py   # Produces architecture.png
├── requirements.txt
├── README.md
├── architecture.png           # Generated diagram
│
├── pipeline/
│   ├── detector.py            # YOLOv8 detection + grid fallback
│   ├── classifier.py          # CLIP zero-shot brand classification
│   ├── ocr_extractor.py       # EasyOCR price/label extraction
│   ├── segmenter.py           # DBSCAN shelf-row segmentation + share-of-shelf
│   └── analyzer.py            # Orchestrates all four stages
│
├── utils/
│   └── visualizer.py          # Annotated image generation (bboxes, legend, rows)
│
├── tests/
│   ├── test_detector.py
│   ├── test_classifier.py
│   ├── test_ocr_extractor.py
│   ├── test_segmenter.py
│   └── test_analyzer.py
│
├── images/                    # Provided test shelf images
│   ├── img_1 (1).jpg          # Beverage shelf
│   ├── img_2 (1).jpg          # Snacks shelf
│   └── img_3 (1).jpg          # Dairy shelf
│
└── outputs/                   # Generated outputs (annotated images + JSON)
```

---

## Assumptions and Limitations

| # | Assumption / Limitation |
|---|-------------------------|
| 1 | COCO-trained YOLOv8n does not have dedicated classes for snack bags or tetra packs; the grid-fallback generates region proposals for dense shelves |
| 2 | CLIP brand classification is zero-shot and may confuse visually similar brands (e.g. Sprite ↔ 7UP, Pepsi ↔ Coca-Cola dark bottles) |
| 3 | OCR is run on the full image; very small price tags (<12 px font) may be missed |
| 4 | Share-of-shelf is approximated as bounding-box area ratio — not pixel-level segmentation masks |
| 5 | Brand list is configured for the three provided test images; extending to new brands only requires adding entries to `BRANDS` and `BRAND_PROMPTS` in `config.py` |
| 6 | The pipeline is designed for offline batch analysis; real-time use would require CLIP batching or a lighter classifier |
| 7 | DBSCAN `eps=55 px` is calibrated for ~600–1000 px tall images with 3–5 shelf rows |

---

## Extending the Pipeline

**Add a new brand**

```python
# config.py
BRANDS = [..., "Red Bull"]
BRAND_PROMPTS["Red Bull"] = "a photo of a Red Bull energy drink silver and blue can"
BRAND_COLORS["Red Bull"] = (200, 200, 20)  # yellow-silver
```

No retraining required — CLIP handles it zero-shot.

**Replace the detector**

Swap `ShelfDetector` for a `GroundingDinoDetector` or `YOLOWorldDetector` by
implementing the same `detect(image: np.ndarray) -> List[Detection]` interface.
`ShelfAnalyzer` uses only that interface, so the rest of the pipeline is unaffected.
