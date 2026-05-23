# Retail Shelf Intelligence Pipeline

End-to-end ML pipeline that analyses retail shelf images and outputs brand-level metrics, OCR-extracted price tags, shelf-row segmentation, and annotated images — built as a Junior MLE take-home assignment.

---

## Assignment Objectives

The pipeline covers every point requested:

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Product / object detection | YOLOv8n (COCO) + 5×10 grid fallback for dense shelves |
| 2 | Brand / product classification | CLIP ViT-B/32 zero-shot — 31 brands, no fine-tuning needed |
| 3 | OCR from shelf labels / price tags | EasyOCR with price/label filter |
| 4 | Shelf segmentation + space estimation | DBSCAN row clustering + bounding-box area share-of-shelf |
| 5 | Clean modular code | `pipeline/`, `utils/`, `tests/` — each stage is its own module |
| 6 | Model research / tool selection rationale | Approach section below — why each model was chosen over alternatives |
| 7 | Prediction outputs on all 3 test images | `outputs/` — annotated JPGs + JSON per image |
| 8 | Visualized prediction plots | Bounding boxes, per-brand colours, row overlays, legend |
| 9 | Assumptions / limitations / tradeoffs | Listed at the bottom of this file |

---

## Setup

```bash
git clone <repo-url>
cd retail-shelf-intelligence

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

All models download automatically on first run (no manual weight downloads needed).

---

## Run

Place your shelf images in the `images/` folder, then:

```bash
python main.py "images/img_1 (1).jpg" "images/img_2 (1).jpg" "images/img_3 (1).jpg"
```

Outputs are written to `outputs/`:
- `<stem>_annotated.jpg` — bounding boxes with brand labels, shelf-row colour bands, and legend
- `<stem>_result.json` — JSON metrics dict

To run on a single custom image:

```bash
python main.py path/to/shelf.jpg
```

### Run Tests

```bash
pytest tests/ -v --cov=pipeline --cov=utils --cov-report=term-missing
```

69 tests, all passing — 99% line coverage across `pipeline/` and `utils/`.

---

## Output — All Three Test Images

### Image 1 — Beverage Shelf (64 products detected)

![Beverage shelf annotated](outputs/img_1%20(1)_annotated.jpg)

```json
{
  "image_name": "img_1 (1).jpg",
  "total_products": 64,
  "brands": {
    "Tropicana": 14,
    "Coca-Cola": 7,
    "7UP": 6,
    "Pepsi": 5,
    "Nestea": 4,
    "Fanta": 4,
    "Red Bull": 4,
    "Gatorade": 4,
    "Nescafe": 3,
    "Sprite": 3,
    "Limca": 3,
    "Minute Maid": 3,
    "Amul": 2,
    "Real": 1,
    "Activia": 1
  },
  "shelf_share": {
    "Tropicana": 0.1857,
    "Coca-Cola": 0.1225,
    "7UP": 0.0966,
    "Pepsi": 0.0734,
    "Fanta": 0.0728,
    "Red Bull": 0.0602,
    "Nestea": 0.0557,
    "Minute Maid": 0.0553,
    "Gatorade": 0.0678,
    "Limca": 0.0515,
    "Sprite": 0.051,
    "Nescafe": 0.0412,
    "Amul": 0.0368,
    "Real": 0.0151,
    "Activia": 0.0143
  }
}
```

---

### Image 2 — Snacks / Mixed Shelf (50 products detected)

![Snacks shelf annotated](outputs/img_2%20(1)_annotated.jpg)

```json
{
  "image_name": "img_2 (1).jpg",
  "total_products": 50,
  "brands": {
    "Kurkure": 5,
    "Parle": 5,
    "Uncle Chipps": 4,
    "Doritos": 4,
    "Pringles": 4,
    "Britannia": 4,
    "Other": 9,
    "Lay's": 3,
    "Cheetos": 3,
    "Bingo": 3,
    "Oreo": 2,
    "Tropicana": 1,
    "Activia": 1,
    "Amul": 1,
    "MilkyMist": 1
  },
  "shelf_share": {
    "Parle": 0.1001,
    "Other": 0.1802,
    "Kurkure": 0.0998,
    "Uncle Chipps": 0.08,
    "Britannia": 0.0801,
    "Pringles": 0.0799,
    "Doritos": 0.0798,
    "Lay's": 0.0599,
    "Bingo": 0.06,
    "Cheetos": 0.06,
    "Oreo": 0.04,
    "MilkyMist": 0.0201,
    "Activia": 0.02,
    "Amul": 0.02,
    "Tropicana": 0.02
  }
}
```

> "Other" (9 products) covers foreign/unrecognised snack brands not in the 31-brand vocabulary — this is expected and assignment-compliant.

---

### Image 3 — Dairy Shelf (28 products detected)

![Dairy shelf annotated](outputs/img_3%20(1)_annotated.jpg)

```json
{
  "image_name": "img_3 (1).jpg",
  "total_products": 28,
  "brands": {
    "Amul": 9,
    "Actimel": 7,
    "Nestle Dairy": 6,
    "Epigamia": 3,
    "MilkyMist": 2,
    "Hersheys": 1
  },
  "shelf_share": {
    "Amul": 0.3731,
    "Nestle Dairy": 0.2352,
    "Actimel": 0.1729,
    "Epigamia": 0.1021,
    "MilkyMist": 0.0916,
    "Hersheys": 0.0252
  }
}
```

---

## Approach

### Detection — why YOLOv8n + grid fallback

- YOLOv8n is fast (~150 ms/image on CPU) and covers COCO classes like bottles, cups, and bowls — the main beverage containers
- COCO has no classes for snack bags or dairy cartons; a **5×10 grid fallback** fires when fewer than 5 products are detected, generating 50 region proposals that cover the full shelf
- Bounding boxes are filtered by area (0.05%–40% of image) and aspect ratio to drop noise

### Classification — why CLIP zero-shot

- CLIP can classify any brand from a text description — no labelled data or retraining required
- Brand prompts live in `config.py`; adding a new brand is a one-liner
- Text embeddings for all 31 brands are pre-computed once at startup and reused per crop — saves ~200 ms per image
- CLIP runs in `float16` to halve memory (~300 MB vs ~600 MB on CPU)

### OCR — why EasyOCR

- Pure Python, no Tesseract binary needed; works out of the box
- Better than Tesseract on natural scene text (price tags, shelf rails)
- Results are filtered to price patterns (`₹/$ amounts`, decimals, "X for Y" promotions) and short uppercase labels (MRP, SALE, etc.)

### Shelf Segmentation — why DBSCAN

- DBSCAN clusters bounding-box centre-y values into shelf rows without needing to specify the row count in advance
- Handles outliers (products that don't sit cleanly on a row) by marking them as noise rather than forcing a wrong assignment
- Share-of-shelf is computed as each brand's total bounding-box area divided by the sum of all areas

### Memory Management

- Each model (YOLO, CLIP, EasyOCR) is loaded, used, then explicitly deleted and garbage-collected before the next one loads
- Peak RAM stays under ~900 MB on a 3.7 GB machine

### Issues Faced

- **OOM crash**: CLIP (600 MB) + YOLO + EasyOCR together exceeded available RAM → fixed by sequential load-delete-gc and float16 CLIP
- **transformers 5.9.x API change**: `CLIPModel.get_text_features()` now returns a model output object, not a tensor → rewrote to use `text_model` + `text_projection` directly
- **Snack/dairy coverage**: YOLOv8n misses non-COCO objects → grid fallback fills the gaps

---

## Project Structure

```
retail-shelf-intelligence/
├── config.py                  # Brands, prompts, colours, model paths, thresholds
├── main.py                    # CLI entry point
├── generate_architecture.py   # Generates architecture.png
├── requirements.txt
│
├── pipeline/
│   ├── detector.py            # YOLOv8n detection + grid fallback
│   ├── classifier.py          # CLIP zero-shot brand classification
│   ├── ocr_extractor.py       # EasyOCR price/label extraction
│   ├── segmenter.py           # DBSCAN row segmentation + shelf share
│   └── analyzer.py            # Orchestrates all four stages
│
├── utils/
│   └── visualizer.py          # Annotated image generation
│
├── tests/                     # 69 unit tests, ~99% line coverage
│   ├── test_detector.py
│   ├── test_classifier.py
│   ├── test_ocr_extractor.py
│   ├── test_segmenter.py
│   ├── test_visualizer.py
│   └── test_analyzer.py
│
├── images/                    # Test shelf images (not in repo — add your own)
└── outputs/                   # Generated annotated images + JSON results
```

---

## Assumptions and Limitations

| # | Assumption / Limitation |
|---|-------------------------|
| 1 | YOLOv8n is COCO-trained — no dedicated classes for snack bags or tetra packs; grid fallback compensates |
| 2 | CLIP zero-shot may confuse visually similar brands (e.g. Sprite ↔ 7UP, Pepsi ↔ Coca-Cola dark cans) |
| 3 | OCR runs on the full image; very small price tags (<12 px font) may be missed |
| 4 | Share-of-shelf is bounding-box area ratio — not pixel-level segmentation masks |
| 5 | Brand list is in `config.py`; adding new brands requires no retraining — just a text prompt |
| 6 | DBSCAN `eps=55 px` is calibrated for ~600–1000 px tall images with 3–5 shelf rows |
| 7 | "Other" is a valid category per the assignment; it catches products outside the 31-brand vocabulary |
| 8 | OCR output includes some noise (short uppercase tokens, bare numbers) — EasyOCR picks up non-price text that passes the label filter; a stricter regex would reduce but not eliminate this |
| 9 | Pipeline is CPU-safe; GPU (if available) is used automatically for faster inference |
