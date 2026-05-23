from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"

# --- Detection ---
YOLO_MODEL = "yolov8n.pt"
DETECTION_CONFIDENCE = 0.20
# Size constraints relative to image area
MIN_BBOX_AREA_RATIO = 0.0005   # ignore noise smaller than 0.05% of image
MAX_BBOX_AREA_RATIO = 0.40     # ignore shelf-level regions larger than 40%
MAX_BBOX_WIDTH_RATIO = 0.65    # ignore full-width shelf strips
MAX_BBOX_HEIGHT_RATIO = 0.65   # ignore full-height objects

# Grid fallback: used when YOLO finds fewer than this many products
GRID_FALLBACK_THRESHOLD = 5
GRID_ROWS = 4
GRID_COLS = 8
GRID_VARIANCE_THRESHOLD = 120  # minimum pixel variance for a non-empty grid cell

# --- Brand Classification (CLIP) ---
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Brands tailored to the three provided shelf images
BRANDS = ["Coca-Cola", "Pepsi", "Lay's", "Doritos", "Amul", "Britannia", "Other"]

BRAND_PROMPTS = {
    "Coca-Cola": (
        "a photo of a Coca-Cola, Coke, Sprite, Fanta, or Limca "
        "beverage bottle or can on a store shelf"
    ),
    "Pepsi": (
        "a photo of a Pepsi cola, 7UP, or Mountain Dew "
        "bottle or can on a store shelf"
    ),
    "Lay's": "a photo of a Lay's potato chips yellow bag on a store shelf",
    "Doritos": (
        "a photo of a Doritos tortilla chips bag with triangular chip logo "
        "on a store shelf"
    ),
    "Amul": (
        "a photo of an Amul dairy product — milk bottle, butter, "
        "cheese, or yogurt — with the Amul cartoon girl logo"
    ),
    "Britannia": (
        "a photo of a Britannia biscuit or cookie package "
        "such as Good Day, Marie Gold, or Tiger on a store shelf"
    ),
    "Other": (
        "a photo of an unidentified consumer retail product on a store shelf"
    ),
}

# --- OCR ---
OCR_LANGUAGES = ["en"]
MIN_OCR_CONFIDENCE = 0.35

# --- Shelf Row Segmentation ---
ROW_CLUSTER_EPS = 55    # pixels: max y-distance to be in the same row
ROW_CLUSTER_MIN_SAMPLES = 1

# --- Visualisation ---
BRAND_COLORS: dict[str, tuple[int, int, int]] = {
    "Coca-Cola":  (220,  30,  30),   # red
    "Pepsi":      ( 30,  50, 200),   # blue
    "Lay's":      (230, 200,   0),   # yellow
    "Doritos":    (230, 100,   0),   # orange
    "Amul":       ( 30, 170,  60),   # green
    "Britannia":  (180,   0, 180),   # purple
    "Other":      (140, 140, 140),   # grey
}
