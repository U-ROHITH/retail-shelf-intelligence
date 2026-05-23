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

# Grid fallback: used when YOLO finds fewer products than this threshold.
# 5 rows × 10 cols = 50 cells covers all three provided shelf images
# (img_2 snacks ~35 items, img_3 dairy ~45 items).
GRID_FALLBACK_THRESHOLD = 5
GRID_ROWS = 5
GRID_COLS = 10
GRID_VARIANCE_THRESHOLD = 100  # minimum pixel variance for a non-empty cell

# --- Brand Classification (CLIP) ---
CLIP_MODEL = "openai/clip-vit-base-patch32"

# All brands visible across the three provided shelf images.
# "Other" is the assignment-specified fallback for unidentifiable products.
BRANDS = [
    # ── Beverages – Coca-Cola family ──────────────────────────────────────
    "Coca-Cola",
    "Sprite",
    "Fanta",
    "Limca",
    "Minute Maid",
    # ── Beverages – PepsiCo family ────────────────────────────────────────
    "Pepsi",
    "7UP",
    "Tropicana",
    "Gatorade",
    # ── Beverages – others ────────────────────────────────────────────────
    "Red Bull",
    "Nestea",
    "Real",
    "Nescafe",
    # ── Snacks – PepsiCo / Frito-Lay ──────────────────────────────────────
    "Lay's",
    "Doritos",
    "Cheetos",
    "Kurkure",
    "Uncle Chipps",
    # ── Snacks – others ───────────────────────────────────────────────────
    "Bingo",
    "Pringles",
    # ── Biscuits / Cookies ────────────────────────────────────────────────
    "Britannia",
    "Parle",
    "Oreo",
    # ── Dairy ─────────────────────────────────────────────────────────────
    "Amul",
    "Activia",
    "Actimel",
    "Nestle Dairy",
    "Epigamia",
    "Hersheys",
    "MilkyMist",
    # ── Catch-all (assignment-specified category) ─────────────────────────
    "Other",
]

BRAND_PROMPTS: dict[str, str] = {
    # Coca-Cola family
    "Coca-Cola":    "a Coca-Cola bottle or can with red label and white Coca-Cola script logo",
    "Sprite":       "a Sprite green bottle with green and white Sprite label",
    "Fanta":        "a Fanta orange soft drink bottle with orange Fanta label",
    "Limca":        "a Limca lemon drink light-green bottle",
    "Minute Maid":  "a Minute Maid juice carton or bottle with blue and white label",
    # PepsiCo family
    "Pepsi":        "a Pepsi cola bottle or can with blue label and Pepsi globe logo",
    "7UP":          "a 7UP green bottle with 7UP logo and bubbles",
    "Tropicana":    "a Tropicana fruit juice tetra-pack carton with orange and tropical fruit imagery",
    "Gatorade":     "a Gatorade sports drink bottle with lightning bolt Gatorade logo",
    # Other beverages
    "Red Bull":     "a Red Bull energy drink slim silver and blue can with two red bulls logo",
    "Nestea":       "a Nestea iced tea bottle or can with Nestea logo",
    "Real":         "a Real fruit juice colorful carton with fruit imagery and Real logo",
    "Nescafe":      "a Nescafe coffee jar or bottle with red Nescafe logo",
    # PepsiCo snacks
    "Lay's":        "a Lay's potato chips bag in yellow with Lay's logo and smiling face",
    "Doritos":      "a Doritos tortilla chips bag with triangular chips and Doritos logo",
    "Cheetos":      "a Cheetos cheese puffs orange bag with Chester Cheetah mascot",
    "Kurkure":      "a Kurkure Indian spicy snack red and orange twisted snack pack",
    "Uncle Chipps": "a Uncle Chipps Indian potato chips bag with uncle character",
    # Other snacks
    "Bingo":        "a Bingo Indian snack bag with Bingo logo",
    "Pringles":     "a Pringles chips cylindrical tube canister with Pringles mustachio man",
    # Biscuits
    "Britannia":    "a Britannia biscuit pack — Good Day cookies, Marie Gold or Tiger biscuits",
    "Parle":        "a Parle biscuit pack — Monaco crackers or Parle-G glucose biscuits",
    "Oreo":         "a Oreo cream biscuit blue pack with white Oreo cookies logo",
    # Dairy
    "Amul":         "an Amul dairy product — milk bottle, butter block or cheese pack with Amul cartoon girl",
    "Activia":      "an Activia Danone yogurt white and green cup or bottle",
    "Actimel":      "an Actimel probiotic small yogurt drink bottle",
    "Nestle Dairy": "a Nestle dairy product — Nestle Dahi yogurt with blue and white Nestle label",
    "Epigamia":     "an Epigamia Greek yogurt white clean-design bottle or cup",
    "Hersheys":     "a Hersheys chocolate milk or syrup brown bottle with Hersheys logo",
    "MilkyMist":    "a MilkyMist dairy product — paneer, butter or cheese with blue MilkyMist label",
    # Fallback
    "Other":        "an unidentified retail consumer product on a store shelf",
}

# --- OCR ---
OCR_LANGUAGES = ["en"]
MIN_OCR_CONFIDENCE = 0.35

# --- Shelf Row Segmentation ---
ROW_CLUSTER_EPS = 55    # pixels: max y-distance to be in the same shelf row
ROW_CLUSTER_MIN_SAMPLES = 1

# --- Visualisation ---
# Each brand gets a distinct colour for bounding boxes and legend.
BRAND_COLORS: dict[str, tuple[int, int, int]] = {
    # Coca-Cola family — reds/greens
    "Coca-Cola":    (220,  30,  30),
    "Sprite":       ( 50, 200,  50),
    "Fanta":        (255, 140,   0),
    "Limca":        (100, 220, 120),
    "Minute Maid":  (255, 210,  50),
    # PepsiCo family — blues
    "Pepsi":        ( 30,  50, 210),
    "7UP":          (  0, 190,  80),
    "Tropicana":    (255, 140,  30),
    "Gatorade":     (  0, 200, 200),
    # Other beverages
    "Red Bull":     (200, 200,  20),
    "Nestea":       ( 60, 140,  60),
    "Real":         (200,  60, 160),
    "Nescafe":      (140,  60,  20),
    # PepsiCo snacks — yellows/oranges
    "Lay's":        (240, 215,   0),
    "Doritos":      (230,  90,   0),
    "Cheetos":      (255, 110,  30),
    "Kurkure":      (210,  50,  50),
    "Uncle Chipps": (240, 170,  30),
    # Other snacks
    "Bingo":        (255, 190,  50),
    "Pringles":     (170,  20,  20),
    # Biscuits — purples/magentas
    "Britannia":    (180,   0, 180),
    "Parle":        (200, 100, 200),
    "Oreo":         ( 40,  40,  40),
    # Dairy — greens/teals
    "Amul":         ( 30, 170,  60),
    "Activia":      (180, 220,  80),
    "Actimel":      (  0, 180, 120),
    "Nestle Dairy": ( 30, 130, 200),
    "Epigamia":     (160, 220, 180),
    "Hersheys":     (120,  60,  20),
    "MilkyMist":    ( 50, 160, 220),
    # Fallback
    "Other":        (140, 140, 140),
}
