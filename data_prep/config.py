import os

# ── Root paths ────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(_ROOT, "dataset")
OUTPUT_DIR   = os.path.join(_ROOT, "output")
CKPT_DIR     = os.path.join(OUTPUT_DIR, "checkpoints", "fasterrcnn")
YOLONAS_CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints", "yolonas")
YOLOV5_CKPT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints", "yolov5")

# Cropped images live outside output/ and are organised by dataset name:
#   crops/Bird_A/Bird_A_DJI_0001_000_000.jpg
#   crops/Bird_G/Bird_G_...jpg  etc.
# COCO JSON split files (no images) stay under output/.
CROPS_IMG_DIR  = os.path.join(_ROOT, "crops")          # root for per-dataset image folders
CROPS_JSON_DIR = os.path.join(OUTPUT_DIR, "crops_json") # train/val/test coco.json files

# ── Datasets to use ───────────────────────────────────────────────────────────
# All sub-folders of DATASET_ROOT that contain images.
# Bird_I is the incomplete version of Bird_I_complete — exclude it.
EXCLUDED_DATASETS = {"Bird_I"}

# ── Image cropping (Section 4.3) ──────────────────────────────────────────────
CROP_SIZE    = 512
CROP_OVERLAP = 0.20     # 20% total overlap for middle crops (10% each side) → stride = 460 px

# ── Train / val / test split ──────────────────────────────────────────────────
# The CSVs provide train/test via 'bbox_split_Robert'.
# We further split the CSV-train portion into train (75%) and val (25%)
# so the final ratio is ≈ 60 / 20 / 20.
VAL_FRACTION = 0.25
RANDOM_SEED  = 42

# ── Faster R-CNN anchor settings (Section 4.1) ────────────────────────────────
# Paper reduced default [32,64,128,256,512] → [8,16,32,64,128] for small birds
ANCHOR_SIZES         = ((8,), (16,), (32,), (64,), (128,))
ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5

# ── RPN settings (Section 4.1) ────────────────────────────────────────────────
RPN_FG_IOU_THRESH        = 0.5   # lowered from 0.7; birds are 20-30px, max anchor IoU ~0.50
RPN_BG_IOU_THRESH        = 0.3
RPN_BATCH_SIZE_PER_IMAGE = 512   # paper: 256 → 512
RPN_POSITIVE_FRACTION    = 0.8   # paper: 0.5 → 0.8
RPN_PRE_NMS_TOP_N_TRAIN  = 2000
RPN_POST_NMS_TOP_N_TRAIN = 1000
RPN_PRE_NMS_TOP_N_TEST   = 1000
RPN_POST_NMS_TOP_N_TEST  = 300

# ── Training (Section 4.1) ────────────────────────────────────────────────────
NUM_EPOCHS          = 100   # paper cap; early stopping almost always ends sooner
EARLY_STOP_PATIENCE = 30     # paper Section 4.1: tolerance = 30 epochs
LEARNING_RATE       = 0.001
BATCH_SIZE          = 4
NUM_WORKERS         = 4
NUM_CLASSES         = 2      # background (0) + bird (1)

# ── YOLO-NAS settings (super-gradients) ───────────────────────────────────────
# Variant: one of "yolo_nas_s", "yolo_nas_m", "yolo_nas_l".
YOLONAS_MODEL       = "yolo_nas_m"
YOLONAS_BATCH_SIZE  = 8        # YOLO-NAS is lighter than Faster R-CNN; larger batch is fine
YOLONAS_LR          = 0.0005   # cosine-decayed; tuned for Adam + small birds
# Post-processing for mAP evaluation (mirrors Faster R-CNN's low score threshold so
# COCO-style AP integrates the full PR curve).
YOLONAS_NMS_IOU     = 0.7
YOLONAS_NMS_TOPK    = 1000
YOLONAS_MAX_PRED    = 300

# ── YOLOv5 settings (Ultralytics) ─────────────────────────────────────────────
# Variant: one of "yolov5su", "yolov5mu", "yolov5lu" (Ultralytics' anchor-free
# YOLOv5u models — the actively-maintained successors to the original anchor-based
# YOLOv5s/m/l).  Ultralytics auto-downloads "<name>.pt" from its GitHub release
# assets on first use.
YOLOV5_MODEL        = "yolov5mu"
YOLOV5_BATCH_SIZE   = 8
YOLOV5_LR           = 0.0005   # ultralytics lr0; Adam, tuned for small birds (matches YOLO-NAS)
# Ultralytics-format dataset (YOLO txt labels + symlinked images) is mirrored here
# from the shared COCO crops so training never touches the originals.  Kept beside
# `crops/` (not under output/) since it is a sizeable derived dataset, not a run artifact.
YOLOV5_DATA_DIR     = os.path.join(_ROOT, "yolov5_data")
YOLOV5_NMS_IOU      = 0.7      # NMS IoU during inference (matches YOLONAS_NMS_IOU)
YOLOV5_MAX_PRED     = 300      # max detections per image (matches YOLONAS_MAX_PRED)

# ── Evaluation (Section 4.4) ──────────────────────────────────────────────────
IOU_THRESHOLD_MAP = 0.30     # mAP30: IoU threshold = 0.30
# Keep the *scoring* threshold low so the full precision/recall curve is available
# to COCOeval (this is what mAP integrates over — it must see low-conf boxes).
CONF_THRESHOLD    = 0.05     # used as SCORE_THRESH_TEST during mAP evaluation
# Threshold used only for the human-facing example panels written during --eval.
# This is what was causing the "lots of false alarms" look: drawing every box
# above 0.01.  Visualise only confident detections.
DISPLAY_CONF_THRESHOLD = 0.5
