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

# ── MegaDetector-Overhead / OWL settings (Microsoft AI for Good, super-gradients-free) ──
# MegaDetector-Overhead (formerly "OWL", Overhead Wildlife Locator) is a *point-based*
# detector: it localises animals as points, not boxes.  It ships as a uv-managed
# research repo (Python 3.11, vendored DINOv3) that will not coexist with the
# detectron2 / super-gradients / ultralytics stack in the `waterfowl` env, so it lives
# in its own interpreter.  This package shells into that interpreter for train/eval.
#
# Layout:
#   third_party/MegaDetector-Overhead/         ← cloned repo (gitignored)
#   third_party/MegaDetector-Overhead/.venv/   ← `uv sync --group gpu` env (Python 3.11)
#   third_party/MegaDetector-Overhead/weights/OWL-C.pth  ← pretrained (Zenodo 20802844)
MDO_REPO_DIR   = os.path.join(_ROOT, "third_party", "MegaDetector-Overhead")
MDO_PYTHON     = os.path.join(MDO_REPO_DIR, ".venv", "bin", "python")
MDO_PRETRAINED = os.path.join(MDO_REPO_DIR, "weights", "OWL-C.pth")
# Legacy alias — per-model dirs live in MDO_MODELS below (mdo_owl_c / mdo_owl_t / mdo_owl_d).
MDO_CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints", "mdo_owl_c")
# Ultralytics-style flat mirror of the shared crops in the point-CSV layout OWL expects
# (flat image folder + gt.csv with images,x,y,labels).  Beside crops/, like yolov5_data.
MDO_DATA_DIR   = os.path.join(_ROOT, "mdo_data")

# Model (OWL-C = HerdNet detection branch, DLA-34).  OWL-T / OWL-D also exist and are
# selectable via the registry below (--model on the CLI); OWL-C is the default baseline
# and the only variant whose weights ship by default (weights/OWL-C.pth).
MDO_MODEL      = "OWLC"   # default selected variant; override with --model {OWLC,OWLT,OWLD_S}
MDO_NUM_LAYERS = 34
MDO_DOWN_RATIO = 2       # heatmap is input/down_ratio; detected points come out in that space
MDO_HEAD_CONV  = 64
MDO_BATCH_SIZE = 8       # DLA-34 is light; fits a 16 GB card comfortably at 512px
MDO_LR         = 0.0005  # Adam; matches YOLO-NAS / YOLOv5 for a fair comparison
MDO_EPOCHS     = NUM_EPOCHS  # same 100-epoch cap; best checkpoint kept by val F1 (+ auto-lr)

# ── OWL model registry (OWL-C / OWL-T / OWL-D) ────────────────────────────────
# Each OWL variant is a *different architecture* with different constructor kwargs and its
# own checkpoint folder, so runs never mix.  Two kwarg sets per model:
#   • kwargs      → build the model for TRAINING (bare arch; the fine-tuning start checkpoint
#                   named by `load_from` is then loaded on top).
#   • eval_kwargs → rebuild the bare architecture at EVAL time; the fine-tuned checkpoint is
#                   loaded on top.
#
# All three fine-tune from the *general overhead* benchmark checkpoints Microsoft released on
# Zenodo (record 20802844, CC BY-NC-SA 4.0) — the same idea for each: warm-start from the
# published weights, then fine-tune on the waterfowl crops.  `--fetch-weights` downloads the
# one named by `zenodo_file` to `requires_weights` (OWL-C.pth already ships).
#
#   OWL-C   DLA-34                        <- OWL-C.pth  (206 MB)
#   OWL-T   DLA-34 + Swin refinement      <- OWL-T.pth  (339 MB)
#   OWL-D   DINOv3 ViT-H+/16 + DPT head   <- OWL-D.pth  (3.3 GB, = OWLD_H)
#
# OWL-D note: OWL-D.pth is a *full-model* checkpoint that already contains its (frozen) ViT-H+
# backbone, so it loads with pretrained=False and needs NO separate (gated) Meta DINOv3
# download.  The gated DINOv3 backbones + fetch_dinov3_weights.py are only for training an
# OWL-D variant *from scratch* (not our path); the constants below remain for that use.
MDO_ZENODO_RECORD  = "20802844"
_ZENODO_FILE_URL   = "https://zenodo.org/api/records/{rec}/files/{name}/content"

MDO_OWLT_WEIGHTS   = os.path.join(MDO_REPO_DIR, "weights", "OWL-T.pth")
MDO_OWLD_WEIGHTS   = os.path.join(MDO_REPO_DIR, "weights", "OWL-D.pth")

MDO_DINOV3_ROOT    = os.path.join(MDO_REPO_DIR, "dinov3")   # importable root of vendored dinov3
MDO_DINOV3_WEIGHTS = os.path.join(
    MDO_REPO_DIR, "weights", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
# HuggingFace source for the gated DINOv3 ViT-S/16 backbone (fetch_dinov3_weights.py) — only
# needed to train an OWL-D variant from scratch.  Accept the license + authenticate first.
MDO_DINOV3_HF_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
MDO_DINOV3_HF_FILE = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

MDO_MODELS = {
    "OWLC": {
        "name": "OWLC",
        "ckpt_dir": os.path.join(OUTPUT_DIR, "checkpoints", "mdo_owl_c"),
        "load_from": MDO_PRETRAINED,
        "requires_weights": MDO_PRETRAINED,
        "zenodo_file": "OWL-C.pth",
        "kwargs": {"num_layers": MDO_NUM_LAYERS, "pretrained": False,
                   "down_ratio": MDO_DOWN_RATIO, "head_conv": MDO_HEAD_CONV},
        "eval_kwargs": {"num_layers": MDO_NUM_LAYERS, "pretrained": False,
                        "down_ratio": MDO_DOWN_RATIO, "head_conv": MDO_HEAD_CONV},
    },
    "OWLT": {
        "name": "OWLT",
        "ckpt_dir": os.path.join(OUTPUT_DIR, "checkpoints", "mdo_owl_t"),
        "load_from": MDO_OWLT_WEIGHTS,        # fine-tune from the released overhead OWL-T
        "requires_weights": MDO_OWLT_WEIGHTS,
        "zenodo_file": "OWL-T.pth",
        # pretrained_cnn=False: the full OWL-T checkpoint supplies DLA+Swin+head, so we skip
        # the ImageNet DLA download that would just be overwritten.
        "kwargs": {"num_layers": MDO_NUM_LAYERS, "pretrained_cnn": False,
                   "down_ratio": MDO_DOWN_RATIO, "head_conv": MDO_HEAD_CONV},
        "eval_kwargs": {"num_layers": MDO_NUM_LAYERS, "pretrained_cnn": False,
                        "down_ratio": MDO_DOWN_RATIO, "head_conv": MDO_HEAD_CONV},
    },
    "OWLD_H": {
        "name": "OWLD_H",
        "ckpt_dir": os.path.join(OUTPUT_DIR, "checkpoints", "mdo_owl_d"),
        "load_from": MDO_OWLD_WEIGHTS,        # released full checkpoint (backbone incl.)
        "requires_weights": MDO_OWLD_WEIGHTS,
        "zenodo_file": "OWL-D.pth",
        # ViT-H+/16 is heavy but the backbone stays frozen (as released), so only the DPT
        # decoder + head (~15M params) train and activations stay small: a measured training
        # step at batch=8 peaks at ~8.3 GiB on a 16 GB card. Same batch as the other models.
        "kwargs": {"pretrained": False, "dinov3_root": MDO_DINOV3_ROOT,
                   "down_ratio": MDO_DOWN_RATIO, "head_conv": MDO_HEAD_CONV,
                   "freeze_backbone": True, "unfreeze_last_n": 0},
        "eval_kwargs": {"pretrained": False, "dinov3_root": MDO_DINOV3_ROOT,
                        "down_ratio": MDO_DOWN_RATIO, "head_conv": MDO_HEAD_CONV,
                        "freeze_backbone": True, "unfreeze_last_n": 0},
    },
}


def mdo_zenodo_url(zenodo_file: str) -> str:
    """Direct download URL for a file in the OWL benchmark Zenodo record."""
    return _ZENODO_FILE_URL.format(rec=MDO_ZENODO_RECORD, name=zenodo_file)


def mdo_model_spec(key: str | None = None) -> dict:
    """Return the registry spec for an OWL variant (defaults to MDO_MODEL)."""
    key = key or MDO_MODEL
    if key not in MDO_MODELS:
        raise KeyError(f"Unknown OWL model '{key}'. Choices: {list(MDO_MODELS)}")
    return MDO_MODELS[key]

# Point-detection post-processing (LMDS: local-maxima detection over the heatmap).
MDO_LMDS_KERNEL   = 3     # local-maxima kernel (odd)
MDO_LMDS_ADAPT_TS = 0.3   # adaptive peak threshold (fraction of per-image max)
MDO_STITCH_OVERLAP = 160  # patch stitching overlap in px (repo default for 512 patches)

# Evaluation.  OWL is scored two ways (see megadetector_overhead/evaluate.py):
#   1) native point metric — a prediction is a TP if within MDO_POINT_RADIUS of a GT
#      point (in heatmap space; ×down_ratio ≈ full-res px).  Gives precision/recall/F1.
#   2) pseudo-box mAP30 — each detected point is wrapped in an MDO_PSEUDO_BOX-px box and
#      run through the *same* COCOeval@IoU=0.30 the other three models use, so it lands
#      in the same table (an approximation: a point model has no real extent).
MDO_POINT_RADIUS = 10    # TP match radius in heatmap px (down_ratio=2 → ≈20 px full-res)
MDO_PSEUDO_BOX   = 28    # side length (full-res px) of the box drawn around each point

# ── OWL-paper point metrics (Chacón et al., Section 4.3) ──────────────────────
# Shared protocol reported by every model's --eval (data_prep/point_metrics.py):
# MAE/RMSE per-image counting errors, AP / AUC-PR / P / R / F1 with greedy point
# matching, and bootstrap 95% CIs.  Box models participate via box centres.
PAPER_TAU        = 40    # TP matching radius in full-res px (paper: 40; sens. {20,40,60})
PAPER_BOOTSTRAP  = 1000  # image-level bootstrap resamples (paper: B = 1,000)

# ── Evaluation (Section 4.4) ──────────────────────────────────────────────────
IOU_THRESHOLD_MAP = 0.30     # mAP30: IoU threshold = 0.30
# Keep the *scoring* threshold low so the full precision/recall curve is available
# to COCOeval (this is what mAP integrates over — it must see low-conf boxes).
CONF_THRESHOLD    = 0.05     # used as SCORE_THRESH_TEST during mAP evaluation
# Threshold used only for the human-facing example panels written during --eval.
# This is what was causing the "lots of false alarms" look: drawing every box
# above 0.01.  Visualise only confident detections.
DISPLAY_CONF_THRESHOLD = 0.5
