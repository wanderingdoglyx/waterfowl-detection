# Waterfowl Detection — Faster R-CNN, YOLO-NAS & YOLOv5

Three models share the same data pipeline and evaluation protocol (mAP30, IoU = 0.30):

- **Faster R-CNN** (Detectron2) — `faster_rcnn/`
- **YOLO-NAS** (super-gradients) — `yolo_nas/`
- **YOLOv5** (Ultralytics) — `yolov5/`

All three consume the identical 512×512 COCO crops produced by `data_prep/`, so data
preparation only needs to be run once. YOLOv5 additionally mirrors those crops into
the Ultralytics YOLO layout (see below), built automatically from the same crops.

---

## Project Layout

```
rebuild/
├── dataset/                    # Raw aerial-image datasets (read-only)
│   ├── Bird_A/
│   ├── Bird_B/
│   ├── Bird_C/
│   ├── Bird_D/
│   ├── Bird_E/
│   ├── Bird_F/
│   ├── Bird_G/
│   ├── Bird_H/
│   ├── Bird_I_complete/        # Bird_I is excluded (incomplete); use this instead
│   └── Bird_J/
│
├── data_prep/                  # Shared data utilities (reusable by future models)
│   ├── config.py               # All hyperparameters (anchors, RPN, LR, crop size…)
│   └── prepare_data.py         # Crops images → 512×512 patches + COCO JSON
│
├── faster_rcnn/                # Faster R-CNN model package (Detectron2)
│   ├── dataset.py              # Registers COCO splits with Detectron2
│   ├── model.py                # Builds Detectron2 CfgNode (ResNet-50 + FPN)
│   ├── evaluator.py            # Custom MAP30Evaluator (IoU threshold = 0.30)
│   ├── train.py                # WaterfowlTrainer + EarlyStoppingHook
│   └── main.py                 # Entry point for this model
│
├── yolo_nas/                   # YOLO-NAS model package (super-gradients)
│   ├── dataset.py              # super-gradients dataloaders from the same COCO crops
│   ├── model.py                # YOLO-NAS model, PPYoloELoss, mAP30 metric, train params
│   ├── train.py                # Trainer wrapper + per-run checkpoints
│   └── main.py                 # Entry point for this model
│
├── yolov5/                     # YOLOv5 model package (Ultralytics)
│   ├── dataset.py              # Mirrors the COCO crops → Ultralytics YOLO layout
│   ├── evaluate.py             # mAP30 via pycocotools COCOeval (IoU = 0.30)
│   ├── train.py                # Ultralytics YOLO.train() wrapper + per-run checkpoints
│   └── main.py                 # Entry point for this model
│
├── crops/                      # Created automatically — crop images, organised by dataset
│   ├── Bird_A/
│   │   ├── A_DJI_0001_0_0.JPG
│   │   ├── A_DJI_0001_0_460.JPG
│   │   └── ...
│   ├── Bird_B/
│   └── ...  (one sub-folder per dataset)
│
├── yolov5_data/                # Created automatically — Ultralytics YOLO mirror of the crops
│   ├── data.yaml               # Dataset descriptor consumed by Ultralytics
│   ├── images/{train,val,test}/  # Symlinks back to crops/ (no images duplicated)
│   └── labels/{train,val,test}/  # YOLO txt labels (one "0 cx cy w h" line per box)
│
└── output/                     # Created automatically — model artefacts only
    ├── crops_json/             # COCO annotation files (no images)
    │   ├── train/
    │   │   ├── coco.json               # Full training split (all datasets)
    │   │   └── coco_Bird_A_Bird_B.json # Filtered split (generated on demand)
    │   ├── val/
    │   │   ├── coco.json
    │   │   └── coco_Bird_A_Bird_B.json
    │   └── test/coco.json              # Test split is never filtered
    └── checkpoints/
        ├── fasterrcnn/                 # All Faster R-CNN runs live here
        │   └── 2026-06-23_14-36-59/   # One timestamped folder per training run
        │       ├── model_best.pth      # Best checkpoint (by val mAP30)
        │       ├── model_final.pth
        │       ├── metrics.json
        │       └── test_inference/     # Written by --eval
        ├── yolonas/                    # All YOLO-NAS runs live here
        │   └── 2026-06-28_10-30-00/   # One timestamped folder per training run
        │       └── RUN_<timestamp>/    # super-gradients nests checkpoints one level down
        │           ├── ckpt_best.pth   # Best checkpoint (by val mAP30)
        │           ├── ckpt_latest.pth
        │           └── average_model.pth
        └── yolov5/                     # All YOLOv5 runs live here
            └── 2026-07-11_12-00-00/   # One timestamped folder per training run
                ├── weights/            # Ultralytics writes checkpoints here
                │   ├── best.pt         # Best checkpoint (by Ultralytics fitness)
                │   └── last.pt
                ├── run_info.json
                ├── results.csv         # Per-epoch Ultralytics metrics
                └── examples/           # Written by --eval
```

Each dataset sub-folder contains:

- JPEG or PNG aerial images
- A `.txt` annotation file per image (`bird,x1,y1,x2,y2` per line)
- `image_info.csv` with a `bbox_split_Robert` column (`train` / `test`)

---

## Environment

All three models run in a single conda environment, `waterfowl`:

| Package         | Version           | Used by       |
| --------------- | ----------------- | ------------- |
| Python          | 3.10.20           | all           |
| PyTorch         | 2.1.2 + CUDA 12.1 | all           |
| Detectron2      | 0.6               | Faster R-CNN  |
| super-gradients | 3.7.1             | YOLO-NAS      |
| ultralytics     | 8.4.82            | YOLOv5        |
| pycocotools     | 2.0.11            | all (mAP30)   |
| Pillow          | 12.2.0            | all           |
| opencv-python   | 4.11.0.86         | all           |

Create and populate it with:

```bash
conda create -n waterfowl python=3.10 -y
conda activate waterfowl
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install super-gradients ultralytics pycocotools pillow opencv-python
# Detectron2 (for Faster R-CNN) — see the official install matrix for your torch/CUDA:
#   https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

Notes:
- **YOLOv5 (Ultralytics)** auto-downloads its COCO-pretrained weights (`yolov5mu.pt`,
  etc.) from GitHub on first use — no manual download required.
- **YOLO-NAS (super-gradients)** pretrained weights are normally fetched from Deci's
  model hub, which is offline; place `yolo_nas_s_coco.pth` in
  `~/.cache/torch/hub/checkpoints/` (mirror: `https://d2gjn4b69gu75n.cloudfront.net/models/`)
  so the download is skipped.

---

## Running the Pipeline (Faster R-CNN)

All commands are run from the `rebuild/` directory.

### Step 1 — Prepare data (one-time)

Crops all datasets into 512×512 patches with 20% overlap (stride = 460 px) and writes
COCO-format JSON files for train / val / test splits.

```bash
./faster_rcnn/main.py --prepare
```

- Images are split **train / val / test ≈ 60 / 20 / 20** using the
  `bbox_split_Robert` column from each dataset's `image_info.csv`.
  The CSV train set is further divided 75 / 25 → train / val.
- Bird_I is excluded; Bird_I_complete is used instead.
- Only needs to be run once. The resulting COCO JSONs are reused by all training runs.

### Step 2 — Train

```bash
# Train on all datasets
./faster_rcnn/main.py --train

# Train on a subset of datasets by subfolder name
./faster_rcnn/main.py --train --datasets Bird_A Bird_B Bird_G
```

- **Backbone**: ResNet-50 + FPN, initialised from COCO-pretrained Faster R-CNN weights (`faster_rcnn_R_50_FPN_3x` from Detectron2 model zoo)
- **Anchor sizes**: 8, 16, 32, 64, 128 px
- **Epochs**: 100 (early stop if val mAP30 stalls for 30 consecutive evaluations)
- **Learning rate**: 0.001 (SGD + cosine decay)
- **Batch size**: 4
- Each run creates a new timestamped folder under `output/checkpoints/fasterrcnn/`
  (e.g. `2026-06-24_09-00-00/`). Previous runs are never overwritten.
- When `--datasets` is used, a filtered COCO JSON is written alongside the original
  (e.g. `coco_Bird_A_Bird_B_Bird_G.json`). The original `coco.json` is not modified.
  The test split is always the full set for a fair final evaluation.

### Step 3 — Evaluate on test set

```bash
# Evaluate the most recent training run
./faster_rcnn/main.py --eval

# Evaluate a specific run by its timestamp
./faster_rcnn/main.py --eval --run 2026-06-23_14-36-59
```

Loads `model_best.pth` from the selected run and reports **mAP30** (mean Average
Precision at IoU threshold = 0.30, matching the paper's metric).

### All steps in one command

```bash
./faster_rcnn/main.py --prepare --train --eval
```

---

## Running the Pipeline (YOLO-NAS)

The YOLO-NAS entry point follows the **same `--prepare / --train / --eval`**
interface and reuses the **same crops** as Faster R-CNN. If you already ran
`--prepare` for Faster R-CNN, skip straight to `--train`.

```bash
# Step 1 — prepare data (shared with Faster R-CNN; only needed once)
./yolo_nas/main.py --prepare

# Step 2 — train (saved to output/checkpoints/yolonas/<timestamp>/)
./yolo_nas/main.py --train

# Step 3 — evaluate best checkpoint on the test set
./yolo_nas/main.py --eval                       # latest run
./yolo_nas/main.py --eval --run 2026-06-28_10-30-00

# All-in-one
./yolo_nas/main.py --prepare --train --eval
```

- **Model**: YOLO-NAS (`yolo_nas_s` by default), initialised from COCO-pretrained weights
- **Epochs**: 100 (early stop if val mAP30 stalls for 30 consecutive epochs)
- **Learning rate**: 0.0005 (Adam + cosine decay), batch size 8
- **Evaluation**: mAP30 (IoU = 0.30), identical protocol to Faster R-CNN
- `--eval` also writes `N` side-by-side example panels (`--examples N`, default 40;
  left = ground truth, right = predictions). Each run gets its own timestamped folder
  under `output/checkpoints/yolonas/`.
- The variant and YOLO-NAS hyperparameters live in `data_prep/config.py`
  (`YOLONAS_MODEL`, `YOLONAS_BATCH_SIZE`, `YOLONAS_LR`, …).

---

## Running the Pipeline (YOLOv5)

The YOLOv5 entry point follows the **same `--prepare / --train / --eval`** interface
and reuses the **same crops** as the other models. Unlike Faster R-CNN / YOLO-NAS,
Ultralytics cannot read COCO JSON directly, so `--prepare` (and `--train`) also builds
a YOLO-format mirror of the crops under `yolov5_data/` (symlinked images + txt labels

+ `data.yaml`). This is automatic and idempotent — no images are duplicated.

```bash
# Step 1 — prepare data: shared COCO crops + Ultralytics YOLO mirror
./yolov5/main.py --prepare

# Step 2 — train (saved to output/checkpoints/yolov5/<timestamp>/)
./yolov5/main.py --train

# Step 3 — evaluate best checkpoint on the test set (mAP30)
./yolov5/main.py --eval                       # latest run
./yolov5/main.py --eval --run 2026-07-11_12-00-00

# All-in-one
./yolov5/main.py --prepare --train --eval
```

- **Model**: YOLOv5 (`yolov5mu` by default — Ultralytics' anchor-free YOLOv5u variant),
  initialised from COCO-pretrained weights that Ultralytics auto-downloads.
- **Epochs**: 100 (early stopping, patience 30)
- **Learning rate**: 0.0005 (Adam), batch size 8, image size 512
- **Evaluation**: mAP30 (IoU = 0.30) computed by `yolov5/evaluate.py` with pycocotools
  `COCOeval`, the **identical protocol** to Faster R-CNN and YOLO-NAS.
- `--eval` also writes `N` side-by-side example panels (`--examples N`, default 40;
  left = ground truth, right = predictions). Each run gets its own timestamped folder
  under `output/checkpoints/yolov5/`, with weights in `weights/best.pt`.
- The variant and YOLOv5 hyperparameters live in `data_prep/config.py`
  (`YOLOV5_MODEL`, `YOLOV5_BATCH_SIZE`, `YOLOV5_LR`, `YOLOV5_DATA_DIR`, …).

> **Two protocol caveats** specific to YOLOv5:
>
> 1. **Early stopping watches Ultralytics' internal fitness** (a blend of mAP@0.50 and
>    mAP@0.50:0.95), not mAP30 — Ultralytics does not expose a single-IoU monitor.
>    Only the **final** test report (`--eval`) is computed at mAP30. Faster R-CNN and
>    YOLO-NAS watch mAP30 directly during training.
> 2. **mAP30 uses pycocotools' default 100-detection-per-image cap.** Some crops hold
>    400+ birds, so this slightly understates recall on the densest scenes — but the
>    same cap applies to the Faster R-CNN evaluator, so the comparison stays fair.

---

## Key Hyperparameters (from the paper, Section 4.1)

| Parameter             | Paper value                                   | Location                |
| --------------------- | --------------------------------------------- | ----------------------- |
| Anchor sizes          | [8, 16, 32, 64, 128]                          | `data_prep/config.py` |
| RPN batch size        | 512                                           | `data_prep/config.py` |
| RPN positive fraction | 0.8                                           | `data_prep/config.py` |
| Training epochs       | 100                                           | `data_prep/config.py` |
| Early-stop patience   | 30                                            | `data_prep/config.py` |
| Learning rate         | 0.001                                         | `data_prep/config.py` |
| Batch size            | 4                                             | `data_prep/config.py` |
| Crop size             | 512 × 512 px                                 | `data_prep/config.py` |
| Crop overlap          | 20% total / 52 px each side (stride = 460 px) | `data_prep/config.py` |
| Evaluation IoU        | 0.30 (mAP30)                                  | `data_prep/config.py` |

---

## Adding a New Model

The `data_prep/` package is model-agnostic — `yolo_nas/` and `yolov5/` are worked
examples of adding models alongside `faster_rcnn/`. To add another:

1. Create a new package folder (e.g. `retinanet/`)
2. Import `data_prep.config` and `data_prep.prepare_data` as needed
3. Reuse the same COCO JSON files in `output/crops_json/` — no need to re-run `--prepare`.
   If the framework can't read COCO directly (as with Ultralytics), add a converter
   that mirrors the crops into its expected layout, like `yolov5/dataset.py`
4. Add a `main.py` entry point following the same `--prepare / --train / --eval` pattern
5. Add a `<MODEL>_CKPT_DIR` to `data_prep/config.py` so checkpoints land under
   `output/checkpoints/<model>/`, one timestamped folder per run
6. Evaluate with the same mAP30 protocol (IoU = 0.30) for fair comparison across models.
   `yolov5/evaluate.py` is a reusable pycocotools implementation if the framework
   doesn't compute mAP at a single IoU threshold natively
