# Waterfowl Detection — Faster R-CNN, YOLO-NAS, YOLOv5 & MegaDetector-Overhead

Four models share the same 512×512 COCO crops produced by `data_prep/`, so data
preparation only needs to be run once:

- **Faster R-CNN** (Detectron2) — `faster_rcnn/`
- **YOLO-NAS** (super-gradients) — `yolo_nas/`
- **YOLOv5** (Ultralytics) — `yolov5/`
- **MegaDetector-Overhead / OWL** (Microsoft AI for Good) — `megadetector_overhead/`
  — three variants via `--model`: **OWL-C** (DLA-34), **OWL-T** (DLA-34 + Swin),
  **OWL-D** (DINOv3 ViT-H+)

The first three are **box** detectors compared on a single protocol (mAP30, IoU = 0.30).
YOLOv5 mirrors the crops into the Ultralytics YOLO layout; MegaDetector-Overhead mirrors
them into a point-CSV layout (both built automatically from the same crops).

**MegaDetector-Overhead is different in kind** — it is a *point* detector (formerly "OWL",
Overhead Wildlife Locator): it localises each bird as a point, not a box. It therefore
cannot use the box-IoU mAP30 protocol directly, and it needs its own Python 3.11
environment (vendored DINOv3, incompatible with the detectron2/super-gradients/ultralytics
stack). It is scored two ways — a native point precision/recall/F1 and a bridging
"pseudo-box" mAP30 — and runs in a separate interpreter. See
[Running the Pipeline (MegaDetector-Overhead)](#running-the-pipeline-megadetector-overhead).

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
├── megadetector_overhead/      # MegaDetector-Overhead / OWL package (point detector)
│   ├── dataset.py              # Mirrors the COCO crops → OWL point-CSV layout (box centres)
│   ├── train.py                # Generates a Hydra config + drives the OWL repo's trainer
│   ├── _eval_owl.py            # Runs UNDER the OWL .venv: point metric + detections
│   ├── evaluate.py             # Pseudo-box mAP30 via the same COCOeval (IoU = 0.30)
│   └── main.py                 # Entry point (shells into the OWL Python 3.11 env)
│
├── third_party/                # Gitignored — cloned research repos
│   └── MegaDetector-Overhead/  # `git clone` + `uv sync` (.venv, vendored DINOv3, weights/)
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
├── mdo_data/                   # Created automatically — OWL point-CSV mirror of the crops
│   └── {train,val,test}/
│       ├── images/             # Symlinks back to crops/ (flat; no images duplicated)
│       └── gt.csv              # Point labels: images,x,y,labels (box centre per bird)
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
        ├── yolov5/                     # All YOLOv5 runs live here
        │   └── 2026-07-11_12-00-00/   # One timestamped folder per training run
        │       ├── weights/            # Ultralytics writes checkpoints here
        │       │   ├── best.pt         # Best checkpoint (by Ultralytics fitness)
        │       │   └── last.pt
        │       ├── run_info.json
        │       ├── results.csv         # Per-epoch Ultralytics metrics
        │       └── examples/           # Written by --eval
        └── mdo_owl_c/                  # MegaDetector-Overhead runs, one folder per model
            │                           # (mdo_owl_c / mdo_owl_t / mdo_owl_d — see --model)
            └── 2026-07-21_20-55-09/   # One timestamped folder per training run
                ├── weights/best.pth    # Best checkpoint (by val point-F1), stable handle
                ├── best_model.pth       # Raw checkpoint written by the OWL trainer
                ├── mdo_train_config.yaml # Generated Hydra config for this run
                ├── run_info.json
                └── eval/                # Written by --eval
                    ├── owl_eval.json    # Point metric + per-point detections
                    ├── metrics.json     # Point P/R/F1 + pseudo-box mAP30
                    └── ../examples/     # GT boxes | predicted points panels
```

Each dataset sub-folder contains:

- JPEG or PNG aerial images
- A `.txt` annotation file per image (`bird,x1,y1,x2,y2` per line)
- `image_info.csv` with a `bbox_split_Robert` column (`train` / `test`)

---

## Environment

The three **box** models run in a single conda environment, `waterfowl`. MegaDetector-
Overhead runs in its **own** Python 3.11 environment (`third_party/MegaDetector-Overhead/.venv`,
created by `uv`) — its vendored DINOv3 + geospatial stack cannot coexist with
detectron2/super-gradients/ultralytics. See
[Running the Pipeline (MegaDetector-Overhead)](#running-the-pipeline-megadetector-overhead)
for its setup.

| Package         | Version           | Used by       |
| --------------- | ----------------- | ------------- |
| Python          | 3.10.20           | box models    |
| PyTorch         | 2.1.2 + CUDA 12.1 | box models    |
| Detectron2      | 0.6               | Faster R-CNN  |
| super-gradients | 3.7.1             | YOLO-NAS      |
| ultralytics     | 8.4.82            | YOLOv5        |
| pycocotools     | 2.0.11            | all (mAP30)   |
| Pillow          | 12.2.0            | box models    |
| opencv-python   | 4.11.0.86         | all           |
| _(separate env)_ PyTorch | 2.5.1 + CUDA 12.1 | MegaDetector-Overhead |
| _(separate env)_ Python  | 3.11              | MegaDetector-Overhead |

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

## Running the Pipeline (MegaDetector-Overhead)

MegaDetector-Overhead / OWL is a **point** detector, so it does not fit the box-IoU
protocol the same way. Its entry point still follows the **same `--prepare / --train /
--eval`** interface and reuses the **same crops**, but it runs in a **separate Python
3.11 environment** and is scored two ways.

### One-time setup (separate environment + weights)

```bash
# 1. Clone the repo into third_party/ (gitignored)
mkdir -p third_party && cd third_party
git clone https://github.com/microsoft/MegaDetector-Overhead
cd MegaDetector-Overhead

# 2. Build its Python 3.11 GPU environment (installs torch 2.5.1+cu121, animaloc,
#    vendored DINOv3). Needs `uv` (curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync --no-default-groups --group gpu

# 3. Download the pretrained weights from Zenodo into weights/ — either per model via
#    the pipeline (recommended):
cd ../..
./megadetector_overhead/main.py --fetch-weights --model OWLC     # OWL-C.pth  (~216 MB)
./megadetector_overhead/main.py --fetch-weights --model OWLT     # OWL-T.pth  (~355 MB)
./megadetector_overhead/main.py --fetch-weights --model OWLD_H   # OWL-D.pth  (~3.5 GB)
#    or manually:
#    curl -L -o weights/OWL-C.pth "https://zenodo.org/records/20802844/files/OWL-C.pth?download=1"
```

Paths for all of the above live in `data_prep/config.py` (`MDO_REPO_DIR`, `MDO_PYTHON`,
`MDO_PRETRAINED`, `MDO_MODELS`). The main entry point runs under the `waterfowl` env like
the others and shells into `MDO_PYTHON` (`.venv/bin/python`) for training and inference.

### OWL model variants (`--model`)

Three OWL architectures are wired up. Each fine-tunes from its released overhead-benchmark
checkpoint (Zenodo [record 20802844](https://zenodo.org/records/20802844), **CC BY-NC-SA
4.0** — non-commercial) and writes to its **own** checkpoint folder, so runs never mix:

| `--model` | Architecture | Start checkpoint | Params (trainable) | Runs folder |
|---|---|---|---|---|
| `OWLC` *(default)* | HerdNet detection branch, DLA-34 | `OWL-C.pth` | 18M (18M) | `output/checkpoints/mdo_owl_c/` |
| `OWLT` | DLA-34 + Swin multiscale residual | `OWL-T.pth` | 30M (30M) | `output/checkpoints/mdo_owl_t/` |
| `OWLD_H` | DINOv3 ViT-H+/16 + DPT decoder | `OWL-D.pth` | 855M (15M) | `output/checkpoints/mdo_owl_d/` |

Notes:

- All three share the same data, losses, LMDS post-processing, and both evaluation metrics —
  only the network (and its start weights) differs.
- `OWLD_H`'s checkpoint bundles its **frozen** DINOv3 backbone, so it needs no separate
  (license-gated) Meta DINOv3 download; only the DPT decoder + head (~15M params) train.
  A training step at batch 8 peaks at ~8.3 GiB GPU memory, but the big backbone makes it
  markedly slower per epoch than OWL-C/T.
- The registry (`MDO_MODELS` in `data_prep/config.py`) defines each variant's constructor
  kwargs, start checkpoint, and folder; `megadetector_overhead/fetch_dinov3_weights.py`
  exists only for the advanced case of training an OWL-D variant from scratch.

### Run

```bash
# Step 1 — prepare: shared COCO crops + OWL point-CSV mirror (mdo_data/)
./megadetector_overhead/main.py --prepare

# Step 2 — fine-tune from pretrained weights (output/checkpoints/mdo_owl_{c,t,d}/<timestamp>/)
./megadetector_overhead/main.py --train                      # OWL-C (default)
./megadetector_overhead/main.py --train --model OWLT         # OWL-T
./megadetector_overhead/main.py --train --model OWLD_H       # OWL-D (ViT-H+)

# Step 3 — evaluate best checkpoint on the test set (point P/R/F1 + pseudo-box mAP30)
./megadetector_overhead/main.py --eval                       # latest OWL-C run
./megadetector_overhead/main.py --eval --model OWLT          # latest OWL-T run
./megadetector_overhead/main.py --eval --run 2026-07-21_20-55-09

# All-in-one
./megadetector_overhead/main.py --prepare --train --eval
```

- **Model**: selected with `--model` (see the variants table above; default `OWLC`),
  fine-tuned from its pretrained overhead-benchmark weights. Each box in the shared crops
  becomes one **point** at the box centre (`megadetector_overhead/dataset.py`).
- **Epochs**: 100 cap; the OWL trainer keeps the best checkpoint by **validation point-F1**
  and decays the LR on plateau (`auto_lr`) — this plays the role Faster R-CNN's
  early-stopping hook plays for the other models.
- **Learning rate**: 0.0005 (Adam), batch size 8, image size 512 — matched to YOLO-NAS/YOLOv5.
- **Two evaluation metrics** (`--eval` prints both and writes `eval/metrics.json`):
  1. **Point precision / recall / F1** — a prediction is a true positive when it lands
     within `MDO_POINT_RADIUS` of a ground-truth point (≈20 px full-res at `down_ratio` 2).
     This is the honest metric for a point model, computed by the OWL repo's own evaluator.
  2. **Pseudo-box mAP30** — each detected point is wrapped in an `MDO_PSEUDO_BOX`-px box and
     run through the **identical** `COCOeval`@IoU=0.30 as the other three, so OWL lands in
     the same table. Read it as an approximate bridge: a point has no real extent, so the
     absolute value depends on the (fixed) pseudo-box size.
- `--eval` also writes `N` example panels (`--examples N`, default 40; left = GT **boxes**,
  right = predicted **points**).
- OWL hyperparameters live in `data_prep/config.py` (`MDO_MODEL`, `MDO_MODELS`,
  `MDO_BATCH_SIZE`, `MDO_LR`, `MDO_POINT_RADIUS`, `MDO_PSEUDO_BOX`, `MDO_LMDS_*`, …).

> **Protocol caveats** specific to MegaDetector-Overhead:
>
> 1. **It is a point detector.** The point metric (1 above) is the faithful score; the
>    pseudo-box mAP30 (2) exists only to place it on the same axis as the box models and
>    is sensitive to `MDO_PSEUDO_BOX`. Don't read the two models' mAP30 as strictly
>    like-for-like.
> 2. **Separate environment.** Unlike the other three, its weights, deps, and interpreter
>    live under `third_party/MegaDetector-Overhead/` (all gitignored). The `waterfowl` env
>    is untouched.
> 3. **License.** The OWL pretrained weights are **CC-BY-NC-SA 4.0 (non-commercial)** —
>    stricter than the other models' weights; check before any commercial use.

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
examples of adding **box** models alongside `faster_rcnn/`, and `megadetector_overhead/`
is a worked example of a **point** model that needs its own environment and a second
(point-based) metric alongside the shared mAP30. To add another:

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
