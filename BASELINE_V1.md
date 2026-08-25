# Waterfowl Detection Baseline v1 — Frozen Specification

The eight-model benchmark, fixed as a reference point. Every value below was read from
the code and run artifacts that produced the committed results, not from intent or
recollection. Where the current repository has since drifted from what produced a result,
that is stated explicitly rather than smoothed over — see [Section 16](#16-drift-since-the-freeze).

**Frozen:** 24 August 2026 · 8 models · 125,746 crops · 88,195 test birds

---

## 1. Training / validation / test split

| Split | Crops | Source images |
|---|---|---|
| train | 75,664 | 1,170 |
| val | 24,374 | 389 |
| test | **25,708** | 396 |
| **total** | **125,746** | **1,955** |

Test set contains **88,195 annotated birds**.

**Derivation.** The test set is *not* chosen randomly. Each dataset ships an
`image_info.csv` in which every image is already marked as either **training** or
**testing** by the people who annotated the data; the pipeline reads that decision and
honours it. The images they marked for training are then shuffled with a fixed random
seed (42) and divided **75/25** into the training and validation sets.

Reproducing that shuffle requires the pool of images to be assembled in the same order
every time, which two things guarantee: datasets are visited in sorted name order, and
images within a dataset keep their original row order from the CSV.

**Datasets included (8).** Bird_A, Bird_D, Bird_E, Bird_F, Bird_G, Bird_H,
Bird_I (171 images), Bird_J.

**Datasets absent (2).** Bird_B and Bird_C — excluded not by choice but by a loader
defect. The loader guessed each image's label file by swapping the extension to `.txt`,
but these two datasets name their label files differently, so every image in both was
skipped without an error. 140 images / 3,138 birds. Fixed after the freeze.

**Known defect inside the freeze.** 30 Bird_I images use a 4-field annotation layout
(`x1,y1,x2,y2`, no `bird,` prefix) that the parser rejected. They entered the splits as
**2,781 crops asserted to contain no birds**, while in fact holding 2,646 birds —
1,393 in train, 820 in val, 433 in test. Baseline v1 therefore trains against ~2,200
crops of mislabelled background and scores 576 test crops as empty when they are not.
This is a property of the frozen baseline and must be cited whenever it is quoted.

**Authoritative artifact.** `output/crops_json/{train,val,test}/coco.json`. These files
*are* the split. Reproducing v1 means reading them, not re-deriving them.

## 2. Image crop size and overlap

| Parameter | Value |
|---|---|
| `CROP_SIZE` | 512 × 512 px |
| `CROP_OVERLAP` | 0.20 (20% total, 10% per side) |
| Stride | `int(512 × (1 − 0.20/2))` = **460 px** |
| Tiling | row-major, last tile clamped to the image edge |

Bird_E ships as 512×512 tiles already and is passed through without re-tiling.

## 3. Preprocessing

- Crops written as JPEG at **quality 95** into `crops/<Bird_X>/`
- Naming: `{Letter}_{Stem}_{row}_{col}.JPG`, row = y offset, col = x offset, unpadded
- A ground-truth box is kept in a crop when its intersection with that crop has positive
  area; coordinates are re-expressed relative to the crop origin. Boxes are **clipped, not
  dropped**, so a bird spanning a boundary appears in both crops — this is why the
  crop-level bird count (412,495) exceeds the image-level count.
- No resizing, colour conversion, or normalisation at crop time. Per-model normalisation
  happens inside each framework: Ultralytics and detectron2 apply their own defaults;
  OWL applies ImageNet mean/std via an albumentations `Normalize`.

## 4. Augmentation

| Model | Augmentation |
|---|---|
| YOLOv5m / YOLO11m / YOLO26m | Ultralytics defaults (mosaic, HSV jitter, scale, translate, fliplr). **Not overridden** — no augmentation arguments passed to `model.train()`. |
| YOLO-NAS-m | super-gradients recipe defaults |
| Faster R-CNN | detectron2 default train mapper (`ResizeShortestEdge` + `RandomFlip`). No custom augmentation. |
| OWL-C / OWL-T / OWL-D | `Normalize(ImageNet mean/std)` only — no geometric or photometric augmentation |

Augmentation was deliberately left at framework defaults rather than tuned per model, so
differences between models are architectural rather than recipe-driven. This is a
limitation as much as a control: the YOLO family receives substantially heavier
augmentation than OWL.

## 5. Model versions

| Model | Architecture | Framework |
|---|---|---|
| Faster R-CNN | ResNet-50 + FPN | detectron2 0.6 |
| YOLO-NAS-m | YOLO-NAS medium | super-gradients 3.7.1 |
| YOLOv5m | `yolov5mu` (anchor-free YOLOv5u) | ultralytics 8.4.82 |
| YOLO11m | `yolo11m` | ultralytics 8.4.82 |
| YOLO26m | `yolo26m`, end-to-end / NMS-free | ultralytics 8.4.82 |
| OWL-C | HerdNet detection branch, DLA-34 | animaloc (MegaDetector-Overhead) |
| OWL-T | DLA-34 + Swin refinement | animaloc |
| OWL-D | DINOv3 ViT-H+/16 + DPT head, backbone frozen | animaloc |

Environment: Python 3.10.20, PyTorch 2.1.2+cu121 for the five box models;
Python 3.11 + PyTorch 2.5.1+cu121 in a separate venv for the three OWL variants.

## 6. Pretrained checkpoints (starting weights)

| Model | Source | Pretraining |
|---|---|---|
| Faster R-CNN | `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` (detectron2 model zoo) | COCO |
| YOLO-NAS-m | `pretrained_weights="coco"` | COCO |
| YOLOv5m | `yolov5mu.pt` | COCO |
| YOLO11m | `yolo11m.pt` | COCO |
| YOLO26m | `yolo26m.pt` | COCO |
| OWL-C | `weights/OWL-C.pth`, SHA-256 `de97b6bb…` | general overhead wildlife, not caribou |
| OWL-T | `weights/OWL-T.pth` | general overhead wildlife |
| OWL-D | `weights/OWL-D.pth` (3.3 GB, includes frozen ViT-H+) | general overhead wildlife |

OWL weights are Zenodo record **20802844** (Microsoft AI for Good, CC BY-NC-SA 4.0).
All three load with `partial_load=False` — the full checkpoint fits the architecture.

## 7. Training hyperparameters

Common: **100 epochs max**, early-stop patience **30**, seed **42**, crop size **512**.

| Model | Optimiser | LR | Schedule | Batch | Best-checkpoint criterion |
|---|---|---|---|---|---|
| Faster R-CNN | SGD, momentum 0.9, wd 1e-4 | 0.001 | step ×0.1 at 70% / 90% of iters; warmup ≤1000 iters | 4 | val mAP30 |
| YOLO-NAS-m | Adam, wd 1e-4 | 0.0005 | cosine, final ratio 0.01, 3 warmup epochs | 8 | val mAP30 |
| YOLOv5m | Adam | 0.0005 | Ultralytics default | 8 | Ultralytics fitness |
| YOLO11m | Adam | 0.0005 | Ultralytics default | 8 | Ultralytics fitness |
| YOLO26m | Adam | 0.0005 | Ultralytics default | 8 | Ultralytics fitness |
| OWL-C/T/D | Adam | 0.0005 | plateau auto-LR, cooldown 10, min 1e-5 | 8 | val point-F1 |

Faster R-CNN RPN, retuned for 20–30 px birds: anchors `[8,16,32,64,128]` (default
`[32…512]`), aspect ratios `[0.5, 1.0, 2.0]`, batch/img 512, positive fraction 0.8,
IoU thresholds `[0.3, 0.5]`.

**Note.** YOLO26 ships a different default optimiser upstream; it was overridden to Adam
to match the rest. That is a fairness choice, not a tuned setting, and means YOLO26 is not
benchmarked at its intended best.

## 8. Fine-tuned checkpoints (the frozen artifacts)

| Model | Run directory | Weights file |
|---|---|---|
| Faster R-CNN | `2026-06-27_16-56-51` | `model_best.pth` |
| YOLO-NAS-m | `2026-06-28_21-14-03` | `RUN_*/ckpt_best.pth` |
| YOLOv5m | `2026-07-11_13-18-41` | `weights/best.pt` |
| YOLO11m | `2026-08-14_00-25-32` | `weights/best.pt` |
| YOLO26m | `2026-08-15_21-24-35` | `weights/best.pt` |
| OWL-C | `2026-07-21_22-59-50` | `weights/best.pth` |
| OWL-T | `2026-07-24_10-39-49` | `weights/best.pth` |
| OWL-D | `2026-07-26_18-56-20` | `weights/best.pth` |

Under `output/checkpoints/<model>/`. Weights are gitignored and mirrored to the Hugging
Face Hub (`juliamonson/waterfowl-uav-checkpoints`); the metrics, configs, curves and logs
beside them are committed.

Training is **bit-reproducible**: an independent re-run of YOLO26 reproduced its original
run exactly across every loss and metric column, AP agreeing to 16 significant figures
(`seed=42`, `deterministic=true`, same data and hardware).

## 9. Inference settings

| Setting | Value |
|---|---|
| Input size | 512 × 512 (native crop, no resize) |
| `MAX_DETECTIONS_PER_IMAGE` | **500** — one project-wide constant |
| NMS IoU | 0.7 (YOLOv5, YOLO11, YOLO-NAS); inert for YOLO26 (end-to-end) |
| YOLO-NAS NMS top-k | 1000 |
| Faster R-CNN RPN at test | pre-NMS 2000, post-NMS **500** |
| Ultralytics batching | **one image per `predict()` call** |
| OWL post-processing | LMDS local maxima, kernel 3, adaptive threshold 0.3, stitch overlap 160 px, down-ratio 2 |

The 500 limit is set above the densest crop (422 birds) so it cannot discard a real
detection. Faster R-CNN needs **both** its post-NMS detection budget and its RPN proposal
budget raised — the latter sits upstream and otherwise bounds detections on its own.

`bs=1` for Ultralytics is deliberate, not an oversight: Ultralytics NMS uses a cumulative
per-batch time budget, and dense crops trip it, silently returning zero detections for
every remaining image in the batch.

## 10. Confidence thresholds

| Purpose | Value |
|---|---|
| Scoring (`CONF_THRESHOLD`) | **0.05** |
| Counting (`t*`) | selected per model, see §12 |
| Visualisation only | 0.5 |

The scoring threshold is deliberately low so AP integrates the full precision–recall
curve. It is not an operating point.

## 11. Post-processing

- Box models: NMS at IoU 0.7, then truncation to 500 detections. YOLO26 is NMS-free.
- Box → point: detections become **box centres** `((x1+x2)/2, (y1+y2)/2)` for the point
  protocol. No other conversion.
- OWL: emits points natively. For the mAP30 bridge only, each point is wrapped in a
  **28 px square** pseudo-box. That number is arbitrary and the resulting mAP30 is *not*
  comparable to a true box mAP.
- No test-time augmentation, no model ensembling, no score calibration anywhere.

## 12. Point-matching protocol

Implemented once in `data_prep/point_metrics.py` and used by every model, so all eight are
scored by identical code.

1. Detections sorted by **descending score** within each image.
2. Each detection greedily matched to its **nearest unclaimed** ground-truth point.
3. A match counts as a true positive when the distance is **≤ τ**. Matching is
   **one-to-one**: a GT point, once claimed, cannot be matched again.
4. Unmatched detections are false positives; unmatched GT are false negatives.
5. Background-only crops **must** be present in the GT dict with an empty list, so their
   false positives are counted.

**Threshold selection.** A 256-point score grid is built from the empirical score
quantiles. `t*` is chosen as **argmin MAE** over that grid — i.e. the counting-optimal
threshold, not the F1-optimal one. Precision, recall, F1 and every counting figure are
reported at that same `t*`.

## 13. Matching distance threshold

**τ = 40 px**, at full crop resolution, matching the OWL paper. Applied identically to all
models. The paper's sensitivity range is {20, 40, 60} px; only 40 is frozen here.

For OWL's *native* point metric (reported separately from the shared protocol), the radius
is 10 px in heatmap space at down-ratio 2, ≈ 20 px full-resolution.

## 14. Bootstrap procedure

- **B = 1,000** resamples, **image-level**, seed **42** (`np.random.default_rng`)
- Implemented as multinomial weights over the N test images rather than index resampling
- Percentile method: **2.5th and 97.5th** percentiles → 95% CI
- CIs reported for **MAE, RMSE and AP**
- `t*` is held **fixed** at its full-sample value across resamples, as the paper prescribes
  — the CI reflects sampling variability, not threshold re-selection

## 15. Evaluation metrics

**Primary — OWL-paper point protocol** (all eight models, identical code):

| Metric | Definition |
|---|---|
| AP | area under the precision–recall curve over the full score sweep |
| AUC-PR | trapezoidal AUC of the same curve |
| Precision / Recall / F1 | at `t*` |
| MAE | mean absolute per-image count error at `t*` |
| RMSE | root mean square per-image count error at `t*` |
| Pred. (Err %) | total predicted count and signed % error vs GT |

**Secondary — mAP30**: pycocotools COCOeval at a single IoU of **0.30**, AP/AR read from
the accumulated arrays at area range "all". For OWL this is the 28 px pseudo-box bridge.

> **mAP30 carries its own cap.** It reads pycocotools' **100-detections-per-image** slot,
> independent of `MAX_DETECTIONS_PER_IMAGE`. On crops holding 400+ birds this understates
> it. The limit applies equally to every box model, so the comparison is internally fair,
> but mAP30 is not an uncapped metric.

## 16. Drift since the freeze

The repository has moved on. These differences mean re-running the pipeline today will
**not** reproduce Baseline v1:

| Change | Effect on reproduction |
|---|---|
| `DATASET_VERSION = "full"` | `--prepare` reads 10 datasets, not 8 |
| Bird_B / Bird_C loader fix | +140 images, +3,138 birds enter the pool |
| Bird_I 4-field parser fix | +2,646 boxes, including on 6 test images frozen as empty |
| Bird_I folder renamed | regenerated crops land under a different directory name than the frozen ones |
| Detection limit 300→500, RPN 300→500, detectron2 100→500 | all eight models re-evaluated after the freeze |

Adding datasets changes the seeded shuffle for *every* dataset, so train/val membership
shifts corpus-wide and the test set grows from 396 to 426 images.

**To reproduce Baseline v1:** use the committed `output/crops_json/*/coco.json` and the
checkpoints in §8. Do **not** re-run `--prepare`. The reported metrics were regenerated at
the final inference settings in §9, so the committed `eval_results.txt` and
`paper_point_metrics.json` are self-consistent with this document.

**To move beyond it:** re-run `--prepare` on `dataset_full`, retrain all eight models, and
freeze the result as Baseline v2. The two are not comparable metric-for-metric.
