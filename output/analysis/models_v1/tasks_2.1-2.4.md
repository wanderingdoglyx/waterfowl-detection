# Benchmark analysis — Tasks 2.1-2.4

Manifest: `models_v1.docx`. All models scored on the current test split: 28,588 crops, 89,884 annotated birds. Accuracy is recomputed here from cached predictions rather than read from each run's stored JSON, so it describes the same images as the size and density breakdowns.

## Task 2.1 — Efficiency

### Measurement environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER (15.6 GB, SM 8.9) |
| CPU / RAM | 32 cores, 62.5 GB |
| Platform | Linux-6.8.0-90-generic-x86_64-with-glibc2.35 |
| CUDA / driver | 12.1 / 570.195.03 |
| Python | 3.10.20 |
| Libraries | cv2 4.11.0, detectron2 0.6, numpy 1.23.0, pycocotools unknown, super_gradients 3.7.1, torch 2.1.2+cu121, ultralytics 8.4.82 |
| Code revision | 9484535 (working tree dirty) |
| Benchmark run | `2026-08-25_00-44-06` |

Timing protocol: 200 crops per pass, 3 repeats after 10 warm-up crops, reported as the mean (conf 0.05, max_det 500, crop 512 px). Every timed region is bracketed by `torch.cuda.synchronize()`.

All runs in this manifest trained on **NVIDIA GeForce RTX 4080 SUPER** (`cengr-dp6smw3.engineering.missouri.edu`).

> Training hardware was not recorded for Faster R-CNN, OWL-C, OWL-D, OWL-T, YOLO26, YOLO-NAS, YOLOv5 — those runs predate hardware capture, or were launched with an interpreter that could not identify the GPU.

### Required metrics

| Model | Params (M) | Checkpoint (MB) | Peak GPU (MB) | ms / 512×512 crop | Throughput (crops/s) |
|---|---|---|---|---|---|
| OWL-C | 18.4 | 206 | 803 | 3.38 ± 0.04 | 296 |
| YOLOv5 | 25.0 | 48 | 505 | 4.18 ± 0.01 | 240 |
| YOLO11 | 20.0 | 39 | 631 | 4.32 ± 0.01 | 232 |
| YOLO26 | 20.4 | 42 | 814 | 4.73 ± 0.03 | 212 |
| OWL-T | 29.9 | 339 | 896 | 6.88 ± 0.01 | 145 |
| Faster R-CNN | 41.3 | 158 | 332 | 9.09 ± 0.15 | 110 |
| YOLO-NAS | 51.1 | 650 | 459 | 15.77 ± 0.18 | 63 |
| OWL-D | 855.5 | 3377 | 4249 | 100.39 ± 0.36 | 10 |

### Recommended metrics

| Model | Preprocess (ms) | Postprocess (ms) | Training (h) | Peak training GPU (MB) |
|---|---|---|---|---|
| YOLO11 | 0.324 | 0.210 | — | — |
| YOLO26 | 0.326 | 0.095 | — | — |
| YOLOv5 | 0.331 | 0.220 | — | — |

#### Batch-size sensitivity (crops/s)

Measured over 256 crops per batch size, warmed up at each shape — cuDNN retunes per input shape, so a shared warm-up would flatter whichever size ran first. Evaluation itself deliberately runs at batch 1 for correctness, so this is headroom the eval path forgoes, not throughput it achieves.

| Model | bs 1 | bs 2 | bs 4 | bs 8 | bs 16 | Best |
|---|---|---|---|---|---|---|
| OWL-C | 279 | 314 | 312 | 285 | 242 | bs 2 |
| OWL-D | 10 | 10 | 11 | 10 | 10 | bs 4 |
| OWL-T | 141 | 157 | 155 | 146 | 132 | bs 2 |

> Preprocess/postprocess split is unavailable for Faster R-CNN, OWL-C, OWL-D, OWL-T, YOLO-NAS: only Ultralytics reports the stages separately, the other frameworks expose a single end-to-end call. Their `ms / crop` above is the whole process (disk read, preprocess, forward, decode).

## Task 2.2 — Accuracy vs efficiency

A model is on the **Pareto frontier** for a trade-off when no other model beats it on both axes at once. Only frontier models can ever be the right choice; the rest are beaten outright.

| Trade-off | On the frontier |
|---|---|
| AP (%) vs Parameters (M) | **OWL-C**, **YOLO11**, **YOLOv5**, **YOLO-NAS** |
| F1 (%) vs Checkpoint size (MB) | **YOLO11**, **YOLOv5** |
| MAE (birds/crop) vs Inference (ms/crop) | **OWL-D**, **YOLOv5**, **OWL-C** |
| RMSE (birds/crop) vs Inference (ms/crop) | **OWL-D**, **YOLOv5**, **OWL-C** |
| Recall (%) vs Peak GPU memory (MB) | **Faster R-CNN**, **YOLO-NAS**, **YOLOv5** |
| Counting accuracy (%) vs Throughput (crops/s) | **OWL-C**, **YOLOv5**, **OWL-T**, **Faster R-CNN** |

### How often each model is non-dominated

| Model | Frontiers | Of | Trade-offs |
|---|---|---|---|
| YOLOv5 | 6 | 6 | AP (%); F1 (%); MAE (birds/crop); RMSE (birds/crop); Recall (%); Counting accuracy (%) |
| OWL-C | 4 | 6 | AP (%); MAE (birds/crop); RMSE (birds/crop); Counting accuracy (%) |
| Faster R-CNN | 2 | 6 | Recall (%); Counting accuracy (%) |
| OWL-D | 2 | 6 | MAE (birds/crop); RMSE (birds/crop) |
| YOLO11 | 2 | 6 | AP (%); F1 (%) |
| YOLO-NAS | 2 | 6 | AP (%); Recall (%) |
| OWL-T | 1 | 6 | Counting accuracy (%) |
| YOLO26 | 0 | 6 | — |

### Best choice under a constraint

| Constraint | Pick | AP (%) | Cost |
|---|---|---|---|
| Peak GPU ≤ 512 MB | YOLO-NAS | 93.31 | 459 MB |
| Latency ≤ 5 ms/crop | YOLOv5 | 92.38 | 4.18 ms |
| Checkpoint ≤ 100 MB | YOLOv5 | 92.38 | 48 MB |
| ≤ 25 M parameters | YOLO11 | 92.33 | 20.0 M |

## Task 2.3 — Performance vs apparent bird size

Apparent size is **√(bounding-box area)** — the side of the equivalent square, in crop pixels. It reduces a box to one comparable number, stays in units a reader can picture, and needs only the existing annotations.

Bins hold equal counts of the 89,884 ground-truth birds, so each is estimated from a comparable sample rather than splitting a skewed distribution at round numbers.

| Size bin (px) | Birds | Faster R-CNN | OWL-C | OWL-D | OWL-T | YOLO11 | YOLO26 | YOLO-NAS | YOLOv5 |
|---|---|---|---|---|---|---|---|---|---|
| 1–14 | 14,733 | 60.24 | 77.78 | 77.80 | 79.47 | 76.99 | 75.67 | 72.94 | 76.58 |
| 14–17 | 14,495 | 85.83 | 91.04 | 91.40 | 93.02 | 92.88 | 91.98 | 92.16 | 92.83 |
| 17–19 | 15,189 | 91.41 | 92.09 | 92.84 | 93.19 | 93.17 | 92.33 | 92.86 | 93.44 |
| 19–23 | 15,447 | 91.91 | 91.59 | 92.07 | 92.72 | 92.17 | 91.78 | 91.61 | 92.69 |
| 23–38 | 15,059 | 89.89 | 84.14 | 88.60 | 85.37 | 87.13 | 87.37 | 85.21 | 87.64 |
| 38–306 | 14,961 | 95.18 | 83.00 | 89.11 | 84.83 | 90.02 | 89.23 | 90.69 | 90.35 |

Values are **recall (%)** — the share of birds in that size bin the model found at its own operating point.

### Where performance falls away

`Deficit` is the smallest bin's recall minus that model's best bin — how much accuracy the smallest birds cost it.

| Model | Recall, smallest bin | Best bin | Deficit | Bins below 80% |
|---|---|---|---|---|
| Faster R-CNN | 60.24 | 95.18 | -34.9 | 1–14 |
| OWL-C | 77.78 | 92.09 | -14.3 | 1–14 |
| OWL-D | 77.80 | 92.84 | -15.0 | 1–14 |
| OWL-T | 79.47 | 93.19 | -13.7 | 1–14 |
| YOLO11 | 76.99 | 93.17 | -16.2 | 1–14 |
| YOLO26 | 75.67 | 92.33 | -16.7 | 1–14 |
| YOLO-NAS | 72.94 | 92.86 | -19.9 | 1–14 |
| YOLOv5 | 76.58 | 93.44 | -16.9 | 1–14 |

### Key question — where does performance start to deteriorate?

"Significant" = more than 5 points below that model's own best size bin.

| Model | Degrades below (px) | Recall there | Also degrades on largest bin |
|---|---|---|---|
| Faster R-CNN | 17 | 60.24 | no |
| OWL-C | 14 | 77.78 | yes |
| OWL-D | 14 | 77.80 | no |
| OWL-T | 14 | 79.47 | yes |
| YOLO11 | 14 | 76.99 | no |
| YOLO26 | 14 | 75.67 | no |
| YOLO-NAS | 14 | 72.94 | no |
| YOLOv5 | 14 | 76.58 | no |

**Detection holds from about 14 px of apparent size upward and falls away sharply below it.** 8 of 8 models degrade in the smallest bin — recall drops to 60–79% there against 91–95% one bin up. Since 14 px is √(box area), that is a bird covering roughly 210 px² — a few hundred pixels in a 512×512 crop.

Separately, 2 of 8 also lose recall on the largest bin. That is a different failure — very large birds are rare, often clipped by the crop boundary, and under-represented in training — and should not be read as part of the small-object limit.

## Task 2.4 — Performance vs bird density

Density is the number of ground-truth birds in a crop. Bands come from the observed distribution, weighted towards the dense tail — quartile-width bands would put every crop from 18 to 422 birds in a single bucket, which is the regime this task is about. Empty crops are kept separate: they can only ever contribute false positives.

| Birds per crop | Crops | Birds | Faster R-CNN | OWL-C | OWL-D | OWL-T | YOLO11 | YOLO26 | YOLO-NAS | YOLOv5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (background) | 19,700 | 0 | — | — | — | — | — | — | — | — |
| 1 | 2,257 | 2,257 | 85.56 | 72.22 | 73.55 | 69.69 | 73.99 | 73.90 | 77.67 | 75.37 |
| 2–3 | 2,412 | 5,778 | 86.81 | 72.88 | 78.28 | 73.83 | 78.68 | 77.02 | 80.22 | 79.01 |
| 4–7 | 1,933 | 9,987 | 87.68 | 76.86 | 84.33 | 77.90 | 82.03 | 81.40 | 81.04 | 82.83 |
| 8–12 | 855 | 8,324 | 87.36 | 80.73 | 86.87 | 82.53 | 83.89 | 83.28 | 81.52 | 84.66 |
| 13–24 | 802 | 13,695 | 88.33 | 84.03 | 87.82 | 85.81 | 87.00 | 86.13 | 84.17 | 86.85 |
| 25–53 | 362 | 12,295 | 88.48 | 87.12 | 88.75 | 89.12 | 89.70 | 88.44 | 87.97 | 89.78 |
| 54–216 | 221 | 25,167 | 84.49 | 92.94 | 92.45 | 94.83 | 93.28 | 92.96 | 92.38 | 93.47 |
| 217–422 | 46 | 12,381 | 80.44 | 97.20 | 94.24 | 98.17 | 96.70 | 96.49 | 96.12 | 96.31 |

Values are **recall (%)**. Background crops have no birds to recall, so that row is blank by construction.

### Systematic count error (%) — negative means undercount

| Birds per crop | Faster R-CNN | OWL-C | OWL-D | OWL-T | YOLO11 | YOLO26 | YOLO-NAS | YOLOv5 |
|---|---|---|---|---|---|---|---|---|
| 0 (background) | — | — | — | — | — | — | — | — |
| 1 | +18.9 | -14.1 | -9.3 | -17.8 | -12.5 | -14.4 | -5.6 | -10.9 |
| 2–3 | +4.4 | -17.7 | -7.3 | -17.4 | -12.1 | -14.0 | -9.4 | -11.1 |
| 4–7 | -1.1 | -15.8 | -5.3 | -14.7 | -10.7 | -11.3 | -11.9 | -9.4 |
| 8–12 | -2.6 | -10.2 | -3.0 | -8.2 | -8.2 | -8.3 | -11.6 | -7.1 |
| 13–24 | -2.8 | -6.7 | -2.9 | -4.5 | -4.8 | -5.5 | -8.9 | -5.0 |
| 25–53 | -6.3 | -5.9 | -3.7 | -3.0 | -4.9 | -5.6 | -6.7 | -4.9 |
| 54–216 | -14.4 | -1.2 | -2.7 | +2.2 | -2.4 | -1.6 | -3.4 | -2.3 |
| 217–422 | -19.4 | +3.2 | -2.6 | +4.6 | +1.3 | +2.5 | +0.2 | +0.3 |

### Key question — do detector families fail differently?

| Model | Family | Recall trend (pts/band) | Sparse → dense recall | Count error, densest band |
|---|---|---|---|---|
| Faster R-CNN | two-stage box | -0.52 | 85.6 → 80.4 | -19.4% |
| OWL-C | point | +3.68 | 72.2 → 97.2 | +3.2% |
| OWL-D | point | +2.74 | 73.5 → 94.2 | -2.6% |
| OWL-T | point | +4.06 | 69.7 → 98.2 | +4.6% |
| YOLO11 | single-stage box | +3.07 | 74.0 → 96.7 | +1.3% |
| YOLO26 | single-stage box | +3.12 | 73.9 → 96.5 | +2.5% |
| YOLO-NAS | single-stage box | +2.54 | 77.7 → 96.1 | +0.2% |
| YOLOv5 | single-stage box | +2.88 | 75.4 → 96.3 | +0.3% |

**Yes — and in opposite directions.** Faster R-CNN loses recall as crops get denser, while OWL-C, OWL-D, OWL-T, YOLO11, YOLO26, YOLO-NAS, YOLOv5 gain it. A single aggregate score hides this completely: the models that look closest overall behave least alike in the dense flocks that dominate the bird count.
