# Model comparison

Manifest: `models_v1.docx` — 8 models.

Point protocol: τ = 40 px, 25,708 test images, 88,195 ground-truth points, score-ordered greedy one-to-one matching.

## OWL-paper point metrics (τ = 40 px)

Comparable across every model. **Bold** marks the best value per column.

| Model | AP | F1 | Precision | Recall | MAE | RMSE |
|---|---|---|---|---|---|---|
| YOLO-NAS | **93.24** | 89.44 | 90.79 | 88.12 | 0.500 | 1.746 |
| YOLOv5 | 92.59 | **90.80** | **92.59** | 89.07 | 0.434 | **1.593** |
| YOLO11 | 92.52 | 90.71 | 92.25 | **89.22** | 0.436 | 1.635 |
| YOLO26 | 91.81 | 90.21 | 92.46 | 88.08 | 0.449 | 1.681 |
| OWL-T | 90.29 | 89.52 | 90.30 | 88.76 | 0.498 | 2.006 |
| Faster R-CNN | 89.54 | 86.41 | 87.20 | 85.63 | 0.762 | 3.594 |
| OWL-C | 88.70 | 88.78 | 89.79 | 87.80 | 0.520 | 1.936 |
| OWL-D | 83.88 | 89.89 | 90.67 | 89.12 | **0.427** | 1.596 |

## Box detection mAP30 (COCOeval @ IoU = 0.30)

| Box detector | mAP30 | mAR30 |
|---|---|---|
| YOLO-NAS | 77.46 | 81.11 |
| YOLOv5 | 76.79 | 78.68 |
| YOLO11 | 76.76 | 78.58 |
| YOLO26 | 75.85 | 77.86 |
| Faster R-CNN | 75.59 | 80.12 |

Point detectors have no boxes to score. The values below come from inflating predicted points to fixed 28 px boxes, so they measure localisation of a point, **not** box quality — they are not comparable with the box detectors above.

| Point detector | pseudo-mAP30 | pseudo-mAR30 |
|---|---|---|
| OWL-T | 49.27 | 59.08 |
| OWL-C | 48.21 | 58.05 |
| OWL-D | 39.85 | 56.94 |

## Counting

| Model | MAE | RMSE | Count error | Operating point t\* |
|---|---|---|---|---|
| OWL-D | 0.427 | 1.596 | -1.7% | 0.321 |
| YOLOv5 | 0.434 | 1.593 | -3.8% | 0.227 |
| YOLO11 | 0.436 | 1.635 | -3.3% | 0.202 |
| YOLO26 | 0.449 | 1.681 | -4.7% | 0.216 |
| OWL-T | 0.498 | 2.006 | -1.7% | 0.257 |
| YOLO-NAS | 0.500 | 1.746 | -2.9% | 0.403 |
| OWL-C | 0.520 | 1.936 | -2.2% | 0.265 |
| Faster R-CNN | 0.762 | 3.594 | -1.8% | 0.579 |

## Runs analysed

| Model | Run | Epochs | Train crops | Commit |
|---|---|---|---|---|
| Faster R-CNN | `2026-06-27_16-56-51` | — | 75,664 | — |
| OWL-C | `2026-07-21_22-59-50` | 100 | 75,664 | — |
| OWL-D | `2026-07-26_18-56-20` | 100 | 75,664 | — |
| OWL-T | `2026-07-24_10-39-49` | 100 | 75,664 | — |
| YOLO11 | `2026-08-14_00-25-32` | 100 | 75,664 | 9484535 |
| YOLO26 | `2026-08-15_21-24-35` | 100 | 75,664 | — |
| YOLO-NAS | `2026-06-28_21-14-03` | — | 75,664 | — |
| YOLOv5 | `2026-07-11_13-18-41` | — | 75,664 | — |
