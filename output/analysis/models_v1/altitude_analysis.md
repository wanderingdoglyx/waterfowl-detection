*Manifest: `models_v1.docx`. All 8 models scored on the current test split: 28,588 crops from 426 source photographs, 89,884 annotated birds. Metrics are recomputed from the cached predictions with the project's own protocol, so every table describes the same runs on the same images as the benchmark report.*

*Precision, recall and F1 are reported at each model's **global** operating threshold, not retuned per subgroup — otherwise a comparison of habitats would partly be a comparison of thresholds. AP is threshold-free.*

## Altitude analysis

**Coverage.** 28,588 of 28,588 crops (100.0%) and 89,884 of 89,884 birds (100.0%), from Bird_A, Bird_B, Bird_C, Bird_D, Bird_E, Bird_F, Bird_G, Bird_H, Bird_I, Bird_J. Source: —.

Altitude is the strongest covariate in this corpus: it is recorded for every image in all ten datasets, spans 14-150 m, and — uniquely among the metadata fields — varies *inside* six datasets, which makes a comparison possible that no between-dataset difference can explain. Heights are snapped to the nominal survey ladder within 2 m (14 -> 15, 29 -> 30, 59 -> 60, 89 -> 90, 119 -> 120), which merges recording jitter without inventing bands; Bird_A's genuinely distinct 40/50/70/80/100/110/130/140 m heights are left as they are.

| Altitude | Crops | Source images | Birds | Sub-datasets | Usable |
|---|--:|--:|--:|---|:--:|
| 15 | 7,527 | 80 | 9,765 | G,H,I,J | yes |
| 30 | 3,627 | 39 | 11,919 | B,C,G,H,J | yes |
| 40 | 96 | 1 | 68 | A | **no** |
| 45 | 192 | 2 | 130 | B | **no** |
| 50 | 288 | 3 | 302 | A | yes |
| 60 | 2,760 | 29 | 10,603 | A,B,C,G,H,J | yes |
| 70 | 384 | 4 | 287 | A | yes |
| 80 | 288 | 3 | 194 | A | **no** |
| 90 | 11,218 | 242 | 55,204 | A,B,C,D,E,F,G,H,J | yes |
| 100 | 384 | 4 | 275 | A | yes |
| 110 | 576 | 6 | 455 | A | yes |
| 120 | 1,056 | 11 | 558 | A,B,C | yes |
| 130 | 96 | 1 | 59 | A | **no** |
| 140 | 96 | 1 | 65 | A | **no** |

### AP (%)

| Model | 15 | 30 | 40 | 45 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 88.7 | 89.9 | (89.4) | (96.4) | 96.1 | 90.6 | 90.1 | (82.9) | 92.2 | 75.9 | 74.7 | 66.6 | (80.7) | (57.7) |
| OWL-C | 76.3 | 80.4 | (88.9) | (97.0) | 93.6 | 88.2 | 94.0 | (95.9) | 92.0 | 96.8 | 91.6 | 74.2 | (96.2) | (98.2) |
| OWL-D | 73.1 | 79.0 | (86.8) | (94.5) | 94.4 | 84.1 | 93.2 | (93.4) | 86.5 | 86.9 | 88.1 | 76.7 | (80.6) | (78.5) |
| OWL-T | 79.0 | 83.6 | (90.7) | (96.6) | 94.5 | 90.5 | 94.9 | (95.0) | 93.0 | 98.4 | 93.6 | 76.8 | (96.3) | (99.5) |
| YOLO11 | 83.8 | 88.0 | (88.0) | (97.6) | 96.4 | 92.7 | 94.8 | (96.3) | 94.8 | 98.7 | 95.3 | 88.3 | (99.8) | (98.7) |
| YOLO26 | 82.5 | 87.3 | (89.5) | (97.5) | 96.6 | 92.5 | 93.8 | (96.0) | 94.2 | 97.8 | 94.4 | 86.9 | (99.9) | (97.1) |
| YOLO-NAS | 85.7 | 88.5 | (93.3) | (97.0) | 96.7 | 93.0 | 94.5 | (95.6) | 95.7 | 93.0 | 90.8 | 79.5 | (96.7) | (93.3) |
| YOLOv5 | 82.9 | 88.3 | (91.1) | (97.6) | 96.3 | 92.9 | 94.8 | (96.6) | 94.8 | 98.8 | 96.0 | 88.2 | (98.6) | (99.5) |

### Recall (%) at each model's global t*

| Model | 15 | 30 | 40 | 45 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 87.5 | 87.0 | (86.8) | (96.9) | 92.7 | 84.8 | 89.9 | (91.2) | 85.4 | 89.5 | 86.4 | 83.5 | (96.6) | (98.5) |
| OWL-C | 70.9 | 79.6 | (86.8) | (96.9) | 93.0 | 87.0 | 93.7 | (97.4) | 90.7 | 96.4 | 92.5 | 78.7 | (96.6) | (96.9) |
| OWL-D | 79.4 | 86.4 | (86.8) | (96.9) | 94.7 | 90.2 | 94.1 | (94.3) | 90.5 | 86.5 | 91.2 | 86.0 | (81.4) | (80.0) |
| OWL-T | 72.5 | 82.0 | (89.7) | (96.9) | 94.0 | 89.9 | 93.0 | (95.9) | 91.8 | 98.2 | 93.8 | 79.0 | (96.6) | (100.0) |
| YOLO11 | 77.5 | 84.8 | (86.8) | (96.9) | 95.7 | 88.3 | 93.0 | (96.9) | 91.5 | 98.5 | 95.8 | 89.1 | (100.0) | (98.5) |
| YOLO26 | 75.8 | 84.8 | (86.8) | (96.2) | 94.7 | 88.5 | 92.3 | (96.4) | 90.7 | 98.2 | 94.1 | 87.3 | (100.0) | (96.9) |
| YOLO-NAS | 80.3 | 81.8 | (86.8) | (95.4) | 93.4 | 86.7 | 92.0 | (95.4) | 90.5 | 85.1 | 80.0 | 78.0 | (76.3) | (78.5) |
| YOLOv5 | 78.2 | 85.7 | (88.2) | (96.9) | 95.7 | 88.7 | 94.4 | (96.9) | 91.4 | 99.3 | 95.6 | 88.9 | (100.0) | (98.5) |

### Precision (%) at each model's global t*

| Model | 15 | 30 | 40 | 45 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 82.8 | 82.5 | (90.8) | (82.4) | 90.9 | 86.7 | 77.5 | (59.4) | 90.8 | 58.2 | 52.6 | 49.6 | (58.2) | (46.7) |
| OWL-C | 91.3 | 87.5 | (83.1) | (85.1) | 89.5 | 89.6 | 90.3 | (86.7) | 91.8 | 89.5 | 89.2 | 79.4 | (89.1) | (88.7) |
| OWL-D | 86.4 | 87.6 | (100.0) | (93.3) | 93.8 | 90.0 | 93.4 | (92.0) | 92.3 | 94.8 | 90.6 | 82.6 | (92.3) | (96.3) |
| OWL-T | 91.9 | 87.6 | (91.0) | (87.5) | 87.4 | 87.7 | 90.8 | (86.1) | 91.8 | 92.2 | 90.7 | 79.0 | (86.4) | (87.8) |
| YOLO11 | 91.1 | 88.8 | (92.2) | (94.7) | 94.1 | 92.9 | 91.1 | (87.4) | 93.4 | 86.0 | 86.7 | 82.1 | (88.1) | (80.0) |
| YOLO26 | 91.3 | 89.6 | (96.7) | (95.4) | 92.9 | 92.2 | 92.3 | (85.8) | 92.7 | 86.3 | 83.3 | 84.1 | (90.8) | (85.1) |
| YOLO-NAS | 88.5 | 88.4 | (100.0) | (94.7) | 92.2 | 90.6 | 89.8 | (85.6) | 92.9 | 86.7 | 89.4 | 78.0 | (91.8) | (89.5) |
| YOLOv5 | 90.9 | 88.9 | (100.0) | (96.9) | 91.2 | 91.4 | 91.6 | (86.6) | 93.6 | 87.2 | 89.0 | 82.9 | (89.4) | (84.2) |

### MAE (birds per crop)

| Model | 15 | 30 | 40 | 45 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 0.320 | 0.722 | (0.156) | (0.161) | 0.167 | 0.882 | 0.260 | (0.451) | 0.970 | 0.516 | 0.688 | 0.487 | (0.448) | (0.750) |
| OWL-C | 0.392 | 0.709 | (0.156) | (0.115) | 0.174 | 0.549 | 0.107 | (0.104) | 0.561 | 0.096 | 0.113 | 0.187 | (0.094) | (0.083) |
| OWL-D | 0.339 | 0.536 | (0.094) | (0.068) | 0.122 | 0.462 | 0.089 | (0.052) | 0.475 | 0.109 | 0.109 | 0.143 | (0.156) | (0.135) |
| OWL-T | 0.378 | 0.661 | (0.073) | (0.104) | 0.191 | 0.576 | 0.107 | (0.111) | 0.542 | 0.057 | 0.104 | 0.197 | (0.115) | (0.094) |
| YOLO11 | 0.333 | 0.611 | (0.146) | (0.057) | 0.108 | 0.502 | 0.104 | (0.094) | 0.468 | 0.120 | 0.125 | 0.139 | (0.083) | (0.156) |
| YOLO26 | 0.352 | 0.577 | (0.115) | (0.057) | 0.132 | 0.483 | 0.099 | (0.097) | 0.492 | 0.109 | 0.161 | 0.132 | (0.062) | (0.115) |
| YOLO-NAS | 0.324 | 0.701 | (0.094) | (0.057) | 0.153 | 0.624 | 0.122 | (0.111) | 0.509 | 0.195 | 0.222 | 0.203 | (0.188) | (0.167) |
| YOLOv5 | 0.329 | 0.574 | (0.083) | (0.042) | 0.142 | 0.548 | 0.096 | (0.087) | 0.461 | 0.109 | 0.101 | 0.133 | (0.073) | (0.115) |

### Count bias (%, + = overcount)

| Model | 15 | 30 | 40 | 45 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | +5.7 | +5.5 | (-4.4) | (+17.7) | +2.0 | -2.2 | +16.0 | (+53.6) | -6.0 | +53.8 | +64.2 | +68.5 | (+66.1) | (+110.8) |
| OWL-C | -22.3 | -9.1 | (+4.4) | (+13.8) | +4.0 | -2.8 | +3.8 | (+12.4) | -1.2 | +7.6 | +3.7 | -0.9 | (+8.5) | (+9.2) |
| OWL-D | -8.1 | -1.5 | (-13.2) | (+3.8) | +1.0 | +0.2 | +0.7 | (+2.6) | -2.0 | -8.7 | +0.7 | +4.1 | (-11.9) | (-16.9) |
| OWL-T | -21.2 | -6.4 | (-1.5) | (+10.8) | +7.6 | +2.5 | +2.4 | (+11.3) | -0.0 | +6.5 | +3.5 | +0.0 | (+11.9) | (+13.8) |
| YOLO11 | -15.0 | -4.6 | (-5.9) | (+2.3) | +1.7 | -4.9 | +2.1 | (+10.8) | -2.1 | +14.5 | +10.5 | +8.4 | (+13.6) | (+23.1) |
| YOLO26 | -17.0 | -5.3 | (-10.3) | (+0.8) | +2.0 | -4.0 | +0.0 | (+12.4) | -2.2 | +13.8 | +13.0 | +3.8 | (+10.2) | (+13.8) |
| YOLO-NAS | -9.3 | -7.5 | (-13.2) | (+0.8) | +1.3 | -4.3 | +2.4 | (+11.3) | -2.6 | -1.8 | -10.5 | +0.0 | (-16.9) | (-12.3) |
| YOLOv5 | -13.9 | -3.7 | (-11.8) | (+0.0) | +5.0 | -3.0 | +3.1 | (+11.9) | -2.3 | +13.8 | +7.5 | +7.2 | (+11.9) | (+16.9) |

### The chain: altitude -> apparent size -> recall -> counting error

The mechanism the task asks about is not a direct one. Altitude does not degrade a detector by itself; it shrinks the birds, and small birds are what the detector misses (Task 2.3 measured that directly). The table follows the chain one link at a time, on the ladder heights that carry enough data.

| Altitude | Median apparent size (px) | Birds | Mean recall (%) | Mean MAE | Mean count bias (%) |
|---|--:|--:|--:|--:|--:|
| 15 m | 66.25 | 9,765 | 77.76 | 0.346 | -12.6 |
| 30 m | 37.67 | 11,919 | 84.00 | 0.636 | -4.1 |
| 60 m | 22.65 | 10,603 | 88.01 | 0.578 | -2.3 |
| 90 m | 17.29 | 55,204 | 90.29 | 0.560 | -2.3 |
| 120 m | 12.37 | 558 | 83.80 | 0.202 | +11.4 |

Across the 5 ladder heights: altitude vs apparent size r = -0.89; apparent size vs recall r = -0.81; altitude vs recall r = +0.54; altitude vs MAE r = -0.40.

**Read this as association, not as a controlled experiment.** The pooled heights differ in which datasets contribute them — 90 m is the only height flown by Bird_D, Bird_E and Bird_F, so the 90 m column carries those three datasets' scenes and nothing else does. The within-dataset comparison below is the part of this analysis that survives that objection.

### Within-dataset altitude effect (the confound-free comparison)

Each row below is one sub-dataset compared against itself: same site, same camera, same annotators, same season, different flight height. Any difference here is attributable to altitude in a way the pooled table above is not. Recall is averaged over all models, at each model's own global operating point.

| Dataset | Heights flown | Recall lowest -> highest (%) | Change | Shape | Median size low -> high (px) |
|---|---|--:|--:|---|---|
| Bird_A | 50, 70, 90, 100, 110, 120 | 94.2 -> 92.1 | -2.1 | U-shaped | 24.13 -> 12.00 |
| Bird_G | 15, 30, 60, 90 | 86.7 -> 92.9 | +6.2 | non-monotone | 61.51 -> 15.49 |
| Bird_H | 15, 30, 60, 90 | 80.2 -> 80.8 | +0.6 | U-shaped | 76.72 -> 16.73 |
| Bird_J | 15, 30, 60, 90 | 75.2 -> 86.7 | +11.5 | peaked | 57.31 -> 17.94 |

**The within-dataset picture is weaker than the pooled one.** Only 0 of 4 datasets rise monotonically with height; the rest turn somewhere in the middle, and Bird_A — the one dataset whose ladder sits entirely in the small-bird regime above 50 m — falls. Endpoint differences in the table above therefore overstate how orderly the effect is. Taken together with the pooled trend, the defensible statement is that **recall is worst at the extremes of the altitude range and best in the middle of it**, where both the training data and the apparent-size distribution are densest — not that flying higher is better.

### Why the middle of the range wins: the training distribution

Recall peaks in the middle of the altitude range rather than falling steadily with height, which apparent size alone cannot explain — birds shrink monotonically from 66 px to 12 px across this range, and Task 2.3 showed smaller is harder, so size predicts a monotone decline that does not happen. The competing explanation is that the models never saw the extremes in proportion: the training split is concentrated on exactly the heights where test recall is highest.

| Altitude | Share of training birds | Median apparent size (px) | Mean test recall (%) |
|---|--:|--:|--:|
| 15 m | 12.5% | 66.25 | 77.8 |
| 30 m | 15.1% | 37.67 | 84.0 |
| 60 m | 25.3% | 22.65 | 88.0 |
| 90 m | 44.2% | 17.29 | 90.3 |
| 120 m | 0.4% | 12.37 | 83.8 |

Correlation with mean recall across these 5 bands: training share r = +0.72, altitude r = +0.54. Both are computed on 5 points and neither is strong evidence on its own.

The relationship is much cleaner once the bands with almost no training data are set aside: across the 4 bands holding at least 5% of training birds, training share against recall gives r = +0.86. The exception is 120 m — 0.4% of training birds yet 83.8% recall, above two better-represented bands. That band holds only 558 birds, so it is both the noisiest point on the curve and the one where apparent size is smallest; it should not be read as contradicting the pattern, nor as confirming it.

The reading this supports: **the low-altitude deficit is most likely a sampling artefact of the training set rather than evidence that low flights are intrinsically harder.** The 15 m band holds 12.5% of training birds against 90 m's 44%, and 15 m birds are roughly four times the linear size the detectors mostly trained on, so they are out of distribution in the one direction the augmentation pipeline does not cover. It remains a hypothesis: the pooled bands differ in dataset composition as well as in training share.

This is testable and worth testing before any Arkansas flight plan fixes an altitude: re-train with altitude-balanced sampling, or with scale augmentation reaching the low-altitude size range, and see whether the 15-30 m deficit closes. If it does, altitude selection is a data decision rather than an operational constraint.
