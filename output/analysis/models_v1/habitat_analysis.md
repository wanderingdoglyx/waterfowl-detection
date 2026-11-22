*Manifest: `models_v1.docx`. All 8 models scored on the current test split: 28,588 crops from 426 source photographs, 89,884 annotated birds. Metrics are recomputed from the cached predictions with the project's own protocol, so every table describes the same runs on the same images as the benchmark report.*

*Precision, recall and F1 are reported at each model's **global** operating threshold, not retuned per subgroup — otherwise a comparison of habitats would partly be a comparison of thresholds. AP is threshold-free.*

## Habitat analysis

**Coverage.** 10,890 of 28,588 crops (38.1%) and 60,477 of 89,884 birds (67.3%), from Bird_G, Bird_H, Bird_J. Source: `background_info` column (Bird_H) and filename token 2 (Bird_G, Bird_J); no other dataset records habitat.

Habitat is recorded three different ways and only for three datasets: a curated `background_info` column in Bird_H, and a filename token in Bird_G and Bird_J. Two vocabulary problems documented in the inventory are fixed before anything is counted — Bird_G's `WaterSubVeg`/`WaterSubVege` are one class spelled two ways and are merged, and Bird_J's `Dense` describes bird density rather than habitat and is excluded rather than analysed as a habitat.

Groups holding fewer than 200 birds or fewer than 3 source images are shown in parentheses throughout: they were measured, but they are statements about a handful of photographs and are excluded from every ranking below.

| Habitat | Crops | Source images | Birds | Sub-datasets | Usable |
|---|--:|--:|--:|---|:--:|
| MoistSoil | 2,016 | 21 | 16,521 | H,J | yes |
| Ice | 504 | 6 | 14,820 | G | yes |
| StandingCorn | 1,728 | 18 | 8,419 | H,J | yes |
| WaterVegetation | 804 | 9 | 5,744 | G | yes |
| OpenWater | 1,158 | 13 | 4,442 | G,H,J | yes |
| Land | 996 | 11 | 2,867 | G,J | yes |
| HarvestedCrop | 864 | 9 | 2,283 | J | yes |
| WaterSubVeg | 348 | 4 | 1,955 | G | yes |
| Wooded | 576 | 6 | 1,052 | H,J | yes |
| WaterCorn | 168 | 2 | 778 | G | **no** |
| Trees | 672 | 7 | 732 | H | yes |
| ShrubScrub | 480 | 5 | 689 | H,J | yes |
| Lotus | 576 | 6 | 175 | J | **no** |

### AP (%)

| Model | MoistSoil | Ice | StandingCorn | WaterVegetation | OpenWater | Land | HarvestedCrop | WaterSubVeg | Wooded | WaterCorn | Trees | ShrubScrub | Lotus |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 92.9 | 96.1 | 88.9 | 92.5 | 94.2 | 82.8 | 87.9 | 95.6 | 78.6 | (92.9) | 79.8 | 81.2 | (77.9) |
| OWL-C | 86.9 | 95.5 | 79.9 | 85.9 | 89.0 | 73.0 | 81.1 | 91.0 | 66.3 | (85.7) | 72.6 | 63.2 | (73.5) |
| OWL-D | 84.4 | 92.0 | 71.6 | 82.1 | 87.4 | 81.1 | 77.1 | 89.5 | 57.8 | (81.2) | 72.3 | 62.6 | (72.6) |
| OWL-T | 89.1 | 96.7 | 82.7 | 88.4 | 90.9 | 75.0 | 85.4 | 92.3 | 72.6 | (87.9) | 76.4 | 68.5 | (69.7) |
| YOLO11 | 92.1 | 97.6 | 89.0 | 90.8 | 90.6 | 86.5 | 88.4 | 93.2 | 78.5 | (89.6) | 85.1 | 79.7 | (84.4) |
| YOLO26 | 91.1 | 97.5 | 88.3 | 89.6 | 89.7 | 85.2 | 87.9 | 92.4 | 77.7 | (89.5) | 83.7 | 79.4 | (83.7) |
| YOLO-NAS | 94.2 | 97.7 | 89.3 | 92.2 | 94.3 | 87.5 | 89.2 | 93.7 | 80.2 | (89.6) | 81.2 | 79.1 | (80.2) |
| YOLOv5 | 92.5 | 97.6 | 88.7 | 90.9 | 90.6 | 86.3 | 87.8 | 93.2 | 79.2 | (89.3) | 85.8 | 80.0 | (83.9) |

### Precision (%) at each model's global t*

| Model | MoistSoil | Ice | StandingCorn | WaterVegetation | OpenWater | Land | HarvestedCrop | WaterSubVeg | Wooded | WaterCorn | Trees | ShrubScrub | Lotus |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 94.7 | 98.5 | 76.7 | 88.5 | 96.3 | 90.6 | 79.8 | 94.7 | 66.9 | (78.7) | 71.7 | 66.6 | (33.2) |
| OWL-C | 94.4 | 95.8 | 82.0 | 88.4 | 97.1 | 93.7 | 88.5 | 94.5 | 81.2 | (85.7) | 83.2 | 82.7 | (74.1) |
| OWL-D | 91.7 | 96.7 | 82.4 | 87.8 | 95.1 | 91.7 | 89.3 | 93.6 | 71.3 | (83.5) | 79.0 | 73.9 | (75.3) |
| OWL-T | 94.4 | 94.9 | 81.8 | 87.9 | 97.7 | 92.9 | 86.6 | 95.2 | 79.0 | (85.7) | 82.8 | 84.5 | (64.0) |
| YOLO11 | 95.6 | 96.1 | 85.2 | 88.2 | 97.7 | 94.1 | 87.2 | 95.4 | 83.4 | (84.8) | 88.1 | 81.3 | (82.5) |
| YOLO26 | 95.4 | 94.0 | 85.1 | 87.5 | 96.7 | 94.4 | 90.8 | 95.5 | 83.5 | (83.9) | 88.6 | 83.0 | (80.7) |
| YOLO-NAS | 95.3 | 96.0 | 81.9 | 89.4 | 97.4 | 93.1 | 81.7 | 96.5 | 84.9 | (85.4) | 83.0 | 83.8 | (66.2) |
| YOLOv5 | 95.6 | 96.6 | 84.2 | 88.4 | 97.8 | 93.7 | 89.6 | 95.8 | 82.1 | (84.5) | 85.5 | 79.2 | (81.0) |

### Recall (%) at each model's global t*

| Model | MoistSoil | Ice | StandingCorn | WaterVegetation | OpenWater | Land | HarvestedCrop | WaterSubVeg | Wooded | WaterCorn | Trees | ShrubScrub | Lotus |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 85.7 | 81.1 | 87.2 | 90.6 | 89.3 | 77.5 | 83.6 | 92.1 | 77.4 | (93.2) | 81.1 | 82.0 | (83.4) |
| OWL-C | 84.5 | 94.6 | 80.2 | 85.0 | 84.1 | 69.9 | 78.1 | 88.3 | 63.3 | (88.4) | 68.3 | 61.2 | (72.0) |
| OWL-D | 87.9 | 92.9 | 82.1 | 90.4 | 88.9 | 85.3 | 82.9 | 93.6 | 73.4 | (95.0) | 85.0 | 76.3 | (85.1) |
| OWL-T | 86.9 | 96.3 | 82.5 | 87.3 | 86.2 | 71.9 | 82.3 | 89.5 | 69.5 | (89.3) | 74.2 | 62.6 | (68.0) |
| YOLO11 | 87.8 | 95.1 | 85.3 | 89.7 | 81.5 | 80.6 | 83.2 | 88.8 | 74.0 | (90.5) | 79.9 | 77.6 | (83.4) |
| YOLO26 | 86.6 | 95.4 | 84.4 | 88.6 | 82.0 | 79.2 | 81.9 | 89.2 | 74.0 | (90.6) | 79.4 | 77.5 | (81.1) |
| YOLO-NAS | 86.2 | 94.2 | 84.3 | 87.2 | 87.0 | 75.1 | 83.1 | 85.3 | 70.7 | (88.9) | 71.9 | 72.1 | (80.6) |
| YOLOv5 | 88.4 | 94.8 | 85.5 | 89.5 | 83.4 | 80.9 | 82.6 | 88.1 | 75.3 | (90.4) | 80.5 | 78.7 | (82.9) |

### F1 (%) at each model's global t*

| Model | MoistSoil | Ice | StandingCorn | WaterVegetation | OpenWater | Land | HarvestedCrop | WaterSubVeg | Wooded | WaterCorn | Trees | ShrubScrub | Lotus |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 89.9 | 89.0 | 81.6 | 89.5 | 92.7 | 83.5 | 81.7 | 93.4 | 71.8 | (85.3) | 76.2 | 73.5 | (47.5) |
| OWL-C | 89.2 | 95.2 | 81.1 | 86.6 | 90.1 | 80.0 | 83.0 | 91.3 | 71.2 | (87.0) | 75.0 | 70.4 | (73.0) |
| OWL-D | 89.7 | 94.8 | 82.2 | 89.1 | 91.9 | 88.4 | 86.0 | 93.6 | 72.3 | (88.9) | 81.9 | 75.1 | (79.9) |
| OWL-T | 90.5 | 95.6 | 82.2 | 87.6 | 91.6 | 81.1 | 84.4 | 92.3 | 74.0 | (87.5) | 78.2 | 71.9 | (65.9) |
| YOLO11 | 91.6 | 95.6 | 85.3 | 89.0 | 88.9 | 86.8 | 85.1 | 92.0 | 78.4 | (87.6) | 83.8 | 79.4 | (83.0) |
| YOLO26 | 90.8 | 94.7 | 84.7 | 88.1 | 88.8 | 86.2 | 86.1 | 92.2 | 78.5 | (87.1) | 83.7 | 80.2 | (80.9) |
| YOLO-NAS | 90.5 | 95.1 | 83.1 | 88.3 | 91.9 | 83.1 | 82.4 | 90.6 | 77.2 | (87.2) | 77.0 | 77.5 | (72.7) |
| YOLOv5 | 91.9 | 95.7 | 84.8 | 88.9 | 90.0 | 86.8 | 86.0 | 91.8 | 78.5 | (87.3) | 82.9 | 79.0 | (81.9) |

### MAE (birds per crop)

| Model | MoistSoil | Ice | StandingCorn | WaterVegetation | OpenWater | Land | HarvestedCrop | WaterSubVeg | Wooded | WaterCorn | Trees | ShrubScrub | Lotus |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 1.183 | 5.562 | 1.208 | 0.975 | 0.492 | 0.774 | 0.733 | 0.517 | 0.681 | (1.065) | 0.455 | 0.627 | (0.533) |
| OWL-C | 1.057 | 1.401 | 1.049 | 1.138 | 0.598 | 0.856 | 0.517 | 0.675 | 0.625 | (0.804) | 0.397 | 0.581 | (0.151) |
| OWL-D | 0.778 | 1.694 | 0.942 | 0.917 | 0.464 | 0.481 | 0.468 | 0.474 | 0.585 | (0.756) | 0.275 | 0.552 | (0.123) |
| OWL-T | 0.902 | 1.437 | 1.019 | 1.117 | 0.533 | 0.813 | 0.484 | 0.612 | 0.533 | (0.792) | 0.339 | 0.581 | (0.182) |
| YOLO11 | 0.840 | 1.371 | 0.863 | 1.083 | 0.731 | 0.599 | 0.556 | 0.655 | 0.469 | (0.667) | 0.253 | 0.440 | (0.097) |
| YOLO26 | 0.906 | 1.593 | 0.879 | 1.146 | 0.696 | 0.608 | 0.498 | 0.638 | 0.477 | (0.750) | 0.244 | 0.408 | (0.106) |
| YOLO-NAS | 0.985 | 1.425 | 0.995 | 1.075 | 0.519 | 0.747 | 0.685 | 0.816 | 0.497 | (0.702) | 0.399 | 0.421 | (0.174) |
| YOLOv5 | 0.795 | 1.272 | 0.931 | 1.058 | 0.651 | 0.575 | 0.501 | 0.687 | 0.495 | (0.690) | 0.296 | 0.440 | (0.101) |

### RMSE (birds per crop)

| Model | MoistSoil | Ice | StandingCorn | WaterVegetation | OpenWater | Land | HarvestedCrop | WaterSubVeg | Wooded | WaterCorn | Trees | ShrubScrub | Lotus |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 3.527 | 16.560 | 2.594 | 1.791 | 3.187 | 4.302 | 1.892 | 0.977 | 1.419 | (2.120) | 0.994 | 1.098 | (1.210) |
| OWL-C | 2.541 | 3.114 | 2.221 | 2.103 | 1.647 | 4.422 | 1.631 | 1.242 | 1.445 | (1.624) | 1.235 | 1.077 | (0.439) |
| OWL-D | 1.870 | 4.737 | 2.240 | 1.776 | 1.851 | 2.197 | 1.383 | 0.879 | 1.293 | (1.568) | 0.797 | 1.007 | (0.402) |
| OWL-T | 2.099 | 3.455 | 2.130 | 2.102 | 1.303 | 4.538 | 1.437 | 1.161 | 1.258 | (1.533) | 0.976 | 1.077 | (0.515) |
| YOLO11 | 2.174 | 3.430 | 1.966 | 2.146 | 2.223 | 3.099 | 1.921 | 1.160 | 1.075 | (1.371) | 0.820 | 0.830 | (0.349) |
| YOLO26 | 2.386 | 4.115 | 1.953 | 2.211 | 2.089 | 3.086 | 1.384 | 1.147 | 1.091 | (1.480) | 0.796 | 0.793 | (0.370) |
| YOLO-NAS | 2.399 | 3.433 | 2.130 | 1.939 | 1.441 | 3.384 | 2.763 | 1.454 | 1.147 | (1.456) | 1.138 | 0.834 | (0.514) |
| YOLOv5 | 2.100 | 3.013 | 2.053 | 2.026 | 2.060 | 3.101 | 1.504 | 1.319 | 1.139 | (1.380) | 0.844 | 0.845 | (0.333) |

### Count bias (%, + = overcount)

| Model | MoistSoil | Ice | StandingCorn | WaterVegetation | OpenWater | Land | HarvestedCrop | WaterSubVeg | Wooded | WaterCorn | Trees | ShrubScrub | Lotus |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | -9.5 | -17.7 | +13.7 | +2.4 | -7.2 | -14.5 | +4.8 | -2.8 | +15.6 | (+18.4) | +13.1 | +23.1 | (+151.4) |
| OWL-C | -10.5 | -1.2 | -2.3 | -3.8 | -13.4 | -25.4 | -11.7 | -6.5 | -22.1 | (+3.2) | -17.9 | -26.0 | (-2.9) |
| OWL-D | -4.1 | -4.0 | -0.4 | +2.9 | -6.5 | -7.0 | -7.2 | -0.1 | +2.9 | (+13.8) | +7.5 | +3.3 | (+13.1) |
| OWL-T | -7.9 | +1.5 | +0.9 | -0.6 | -11.8 | -22.6 | -4.9 | -6.0 | -12.1 | (+4.2) | -10.4 | -26.0 | (+6.3) |
| YOLO11 | -8.2 | -1.0 | +0.1 | +1.8 | -16.5 | -14.3 | -4.6 | -6.9 | -11.2 | (+6.7) | -9.3 | -4.5 | (+1.1) |
| YOLO26 | -9.2 | +1.4 | -0.8 | +1.3 | -15.2 | -16.1 | -9.8 | -6.5 | -11.3 | (+8.0) | -10.4 | -6.7 | (+0.6) |
| YOLO-NAS | -9.5 | -1.9 | +2.9 | -2.4 | -10.7 | -19.4 | +1.8 | -11.6 | -16.7 | (+4.1) | -13.4 | -13.9 | (+21.7) |
| YOLOv5 | -7.6 | -1.8 | +1.5 | +1.3 | -14.7 | -13.6 | -7.8 | -8.0 | -8.3 | (+6.9) | -5.9 | -0.7 | (+2.3) |

### Does the ranking change between habitats?

| Habitat | Best | 2nd | 3rd | Kendall tau vs whole split |
|---|---|---|---|--:|
| MoistSoil | YOLOv5 | YOLO11 | YOLO26 | +0.71 |
| Ice | YOLOv5 | YOLO11 | OWL-T | +0.36 |
| StandingCorn | YOLO11 | YOLOv5 | YOLO26 | +0.79 |
| WaterVegetation | Faster R-CNN | OWL-D | YOLO11 | +0.14 |
| OpenWater | Faster R-CNN | OWL-D | YOLO-NAS | -0.36 |
| Land | OWL-D | YOLOv5 | YOLO11 | +0.57 |
| HarvestedCrop | YOLO26 | OWL-D | YOLOv5 | +0.57 |
| WaterSubVeg | OWL-D | Faster R-CNN | OWL-T | -0.21 |
| Wooded | YOLOv5 | YOLO26 | YOLO11 | +0.71 |
| Trees | YOLO11 | YOLO26 | YOLOv5 | +0.71 |
| ShrubScrub | YOLO26 | YOLO11 | YOLOv5 | +0.57 |

**The ranking is not stable across habitats.** Different models take first place in different habitats (YOLOv5 in 3, YOLO11 in 2, Faster R-CNN in 2, OWL-D in 2, YOLO26 in 2). Mean Kendall tau against the whole-split order is +0.42 — a single overall ranking hides habitat-specific reversals, and a model should be chosen for the habitat it will fly over.

### Confounding with dataset identity

7 of the 13 habitats appear in exactly one sub-dataset (Ice, WaterVegetation, HarvestedCrop, WaterSubVeg, WaterCorn, Trees, Lotus), so for those a habitat difference is indistinguishable from a difference between datasets — different sites, dates, cameras and annotators. Habitats that appear in more than one dataset are the only ones where the comparison is even partly clean:

| Habitat | Datasets | Crops per dataset |
|---|---|---|
| MoistSoil | 2 | Bird_H: 960, Bird_J: 1,056 |
| StandingCorn | 2 | Bird_H: 480, Bird_J: 1,248 |
| OpenWater | 3 | Bird_G: 168, Bird_H: 864, Bird_J: 126 |
| Land | 2 | Bird_G: 336, Bird_J: 660 |
| Wooded | 2 | Bird_H: 288, Bird_J: 288 |
| ShrubScrub | 2 | Bird_H: 192, Bird_J: 288 |

### Relevance to the Arkansas proposal

The proposal names bottomland hardwood forest, rice and soybean fields as the hard cases for generalisation. **None of those appear verbatim in the MDC data**, so the table below groups the habitats that do exist into the nearest analogue and labels the mapping as an analogue rather than a match. The row-crop group is corn, not rice or soybean; the forested group is the closest available stand-in for bottomland hardwood.

| Proposal environment | MDC habitats used | Crops | Birds | Best F1 | Worst F1 |
|---|---|--:|--:|---|---|
| Forested / bottomland hardwood | Wooded, Trees | 1,248 | 1,784 | YOLO11 80.7 | OWL-C 72.8 |
| Row-crop agriculture (rice / soybean analogue) | StandingCorn, HarvestedCrop, WaterCorn | 2,760 | 11,480 | YOLO11 85.4 | OWL-C 81.9 |
| Moist-soil / managed wetland | MoistSoil | 2,016 | 16,521 | YOLOv5 91.9 | OWL-C 89.2 |
| Open water | Ice, OpenWater | 1,662 | 19,262 | OWL-T 94.7 | Faster R-CNN 89.9 |
| Emergent / submerged vegetation | WaterVegetation, WaterSubVeg, ShrubScrub, Lotus | 2,208 | 8,563 | OWL-D 88.8 | OWL-C 86.2 |
| Dry land / levee | Land | 996 | 2,867 | OWL-D 88.4 | OWL-C 80.0 |
