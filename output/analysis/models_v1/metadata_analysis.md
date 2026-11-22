*Manifest: `models_v1.docx`. All 8 models scored on the current test split: 28,588 crops from 426 source photographs, 89,884 annotated birds. Metrics are recomputed from the cached predictions with the project's own protocol, so every table describes the same runs on the same images as the benchmark report.*

*Precision, recall and F1 are reported at each model's **global** operating threshold, not retuned per subgroup — otherwise a comparison of habitats would partly be a comparison of thresholds. AP is threshold-free.*

## Other survey metadata

The Week 1 audit determined which further variables the data can actually support. This section analyses only those, and states plainly which requested variables cannot be analysed at all rather than substituting a weak proxy.

### Weather (sun / cloud)

**Coverage.** 7,434 of 28,588 crops (26.0%) and 55,148 of 89,884 birds (61.4%), from Bird_G, Bird_J. Source: filename token 1, Bird_G and Bird_J only (Sun/Cloud; no other conditions recorded).

Sun/cloud is the only environmental condition recorded anywhere in the corpus, and only as a filename token in Bird_G and Bird_J. Both datasets contain both conditions, so this is one of the few fields where a within-dataset check is possible.

| Weather | Crops | Source images | Birds | Sub-datasets | Usable |
|---|--:|--:|--:|---|:--:|
| Cloud | 4,350 | 48 | 42,096 | G,J | yes |
| Sun | 3,084 | 33 | 13,052 | G,J | yes |

| Model | Cloud | Sun |
|---|--:|--:|
| Faster R-CNN | 88.7 | 84.9 |
| OWL-C | 90.3 | 83.9 |
| OWL-D | 90.7 | 86.8 |
| OWL-T | 91.0 | 85.4 |
| YOLO11 | 91.6 | 87.4 |
| YOLO26 | 90.8 | 87.3 |
| YOLO-NAS | 91.1 | 85.2 |
| YOLOv5 | 91.6 | 87.6 |

Per dataset, holding the dataset fixed:

| Dataset | Condition | Crops | Birds | Mean F1 (%) |
|---|---|--:|--:|--:|
| Bird_G | Cloud | 1,548 | 20,939 | 92.97 |
| Bird_G | Sun | 780 | 5,557 | 90.97 |
| Bird_J | Cloud | 2,802 | 21,157 | 88.43 |
| Bird_J | Sun | 2,304 | 7,495 | 82.37 |

### Survey site (Bird_H)

Bird_H is the only dataset with named sites that also has curated habitat, a survey date and a known altitude for every image, so a site comparison there is interpretable rather than a relabelled dataset difference.

| Site | Crops | Source images | Birds | Sub-datasets | Usable |
|---|--:|--:|--:|---|:--:|
| EagleBluffs | 1,824 | 19 | 2,961 | H | yes |
| TedShanks | 384 | 4 | 484 | H | yes |
| TenMilePond | 1,248 | 13 | 1,884 | H | yes |

| Model | EagleBluffs | TedShanks | TenMilePond |
|---|--:|--:|--:|
| Faster R-CNN | 83.4 | 72.6 | 79.3 |
| OWL-C | 81.2 | 68.7 | 82.2 |
| OWL-D | 85.9 | 76.0 | 83.6 |
| OWL-T | 83.3 | 69.0 | 83.3 |
| YOLO11 | 88.5 | 80.5 | 86.5 |
| YOLO26 | 88.1 | 80.3 | 85.8 |
| YOLO-NAS | 84.9 | 77.9 | 84.4 |
| YOLOv5 | 88.7 | 79.4 | 86.7 |

### Survey date (Bird_H)

Four survey dates spanning October 2021 to March 2022, flown with the same platform over the same set of sites. Season is confounded with site here (not every site was flown on every date), so this is a description of the four surveys rather than a seasonal effect.

| Date | Crops | Source images | Birds | Sub-datasets | Usable |
|---|--:|--:|--:|---|:--:|
| 03032022 | 384 | 4 | 484 | H | yes |
| 03082022 | 672 | 7 | 732 | H | yes |
| 10252021 | 1,152 | 12 | 2,229 | H | yes |
| 11302021 | 1,248 | 13 | 1,884 | H | yes |

| Model | 03032022 | 03082022 | 10252021 | 11302021 |
|---|--:|--:|--:|--:|
| Faster R-CNN | 72.6 | 76.2 | 86.1 | 79.3 |
| OWL-C | 68.7 | 75.0 | 83.2 | 82.2 |
| OWL-D | 76.0 | 81.9 | 87.3 | 83.6 |
| OWL-T | 69.0 | 78.2 | 85.0 | 83.3 |
| YOLO11 | 80.5 | 83.8 | 90.1 | 86.5 |
| YOLO26 | 80.3 | 83.7 | 89.6 | 85.8 |
| YOLO-NAS | 77.9 | 77.0 | 87.5 | 84.4 |
| YOLOv5 | 79.4 | 82.9 | 90.7 | 86.7 |

### Sub-dataset (the scale of the confounder)

Not a survey condition, but the reference every other table in this document has to be read against: if performance varies more between sub-datasets than between habitats or altitudes, then a habitat or altitude finding drawn across datasets is mostly measuring dataset identity.

| Dataset | Crops | Source images | Birds | Sub-datasets | Usable |
|---|--:|--:|--:|---|:--:|
| Bird_A | 3,168 | 33 | 2,474 | A | yes |
| Bird_B | 864 | 9 | 719 | B | yes |
| Bird_C | 2,016 | 21 | 265 | C | yes |
| Bird_D | 7,767 | 68 | 7,520 | D | yes |
| Bird_E | 139 | 139 | 15,538 | E | yes |
| Bird_F | 480 | 5 | 552 | F | yes |
| Bird_G | 2,328 | 27 | 26,496 | G | yes |
| Bird_H | 3,456 | 36 | 5,329 | H | yes |
| Bird_I | 3,264 | 34 | 2,339 | I | yes |
| Bird_J | 5,106 | 54 | 28,652 | J | yes |

| Model | Bird_A | Bird_B | Bird_C | Bird_D | Bird_E | Bird_F | Bird_G | Bird_H | Bird_I | Bird_J |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Faster R-CNN | 74.5 | 79.6 | 44.0 | 80.4 | 92.4 | 89.8 | 89.7 | 80.8 | 81.7 | 86.0 |
| OWL-C | 91.5 | 86.3 | 51.9 | 84.2 | 95.2 | 88.4 | 92.4 | 80.4 | 74.1 | 85.3 |
| OWL-D | 91.7 | 92.1 | 76.1 | 84.1 | 96.0 | 90.0 | 93.1 | 84.2 | 73.1 | 86.7 |
| OWL-T | 92.0 | 86.0 | 50.9 | 84.5 | 94.8 | 91.1 | 93.0 | 82.0 | 75.0 | 86.6 |
| YOLO11 | 92.0 | 92.4 | 55.7 | 85.8 | 95.9 | 91.7 | 93.4 | 87.0 | 77.1 | 87.8 |
| YOLO26 | 91.3 | 93.2 | 63.1 | 85.3 | 95.4 | 90.3 | 92.7 | 86.6 | 74.7 | 87.4 |
| YOLO-NAS | 89.0 | 88.0 | 41.4 | 84.2 | 95.7 | 90.7 | 92.7 | 84.1 | 81.1 | 86.8 |
| YOLOv5 | 92.7 | 92.0 | 54.9 | 85.0 | 96.1 | 91.6 | 93.5 | 87.1 | 78.0 | 88.0 |

**Scale of the confounder.** Mean F1 spread across usable groups: sub-dataset 40.8 points, habitat 20.1, altitude 14.6. Any pooled habitat or altitude difference smaller than the dataset spread should be treated as unproven, because habitat and altitude are not evenly distributed across the datasets.

### Requested variables the data cannot support

These were asked for and are not analysed. Each is absent from the data, not omitted for convenience:

- **decoy vs real** — No marker exists in any filename, CSV column or annotation file in any of the ten datasets. Distinguishing decoys would require new annotation or a field record from MDC.
- **camera sensor** — EXIF survives for only 191 of 2,095 images (9%), covering two camera models in Bird_B/Bird_G. Too thin, and fully confounded with dataset identity.
- **environment beyond sun cloud** — No wind, temperature, water level or light-angle data exists in any dataset.
