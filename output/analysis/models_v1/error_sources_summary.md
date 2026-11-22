*Manifest: `models_v1.docx`. All 8 models scored on the current test split: 28,588 crops from 426 source photographs, 89,884 annotated birds. Metrics are recomputed from the cached predictions with the project's own protocol, so every table describes the same runs on the same images as the benchmark report.*

*Precision, recall and F1 are reported at each model's **global** operating threshold, not retuned per subgroup — otherwise a comparison of habitats would partly be a comparison of thresholds. AP is threshold-free.*

## Preliminary summary — the major sources of error

Each factor below is scored the same way, so they can be ranked against each other instead of merely listed.

**How the spread is calculated.** For each factor: (1) partition the test split by that factor — habitat gives 13 classes, altitude gives the nominal heights; (2) discard groups holding fewer than 200 birds or fewer than 3 source photographs, leaving the *usable* groups; (3) for one model, compute F1 in every usable group at that model's own global operating threshold, and take the difference between its best and worst group; (4) average that difference over all eight models. It reads as: how many F1 points does this factor move a typical model between the easiest and hardest condition it defines. It is a spread, not an effect size — it says how much performance varies with the factor, not that the factor caused the variation.

| Rank | Factor | Mean F1 spread (points) | What it is |
|--:|---|--:|---|
| 1 | dataset | 40.8 | Which sub-dataset the crop came from — site, camera, annotator and season together |
| 2 | habitat | 20.1 | Habitat class (Bird_G/H/J only) |
| 3 | density | 16.8 | Birds per crop (density band) |
| 4 | altitude | 14.6 | Nominal flight altitude |
| 5 | date | 12.0 | Survey date within Bird_H |
| 6 | site | 10.1 | Survey site within Bird_H |
| 7 | weather | 4.7 | Sun vs cloud (Bird_G/J only) |

**Is the ordering an artefact of how finely each factor is divided?** A max-minus-min range widens as a factor gains groups — weather has only two and cannot produce a large range whatever its true effect. The group count and a standard deviation, which is far less sensitive to that, are shown beside the plotted range:

| Factor | Usable groups | Mean F1 range | Mean F1 std. dev. |
|---|--:|--:|--:|
| dataset | 10 | 40.8 | 12.0 |
| habitat | 11 | 20.1 | 6.6 |
| density | 8 | 16.8 | 5.6 |
| altitude | 9 | 14.6 | 5.1 |
| date | 4 | 12.0 | 5.3 |
| site | 3 | 10.1 | 5.4 |
| weather | 2 | 4.7 | 3.3 |

Two checks say the headline survives: habitat is divided into 11 usable groups against sub-dataset's 10, so sub-dataset's larger spread cannot come from having more groups to spread across; and ranking by standard deviation instead preserves the top three in order (dataset, habitat, density). What does change is the margin: on range sub-dataset leads habitat 40.8 to 20.1, on standard deviation 12.0 to 6.6. Treat the ordering as robust and the ratio between factors as indicative only.

### What this means

**Dataset identity dominates every survey variable measured.** It moves F1 by 40.8 points on average, against 20.1 for habitat and 14.6 for altitude. Sub-dataset is not one variable — it bundles site, camera, annotator, season and labelling convention — so this is the honest headline: **the largest single determinant of measured accuracy is which survey the imagery came from, not what was in it.** For a new deployment in Arkansas, that is the risk to plan around; it predicts a drop on first contact with new imagery larger than anything habitat or altitude selection can recover.

Ordered by what the evidence in this document supports:

1. **Domain shift between surveys** is the dominant term, and it is the least studied here because every model was trained and tested on the same ten datasets. Nothing in this analysis measures transfer to genuinely new imagery.

2. **Apparent bird size** is the dominant *within-image* term, and it is the mechanism behind most of the altitude story: birds shrink from 66 px at 15 m to 12 px at 120 m, and Task 2.3 showed recall falls with size. But the altitude relationship measured here runs the *opposite* way to the naive expectation — recall rises with height — because the training set is 44% 90 m imagery and only 12.5% 15 m. **That is a sampling artefact, not a physical finding**, and it is the most actionable item in this document: rebalancing the training altitudes is cheap and testable.

3. **Habitat complexity** is real and consistent in direction: structurally broken backgrounds (Trees, Wooded, ShrubScrub) score worst for every model, open water and ice best. The effect is smaller than dataset identity, and 7 of 13 habitats occur in a single sub-dataset, so the magnitude is not cleanly separable from domain shift.

4. **Density** drives counting error more than detection error. Dense crops are where absolute count error concentrates, and the failure library's dense-flock case shows the mechanism: birds are visible, boundaries between them are not.

5. **Background false positives** matter out of proportion to their rate. Empty crops are the majority of any survey mosaic, so a small per-crop false-positive rate becomes a large absolute overcount at mosaic scale.

6. **Weather, site and date** are measurable but small, and each is confined to one or two datasets. None supports a claim at corpus scale.

**The ranking question, answered.** No single model wins everywhere: different architectures take first place in different habitats, and the box and point families fail on different crops rather than on the same crops by different amounts. A single leaderboard number hides that, which is the strongest argument in this document for reporting conditional performance rather than one AP.
