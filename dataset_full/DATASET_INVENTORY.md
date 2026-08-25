# MDC Waterfowl Dataset Inventory

Complete metadata inventory of the ten sub-datasets in `dataset_full/`, compiled to
determine which subgroup analyses the data can actually support.

**Method.** Every value was measured directly from the files: image counts by extension,
bird counts by parsing annotation files through the project's own loader, resolution and
EXIF via PIL, and filename-encoded fields by parsing the filenames. **Nothing is
inferred.** Absent fields are recorded as unavailable rather than guessed; partial fields
carry their covered fraction.

Compiled 24 August 2026 · 10 datasets · 2,095 original images · 317,387 labelled birds.

---

## 1. Dataset totals

| Dataset | Original images | Birds | Resolution | Flight altitudes (m) |
|---|---|---|---|---|
| Bird_A | 161 | 8,952 | 5120×3584 | 40–150 |
| Bird_B | 36 | 2,175 | 5472×3648 | 28–120 |
| Bird_C | 104 | 963 | 5472×3648 | 30/60/90/120 |
| Bird_D | 339 | 28,800 | 5760×3840, 5472×3648 | 90 |
| Bird_E | 694 | 79,424 | 512×512 (pre-tiled) | 90 |
| Bird_F | 26 | 1,936 | 5472×3648 | 90 |
| Bird_G | 125 | 62,742 | 5472×3078, 5472×3648 | 14–90 |
| Bird_H | 171 | 16,560 | 5472×3648 | 15/30/60/90 |
| Bird_I | 171 | 7,038 | 5472×3648, 4000×3000 | 15 |
| Bird_J | 268 | 108,797 | 5472×3648, 5472×3078 | 15/30/60/90 |
| **Total** | **2,095** | **317,387** | | **14–150** |

Crop counts are not listed per dataset because crops are a derived artifact: `--prepare`
tiles these images into 512×512 crops with 20% overlap, and the count depends on the tiling
settings rather than on the source data. Bird_E is already distributed as 512×512 tiles and
is not re-tiled.

---

## 2. Availability matrix

`Y` available for the whole dataset · `P` partial (fraction given) · `—` not available

| Information | A | B | C | D | E | F | G | H | I | J |
|---|---|---|---|---|---|---|---|---|---|---|
| Number of original images | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Number of image crops | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Number of labeled birds | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Bounding boxes / points | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Species labels | — | — | — | — | — | — | — | **Y** | **Y** | — |
| Sex labels | — | — | — | — | — | — | — | **Y** | **Y** | — |
| Habitat | — | — | — | — | — | — | **Y** | **Y** | — | **Y** |
| Flight altitude | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Drone platform | — | **Y** | P | — | — | P | **Y** | P | P 30/171 | P |
| Camera / sensor | — | **Y** | — | — | — | — | **Y** | — | P 30/171 | — |
| Image resolution | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Survey date | — | **Y** | — | **Y** | **Y** | — | **Y** | **Y** | P 30/171 | — |
| Geographic / site identifier | — | **Y** | — | **Y** | **Y** | — | **Y** | **Y** | P 30/171 | — |
| Decoy vs. real bird | — | — | — | — | — | — | — | — | — | — |
| Environmental conditions | — | — | — | — | — | — | **Y** | — | — | **Y** |

### Where each field comes from

- **Bounding boxes** — every dataset annotates with axis-aligned boxes, one line per bird.
  Two layouts occur and both are read: `bird,x1,y1,x2,y2` (most files) and a bare
  `x1,y1,x2,y2` used by part of Bird_I. No dataset uses point annotations; the point
  metrics elsewhere in this project derive points from box centres.
- **Flight altitude** — the `height` column of each `image_info.csv`, populated for
  **100% of rows in all ten datasets**. The single most reliable covariate available,
  spanning 14–150 m. Six datasets vary altitude internally (A, B, C, G, H, J), which is
  what makes within-dataset comparison possible. Independently confirmed in the filenames
  of A (`140_1.jpg`), C (`_120m`), G (`_15m_`), H (`_30meters_`) and J (`_60m_`).
- **Species and sex** — `*_class.txt` files in **Bird_H** and **Bird_I** only, same box
  format with a species label in place of `bird`:
  `Pintail Male,4758,2185,4816,2212`. Sex is embedded in the species string
  (`Mallard Male`, `Shoveler_Female`) rather than held as its own field, so it must be
  parsed out and exists only for sexually dimorphic species.
  - Bird_H: 16,560 labelled birds, 11 distinct labels — Green-winged teal (2,270),
    Gadwall (1,853), Mallard M/F (1,473/1,080), Shoveler M/F, Pintail M/F, American
    Widgeon M/F, and **Unknown (3,551, 21%)**. Sex: 5,212 M, 3,674 F, 7,674 unspecified.
  - Bird_I: 7,006 labelled birds, 30 distinct labels — Mallard M/F, Pintail M/F,
    Ring-necked duck, White-fronted Goose, Canada Goose, Gadwall, Shoveler, Canvasback,
    Coot, Snow/Ross, Ruddy, Hooded Merganser, and **Unknown (470, 7%)**.
    Sex: 2,804 M, 1,219 F, 2,983 unspecified.
- **Habitat** — three sources, all categorical:
  - Bird_H: recorded in a **dedicated column** of its `image_info.csv`, which makes it the
    most reliable of the three — MoistSoil 46, OpenWater 34,
    StandingCorn 32, Trees 28, ShrubScrub 21, Wooded 10 images. Repeated in filenames.
  - Bird_G: filename token 2 — Ice, Land, OpenWater, Rocks, WaterCorn, WaterSubVeg,
    WaterSubVege, WaterVegetation. *`WaterSubVeg` and `WaterSubVege` are almost certainly
    one class spelled two ways and must be merged before use.*
  - Bird_J: filename token 2 — Dense, HarvestedCrop, Land, Lotus, MoistSoil, OpenWater,
    ShrubScrub, StandingCorn, Wooded. *`Dense` describes bird density, not habitat, so
    this vocabulary mixes two concepts.*
- **Environmental conditions** — sun/cloud only, from filenames only, in **Bird_G**
  (Sun 68, Cloud 57) and **Bird_J** (Cloud 146, Sun 122). No wind, temperature, water
  level or light-angle data exists in any dataset.
- **Survey date** — three sources of differing precision:
  - EXIF `DateTimeOriginal`, to the second: Bird_B (36/36, September 2020), Bird_G
    (125/125, January 2021), Bird_I (30/171).
  - Filenames, to the day: Bird_D (`G135_11Nov2016_0288`), Bird_H (`_03082022_`, four
    survey dates: 10252021, 11302021, 03032022, 03082022).
  - Filenames, to the month: Bird_E (`mar2019_`).
- **Geographic / site identifier** — GPS coordinates in EXIF for Bird_B, Bird_G and 30
  Bird_I images. Named or coded sites in filenames for Bird_H (EagleBluffs, TedShanks,
  TenMilePond), Bird_D (plot codes G135, G144, G164, …) and Bird_E (MODOC).
- **Camera / sensor** — EXIF `Make`/`Model` only. Bird_B and Bird_G are 100% Hasselblad
  L1D-20c; Bird_I has 30/171 tagged L1D-20c or FC3170. All others have no EXIF (Section 4).
- **Drone platform** — full identification only via EXIF, as above. Datasets C, F, H, I
  and J carry `DJI_` filename prefixes, which identify the manufacturer's default file
  naming but **not the airframe**; marked `P` and not usable as a platform label.

---

## 3. Availability tiers

### 3.1 Reliably available — every dataset, usable now

- Number of original images, image crops, and labelled birds
- Bounding-box annotations
- **Flight altitude** (`height`, 100% coverage across all ten datasets)
- Image resolution
- The annotators' own train/test assignment for every image

### 3.2 Available for only some datasets

| Field | Datasets | Coverage |
|---|---|---|
| Species | H, I | 23,566 birds — **7.4%** of all birds |
| Sex | H, I | 12,909 sexed of 23,566 labelled (55%) |
| Habitat | G, H, J | 564 images — **27%** of images |
| Weather (sun/cloud) | G, J | 393 images — **19%** of images |
| Survey date | B, D, E, G, H, + 30 of I | 1,395 images — **67%** of images |
| Site identifier | B, D, E, G, H, + 30 of I | as above |
| Camera model / GPS | B, G, + 30 of I | 191 images — **9%** of images |

### 3.3 Potentially recoverable from originals / EXIF

- **GPS → named site, for Bird_B, Bird_G and 30 Bird_I images.** Coordinates are present
  and readable; mapping them to named MDC sites needs a site-boundary reference this
  project does not hold.
- **Ground sample distance.** Bird_E filenames carry `0015GSD`, apparently a GSD encoding
  but undocumented and unverified. Where EXIF exists, GSD could be derived from focal
  length (10.26 mm recorded), sensor size and altitude.
- **Exact capture time** for the 191 EXIF-bearing images, enabling time-of-day analysis.
- Nothing further is recoverable for A, C, D, E, F, H, J or 141 of Bird_I: those images
  carry **no EXIF whatsoever**, so date, camera and location cannot be read back from the
  files as held.

### 3.4 Unavailable

- **Decoy versus real bird.** No marker exists anywhere — not in filenames, not in any CSV
  column, not in any annotation file. If decoys appear in these surveys, the data as held
  cannot distinguish them.
- **Environmental conditions** beyond the sun/cloud binary in Bird_G and Bird_J.
- **Species and sex for 92.6% of birds** — everything outside Bird_H and Bird_I.
- **Observer / annotator identity**, annotation date, inter-annotator agreement.
- **Bird behaviour or posture** (flying, swimming, roosting).
- **Camera, GPS and date for the seven datasets with no EXIF**, unless original
  unprocessed files survive elsewhere.

---

## 4. EXIF status

| Dataset | Images | With EXIF | GPS | DateTime | Camera model |
|---|---|---|---|---|---|
| Bird_B | 36 | 36 (100%) | 36 | 36 | Hasselblad L1D-20c |
| Bird_G | 125 | 125 (100%) | 125 | 125 | Hasselblad L1D-20c |
| Bird_I | 171 | 30 (18%) | 30 | 30 | FC3170, L1D-20c |
| A, C, D, E, F, H, J | 1,763 | **0** | 0 | 0 | — |

EXIF survives in only **191 of 2,095 images (9%)**. The metadata was almost certainly
stripped by earlier processing — Bird_E is distributed as clipped PNG tiles, and the
partial survival within Bird_I (exactly the 30 images that also use the second annotation
layout) indicates its images reached the archive by two different routes.

**Whether the original camera files still exist upstream at MDC is the single
highest-value question to put to the data providers.**

---

## 5. Which subgroup analyses are supported

**Fully supported — all 2,095 images / 317,387 birds**

- **Performance versus flight altitude.** By far the strongest analysis available.
  Altitude is 100% covered, spans 14–150 m, and six datasets vary it *internally*
  (A, B, C, G, H, J), permitting within-dataset comparison not confounded by site or date.
- Performance versus bird density per crop, and versus image resolution.

**Supported on a subset — report as subset findings, not corpus-wide**

- **Habitat** (G, H, J — 564 images, 27%). Bird_H is soundest: its habitat comes from a
  real column and its site, date and altitude are all known, so habitat can be partly
  separated from confounders. G and J rely on filename tokens needing cleanup first.
- **Species and sex** (H, I — 23,566 birds, 7.4%). Enough for per-species detection rates
  on common species; note Bird_H is 21% `Unknown` and Bird_I 7%. Sex analysis is limited
  to dimorphic species and cannot be generalised.
- **Weather** (G, J only), **survey date / seasonality** and **site** (67% of images).
- **Camera / platform** (B, G, + 30 of I — 9% of images, two camera models). Thin enough
  that any finding is confounded with everything else distinguishing those datasets.

**Not supported**

- **Decoy versus real bird** — no data exists. Would require new annotation or a field
  record from MDC.
- Species or sex analysis outside Bird_H and Bird_I.
- Any environmental analysis beyond sun/cloud.

**Principal confounder.** Habitat, weather, species and site each vary *between* datasets
far more than within them, so a difference attributed to habitat may really be a
difference between Bird_G and Bird_J — different sites, dates, cameras and annotators.
Only altitude is cleanly manipulable within datasets. Subgroup findings on these fields
should be reported per dataset before any pooling.

---

## 6. Recommended follow-up with MDC

1. Do the **original unprocessed image files** still exist? That determines whether EXIF —
   and with it date, GPS and camera — is recoverable for the 1,904 images now missing it.
2. Are **decoys** present in any of these surveys, and is there a field record identifying
   which images or locations contain them?
3. Can the **plot codes in Bird_D** (G135, G144, …) and the **MODOC identifier in Bird_E**
   be resolved to named sites and coordinates?
4. Is `0015GSD` in the Bird_E filenames a ground sample distance, and in what units?
5. Are species labels available for any dataset beyond Bird_H and Bird_I, and what does
   `Unknown` denote — unidentifiable in the image, or not yet reviewed?
