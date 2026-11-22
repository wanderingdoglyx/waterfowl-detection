# `waterfowl_detection.sh` — User Manual

`waterfowl_detection.sh` starts the **Waterfowl Workbench**, a menu-driven terminal program
that covers the whole project from one place:

- bring in aerial imagery,
- train and evaluate the eight bird detectors,
- count birds in new survey photographs,
- review and correct those counts bird by bird,
- export count reports, corrected labels and model comparisons.

This manual explains how to do each of these. For how the workbench is built and why it
behaves the way it does, see [`workbench/README.md`](workbench/README.md).

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Starting the program](#2-starting-the-program)
3. [Moving around](#3-moving-around)
4. [The home screen](#4-the-home-screen)
5. [Walkthrough A — count the birds in a new flight](#5-walkthrough-a--count-the-birds-in-a-new-flight)
6. [Walkthrough B — train and compare models](#6-walkthrough-b--train-and-compare-models)
7. [Walkthrough C — turn reviewed counts into training data](#7-walkthrough-c--turn-reviewed-counts-into-training-data)
8. [Menu reference](#8-menu-reference)
9. [The reviewer](#9-the-reviewer)
10. [Background jobs](#10-background-jobs)
11. [Where files are saved and how to find them](#11-where-files-are-saved-and-how-to-find-them)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick reference card](#13-quick-reference-card)

---

## 1. Before you start

| You need                                                                         | Why                                                                                            |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| The`waterfowl` conda environment (see the main [README](README.md#environment)) | The workbench and most models run in it. It needs the`rich` and `prompt_toolkit` packages. |
| The OWL environment (Python 3.11), only for OWL-C / OWL-T / OWL-D                | The OWL models run in their own interpreter.                                                   |
| The imagery in`dataset_full/`                                                  | For training and evaluation. Download it as described in the main [README](README.md#data).     |
| A terminal of at least**110 × 32** characters                             | For the full-screen reviewer. Other screens work in smaller windows.                           |
| A GPU                                                                            | For training, evaluation and detection. Menus, review and reports work without one.            |

It works over plain SSH and in VS Code's integrated terminal.

---

## 2. Starting the program

From the repository root:

```bash
./waterfowl_detection.sh              # open the main menu
./waterfowl_detection.sh <screen>     # open one screen directly, then the main menu
./waterfowl_detection.sh --help       # list the options
./waterfowl_detection.sh --version    # print the version
```

`<screen>` is one of:

| Screen       | Opens                         |
| ------------ | ----------------------------- |
| `data`     | Import imagery & prepare data |
| `train`    | Train a model                 |
| `evaluate` | Evaluate & compare            |
| `detect`   | Detect birds in new imagery   |
| `review`   | Review & correct detections   |
| `export`   | Export reports                |
| `jobs`     | Jobs & logs                   |
| `settings` | Settings & health check       |

For example, `./waterfowl_detection.sh jobs` goes straight to the job list, which is the
quickest way to check on a training run.

You can run the script from any directory, or through a symlink. It always switches to
the repository root first.

### Which Python it uses

The script looks for a Python interpreter in this order:

1. The `WATERFOWL_PYTHON` environment variable, if set.
2. The active conda environment, if it is called `waterfowl`.
3. `envs/waterfowl/bin/python` under `~/anaconda3`, `~/miniconda3`, `~/miniforge3`,
   `~/mambaforge`, `/opt/conda`, or the installation `$CONDA_EXE` points to.
4. Otherwise `python3` from your `PATH`, with a warning.

To choose one yourself:

```bash
WATERFOWL_PYTHON=/path/to/envs/waterfowl/bin/python ./waterfowl_detection.sh
```

The equivalent without the script, from inside the `waterfowl` environment, is
`python -m workbench`.

### A desktop icon (optional)

On a machine with a graphical desktop, *Settings & health check → Create a desktop
launcher* adds **Waterfowl Workbench** to the applications menu, and optionally to the
Desktop. Clicking it opens a terminal window with the workbench running. If you move the
repository, create the launcher again.

---

## 3. Moving around

Every menu uses the same keys:

| Key                            | Action                                          |
| ------------------------------ | ----------------------------------------------- |
| `↑` `↓` (or `k` `j`) | move                                            |
| `1`–`9`                   | choose an option by its number                  |
| `Enter`                      | select                                          |
| `Esc`                        | go back one level (on the home screen: quit)    |
| `Space`                      | tick or untick, in lists where you pick several |
| `a`                          | tick all / none, in those lists                 |
| `Ctrl+C`                     | leave the current screen                        |

Each screen begins with a breadcrumb showing where you are, for example
`Waterfowl Workbench › Evaluate & compare › Leaderboard`.

If something goes wrong inside a screen, the workbench shows the error and returns to the
menu instead of closing. The full traceback is written to `output/workbench/errors.log`.

---

## 4. The home screen

The home screen shows the state of the project before it asks what to do next:

| Line              | Tells you                                                                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data**    | the active data folder, its number of collections and photographs.*(chosen in the workbench)* means it differs from the default in `data_prep/config.py`. |
| **Crops**   | whether the 512 px training crops are prepared.                                                                                                               |
| **Models**  | how many training runs exist, how many are evaluated, how many are training now, and the best point AP.                                                       |
| **GPUs**    | load and memory for each card.                                                                                                                                |
| **Jobs**    | workbench jobs running, plus pipeline processes started outside the workbench.                                                                                |
| **Surveys** | imported surveys, detection results, and how many are being reviewed.                                                                                         |

Below it are the eight menus described in [section 8](#8-menu-reference). Choosing
**Quit** (or `Esc`) leaves the program. Jobs keep running after you quit.

---

## 5. Walkthrough A — count the birds in a new flight

The everyday task: a drone survey comes back and you need a bird count you can trust.

**Step 1 — Import the photographs.**
*Import imagery & prepare data → Import survey photos to count birds in.*
Give the folder of photographs and a name for the survey. Choose **link** (the default,
uses no disk space) or **copy** (safer if the original folder may move). If label files
sit next to the photos, the survey is marked as *labelled*, and the report will score the
model against them.

*Saved in:* `output/workbench/surveys/<survey>/`. Your original photos are never moved or
changed.

**Step 2 — Count the birds.**
*Detect birds in new imagery → Count birds in a survey.*

1. Pick the survey.
2. Pick the model. Evaluated versions are listed with the best counting error first, and
   each shows an estimated run time. *Another version or checkpoint…* lists every trained
   version, including untested ones.
3. Accept the threshold. The default is that checkpoint's **t\***, the score cut that gave
   the lowest counting error on the test split.
4. Pick a GPU. The idlest one is pre-selected.

Detection runs as a background job (see [section 10](#10-background-jobs)). Each
photograph is cut into 512 × 512 tiles, every tile is scored, and birds on tile overlaps
are counted once.

*Saved in:* `output/workbench/detections/<date>_<time>_<survey>_<model>/`. The result's
page under *Detect birds in new imagery* shows this folder.

**Step 3 — Review the count.**
When the job finishes, choose *Review these detections* from the job page, or open
*Review & correct detections* and pick the result. Choose which cells to visit
(*Cells with detections* is the usual pass) and work through them in the reviewer
([section 9](#9-the-reviewer)). You can stop at any point. Every key press is saved.

*Saved in:* `review.json`, in the same detection folder as Step 2.

**Step 4 — Export the report.**
*Export reports → Bird count report for a survey.* The report goes to a new folder,
`reports/<date>_<survey>_<model>/`, for example
`reports/2026-09-29_lake_north_yolov5/`:

- `report.html` — the report to open or send: totals before and after review, how much was
  reviewed, the model and threshold, a table per photograph, and an annotated picture of
  each photograph (plus precision, recall and counting error for a labelled survey);
- `report.md` — the same report as plain text (Markdown);
- `counts_<survey>.csv` — one row per photograph, for Excel;
- `detections.csv` — every detection with its score and review decision;
- `images/` — the annotated pictures the report shows.

The program prints the folder when it's done and, in VS Code, offers to open the report.
[Section 11.3](#113-whats-inside-a-report-folder) describes every column.

> **Tip — test a model on whole photographs first.** *Import imagery → Sample held-out
> test photos as a survey* builds a labelled survey from test-split photographs the model
> has never seen. Running Steps 2–4 on it shows how well a model counts whole photographs
> before you rely on it for a real flight.

---

## 6. Walkthrough B — train and compare models

**Step 1 — Get the data ready** (once).

- If `dataset_full/` is empty, download it first (main [README](README.md#data)).
- *Import imagery → Prepare crops for all models.* This builds the 512 px crops, the
  train/validation/test splits and each model family's format. The home screen shows
  **Crops: ready** when it is done. *Saved in:* `crops/`, `output/crops_json/`,
  `yolov5_data/` and `mdo_data/`.
- Optional: *Choose which collections experiments use* builds a data folder with only
  some collections, and *Switch the active data folder* makes jobs read it.

**Step 2 — Train.**
*Train a model*, then pick one of the eight:

| Family                                  | Models                 |
| --------------------------------------- | ---------------------- |
| Ultralytics                             | YOLOv5, YOLO11, YOLO26 |
| Other box detectors                     | YOLO-NAS, Faster R-CNN |
| Point detectors (MegaDetector-Overhead) | OWL-C, OWL-T, OWL-D    |

Then choose:

- **what the run starts from** — the public pretrained weights, a trained version
  (fine-tuning), or a checkpoint file;
- the number of **epochs**, for models whose training command accepts it;
- whether to **evaluate when done**;
- the **GPU**.

The screen shows the full training plan, including hyperparameters from
`data_prep/config.py` and how long the previous run took, before anything starts.
Training runs as a background job. You can quit the workbench, close VS Code or lose the
SSH connection and it keeps going.

*Saved in:* `output/checkpoints/<model>/<date>_<time>/`, named after the moment training
started. This folder is the **version** the rest of the interface refers to. When the job
finishes, its page offers *Inspect the run this job made*.
[Section 11.4](#114-inside-a-trained-models-folder) lists what's inside.

An interrupted OWL run can be continued with *Train a model → Resume an OWL run*.

**Step 3 — Test.**
If you did not choose *evaluate when done*: *Evaluate & compare → Test a trained version*
→ model → version → checkpoint.

- **best** (the highest validation score) writes the version's official scores into the
  version's folder (for the OWL models, into its `eval/` subfolder).
- **last** (end of training) writes to `<version folder>/eval_last/` and appears as a
  separate *(last)* row, so it never replaces the official scores.

The quickest file to read is `eval_results.txt`, which holds the scores as plain text.

**Step 4 — Compare.**
*Evaluate & compare → Leaderboard* lists every evaluated version. Runs scored on the
older, smaller test split are shown in a separate table, since their numbers are not
directly comparable. From there you can:

- *Inspect a trained version* — its lineage, both checkpoints' scores, example panels;
- run *Zero-shot baselines* — each model before any training on waterfowl
  (saved in `output/baselines/`);
- run *Cross-model analysis* — tick one run per model to compare them in detail
  (saved in `output/analysis/<name>/`);
- run *Efficiency benchmarks* — speed and memory (saved in `output/efficiency/` or
  `output/analysis/bench_*/`).

*Export reports → Model comparison* saves the leaderboard as HTML, Markdown and CSV in
`reports/<date>_leaderboard/`.

---

## 7. Walkthrough C — turn reviewed counts into training data

Reviewed surveys can improve the models:

1. Review a survey thoroughly (Walkthrough A, Step 3). *Every cell, empty ones too* also
   catches birds the model missed.
2. *Export reports → Corrected labels for training.* By default only photographs whose
   cells are all reviewed are included. This writes a collection in the same format as
   `dataset_full/`, to `reports/<date>_<survey>_labels/<survey>_reviewed/`.
3. *Import imagery → Import labelled photos as a training collection* and give it that
   `<survey>_reviewed` folder. The collection is added to a data folder, either a new
   `dataset_<name>/` or an existing one you choose. `dataset_full/` itself is never
   changed.
4. *Prepare crops* again, then *Train a model → Start from: a trained version* to
   fine-tune an existing model on the new data.

A fine-tuned version is marked with **↳** in lists, and its page shows which version it
came from.

> **Note:** The YOLOv5/11/26 models often score lower after the first fine-tuning epoch
> and recover over the following epochs. Judge a fine-tune by its *best* checkpoint, not
> its first epochs. See [`workbench/README.md`](workbench/README.md#versions-fine-tuning-and-checkpoints)
> for measured numbers.

---

## 8. Menu reference

### 1 · Import imagery & prepare data

| Option                                          | What it does                                                          |
| ----------------------------------------------- | --------------------------------------------------------------------- |
| Data overview                                   | collections, crops, experiment selections and surveys at a glance     |
| Import survey photos to count birds in          | registers a folder of photographs as a survey                         |
| Sample held-out test photos as a survey         | builds a labelled survey from test-split photographs                  |
| Import labelled photos as a training collection | adds photos with`bird,x1,y1,x2,y2` label files to the training data |
| Choose which collections experiments use        | builds a data folder from the collections you tick                    |
| Switch the active data folder                   | chooses which data folder every job reads                             |
| Prepare crops for all models                    | 512 px crops, COCO splits and each model's format                     |
| Check the data                                  | describe · coverage · conformance · verify                         |
| Manage imported surveys                         | list, inspect, delete                                                 |

### 2 · Train a model

Pick a model to train it, *Fine-tune a trained version*, or *Resume an OWL run*.
See [Walkthrough B](#6-walkthrough-b--train-and-compare-models).

### 3 · Evaluate & compare

Leaderboard · Inspect a trained version · Test a trained version · Zero-shot baselines ·
Cross-model analysis · Efficiency benchmarks.

### 4 · Detect birds in new imagery

Lists earlier detection results (survey, model, photos done, birds counted, state, cells
reviewed). *Count birds in a survey* starts a new count. *Import a new survey first* is a
shortcut to the import screen. Choosing an earlier result opens it.

### 5 · Review & correct detections

Pick a detection result, then which cells to visit:

| Choice                     | Use it for                                                                       |
| -------------------------- | -------------------------------------------------------------------------------- |
| Cells with detections      | the usual pass                                                                   |
| Borderline cells first     | cells with a score within ±0.15 of the threshold, where the model is least sure |
| Cells not reviewed yet     | picking up where you left off                                                    |
| Every cell, empty ones too | an audit that also finds birds the model missed entirely                         |

### 6 · Export reports

| Option                         | Produces                                                                     | Saved in                              |
| ------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------- |
| Bird count report for a survey | HTML, Markdown, CSV and annotated photographs                                | `reports/<date>_<survey>_<model>/`  |
| Corrected labels for training  | the reviewed birds as a`dataset_full`-format collection plus `coco.json` | `reports/<date>_<survey>_labels/`   |
| Model comparison               | every evaluated run, with its caveats                                        | `reports/<date>_leaderboard/`       |
| One version's report           | lineage, the scores of each checkpoint, example panels                       | `reports/<date>_<model>_<version>/` |
| Open the reports folder        | shown only in VS Code or on a desktop                                        | —                                    |

Details of every file are in [section 11.3](#113-whats-inside-a-report-folder).

### 7 · Jobs & logs

See [section 10](#10-background-jobs).

### 8 · Settings & health check

| Option                    | What it does                                                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| Active data folder        | same as*Switch the active data folder*                                                      |
| Preferred GPU             | pre-select a card for new jobs, or the idlest one at launch                                   |
| Image viewer              | how images and reports open: VS Code, the desktop viewer, or not at all (the path is printed) |
| Health check              | checks the environments, GPUs, weights, data and disk space                                   |
| Self-test                 | checks the workbench's own logic in seconds, without a GPU                                    |
| Create a desktop launcher | adds the workbench to the applications menu                                                   |
| About                     | version and where things are kept                                                             |

---

## 9. The reviewer

The reviewer shows one **cell** at a time, the part of a photograph one tile is
responsible for, plus a thin margin of its neighbours and a magnified **loupe** on the
selected detection. Detections are listed least confident first, since that is where the
model's mistakes usually are. Maximise the terminal before opening it.

**Marker colours**

| Marker             | Meaning                                   |
| ------------------ | ----------------------------------------- |
| amber ring         | counted by the model, not reviewed yet    |
| green ring         | you confirmed it is a bird                |
| red ring           | you rejected it — not a bird             |
| grey ring          | below the threshold (shown with`b`)     |
| cyan ring          | a bird you added by hand                  |
| magenta dot or box | an annotated bird (labelled surveys only) |

**Keys**

| Key                        | Action                          | Key           | Action                                                             |
| -------------------------- | ------------------------------- | ------------- | ------------------------------------------------------------------ |
| `↑` `↓` `k` `j`  | select a detection              | `y` / `n` | bird / not a bird                                                  |
| `Enter`                  | cell done, go to the next       | `c`         | clear the decision                                                 |
| `→` `←`              | next / previous cell            | `a`         | add a missed bird (arrows move the cross, Enter places it)         |
| `]` `[`                | next / previous photograph      | mouse click   | select a marker, or add a bird on an empty spot                    |
| `+` `-` or mouse wheel | zoom the loupe                  | `x` `Del` | delete a bird you added                                            |
| `b`                      | show below-threshold detections | `g`         | show/hide annotations                                              |
| `<` `>`                | threshold −/+ 0.05             | `T`         | reset the threshold to t\*                                         |
| `u`                      | undo                            | `o`         | open a sharp full-resolution view in VS Code or the desktop viewer |
| `?`                      | list every key                  | `q` `Esc` | save and leave                                                     |

**How the count is worked out**

    count = detections ≥ threshold − rejected + below-threshold confirmed + added

A detection you never decide on keeps the model's verdict, so a partly reviewed survey
still has a sensible count. If a cell is already right, one `Enter` accepts it. Reports
say how many cells were checked by a person.

**Saving.** Decisions are written to disk after every key press. Closing the terminal
loses nothing. Reopen the result and choose *Cells not reviewed yet* to continue.

**Image quality.** The terminal picture is coarse. Press `o` for a sharp view of the
current cell; that view then follows you as you move through cells.

---

## 10. Background jobs

Anything that takes more than a few seconds (preparing crops, training, evaluation,
detection, analysis) runs as a **background job** with its own log. Quitting the
workbench, closing VS Code or losing the SSH connection does not stop it.

*Jobs & logs* lists every job and its status:

| Status    | Meaning                                               |
| --------- | ----------------------------------------------------- |
| running   | still working                                         |
| succeeded | finished normally                                     |
| failed    | finished with an error; the exit code is shown        |
| cancelled | you stopped it                                        |
| stopped   | it died without reporting, for example after a reboot |

Pipeline processes started outside the workbench, such as a training script run by hand,
are listed as well.

Opening a job shows its command, including the environment variables it ran with, and the
latest lines of its log. You can then:

- **Follow the log live** (refreshes every second);
- **Page through the whole log** (`q` to return);
- **Open the log** in VS Code or a viewer;
- **Inspect the run this job made** / **Review these detections**, when it succeeded;
- **Cancel this job** — stops it and everything it started, including data-loader
  workers;
- **Run the same command again**;
- **Remove this job's record and log**.

**GPUs.** Each job uses one GPU. The picker shows every card's load and pre-selects the
idlest. Choosing a busy card asks you to confirm.

**Which data a job reads.** Jobs read the *active data folder*. The workbench never edits
`data_prep/config.py`. When the active folder differs from the default, it passes
`WATERFOWL_DATASET=<folder>` to the job, and the job's log shows it.

---

## 11. Where files are saved and how to find them

### 11.1 The project folder

Every path in this manual is relative to the **project folder**: the folder that holds
`waterfowl_detection.sh`. On the lab server that is `/home/zygfd/waterfowl/detection`.
The program always works from there, whichever directory you start it from. Everything
it saves goes inside this folder, apart from the optional desktop launcher.

```
<project folder>/
├── reports/                  everything you export: count reports, labels, comparisons
├── output/
│   ├── workbench/            the interface's own records: surveys, detections, reviews, jobs
│   ├── checkpoints/          trained models and their test scores
│   ├── baselines/            zero-shot baseline results
│   ├── analysis/             cross-model analysis and benchmarks
│   ├── efficiency/           the Ultralytics efficiency benchmark
│   └── crops_json/           train/validation/test lists made by Prepare crops
├── crops/                    512 px training crops (made by Prepare crops)
├── yolov5_data/  mdo_data/   the same crops laid out for the YOLO and OWL models
├── dataset_full/             the imagery
└── dataset_<name>/           data folders you create
```

In folder names, `<date>` is written `2026-09-29` and `<date>_<time>` is written
`2026-09-29_14-05-33`.

### 11.2 Where each action saves

**Counting birds**

| What you did                | Where to look                                                   | What's there                                                                                                                                |
| --------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Import survey photos        | `output/workbench/surveys/<survey>/`                          | `survey.json` (the list of photos), `images/` (links to your photos, or copies), `labels/` (only if the photos came with label files) |
| Sample held-out test photos | `output/workbench/surveys/heldout_sample_<n>/` (default name) | the same, always with labels                                                                                                                |
| Count birds in a survey     | `output/workbench/detections/<date>_<time>_<survey>_<model>/` | `detect.json` (model, threshold, settings) and `detections.jsonl` (every detection, one line per photo)                                 |
| Review                      | the same detection folder                                       | `review.json`, your decisions, saved after every key press                                                                                |
| `o` in the reviewer       | `output/workbench/tmp/review_preview.png`                     | the sharp picture of the current cell, replaced as you move                                                                                 |
| Export anything             | `reports/<date>_<name>/`                                      | see[11.3](#113-whats-inside-a-report-folder)                                                                                                 |

**Training and testing**

| What you did                       | Where to look                                                                                                                                                                      | What's there                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Train a model                      | `output/checkpoints/<model>/<date>_<time>/`                                                                                                                                      | see[11.4](#114-inside-a-trained-models-folder)                                                                             |
| Test a version,*best* checkpoint | the version's folder (OWL models: its`eval/` subfolder)                                                                                                                          | the version's official scores                                                                                             |
| Test a version,*last* checkpoint | `<version folder>/eval_last/`                                                                                                                                                    | the last checkpoint's scores                                                                                              |
| Zero-shot baselines                | `output/baselines/<model>/<date>_<time>/`; the before/after table in `output/baselines/summary.txt`                                                                            | scores of each model before training                                                                                      |
| Cross-model analysis               | `output/analysis/<name>/`                                                                                                                                                        | tables (`.csv`, `.md`) and `figures/`. The runs you ticked are listed in `output/workbench/manifests/<name>.txt`. |
| Efficiency benchmarks              | Ultralytics:`output/efficiency/<date>_<time>/`; all eight models: `output/analysis/bench_extra/<date>_<time>/`; training speed: `output/analysis/bench_train/<date>_<time>/` | timings and memory (`efficiency.txt` and `.json`, `bench_extra.json`, `bench_train.json`)                         |

**Data**

| What you did                                    | Where to look                                                                                                                                                                                                                                           |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prepare crops                                   | `crops/<collection>/` (the crop images), `output/crops_json/{train,val,test}/coco.json` (the splits), `yolov5_data/`, `mdo_data/`. With a data folder other than `dataset_full` active, the subset is kept in `output/experiments/<name>/`. |
| Choose which collections experiments use        | `dataset_<name>/`, holding shortcuts to the collections you ticked                                                                                                                                                                                    |
| Import labelled photos as a training collection | `dataset_<name>/<collection>/` for a new data folder, or `<data folder>/<collection>/` for an existing one                                                                                                                                          |

**The interface itself**

| What                    | Where                                                                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Any background job      | `output/workbench/jobs/<date>_<time>_<kind>/`: `job.json` (the exact command), `log.txt` (everything the job printed), `exit_code` (once it has finished) |
| Settings                | `output/workbench/settings.json`, created the first time you change a setting                                                                                   |
| Errors in the interface | `output/workbench/errors.log`                                                                                                                                   |
| Desktop launcher        | `~/.local/share/applications/waterfowl-workbench.desktop`, and a copy in `~/Desktop/` if you asked for one                                                    |

### 11.3 What's inside a report folder

Each export makes a **new** folder in `reports/`. Nothing is overwritten: exporting the
same thing again on the same day creates `…-2`, then `…-3`.

**Bird count report** — `reports/<date>_<survey>_<model>/`

| File                    | Contents                                                                                                                                                                                                                                                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report.html`         | The report to open or send. Open it in any web browser.                                                                                                                                                                                                                                                   |
| `report.md`           | The same report as Markdown text. In VS Code, press Ctrl+Shift+V to see it formatted.                                                                                                                                                                                                                     |
| `counts_<survey>.csv` | One row per photo:`photo`, `width`, `height`, `cells`, `cells_reviewed`, `model_count`, `rejected`, `promoted` (below-threshold detections you confirmed), `added` (birds you added), `reviewed_count` (the final count), and `annotated_count` for a labelled survey.              |
| `detections.csv`      | One row per detection:`photo`, `x`, `y` (centre, in pixels), `w`, `h` (box size; empty for point models), `score`, `source` (`model`, or `reviewer` for a bird you added), `decision` (`bird`, `not_bird`, or empty if not reviewed), `counted` (1 if it's in the final count). |
| `images/<photo>.jpg`  | Each photo with markers drawn on: amber = counted, not reviewed; green = confirmed; red cross = rejected; cyan = added by you.                                                                                                                                                                            |

`report.html` shows the pictures from the `images/` folder next to it, so keep the folder
together when you copy or send it.

**Corrected labels** — `reports/<date>_<survey>_labels/`

| File                   | Contents                                                                                                                                                                    |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<survey>_reviewed/` | The new training collection: each photo (a shortcut to the survey's photo), one`<photo>.txt` per photo with a `bird,x1,y1,x2,y2` line per bird, and `image_info.csv`. |
| `coco.json`          | The same boxes in COCO format.                                                                                                                                              |
| `README.md`          | What was exported: how many photos and birds, from which result and threshold, and how boxes were sized for birds that had only a point.                                    |

**Model comparison** — `reports/<date>_leaderboard/`: `leaderboard.html`,
`leaderboard.md` and `leaderboard.csv`, one row per evaluated version.

**One version's report** — `reports/<date>_<model>_<version>/`: `report.html`,
`report.md` and `examples/`, up to six example pictures.

### 11.4 Inside a trained model's folder

A version lives in `output/checkpoints/<model>/<date>_<time>/`. `<model>` is one of
`yolov5`, `yolo11`, `yolo26`, `yolonas`, `fasterrcnn`, `mdo_owl_c`, `mdo_owl_t` or
`mdo_owl_d`, and the date and time are when training started.

| File                                   | What it is                                                                                                                                                      |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval_results.txt`                   | The test scores as plain text: AP, precision, recall, F1, counting error (MAE, RMSE) and t\*. The quickest file to read. OWL models: `eval/eval_results.txt`. |
| `paper_point_metrics.json`           | The same scores for programs, with 95% intervals. OWL models:`eval/metrics.json`.                                                                             |
| `examples/`                          | Example test crops with the model's detections drawn on.                                                                                                        |
| `run_info.json`, `experiment.json` | How the run was made: model, starting weights, data, epochs, and where it was fine-tuned from.                                                                  |
| `eval_last/`                         | The last checkpoint's scores, if you tested it.                                                                                                                 |

The model weights themselves:

| Family                 | best checkpoint (used by default)                              | last checkpoint                                         |
| ---------------------- | -------------------------------------------------------------- | ------------------------------------------------------- |
| YOLOv5, YOLO11, YOLO26 | `weights/best.pt`                                            | `weights/last.pt`                                     |
| YOLO-NAS               | `RUN_<…>/ckpt_best.pth`                                     | `RUN_<…>/ckpt_latest.pth`                            |
| Faster R-CNN           | `model_best.pth`                                             | `model_final.pth`, or the newest `model_<step>.pth` |
| OWL-C, OWL-T, OWL-D    | `weights/best.pth` (`best_model.pth` while still training) | `latest_model.pth`                                    |

### 11.5 Finding a file

- **The program tells you.** After every save it prints the path, for example
  `✓ Wrote reports/2026-09-29_lake_north_yolov5/`, followed by the files it wrote.
- **Open it from the menu.** In VS Code's terminal (or on a desktop), each export offers to
  open what it just wrote, and *Export reports → Open the reports folder* opens the whole
  folder. A detection result's page shows its *Folder*. A finished training job offers
  *Inspect the run this job made*.
- **In VS Code**, the Explorer side bar shows the project folder. Expand `reports/` or
  `output/`.
- **From a terminal**, list the newest first:
  ```bash
  ls -t reports/ | head                   # latest exports
  ls -t output/checkpoints/yolo11/ | head  # latest YOLO11 versions
  ```

### 11.6 Copying results to your own computer

The files are on the server. To get a copy:

- **VS Code:** in the Explorer side bar, right-click a file or folder → *Download…*.
- **Any terminal on your computer** (Windows PowerShell, macOS Terminal or Linux), with
  your own user name, server and project folder:
  ```bash
  scp -r <user>@<server>:<project folder>/reports/2026-09-29_lake_north_yolov5 .
  ```

Copy the whole report folder, not only `report.html`, so the pictures come along. The
photos in a *Corrected labels* export are shortcuts. To take that collection to another
computer, first make a copy with real files on the server
(`cp -rL reports/<folder> ~/labels_copy`) and download that.

### 11.7 What's kept in git, and what you can delete

| Location                                                             | In git                                              | Can it be deleted?                                                                                    |
| -------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `reports/`                                                         | only if you commit it                               | Yes. Export again to recreate a report, as long as its detection result still exists.                 |
| `output/workbench/surveys/`                                        | no                                                  | Use*Import imagery → Manage imported surveys → delete*. Your original photos are not touched.     |
| `output/workbench/detections/`                                     | no                                                  | Only when you're finished with it:`review.json` holds your review decisions and can't be recreated. |
| `output/workbench/jobs/`                                           | no                                                  | Use*Jobs & logs → Remove this job's record and log*.                                               |
| `output/checkpoints/`                                              | scores and logs yes, weights (`.pt`, `.pth`) no | No. The weights exist only here and on the Hugging Face checkpoints repository.                       |
| `crops/`, `yolov5_data/`, `mdo_data/`, `output/experiments/` | no                                                  | Yes.*Prepare crops* rebuilds them.                                                                  |

Everything under `output/workbench/` belongs to this machine and is never committed.

---

## 12. Troubleshooting

| Symptom                                                   | Fix                                                                                                                                                  |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `note: the 'waterfowl' conda environment was not found` | Create the environment (main[README](README.md#environment)), or point `WATERFOWL_PYTHON` at the right interpreter.                                 |
| `The workbench needs 'rich' and 'prompt_toolkit'`       | Run the`pip install` command it prints.                                                                                                            |
| `Permission denied` when starting                       | `chmod +x waterfowl_detection.sh`                                                                                                                  |
| `(not a terminal: menus fall back to numbered prompts)` | Input is not an interactive terminal, e.g. piped. Run it directly in a terminal.                                                                     |
| The reviewer looks cramped or cut off                     | Enlarge the terminal to at least 110 × 32.                                                                                                          |
| **Crops: not prepared** on the home screen          | *Import imagery & prepare data → Prepare crops for all models.*                                                                                   |
| A job shows**stopped**                              | It was killed or the machine restarted. Open it and choose*Run the same command again* (OWL runs: *Train → Resume an OWL run*).                 |
| A job**failed**                                     | Open it and page through the log. The exact command is shown, so you can rerun it by hand to debug.                                                  |
| An OWL model ignored the data folder I chose              | Known limitation: OWL-C/T/D always use the full corpus. The training and test screens warn about this.                                               |
| `o` in the reviewer or *Open…* options are missing   | No viewer was found. Set one under*Settings → Image viewer*, or open the printed path yourself.                                                   |
| I can't find a file I exported                            | Exports are in`reports/`, newest first with `ls -t reports/ \| head`. [Section 11.2](#112-where-each-action-saves) lists where every action saves. |
| The desktop icon stopped working                          | The repository moved or the script was renamed. Run*Settings → Create a desktop launcher* again.                                                  |
| Anything else                                             | Run*Settings → Health check* and *Self-test*, and check `output/workbench/errors.log`.                                                        |

---

## 13. Quick reference card

```
START      ./waterfowl_detection.sh [data|train|evaluate|detect|review|export|jobs|settings]
PYTHON     WATERFOWL_PYTHON=/path/to/python ./waterfowl_detection.sh

MENUS      ↑↓ / j k move · 1-9 choose · Enter select · Esc back · Space tick · a all

COUNT      Import imagery → Import survey photos
           Detect birds → Count birds in a survey
           Review & correct → Cells with detections
           Export → Bird count report

TRAIN      Import imagery → Prepare crops
           Train a model → <model>
           Evaluate & compare → Test a trained version → Leaderboard

REVIEWER   y/n bird or not · Enter cell done · a add · u undo · b below-threshold
           </> threshold · T reset t* · o sharp view · ? all keys · q save & leave

SAVED IN   reports/<date>_<name>/                      exports (count reports, labels, comparisons)
           output/checkpoints/<model>/<date>_<time>/   trained models and their scores
           output/workbench/surveys|detections|jobs/   surveys, counts + reviews, job logs
FIND       the path is printed after every save · ls -t reports/ | head
```
