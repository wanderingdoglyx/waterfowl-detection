# Waterfowl Workbench — the terminal interface

One program for the whole workflow: bring imagery in, train and evaluate the eight
detectors, count birds in new survey photographs, review and correct those counts, and
export reports.

```bash
./waterfowl_detection.sh        # the main menu
./waterfowl_detection.sh detect # open one screen directly (data, train, evaluate, detect,
                                #   review, export, jobs, settings), then the main menu
python -m workbench             # the same, from the waterfowl conda environment
```

It runs in the `waterfowl` environment and installs nothing: menus and questions are
`prompt_toolkit`, everything printed is `rich`, and both were already there. It works
over plain SSH and in VS Code's terminal. For the reviewer, maximise the terminal:
110 × 32 or larger. On a machine with a desktop, *Settings → Create a desktop launcher*
adds it to the applications menu. Clicking it opens a terminal window with the
workbench running.

Arrow keys (or `j`/`k`) move, a number key chooses, Enter confirms, Esc goes back. Every
screen starts with where you are, e.g. `Waterfowl Workbench › Evaluate & compare ›
Leaderboard`.

## The workflow

```
 Import imagery ──► Train ──► Evaluate & compare ──► Export (comparison, run reports)
      │                             │
      │ survey photos               │ a trained run
      ▼                             ▼
   Survey ───────────────────► Detect birds ──► Review & correct ──► Export
                                (counts)         (fixes the counts)    (count report,
                                                                        corrected labels)
                                                                           │
            Import labelled photos as a training collection  ◄─────────────┘
```

The research side (train → evaluate → compare) uses the project's existing entry points,
unchanged. The operational side (a new flight → a count someone can stand behind) is
what the workbench adds. The two meet in the loop at the bottom: reviewed counts can be
exported as labels and imported as training data.

## Menu map

| Home | What it offers | What it runs |
|---|---|---|
| **1 Import imagery & prepare data** | data overview · import survey photos · sample held-out test photos as a labelled survey · import labelled photos as a training collection · choose collections · switch the active data folder · prepare crops · check the data · manage surveys | `pipeline.cli folder / prepare / describe / coverage / conform`, `pipeline.verify`, `workbench.prepare` |
| **2 Train a model** | the eight models with their versions; what the run starts from — the public pretrained weights, a trained version (fine-tuning) or a checkpoint file; the training plan (hyperparameters from `data_prep/config.py`, data folder, output folder, previous duration); epochs where the entry point takes them; evaluate-when-done; GPU; resume an interrupted OWL run | `<model>/main.py --train [--init-from CKPT] [--epochs N] [--eval]` |
| **3 Evaluate & compare** | leaderboard (current and earlier test split kept apart; last checkpoints and fine-tuned versions marked) · inspect a version (lineage, checkpoints, scores of each, example panels) · test a trained version: model → version → checkpoint · zero-shot baselines · cross-model analysis from runs you tick · efficiency benchmarks | `<model>/main.py --eval --run X [--checkpoint last]`, `baseline/main.py`, `analysis/main.py -m <manifest>`, `analysis.bench_*` |
| **4 Detect birds in new imagery** | choose a survey and the weights: a scored version (sorted by counting error, with a time estimate) or any version and checkpoint; threshold (default: that checkpoint's t*); GPU | `python -m workbench.detect <result>` |
| **5 Review & correct detections** | choose a result and which cells to visit, then the full-screen reviewer | — |
| **6 Export reports** | bird count report · corrected labels for training · model comparison · one run's report | — |
| **7 Jobs & logs** | every job with its status and latest output; follow a log live; page it; cancel; run again; processes started elsewhere (e.g. a keep-alive training script) | — |
| **8 Settings & health check** | active data folder · preferred GPU · image viewer · health check · self-test · desktop launcher | `python -m workbench.selftest` |

## Background jobs

Anything slower than a few seconds runs as a detached job: its own session, its own log.
Quitting the workbench, closing VS Code or losing the SSH connection does not stop it.
Reopening the workbench finds it again under *Jobs & logs*, where the exact command is
shown (environment included) so any job can be reproduced by hand. Status is worked out
from the job's record rather than stored: *running*, *succeeded*, *failed* (with its exit
code), *cancelled*, or *stopped* if it died without reporting, e.g. on a reboot.
Cancelling stops the job's whole process group, including DataLoader workers and the OWL
interpreter.

Each job runs on one GPU. The GPU picker shows every card's load and pre-selects the
idlest. A busy card needs an explicit "use it anyway".

**The data a job reads** is the *active data folder*. The workbench never edits
`data_prep/config.py`. It passes `WATERFOWL_DATASET=<folder>` to each job when the folder
differs from the configured default, so the command in the job's log says which data it
read.

## Versions, fine-tuning and checkpoints

A **version** is one training run: a timestamped folder under
`output/checkpoints/<model>/`. Each holds two checkpoints:

| | Ultralytics (YOLOv5/11/26) | YOLO-NAS | Faster R-CNN | OWL |
|---|---|---|---|---|
| **best** — highest validation score; what the run's scores describe | `weights/best.pt` | `ckpt_best.pth` | `model_best.pth` | `weights/best.pth` (`best_model.pth` while unfinished) |
| **last** — the end of training, or the latest epoch of a run still going | `weights/last.pt` | `ckpt_latest.pth` | `model_final.pth`, else the newest periodic `model_<iter>.pth` | `latest_model.pth` |

**Fine-tuning** starts a new version from a trained one instead of from the public
pretrained weights — e.g. to continue on new or corrected data. Use *Train → pick a model →
Start from: a trained version*, or *Train → Fine-tune a trained version*, or *Fine-tune a
new version from it* on a version's page. A checkpoint file works too, such as one from
the Hugging Face checkpoints repo. It runs the entry point with
`--init-from <checkpoint>`. That is a new run folder with a fresh optimiser and
learning-rate schedule, where only the weights carry over and early stopping still
applies. The new version records where it started (`init_from` in `run_info.json`, and
`pretrained` in `experiment.json`). Lists mark it with ↳, and its page says which version
it came from and which versions were fine-tuned from it.

**What to expect from a fine-tune.** Measured on one collection (Bird_E, 417 training
crops), starting from each family's published version:

| Family | Validation after one epoch | For reference | |
|---|---|---|---|
| YOLO-NAS | mAP@0.30 0.936 | its original run's first epoch from COCO: 0.673 | carries its level over |
| Faster R-CNN | AP30 94.6 | the parent on this collection's test crops: 93.8 | carries its level over (its backbone BatchNorm is frozen) |
| OWL-C | point-F1 0.929 | the parent on this collection's test crops: 0.945 | carries its level over |
| YOLO11 | mAP50 0.809 | the parent on the same validation crops: 0.929 | dips, then recovers |

The Ultralytics models (YOLOv5/11/26) dip because a new run restarts mosaic augmentation
and the BatchNorm statistics re-adapt to it. They move even at a learning rate of zero,
which took one epoch to 0.756. Over a 15-epoch fine-tune YOLO11 went 0.809 → 0.912 →
(0.850 when mosaic switched off for the last ten epochs) → 0.925. Its best checkpoint
scored 0.937 against its parent's 0.929. The run keeps its best checkpoint, so judge a
fine-tune by that, not by its first epochs.

**Testing a chosen version**: *Evaluate & compare → Test a trained version* asks for the
model, then the version, then the checkpoint. Testing *best* writes the version's own
scores, as `--eval` always has. Testing *last* runs `--eval --checkpoint last`, whose
results go to `<run>/eval_last/`, so they never overwrite the version's own scores. The
leaderboard shows them as a separate "(last)" row. A version still training can be
tested too — its last checkpoint only, as a snapshot.

*Detect birds* uses a version's best checkpoint by default. *Another version or
checkpoint…* reaches every trained version, tested or not, and either checkpoint. The
counting threshold defaults to that checkpoint's own t*. Failing that, it offers the best
checkpoint's t*; if neither exists, you choose one.

## Counting birds in survey photographs

A **survey** is a folder of photographs registered under `output/workbench/surveys/`.
Photos are linked by default, or copied. Surveys never join the training data, because
adding photographs to `dataset_full/` would reshuffle the published train/validation
split. Label files next to the photos are optional. When present, the count report
scores the model against them.

**Detection** cuts each photograph into 512 × 512 tiles at a 460 px stride, with the
same edge rule as the training crops. `python -m workbench.selftest` checks this against
the crop file names under `crops/`. Each tile is scored with that model's own evaluation
settings: score floor 0.05, its NMS IoU, up to 500 detections per tile, and one tile per
Ultralytics call. The OWL variants run in their own Python 3.11 environment
(`workbench/_owl_detect.py`), with the stitcher and LMDS settings their evaluation uses.

Neighbouring tiles overlap, so each bird must be counted once. Each tile keeps only the
detections in the part of the photograph it *owns*: its **cell**, bounded by the middle
of each overlap. That boundary is at least 26 px inside every tile, so a bird owned by a
tile is seen whole by it. A bird seen by two tiles right at a boundary is merged: boxes
by overlap, points within 6 px.

Every detection above 0.05 is kept with its score. The counting threshold therefore
belongs to the result, not to the run, and can be changed in the reviewer or a report
without running the model again. The default is the run's **t\***: the score cut that
minimised counting error on the test split, where its published P/R/F1/MAE are read.

**Checked on held-out data.** Six test-split photographs from four collections
(825 annotated birds, 1 to 495 per photograph) were imported as a labelled survey.
Every model family counted them with the recall its test-split evaluation would lead
you to expect:

| Run | Recall | Precision |
|---|---:|---:|
| YOLO11 2026-08-27 | 99.0% | 88.8% |
| YOLOv5 2026-08-27 | 98.9% | 89.3% |
| YOLO26 2026-08-28 | 98.4% | 90.2% |
| YOLO-NAS 2026-08-26 | 97.2% | 87.7% |
| Faster R-CNN 2026-08-25 | 96.1% | 83.6% |
| OWL-C 2026-08-30 | 90.4% | 89.4% |

None of the missed birds lay near a cell boundary, and the unmatched detections were no
more common near boundaries than elsewhere. So the tile merge neither drops nor doubles
birds. Several unmatched detections are plainly unlabelled birds, which matches the
annotation gaps the cross-model analysis found.

A wider check — twelve held-out photographs, one or two from each of the ten collections,
1,872 annotated birds — gave YOLOv5 2026-08-27 point AP 91.4%, recall 90.7% and precision
88.4% on whole photographs. Its test-split crop evaluation gives 93.0%, 89.6% and 93.2%.
And on a single 15 m photograph, YOLOv5 run 2026-07-11 found 30.4% of the birds through
the workbench's whole-photograph path. The project's crop-level evaluation of the same
photograph's 96 crops gives 25.0%. The low number is that run's low-altitude weakness,
not the tiling; the August run reaches 66% on the same photograph.
*Import imagery → Sample held-out test photos as a survey* builds such a check in one
step, for any run.

## Reviewing

The reviewer shows one cell at a time with a 16 px margin of its neighbours, plus a
magnified loupe on the selected detection:

- **amber** ring — counted by the model, not reviewed yet
- **green** ring — confirmed as a bird
- **red** ring — rejected, not a bird
- **grey** ring — below the threshold (shown with `b`); `y` counts one anyway
- **cyan** ring — a bird added by hand
- **magenta** dot or box — an annotated bird (labelled surveys)

| Key | | Key | |
|---|---|---|---|
| `↑` `↓` `k` `j` | select a detection | `y` / `n` | bird / not a bird |
| `Enter` | cell done, go to the next | `c` | clear the decision |
| `→` `←` | next / previous cell | `a` | add a missed bird (arrows move the cross, Enter places) |
| `]` `[` | next / previous photograph | click | select a marker, or add a bird where there is none |
| `+` `-` or wheel | zoom the loupe | `x` `Del` | delete a bird you added |
| `b` `g` | below-threshold detections / annotations | `u` | undo |
| `<` `>` `T` | threshold −/+ 0.05, reset to t* | `o` | live full-resolution preview in VS Code |
| `?` | every key | `q` `Esc` | save and leave |

Detections in a cell are listed least confident first, because that is where a model's
mistakes are. A cell the model got right costs one key (Enter). A detection nobody
decides keeps the model's verdict, so a partly reviewed survey still has a sensible
count. Reports say how many cells a person checked.

    count = detections ≥ threshold − rejected + below-threshold confirmed + added

Decisions are written to `review.json` after every key press, so closing the terminal
loses nothing. The terminal image is coarse: about five image pixels per terminal pixel
in the cell view. `o` writes a 2× full-resolution picture of the current cell and opens
it in VS Code (or the desktop viewer); it follows the reviewer from then on.

Which cells to visit: *cells with detections* (the usual pass), *borderline cells first*
(scores within ±0.15 of the threshold), *cells not reviewed yet*, or *every cell*, an
audit that also finds birds the model missed entirely.

## Exports

Everything goes to `reports/<date>_<name>/`.

- **Bird count report** — `report.md` and `report.html`: totals before and after review,
  review coverage, the model and threshold, a per-photograph table and an annotated
  overview of each photograph. For a labelled survey it adds precision, recall and
  counting error against the labels. Also `counts_<survey>.csv` (one row per photograph)
  and `detections.csv` (every detection with its score and decision).
- **Corrected labels** — a collection in the `dataset_full/` format: photos (symlinks),
  one `bird,x1,y1,x2,y2` file each and an `image_info.csv`, plus `coco.json`. Only fully
  reviewed photographs by default. It reads back through the pipeline's own adapter, and
  *Import labelled photos as a training collection* takes it directly.
- **Model comparison** and **run report** — the leaderboard, or one run's scores and
  provenance, as Markdown, HTML and CSV.

## Where things are kept

| Path | Holds | In git |
|---|---|---|
| `output/workbench/jobs/<id>/` | a job's record, log and exit code | no (the folder ignores itself) |
| `output/workbench/surveys/<name>/` | a survey: `survey.json`, `images/` links, `labels/` | no |
| `output/workbench/detections/<id>/` | `detect.json`, `detections.jsonl` (one line per photograph), `review.json` | no |
| `output/workbench/settings.json` | active data folder, preferred GPU, viewer | no |
| `reports/` | exports | your call |

## Design notes

- **No second way of doing things.** Preparing data, training, evaluation, baselines and
  analysis are the existing entry points, launched with the flags a person would type. A
  run started here is indistinguishable from one started by hand. The one new command,
  `workbench.prepare`, calls the same `pipeline.auto.ensure_data` every model calls at
  startup. `pipeline.cli select` only filters crops that already exist, so it would leave
  newly imported photographs out.
- **Scores are read, never recomputed.** The leaderboard reads each run's
  `paper_point_metrics.json` / `eval/metrics.json`. Runs scored on the earlier
  25,708-crop test split are kept in a separate table from those on the current
  28,588-crop split, and the OWL pseudo-box mAP30 is marked as such.
- **Two OWL `--resume` syntaxes exist across machines** (`--resume TIMESTAMP` and
  `--resume --run TIMESTAMP`). *Resume* reads the parser in `megadetector_overhead/main.py`
  and uses whichever this checkout has.
- **Checks:** `python -m workbench.selftest` (seconds, no GPU) covers tiling against the
  real crops, cell ownership, the merge, the counting rule, label round trips and the job
  manager. *Settings → Health check* covers the environments, weights, data and disk.

## Limits

- **OWL ignores the chosen data folder.** `megadetector_overhead/main.py` never calls
  `pipeline.auto.ensure_data()`, and `megadetector_overhead/dataset.py` fixes its data
  paths as default arguments when it is imported. So OWL-C/T/D always train and test on
  the whole corpus, whatever folder is active; the training and test screens say so. They
  do honour `WATERFOWL_EXPERIMENT`, which `data_prep/config.py` resolves at import. The fix
  — an `ensure_data(mirror="mdo")` call at the top of `main()` and paths read at call time —
  is left to you, because those two files differ between machines (see *Design notes*).
- A survey is counted at whole-photograph level. The accuracy numbers each run carries
  were measured on 512 px test crops, and the report says so.
- The reviewer has one reviewer per result and no per-photograph notes.
- Reviewing the *test split* itself — e.g. the 1,639 detections all eight models agree on
  that no label covers — would reuse the reviewer, with `output/analysis/predictions/`
  as input. It is not built yet.
