"""
Import imagery & prepare data.

Two kinds of imagery come in, and they go to different places on purpose:

* **Survey photos** — photographs to count birds in.  They become a survey
  (workbench/surveys.py), linked rather than copied, and never touch the training data.
* **Labelled photos** — photographs with `bird,x1,y1,x2,y2` label files that should
  become training data.  They become a new collection inside a data-selection folder
  (`dataset_<name>/`), next to symlinks of whichever existing collections should train
  with them.  dataset_full/ itself is never modified: adding photographs to it would
  reshuffle the published train/validation split for every collection.

Which folder the models read is the workbench's "active data folder"
(workbench/settings.py), passed to every job as WATERFOWL_DATASET.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from data_prep import config
from data_prep.prepare_data import read_annotations
from workbench import catalog, settings, surveys, ui
from workbench.jobs import python_cmd
from workbench.paths import ROOT, rel
from workbench.screens import common
from workbench.util import ago, human_bytes, slug

PROTECTED_FOLDERS = ("dataset_full", "dataset_v1")   # the corpus and its frozen v1 copy


def run() -> None:
    while True:
        ui.title("Import imagery & prepare data")
        folder = settings.data_folder()
        ui.show(ui.kv([("Active data folder", f"{folder}  (every training and evaluation "
                                              "job reads this)"),
                       ("Crops", "ready" if catalog.splits_status().ready else "not prepared"),
                       ("Surveys", f"{len(surveys.all_surveys())} imported")]))
        ui.console.print()
        options = [
            ("overview", "Data overview", "collections, crops, selections, surveys"),
            ("survey", "Import survey photos to count birds in", "→ then Detect birds"),
            ("sample", "Sample held-out test photos as a survey",
             "labelled; shows how a model does on whole photos"),
            ("labelled", "Import labelled photos as a training collection",
             "bird,x1,y1,x2,y2 label files"),
            ("choose", "Choose which collections experiments use", "builds a data folder"),
            ("switch", "Switch the active data folder", f"now: {folder}"),
            ("prepare", "Prepare crops for all models", "512 px crops, COCO splits, mirrors"),
            ("check", "Check the data", "describe · coverage · conformance · verify"),
            ("surveys", "Manage imported surveys", "list, inspect, delete"),
        ]
        picked = ui.choose("What do you want to do?", options)
        if picked is None:
            return
        {"overview": overview, "survey": import_survey, "sample": sample_survey,
         "labelled": import_labelled,
         "choose": choose_collections, "switch": switch_folder, "prepare": prepare,
         "check": check, "surveys": manage_surveys}[picked]()


# ── Overview ──────────────────────────────────────────────────────────────────

def collections_table(root: str, title: str):
    cols = catalog.collections(root)
    rows = [[c.name, f"{c.photos:,}", f"{c.label_files:,}", "yes" if c.has_index else "no",
             c.link or ""] for c in cols]
    rows.append(["[bold]total[/]", f"[bold]{sum(c.photos for c in cols):,}[/]",
                 f"{sum(c.label_files for c in cols):,}", "", ""])
    return ui.table(["Collection", ("Photos", "right"), ("Label files", "right"),
                     "image_info.csv", "Links to"], rows, title=title)


def overview() -> None:
    ui.title("Import imagery & prepare data", "Overview")
    ui.show(collections_table(settings.data_root(),
                              f"Active data folder: {settings.data_folder()}"))
    split = catalog.splits_status()
    ui.show(ui.kv([
        ("COCO splits", ("ready in " + rel(config.COMMITTED_CROPS_JSON_DIR) +
                         f" (updated {ago(split.updated)})") if split.ready else "not prepared"),
        ("Crop images", rel(config.CROPS_IMG_DIR)),
        ("Selections", ", ".join(split.experiments) or "none"),
        ("Data folders", ", ".join(catalog.data_folders())),
    ]))
    found = surveys.all_surveys()
    if found:
        ui.console.print()
        ui.show(ui.table(["Survey", ("Photos", "right"), ("Labelled", "right"), "Imported",
                          "From"],
                         [[s.name, str(len(s.photos)), str(s.labelled), ago(s.created),
                           s.source] for s in found], title="Imported surveys"))
    ui.pause()


# ── Survey photos ─────────────────────────────────────────────────────────────

def _scan_with_status(folder: str, recursive: bool) -> surveys.Scan:
    with ui.console.status(f"Looking for photographs in {folder}…"):
        return surveys.scan(folder, recursive=recursive)


def _has_subfolders_with_images(folder: str) -> bool:
    for entry in os.scandir(folder):
        if entry.is_dir():
            try:
                if any(n.lower().endswith(catalog.IMAGE_EXTS) for n in os.listdir(entry.path)):
                    return True
            except OSError:
                continue
    return False


def _progress() -> Progress:
    return Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                    TimeRemainingColumn(), console=ui.console, transient=True)


def import_survey() -> None:
    ui.title("Import imagery & prepare data", "Survey photos")
    ui.dim("Photographs to count birds in — one flight, site or day per survey.  They are "
           "linked, not copied, unless you choose otherwise, and never join the training "
           "data.  Label files next to them (optional) let the report check the model's "
           "accuracy on them.")
    ui.console.print()
    folder = ui.ask_path("Folder with the photographs:")
    if not folder:
        return
    recursive = False
    if _has_subfolders_with_images(folder):
        recursive = ui.confirm("It has sub-folders with images too. Include them?", default=True)
    found = _scan_with_status(folder, recursive)
    if not found.images:
        ui.error(f"No photographs ({', '.join(catalog.IMAGE_EXTS)}) found in {folder}.")
        ui.pause()
        return
    ui.show(ui.kv([("Photographs", f"{len(found.images):,} ({human_bytes(found.total_bytes)})"),
                   ("With label files", f"{len(found.labels):,}"),
                   ("First few", ", ".join(os.path.basename(p) for p in found.images[:4])
                    + (" …" if len(found.images) > 4 else ""))]))
    ui.console.print()
    name = ui.ask("Name for this survey:", default=slug(os.path.basename(folder.rstrip("/"))),
                  validate=surveys.valid_name,
                  hint="letters, digits, - _ .")
    if not name:
        return
    meta = {}
    if ui.confirm("Add a site, date, altitude or notes? (optional)", default=False):
        for key, question in (("site", "Site:"), ("date", "Date (YYYY-MM-DD):"),
                              ("altitude_m", "Flight altitude in metres:"), ("notes", "Notes:")):
            answer = ui.ask(question)
            if answer:
                meta[key] = answer
    mode = ui.choose("Link or copy the photographs?", [
        ("link", "Link to them", "no disk space used; the originals must stay where they are"),
        ("copy", "Copy them", f"uses {human_bytes(found.total_bytes)}; safe if the originals move"),
    ], default="link")
    if mode is None:
        return
    with _progress() as bar:
        task = bar.add_task("Importing", total=len(found.images))
        survey, skipped = surveys.create(name, found, mode=mode, meta=meta,
                                         progress=lambda i, n: bar.update(task, completed=i))
    ui.ok(f"Survey {survey.name}: {len(survey.photos):,} photographs"
          + (f", {survey.labelled} with labels" if survey.labelled else ""))
    for path in skipped[:5]:
        ui.warn(f"Could not read {path} — skipped")
    if len(skipped) > 5:
        ui.warn(f"… and {len(skipped) - 5} more unreadable files skipped")
    if ui.confirm("Count the birds in it now?", default=True):
        from workbench.screens import detect
        detect.start(survey=survey)


def sample_survey() -> None:
    ui.title("Import imagery & prepare data", "Held-out sample")
    ui.dim("A survey made of photographs the annotators held out for testing — no model "
           "was trained on them — with their labels.  Count the birds in it with any run "
           "and the count report scores the run against the labels, on whole photographs.")
    ui.console.print()
    n = ui.ask_int("How many photographs?", default=12, lo=1, hi=500)
    if n is None:
        return
    existing = catalog.collections(config.DEFAULT_DATASET_ROOT)
    chosen = ui.choose_many("From which collections?",
                            [(c.name, c.name, f"{c.photos} photos") for c in existing],
                            selected=[c.name for c in existing])
    if not chosen:
        return
    name = ui.ask("Name for this survey:", default=f"heldout_sample_{n}",
                  validate=surveys.valid_name)
    if not name:
        return
    with ui.console.status("Choosing held-out photographs…"):
        found = surveys.sample_held_out(n, chosen)
    if not found.images:
        ui.error("No held-out test photographs found in those collections.")
        ui.pause()
        return
    with _progress() as bar:
        task = bar.add_task("Importing", total=len(found.images))
        survey, skipped = surveys.create(
            name, found, mode="link", prefix_folder=True,
            meta={"notes": "held-out test photographs with their labels, sampled by the "
                           "workbench"},
            progress=lambda i, total: bar.update(task, completed=i))
    ui.ok(f"Survey {survey.name}: {len(survey.photos)} held-out photographs from "
          f"{found.subfolders} collection(s), {survey.labelled} labelled")
    if ui.confirm("Count the birds in it now?", default=True):
        from workbench.screens import detect
        detect.start(survey=survey)


def manage_surveys() -> None:
    while True:
        ui.title("Import imagery & prepare data", "Surveys")
        found = surveys.all_surveys()
        if not found:
            ui.dim("No surveys imported yet.")
            ui.pause()
            return
        picked = ui.choose("Which survey?", [(s, s.name, f"{len(s.photos)} photos · "
                                             f"{s.labelled} labelled · {ago(s.created)}")
                                            for s in found])
        if picked is None:
            return
        ui.title("Import imagery & prepare data", "Surveys", picked.name)
        ui.show(ui.kv([("Photographs", f"{len(picked.photos):,}"),
                       ("Labelled", str(picked.labelled)),
                       ("From", picked.source), ("Stored as", "links" if picked.mode == "link"
                                                  else "copies"),
                       ("Folder", rel(picked.dir))]
                      + [(k.capitalize(), v) for k, v in picked.meta.items()]))
        action = ui.choose("Next:", [("detect", "Count birds in it", ""),
                                     ("delete", "Delete this survey",
                                      "removes its links/copies; originals untouched")])
        if action == "detect":
            from workbench.screens import detect
            detect.start(survey=picked)
        elif action == "delete":
            from workbench import detections
            users = [r for r in detections.all_results() if r.meta.get("survey") == picked.name]
            if users:
                ui.warn(f"{len(users)} detection result(s) use it and would lose their photos.")
            if ui.confirm(f"Delete survey {picked.name}?", default=False):
                surveys.delete(picked)


# ── Labelled photos → a training collection ───────────────────────────────────

def _label_report(found: surveys.Scan) -> dict:
    from PIL import Image
    boxes = outside = 0
    for image, label in found.labels.items():
        rows = read_annotations(label)
        boxes += len(rows)
        with Image.open(image) as im:
            w, h = im.size
        outside += sum(1 for x1, y1, x2, y2 in rows if x1 < 0 or y1 < 0 or x2 > w or y2 > h)
    return {"boxes": boxes, "outside": outside}


def import_labelled(folder: str | None = None) -> None:
    ui.title("Import imagery & prepare data", "Labelled training collection")
    ui.dim("Photographs with label files (one bird,x1,y1,x2,y2 line per bird, as in "
           "dataset_full/) become a new collection inside a data folder.  dataset_full/ "
           "itself is never changed.")
    ui.console.print()
    folder = ui.ask_path("Folder with the labelled photographs:", default=folder or "")
    if not folder:
        return
    found = _scan_with_status(folder, recursive=False)
    if not found.images:
        ui.error("No photographs found there.")
        ui.pause()
        return
    with ui.console.status("Reading the label files…"):
        report = _label_report(found)
    unlabelled = len(found.images) - len(found.labels)
    ui.show(ui.kv([
        ("Photographs", f"{len(found.images):,}"),
        ("With label files", f"{len(found.labels):,} holding {report['boxes']:,} birds"),
        ("Without labels", f"{unlabelled:,}" + ("  — skipped by the pipeline" if unlabelled else "")),
        ("image_info.csv", "yes — its train/test marks are honoured" if found.has_index else
         "no — every photograph joins train/validation (none is held out for test)"),
    ]))
    if report["outside"]:
        ui.warn(f"{report['outside']} box(es) extend past their photograph's edge; the tiler "
                "clips them.")
    if not found.labels:
        ui.error("None of the photographs has a label file, so there is nothing to train on.")
        ui.pause()
        return
    ui.console.print()

    name = ui.ask("Collection name:", default=slug(os.path.basename(folder.rstrip("/"))),
                  validate=lambda s: None if s and slug(s) == s else
                  "letters, digits, '-', '_' and '.' only",
                  hint="e.g. Bird_K, or the survey's name")
    if not name:
        return
    targets = [f for f in catalog.data_folders() if f not in PROTECTED_FOLDERS]
    options = [("new", "A new data folder", "this collection + any existing ones you pick")]
    options += [(f, f"Add to {f}", f"{len(catalog.collections(os.path.join(ROOT, f)))} "
                                   "collections") for f in targets]
    where = ui.choose("Where should it go?", options)
    if where is None:
        return
    if where == "new":
        folder_name = ui.ask("New data folder: dataset_", default=slug(name).lower(),
                             validate=lambda s: None if s and slug(s) == s and not os.path.exists(
                                 os.path.join(ROOT, f"dataset_{s}")) else "unused name, letters/digits/_ only")
        if not folder_name:
            return
        target = os.path.join(ROOT, f"dataset_{folder_name}")
        existing = catalog.collections(config.DEFAULT_DATASET_ROOT)
        include = ui.choose_many(
            "Train it together with which existing collections? (Space ticks)",
            [(c, c.name, f"{c.photos} photos") for c in existing],
            selected=existing, min_count=0)
        if include is None:
            return
    else:
        target = os.path.join(ROOT, where)
        include = []
    dest = os.path.join(target, name)
    if os.path.exists(dest):
        ui.error(f"{rel(dest)} already exists.")
        ui.pause()
        return
    mode = ui.choose("Link or copy the photographs?", [
        ("copy", "Copy them", "safe if the originals move (recommended for training data)"),
        ("link", "Link to them", "no disk space used"),
    ], default="copy")
    if mode is None:
        return

    os.makedirs(dest)
    for c in include:
        link = os.path.join(target, c.name)
        if not os.path.exists(link):
            os.symlink(os.path.relpath(os.path.realpath(c.path), target), link)
    with _progress() as bar:
        task = bar.add_task("Importing", total=len(found.labels))
        for i, (image, label) in enumerate(sorted(found.labels.items())):
            bar.update(task, completed=i)
            for src in (image, label):
                out = os.path.join(dest, os.path.basename(src))
                if mode == "copy":
                    shutil.copy2(src, out)
                else:
                    os.symlink(os.path.realpath(src), out)
        if found.has_index:
            shutil.copy2(os.path.join(folder, "image_info.csv"), dest)
    ui.ok(f"{rel(dest)}/ — {len(found.labels):,} labelled photographs"
          + (f", next to {len(include)} linked collection(s)" if include else ""))

    folder_name = os.path.basename(target)
    if ui.confirm(f"Make {folder_name} the active data folder for training and evaluation?",
                  default=True):
        _set_folder(folder_name)
        if ui.confirm("Prepare its crops now (a background job)?", default=True):
            prepare()


# ── Choosing data ─────────────────────────────────────────────────────────────

def _set_folder(folder: str) -> None:
    settings.put("data_folder", None if folder == settings.configured_data_folder() else folder)
    ui.ok(f"Active data folder: {folder}")


def choose_collections() -> None:
    ui.title("Import imagery & prepare data", "Choose collections")
    ui.dim("An experiment's data is a folder of collections (symlinks; nothing is "
           "copied).  Every model trains and evaluates on whatever the folder holds, with "
           "each photograph keeping its published train/validation/test role.")
    ui.console.print()
    existing = catalog.collections(config.DEFAULT_DATASET_ROOT)
    chosen = ui.choose_many("Tick the collections to use:",
                            [(c.name, c.name, f"{c.photos} photos") for c in existing])
    if not chosen:
        return
    default = "_and_".join(c.replace("Bird_", "").lower() for c in chosen) \
        if len(chosen) <= 3 else f"{len(chosen)}_collections"
    name = ui.ask("Folder name: dataset_", default=default,
                  validate=lambda s: None if s and slug(s) == s else "letters, digits, _ only")
    if not name:
        return
    out = subprocess.run(python_cmd("-m", "pipeline.cli", "folder", "--name", name,
                                    "--collections", *chosen), cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        ui.error((out.stderr or out.stdout).strip())
        ui.pause()
        return
    ui.ok(out.stdout.strip().splitlines()[0])
    folder = f"dataset_{name}"
    if ui.confirm(f"Make {folder} the active data folder?", default=True):
        _set_folder(folder)
        ui.dim("  Models prepare the folder themselves on their first run; preparing it now "
               "just does that work up front.")
        if ui.confirm("Prepare it now (a background job)?", default=False):
            prepare()
    else:
        ui.pause()


def switch_folder() -> None:
    ui.title("Import imagery & prepare data", "Active data folder")
    current = settings.data_folder()
    options = []
    for folder in catalog.data_folders():
        cols = catalog.collections(os.path.join(ROOT, folder))
        hint = f"{len(cols)} collections · {sum(c.photos for c in cols):,} photos"
        if folder == settings.configured_data_folder():
            hint += " · config.py default"
        if folder == "dataset_v1":
            hint += " · frozen Baseline v1 input"
        options.append((folder, folder + ("  ← active" if folder == current else ""), hint))
    picked = ui.choose("Which folder should jobs read?", options, default=current)
    if picked:
        _set_folder(picked)
        ui.pause()


def prepare() -> None:
    ui.title("Import imagery & prepare data", "Prepare crops")
    folder = settings.data_folder()
    if folder == settings.configured_data_folder():
        split = catalog.splits_status()
        if split.ready:
            ui.note("The published crops and COCO splits already exist; preparing again only "
                    "(re)builds the Ultralytics and OWL mirrors. It never re-cuts the "
                    "published split.")
        cmd = python_cmd("-m", "pipeline.cli", "prepare", "--mirrors")
        title = "Prepare crops + mirrors (whole corpus)"
    else:
        ui.note(f"{folder} is a data selection: its crops are filtered from the published "
                "ones, and any photograph not cut yet is tiled — the same step every model "
                "runs when it starts, done once up front.")
        cmd = python_cmd("-m", "workbench.prepare")
        title = f"Prepare data folder {folder}"
    if ui.confirm(f"Start: {title}?", default=True):
        common.launch(title, "prepare", cmd, env={})


# ── Checks ────────────────────────────────────────────────────────────────────

def check() -> None:
    folder = settings.data_folder()
    source = ["--source", "mdc"] + ([] if folder == settings.configured_data_folder()
                                    else ["--from", folder])
    while True:
        ui.title("Import imagery & prepare data", "Check the data")
        picked = ui.choose("Which check?", [
            ("describe", "Describe the source", "collections, counts, crosswalks (~20 s)"),
            ("coverage", "Metadata coverage", "what the files actually record (~5 s)"),
            ("conform", "Conformance suite", "every adapter check; background job"),
            ("verify", "Verify against the published pipeline", "python -m pipeline.verify"),
        ])
        if picked is None:
            return
        if picked in ("describe", "coverage"):
            with ui.console.status(f"Running pipeline.cli {picked}…"):
                out = subprocess.run(python_cmd("-m", "pipeline.cli", picked, *source,
                                                "--no-exif"), cwd=ROOT, capture_output=True,
                                     text=True)
            ui.console.print((out.stdout + out.stderr).rstrip(), markup=False, highlight=False)
            ui.pause()
        elif picked == "conform":
            common.launch("Conformance suite", "check",
                          python_cmd("-m", "pipeline.cli", "conform", *source))
        else:
            quick = ui.confirm("Quick version (skips the two slow checks)?", default=True)
            common.launch("Verify the dataset interface", "check",
                          python_cmd("-m", "pipeline.verify", *(["--quick"] if quick else [])))
