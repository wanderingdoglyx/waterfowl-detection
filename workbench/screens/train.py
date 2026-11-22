"""
Train a model: pick one of the eight, choose what it starts from, and launch it in the
background.

The command launched is the model's own entry point with the flags a person would type
(`./yolov5/main.py --train --eval`, `./megadetector_overhead/main.py --train --model OWLT`,
...), so a run started here is indistinguishable from one started by hand: same folder
under output/checkpoints/, same run_info.json and experiment.json.  Hyperparameters are
shown, not edited — they live in data_prep/config.py, where every model reads them, and
the only per-run override offered is the one the entry point itself accepts (--epochs).

What a run starts from is a choice.  Every published run started from public pretrained
weights (COCO for the box models, the released Zenodo checkpoint for OWL).  Starting from
a trained version instead is fine-tuning: the entry point's `--init-from` loads that
version's checkpoint as the new run's weights, in a new run folder, with a fresh optimiser
and learning-rate schedule.  The new run records where it started (`init_from` in
run_info.json), which is the lineage the workbench shows ("↳ fine-tuned from …").
"""

from __future__ import annotations

import glob
import os
import re

from data_prep import config
from workbench import catalog, settings, ui
from workbench.jobs import python_cmd
from workbench.paths import ROOT, rel
from workbench.screens import common
from workbench.util import human_duration, pct


def hyperparameters(model: catalog.Model) -> list[tuple[str, str]]:
    patience = f"early stop after {config.EARLY_STOP_PATIENCE} epochs without improvement"
    if model.key == "fasterrcnn":
        return [("Network", "Faster R-CNN, ResNet-50 FPN"),
                ("Epochs", f"up to {config.NUM_EPOCHS}; {patience} in val mAP30"),
                ("Optimiser", f"SGD, learning rate {config.LEARNING_RATE}, cosine decay"),
                ("Batch size", str(config.BATCH_SIZE)),
                ("Anchors", ", ".join(str(a[0]) for a in config.ANCHOR_SIZES) + " px")]
    if model.key == "yolonas":
        return [("Network", config.YOLONAS_MODEL),
                ("Epochs", f"up to {config.NUM_EPOCHS}; {patience} in val mAP30"),
                ("Optimiser", f"Adam, learning rate {config.YOLONAS_LR}, cosine decay"),
                ("Batch size", str(config.YOLONAS_BATCH_SIZE))]
    if model.key == "yolov5":
        return [("Network", config.YOLOV5_MODEL),
                ("Epochs", f"up to {config.NUM_EPOCHS}; {patience} (Ultralytics fitness)"),
                ("Optimiser", f"Adam, learning rate {config.YOLOV5_LR}"),
                ("Batch size", str(config.YOLOV5_BATCH_SIZE))]
    if model.key in config.ULTRALYTICS_MODELS:
        spec = config.ULTRALYTICS_MODELS[model.key]
        return [("Network", spec["weights"]
                 + (" (end-to-end, NMS-free)" if spec.get("end2end") else "")),
                ("Epochs", f"up to {config.NUM_EPOCHS}; {patience} (Ultralytics fitness)"),
                ("Optimiser", f"Adam, learning rate {spec['lr']}"),
                ("Batch size", str(spec["batch_size"]))]
    spec = config.MDO_MODELS[model.mdo_key]
    return [("Network", model.about.split(" · ")[1]),
            ("Epochs", f"up to {config.MDO_EPOCHS}; best checkpoint kept by val point-F1, "
                       "learning rate reduced on plateau"),
            ("Optimiser", f"Adam, learning rate {config.MDO_LR}"),
            ("Batch size", str(spec.get("batch_size", config.MDO_BATCH_SIZE)))]


def public_weights(model: catalog.Model) -> str:
    """The weights a run starts from when it is not fine-tuning a trained version."""
    if model.key == "fasterrcnn":
        return "COCO-pretrained Faster R-CNN R50-FPN (Detectron2 model zoo)"
    if model.key == "yolonas":
        return f"COCO-pretrained {config.YOLONAS_MODEL}"
    if model.key == "yolov5":
        return f"COCO-pretrained {config.YOLOV5_MODEL}"
    if model.key in config.ULTRALYTICS_MODELS:
        return f"COCO-pretrained {config.ULTRALYTICS_MODELS[model.key]['weights']}"
    load_from = config.MDO_MODELS[model.mdo_key]["load_from"]
    return f"released {os.path.basename(load_from)} (Zenodo 20802844, CC BY-NC-SA 4.0)"


def data_line(model: catalog.Model, what: str) -> str:
    """The data a job of this model will really read, for a plan's "Data" row."""
    folder = settings.data_folder()
    if folder != settings.configured_data_folder() and not catalog.reads_data_folder(model):
        return (f"[yellow]the whole corpus ({settings.configured_data_folder()}) — "
                f"{model.script} does not read the chosen folder, so {folder} is "
                "ignored[/]")
    return f"{folder} ({what})"


def default_epochs(model: catalog.Model) -> int:
    return config.MDO_EPOCHS if model.owl else config.NUM_EPOCHS


def previous_duration(run: catalog.Run) -> float | None:
    """Roughly how long a finished run trained: its last checkpoint write minus its start."""
    names = ("weights/last.pt", "model_final.pth", "latest_model.pth", "*/ckpt_latest.pth")
    stamps = [os.path.getmtime(p) for n in names for p in glob.glob(os.path.join(run.path, n))]
    if not stamps:
        return None
    return max(stamps) - catalog.run_started(run.run_id)


def resume_syntax() -> list[str]:
    """How this checkout's OWL entry point spells "resume run X".

    Two implementations exist across machines: `--resume TIMESTAMP` and
    `--resume --run TIMESTAMP`.  Reading the parser declaration picks the right one.
    """
    with open(os.path.join(ROOT, "megadetector_overhead", "main.py")) as f:
        source = f.read()
    if re.search(r'"--resume",\s*nargs="\?"', source):
        return ["--resume", "{run}"]
    return ["--resume", "--run", "{run}"]


def owl_epoch(run: catalog.Run) -> int | None:
    """The last epoch an OWL run's training log reports."""
    logs = sorted(glob.glob(os.path.join(run.path, "*_training.log")), key=os.path.getmtime)
    if not logs:
        return None
    with open(logs[-1], "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 64 * 1024))
        tail = f.read().decode(errors="replace")
    hits = re.findall(r"Epoch: \[(\d+)\]", tail)
    return int(hits[-1]) if hits else None


def run() -> None:
    while True:
        ui.title("Train a model")
        procs = catalog.pipeline_processes()
        all_runs = catalog.runs(procs=procs)
        rows = []
        for m in catalog.MODELS:
            mine = [r for r in all_runs if r.model is m]
            current = [r for r in mine if r.metrics and r.metrics.split == "current"]
            top = catalog.best(current, "ap")
            tuned = sum(1 for r in mine if r.parent)
            rows.append([m.label, m.about.split(" · ", 1)[1],
                         str(len(mine)) + (f" ({tuned} fine-tuned)" if tuned else ""),
                         mine[0].run_id if mine else "—",
                         common.styled(mine[0].status) if mine else "—",
                         pct(top.metrics.ap) if top else "—"])
        ui.show(ui.table(["Model", "Framework · type", ("Versions", "right"), "Latest",
                          "Latest status", ("Best point AP", "right")], rows,
                         caption="A version is one training run.  Best point AP among "
                                 "versions scored on the current test split."))
        ui.note(f"Training reads the data folder [bold]{settings.data_folder()}[/] "
                "(change it under Import imagery & prepare data).")
        ui.console.print()

        trained = [r for r in all_runs if r.checkpoints() and not r.active
                   and catalog.supports(r.model, "--init-from")]
        resumable = [r for r in all_runs if r.model.resumable and not r.active
                     and not r.metrics and os.path.exists(os.path.join(r.path, "latest_model.pth"))]
        options = [(m, m.label, m.about) for m in catalog.MODELS]
        if trained:
            options.append(("finetune", "Fine-tune a trained version",
                            f"start a new run from one of {len(trained)} trained versions"))
        if resumable:
            options.append(("resume", "Resume an interrupted OWL run",
                            f"{len(resumable)} can continue from their last epoch"))
        picked = ui.choose("Which model do you want to train?", options)
        if picked is None:
            return
        if picked == "finetune":
            fine_tune()
        elif picked == "resume":
            resume(resumable)
        else:
            plan(picked, [r for r in all_runs if r.model is picked])


def _ready_to_train(model: catalog.Model) -> bool:
    split = catalog.splits_status()
    if not split.ready:
        ui.error("The crops and COCO splits have not been prepared yet.")
        ui.note("Import imagery & prepare data → Prepare crops for all models.")
        ui.pause()
        return False
    if model.owl and not catalog.owl_env_ready():
        ui.error(f"The OWL environment is missing ({rel(config.MDO_PYTHON)}).")
        ui.note(f"Set it up with: cd {rel(config.MDO_REPO_DIR)} && "
                "uv sync --no-default-groups --group gpu")
        ui.pause()
        return False
    return True


def _release_weights_ready(model: catalog.Model) -> bool:
    """OWL starts from a released checkpoint that has to be downloaded once."""
    weights = catalog.owl_weights(model)
    if not weights or os.path.exists(weights):
        return True
    ui.warn(f"{model.label} starts from {os.path.basename(weights)}, which is not "
            "downloaded yet.")
    if ui.confirm("Download it now from Zenodo (a background job)?", default=True):
        common.launch(f"Download {os.path.basename(weights)}", "download",
                      python_cmd(model.script, *model.variant, "--fetch-weights"),
                      links={"model": model.key})
    return False


def choose_start(model: catalog.Model) -> tuple[str, str | None, str] | None:
    """What the new run starts from: (description, checkpoint path or None for the public
    pretrained weights, a short name for job titles).  None when the user backs out."""
    trained = [r for r in catalog.runs(model) if r.checkpoints() and not r.active]
    options = [("public", "The public pretrained weights",
                public_weights(model) + " — how every published run started")]
    if not catalog.supports(model, "--init-from"):
        ui.dim(f"  Fine-tuning needs the --init-from flag, which this checkout's "
               f"{model.script} does not have yet — sync the updated entry points.")
    else:
        if trained:
            options.append(("version", f"A trained {model.label} version (fine-tune)",
                            f"continue from one of {len(trained)} — e.g. on new or "
                            "corrected data"))
        options.append(("file", "A checkpoint file…",
                        f"a {model.label} .pt/.pth, e.g. from the Hugging Face checkpoints repo"))
    picked = ui.choose("What should the new run start from?", options)
    if picked is None:
        return None
    if picked == "public":
        return public_weights(model), None, ""
    if picked == "version":
        got = common.pick_version("start from", model=model)
        if got is None:
            return None
        run, ckpt = got
        return (f"{model.label} {run.run_id}, {ckpt.kind} checkpoint ({ckpt.file})",
                ckpt.path, run.run_id)
    path = ui.ask_path("Checkpoint file:", dirs_only=False)
    if not path:
        return None
    if not path.endswith((".pt", ".pth")):
        ui.warn(f"{os.path.basename(path)} does not look like a checkpoint (.pt / .pth).")
        if not ui.confirm("Use it anyway?", default=False):
            return None
    ui.dim(f"  It must be a {model.label} checkpoint: the entry point loads it into the "
           f"{model.label} network, and the run is filed under {rel(model.ckpt_dir)}/.")
    return rel(path), path, os.path.basename(path)


def fine_tune() -> None:
    ui.title("Train a model", "Fine-tune a trained version")
    ui.dim("A new run that starts from a trained version's weights instead of the public "
           "pretrained ones — to continue on new or corrected data, say.  It gets its own "
           "folder and scores; the version it starts from is left as it is.")
    ui.console.print()
    got = common.pick_version("fine-tune")
    if got is None:
        return
    run, ckpt = got
    plan(run.model, catalog.runs(run.model),
         start=(f"{run.model.label} {run.run_id}, {ckpt.kind} checkpoint ({ckpt.file})",
                ckpt.path, run.run_id))


def plan(model: catalog.Model, runs: list[catalog.Run],
         start: tuple[str, str | None, str] | None = None) -> None:
    ui.title("Train a model", model.label)
    if not _ready_to_train(model):
        return
    if start is None:
        start = choose_start(model)
        if start is None:
            return
        ui.title("Train a model", model.label)
    start_text, init_from, start_name = start
    if init_from is None and model.owl and not _release_weights_ready(model):
        return

    finished = [r for r in runs if not r.active and r.checkpoint]
    took = next((d for d in (previous_duration(r) for r in finished) if d and d > 0), None)
    pairs = [("Model", model.about),
             ("Start from", start_text + (" — fine-tune" if init_from else "")),
             *hyperparameters(model),
             ("Data", data_line(model, "train and validation splits")),
             ("Output", f"{rel(model.ckpt_dir)}/<start time>/  (a new version)"),
             ("Previous run", f"took ≈ {human_duration(took)}" if took else "none finished")]
    ui.show(ui.kv(pairs, title="Fine-tuning plan" if init_from else "Training plan"))
    ui.dim("  Hyperparameters are set in data_prep/config.py and shared by every run.")
    if init_from:
        ui.dim("  Only the weights carry over: the optimiser, learning-rate schedule and "
               "epoch count start afresh, and early stopping still applies.")
        if model.key in ("yolov5", "yolo11", "yolo26"):
            ui.dim("  Ultralytics restarts mosaic augmentation and the BatchNorm statistics "
                   "re-adapt to it, so the first epochs score below the starting version "
                   "before recovering.  The run keeps its best checkpoint: judge it by that.")
    ui.console.print()

    busy = [r for r in runs if r.active]
    if busy:
        ui.warn(f"A {model.label} run is already in progress ({busy[0].run_id}).")

    epochs = None
    if model.epochs_flag:
        epochs = ui.ask_int("Maximum epochs:", default=default_epochs(model), lo=1, hi=1000)
        if epochs is None:
            return
        if epochs == default_epochs(model):
            epochs = None
    evaluate_after = ui.confirm("Evaluate on the test set when training finishes?", default=True)
    gpu = common.pick_gpu("training")
    if gpu is None:
        return

    cmd = python_cmd(model.script, *model.variant, "--train")
    if init_from:
        cmd += ["--init-from", init_from]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if evaluate_after:
        cmd.append("--eval")
    ui.console.print()
    ui.dim("  " + " ".join(["CUDA_VISIBLE_DEVICES=" + gpu, *[rel(c) for c in cmd[1:]]]))
    verb = "fine-tuning" if init_from else "training"
    if not ui.confirm(f"Start {verb} {model.label}?", default=True):
        return
    title = f"Fine-tune {model.label} from {start_name}" if init_from else f"Train {model.label}"
    links = {"model": model.key}
    if init_from:
        links["init_from"] = init_from
    common.launch(title + (" + evaluate" if evaluate_after else ""), "train", cmd, gpu=gpu,
                  links=links)


def resume(candidates: list[catalog.Run]) -> None:
    ui.title("Train a model", "Resume an OWL run")
    ui.dim("OWL checkpoints every epoch, so an interrupted run loses at most the epoch in "
           "flight; resuming restores the optimiser state and best score so far.")
    ui.console.print()
    options = []
    for r in candidates:
        epoch = owl_epoch(r)
        cap = r.info.get("epochs", config.MDO_EPOCHS)
        options.append((r, f"{r.model.label} {r.run_id}",
                        f"reached epoch {epoch}/{cap}" if epoch is not None else "epoch unknown"))
    picked = ui.choose("Which run should continue?", options)
    if picked is None:
        return
    gpu = common.pick_gpu("training")
    if gpu is None:
        return
    flags = [f.format(run=picked.run_id) for f in resume_syntax()]
    cmd = python_cmd(picked.model.script, *picked.model.variant, "--train", *flags)
    if ui.confirm(f"Resume {picked.model.label} {picked.run_id}?", default=True):
        common.launch(f"Resume {picked.model.label} {picked.run_id}", "train", cmd, gpu=gpu,
                      links={"model": picked.model.key, "run": picked.run_id})
