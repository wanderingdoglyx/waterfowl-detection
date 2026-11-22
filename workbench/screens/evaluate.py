"""
Evaluate & compare: the leaderboard, one run in detail, test-set evaluation, the zero-shot
baselines, the cross-model analysis and the efficiency benchmarks.

Everything that computes runs the project's own entry point as a background job
(`<model>/main.py --eval --run X`, `baseline/main.py`, `analysis/main.py -m <manifest>`,
`python -m analysis.bench_extra`); everything that only reads — the leaderboard, a run's
details, the baseline before/after summary — is shown directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from rich.panel import Panel
from rich.text import Text

from data_prep import config
from workbench import catalog, imaging, paths, ui
from workbench.jobs import python_cmd
from workbench.paths import ROOT, rel
from workbench.screens import common
from workbench.screens import train as train_screen
from workbench.util import ago, num, pct, slug, timestamp


def run() -> None:
    while True:
        ui.title("Evaluate & compare")
        options = [
            ("board", "Leaderboard", "every scored version, ranked"),
            ("inspect", "Inspect a trained version", "lineage, scores, checkpoints, panels"),
            ("eval", "Test a trained version", "choose model, version and checkpoint to load"),
            ("baseline", "Zero-shot baselines", "pretrained checkpoints before fine-tuning"),
            ("analysis", "Cross-model analysis", "accuracy vs cost, size, density, conditions"),
            ("bench", "Efficiency benchmarks", "speed and memory; needs an idle GPU"),
        ]
        picked = ui.choose("What do you want to do?", options)
        if picked is None:
            return
        {"board": leaderboard, "inspect": inspect_pick, "eval": evaluate_pick,
         "baseline": baselines, "analysis": analysis, "bench": benchmarks}[picked]()


# ── Leaderboard ───────────────────────────────────────────────────────────────

SORTS = {
    "ap": ("point AP", False), "f1": ("F1 at t*", False), "mae": ("MAE (counting)", True),
    "rmse": ("RMSE (counting)", True), "map30": ("mAP30", False),
}


def leaderboard() -> None:
    key = "ap"
    while True:
        ui.title("Evaluate & compare", "Leaderboard")
        scored = common.scored_entries(catalog.runs())
        label, lower = SORTS[key]

        def order(runs):
            present = [r for r in runs if getattr(r.metrics, key) is not None]
            missing = [r for r in runs if getattr(r.metrics, key) is None]
            return sorted(present, key=lambda r: getattr(r.metrics, key),
                          reverse=not lower) + missing

        current = order([r for r in scored if r.metrics.split == "current"])
        legacy = order([r for r in scored if r.metrics.split == "v1"])
        subset = sorted(order([r for r in scored if r.metrics.split == "subset"]),
                        key=lambda r: -(r.metrics.n_images or 0))
        if current:
            ui.show(common.runs_table(current, title=f"Current test split "
                                                     f"({catalog.CURRENT_TEST_CROPS:,} crops) — by {label}"))
        if legacy:
            ui.show(common.runs_table(legacy, title=f"Earlier test split "
                                                    f"({catalog.LEGACY_TEST_CROPS:,} crops, Baseline v1) — "
                                                    "not comparable with the table above"))
        if subset:
            ui.show(common.runs_table(subset, title="Data-folder test splits — compare only "
                                                    "rows with the same number of test crops"))
        ui.dim(common.RUNS_CAPTION)
        ui.console.print()
        options = [(k, f"Sort by {v[0]}", "lower is better" if v[1] else "")
                   for k, v in SORTS.items() if k != key]
        options.append(("export", "Export this leaderboard", "Markdown, CSV and HTML"))
        picked = ui.choose("Next:", options)
        if picked is None:
            return
        if picked == "export":
            from workbench import reports
            out = reports.leaderboard_report(current + legacy + subset, sort_key=key)
            ui.ok(f"Wrote {rel(out)}/")
            ui.note(imaging.open_external(os.path.join(out, "leaderboard.md")))
            ui.pause()
        else:
            key = picked


# ── One version in detail ─────────────────────────────────────────────────────

def inspect_pick() -> None:
    ui.title("Evaluate & compare", "Inspect a trained version")
    got = common.pick_version("inspect", include_active=True, with_checkpoint=False)
    if got:
        inspect(got[0])


def _results_panel(text: str, title: str) -> Panel:
    return Panel(Text(text), title=f"[bold]{title}[/]", title_align="left",
                 border_style="grey50")


def inspect(run: catalog.Run) -> None:
    while True:
        run = catalog.load_run(run.model, run.run_id) or run
        ui.title("Evaluate & compare", run.name)
        pairs = [(k, common.styled(v) if k == "Status" else v) for k, v in run.details()]
        children = [r for r in catalog.runs(run.model)
                    if r.parent and r.parent[1] == run.run_id]
        if children:
            pairs.append(("Fine-tuned into", ", ".join(c.run_id for c in children)))
        pairs.append(("Last activity", ago(run.last_activity)))
        ui.show(ui.kv(pairs, title=run.name))
        best_text, last_text = run.eval_text("best"), run.eval_text("last")
        if best_text:
            ui.show(_results_panel(best_text, "test scores — best checkpoint"))
        if last_text:
            ui.show(_results_panel(last_text, "test scores — last checkpoint (eval_last/)"))
        if not best_text and not last_text:
            ui.dim("  Not tested yet.")
        checkpoints = run.checkpoints()
        options = []
        if checkpoints:
            options.append(("test", "Test this version on the test set",
                            "choose which checkpoint to load"))
        if checkpoints and not run.active:
            options.append(("finetune", "Fine-tune a new version from it",
                            "a new run starting from these weights"))
        for which in ("best", "last"):
            panels = run.examples(which)
            if panels:
                options.append((f"ex-{which}", f"Look at the example panels ({len(panels)}, "
                                               f"{which} checkpoint)", "ground truth | predictions"))
        if imaging.viewer_available():
            options.append(("open", "Open the version's folder in VS Code / a viewer", ""))
        if checkpoints and not run.active:
            options.append(("detect", "Count birds in a survey with this version", ""))
        if run.metrics:
            options.append(("report", "Export a report for this version", ""))
        picked = ui.choose("Next:", options)
        if picked is None:
            return
        if picked in ("test", "finetune", "detect"):
            usable = [c for c in checkpoints if not run.active or c.kind == "last"] or checkpoints
            ckpt = usable[0] if len(usable) == 1 else ui.choose(
                "Which checkpoint?", [(c, c.kind, c.what) for c in usable], default=usable[0])
            if ckpt is None:
                continue
            if picked == "test":
                evaluate(run, ckpt)
            elif picked == "finetune":
                from workbench.screens import train
                train.plan(run.model, catalog.runs(run.model),
                           start=(f"{run.name}, {ckpt.kind} checkpoint ({ckpt.file})",
                                  ckpt.path, run.run_id))
            else:
                from workbench.screens import detect
                detect.start(run=run, checkpoint=ckpt)
        elif picked.startswith("ex-"):
            which = picked[3:]
            browse_images(run.examples(which), f"{run.name} ({which})")
        elif picked == "open":
            ui.note(imaging.open_external(run.path))
            ui.pause()
        elif picked == "report":
            from workbench import reports
            out = reports.run_report(run)
            ui.ok(f"Wrote {rel(out)}")
            ui.note(imaging.open_external(out))
            ui.pause()


def browse_images(files: list[str], label: str) -> None:
    i = 0
    while True:
        ui.title("Evaluate & compare", label, f"example {i + 1}/{len(files)}")
        width = ui.console.size.width - 2
        height = max(8, ui.console.size.height - 12)
        ui.show(imaging.rich_image(files[i], width, height))
        ui.dim(f"{os.path.basename(files[i])} — left: ground truth · right: predictions "
               "(terminal preview; open it for full resolution)")
        options = [("next", "Next", ""), ("prev", "Previous", "")]
        if imaging.viewer_available():
            options.append(("open", "Open at full resolution", ""))
        picked = ui.choose("", options, default="next")
        if picked is None:
            return
        if picked == "open":
            ui.note(imaging.open_external(files[i]))
        else:
            i = (i + (1 if picked == "next" else -1)) % len(files)


# ── Test-set evaluation ───────────────────────────────────────────────────────

def evaluate_pick() -> None:
    ui.title("Evaluate & compare", "Test a trained version")
    ui.dim("Scores one set of weights on the test split — mAP30 and the OWL-paper point "
           "metrics.  Pick the model, the version (one training run), and which of its "
           "checkpoints to load: best (what its published scores describe) or last.")
    ui.console.print()
    got = common.pick_version("test", include_active=True)
    if got:
        evaluate(*got)


def evaluate(run: catalog.Run, ckpt: catalog.Checkpoint) -> None:
    ui.title("Evaluate & compare", f"Test {run.name}")
    last = ckpt.kind == "last"
    where = rel(catalog.results_dir(run.path, run.model.kind, ckpt.kind))
    existing = run.last_metrics if last else run.metrics
    ui.show(ui.kv([
        ("Version", f"{run.name} — started from {run.started_from()}"),
        ("Checkpoint", f"{ckpt.kind}: {ckpt.what}"),
        ("Test split", train_screen.data_line(run.model, "its test split")),
        ("Scores go to", f"{where}/  " + ("(kept apart from the version's own scores)" if last
                                          else "(the version's own scores)")),
    ]))
    ui.console.print()
    if run.active:
        ui.note("This version is still training: this scores a snapshot of its latest "
                "epoch, and testing it again later replaces the snapshot's scores.")
    elif existing and existing.split != "current" and not last:
        ui.note(f"These weights were scored on a {existing.n_images:,}-crop test split; testing "
                "again scores them on the test split of the data below, and replaces those "
                "scores.")
    elif existing:
        ui.warn("These weights already have scores; testing again replaces them.")
    examples = ui.ask_int("How many example panels to save (0 for none)?", default=40, lo=0,
                          hi=1000)
    if examples is None:
        return
    gpu = common.pick_gpu("test")
    if gpu is None:
        return
    cmd = python_cmd(run.model.script, *run.model.variant, "--eval", "--run", run.run_id,
                     "--examples", str(examples))
    if last:
        cmd += ["--checkpoint", "last"]
    title = f"Test {run.name}" + (" (last checkpoint)" if last else "")
    if ui.confirm(f"{title}?", default=True):
        common.launch(title, "eval", cmd, gpu=gpu,
                      links={"model": run.model.key, "run": run.run_id,
                             "checkpoint": ckpt.kind})


# ── Zero-shot baselines ───────────────────────────────────────────────────────

def _baseline_runs() -> dict[str, list[str]]:
    found = {}
    for key in config.BASELINE_MODELS:
        d = os.path.join(config.BASELINE_DIR, key)
        found[key] = sorted((e.name for e in os.scandir(d) if e.is_dir()), reverse=True) \
            if os.path.isdir(d) else []
    return found


def baselines() -> None:
    while True:
        ui.title("Evaluate & compare", "Zero-shot baselines")
        ui.dim("Each model's starting checkpoint, scored on the test split before any "
               "fine-tuning — the documented 'before' behind every result.")
        ui.console.print()
        done = _baseline_runs()
        rows = [[spec["label"], spec["family"], done[k][0] if done[k] else "—"]
                for k, spec in config.BASELINE_MODELS.items()]
        ui.show(ui.table(["Pretrained checkpoint", "Family", "Latest baseline run"], rows))
        options = [
            ("summary", "Show the before/after table", "reads existing results; instant"),
            ("one", "Score one pretrained checkpoint", "background job"),
            ("all", "Score all eight", "background job"),
            ("save", "Save the before/after table", "output/baselines/summary.txt"),
        ]
        picked = ui.choose("What next?", options)
        if picked is None:
            return
        if picked in ("summary", "save"):
            flag = "--summary" if picked == "summary" else "--save"
            out = subprocess.run(python_cmd("baseline/main.py", flag), cwd=ROOT,
                                 capture_output=True, text=True)
            text = (out.stdout + out.stderr).rstrip()
            if picked == "summary":
                _page(text)
            else:
                ui.ok(text.splitlines()[-1] if text else "saved")
                ui.pause()
        elif picked == "one":
            key = ui.choose("Which checkpoint?", [(k, s["label"], s["family"])
                                                 for k, s in config.BASELINE_MODELS.items()])
            if key:
                gpu = common.pick_gpu("baseline")
                if gpu:
                    common.launch(f"Baseline {config.BASELINE_MODELS[key]['label']}", "baseline",
                                  python_cmd("baseline/main.py", "--model", key), gpu=gpu)
        elif picked == "all":
            gpu = common.pick_gpu("baselines")
            if gpu:
                common.launch("Baselines: all eight pretrained checkpoints", "baseline",
                              python_cmd("baseline/main.py", "--all"), gpu=gpu)


def _page(text: str) -> None:
    """Show wide text: straight to the screen if it fits, otherwise through less -S."""
    width = max((len(line) for line in text.splitlines()), default=0)
    if width <= ui.console.size.width or shutil.which("less") is None:
        ui.console.print(Text(text), soft_wrap=False, overflow="ignore", crop=False)
        ui.pause()
        return
    ui.note("The table is wider than this terminal: ←/→ scroll, q returns.")
    subprocess.run(["less", "-S", "-R", "-P", "←/→ scroll · q back"], input=text, text=True)


# ── Cross-model analysis ──────────────────────────────────────────────────────

def analysis() -> None:
    ui.title("Evaluate & compare", "Cross-model analysis")
    ui.dim("Compares finished runs: accuracy against cost, by bird size and density, and "
           "(optionally) by survey conditions with a failure-case library.  The runs are "
           "chosen by a manifest; the workbench writes one from your choice.")
    ui.console.print()
    source = ui.choose("Which runs?", [
        ("pick", "Choose runs", "writes a new manifest"),
        ("v1", "The published set (models_v1.docx)", "the Baseline v1 comparison"),
    ])
    if source is None:
        return
    if source == "v1":
        manifest, name = os.path.join(ROOT, "models_v1.docx"), "models_v1"
    else:
        scored = catalog.evaluated()
        newest = {}
        for r in scored:
            if r.metrics.split == "current":
                newest.setdefault(r.model.key, r)
        options = [(r, f"{r.model.label:13s} {r.run_id}",
                    f"AP {pct(r.metrics.ap)} · MAE {num(r.metrics.mae)}"
                    + {"current": "", "v1": " · earlier split",
                       "subset": " · data-folder split"}[r.metrics.split])
                   for r in scored]
        chosen = ui.choose_many("Tick the runs to compare (one per model):", options,
                                selected=list(newest.values()), min_count=2)
        if not chosen:
            return
        keys = [r.model.key for r in chosen]
        if len(set(keys)) != len(keys):
            ui.error("A manifest takes one run per model — untick the duplicates.")
            ui.pause()
            return
        name = ui.ask("Name for this comparison:", default=f"workbench_{timestamp()[:10]}",
                      validate=lambda s: None if s and slug(s) == s else
                      "letters, digits, '-', '_' and '.' only")
        if not name:
            return
        os.makedirs(paths.MANIFESTS_DIR, exist_ok=True)
        manifest = os.path.join(paths.MANIFESTS_DIR, f"{name}.txt")
        with open(manifest, "w") as f:
            f.write("# model, run  — written by the workbench\n")
            for r in chosen:
                f.write(f"{r.model.key}, {r.run_id}\n")

    scope = ui.choose("How much?", [
        ("accuracy", "Accuracy tables only", "from stored metrics; seconds, no GPU"),
        ("tasks", "+ Tasks 2.1–2.4", "Pareto, size, density; runs inference if not cached"),
        ("all", "+ Tasks and survey conditions", "habitat, altitude, failure library"),
    ])
    if scope is None:
        return
    cmd = python_cmd("analysis/main.py", "-m", manifest)
    if scope in ("tasks", "all"):
        cmd.append("--tasks")
    if scope == "all":
        cmd.append("--conditions")
    if scope != "accuracy":
        ui.note("Models without a cached prediction set are run over the whole test split "
                "first (≈5 min for OWL-C/YOLO-NAS, 25 for Faster R-CNN, 96 for OWL-D).")
    gpu = common.pick_gpu("analysis") if scope != "accuracy" else None
    if scope != "accuracy" and gpu is None:
        return
    ui.note(f"Results will be written to {rel(os.path.join(config.OUTPUT_DIR, 'analysis', name))}/")
    if ui.confirm("Start the analysis?", default=True):
        common.launch(f"Analysis: {name} ({scope})", "analysis", cmd, gpu=gpu)


# ── Efficiency benchmarks ─────────────────────────────────────────────────────

def benchmarks() -> None:
    ui.title("Evaluate & compare", "Efficiency benchmarks")
    ui.warn("Benchmarks must run on an idle GPU — anything else on the card corrupts the "
            "timings.  Busy GPUs are marked below.")
    ui.console.print()
    which = ui.choose("Which benchmark?", [
        ("eff", "Ultralytics efficiency", "python -m data_prep.efficiency"),
        ("extra", "All eight models: batch sweep + end-to-end", "python -m analysis.bench_extra"),
        ("train", "Training-step throughput", "python -m analysis.bench_train"),
    ])
    if which is None:
        return
    gpu = common.pick_gpu("benchmark")
    if gpu is None:
        return
    module = {"eff": "data_prep.efficiency", "extra": "analysis.bench_extra",
              "train": "analysis.bench_train"}[which]
    if ui.confirm(f"Start python -m {module}?", default=True):
        common.launch(f"Benchmark: {module}", "benchmark", python_cmd("-m", module), gpu=gpu)
