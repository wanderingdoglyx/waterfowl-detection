"""
What exists right now: the eight models, their runs and scores, the data folders, the
GPUs, and any pipeline process already running.  Read-only — nothing here writes a file.

Scores are read from the files each model's own `--eval` writes, never recomputed:

    box models   <run>/paper_point_metrics.json  +  mAP30 from <run>/eval_results.txt
    OWL models   <run>/eval/metrics.json  (paper_point_metrics + pseudo_box_map30)

so a number shown here is the number in that run's eval_results.txt, and a run that was
never evaluated shows no number rather than a guessed one.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field

from analysis import registry
from data_prep import config
from workbench import paths
from workbench.util import read_json

RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# The two test splits a published run may have been scored on (README, "Are the splits
# fixed?").  Scores from different splits are not comparable, so tables mark them.
CURRENT_TEST_CROPS = 28588
LEGACY_TEST_CROPS = 25708


# ── Models ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Model:
    key: str                  # checkpoint folder under output/checkpoints/
    script: str               # entry point, relative to the repository
    variant: tuple[str, ...]  # flags selecting this variant of a shared entry point
    baseline: str             # key in config.BASELINE_MODELS
    about: str
    ms_per_tile: float        # measured inference cost (README "Caches and runtimes")
    epochs_flag: bool = False # entry point accepts --epochs
    resumable: bool = False   # entry point accepts --resume

    @property
    def spec(self) -> registry.ModelSpec:
        return registry.resolve(self.key)

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def owl(self) -> bool:
        return self.kind == "point"

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(config.OUTPUT_DIR, "checkpoints", self.key)

    @property
    def mdo_key(self) -> str | None:
        return self.variant[1] if self.owl else None


# ms_per_tile: test-split inference time / 28,588 crops on an RTX 4080 SUPER, including
# image loading — a planning figure for "how long will this survey take", not a benchmark.
MODELS: tuple[Model, ...] = (
    Model("yolonas", "yolo_nas/main.py", (), "yolonas",
          "YOLO-NAS-m · super-gradients · box", 10.5),
    Model("yolov5", "yolov5/main.py", (), "yolov5",
          "YOLOv5m (anchor-free u) · Ultralytics · box", 31.5),
    Model("yolo11", "yolo_ultralytics/main.py", ("--model", "yolo11"), "yolo11",
          "YOLO11m · Ultralytics · box", 12.6, epochs_flag=True),
    Model("yolo26", "yolo_ultralytics/main.py", ("--model", "yolo26"), "yolo26",
          "YOLO26m · Ultralytics · box, NMS-free", 14.7, epochs_flag=True),
    Model("fasterrcnn", "faster_rcnn/main.py", (), "fasterrcnn",
          "Faster R-CNN R50-FPN · Detectron2 · box", 52.5),
    Model("mdo_owl_c", "megadetector_overhead/main.py", ("--model", "OWLC"), "owl_c",
          "OWL-C · DLA-34 · point detector", 10.5, epochs_flag=True, resumable=True),
    Model("mdo_owl_t", "megadetector_overhead/main.py", ("--model", "OWLT"), "owl_t",
          "OWL-T · DLA-34 + Swin · point detector", 23.1, epochs_flag=True, resumable=True),
    Model("mdo_owl_d", "megadetector_overhead/main.py", ("--model", "OWLD_H"), "owl_d",
          "OWL-D · DINOv3 ViT-H+ · point detector (slow)", 201.5,
          epochs_flag=True, resumable=True),
)

_BY_KEY = {m.key: m for m in MODELS}


def model(key: str) -> Model:
    return _BY_KEY[registry.resolve(key).key]


_SOURCES: dict[str, str] = {}


def supports(model_: Model, flag: str) -> bool:
    """Whether this checkout's entry point for the model declares `flag`.

    The code is edited on more than one machine, so a flag the workbench relies on
    (--init-from, --checkpoint) may not have reached every copy yet; reading the parser
    declaration lets the workbench leave out what the entry point cannot do instead of
    launching a job that fails on an unknown argument.
    """
    path = os.path.join(paths.ROOT, model_.script)
    if path not in _SOURCES:
        try:
            with open(path) as f:
                _SOURCES[path] = f.read()
        except OSError:
            _SOURCES[path] = ""
    return f'"{flag}"' in _SOURCES[path]


def reads_data_folder(model_: Model) -> bool:
    """Whether the model's entry point reads the chosen data folder (WATERFOWL_DATASET).

    Entry points do so by calling pipeline.auto.ensure_data() when they start.  One that
    does not (megadetector_overhead/main.py today) trains and tests on the whole corpus
    whatever folder is chosen, so the workbench says so instead of implying otherwise.
    """
    supports(model_, "")                       # fills the source cache
    return "ensure_data(" in _SOURCES[os.path.join(paths.ROOT, model_.script)]


# ── Runs and their scores ─────────────────────────────────────────────────────

@dataclass
class Metrics:
    """One run's test-set scores, normalised across the two families' file layouts."""
    ap: float | None = None
    auc_pr: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    mae: float | None = None
    rmse: float | None = None
    t_star: float | None = None
    count_error_pct: float | None = None
    n_images: int | None = None
    n_gt: int | None = None
    map30: float | None = None        # percent
    mar30: float | None = None        # percent
    map30_pseudo: bool = False        # OWL: points wrapped in 28 px boxes
    ap_ci: tuple | None = None
    mae_ci: tuple | None = None
    source: str = ""

    @property
    def split(self) -> str:
        """Which test split the scores come from: "current" (the whole corpus's test
        split), "v1" (the earlier one) or "subset" (a data folder's own test photographs).
        Only scores from the same split are comparable."""
        if self.n_images == CURRENT_TEST_CROPS:
            return "current"
        if self.n_images == LEGACY_TEST_CROPS:
            return "v1"
        return "subset"


def _from_paper(pm: dict, source: str) -> Metrics:
    det, cnt, ci = pm.get("detection", {}), pm.get("counting", {}), pm.get("ci95", {})
    proto = pm.get("protocol", {})
    return Metrics(
        ap=det.get("ap"), auc_pr=det.get("auc_pr"),
        precision=det.get("precision_at_t_star"), recall=det.get("recall_at_t_star"),
        f1=det.get("f1_at_t_star"), mae=cnt.get("mae"), rmse=cnt.get("rmse"),
        t_star=cnt.get("t_star"), count_error_pct=cnt.get("signed_pct_error"),
        n_images=proto.get("n_images"), n_gt=proto.get("n_gt_points"),
        ap_ci=tuple(ci["ap"]) if "ap" in ci else None,
        mae_ci=tuple(ci["mae"]) if "mae" in ci else None,
        source=source,
    )


_MAP_RE = re.compile(r"mAP30\s*:\s*([\d.]+)%")
_MAR_RE = re.compile(r"mAR30\s*:\s*([\d.]+)%")


def results_dir(run_dir: str, kind: str, which: str = "best") -> str:
    """Where an evaluation of the run's `which` checkpoint writes its scores.

    The best checkpoint's scores are the run's own: the run folder for box models,
    eval/ for OWL.  `--eval --checkpoint last` writes to eval_last/ for every family,
    so a second evaluation never overwrites the first.
    """
    if which == "last":
        return os.path.join(run_dir, "eval_last")
    return os.path.join(run_dir, "eval") if kind == "point" else run_dir


def read_metrics(folder: str, kind: str) -> Metrics | None:
    """Scores from one results folder (see results_dir)."""
    if kind == "point":
        path = os.path.join(folder, "metrics.json")
        payload = read_json(path)
        if not payload or "paper_point_metrics" not in payload:
            return None
        m = _from_paper(payload["paper_point_metrics"], path)
        pseudo = payload.get("pseudo_box_map30") or {}
        m.map30, m.mar30, m.map30_pseudo = pseudo.get("ap30"), pseudo.get("ar30"), True
        return m

    path = os.path.join(folder, "paper_point_metrics.json")
    payload = read_json(path)
    if not payload:
        return None
    m = _from_paper(payload, path)
    try:
        with open(os.path.join(folder, "eval_results.txt")) as f:
            text = f.read()
        if (hit := _MAP_RE.search(text)):
            m.map30 = float(hit.group(1))
        if (hit := _MAR_RE.search(text)):
            m.mar30 = float(hit.group(1))
    except OSError:
        pass
    return m


def find_checkpoint(model_: Model, run_dir: str) -> str | None:
    """The weights this run's own --eval would load (same rule as analysis.predict)."""
    from analysis.predict import checkpoint_for
    try:
        return checkpoint_for(model_.spec, os.path.basename(run_dir))
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class Checkpoint:
    """One set of weights inside a run: what `--checkpoint best|last` would load."""
    kind: str        # "best" or "last"
    path: str
    what: str        # one line for a menu

    @property
    def file(self) -> str:
        return os.path.basename(self.path)


def _first(*candidates: str | None) -> str | None:
    return next((c for c in candidates if c and os.path.exists(c)), None)


def run_checkpoints(model_: Model, run_dir: str, active: bool = False) -> list[Checkpoint]:
    """The run's best and last weights, resolved exactly as its entry point's --eval
    resolves `--checkpoint best` and `--checkpoint last`.  `last` is listed only when it
    is a different file from `best`."""
    import glob

    def join(*p):
        return os.path.join(run_dir, *p)

    key = model_.key
    if key in ("yolov5", "yolo11", "yolo26"):
        best = _first(join("weights", "best.pt"), join("weights", "last.pt"))
        last = _first(join("weights", "last.pt"))
        best_why = "best Ultralytics fitness on validation"
    elif key == "yolonas":
        def sg(name):
            direct = join(name)
            nested = sorted(glob.glob(join("RUN_*", name)), reverse=True)
            return _first(direct, *nested)
        best = sg("ckpt_best.pth") or sg("ckpt_latest.pth")
        last = sg("ckpt_latest.pth")
        best_why = "best validation mAP30"
    elif key == "fasterrcnn":
        periodic = sorted(glob.glob(join("model_[0-9]*.pth")))
        best = _first(join("model_best.pth"), join("model_final.pth"))
        last = _first(join("model_final.pth"), periodic[-1] if periodic else None)
        best_why = "best validation mAP30"
    else:
        best = _first(join("weights", "best.pth"), join("best_model.pth"))
        last = _first(join("latest_model.pth"))
        best_why = "best validation point-F1"
    found = []
    if best:
        found.append(Checkpoint("best", best, f"{best_why}"
                                + (" so far" if active else "") + f" — {os.path.basename(best)}"))
    if last and (not best or os.path.realpath(last) != os.path.realpath(best)):
        if active:
            when = "the latest epoch so far"
        elif key == "fasterrcnn" and not last.endswith("model_final.pth"):
            when = "the newest periodic checkpoint (saved every 5 epochs)"
        else:
            when = "the end of training"
        found.append(Checkpoint("last", last, f"{when} — {os.path.basename(last)}"))
    return found


@dataclass
class Run:
    model: Model
    run_id: str
    path: str
    checkpoint: str | None
    metrics: Metrics | None
    info: dict
    experiment: dict
    last_activity: float
    active: bool = False       # a training/eval process is working on it right now
    last_metrics: Metrics | None = None   # scores of --checkpoint last (eval_last/)

    def checkpoints(self) -> list[Checkpoint]:
        return run_checkpoints(self.model, self.path, self.active)

    # ── Lineage ──
    @property
    def init_from(self) -> str | None:
        """The checkpoint a fine-tuned run started from (None: the public weights).

        run_info.json records it; runs whose run_info is written only when training ends
        (YOLO11/26) have it in experiment.json from the start.
        """
        path = self.info.get("init_from")
        if not path:
            weights = (self.experiment.get("pretrained") or {}).get("weights") or ""
            if weights.startswith(os.path.join(config.OUTPUT_DIR, "checkpoints") + os.sep):
                path = weights
        return path or None

    @property
    def parent(self) -> tuple[Model, str] | None:
        """(model, run id) of the version this one was fine-tuned from."""
        path = self.init_from
        if not path:
            return None
        rel_path = os.path.relpath(path, os.path.join(config.OUTPUT_DIR, "checkpoints"))
        parts = rel_path.split(os.sep)
        if rel_path.startswith("..") or len(parts) < 3 or not RUN_ID_RE.match(parts[1]):
            return None
        try:
            return model(parts[0]), parts[1]
        except KeyError:
            return None

    def started_from(self) -> str:
        if self.init_from:
            parent = self.parent
            if parent:
                return f"{parent[0].label} {parent[1]} ({os.path.basename(self.init_from)})"
            return paths.rel(self.init_from)
        weights = (self.experiment.get("pretrained") or {}).get("weights")
        if weights:
            return f"public pretrained weights: {os.path.basename(weights)}"
        return "public pretrained weights"

    @property
    def name(self) -> str:
        return f"{self.model.label} {self.run_id}"

    @property
    def status(self) -> str:
        if self.active:
            return "running"
        if self.metrics:
            return "evaluated"
        if self.checkpoint:
            return "trained"
        return "no checkpoint"

    @property
    def data_folder(self) -> str | None:
        root = (self.experiment.get("dataset") or {}).get("root")
        return os.path.basename(root) if root else None

    @property
    def epochs_done(self) -> int | None:
        csv_path = os.path.join(self.path, "results.csv")      # Ultralytics
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                return max(0, sum(1 for _ in f) - 1)
        return None

    def eval_text(self, which: str = "best") -> str | None:
        """eval_results.txt of the best checkpoint (the run's own) or of the last."""
        path = os.path.join(results_dir(self.path, self.model.kind, which), "eval_results.txt")
        if os.path.exists(path):
            with open(path) as f:
                return f.read().rstrip()
        return None

    def examples(self, which: str = "best") -> list[str]:
        """The ground truth | prediction panels an --eval saved."""
        import glob
        base = self.path if which == "best" else os.path.join(self.path, "eval_last")
        found = []
        for pattern in ("examples/*.jpg", "examples/*.png"):
            found += glob.glob(os.path.join(base, pattern))
        return sorted(found)

    def details(self) -> list[tuple[str, str]]:
        """Provenance as label/value pairs: where it is, what it read, what made it."""
        info, exp = self.info, self.experiment
        code, hw = exp.get("code", {}), exp.get("hardware", {})
        ckpts = []
        for c in self.checkpoints():
            size = os.path.getsize(c.path) / 1e6
            ckpts.append(f"{c.kind}: {paths.rel(c.path)} ({size:,.0f} MB)")
        crops = "—"
        if info.get("n_train"):
            crops = (f"train {info['n_train']:,} · val {info.get('n_val', 0):,} · "
                     f"test {info.get('n_test', 0):,}")
        pairs = [("Status", self.status), ("Folder", paths.rel(self.path)),
                 ("Started from", self.started_from()),
                 ("Checkpoints", "\n".join(ckpts) or "—"),
                 ("Data", self.data_folder or "not recorded"),
                 ("Crops", crops), ("Epoch cap", str(info.get("epochs", "—")))]
        if self.epochs_done is not None:
            pairs.append(("Epochs run", str(self.epochs_done)))
        if code:
            pairs.append(("Code", f"{code.get('git_commit_short', '?')}"
                                  + (" (uncommitted changes)" if code.get("git_dirty") else "")))
        gpu = (hw.get("gpus") or [{}])[0].get("name")
        if gpu or hw.get("hostname"):
            pairs.append(("Trained on",
                          f"{gpu or '?'} · {(hw.get('hostname') or '?').split('.')[0]}"))
        return pairs


def _last_activity(run_dir: str) -> float:
    latest = 0.0
    try:
        with os.scandir(run_dir) as it:
            for entry in it:
                try:
                    latest = max(latest, entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    pass
    except OSError:
        pass
    return latest


def load_run(model_: Model, run_id: str, procs: list | None = None) -> Run | None:
    run_dir = os.path.join(model_.ckpt_dir, run_id)
    if not os.path.isdir(run_dir):
        return None
    run = Run(
        model=model_, run_id=run_id, path=run_dir,
        checkpoint=find_checkpoint(model_, run_dir),
        metrics=read_metrics(results_dir(run_dir, model_.kind), model_.kind),
        info=read_json(os.path.join(run_dir, "run_info.json"), {}) or {},
        experiment=read_json(os.path.join(run_dir, "experiment.json"), {}) or {},
        last_activity=_last_activity(run_dir),
        last_metrics=read_metrics(results_dir(run_dir, model_.kind, "last"), model_.kind),
    )
    run.active = _run_is_active(run, procs if procs is not None else pipeline_processes())
    return run


def runs(model_: Model | None = None, procs: list | None = None) -> list[Run]:
    """Every run folder, newest first within each model (models in MODELS order)."""
    procs = procs if procs is not None else pipeline_processes()
    out: list[Run] = []
    for m in (model_,) if model_ else MODELS:
        if not os.path.isdir(m.ckpt_dir):
            continue
        ids = sorted((e.name for e in os.scandir(m.ckpt_dir)
                      if e.is_dir() and RUN_ID_RE.match(e.name)), reverse=True)
        for run_id in ids:
            run = load_run(m, run_id, procs)
            if run:
                out.append(run)
    return out


def evaluated(all_runs: list[Run] | None = None) -> list[Run]:
    return [r for r in (all_runs if all_runs is not None else runs()) if r.metrics]


def best(all_runs: list[Run], key: str = "ap", lower_is_better: bool = False) -> Run | None:
    scored = [r for r in all_runs if r.metrics and getattr(r.metrics, key) is not None]
    if not scored:
        return None
    pick = min if lower_is_better else max
    return pick(scored, key=lambda r: getattr(r.metrics, key))


# ── Pipeline processes already running ────────────────────────────────────────

@dataclass
class Proc:
    pid: int
    started: float           # epoch seconds
    args: list[str]
    workbench_job: str | None = None
    children: int = 0

    @property
    def command(self) -> str:
        return " ".join(self.args)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def describe(self) -> str:
        text = self.command
        for m in MODELS:
            if m.script in text and all(v in text for v in m.variant):
                verb = ("training" if "--train" in self.args else
                        "evaluating" if "--eval" in self.args else "running")
                return f"{m.label} {verb}"
        if "tools/train.py" in text:
            return "OWL trainer"
        for marker, label in (("baseline/main.py", "zero-shot baselines"),
                              ("analysis/main.py", "cross-model analysis"),
                              ("pipeline.cli", "data preparation"),
                              ("workbench.detect", "bird detection"),
                              ("analysis.bench", "benchmark"),
                              ("data_prep.efficiency", "benchmark")):
            if marker in text:
                return label
        return os.path.basename(self.args[1]) if len(self.args) > 1 else self.args[0]


_PIPELINE_MARKERS = tuple(dict.fromkeys(m.script for m in MODELS)) + (
    "baseline/main.py", "analysis/main.py", "pipeline.cli", "pipeline/cli.py",
    "workbench.detect", "analysis.bench_extra", "analysis.bench_train",
    "data_prep.efficiency", "tools/train.py", "keepalive_resume.sh",
)


def _boot_time() -> float:
    with open("/proc/uptime") as f:
        return time.time() - float(f.read().split()[0])


def pipeline_processes() -> list[Proc]:
    """This user's processes running a pipeline entry point, collapsed to one per job
    (DataLoader workers and interpreter hops are folded into their top-most ancestor)."""
    if not os.path.isdir("/proc"):
        return []
    uid, boot, hz = os.getuid(), _boot_time(), os.sysconf("SC_CLK_TCK")
    found: dict[int, tuple[Proc, int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if entry.stat().st_uid != uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
            if not args or not any(mk in " ".join(args) for mk in _PIPELINE_MARKERS):
                continue
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            ppid, start_ticks = int(stat[1]), int(stat[19])
            job = None
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    for item in f.read().split(b"\0"):
                        if item.startswith(b"WB_JOB_ID="):
                            job = item.split(b"=", 1)[1].decode()
            except OSError:
                pass
            found[pid] = (Proc(pid, boot + start_ticks / hz, args, job), ppid)
        except (OSError, ValueError, IndexError):
            continue

    def top(pid: int) -> int:
        seen = set()
        while found[pid][1] in found and pid not in seen:
            seen.add(pid)
            pid = found[pid][1]
        return pid

    roots: dict[int, Proc] = {}
    for pid in found:
        root = top(pid)
        roots.setdefault(root, found[root][0])
        if root != pid:
            roots[root].children += 1
    return sorted(roots.values(), key=lambda p: p.started)


def run_started(run_id: str) -> float:
    """A run id is the local time its trainer created the folder."""
    return time.mktime(time.strptime(run_id, "%Y-%m-%d_%H-%M-%S"))


def _without_init_from(args: list[str]) -> str:
    """The command minus any --init-from value: a fine-tune names its parent's checkpoint,
    and the parent is not the run being worked on."""
    kept, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--init-from":
            skip = True
            continue
        if not a.startswith("--init-from="):
            kept.append(a)
    return " ".join(kept)


def _run_is_active(run: Run, procs: list[Proc]) -> bool:
    started = run_started(run.run_id)
    for p in procs:
        text = _without_init_from(p.args)
        if run.path in text:
            return True
        # a run id alone is not enough — two models' runs can share a timestamp
        mine = run.model.script in text and all(v in text for v in run.model.variant)
        if mine and run.run_id in text:
            return True
        if (mine and "--train" in p.args and "--resume" not in p.args
                and p.started - 5 <= started <= time.time() + 60):
            return True          # a fresh --train made this folder after it started
    return False


# ── GPUs ──────────────────────────────────────────────────────────────────────

@dataclass
class GPU:
    index: int
    name: str
    used_mb: int
    total_mb: int
    util: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb

    @property
    def busy(self) -> bool:
        return self.util >= 20 or self.used_mb >= 1024

    def describe(self) -> str:
        return (f"GPU {self.index} · {self.name} · {self.util}% busy · "
                f"{self.used_mb / 1024:.1f}/{self.total_mb / 1024:.0f} GB used")


def gpus() -> list[GPU]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            found.append(GPU(int(parts[0]), parts[1], int(parts[2]), int(parts[3]),
                             int(parts[4])))
        except (ValueError, IndexError):
            continue
    return found


def suggest_gpu(found: list[GPU] | None = None) -> GPU | None:
    """The idlest GPU: least busy first, then most free memory."""
    found = gpus() if found is None else found
    if not found:
        return None
    return min(found, key=lambda g: (g.busy, g.util, -g.free_mb))


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class Collection:
    name: str
    path: str
    photos: int
    label_files: int
    has_index: bool
    link: str | None = None           # where a symlinked collection points


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def collections(root: str) -> list[Collection]:
    """Sub-folders of a data folder that hold images (what the MDC adapter reads)."""
    out = []
    if not os.path.isdir(root):
        return out
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        photos = labels = 0
        index = False
        try:
            for f in os.scandir(entry.path):
                if _is_image(f.name):
                    photos += 1
                elif f.name.endswith(".txt"):
                    labels += 1
                elif f.name == "image_info.csv":
                    index = True
        except OSError:
            continue
        if photos:
            link = os.readlink(entry.path) if entry.is_symlink() else None
            out.append(Collection(entry.name, entry.path, photos, labels, index, link))
    return out


def data_folders() -> list[str]:
    """Candidate data folders: dataset_* directories at the repository root."""
    return sorted(e.name for e in os.scandir(paths.ROOT)
                  if e.is_dir() and e.name.startswith("dataset_"))


@dataclass
class SplitStatus:
    ready: bool
    updated: float | None
    experiments: list[str] = field(default_factory=list)


def splits_status() -> SplitStatus:
    base = config.COMMITTED_CROPS_JSON_DIR
    files = [os.path.join(base, s, "coco.json") for s in ("train", "val", "test")]
    ready = all(os.path.exists(f) for f in files)
    updated = max((os.path.getmtime(f) for f in files if os.path.exists(f)), default=None)
    experiments = []
    if os.path.isdir(config.EXPERIMENTS_DIR):
        experiments = sorted(e.name for e in os.scandir(config.EXPERIMENTS_DIR)
                             if e.is_dir()
                             and os.path.exists(os.path.join(e.path, "selection.json")))
    return SplitStatus(ready, updated, experiments)


def owl_env_ready() -> bool:
    return os.path.exists(config.MDO_PYTHON)


def owl_weights(model_: Model) -> str | None:
    """The released OWL checkpoint a fine-tuning run of this variant starts from."""
    if not model_.owl:
        return None
    return config.MDO_MODELS[model_.mdo_key].get("requires_weights")
