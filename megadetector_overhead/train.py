"""
Train / fine-tune OWL-C on the waterfowl crops by driving the MegaDetector-Overhead
repo's own Hydra entry point (tools/train.py) under its .venv (Python 3.11).

Rather than reimplement animaloc's training loop, we generate a self-contained Hydra
config from data_prep.config and hand it to the repo's tested trainer.  The pretrained
OWL-C checkpoint is used as the fine-tuning start point (`model.load_from`); OWL-C's
head is already single-class, so it loads whole (no partial load needed).

The trainer keeps the best checkpoint by validation F1 (point metric) and, together with
auto-lr plateau decay, plays the role Faster R-CNN's early-stopping hook plays for the
other models.  Everything lands in the timestamped run dir under the selected model's folder
(output/checkpoints/mdo_owl_c / mdo_owl_t / mdo_owl_d).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import yaml

from data_prep import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_config(train_imgs: str, train_csv: str,
                       val_imgs: str, val_csv: str,
                       run_tag: str, model_spec: dict) -> dict:
    """Assemble the Hydra `train:` config for a fine-tuning run (concrete values, no
    interpolations, so it composes on its own via --config-name).

    `model_spec` is one entry of config.MDO_MODELS — it supplies the animaloc class name,
    the fine-tuning start checkpoint (`load_from`), and the architecture-specific kwargs
    (DLA args for OWL-C/T, DINOv3 args for OWL-D)."""
    dr = config.MDO_DOWN_RATIO
    load_from = model_spec.get("load_from")
    if load_from and not os.path.exists(load_from):
        load_from = None
    return {
        "train": {
            "wandb_project": "waterfowl-owl",
            "wandb_entity": "anonymous",
            "wandb_run": run_tag,
            "seed": config.RANDOM_SEED,
            "device_name": "cuda",
            "model": {
                "name": model_spec["name"],
                "from_torchvision": False,
                "load_from": load_from,
                "partial_load": False,
                "resume_from": None,
                "kwargs": dict(model_spec["kwargs"]),
                "freeze": None,
            },
            "losses": {
                "FocalLoss": {
                    "print_name": "focal_loss",
                    "from_torch": False,
                    "output_idx": 0,
                    "target_idx": 0,
                    "lambda_const": 1.0,
                    "kwargs": {"reduction": "mean", "normalize": False},
                }
            },
            "datasets": {
                "img_size": [config.CROP_SIZE, config.CROP_SIZE],
                "anno_type": "point",
                "collate_fn": None,
                "class_def": {1: "bird"},
                "train": {
                    "name": "FolderDataset",
                    "csv_file": train_csv,
                    "root_dir": train_imgs,
                    "sampler": None,
                    "albu_transforms": {"Normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}},
                    "end_transforms": {"FIDT": {"add_bg": False, "down_ratio": dr}},
                },
                "validate": {
                    "name": "FolderDataset",
                    "csv_file": val_csv,
                    "root_dir": val_imgs,
                    "albu_transforms": {"Normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}},
                    "end_transforms": {"DownSample": {"down_ratio": dr, "anno_type": "point"}},
                },
            },
            "training_settings": {
                "valid_freq": 1,
                "trainer": "Trainer",
                "print_freq": 50,
                "batch_size": model_spec.get("batch_size", config.MDO_BATCH_SIZE),
                "num_workers": config.NUM_WORKERS,
                "pin_memory": True,
                "optimizer": "adam",
                "lr": config.MDO_LR,
                "weight_decay": 0.0005,
                "auto_lr": {
                    "mode": "max", "patience": 10, "threshold": 0.0001,
                    "threshold_mode": "rel", "cooldown": 10, "min_lr": 1.0e-05,
                    "verbose": True,
                },
                "warmup_iters": 100,
                "vizual_fn": None,
                "epochs": config.MDO_EPOCHS,
                "evaluator": {
                    "name": "HerdNet_Detection_Branch_Evaluator",
                    "threshold": config.MDO_POINT_RADIUS,
                    "select_mode": "max",
                    "validate_on": "f1_score",
                    "kwargs": {
                        "print_freq": 50,
                        "lmds_kwargs": {
                            "kernel_size": [config.MDO_LMDS_KERNEL, config.MDO_LMDS_KERNEL],
                            "adapt_ts": config.MDO_LMDS_ADAPT_TS,
                        },
                    },
                },
                "stitcher": {
                    "name": "HerdNet_Detection_Branch_Stitcher",
                    "kwargs": {
                        "overlap": config.MDO_STITCH_OVERLAP,
                        "down_ratio": dr, "up": False, "reduction": "mean",
                    },
                },
            },
        }
    }


def _find_best_checkpoint(run_dir: str) -> str | None:
    """Locate best_model.pth the trainer wrote (CWD is the run dir via hydra.job.chdir)."""
    for root, _dirs, files in os.walk(run_dir):
        if "best_model.pth" in files:
            return os.path.join(root, "best_model.pth")
    # fall back to latest_model.pth if best was never beaten
    for root, _dirs, files in os.walk(run_dir):
        if "latest_model.pth" in files:
            return os.path.join(root, "latest_model.pth")
    return None


def train(run_dir: str, epochs: int | None = None, model_key: str | None = None) -> str:
    """
    Fine-tune the selected OWL variant, writing the config + checkpoints under run_dir.
    Returns the path to the best checkpoint (also copied to run_dir/weights/best.pth for a
    stable handle).  `model_key` selects the config.MDO_MODELS entry (defaults to MDO_MODEL).
    """
    from megadetector_overhead.dataset import split_paths

    model_spec = config.mdo_model_spec(model_key)

    train_imgs, train_csv = split_paths("train")
    val_imgs, val_csv = split_paths("val")

    cfg = build_train_config(train_imgs, train_csv, val_imgs, val_csv,
                             run_tag=os.path.basename(run_dir), model_spec=model_spec)
    if epochs is not None:
        cfg["train"]["training_settings"]["epochs"] = epochs

    cfg_path = os.path.join(run_dir, "mdo_train_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MDO_REPO_DIR"] = config.MDO_REPO_DIR
    # OWL-D's DINOv3 backbone resolves its package/weights via DINOV3_ROOT; point it at the
    # vendored copy so the subprocess finds it regardless of CWD (harmless for OWL-C/T).
    env["DINOV3_ROOT"] = config.MDO_DINOV3_ROOT
    # /dev/shm can be tiny in containers; route DataLoader IPC through the filesystem.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # animaloc rebuilds an albumentations Compose per __getitem__; with only a Normalize
    # transform it warns "no transform to process keypoints" once per crop, flooding the
    # log. Likewise sklearn >= 1.9 warns on every 1x1 confusion matrix — which is EVERY
    # image for our single-class (bird-only) setup, even though animaloc correctly passes
    # labels=[1] — flooding validation output. Silence just those two modules' UserWarnings
    # in the trainer and its DataLoader workers (PYTHONWARNINGS is read at each interpreter
    # startup); the computed metrics are unaffected.
    env["PYTHONWARNINGS"] = ",".join(filter(None, [
        env.get("PYTHONWARNINGS"),
        "ignore::UserWarning:albumentations.core.composition",
        "ignore::UserWarning:sklearn.metrics._classification",
    ]))

    cmd = [
        config.MDO_PYTHON, os.path.join(config.MDO_REPO_DIR, "tools", "train.py"),
        "--config-dir", os.path.abspath(run_dir),
        "--config-name", "mdo_train_config",
        f"hydra.run.dir={os.path.abspath(run_dir)}",
        "hydra.output_subdir=.hydra",
        "hydra.job.chdir=True",
    ]
    print(f"[MDO] launching trainer:\n  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=config.MDO_REPO_DIR, env=env, check=True)

    best = _find_best_checkpoint(run_dir)
    if best is None:
        raise RuntimeError(f"Training finished but no checkpoint found under {run_dir}")

    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    stable = os.path.join(weights_dir, "best.pth")
    shutil.copy2(best, stable)
    return stable


if __name__ == "__main__":
    # Direct invocation: megadetector_overhead/train.py <run_dir> [epochs] [model_key]
    _run_dir = sys.argv[1]
    _epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    _model_key = sys.argv[3] if len(sys.argv) > 3 else None
    print(train(_run_dir, epochs=_epochs, model_key=_model_key))
