#!/usr/bin/env python
"""
Fetch the gated DINOv3 ViT-S/16 backbone weights for training an OWL-D variant FROM SCRATCH.

NOTE: the default OWL-D path here (OWLD_H, `--model OWLD_H`) fine-tunes from the released
OWL-D.pth checkpoint, which already bundles its frozen ViT-H+ backbone — so it does NOT need
this download.  This script is only for the advanced case of training an OWLD_S/B/L from a
Meta DINOv3 backbone.  Get the released checkpoints instead with:
    ./megadetector_overhead/main.py --fetch-weights --model OWLD_H

RUNS UNDER THE MegaDetector-Overhead .venv (Python 3.11) — that is the only interpreter
here with `huggingface_hub` installed.  Invoke it directly:

    third_party/MegaDetector-Overhead/.venv/bin/python \
        megadetector_overhead/fetch_dinov3_weights.py

The weights are license-gated on HuggingFace, so two things must be true first:
  1. You have accepted the model license at
       https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
  2. This machine is authenticated, via either
       - `third_party/MegaDetector-Overhead/.venv/bin/hf auth login`  (stores a token file), or
       - an HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable, or
       - the --token flag (least preferred; ends up in shell history).

On success the file lands at config.MDO_DINOV3_WEIGHTS, exactly where the OWL-D registry
entry points, so `--train --model OWLD_S` then just works.  Re-running is a no-op if the
file is already present and non-empty (use --force to re-download).
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data_prep.config as config


def _resolve_token(cli_token: str | None) -> str | None:
    """Token precedence: explicit flag > env > stored `hf auth login` file."""
    if cli_token:
        return cli_token
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        from huggingface_hub import get_token  # reads ~/.cache/huggingface/token
        return get_token()
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Download gated DINOv3 ViT-S/16 weights for OWL-D")
    p.add_argument("--token", default=None,
                   help="HF token (else HF_TOKEN env, else stored `hf auth login`)")
    p.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = p.parse_args()

    dest = config.MDO_DINOV3_WEIGHTS
    weights_dir = os.path.dirname(dest)
    os.makedirs(weights_dir, exist_ok=True)

    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not args.force:
        print(f"[fetch] Already present ({os.path.getsize(dest) / 1e6:.0f} MB): {dest}")
        print("[fetch] Nothing to do (use --force to re-download).")
        return 0

    token = _resolve_token(args.token)
    if not token:
        print("[fetch] No HuggingFace token found. Authenticate one of these ways, then re-run:")
        print(f"    {config.MDO_PYTHON.replace('/python', '/hf')} auth login")
        print("    # or:  export HF_TOKEN=hf_xxx")
        print("Also make sure you have accepted the license at:")
        print(f"    https://huggingface.co/{config.MDO_DINOV3_HF_REPO}")
        return 2

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        from huggingface_hub.utils import EntryNotFoundError
    except Exception as e:  # pragma: no cover
        print(f"[fetch] huggingface_hub unavailable in this interpreter: {e}")
        print(f"[fetch] Run with the OWL .venv python: {config.MDO_PYTHON}")
        return 3

    repo, fname = config.MDO_DINOV3_HF_REPO, config.MDO_DINOV3_HF_FILE
    print(f"[fetch] Downloading {fname}\n        from {repo}\n        -> {dest}")
    try:
        # local_dir places the file at weights_dir/<fname> (no cache-symlink indirection).
        path = hf_hub_download(
            repo_id=repo, filename=fname, token=token,
            local_dir=weights_dir,
        )
    except GatedRepoError:
        print("[fetch] 401 GatedRepoError — the token is valid but you have NOT been granted")
        print("        access to this gated repo. Accept the license (same HF account as the")
        print(f"        token) at:  https://huggingface.co/{repo}")
        return 4
    except RepositoryNotFoundError:
        print(f"[fetch] Repo not found (or token lacks access): {repo}")
        return 5
    except EntryNotFoundError:
        print(f"[fetch] File '{fname}' not found in {repo}. Its stored name may differ; list the")
        print(f"        repo files at https://huggingface.co/{repo}/tree/main and update")
        print("        config.MDO_DINOV3_HF_FILE accordingly.")
        return 6
    except Exception as e:
        msg = str(e)
        hint = ""
        if "401" in msg or "Unauthorized" in msg:
            hint = "  (token invalid/expired — re-run `hf auth login`)"
        print(f"[fetch] Download failed: {type(e).__name__}: {msg[:200]}{hint}")
        return 7

    # hf_hub_download with local_dir returns the final path; normalise to our dest name.
    if os.path.abspath(path) != os.path.abspath(dest) and os.path.exists(path):
        os.replace(path, dest)

    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size == 0:
        print(f"[fetch] Downloaded file is empty: {dest}")
        return 8

    # Light sanity check: it should load as a torch state dict.
    try:
        import torch
        obj = torch.load(dest, map_location="cpu", weights_only=False)
        n = len(obj) if hasattr(obj, "__len__") else "?"
        print(f"[fetch] OK — {size / 1e6:.0f} MB, loads as a checkpoint ({n} top-level keys).")
    except Exception as e:
        print(f"[fetch] Saved {size / 1e6:.0f} MB to {dest}, but torch.load warned: "
              f"{type(e).__name__}: {str(e)[:120]}")

    print(f"[fetch] Done. OWL-D is ready:  ./megadetector_overhead/main.py --train --model OWLD_S")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
