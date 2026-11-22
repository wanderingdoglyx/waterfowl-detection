"""
OWL detection on survey photographs — RUNS UNDER THE MegaDetector-Overhead .venv
(Python 3.11), never under the `waterfowl` env.  Started by workbench/detect.py.

The sibling of megadetector_overhead/_eval_owl.py, and deliberately identical to it where
the result depends on it: the same model construction and checkpoint loading, the same
normalisation (the checkpoint's own mean/std through albumentations' Normalize, then a
plain HWC -> CHW transpose, which is all ToTensor does to a float array), the same
detection-branch stitcher and LMDS settings, points scaled back by down_ratio.  What
differs is the input: tiles cut from whole photographs at the origins the caller passes,
with no ground truth needed.

    --work  JSON list of {"photo": path, "key": id, "xs": [...], "ys": [...]}
    --out   JSONL, one line per photograph:
            {"key": id, "tiles": [[tile_x, tile_y, [[x, y, score], ...]], ...], "seconds": s}
            (x, y in tile pixels).  Photographs already in --out are skipped.
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

_REPO = os.environ.get("MDO_REPO_DIR")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import albumentations as A  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import animaloc.models as animaloc_models  # noqa: E402
from animaloc.eval.lmds import HerdNet_Detection_Branch_LMDS  # noqa: E402
from animaloc.eval.stitchers import HerdNet_Detection_Branch_Stitcher  # noqa: E402
from animaloc.models.utils import LossWrapper, load_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OWL detection on survey photographs (mdo env)")
    p.add_argument("--pth", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-name", default="OWLC")
    p.add_argument("--model-kwargs", default="")
    p.add_argument("--down-ratio", type=int, default=2)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=160)
    p.add_argument("--adapt-ts", type=float, default=0.3)
    p.add_argument("--kernel", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.pth, map_location="cpu", weights_only=False)
    mean = ckpt.get("mean", IMAGENET_MEAN) if isinstance(ckpt, dict) else IMAGENET_MEAN
    std = ckpt.get("std", IMAGENET_STD) if isinstance(ckpt, dict) else IMAGENET_STD
    del ckpt
    normalise = A.Normalize(mean=mean, std=std)

    model_cls = animaloc_models.__dict__.get(args.model_name)
    if model_cls is None:
        raise KeyError(f"Model '{args.model_name}' not found in animaloc.models")
    model = model_cls(**json.loads(args.model_kwargs)) if args.model_kwargs else model_cls()
    model = load_model(LossWrapper(model, []), args.pth).to(device)
    model.eval()

    stitcher = HerdNet_Detection_Branch_Stitcher(
        model=model, size=(args.img_size, args.img_size), overlap=args.overlap,
        down_ratio=args.down_ratio, up=False, reduction="mean", device_name=device,
    )
    lmds = HerdNet_Detection_Branch_LMDS(up=False, kernel_size=(args.kernel, args.kernel),
                                         adapt_ts=args.adapt_ts)

    with open(args.work) as f:
        work = json.load(f)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.endswith("\n"):
                    done.add(json.loads(line)["key"])

    size = args.img_size
    todo = [w for w in work if w["key"] not in done]
    print(f"[owl-detect] {len(todo)} photographs on {device}", flush=True)
    for n, item in enumerate(todo, 1):
        start = time.time()
        with Image.open(item["photo"]) as im:
            img = im.convert("RGB")
        tiles = []
        for ty in item["ys"]:
            for tx in item["xs"]:
                arr = np.asarray(img.crop((tx, ty, tx + size, ty + size)))
                arr = normalise(image=arr)["image"]
                tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()
                with torch.no_grad():
                    heatmap = stitcher(tensor)
                _, locs, _, scores = lmds(heatmap)
                pts = [[round(float(w) * args.down_ratio, 1), round(float(h) * args.down_ratio, 1),
                        round(float(s), 4)] for (h, w), s in zip(locs[0], scores[0])]
                tiles.append([tx, ty, pts])
        seconds = time.time() - start
        with open(args.out, "a") as f:
            f.write(json.dumps({"key": item["key"], "tiles": tiles, "seconds": seconds},
                               separators=(",", ":")) + "\n")
        print(f"[owl-detect] {n}/{len(todo)} {os.path.basename(item['photo'])} "
              f"{sum(len(t[2]) for t in tiles)} points · {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
