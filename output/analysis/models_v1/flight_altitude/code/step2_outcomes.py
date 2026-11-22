"""
Step 2 — per-bird detection outcomes on the test split, for every model in models_v1.docx.

Re-uses the benchmark's own matching (analysis.strata: score-ordered greedy nearest neighbour,
tau = 40 px, each model at its global operating threshold t*). Adds, per ground-truth bird:
dataset, source image, recorded altitude, sqrt(box area), whether the box was clipped by the
crop edge, its source-image box, species (Bird_H/Bird_I only), and source resolution/camera.

Output: $FA_WORK/outcomes.pkl
"""
import os

import numpy as np
import pandas as pd
from PIL import Image

from common import DATASET, REPO, WORK, class_file, find_image, image_info, read_class_boxes
from analysis import manifest, metadata, predict, registry, rescore, strata


def main() -> None:
    specs = [(registry.resolve(e.key), e.run_id) for e in manifest.load(os.path.join(REPO, "models_v1.docx"))]
    gt = strata.load_ground_truth()
    metas = metadata.load(predict.TEST_JSON)

    cx, cy, w, h = gt.gt_xy[:, 0], gt.gt_xy[:, 1], gt.gt_w, gt.gt_h
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    m = [metas[int(i)] for i in gt.gt_image_id]
    col = np.array([k.col for k in m])
    row = np.array([k.row for k in m])
    df = pd.DataFrame(dict(
        image_id=gt.gt_image_id, dataset=[k.dataset for k in m], stem=[k.stem for k in m],
        alt=[k.altitude_m for k in m], alt_nom=[k.altitude_nominal for k in m],
        size=gt.gt_size, w=w, h=h,
        # a box touching the 512 px crop border was cut by the tiling: its size is not the bird's
        clipped=(x1 <= 0.5) | (y1 <= 0.5) | (x2 >= 511.5) | (y2 >= 511.5),
        sx1=np.rint(x1 + col).astype(int), sy1=np.rint(y1 + row).astype(int),
        sx2=np.rint(x2 + col).astype(int), sy2=np.rint(y2 + row).astype(int),
    ))

    for spec, run_id in specs:
        pm = rescore.ensure(spec, run_id)
        cache = predict.load(predict.cache_path(spec, run_id))
        out = strata.outcomes_at_t_star(gt, cache, pm["counting"]["t_star"])
        df["hit_" + spec.key] = out.gt_matched
        print(f"{spec.label:14s} t*={pm['counting']['t_star']:.3f}  recall={out.gt_matched.mean() * 100:.2f}%")

    res, cam = {}, {}
    for ds, stem in df[["dataset", "stem"]].drop_duplicates().itertuples(index=False):
        p = find_image(ds, stem)
        im = Image.open(p)
        res[(ds, stem)] = im.size[0]
        cam[(ds, stem)] = (im.getexif().get(0x0110) or "") if p.lower().endswith((".jpg", ".jpeg")) else ""
    df["W"] = [res[k] for k in zip(df.dataset, df.stem)]
    df["exif_cam"] = [cam[k] for k in zip(df.dataset, df.stem)]

    # species: nearest labelled source box within 3 px of the (unclipped) test box centre
    df["species"] = ""
    for ds in ("Bird_H", "Bird_I"):
        rows = {os.path.splitext(r["image_name"])[0]: r for r in image_info(ds)}
        for stem, g in df[(df.dataset == ds) & ~df.clipped].groupby("stem"):
            cf = class_file(ds, rows.get(stem, {"image_name": stem + ".jpg"}))
            boxes = read_class_boxes(cf) if cf else []
            if not boxes:
                continue
            B = np.array([b[1:] for b in boxes], float)
            centres = np.column_stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2])
            for idx, r in g.iterrows():
                d = np.linalg.norm(centres - [(r.sx1 + r.sx2) / 2, (r.sy1 + r.sy2) / 2], axis=1)
                j = int(d.argmin())
                if d[j] <= 3:
                    df.at[idx, "species"] = boxes[j][0]

    df.to_pickle(os.path.join(WORK, "outcomes.pkl"))
    print(f"{len(df):,} test birds, {int(df.clipped.sum()):,} clipped, {int((df.species != '').sum()):,} with species")


if __name__ == "__main__":
    main()
