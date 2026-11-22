"""
Step 3c — is the optimum pixel size set by the image blur, or by the sizes the detector was trained on?

Re-runs YOLOv5 on the same test crops with extra Gaussian blur and refits the detection rate against the
(unchanged) annotated box size. If blur sets the requirement, the curve shifts towards larger boxes; if the
training sizes do, the curve drops but stays put. Needs a GPU; about 10 minutes for 4 blur levels.

Output: results/blur_experiment.csv
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from common import RESULTS, REPO, WORK, fit_curve
from analysis import manifest, predict, registry, rescore, strata
from data_prep import config
import cv2
from ultralytics import YOLO

SIGMAS = (0.0, 1.0, 2.0, 3.0)
N_CROPS = 4000

gt = strata.load_ground_truth()
out = pd.read_pickle(os.path.join(WORK, "outcomes.pkl"))
spec = registry.resolve("yolov5")
run_id = [e.run_id for e in manifest.load(os.path.join(REPO, "models_v1.docx")) if registry.resolve(e.key).key == "yolov5"][0]
t_star = rescore.ensure(spec, run_id)["counting"]["t_star"]
model = YOLO(predict.checkpoint_for(spec, run_id))

coco = json.load(open(predict.TEST_JSON))
path_of = {im["id"]: os.path.join(config.CROPS_IMG_DIR, im["file_name"]) for im in coco["images"]}
with_birds = out[~out.clipped].image_id.unique()
pick = np.random.default_rng(0).choice(with_birds, size=min(N_CROPS, len(with_birds)), replace=False)
keep = out[~out.clipped & out.image_id.isin(pick)].copy()
print(f"{len(pick)} crops, {len(keep)} uncut birds, t* = {t_star:.3f}", flush=True)

gt_idx = {int(i): np.where(gt.gt_image_id == i)[0] for i in pick}
rows = []
for sigma in SIGMAS:
    hits = {}
    for n, iid in enumerate(pick):
        img = cv2.imread(path_of[int(iid)])
        if sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        r = model.predict(img, conf=config.CONF_THRESHOLD, iou=config.YOLOV5_NMS_IOU,
                          max_det=config.YOLOV5_MAX_PRED, imgsz=config.CROP_SIZE, verbose=False)[0]
        b = r.boxes
        xy, sc = b.xywh.cpu().numpy(), b.conf.cpu().numpy()
        k = sc >= t_star
        xy, sc = xy[k][np.argsort(-sc[k])], np.sort(sc[k])[::-1]
        g = gt_idx[int(iid)]
        _, matched, _ = strata._match_one(gt.gt_xy[g], xy[:, :2], strata.TAU)
        for j, m in zip(g, matched):
            hits[j] = bool(m)
        if n % 1000 == 999:
            print(f"  sigma {sigma}: {n+1} crops", flush=True)
    d = keep.copy()
    d["hit"] = [hits.get(i, False) for i in d.index]
    c = fit_curve(d, "hit")
    grid = np.arange(8, 40.01, 0.1)
    r = c(grid) * 100
    peak_i = int(r.argmax())
    knee = grid[np.argmax(r >= 0.95 * r[peak_i])]
    rows.append(dict(sigma=sigma, mean_recall=d.hit.mean() * 100, peak_px=grid[peak_i], peak_rate=r[peak_i],
                     knee95_px=knee, R12=c(12) * 100, R15=c(15) * 100, R18=c(18) * 100, R25=c(25) * 100))
    print(pd.DataFrame(rows[-1:]).round(2).to_string(index=False), flush=True)
t = pd.DataFrame(rows)
t.to_csv(os.path.join(RESULTS, "blur_experiment.csv"), index=False)
print("\n", t.round(2).to_string(index=False))
