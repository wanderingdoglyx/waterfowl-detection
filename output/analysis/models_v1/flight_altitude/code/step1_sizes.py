"""
Step 1 — measure every bird in the project's dataset on its source photograph, and read camera metadata.

The boxes are the ones the benchmark itself uses: the COCO crop annotations in
output/crops_json/{train,val,test}/coco.json, mapped back to source-image pixels by
`coco_boxes` (see that module for why, and for the check that the mapping is exact).  Each
bird therefore carries the split it was scored in, whether any single 512 px crop held it
whole, and whether the photograph's own edge cut it — the last of which is the difference
between a bird's size and the size of what was annotated.

Species labels are the one thing the COCO files do not hold: they live in the *_class.txt
sidecars of Bird_H and Bird_I, which carry the same boxes with a name attached.  They are
joined onto the COCO boxes by overlap (best IoU >= 0.5), so a species row is a bird in the
dataset, never a label the crops never saw.

Outputs
  $FA_WORK/birds.csv          one row per bird (all ten datasets): dataset, image, resolution,
                               EXIF camera, recorded altitude, split, w, h, whole, edge
  $FA_WORK/species.csv        one row per species-labelled bird (Bird_H decoys, Bird_I real birds)
  results/camera_exif.csv     every image that still carries EXIF: camera, focal length, 35 mm
                               equivalent, DJI XMP relative altitude and gimbal pitch
"""
import csv
import os

import numpy as np
from PIL import Image

import coco_boxes
from common import DATASET, RESULTS, WORK, class_file, dji_xmp, find_image, image_info, read_class_boxes

EXIF_MODEL, EXIF_SUB = 0x0110, 0x8769
FOCAL, FOCAL35 = 0x920A, 0xA405
IOU_MIN = 0.5           # a species sidecar box and its COCO box are the same box 99.7% of the time


def match_species(boxes: list, labelled: list) -> dict[int, str]:
    """{index into `boxes` -> species}, by best overlap with the *_class.txt boxes."""
    if not boxes or not labelled:
        return {}
    A = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], float)
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    out = {}
    for sp, x1, y1, x2, y2 in labelled:
        ix1 = np.maximum(A[:, 0], x1); iy1 = np.maximum(A[:, 1], y1)
        ix2 = np.minimum(A[:, 2], x2); iy2 = np.minimum(A[:, 3], y2)
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (area_a + (x2 - x1) * (y2 - y1) - inter)
        j = int(iou.argmax())
        if iou[j] >= IOU_MIN and j not in out:
            out[j] = sp
    return out


def main() -> None:
    src = coco_boxes.load()
    birds, species, cams = [], [], []
    no_crops = 0
    for ds in sorted(d for d in os.listdir(DATASET) if d.startswith("Bird_")):
        for r in image_info(ds):
            stem = os.path.splitext(r["image_name"])[0]
            image = src.get((ds, stem))
            img = find_image(ds, stem)
            if image is None or img is None:
                no_crops += 1
                continue
            im = Image.open(img)
            W, H = im.size
            exif = im.getexif()
            model = exif.get(EXIF_MODEL) or ""
            alt = int(float(r["height"]))
            if model:
                sub = exif.get_ifd(EXIF_SUB)
                cams.append(dict(dataset=ds, image=r["image_name"], W=W, H=H, camera=model,
                                 focal_mm=float(sub.get(FOCAL, 0) or 0), f35_mm=sub.get(FOCAL35),
                                 csv_height=alt, xmp_rel_alt=dji_xmp(img, "RelativeAltitude"),
                                 gimbal_pitch=dji_xmp(img, "GimbalPitchDegree"),
                                 exposure_s=float(sub.get(0x829A, 0) or 0)))
            common = dict(ds=ds, image=r["image_name"], camera=model, alt=alt, split=image.split)
            for b in image.boxes:
                birds.append(dict(**common, W=W, H=H, w=b.w, h=b.h, whole=b.whole, edge=b.edge))
            if ds in ("Bird_H", "Bird_I"):
                cf = class_file(ds, r)
                labelled = read_class_boxes(cf) if cf else []
                for j, sp in match_species(image.boxes, labelled).items():
                    b = image.boxes[j]
                    species.append(dict(ds=ds, image=r["image_name"], res=f"{W}x{H}", cam=model, alt=alt,
                                        split=image.split, hab=r.get("background_info", ""), sp=sp,
                                        w=b.w, h=b.h, whole=b.whole, edge=b.edge))

    for name, rows, folder in (("birds.csv", birds, WORK), ("species.csv", species, WORK),
                               ("camera_exif.csv", cams, RESULTS)):
        with open(os.path.join(folder, name), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"{name}: {len(rows):,} rows -> {folder}")
    print(f"{sum(b['edge'] for b in birds):,} birds cut by the photograph's edge, "
          f"{sum(not b['whole'] for b in birds):,} split across crops, "
          f"{no_crops} images in image_info.csv with no crops in the dataset")


if __name__ == "__main__":
    main()
