"""
Recommended drone flight height for waterfowl counting, from camera and bird species.

    H_rec = k * S * f * N / (w * P_optimum) = k * S * C / P_optimum     metres above the birds
    H_set = H_rec - h_takeoff                                            height to enter on the controller

    S          bird size (m): side of a square with the area of the bird's bounding box
    f          focal length (mm); N image width (pixels, long side); w sensor width (mm, long side)
    C          camera factor = f * N / w, or from EXIF alone C = f35 * image_diagonal_px / 43.27
    P_optimum  optimum bird size in the photo for detection: the box size, sqrt(box area) in px, at which
               the detection rate peaks (YOLOv5: 18 px; the earlier Task 2.3 analysis: best bin 17-19 px)
    k          blur correction = P_optimum / sqrt(P_optimum^2 - b^2) = 1 / sqrt(1 - 1/kappa^2), where
               kappa = P_optimum / b ~ 2.5 in this corpus, so b is never measured in the field (k = 1.09)

Re-derive P_optimum (step3b/step4) whenever the detector is retrained.

Usage:
    python flight_altitude.py                                   # tables for the two calibrated cameras
    python flight_altitude.py --species teal --focal-mm 10.26 --sensor-mm 13.2 --width 5472 --takeoff 3
    python flight_altitude.py --species mallard_m --f35 24 --width 4000 --height 3000
    python flight_altitude.py --size-cm 45 --focal-mm 10.26 --sensor-mm 13.2 --width 5472 --pixels 16
"""
import argparse
import math

P_OPTIMUM = 18.0                  # px box size, YOLOv5
B_PX = 7.29                       # px, blur and box margin
ALTERNATIVES = (16.0, 14.0)       # px box size: higher flights, fewer birds found (92.7% and 86.7% vs 94.6%)

# S (cm): median over real birds, 27 XMP-verified Mavic 2 Pro photos of Bird_I at 14.7–15.8 m
SPECIES_CM = {
    "green-winged teal": 21.09,
    "american wigeon m": 26.48,
    "gadwall": 27.94,
    "mallard f": 27.91,
    "northern pintail f": 28.23,
    "northern pintail m": 29.36,
    "mallard m": 30.06,
}

CAMERAS = {  # focal length (mm), sensor width (mm), image width (px) — verified from EXIF in this corpus
    "DJI Mavic 2 Pro (Hasselblad L1D-20c)": (10.26, 13.2, 5472),
    "DJI Mavic Air 2 (FC3170, 12 MP)": (4.5, 6.4, 4000),
}

LEGAL_CEILING_M = 121.9  # 400 ft above ground level, FAA Part 107


def camera_factor(focal_mm=None, sensor_mm=None, width=None, f35=None, height=None) -> float:
    """C = f * N / w, or from the 35 mm-equivalent focal length and the image diagonal."""
    if focal_mm and sensor_mm and width:
        return focal_mm * width / sensor_mm
    if f35 and width and height:
        return f35 * math.hypot(width, height) / 43.27
    raise ValueError("give (focal_mm, sensor_mm, width) or (f35, width, height)")


def blur_factor(pixels: float) -> float:
    """k = P / sqrt(P^2 - b^2): a bird's box in the photo is slightly larger than the bird itself."""
    return pixels / math.sqrt(pixels ** 2 - B_PX ** 2)


def recommend(species_cm: float, C: float, pixels: float = P_OPTIMUM, takeoff_m: float = 0.0) -> dict:
    k = blur_factor(pixels)
    h = min(k * species_cm / 100 * C / pixels, LEGAL_CEILING_M)
    return dict(height_above_birds_m=round(h, 1), controller_setting_m=round(h - takeoff_m, 1),
                gsd_cm_per_px=round(h / C * 100, 2), k=round(k, 3), capped_at_legal_limit=h >= LEGAL_CEILING_M)


def _key(name: str) -> str:
    """Forgiving species lookup: 'teal', 'mallard_m', 'pintail ♂' all resolve if unambiguous."""
    words = set(name.lower().replace("_", " ").replace("-", " ").replace("♂", " m").replace("♀", " f").split())
    hits = [k for k in SPECIES_CM if words <= set(k.replace("-", " ").split())]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"species {name!r} is {'ambiguous' if hits else 'unknown'}; known: {', '.join(SPECIES_CM)} "
                   "(or pass --size-cm)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species"); ap.add_argument("--size-cm", type=float)
    ap.add_argument("--focal-mm", type=float); ap.add_argument("--sensor-mm", type=float)
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int); ap.add_argument("--f35", type=float)
    ap.add_argument("--pixels", type=float, default=P_OPTIMUM, help="box size in pixels to aim for (default: optimum)")
    ap.add_argument("--takeoff", type=float, default=0.0, help="height of the take-off point above the water (m)")
    a = ap.parse_args()

    if a.species or a.size_cm:
        s = a.size_cm if a.size_cm else SPECIES_CM[_key(a.species)]
        C = camera_factor(a.focal_mm, a.sensor_mm, a.width, a.f35, a.height)
        print(f"camera factor C = {C:,.0f}  bird size S = {s} cm  box size = {a.pixels} px  ->",
              recommend(s, C, a.pixels, a.takeoff))
        return
    for cam, (fmm, sw, w) in CAMERAS.items():
        C = camera_factor(fmm, sw, w)
        print(f"\n{cam}: camera factor C = {C:,.0f}   (H_rec in m at P_optimum {P_OPTIMUM:g} px; "
              f"higher options at {ALTERNATIVES[0]:g} / {ALTERNATIVES[1]:g} px)")
        for sp, s in SPECIES_CM.items():
            r = [recommend(s, C, p) for p in (P_OPTIMUM, *ALTERNATIVES)]
            print(f"  {sp:20s} S = {s:5.2f} cm  GSD_optimum = {r[0]['gsd_cm_per_px']:.2f} cm/px   "
                  f"H_rec = {r[0]['height_above_birds_m']:5.1f}   ({r[1]['height_above_birds_m']:5.1f} / "
                  f"{r[2]['height_above_birds_m']:5.1f})")


if __name__ == "__main__":
    main()
