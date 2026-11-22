"""
Camera optics, and where they come from.

A detector sees pixels, so what matters is not the flight height but the ground sample
distance: altitude x pixel pitch / focal length.  Two surveys are comparable when their
ground sample distances are, whatever heights they were flown at.

EXIF is preferred and read per photograph when available.  Where it is absent, a
collection-level default may be declared — and the record says which it was
(`Camera.source`), so an analysis can exclude assumed optics if it wants to.

The MDC constants below were measured from the EXIF/XMP of this corpus during the
flight-height study: the Mavic 2 Pro's Hasselblad L1D-20c for Bird_G-J, with a handful of
Bird_I frames from a Mavic Air 2 (FC3170).  EXIF survives for only 9% of the corpus, so
most Missouri photographs carry no optics at all and report a ground sample distance of
None rather than a guessed one.
"""

from __future__ import annotations

import os

from pipeline.records import Camera

# Known cameras, keyed by the EXIF model string.
KNOWN: dict[str, Camera] = {
    # DJI Mavic 2 Pro, Hasselblad L1D-20c: 10.26 mm lens, 1" sensor 13.2 mm wide,
    # 5472 px across.  0.235 cm/px per 10 m of height.
    "L1D-20c": Camera("L1D-20c", 10.26, 13.2, 5472, source="exif"),
    # DJI Mavic Air 2 (FC3170): 4.5 mm lens, 1/2" sensor, 4000 px across.
    "FC3170": Camera("FC3170", 4.5, 6.4, 4000, source="exif"),
}


def from_exif(image_path: str) -> Camera:
    """Optics read from the image itself, or an empty Camera when EXIF is absent.

    Only the header is read (PIL does not decode the pixels), so this is cheap enough to
    run over a whole corpus.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            exif = img.getexif()
            width = img.width
    except Exception:
        return Camera()

    if not exif:
        return Camera()

    model = exif.get(272)                       # Model
    focal = exif.get(37386)                     # FocalLength
    model = str(model).strip() if model else None

    known = KNOWN.get(model)
    if known is not None:
        # Trust the measured constants for a camera we know, but keep the frame's own
        # width: the L1D-20c also shoots a 16:9 mode at the same pixel pitch.
        return Camera(known.model, known.focal_length_mm, known.sensor_width_mm,
                      width or known.image_width_px, source="exif")

    if model and focal:
        # An unknown camera still gives a focal length; without the sensor width the
        # ground sample distance stays None rather than being invented.
        return Camera(model, float(focal), None, width, source="exif")
    return Camera(model, None, None, width, source="exif")


def declared(model: str, source: str = "collection") -> Camera:
    """A collection-level default for a known camera, marked as not read from the frame."""
    c = KNOWN[model]
    return Camera(c.model, c.focal_length_mm, c.sensor_width_mm, c.image_width_px,
                  source=source)


def gsd_cm_px(altitude_m: float, camera: Camera) -> float | None:
    """Centimetres of ground per pixel; None unless both the height and the optics exist."""
    if altitude_m is None or not camera.complete:
        return None
    pitch_mm = camera.sensor_width_mm / camera.image_width_px
    return float(altitude_m * 1000.0 * pitch_mm / camera.focal_length_mm / 10.0)


def height_for_pixel_size(bird_size_m: float, pixels: float, camera: Camera,
                          blur_correction: float = 1.09) -> float | None:
    """The flight height at which a bird of `bird_size_m` covers `pixels` pixels.

    The flight-height study's equation, kept here so a new programme can be advised on
    altitudes before it flies: H = k x S x f x N / (w x P).  `pixels` defaults elsewhere
    to 18, the size at which detection rate peaks in this corpus.
    """
    if not camera.complete or not pixels:
        return None
    return float(blur_correction * bird_size_m * camera.focal_length_mm
                 * camera.image_width_px / (camera.sensor_width_mm * pixels))


def image_size(path: str) -> tuple[int | None, int | None]:
    """(width, height) in pixels from the file header, or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None
