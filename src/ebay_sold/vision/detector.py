"""YOLO11 card/field detector: training on auto-labelled tiles, and tiled inference.

Why YOLO11 (ultralytics) instead of the original darknet YOLOv4-tiny: it pip-installs,
trains on CPU, starts from COCO-pretrained weights and reads the plain YOLO
label format that ``vision.dataset`` writes.

Search pages are very tall (a 240-result page is ~60k px), and squeezing one
into a 1024 px network input would shrink text to nothing. ``Detector.detect``
therefore cuts tall images into overlapping horizontal tiles, the same way the
training tiles were cut, and merges the duplicates the overlaps produce: a
field seen whole in two tiles collapses to one box, and the two halves of a
card split by a tile edge are joined back into one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Union

from pydantic import BaseModel

from ..models import Box

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from PIL import Image as PILImage

    ImageInput = Union[str, Path, np.ndarray, PILImage.Image]

log = logging.getLogger(__name__)

# Augmentations that keep text legible. Flips mirror glyphs (and eBay never shows
# mirrored cards), rotation/shear/perspective do not occur in screenshots, and
# strong hue shifts would erase the colour cues (green prices and sold dates).
# Mild scaling stands in for different zoom levels / device scale factors.
TRAIN_DEFAULTS: dict[str, Any] = {
    "fliplr": 0.0,
    "flipud": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.005,
    "hsv_s": 0.3,
    "hsv_v": 0.3,
    "translate": 0.05,
    "scale": 0.25,
    "mosaic": 0.5,
    "seed": 0,
    "exist_ok": True,
}


class Detection(BaseModel):
    cls: str
    conf: float
    box: Box  # image pixels


def train(data_yaml: Path, *, model: str = "yolo11n.pt", epochs: int = 50, imgsz: int = 1024, batch: int = 8,
          device: str = "cpu", project: Path, name: str = "ebay-sold", patience: int = 15, workers: int = 2,
          **overrides: Any) -> Path:
    """Fine-tune a YOLO11 model on a ``vision.dataset`` dataset; returns the path of ``best.pt``.

    ``model`` is a pretrained checkpoint (downloaded by ultralytics when given as
    a bare release name like ``yolo11n.pt``). ``overrides`` go straight to
    ``ultralytics.YOLO.train`` and win over ``TRAIN_DEFAULTS``.
    """
    from ultralytics import YOLO

    if not str(model).startswith(("http://", "https://")) and Path(model).suffix == ".pt" and not Path(model).exists():
        # A bare name like "yolo11n.pt" would be downloaded into the current directory; keep it with the run.
        from ultralytics.utils.downloads import attempt_download_asset

        target = Path(project) / Path(model).name
        target.parent.mkdir(parents=True, exist_ok=True)
        model = attempt_download_asset(str(target)) if Path(model).name == str(model) else model

    args: dict[str, Any] = dict(TRAIN_DEFAULTS)
    # Turn mosaic off for the last third of training so the final epochs see whole, real tiles.
    args["close_mosaic"] = max(1, epochs // 3)
    args.update(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                project=str(Path(project).resolve()), name=name, patience=patience, workers=workers)
    args.update(overrides)
    yolo = YOLO(str(model))
    yolo.train(**args)
    trainer = getattr(yolo, "trainer", None)
    best = Path(trainer.best) if trainer is not None and getattr(trainer, "best", None) else None
    if best is None or not best.exists():
        best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        last = best.with_name("last.pt")
        if last.exists():
            return last
        raise FileNotFoundError(f"training finished but no weights were written under {best.parent}")
    return best


# --- image helpers ------------------------------------------------------------


def load_image(image: "ImageInput") -> "np.ndarray":
    """Return an OpenCV-style BGR ``uint8`` array.

    numpy arrays are taken as BGR (what ``cv2.imread`` returns); grayscale and
    BGRA arrays are converted. PIL images and file paths are converted from RGB.
    """
    import numpy as np

    if isinstance(image, (str, Path)):
        import cv2

        arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"cannot read image {image}")
        return arr
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] == 4:
            return np.ascontiguousarray(arr[:, :, :3])
        return arr
    # PIL image
    rgb = np.asarray(image.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def tile_windows(height: int, tile_height: int = 1280, overlap: int = 160) -> list[tuple[int, int]]:
    """``(y0, y1)`` rows for overlapping horizontal tiles covering ``height`` pixels."""
    if overlap >= tile_height:
        raise ValueError("overlap must be smaller than tile_height")
    # A little slack so an image just over one tile is not cut into a sliver.
    if height <= tile_height + overlap // 2:
        return [(0, height)]
    windows, y = [], 0
    while True:
        y1 = min(height, y + tile_height)
        windows.append((y, y1))
        if y1 >= height:
            return windows
        y += tile_height - overlap


# --- merging duplicates across tiles --------------------------------------------


def iou(a: Box, b: Box) -> float:
    inter = a.intersection(b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def containment(a: Box, b: Box) -> float:
    """Intersection over the smaller box: 1.0 when one box lies inside the other."""
    smaller = min(a.area, b.area)
    return a.intersection(b) / smaller if smaller > 0 else 0.0


def union_box(a: Box, b: Box) -> Box:
    x1, y1 = min(a.x, b.x), min(a.y, b.y)
    x2, y2 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
    return Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _same_object(a: Box, b: Box, iou_thr: float, contain_thr: float) -> bool:
    if iou(a, b) >= iou_thr:
        return True
    # Pieces of one card cut by a tile edge: they overlap almost entirely inside
    # the tile overlap, and are aligned horizontally.
    if containment(a, b) < contain_thr:
        return False
    x_overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    return x_overlap >= 0.8 * min(a.w, b.w)


def merge_detections(dets: Sequence[Detection], *, iou_thr: float = 0.5, contain_thr: float = 0.7) -> list[Detection]:
    """Class-wise merge of duplicate boxes (typically from overlapping tiles).

    Two boxes of one class are the same object when their IoU is high or one
    mostly contains the other; the merged box is their union (re-joining a card
    cut in two by a tile edge) with the higher confidence.
    """
    by_cls: dict[str, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)
    out: list[Detection] = []
    for cls, items in by_cls.items():
        kept: list[Detection] = []
        for d in sorted(items, key=lambda d: -d.conf):
            box = d.box
            # A union can newly overlap other kept boxes; repeat until stable.
            merged_any = True
            conf = d.conf
            while merged_any:
                merged_any = False
                for i, k in enumerate(kept):
                    if _same_object(k.box, box, iou_thr, contain_thr):
                        box = union_box(k.box, box)
                        conf = max(conf, k.conf)
                        kept.pop(i)
                        merged_any = True
                        break
            kept.append(Detection(cls=cls, conf=conf, box=box))
        out.extend(kept)
    return sorted(out, key=lambda d: (d.box.y, d.box.x))


# --- inference ----------------------------------------------------------------------


class Detector:
    """Run a trained card/field detector on screenshots of any height."""

    def __init__(self, weights: Path | str, *, imgsz: int = 1024, conf: float = 0.25, device: str = "cpu",
                 iou: float = 0.6) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"no detector weights at {self.weights}; build a dataset with "
                "`ebay_sold.vision.dataset.build_dataset` and train with `ebay_sold.vision.detector.train`"
            )
        self.model = YOLO(str(self.weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        names = self.model.names
        self.names: dict[int, str] = dict(names) if isinstance(names, dict) else dict(enumerate(names))

    def detect(self, image: "ImageInput", *, tile_height: int = 1280, overlap: int = 160) -> list[Detection]:
        """Detect cards and fields; boxes are in the input image's pixels."""
        arr = load_image(image)
        return self.detect_windows(arr, tile_windows(arr.shape[0], tile_height, overlap))

    def detect_windows(self, arr: "np.ndarray", windows: Sequence[tuple[int, int]], *,
                       batch: int = 4) -> list[Detection]:
        """Detect on the given horizontal strips ``(y0, y1)`` of a BGR array and merge the overlaps."""
        dets: list[Detection] = []
        crops = [(y0, arr[y0:y1]) for y0, y1 in windows if y1 > y0]
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            results = self.model.predict([c for _, c in chunk], imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                         device=self.device, verbose=False)
            for (y0, crop), res in zip(chunk, results):
                if res.boxes is None:
                    continue
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, classes):
                    dets.append(Detection(cls=self.names.get(int(k), str(k)), conf=float(c),
                                          box=Box(x=float(x1), y=float(y1) + y0, w=float(x2 - x1), h=float(y2 - y1))))
        return merge_detections(dets) if len(crops) > 1 else sorted(dets, key=lambda d: (d.box.y, d.box.x))
