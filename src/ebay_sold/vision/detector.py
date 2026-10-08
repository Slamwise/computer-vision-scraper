"""YOLO11 card/field detector: training on auto-labelled tiles, and tiled inference.

Why YOLO11 (ultralytics) instead of the original darknet YOLOv4-tiny: it pip-installs,
trains on CPU, starts from COCO-pretrained weights and reads the plain YOLO
label format that ``vision.dataset`` writes.

Search pages are very tall (a 240-result page is ~60k px), and squeezing one
into a 1024 px network input would shrink text to nothing. ``Detector.detect``
therefore cuts tall images into overlapping horizontal windows, the same way
the training tiles were cut, and merges the duplicates the overlaps produce: a
field seen whole in two windows collapses to one box, and the two pieces of a
card cut by a window edge are joined back into one (each detection records
whether it touches an inner window edge, so pieces are re-joined however
little of them overlaps).

A detector only works near the text size it was trained on. Screenshots come
at device scale factor 1 or 2, zoomed, as single cards or 2560 px wide, so
``detect`` first measures the scale: eBay draws prices, sold dates and
shipping lines at fixed CSS sizes, and their detected height gives image
pixels per CSS pixel. Windows are then cut at the training geometry in CSS
pixels, and the network input size is chosen per image so text reaches the
network at about the training scale.
"""

from __future__ import annotations

import logging
import math
import statistics
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


# Training-tile geometry (CSS px), see vision.dataset.build_dataset.
TILE_CSS = 1280
OVERLAP_CSS = 160
# Height (CSS px) of single-line fields as eBay draws them: the median label height
# in the training set; 90 % of labels are within a pixel of it.
REFERENCE_TEXT_HEIGHT: dict[str, float] = {"price": 24.0, "sold_date": 16.0, "shipping": 16.0}
# Network pixels per CSS pixel the model can handle, relative to the training
# scale (train imgsz / TILE_CSS). Training tiles were 1024-1440 px wide with
# +-25 % scale augmentation; outside this band boxes degrade (measured: shipping
# rows vanish at 0.5x, titles split in two at 1.35x).
NET_SCALE_BAND = (0.62, 1.19)
MAX_IMGSZ = 2560


class Detection(BaseModel):
    cls: str
    conf: float
    box: Box  # image pixels
    # The box reaches an inner edge of the window it was found in, so the object
    # may continue in the neighbouring window; merge_detections re-joins the pieces.
    cut_top: bool = False
    cut_bottom: bool = False


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

    numpy arrays are taken as BGR (what ``cv2.imread`` returns); grayscale, BGRA
    and float arrays are converted (``ocr.as_bgr``). PIL images and file paths
    are converted from RGB.
    """
    import numpy as np

    from .ocr import as_bgr

    if isinstance(image, (str, Path)):
        import cv2

        arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"cannot read image {image}")
        return arr
    if isinstance(image, np.ndarray):
        return as_bgr(image)
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


def _same_object(a: Detection, b: Detection, iou_thr: float, contain_thr: float) -> bool:
    if iou(a.box, b.box) >= iou_thr:
        return True
    # Everything else must be one column: pieces of one card are aligned horizontally.
    x_overlap = min(a.box.x + a.box.w, b.box.x + b.box.w) - max(a.box.x, b.box.x)
    if x_overlap < 0.8 * min(a.box.w, b.box.w):
        return False
    if containment(a.box, b.box) >= contain_thr:
        return True  # a piece seen next to the whole object
    # Two pieces of an object cut by a window seam: the upper one stops at the
    # bottom edge of its window, the lower one starts at the top edge of the next,
    # and both cover the band where the windows overlap.
    upper, lower = (a, b) if a.box.y <= b.box.y else (b, a)
    y_overlap = min(a.box.y + a.box.h, b.box.y + b.box.h) - max(a.box.y, b.box.y)
    return (upper.cut_bottom and lower.cut_top and y_overlap > 0
            and upper.box.y + upper.box.h <= lower.box.y + lower.box.h)


def _join(a: Detection, b: Detection) -> Detection:
    """Union of two detections of one object; edge flags come from whichever box sets that edge."""
    box = union_box(a.box, b.box)
    top = [d for d in (a, b) if d.box.y <= box.y + 1e-6]
    bottom = [d for d in (a, b) if d.box.y + d.box.h >= box.y + box.h - 1e-6]
    return Detection(cls=a.cls, conf=max(a.conf, b.conf), box=box,
                     cut_top=all(d.cut_top for d in top), cut_bottom=all(d.cut_bottom for d in bottom))


def merge_detections(dets: Sequence[Detection], *, iou_thr: float = 0.5, contain_thr: float = 0.7) -> list[Detection]:
    """Class-wise merge of duplicate boxes (typically from overlapping windows or tiles).

    Two boxes of one class are the same object when their IoU is high, when one
    mostly contains the other, or when they are the two pieces of an object cut
    by a window edge (``cut_bottom`` above, ``cut_top`` below). The merged box
    is their union with the higher confidence.
    """
    by_cls: dict[str, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)
    out: list[Detection] = []
    for items in by_cls.values():
        kept: list[Detection] = []
        for d in sorted(items, key=lambda d: -d.conf):
            # A union can newly overlap other kept boxes; repeat until stable.
            merged_any = True
            while merged_any:
                merged_any = False
                for i, k in enumerate(kept):
                    if _same_object(k, d, iou_thr, contain_thr):
                        d = _join(k, d)
                        kept.pop(i)
                        merged_any = True
                        break
            kept.append(d)
        out.extend(kept)
    return sorted(out, key=lambda d: (d.box.y, d.box.x))


# --- inference ----------------------------------------------------------------------


class Detector:
    """Run a trained card/field detector on screenshots of any size and scale."""

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
        ckpt = getattr(self.model, "ckpt", None) or {}
        train_imgsz = (ckpt.get("train_args") or {}).get("imgsz") if isinstance(ckpt, dict) else None
        # Network px per CSS px the model was trained at (training tiles are TILE_CSS tall).
        self.train_scale = float(train_imgsz if isinstance(train_imgsz, (int, float)) else imgsz) / TILE_CSS

    def detect(self, image: "ImageInput", *, tile_height: int | None = None, overlap: int | None = None,
               scale: float | None = None) -> list[Detection]:
        """Detect cards and fields; boxes are in the input image's pixels.

        ``scale`` is image pixels per CSS pixel (the device scale factor of a
        screenshot taken at 100 % zoom); when ``None`` it is measured with
        ``estimate_scale`` (falling back to 1). ``tile_height`` / ``overlap`` are
        the window size in image pixels (default: the training tiles' 1280 / 160
        CSS px times ``scale``).
        """
        arr = load_image(image)
        cache: dict = {}
        if scale is None:
            scale = self.estimate_scale(arr, _cache=cache) or 1.0
        th = int(tile_height or round(TILE_CSS * scale))
        ov = int(overlap if overlap is not None else round(OVERLAP_CSS * scale))
        windows = tile_windows(arr.shape[0], th, ov)
        imgsz = self.imgsz_for(arr.shape[1], max(y1 - y0 for y0, y1 in windows), scale)
        return self.detect_windows(arr, windows, imgsz=imgsz, _cache=cache)

    def estimate_scale(self, image: "ImageInput", *, max_probes: int = 3, _cache: dict | None = None) -> float | None:
        """Image pixels per CSS pixel, from the height of detected price / sold date / shipping lines.

        Probes up to ``max_probes`` training-sized windows from the top until a
        few such lines are found; ``None`` if there are too few. Within 15 % of 1
        counts as 1 (the model is trained with +-25 % scale augmentation).
        """
        arr = load_image(image)
        ratios: list[float] = []
        for window in tile_windows(arr.shape[0], TILE_CSS, OVERLAP_CSS)[:max_probes]:
            (dets,) = self._predict(arr, [window], self.imgsz, _cache)
            ratios += [d.box.h / REFERENCE_TEXT_HEIGHT[d.cls] for d in dets
                       if d.cls in REFERENCE_TEXT_HEIGHT and d.conf >= 0.5]
            if len(ratios) >= 3:
                break
        if len(ratios) < 2:
            return None
        found = statistics.median(ratios)
        return 1.0 if abs(math.log(found)) < math.log(1.15) else round(found, 3)

    def imgsz_for(self, width: int, height: int, scale: float = 1.0) -> int:
        """Network input size for a ``width`` x ``height`` window at ``scale`` image px per CSS px.

        ``self.imgsz`` while that shows text at a scale the model handles;
        otherwise the size that restores the training scale (a 2560 px wide
        page needs ~2048, a single-card crop ~768).
        """
        longest = max(width, height)
        net = self.imgsz * scale / longest
        lo, hi = (self.train_scale * f for f in NET_SCALE_BAND)
        if lo <= net <= hi:
            return self.imgsz
        size = longest * self.train_scale / scale
        return int(max(128, min(MAX_IMGSZ, math.ceil(size / 32) * 32)))

    def detect_windows(self, arr: "np.ndarray", windows: Sequence[tuple[int, int]], *, batch: int = 4,
                       imgsz: int | None = None, _cache: dict | None = None) -> list[Detection]:
        """Detect on the given horizontal strips ``(y0, y1)`` of a BGR array and merge the overlaps."""
        height = arr.shape[0]
        dets: list[Detection] = []
        windows = [(y0, y1) for y0, y1 in windows if y1 > y0]
        for (y0, y1), found in zip(windows, self._predict(arr, windows, imgsz or self.imgsz, _cache, batch)):
            tol = max(2.0, 0.01 * (y1 - y0))
            for d in found:
                b = d.box
                dets.append(d.model_copy(update={"cut_top": y0 > 0 and b.y - y0 <= tol,
                                                 "cut_bottom": y1 < height and y1 - (b.y + b.h) <= tol}))
        return merge_detections(dets)

    def _predict(self, arr: "np.ndarray", windows: Sequence[tuple[int, int]], imgsz: int,
                 cache: dict | None = None, batch: int = 4) -> list[list[Detection]]:
        """Raw per-window detections (image coordinates), memoised in ``cache`` by (y0, y1, imgsz)."""
        cache = {} if cache is None else cache
        todo = [(y0, y1) for y0, y1 in windows if (y0, y1, imgsz) not in cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            results = self.model.predict([arr[y0:y1] for y0, y1 in chunk], imgsz=imgsz, conf=self.conf,
                                         iou=self.iou, device=self.device, verbose=False)
            for (y0, y1), res in zip(chunk, results):
                found: list[Detection] = []
                if res.boxes is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                    for (x1, by1, x2, by2), c, k in zip(xyxy, confs, classes):
                        found.append(Detection(cls=self.names.get(int(k), str(k)), conf=float(c),
                                               box=Box(x=float(x1), y=float(by1) + y0, w=float(x2 - x1),
                                                       h=float(by2 - by1))))
                cache[(y0, y1, imgsz)] = found
        return [cache.get((y0, y1, imgsz), []) for y0, y1 in windows]
