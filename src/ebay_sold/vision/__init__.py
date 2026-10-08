"""Vision fallback: a YOLO11 detector plus OCR that reads sold listings from screenshots.

This replaces the original darknet YOLOv4-tiny + Tesseract pipeline, whose
weights and hand-drawn labels were lost. Nothing here needs hand labelling:

* ``dataset``  renders saved eBay HTML, asks the DOM where every card and field
  is drawn (``capture.card_regions``) and writes YOLO labels for each screenshot tile.
* ``detector`` trains / runs YOLO11 on those tiles (tall pages are tiled and the
  overlaps merged).
* ``ocr``      reads the field crops (RapidOCR by default, Tesseract optional).
* ``extract``  turns detections + OCR into ``Listing`` objects via ``ebay_sold.normalize``.
* ``evaluate`` scores the whole chain against the DOM on pages it was not trained on.

Submodules import their heavy dependencies (ultralytics, torch, cv2, numpy,
rapidocr) lazily, so ``import ebay_sold.vision`` is cheap.
"""

from __future__ import annotations

from ..models import VISION_CLASSES

__all__ = ["VISION_CLASSES", "load_pipeline"]


def load_pipeline(settings=None):  # -> tuple[Detector, OcrEngine]
    """``(Detector, OcrEngine)`` configured from ``Settings.vision`` (weights default: ``<data_dir>/models``)."""
    from ..config import load_settings
    from .detector import Detector
    from .ocr import get_ocr

    settings = settings or load_settings()
    detector = Detector(settings.weights_path, imgsz=settings.vision.imgsz, conf=settings.vision.conf)
    return detector, get_ocr(settings.vision.ocr_backend)
