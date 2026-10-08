"""OCR for detected field crops (one or two short lines of clean screen text).

Two engines behind one small protocol (``OcrEngine.read(bgr_array) -> OcrResult``):

* ``RapidOcrEngine``: PaddleOCR's PP-OCRv3 recogniser run with onnxruntime
  (``rapidocr_onnxruntime``); pip-only, ~35 ms per field on one CPU core.
* ``TesseractEngine``: the engine the original project used; needs the
  ``tesseract`` binary; ~140 ms per field, so crops are read in parallel
  processes (``read_many``). Slightly more exact on titles; the ``"auto"``
  default when installed.

Why the RapidOCR output used to lose its spaces ("SoldApr25,2026",
"20122013HotWheelsZamac..."): ``RapidOCR()(crop)`` first runs its text
*detector* on the crop. On a tiny, tightly cropped field it returns boxes that
are cut through letters and re-crops/warps them, and the recogniser then sees
glyphs squeezed together. Measured on 300 held-out field crops, that pipeline
got the sold date exactly right 25 % of the time and titles 0 %. We skip the
detector: lines are found with a horizontal ink projection, trimmed to their
ink plus a margin, and each line is resized to the recogniser's 48 px input
height. That alone restores the spaces (sold date / shipping / condition /
price text 100 % exact on the same crops). As a backstop, ``spaces="gaps"``
re-inserts a space wherever a wide ink gap falls between two consecutive CTC
character positions; at the default ``gap_ratio`` it never fired wrongly in
our measurements (a lower ratio splits wide capitals: "SKYLIN E").

The remaining RapidOCR title errors come from its mostly-Chinese vocabulary:
"&" is read as "8" ("Roebuck 8 Co."). Tesseract gets those right.

All images are OpenCV-style BGR ``uint8`` arrays.
"""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

log = logging.getLogger(__name__)

SpaceMode = Literal["gaps", "words", "none"]


class OcrResult(BaseModel):
    text: str
    conf: float  # 0..1


@runtime_checkable
class OcrEngine(Protocol):
    name: str

    def read(self, image: "np.ndarray") -> OcrResult: ...


# --- layout helpers (pure numpy) -------------------------------------------------


def ink_mask(image: "np.ndarray", *, min_contrast: int = 60) -> "np.ndarray":
    """Boolean mask of text pixels: whatever differs clearly from the background.

    The background colour is the median of the crop border, so dark-on-light,
    coloured (green prices) and light-on-dark text all work.
    """
    import numpy as np

    img = image if image.ndim == 3 else image[:, :, None]
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.int16)
    bg = np.median(border, axis=0)
    diff = np.abs(img.astype(np.int16) - bg).max(axis=2)
    return diff >= min_contrast


def _runs(flags: "np.ndarray") -> list[tuple[int, int]]:
    """``[start, end)`` index ranges where ``flags`` is true."""
    import numpy as np

    padded = np.concatenate([[False], flags.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def text_lines(mask: "np.ndarray", *, min_height: int = 4) -> list[tuple[int, int]]:
    """Row ranges of the text lines in a crop, from the horizontal ink projection."""
    rows = mask.sum(axis=1) > 0
    runs = _runs(rows)
    if not runs:
        return []
    # Re-join a line that a thin empty row split (e.g. between "i" dot and stem).
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= 1:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    lines = [(a, b) for a, b in merged if b - a >= min_height]
    if not lines:
        return []
    # Fragments much shorter than the tallest line (an underline, the descenders of
    # a line cut off above the crop) are not text we can read; drop them.
    tallest = max(b - a for a, b in lines)
    return [(a, b) for a, b in lines if b - a >= 0.4 * tallest]


def word_gaps(mask: "np.ndarray", *, gap_ratio: float = 0.28) -> list[tuple[int, int]]:
    """Column ranges of the gaps between words on one text line.

    A gap between ink columns is a word gap when it is at least ``gap_ratio``
    times the line's ink height (letter gaps in eBay's fonts are ~0-0.15, word
    gaps ~0.3-0.4 of the line height).
    """
    cols = mask.sum(axis=0) > 0
    ink = _runs(cols)
    if len(ink) < 2:
        return []
    rows = _runs(mask.sum(axis=1) > 0)
    height = (rows[-1][1] - rows[0][0]) if rows else mask.shape[0]
    min_gap = max(2.0, gap_ratio * height)
    return [(a[1], b[0]) for a, b in zip(ink, ink[1:]) if b[0] - a[1] >= min_gap]


def _line_crops(image: "np.ndarray", *, margin_ratio: float = 0.25) -> list[tuple["np.ndarray", "np.ndarray"]]:
    """Split a crop into ``(line_image, line_mask)`` pairs, trimmed to the ink with a margin."""
    mask = ink_mask(image)
    lines = text_lines(mask)
    if not lines:
        return []
    out = []
    h, w = mask.shape
    for y0, y1 in lines:
        cols = _runs(mask[y0:y1].sum(axis=0) > 0)
        if not cols:
            continue
        x0, x1 = cols[0][0], cols[-1][1]
        m = max(2, int(round((y1 - y0) * margin_ratio)))
        ya, yb = max(0, y0 - m), min(h, y1 + m)
        xa, xb = max(0, x0 - m), min(w, x1 + m)
        out.append((image[ya:yb, xa:xb], mask[ya:yb, xa:xb]))
    return out


# --- RapidOCR -----------------------------------------------------------------------


class RapidOcrEngine:
    """PP-OCRv3 recogniser via ``rapidocr_onnxruntime``, run line by line (no text detector).

    ``spaces``: ``"gaps"`` (default) also inserts spaces at wide ink gaps
    between CTC character positions; ``"words"`` recognises each gap-separated
    word on its own (measured slightly worse); ``"none"`` returns the
    recogniser's output as is. ``threads`` caps onnxruntime's CPU threads.
    """

    name = "rapidocr"

    def __init__(self, *, spaces: SpaceMode = "gaps", gap_ratio: float = 0.28, upscale: float = 1.0,
                 min_line_height: int = 0, rec_height: int = 48, threads: int | None = None) -> None:
        from rapidocr_onnxruntime import RapidOCR

        # Only the recogniser is used: RapidOCR's text detector is what loses the
        # spaces and garbles short crops (see the module docstring).
        self._engine = RapidOCR(use_text_det=False, use_angle_cls=False)
        self._rec = self._engine.text_recognizer
        if threads:
            self._limit_threads(threads)
        self.spaces = spaces
        self.gap_ratio = gap_ratio
        self.upscale = upscale
        self.min_line_height = min_line_height
        self.rec_height = rec_height

    def _limit_threads(self, threads: int) -> None:
        import onnxruntime as ort

        wrapper = self._rec.session
        old = wrapper.session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model_path = getattr(old, "_model_path", None) or self._engine_model_path()
        wrapper.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

    @staticmethod
    def _engine_model_path() -> str:
        from pathlib import Path

        import rapidocr_onnxruntime

        return str(Path(rapidocr_onnxruntime.__file__).parent / "models" / "ch_PP-OCRv3_rec_infer.onnx")

    # Raw recogniser call that also returns where (in line pixels) each character was emitted.
    def _recognize(self, line: "np.ndarray") -> tuple[str, list[float], list[float]]:
        import math

        import cv2
        import numpy as np

        h, w = line.shape[:2]
        target_h = self.rec_height
        resized_w = max(8, int(math.ceil(target_h * w / max(h, 1))))
        img = cv2.resize(line, (resized_w, target_h), interpolation=cv2.INTER_CUBIC if target_h > h else cv2.INTER_AREA)
        # Pad to a multiple of the network's width stride; scale like rapidocr does.
        padded_w = int(math.ceil(resized_w / 32.0) * 32)
        x = np.zeros((1, 3, target_h, padded_w), dtype=np.float32)
        x[0, :, :, :resized_w] = (img.astype(np.float32).transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        preds = self._rec.session(x)[0][0]  # (T, C)
        idx = preds.argmax(axis=1)
        prob = preds.max(axis=1)
        chars, confs, xs = [], [], []
        decoder = self._rec.postprocess_op
        step = padded_w / preds.shape[0]
        prev = -1
        for t, k in enumerate(idx):
            if k != 0 and k != prev:
                chars.append(decoder.character[int(k)])
                confs.append(float(prob[t]))
                xs.append((t + 0.5) * step * w / resized_w)
            prev = k
        return "".join(chars), confs, xs

    def _read_line(self, line: "np.ndarray", mask: "np.ndarray") -> tuple[str, list[float]]:
        import cv2

        if self.upscale and self.upscale != 1.0:
            line = cv2.resize(line, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask.astype("uint8"), None, fx=self.upscale, fy=self.upscale,
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        if self.spaces == "words":
            gaps = word_gaps(mask, gap_ratio=self.gap_ratio)
            edges = [0, *[(a + b) // 2 for a, b in gaps], line.shape[1]]
            words, confs = [], []
            for x0, x1 in zip(edges, edges[1:]):
                if x1 - x0 < 2:
                    continue
                text, c, _ = self._recognize(line[:, x0:x1])
                if text.strip():
                    words.append(text.strip())
                    confs.extend(c)
            return " ".join(words), confs
        text, confs, xs = self._recognize(line)
        if self.spaces == "none" or not text:
            return text.strip(), confs
        gaps = word_gaps(mask, gap_ratio=self.gap_ratio)
        out = []
        for i, ch in enumerate(text):
            out.append(ch)
            if i + 1 < len(text) and ch != " " and text[i + 1] != " ":
                if any(xs[i] < (a + b) / 2 < xs[i + 1] for a, b in gaps):
                    out.append(" ")
        return "".join(out).strip(), confs

    def read(self, image: "np.ndarray") -> OcrResult:
        texts, confs = [], []
        for line, mask in _line_crops(image):
            if line.shape[0] < self.min_line_height:
                continue
            text, c = self._read_line(line, mask)
            if text:
                texts.append(text)
                confs.extend(c)
        if not texts:
            return OcrResult(text="", conf=0.0)
        return OcrResult(text=" ".join(texts), conf=round(sum(confs) / len(confs), 4) if confs else 0.0)


# --- Tesseract ----------------------------------------------------------------------


def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


class TesseractEngine:
    """Tesseract (``pytesseract``) on upscaled grayscale lines (``--psm 7``)."""

    name = "tesseract"

    def __init__(self, *, upscale: float = 3.0, lang: str = "eng", oem: int = 1, workers: int | None = None) -> None:
        if not tesseract_available():
            raise RuntimeError("tesseract is not installed (apt install tesseract-ocr, pip install pytesseract)")
        import os

        import pytesseract

        # Tesseract's OpenMP threads make single-line calls 10-100x slower when
        # the CPU is busy (e.g. next to YOLO); one thread per call is fastest here.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        self._tess = pytesseract
        self.workers = workers or os.cpu_count() or 1
        self.upscale = upscale
        self.config = f"--oem {oem} --psm 7"
        self.lang = lang

    def _read_line(self, line: "np.ndarray") -> tuple[str, list[float]]:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        # Tesseract wants dark text on a light background.
        if np.median(np.concatenate([gray[0], gray[-1]])) < 128:
            gray = 255 - gray
        if self.upscale != 1.0:
            gray = cv2.resize(gray, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        data = self._tess.image_to_data(gray, lang=self.lang, config=self.config,
                                        output_type=self._tess.Output.DICT)
        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = (text or "").strip()
            if text:
                words.append(text)
                confs.append(max(0.0, float(conf)) / 100.0)
        return " ".join(words), confs

    def read(self, image: "np.ndarray") -> OcrResult:
        texts, confs = [], []
        for line, _ in _line_crops(image):
            text, c = self._read_line(line)
            if text:
                texts.append(text)
                confs.extend(c)
        if not texts:
            return OcrResult(text="", conf=0.0)
        return OcrResult(text=" ".join(texts), conf=round(sum(confs) / len(confs), 4))


    def read_many(self, images: "list[np.ndarray]") -> list[OcrResult]:
        """Read several crops in parallel (each Tesseract call is a separate process)."""
        if self.workers <= 1 or len(images) < 2:
            return [self.read(img) for img in images]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.read, images))


def read_many(engine: OcrEngine, images: "list[np.ndarray]") -> list[OcrResult]:
    """``engine.read_many`` when the engine has one (parallel Tesseract), else one ``read`` per image."""
    many = getattr(engine, "read_many", None)
    if callable(many):
        return list(many(images))
    return [engine.read(img) for img in images]


# --- selection ------------------------------------------------------------------------

# Measured on a held-out page (64 cards): both engines read every price, sold date,
# shipping and condition right; Tesseract also gets titles exact (mean similarity
# 0.999 vs 0.989, RapidOCR turns "&" into "8") at about the same page time once its
# calls run in parallel. So "auto" prefers Tesseract when the binary is installed
# and falls back to the pip-only RapidOCR.
AUTO_ORDER = ("tesseract", "rapidocr")


def get_ocr(backend: str = "auto") -> OcrEngine:
    """Return an OCR engine: ``"rapidocr"``, ``"tesseract"`` or ``"auto"`` (Tesseract if installed, else RapidOCR)."""
    backend = (backend or "auto").lower()
    if backend == "rapidocr":
        return RapidOcrEngine()
    if backend == "tesseract":
        return TesseractEngine()
    if backend != "auto":
        raise ValueError(f"unknown OCR backend {backend!r}; expected auto, rapidocr or tesseract")
    errors = []
    for name in AUTO_ORDER:
        try:
            return get_ocr(name)
        except Exception as exc:  # missing package / binary
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no OCR engine available: " + "; ".join(errors))
