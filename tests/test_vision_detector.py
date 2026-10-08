"""Detector tiling, scale handling and seam merging, with a fake YOLO model (no weights needed).

The fake "network" finds flat rectangles of known gray levels in whatever
window it is given, so cards cut by a window edge come back clipped exactly as
a real detector reports them, and box heights scale with the image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from ebay_sold.models import VISION_CLASSES, Box  # noqa: E402
from ebay_sold.vision.detector import Detection, Detector, load_image, merge_detections  # noqa: E402

# gray level -> class index
LEVELS = {200: VISION_CLASSES.index("listing"), 70: VISION_CLASSES.index("sold_date"),
          60: VISION_CLASSES.index("price"), 50: VISION_CLASSES.index("shipping"),
          30: VISION_CLASSES.index("title")}


class _Arr:
    """Stands in for a torch tensor: ``.cpu().numpy()``."""

    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Result:
    def __init__(self, rows):
        a = np.array(rows, dtype=np.float32).reshape(-1, 6)
        self.boxes = type("Boxes", (), {"xyxy": _Arr(a[:, :4]), "conf": _Arr(a[:, 4]), "cls": _Arr(a[:, 5])})()


class FakeYolo:
    def __init__(self):
        self.calls: list[tuple[tuple[int, int], int]] = []  # (window shape, imgsz)

    def predict(self, images, *, imgsz, conf, iou, device, verbose):
        out = []
        for img in images:
            self.calls.append((img.shape[:2], imgsz))
            rows = []
            for level, cls in LEVELS.items():
                _n, _, stats, _ = cv2.connectedComponentsWithStats((img[:, :, 0] == level).astype(np.uint8))
                for x, y, w, h, _area in stats[1:]:
                    rows.append((x, y, x + w, y + h, 0.9, cls))
            out.append(_Result(rows))
        return out


def fake_detector(imgsz: int = 1024) -> Detector:
    det = Detector.__new__(Detector)
    det.weights, det.model = Path("fake.pt"), FakeYolo()
    det.imgsz, det.conf, det.iou, det.device = imgsz, 0.25, 0.6, "cpu"
    det.names = dict(enumerate(VISION_CLASSES))
    det.train_scale = 1024 / 1280
    return det


def fill(img, x, y, w, h, level, d):
    img[int(y * d):int((y + h) * d), int(x * d):int((x + w) * d)] = level


def page(card_tops: list[int], *, d: float = 1.0, width: int = 1000, height: int | None = None):
    """A results page at device scale ``d``: cards 256 CSS px tall, fields at eBay's CSS sizes."""
    height = height or max(card_tops) + 400
    img = np.full((int(height * d), int(width * d), 3), 255, np.uint8)
    for y in card_tops:
        fill(img, 236, y, 700, 256, 200, d)
        fill(img, 500, y + 10, 150, 16, 70, d)   # sold date (16 CSS px)
        fill(img, 500, y + 34, 300, 16, 30, d)   # title
        fill(img, 500, y + 70, 100, 24, 60, d)   # price (24 CSS px)
        fill(img, 500, y + 200, 160, 16, 50, d)  # shipping, near the bottom of the card
    return img


def by_cls(dets: list[Detection], cls: str) -> list[Detection]:
    return sorted((d for d in dets if d.cls == cls), key=lambda d: d.box.y)


# --- merging pieces cut by a window seam -------------------------------------------------------------


def test_seam_pieces_are_rejoined_however_small_the_overlap():
    # Window overlap 80 px (1280/160 windows at device scale factor 2), card 256 px centred on it:
    # two 168 px pieces that share only 80 px. Containment (0.48) cannot tell them from two cards.
    top = Detection(cls="listing", conf=0.98, box=Box(x=236, y=1032, w=850, h=168), cut_bottom=True)
    bottom = Detection(cls="listing", conf=0.99, box=Box(x=236, y=1120, w=850, h=168), cut_top=True)
    (card,) = merge_detections([top, bottom])
    assert (card.box.y, card.box.y + card.box.h) == (1032, 1288) and card.conf == 0.99
    assert not card.cut_top and not card.cut_bottom
    # The same boxes without seam flags are two different cards, as before.
    plain = [d.model_copy(update={"cut_top": False, "cut_bottom": False}) for d in (top, bottom)]
    assert len(merge_detections(plain)) == 2
    # Pieces in different columns are never joined.
    other = bottom.model_copy(update={"box": Box(x=1100, y=1120, w=400, h=168)})
    assert len(merge_detections([top, other])) == 2


def test_card_on_an_inner_seam_at_device_scale_factor_2_keeps_its_bottom_fields():
    # The reviewer's failure: 1280/160 *pixel* windows at dsf 2 cut a 256 CSS px card into two
    # pieces that were never re-joined, and its shipping row (in the lower piece) was lost.
    det = fake_detector()
    img = page([100, 472, 840], d=2, height=1200)  # card 2: 944..1456 px, centred on the 1120..1280 overlap
    dets = det.detect(img, tile_height=1280, overlap=160, scale=2.0)
    cards = by_cls(dets, "listing")
    assert len(cards) == 3
    assert (cards[1].box.y, cards[1].box.y + cards[1].box.h) == (944, 1456)
    ships = by_cls(dets, "shipping")
    assert len(ships) == 3 and all(cards[i].box.intersection(ships[i].box) == ships[i].box.area for i in range(3))

    # End to end: every card keeps its shipping cost.
    from datetime import date

    from test_vision_extract import FakeOcr

    from ebay_sold.vision.extract import extract_listings

    class OldWindows:
        def detect(self, image):
            return det.detect(image, tile_height=1280, overlap=160, scale=2.0)

    ocr = FakeOcr({70: "Sold Apr 25, 2026", 30: "Sears catalog", 60: "$65.00", 50: "+$9.45 delivery"})
    listings = extract_listings(img, detector=OldWindows(), ocr=ocr, today=date(2026, 5, 1))
    assert [lst.shipping for lst in listings] == [9.45, 9.45, 9.45]


# --- scale: device scale factor, wide pages, single cards ---------------------------------------------


def test_estimate_scale_reads_the_device_scale_factor_from_text_heights():
    det = fake_detector()
    assert det.estimate_scale(page([100, 400, 700], d=1)) == 1.0
    assert det.estimate_scale(page([100, 400, 700], d=2)) == 2.0
    assert det.estimate_scale(page([100, 400, 700], d=1.5)) == 1.5
    assert det.estimate_scale(np.full((500, 500, 3), 255, np.uint8)) is None  # nothing to measure


def test_default_windows_follow_the_measured_scale():
    det = fake_detector()
    img = page([100, 400, 700, 1000, 1300, 1600, 1900, 2200], d=2)  # 5200 px tall
    dets = det.detect(img)
    assert len(by_cls(dets, "listing")) == 8 and len(by_cls(dets, "shipping")) == 8
    # windows are 1280 / 160 CSS px = 2560 / 320 image px, all at the configured input size
    window_heights = {shape[0] for shape, imgsz in det.model.calls if imgsz == 1024 and shape[0] != 1280}
    assert max(window_heights) == 2560


def test_imgsz_keeps_text_at_the_training_scale():
    det = fake_detector()
    assert det.imgsz_for(1366, 1280, 1.0) == 1024   # ordinary page tile: unchanged
    assert det.imgsz_for(1920, 1280, 1.0) == 1024   # 0.53x: still within the trained range
    assert det.imgsz_for(2732, 2560, 2.0) == 1024   # dsf 2 at 1366 CSS px
    assert det.imgsz_for(2560, 1280, 1.0) == 2048   # full-screen 1440p window: text would be 0.4x
    assert det.imgsz_for(945, 264, 1.0) == 768      # single-card crop: would be upscaled 1.08x
    assert det.imgsz_for(1890, 528, 2.0) == 768     # the same card at dsf 2


def test_wide_page_and_single_card_are_run_at_the_matching_input_size():
    det = fake_detector()
    wide = page([100, 400, 700], d=1, width=2560, height=1200)
    assert len(by_cls(det.detect(wide), "listing")) == 3
    assert det.model.calls[-1][1] == 2048
    card = page([4], d=1, width=945, height=264)[:, :945]
    dets = det.detect(card)
    assert det.model.calls[-1] == ((264, 945), 768)
    assert {d.cls for d in dets} >= {"listing", "price", "sold_date", "shipping", "title"}


def test_overlapping_boxes_of_one_class_in_a_single_window_are_merged():
    # A long title split into two overlapping partial boxes (seen on single-card crops).
    det = fake_detector()

    class TwoTitles(FakeYolo):
        def predict(self, images, **kw):
            res = super().predict(images, **kw)
            title = VISION_CLASSES.index("title")
            for r in res:
                r.boxes = _Result([(277, 34, 714, 50, 0.75, title), (413, 34, 843, 50, 0.5, title)]).boxes
            return res

    det.model = TwoTitles()
    (title,) = det.detect(np.full((264, 945, 3), 255, np.uint8), scale=1.0)
    assert (title.box.x, title.box.x + title.box.w) == (277, 843)


def test_load_image_converts_float_and_gray_arrays():
    assert load_image(np.full((10, 10, 3), 0.8, np.float32)).max() == 204  # normalised [0, 1] floats
    assert load_image(np.full((10, 10, 3), 128.0)).max() == 128              # [0, 255] floats
    gray = load_image(np.full((10, 12), 7, np.uint8))
    assert gray.shape == (10, 12, 3) and gray.dtype == np.uint8
    assert load_image(np.zeros((4, 4, 4), np.uint8)).shape == (4, 4, 3)
