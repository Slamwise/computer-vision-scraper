"""Detections + OCR -> listings, with a fake detector and fake OCR (no model needed)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ebay_sold.models import Box  # noqa: E402
from ebay_sold.vision.detector import Detection, merge_detections, tile_windows  # noqa: E402
from ebay_sold.vision.extract import (  # noqa: E402
    assign_fields,
    extract_listings,
    extract_listings_detailed,
    extract_page,
    reading_order,
)
from ebay_sold.vision.ocr import OcrResult  # noqa: E402

TODAY = date(2026, 5, 1)
WEIGHTS = Path(__file__).resolve().parents[1] / "data" / "models" / "ebay-sold-yolo.pt"


def det(cls: str, x: float, y: float, w: float, h: float, conf: float = 0.9) -> Detection:
    return Detection(cls=cls, conf=conf, box=Box(x=x, y=y, w=w, h=h))


# --- a fake page: each field is a flat rectangle whose gray level identifies its text -------------


class FakeOcr:
    """Reads the text a field's fill colour stands for."""

    name = "fake"

    def __init__(self, table: dict[int, str]) -> None:
        self.table = table
        self.calls = 0

    def read(self, image):
        self.calls += 1
        values, counts = np.unique(image[:, :, 0], return_counts=True)
        keep = values != 255
        if not keep.any():
            return OcrResult(text="", conf=0.0)
        v = int(values[keep][np.argmax(counts[keep])])
        return OcrResult(text=self.table.get(v, ""), conf=0.8)


class FakeDetector:
    """Returns page-level boxes as a real detector would see them in each image.

    Tile images carry their tile index in pixel (0, 0); boxes are clipped to
    the tile, cards kept when >= 30 % visible and fields when >= 50 %.
    """

    def __init__(self, dets: list[Detection], windows: list[tuple[int, int]] | None = None) -> None:
        self.dets = dets
        self.windows = windows
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.windows is None:
            return list(self.dets)
        y0, y1 = self.windows[int(image[0, 0, 1])]
        out = []
        for d in self.dets:
            b = d.box
            top, bottom = max(b.y, y0), min(b.y + b.h, y1)
            if bottom <= top or (bottom - top) / b.h < (0.3 if d.cls == "listing" else 0.5):
                continue
            out.append(Detection(cls=d.cls, conf=d.conf, box=Box(x=b.x, y=top - y0, w=b.w, h=bottom - top)))
        return out


CARD_TEXTS = {
    # gray level -> (field, text)
    "A": {10: ("title", "Hot Wheels Zamac Nissan Skyline R34"), 11: ("price", "$65.00"),
          12: ("sold_date", "SoldApr25,2026"), 13: ("shipping", "+$9.45 delivery"), 14: ("condition", "Brand New")},
    "B": {20: ("title", "2012 2013 Hot Wheels Zamac lot of 2"), 21: ("price", "$3.75 to $23.95"),
          22: ("sold_date", "Sold Apr 21, 2026"), 23: ("shipping", "Free delivery"), 24: ("condition", "Pre-Owned")},
    "C": {30: ("title", "Third card"), 31: ("price", "S12.50"), 32: ("sold_date", "Sold Mar 2, 2026")},
}


def _card(img, x: int, y: int, key: str, *, conf: float = 0.9) -> list[Detection]:
    """Paint one card's fields at (x, y) and return its detections (card + fields)."""
    dets = [det("listing", x, y, 480, 240, conf)]
    for i, (gray, (field, _)) in enumerate(CARD_TEXTS[key].items()):
        fx, fy, fw, fh = x + 200, y + 10 + 30 * i, 200, 20
        img[fy:fy + fh, fx:fx + fw] = gray
        dets.append(det(field, fx, fy, fw, fh, conf))
    img[y + 10:y + 200, x + 10:x + 180] = 128  # product photo
    dets.append(det("image", x + 10, y + 10, 170, 190, conf))
    return dets


def _table() -> dict[int, str]:
    return {g: text for card in CARD_TEXTS.values() for g, (_, text) in card.items()}


# --- merging and grouping --------------------------------------------------------------------------


def test_tile_windows():
    assert tile_windows(1000, 1280, 160) == [(0, 1000)]
    assert tile_windows(1300, 1280, 160) == [(0, 1300)]  # just over one tile: no sliver tile
    assert tile_windows(3000, 1280, 160) == [(0, 1280), (1120, 2400), (2240, 3000)]
    with pytest.raises(ValueError):
        tile_windows(3000, 100, 100)


def test_merge_rejoins_a_card_cut_by_a_tile_edge():
    dets = [
        det("listing", 236, 1100, 850, 180, 0.8),   # top part, seen in tile 0 (ends at 1280)
        det("listing", 236, 1120, 850, 230, 0.9),   # bottom part, seen in tile 1 (starts at 1120)
        det("price", 509, 1200, 70, 22, 0.7),       # same price seen whole in both tiles
        det("price", 508, 1201, 71, 21, 0.95),
        det("listing", 236, 1352, 850, 250, 0.9),   # the next card: touches, must stay separate
        det("price", 509, 1400, 70, 22, 0.9),
    ]
    merged = merge_detections(dets)
    cards = [d for d in merged if d.cls == "listing"]
    prices = [d for d in merged if d.cls == "price"]
    assert len(cards) == 2 and len(prices) == 2
    first = min(cards, key=lambda d: d.box.y)
    assert (first.box.y, first.box.y + first.box.h) == (1100, 1350)  # union of both halves
    assert first.conf == 0.9
    assert max(p.conf for p in prices if p.box.y < 1300) == 0.95


def test_merge_keeps_side_by_side_cards_apart():
    dets = [det("listing", 0, 0, 300, 400), det("listing", 310, 0, 300, 400), det("listing", 5, 3, 295, 390, 0.5)]
    merged = merge_detections(dets)
    assert len(merged) == 2


def test_assign_fields_by_max_containment():
    cards = [det("listing", 0, 0, 500, 250), det("listing", 0, 250, 500, 250)]
    fields = [
        det("price", 200, 100, 80, 20),
        det("price", 200, 245, 80, 20, 0.95),   # straddles the boundary, 75 % in card 2
        det("title", 200, 300, 200, 20),
        det("title", 200, 310, 200, 20, 0.6),   # duplicate in card 2: the more confident one is kept
        det("shipping", 900, 900, 50, 20),      # outside every card: dropped
    ]
    groups = assign_fields(cards + fields)
    assert len(groups) == 2
    (c1, f1), (c2, f2) = groups
    assert set(f1) == {"price"} and f1["price"].box.y == 100
    assert set(f2) == {"price", "title"} and f2["title"].conf == 0.9 and f2["price"].box.y == 245
    # No card box (a single-card crop): all fields form one card spanning the image.
    (card, only), = assign_fields(fields[:3], image_size=(500, 400))
    assert card.box == Box(x=0, y=0, w=500, h=400) and set(only) == {"price", "title"}
    assert assign_fields(fields[:3], allow_synthetic=False) == []


def test_reading_order_rows_then_columns():
    boxes = [Box(x=500, y=5, w=400, h=300), Box(x=0, y=0, w=400, h=300), Box(x=0, y=320, w=400, h=300)]
    assert reading_order(boxes) == [1, 0, 2]


# --- extraction with fakes --------------------------------------------------------------------------


def test_extract_listings_with_fakes():
    img = np.full((800, 1000, 3), 255, np.uint8)
    dets = _card(img, 520, 2, "B") + _card(img, 0, 0, "A") + _card(img, 0, 300, "C")
    ocr = FakeOcr(_table())
    listings = extract_listings(img, detector=FakeDetector(dets), ocr=ocr, today=TODAY)
    assert [lst.title for lst in listings] == [
        "Hot Wheels Zamac Nissan Skyline R34", "2012 2013 Hot Wheels Zamac lot of 2", "Third card"]
    assert [lst.position for lst in listings] == [1, 2, 3]
    a, b, c = listings
    assert (a.price, a.currency, a.shipping, a.sold_date, a.condition) == (65.0, "USD", 9.45, date(2026, 4, 25), "Brand New")
    assert (b.price, b.price_max, b.shipping, b.sold_date) == (3.75, 23.95, 0.0, date(2026, 4, 21))
    assert c.price == 12.5 and c.shipping is None and c.condition is None  # OCR'd "S12.50"
    for lst in listings:
        assert lst.extraction == "vision" and lst.item_id is None
        assert 0 < lst.confidence <= 1
    assert a.price_text == "$65.00" and a.sold_date_text == "SoldApr25,2026"
    assert ocr.calls == 13  # only text fields (5 + 5 + 3) are read, never the photo


def test_single_card_crop_without_card_box():
    img = np.full((250, 500, 3), 255, np.uint8)
    dets = [d for d in _card(img, 0, 0, "A") if d.cls != "listing"]
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=FakeOcr(_table()), today=TODAY)
    assert lst.price == 65.0 and lst.position == 1
    (detailed,) = extract_listings_detailed(img, detector=FakeDetector(dets), ocr=FakeOcr(_table()), today=TODAY)
    assert detailed.synthetic_box and detailed.box == Box(x=0, y=0, w=500, h=250)


def test_cards_without_price_and_title_are_dropped():
    img = np.full((600, 1000, 3), 255, np.uint8)
    dets = _card(img, 0, 0, "A")
    img[310:330, 200:400] = 12  # a lone sold date
    dets += [det("listing", 0, 300, 480, 240), det("sold_date", 200, 310, 200, 20)]
    listings = extract_listings(img, detector=FakeDetector(dets), ocr=FakeOcr(_table()), today=TODAY)
    assert len(listings) == 1 and listings[0].price == 65.0


def test_page_offset_maps_boxes_to_page_css_pixels():
    img = np.full((500, 1000, 3), 255, np.uint8)  # a tile shot at device scale factor 2
    dets = _card(img, 100, 200, "A")
    (e,) = extract_listings_detailed(img, detector=FakeDetector(dets), ocr=FakeOcr(_table()), today=TODAY,
                                     page_offset=Box(x=0, y=1000, w=500, h=250))
    assert e.box == Box(x=50, y=1100, w=240, h=120)


def test_extract_page_merges_cards_split_across_tiles(tmp_path):
    import cv2

    page = np.full((1500, 1000, 3), 255, np.uint8)
    dets = (_card(page, 0, 100, "A") + _card(page, 0, 560, "B")   # B spans 560..800, across the tile seam
            + _card(page, 0, 1200, "C"))
    windows = [(0, 700), (500, 1500)]  # overlap 500..700: B is cut in both tiles
    tiles = []
    for i, (y0, y1) in enumerate(windows):
        tile = page[y0:y1].copy()
        tile[0, 0] = (255, i, 255)  # tile index for the fake detector (green channel)
        path = tmp_path / f"tile{i}.png"
        cv2.imwrite(str(path), tile)
        tiles.append((path, Box(x=0, y=y0, w=1000, h=y1 - y0)))
    detector = FakeDetector(dets, windows)
    listings = extract_page(tiles, detector=detector, ocr=FakeOcr(_table()), today=TODAY)
    assert detector.calls == 2
    assert [lst.title for lst in listings] == [
        "Hot Wheels Zamac Nissan Skyline R34", "2012 2013 Hot Wheels Zamac lot of 2", "Third card"]
    b = listings[1]
    # every field of the split card was read, each from a tile that shows it whole
    assert (b.price, b.sold_date, b.shipping, b.condition) == (3.75, date(2026, 4, 21), 0.0, "Pre-Owned")
    assert [lst.position for lst in listings] == [1, 2, 3]


# --- optional: the trained model on a real rendered page -----------------------------------------------


@pytest.mark.browser
@pytest.mark.vision
@pytest.mark.slow
@pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights at data/models/ebay-sold-yolo.pt")
async def test_trained_detector_reads_a_rendered_tile(require_browser, require_vision, tmp_path):
    from conftest import load_fixture

    from ebay_sold.capture import capture_tiles, render_html
    from ebay_sold.vision.detector import Detector
    from ebay_sold.vision.ocr import get_ocr

    async with render_html(load_fixture("sold_2026-04-26_hot-wheels-r34-zamac"), load_images=False) as page:
        tiles = await capture_tiles(page, tmp_path, tile_height=1280, overlap=160)
    detector = Detector(WEIGHTS)
    listings = extract_listings(tiles[1][0], detector=detector, ocr=get_ocr("auto"), today=TODAY,
                                page_offset=tiles[1][1])
    priced = [lst for lst in listings if lst.price is not None]
    assert len(priced) >= 1
    assert all(lst.extraction == "vision" for lst in listings)


# --- scoring against the DOM ---------------------------------------------------------------------------


def test_match_cards_and_score_pair():
    from ebay_sold.models import CardRegion, Listing
    from ebay_sold.vision.evaluate import match_cards, score_pair
    from ebay_sold.vision.extract import ExtractedListing

    regions = [
        CardRegion(item_id="1", box=Box(x=0, y=0, w=800, h=250),
                   texts={"price": "$65.00", "sold_date": "Sold  Apr 25, 2026", "shipping": "Free delivery",
                          "condition": "New – Open box", "title": "Hot Wheels Zamac R34"}),
        CardRegion(item_id="2", box=Box(x=0, y=260, w=800, h=250), texts={"price": "$1.00"}),
    ]
    good = Listing(title="Hot Wheels Zamac R34", price=65.0, sold_date=date(2026, 4, 25), shipping=0.0,
                   condition="New - Open box", extraction="vision")
    preds = [
        ExtractedListing(listing=Listing(title="x", price=2.0), box=Box(x=0, y=600, w=800, h=250)),  # extra card
        ExtractedListing(listing=good, box=Box(x=2, y=4, w=790, h=245)),
    ]
    assert match_cards(preds, regions) == [(1, 0)]
    scores = score_pair(preds[1], regions[0], today=TODAY)
    assert {k: v[0] for k, v in scores.items()} == {
        "price": True, "sold_date": True, "shipping": True, "condition": True, "title": True}
    wrong = preds[1].model_copy(deep=True)
    wrong.listing.price, wrong.listing.title = 66.0, "Hot Wheels"
    scores = score_pair(wrong, regions[0], today=TODAY)
    assert not scores["price"][0] and not scores["title"][0] and scores["price"][1:] == (65.0, 66.0)


def test_title_crop_stops_at_the_next_field_and_drops_new_listing_tag():
    from ebay_sold.vision.extract import _trim_against

    title = Box(x=500, y=100, w=400, h=50)  # too tall: reaches into the condition line
    condition = Box(x=500, y=140, w=100, h=18)
    trimmed = _trim_against(title, [condition, Box(x=0, y=100, w=100, h=20)])  # other column: ignored
    assert (trimmed.y, trimmed.y + trimmed.h) == (100, 140)

    img = np.full((250, 500, 3), 255, np.uint8)
    img[10:30, 200:400] = 40
    img[40:60, 200:300] = 41
    dets = [det("listing", 0, 0, 480, 240), det("title", 200, 10, 200, 40), det("price", 200, 40, 100, 20)]
    ocr = FakeOcr({40: "NEW LISTING Hot Wheels Zamac", 41: "$5.00"})
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=ocr, today=TODAY)
    assert lst.title == "Hot Wheels Zamac" and lst.price == 5.0
