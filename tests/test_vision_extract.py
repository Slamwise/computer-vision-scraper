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
    extract_page_detailed,
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


@pytest.mark.browser
@pytest.mark.vision
@pytest.mark.slow
@pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights at data/models/ebay-sold-yolo.pt")
async def test_trained_detector_on_retina_tiles_and_single_card_crops(require_browser, require_vision, tmp_path):
    """Regressions found in review: at device scale factor 2 a card cut by an inner window seam lost its
    shipping row, and single-card crops lost the end of long titles. Held-out page, real model."""
    from conftest import load_fixture

    from ebay_sold.capture import capture_cards, capture_tiles, card_regions, render_html
    from ebay_sold.vision.detector import Detector
    from ebay_sold.vision.evaluate import match_cards, score_pair
    from ebay_sold.vision.ocr import get_ocr

    async with render_html(load_fixture("sold_2026-04-24_sears-roebuck-magazine"), device_scale_factor=2,
                           load_images=False) as page:
        regions = await card_regions(page)
        tiles = await capture_tiles(page, tmp_path / "tiles", tile_height=1280, overlap=160)
        cards = await capture_cards(page, tmp_path / "cards", regions=regions[:10])
    detector, ocr = Detector(WEIGHTS), get_ocr("rapidocr")
    preds = extract_page_detailed(tiles, detector=detector, ocr=ocr, today=TODAY)
    pairs = match_cards(preds, regions)
    assert len(pairs) == len(regions) == len(preds)
    for i, j in pairs:
        scores = score_pair(preds[i], regions[j], today=TODAY)
        assert scores["shipping"][0] and scores["price"][0], (regions[j].item_id, scores)
    for region, path in cards:
        (e,) = extract_listings_detailed(path, detector=detector, ocr=ocr, today=TODAY)
        assert score_pair(e, region, today=TODAY)["title"][0], (region.texts["title"], e.listing.title)


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


# --- titles vs conditions: the title's ink decides ---------------------------------------------------------


class LinesOcr:
    """Reads each text line of a crop as the text its fill colour stands for."""

    name = "lines"

    def __init__(self, table: dict[int, str]) -> None:
        self.table = table

    def read(self, image):
        from ebay_sold.vision.ocr import ink_mask, text_lines

        mask = ink_mask(image)
        words = []
        for y0, y1 in text_lines(mask):
            values = image[y0:y1, :, 0][mask[y0:y1]]
            words.append(self.table.get(int(np.bincount(values).argmax()), "?"))
        return OcrResult(text=" ".join(words), conf=0.9)


# Title lines in eBay's near-black (two slightly different values so the OCR fake can tell
# them apart), the condition in eBay's grey, price and sold date in other colours.
INK = {25: "Vintage Sears Catalog 1970 Mint", 26: "Condition Rare", 120: "Pre-Owned", 60: "$29.99",
       90: "Sold Apr 20, 2026"}


def _two_line_title_card(*, condition_line: bool = True):
    """Sold date, a title wrapped onto two lines, a grey condition line, a price."""
    img = np.full((240, 480, 3), 255, np.uint8)
    img[10:26, 200:330] = 90      # sold date
    img[34:50, 200:440] = 25      # title, line 1
    img[54:70, 200:380] = 26      # title, line 2
    if condition_line:
        img[76:92, 200:290] = 120  # condition (grey)
    img[100:124, 200:270] = 60    # price
    base = [det("listing", 0, 0, 480, 240), det("sold_date", 200, 10, 130, 16), det("price", 200, 100, 70, 24)]
    return img, base


def test_second_title_line_detected_as_condition_is_folded_back_into_the_title():
    # Seen at widths 1024 and 1280: the title box covers line 1 only and line 2 is called "condition".
    img, base = _two_line_title_card(condition_line=False)
    dets = base + [det("title", 200, 34, 240, 16), det("condition", 200, 54, 180, 16)]
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=LinesOcr(INK), today=TODAY)
    assert lst.title == "Vintage Sears Catalog 1970 Mint Condition Rare"
    assert lst.condition is None


def test_false_condition_inside_the_title_box_does_not_hide_the_real_condition():
    # Title box spans both lines; the more confident "condition" is the title's second line,
    # the real (grey) condition line below is detected with lower confidence.
    img, base = _two_line_title_card()
    dets = base + [det("title", 200, 34, 240, 36), det("condition", 200, 54, 180, 16, 0.9),
                   det("condition", 200, 76, 90, 16, 0.6)]
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=LinesOcr(INK), today=TODAY)
    assert (lst.title, lst.condition) == ("Vintage Sears Catalog 1970 Mint Condition Rare", "Pre-Owned")


def test_title_box_reaching_into_a_grey_condition_line_is_still_trimmed():
    img = np.full((240, 480, 3), 255, np.uint8)
    img[34:50, 200:440] = 25   # one-line title
    img[54:70, 200:290] = 120  # condition right below it
    img[100:124, 200:270] = 60
    dets = [det("listing", 0, 0, 480, 240), det("title", 200, 34, 240, 32), det("condition", 200, 54, 90, 16),
            det("price", 200, 100, 70, 24)]
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=LinesOcr(INK), today=TODAY)
    assert (lst.title, lst.condition) == ("Vintage Sears Catalog 1970 Mint", "Pre-Owned")


def test_title_box_covering_only_the_first_line_is_extended_over_the_second():
    # Seen at width 800: line 2 is not detected at all.
    img, base = _two_line_title_card()
    dets = base + [det("title", 200, 34, 240, 16), det("condition", 200, 76, 90, 16)]
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=LinesOcr(INK), today=TODAY)
    assert (lst.title, lst.condition) == ("Vintage Sears Catalog 1970 Mint Condition Rare", "Pre-Owned")
    # ...but never over a grey line, nor over a line another field claims.
    img2, base2 = _two_line_title_card()
    img2[54:70, 200:380] = 120
    (lst2,) = extract_listings(img2, detector=FakeDetector(base2 + [det("title", 200, 34, 240, 16)]),
                               ocr=LinesOcr(INK), today=TODAY)
    assert lst2.title == "Vintage Sears Catalog 1970 Mint"


def test_title_box_covering_only_the_second_line_is_extended_upwards_but_not_over_the_sold_date():
    img, base = _two_line_title_card()
    dets = base + [det("title", 200, 54, 180, 16), det("condition", 200, 76, 90, 16)]
    (e,) = extract_listings_detailed(img, detector=FakeDetector(dets), ocr=LinesOcr(INK), today=TODAY)
    assert (e.listing.title, e.listing.condition) == ("Vintage Sears Catalog 1970 Mint Condition Rare", "Pre-Owned")
    assert e.listing.sold_date == date(2026, 4, 20)
    title = e.fields["title"]
    assert (title.x, title.y, title.x + title.w, title.y + title.h) == (200, 34, 440, 70)  # line 1 is the longer one


def test_condition_box_on_a_seller_tagline_moves_to_the_condition_line_below():
    # Seller taglines share the condition's grey and are drawn above it; eBay draws the condition last.
    img = np.full((240, 480, 3), 255, np.uint8)
    img[34:50, 200:440] = 25    # title
    img[54:70, 200:420] = 121   # tagline (grey)
    img[74:90, 200:290] = 120   # condition (grey)
    img[100:124, 200:270] = 60  # price
    dets = [det("listing", 0, 0, 480, 240), det("title", 200, 34, 240, 16), det("condition", 200, 54, 220, 16),
            det("price", 200, 100, 70, 24)]
    ocr = LinesOcr({**INK, 121: "FREE AND FAST SHIPPING. FREE RETURNS."})
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=ocr, today=TODAY)
    assert (lst.title, lst.condition, lst.price) == ("Vintage Sears Catalog 1970 Mint", "Pre-Owned", 29.99)


def test_overlapping_partial_boxes_of_one_field_are_joined():
    # A long title split into two overlapping boxes (single-card crops): keep both halves.
    card = det("listing", 0, 0, 945, 264)
    parts = [det("title", 277, 34, 437, 16, 0.75), det("title", 413, 34, 430, 16, 0.5)]
    ((_, fields),) = assign_fields([card, *parts])
    assert (fields["title"].box.x, fields["title"].box.x + fields["title"].box.w) == (277, 843)
    assert fields["title"].conf == 0.75


# --- conversion guards -------------------------------------------------------------------------------------


def test_rows_mistaken_for_shipping_are_not_a_shipping_cost():
    from ebay_sold.vision.extract import _to_listing

    def ship(text):
        lst = _to_listing({"title": "x", "shipping": text}, site="www.ebay.com", today=TODAY)
        return lst.shipping, lst.shipping_text

    assert ship("Free returns") == (None, None)       # was 0.0: "free shipping"
    assert ship("or Best Offer") == (None, None)
    assert ship("from Canada") == (None, None)
    assert ship("+$5.55 shipping") == (5.55, "+$5.55 shipping")
    assert ship("+$9.45 delivery") == (9.45, "+$9.45 delivery")
    assert ship("Free delivery") == (0.0, "Free delivery")
    assert ship("Free shipping Free returns")[0] == 0.0
    assert ship("+$5.83") == (5.83, "+$5.83")
    assert ship("Delivery in 2-4 days") == (None, "Delivery in 2-4 days")


def test_site_is_normalised_like_the_other_extractors():
    img = np.full((100, 400, 3), 255, np.uint8)
    img[10:30, 10:310] = 1
    img[40:60, 10:110] = 2
    img[70:90, 10:110] = 3
    dets = [det("listing", 0, 0, 400, 100), det("title", 10, 10, 300, 20), det("price", 10, 40, 100, 20),
            det("sold_date", 10, 70, 100, 20)]
    ocr = FakeOcr({1: "Hockey cards lot", 2: "$12.00", 3: "Sold 05/04/2026"})
    (uk,) = extract_listings(img, detector=FakeDetector(dets), ocr=ocr, site="ebay.co.uk", today=TODAY)
    assert (uk.site, uk.currency, uk.sold_date) == ("www.ebay.co.uk", "GBP", date(2026, 4, 5))  # day first
    (us,) = extract_listings(img, detector=FakeDetector(dets), ocr=ocr, site="https://www.ebay.com/", today=TODAY)
    assert (us.site, us.currency, us.sold_date) == ("www.ebay.com", "USD", date(2026, 5, 4))
    with pytest.raises(ValueError):
        extract_listings(img, detector=FakeDetector(dets), ocr=ocr, site="example.com")


def test_ocr_price_without_its_decimal_point():
    from ebay_sold.vision.extract import _to_listing

    assert _to_listing({"title": "x", "price": "$3718"}, site="www.ebay.com", today=TODAY).price == 37.18


def test_extract_page_rejoins_a_card_cut_by_a_tile_edge_with_a_small_overlap(tmp_path):
    import cv2

    # Card B (y 560..800) is cut by the tile edge; the tiles overlap by only 40 px, so its two
    # pieces share far less than the containment threshold. Its shipping row is only in tile 1.
    page = np.full((1500, 1000, 3), 255, np.uint8)
    dets = _card(page, 0, 100, "A") + _card(page, 0, 560, "B") + _card(page, 0, 1200, "C")
    windows = [(0, 700), (660, 1500)]
    tiles = []
    for i, (y0, y1) in enumerate(windows):
        tile = page[y0:y1].copy()
        tile[0, 0] = (255, i, 255)
        path = tmp_path / f"tile{i}.png"
        cv2.imwrite(str(path), tile)
        tiles.append((path, Box(x=0, y=y0, w=1000, h=y1 - y0)))
    detailed = extract_page_detailed(tiles, detector=FakeDetector(dets, windows), ocr=FakeOcr(_table()), today=TODAY)
    assert len(detailed) == 3
    b = detailed[1]
    assert (b.box.y, b.box.y + b.box.h) == (560, 800)
    assert (b.listing.price, b.listing.shipping, b.listing.condition) == (3.75, 0.0, "Pre-Owned")


# --- scoring: invented fields and layouts with other card shapes -------------------------------------------


def test_score_pair_counts_invented_fields_as_errors():
    from ebay_sold.models import CardRegion, Listing
    from ebay_sold.vision.evaluate import score_pair
    from ebay_sold.vision.extract import ExtractedListing

    region = CardRegion(item_id="1", box=Box(x=0, y=0, w=800, h=250),
                        texts={"price": "$65.00", "sold_date": "Sold Apr 25, 2026", "title": "A catalog"})
    invented = ExtractedListing(listing=Listing(title="A catalog", price=65.0, sold_date=date(2026, 4, 25),
                                                condition="Blue MINT", shipping=0.0),
                                box=Box(x=0, y=0, w=800, h=250))
    scores = score_pair(invented, region, today=TODAY)
    assert scores["condition"] == (False, None, "Blue MINT")
    assert scores["shipping"] == (False, None, 0.0)
    assert scores["price"][0] and scores["title"][0]
    clean = invented.model_copy(deep=True)
    clean.listing.condition = clean.listing.shipping = None
    assert all(ok for ok, _, _ in score_pair(clean, region, today=TODAY).values())


def test_match_cards_falls_back_to_field_positions():
    from ebay_sold.models import CardRegion, Listing
    from ebay_sold.vision.evaluate import match_cards
    from ebay_sold.vision.extract import ExtractedListing

    # Legacy layout: the DOM card spans the whole row, the detector's card box also covers the filter
    # sidebar and stops early (IoU ~0.4), but its fields are inside the DOM card.
    regions = [CardRegion(item_id=str(i), box=Box(x=237, y=314 + 250 * i, w=1113, h=230)) for i in range(2)]
    preds = [ExtractedListing(listing=Listing(title="t"), box=Box(x=0, y=320 + 250 * i, w=810, h=223),
                              fields={"title": Box(x=253, y=339 + 250 * i, w=605, h=18),
                                      "price": Box(x=253, y=400 + 250 * i, w=100, h=24)})
             for i in (1, 0)]
    assert sorted(match_cards(preds, regions)) == [(0, 1), (1, 0)]
    # A card box with no fields inside the DOM card is still not a match.
    stray = ExtractedListing(listing=Listing(title="t"), box=Box(x=0, y=320, w=810, h=223),
                             fields={"title": Box(x=20, y=339, w=150, h=18)})
    assert match_cards([stray], regions) == []


def test_struck_out_asking_price_is_the_original_price_not_the_sale_price():
    # "Best offer accepted": eBay strikes out the asking price and does not show what was paid.
    img = np.full((240, 480, 3), 255, np.uint8)
    img[34:50, 200:440] = 25
    for x in range(200, 270, 10):
        img[100:124, x:x + 7] = 60  # glyphs with gaps between them
    img[111:113, 198:272] = 60  # the strike line, a little wider than the digits
    dets = [det("listing", 0, 0, 480, 240), det("title", 200, 34, 240, 16), det("price", 200, 100, 70, 24)]
    (lst,) = extract_listings(img, detector=FakeDetector(dets), ocr=FakeOcr({25: "Controller", 60: "$65.99"}),
                              today=TODAY)
    assert (lst.price, lst.original_price, lst.currency, lst.price_text) == (None, 65.99, "USD", "$65.99")
