"""OCR engines on synthetic text crops, and the line / word-gap helpers."""

from __future__ import annotations

from datetime import date

import pytest

np = pytest.importorskip("numpy")

from ebay_sold.normalize import parse_money, parse_shipping, parse_sold_date  # noqa: E402
from ebay_sold.vision.ocr import (  # noqa: E402
    OcrEngine,
    OcrResult,
    get_ocr,
    ink_mask,
    tesseract_available,
    text_lines,
    word_gaps,
)


def _render(text: str, *, size: int = 15, color=(30, 30, 30), bg=(255, 255, 255), lines: int = 1):
    """Render text roughly the way eBay draws it: small, anti-aliased, on a light background (BGR)."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=size)
    parts = text.split("\n")
    w = int(max(font.getlength(p) for p in parts)) + 12
    h = (size + 6) * len(parts) + 6
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for i, part in enumerate(parts):
        draw.text((6, 4 + i * (size + 6)), part, font=font, fill=color)
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def test_ink_mask_and_lines_on_synthetic_bars():
    img = np.full((40, 100, 3), 255, np.uint8)
    img[5:15, 10:30] = 0  # word 1, line 1
    img[5:15, 32:60] = 0  # 2 px gap at line height 10: a letter gap
    img[5:15, 70:90] = 0  # 10 px gap: a word gap
    img[25:35, 10:50] = (0, 160, 0)  # a green second line
    mask = ink_mask(img)
    assert text_lines(mask) == [(5, 15), (25, 35)]
    assert word_gaps(mask[5:15]) == [(60, 70)]
    assert word_gaps(mask[5:15], gap_ratio=0.15) == [(30, 32), (60, 70)]
    # light text on a dark background
    inv = 255 - img
    assert text_lines(ink_mask(inv)) == [(5, 15), (25, 35)]


def test_get_ocr_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_ocr("paddle")


@pytest.mark.vision
def test_rapidocr_reads_price_and_date(require_vision):
    pytest.importorskip("rapidocr_onnxruntime")
    ocr = get_ocr("rapidocr")
    assert isinstance(ocr, OcrEngine)
    price = ocr.read(_render("$65.00", size=20, color=(0, 120, 0)))
    assert isinstance(price, OcrResult) and 0 < price.conf <= 1
    assert parse_money(price.text).amount == 65.0
    sold = ocr.read(_render("Sold Apr 25, 2026", size=14, color=(0, 110, 0)))
    assert parse_sold_date(sold.text, date(2026, 10, 1)) == date(2026, 4, 25)
    # Spaces survive (the raw RapidOCR pipeline returns "SoldApr25,2026").
    assert sold.text.replace(" ", "") == "SoldApr25,2026" and sold.text.count(" ") >= 2
    ship = ocr.read(_render("Free delivery", size=14, color=(90, 90, 90)))
    assert parse_shipping(ship.text) == 0.0
    two = ocr.read(_render("Hot Wheels 2013 Zamac\nNissan Skyline R34", size=15))
    assert "Wheels" in two.text and "Skyline" in two.text
    assert ocr.read(np.full((20, 60, 3), 255, np.uint8)).text == ""


@pytest.mark.vision
def test_tesseract_reads_price_and_date(require_vision):
    if not tesseract_available():
        pytest.skip("tesseract binary / pytesseract not installed")
    ocr = get_ocr("tesseract")
    assert parse_money(ocr.read(_render("$65.00", size=20, color=(0, 120, 0))).text).amount == 65.0
    sold = ocr.read(_render("Sold Apr 25, 2026", size=14, color=(0, 110, 0))).text
    assert parse_sold_date(sold, date(2026, 10, 1)) == date(2026, 4, 25)


@pytest.mark.vision
def test_auto_prefers_tesseract_then_rapidocr(require_vision):
    pytest.importorskip("rapidocr_onnxruntime")
    assert get_ocr("auto").name == ("tesseract" if tesseract_available() else "rapidocr")


@pytest.mark.vision
def test_tesseract_read_many_matches_read(require_vision):
    if not tesseract_available():
        pytest.skip("tesseract binary / pytesseract not installed")
    from ebay_sold.vision.ocr import read_many

    ocr = get_ocr("tesseract")
    crops = [_render(t, size=16) for t in ("$1.00", "Pre-Owned", "Free delivery", "$23.95")]
    assert [r.text for r in read_many(ocr, crops)] == [ocr.read(c).text for c in crops]
