"""Screenshots -> ``Listing`` objects: detect cards and fields, OCR them, normalise.

The pipeline per image (or per tiled page):

1. ``Detector.detect`` finds ``listing`` boxes (whole cards) and field boxes
   (title, price, sold_date, shipping, condition, image).
2. Each field goes to the card that contains most of it. A crop of a single
   card usually has no ``listing`` box of its own, so fields without any card
   box are treated as one card.
3. Titles are checked against the pixels. eBay draws a title in near-black ink
   and everything around it (condition, shipping, sold date, price) in grey or
   colour, while the detector sometimes boxes only the first line of a wrapped
   title, or calls the second line a "condition". A "condition" drawn in title
   ink directly under the title is folded back into it, and a title box is
   extended over following lines in the same ink.
4. Field crops are cut from the *original* pixels with a few pixels of margin
   (detector boxes are a little tight or loose) and read by an ``OcrEngine``.
5. Text becomes typed values only through ``ebay_sold.normalize``, so vision
   output is parsed exactly like DOM output. Rows the detector mistakes for
   shipping ("Free returns", "or Best Offer") are not read as a shipping cost,
   and a struck-out price (an accepted best offer) is the original price, not
   the sale price, as in the DOM parser.

For a page captured as overlapping tiles (``capture.capture_tiles``),
``extract_page`` detects on every tile, maps the boxes into one page-wide
pixel space, merges the duplicates the overlaps create (a card cut by a tile
edge is re-joined from its two halves), and reads every field from a tile that
shows it whole.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from pydantic import BaseModel, Field

from ..models import Box, Listing
from ..normalize import clean_text, parse_money, parse_shipping, parse_sold_date
from ..urls import SITE_CURRENCY, normalize_site
from .detector import Detection, containment, load_image, merge_detections, union_box
from .ocr import ink_mask, read_many, struck_through, text_lines

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .detector import ImageInput
    from .ocr import OcrEngine

log = logging.getLogger(__name__)

TEXT_FIELDS: tuple[str, ...] = ("title", "price", "sold_date", "shipping", "condition")
# Sites whose all-numeric dates are day-first.
_DAY_FIRST_SITES = {"www.ebay.co.uk", "www.ebay.com.au", "www.ebay.de", "www.ebay.fr", "www.ebay.it", "www.ebay.es"}


class DetectorLike(Protocol):
    """``Detector`` or a stand-in. ``detect`` may also accept ``scale=`` (image px per CSS px)."""

    def detect(self, image: Any) -> list[Detection]: ...


def _detect(detector: DetectorLike, image: Any, scale: float | None) -> list[Detection]:
    """``detector.detect``, passing the known scale when the detector accepts it."""
    import inspect

    if scale is not None:
        try:
            params = inspect.signature(detector.detect).parameters
        except (TypeError, ValueError):
            params = {}
        if "scale" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return detector.detect(image, scale=scale)
    return detector.detect(image)


class ExtractedListing(BaseModel):
    """A listing plus where it was found (for scoring and debugging)."""

    listing: Listing
    box: Box  # page CSS px when a page offset is known, else image px
    fields: dict[str, Box] = Field(default_factory=dict)
    texts: dict[str, str] = Field(default_factory=dict)  # raw OCR text per field
    field_conf: dict[str, float] = Field(default_factory=dict)  # detector conf * OCR conf
    synthetic_box: bool = False  # no card box was detected; the whole image is the card


# --- grouping ------------------------------------------------------------------------


def assign_fields(dets: Sequence[Detection], *, min_containment: float = 0.5,
                  image_size: tuple[int, int] | None = None,
                  allow_synthetic: bool = True) -> list[tuple[Detection, dict[str, Detection]]]:
    """Group field detections under card (``listing``) detections.

    Each field goes to the card containing the largest share of it (at least
    ``min_containment``). Two boxes of one class in one card that overlap are
    two partial readings of one line (the detector can split a long title in
    two) and are joined; otherwise the most confident wins.
    Without any card box (a crop of a single card), all fields form one card
    spanning ``image_size`` (``(width, height)``) or the fields' union, unless
    ``allow_synthetic`` is false.
    """
    cards = [d for d in dets if d.cls == "listing"]
    fields = [d for d in dets if d.cls != "listing"]
    if not cards:
        if not fields or not allow_synthetic:
            return []
        if image_size is not None:
            box = Box(x=0, y=0, w=float(image_size[0]), h=float(image_size[1]))
        else:
            x1 = min(f.box.x for f in fields)
            y1 = min(f.box.y for f in fields)
            x2 = max(f.box.x + f.box.w for f in fields)
            y2 = max(f.box.y + f.box.h for f in fields)
            box = Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        conf = sum(f.conf for f in fields) / len(fields)
        cards = [Detection(cls="listing", conf=conf, box=box)]
        synthetic = True
    else:
        synthetic = False
    groups: list[dict[str, Detection]] = [{} for _ in cards]
    for f in fields:
        if f.box.area <= 0:
            continue
        best_i, best_c = -1, 0.0
        for i, card in enumerate(cards):
            c = f.box.intersection(card.box) / f.box.area
            if c > best_c:
                best_i, best_c = i, c
        if best_i < 0 or (best_c < min_containment and not synthetic):
            continue
        current = groups[best_i].get(f.cls)
        if current is None:
            groups[best_i][f.cls] = f
        elif f.cls != "image" and f.box.intersection(current.box) > 0:
            groups[best_i][f.cls] = Detection(cls=f.cls, conf=max(f.conf, current.conf),
                                              box=union_box(f.box, current.box))
        elif f.conf > current.conf:
            groups[best_i][f.cls] = f
    return list(zip(cards, groups))


def reading_order(boxes: Sequence[Box]) -> list[int]:
    """Indices of ``boxes`` in reading order: rows top to bottom, left to right within a row."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].y, boxes[i].x))
    rows: list[list[int]] = []
    for i in order:
        b = boxes[i]
        if rows:
            ref = boxes[rows[-1][0]]
            overlap = min(ref.y + ref.h, b.y + b.h) - max(ref.y, b.y)
            if overlap > 0.5 * min(ref.h, b.h):
                rows[-1].append(i)
                continue
        rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda j: boxes[j].x)]


# --- reading pixels -------------------------------------------------------------------


class _Pixels:
    """Crop source: one image, or several horizontal tiles of one page."""

    def __init__(self, strips: list[tuple[int, "np.ndarray"]]) -> None:
        self.strips = strips  # (y offset in page pixels, BGR array)
        self.width = max(a.shape[1] for _, a in strips)
        self.height = max(y + a.shape[0] for y, a in strips)

    def crop(self, box: Box, pad: int = 3, pad_x: int | None = None) -> "np.ndarray | None":
        found = self.crop_at(box, pad, pad_x)
        return None if found is None else found[0]

    def crop_at(self, box: Box, pad: int = 3, pad_x: int | None = None) -> "tuple[np.ndarray, int, int] | None":
        """``(crop, x0, y0)``: the pixels around ``box`` and where the crop's top-left corner is."""
        pad_x = pad if pad_x is None else pad_x
        x0, y0 = int(max(0, box.x - pad_x)), int(max(0, box.y - pad))
        x1 = int(min(self.width, box.x + box.w + pad_x + 0.999))
        y1 = int(min(self.height, box.y + box.h + pad + 0.999))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        # Prefer a strip that shows the whole box, with the most margin around it.
        best, best_key = None, None
        for off, arr in self.strips:
            top, bottom = y0 - off, y1 - off
            inside = min(bottom, arr.shape[0]) - max(top, 0)
            whole = top >= 0 and bottom <= arr.shape[0]
            margin = min(top, arr.shape[0] - bottom)
            key = (whole, inside, margin)
            if best_key is None or key > best_key:
                best, best_key = (off, arr), key
        if best is None or best_key[1] < 2:
            return None
        off, arr = best
        top = max(0, y0 - off)
        return arr[top:min(arr.shape[0], y1 - off), x0:min(x1, arr.shape[1])], x0, top + off


def _trim_against(box: Box, others: Sequence[Box]) -> Box:
    """Shrink ``box`` vertically so it stops short of the card's other text fields.

    A title box that reaches into the condition line below it would otherwise
    be OCR'd as "<title> Pre-Owned".
    """
    top, bottom = box.y, box.y + box.h
    cy = box.y + box.h / 2
    for o in others:
        if min(box.x + box.w, o.x + o.w) - max(box.x, o.x) <= 0:
            continue  # not in the same column
        o_cy = o.y + o.h / 2
        if o_cy > cy and o.y < bottom:
            bottom = max(o.y, cy + 1)
        elif o_cy < cy and o.y + o.h > top:
            top = min(o.y + o.h, cy - 1)
    return Box(x=box.x, y=top, w=box.w, h=max(1.0, bottom - top))


def ink_style(image: "np.ndarray | None") -> tuple[float, float] | None:
    """``(contrast, saturation)`` of a crop's text pixels (medians, 0-255); ``None`` if there is no text.

    Tells eBay's text styles apart: titles are near-black (contrast ~170 on
    white, unsaturated), conditions and shipping lines grey (~110), prices and
    sold dates green (saturation ~175).
    """
    import cv2
    import numpy as np

    if image is None or image.size == 0:
        return None
    mask = ink_mask(image)
    if mask.sum() < 4:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg = float(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(np.median(np.abs(gray - bg)[mask])), float(np.median(sat[mask]))


def _same_ink(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    """Same text style: contrast within 20 % and the same colourfulness."""
    if a is None or b is None:
        return False
    return min(a[0], b[0]) >= 0.8 * max(a[0], b[0]) and abs(a[1] - b[1]) <= 60


def _below(title: Box, line: Box, line_h: float) -> bool:
    """``line`` is the title's next line: left-aligned with it, starting inside it or right under it."""
    return (abs(line.x - title.x) <= 1.5 * line_h
            and line.y + line.h / 2 > title.y + 0.5 * line_h
            and line.y < title.y + title.h + 0.75 * line_h)


def _absorb_title_lines(pixels: "_Pixels", dets: Sequence[Detection]) -> list[Detection]:
    """Fold "condition" boxes that are really a title's next line back into the title.

    Runs before fields are grouped by card, so a card's real (grey) condition
    line is still there to be picked once the false one is gone.
    """
    out = list(dets)
    titles = [i for i, d in enumerate(out) if d.cls == "title"]
    drop: set[int] = set()
    for ci, cond in enumerate(dets):
        if cond.cls != "condition" or not titles:
            continue
        near = [ti for ti in titles if _below(out[ti].box, cond.box, cond.box.h)]
        if not near:
            continue
        ti = min(near, key=lambda i: abs(cond.box.y - out[i].box.y - out[i].box.h))
        title = out[ti]
        if not _same_ink(ink_style(pixels.crop(title.box, pad=2)), ink_style(pixels.crop(cond.box, pad=2))):
            continue  # a real condition line (grey), maybe overlapped by a loose title box
        out[ti] = title.model_copy(update={"box": union_box(title.box, cond.box)})
        drop.add(ci)
    return [d for i, d in enumerate(out) if i not in drop]


def _adjacent_line(pixels: "_Pixels", box: Box, others: Sequence[Box], line_h: float, ink: tuple[float, float],
                   *, up: bool, right: float | None = None) -> Box | None:
    """The text line right below (``up=False``) or above ``box`` if it is drawn the same way, else ``None``.

    Same way: starts at the box's left edge, at line spacing, a text line's
    height, the same ink, and not part of another detected field. The line may
    be longer than ``box`` (the first line of a wrapped title is); it is
    followed rightwards up to ``right`` (the card's edge) until a gap wider
    than a word space.
    """
    x0 = box.x - 0.5 * line_h
    width = max(box.w + line_h, (right - x0) if right is not None else 0.0)
    region = Box(x=x0, y=box.y - 2.0 * line_h if up else box.y + box.h, w=width, h=2.0 * line_h)
    found = pixels.crop_at(region, pad=0)
    if found is None:
        return None
    crop, cx, cy = found
    mask = ink_mask(crop)
    # Find the line in the box's own column (other columns of the card may hold text at other heights);
    # skip slivers of the box's own glyphs (descenders, accents) poking out of a tight box.
    column = mask[:, :max(1, int(round(box.x + box.w + 0.5 * line_h - cx)))]
    edge = crop.shape[0] if up else 0
    lines = [(a, b) for a, b in text_lines(column) if not ((b if up else a) == edge and b - a < 0.5 * line_h)]
    if not lines:
        return None
    y0, y1 = lines[-1] if up else lines[0]
    # Ink of a following line starts ~0.6 line heights below a line's box (line spacing + ascender room).
    gap = box.y - (cy + y1) if up else cy + y0 - (box.y + box.h)
    if gap > 0.9 * line_h or not 0.5 * line_h <= y1 - y0 <= 1.5 * line_h:
        return None
    cols = mask[y0:y1].any(axis=0).nonzero()[0]
    gaps = (cols[1:] - cols[:-1] > 1.5 * line_h).nonzero()[0]
    end = cols[gaps[0]] if len(gaps) else cols[-1]  # stop at another column of text
    line = Box(x=float(cx + cols[0]), y=float(cy + y0), w=float(end - cols[0] + 1), h=float(y1 - y0))
    if abs(line.x - box.x) > line_h or any(o.intersection(line) > 0.3 * line.area for o in others):
        return None
    return line if _same_ink(ink, ink_style(pixels.crop(line, pad=2))) else None


def _extend_title(pixels: "_Pixels", title: Box, others: Sequence[Box], line_h: float, *,
                  max_lines: int = 2, right: float | None = None) -> Box:
    """Grow a title box over adjacent lines drawn in the title's ink.

    The detector sometimes boxes only one line of a wrapped title. A line
    belongs to the title when it starts at the title's left edge, follows at
    line spacing, has a text line's height, is drawn in the same near-black
    ink and is not another detected field. eBay draws the sold date (green)
    above a title and the subtitle / condition (grey) below it, so the growth
    stops there.
    """
    if line_h <= 0:
        return title
    ink = ink_style(pixels.crop(title, pad=2))
    if ink is None:
        return title
    for up, limit in ((False, max_lines), (True, 1)):
        for _ in range(limit):
            line = _adjacent_line(pixels, title, others, line_h, ink, up=up, right=right)
            if line is None:
                break
            title = union_box(title, line)
    return title


def _last_subtitle_line(pixels: "_Pixels", cond: Box, others: Sequence[Box], line_h: float, *,
                        max_lines: int = 3) -> Box:
    """Move a condition box down to the last line of the grey subtitle block it is in.

    Sellers can add tagline subtitles ("FREE AND FAST SHIPPING") in the same
    grey as the condition, and eBay always draws the condition line last
    (the DOM parser relies on the same rule), so a box on a tagline moves to
    the line below it.
    """
    ink = ink_style(pixels.crop(cond, pad=2)) if line_h > 0 else None
    if ink is None:
        return cond
    for _ in range(max_lines):
        line = _adjacent_line(pixels, cond, others, line_h, ink, up=False)
        if line is None:
            break
        cond = line
    return cond


# --- conversion -----------------------------------------------------------------------

# eBay prefixes fresh listings with a "NEW LISTING" tag on the title line; the DOM
# parser leaves it out of the title, so vision does too.
_NEW_LISTING_RE = re.compile(r"^\s*new\s*listing\b[\s:-]*", re.IGNORECASE)
# Rows the detector can mistake for the shipping line. "Free returns" must not
# become free shipping (capture.card_regions skips these rows for the same reason).
_RETURNS_RE = re.compile(r"\breturns?\b|r[üu]ckgabe|\bretours?\b|\bresi\b|devoluci", re.IGNORECASE)
_SHIPPING_WORD_RE = re.compile(r"deliver|shipping|postage|versand|livraison|spedizione|env[ií]o|lieferung",
                               re.IGNORECASE)


def _shipping(text: str, currency: str | None) -> tuple[float | None, str | None]:
    """``(cost, text)`` for an OCR'd shipping box; ``(None, None)`` when it is not a shipping line."""
    text = clean_text(text)
    if not text or (_RETURNS_RE.search(text) and not _SHIPPING_WORD_RE.search(text)):
        return None, None
    cost = parse_shipping(text, currency, ocr=True)
    if cost is None and not _SHIPPING_WORD_RE.search(text):
        return None, None  # "or Best Offer", "from Canada": some other row
    return cost, text


def _to_listing(texts: dict[str, str], *, site: str, today: date | None, struck_price: bool = False) -> Listing:
    currency = SITE_CURRENCY.get(site)
    title = _NEW_LISTING_RE.sub("", clean_text(texts.get("title")))
    data: dict[str, Any] = {"site": site, "title": title, "extraction": "vision", "item_id": None}
    if "price" in texts:
        data["price_text"] = clean_text(texts["price"]) or None
        money = parse_money(texts["price"], currency, ocr=True)
        if money and struck_price:
            # An accepted best offer: eBay strikes out the asking price and does not
            # show what was paid. Like the DOM parser, keep it as the original price.
            data.update(original_price=money.amount, currency=money.currency)
        elif money:
            data.update(price=money.amount, price_max=money.amount_max, currency=money.currency)
    if "shipping" in texts:
        data["shipping"], data["shipping_text"] = _shipping(texts["shipping"], currency)
    if "sold_date" in texts:
        data["sold_date_text"] = clean_text(texts["sold_date"]) or None
        data["sold_date"] = parse_sold_date(texts["sold_date"], today, day_first=site in _DAY_FIRST_SITES)
    if "condition" in texts:
        data["condition"] = clean_text(texts["condition"]) or None
    if data.get("currency") is None and (data.get("price") is not None or data.get("original_price") is not None):
        data["currency"] = currency
    return Listing(**data)


def _confidence(card_conf: float, field_conf: dict[str, float], listing: Listing, texts: dict[str, str]) -> float:
    key = [field_conf[k] for k in ("title", "price", "sold_date") if k in field_conf]
    conf = card_conf * (sum(key) / len(key) if key else 0.5)
    if ("price" in texts and listing.price is None and listing.original_price is None
            and "see price" not in texts["price"].lower()):
        conf *= 0.5  # a price box we could not parse
    if "price" not in texts:
        conf *= 0.7
    return round(max(0.0, min(1.0, conf)), 3)


def _extract(pixels: _Pixels, dets: list[Detection], *, ocr: "OcrEngine", site: str, today: date | None,
             scale: float, offset: tuple[float, float], pad: int, image_size: tuple[int, int] | None,
             allow_synthetic: bool) -> list[ExtractedListing]:
    out: list[ExtractedListing] = []
    ox, oy = offset
    synthetic = not any(d.cls == "listing" for d in dets)

    def to_out(b: Box) -> Box:
        return Box(x=ox + b.x / scale, y=oy + b.y / scale, w=b.w / scale, h=b.h / scale)

    groups = assign_fields(_absorb_title_lines(pixels, dets), image_size=image_size,
                           allow_synthetic=allow_synthetic)
    # Cut every crop first and OCR them in one call, so engines can work in parallel.
    jobs: list[tuple[int, str, float]] = []
    crops = []
    struck: set[int] = set()
    for gi, (card, fields) in enumerate(groups):
        text_boxes = [f.box for k, f in fields.items() if k in TEXT_FIELDS]
        # The shortest text box is one line tall (titles can be two).
        line_h = min((b.h for b in text_boxes), default=0.0)
        if "title" in fields:
            others = [f.box for k, f in fields.items() if k != "title"]
            grown = _extend_title(pixels, fields["title"].box, others, line_h, right=card.box.x + card.box.w)
            fields["title"] = fields["title"].model_copy(update={"box": grown})
        if "condition" in fields:
            others = [f.box for k, f in fields.items() if k != "condition"]
            moved = _last_subtitle_line(pixels, fields["condition"].box, others, line_h)
            fields["condition"] = fields["condition"].model_copy(update={"box": moved})
        groups[gi] = (card, fields)
        for name in TEXT_FIELDS:
            det = fields.get(name)
            if det is None:
                continue
            box = _trim_against(det.box, [f.box for k, f in fields.items() if k != name and k in TEXT_FIELDS])
            # Boxes are most often a glyph short at the ends of long lines; widen
            # horizontally by half a line height (eBay leaves more space than that
            # between a field and the product photo or seller column).
            crop = pixels.crop(box, pad=pad, pad_x=max(pad, int(round(0.5 * line_h))))
            if crop is not None:
                jobs.append((gi, name, det.conf))
                crops.append(crop)
                if name == "price" and struck_through(crop):
                    struck.add(gi)
    read: list[tuple[dict[str, str], dict[str, float]]] = [({}, {}) for _ in groups]
    for (gi, name, det_conf), res in zip(jobs, read_many(ocr, crops)):
        if res.text.strip():
            read[gi][0][name] = res.text
            read[gi][1][name] = round(det_conf * res.conf, 4)

    for gi, ((card, fields), (texts, fconf)) in enumerate(zip(groups, read)):
        listing = _to_listing(texts, site=site, today=today, struck_price=gi in struck)
        if listing.price is None and listing.original_price is None and not listing.title:
            continue
        listing.confidence = _confidence(card.conf, fconf, listing, texts)
        out.append(ExtractedListing(
            listing=listing, box=to_out(card.box), fields={k: to_out(v.box) for k, v in fields.items()},
            texts=texts, field_conf=fconf, synthetic_box=synthetic,
        ))
    return out


def _dedupe(items: list[ExtractedListing], *, threshold: float = 0.6) -> list[ExtractedListing]:
    """Drop listings that are the same card seen twice; keep the more complete one."""
    def completeness(e: ExtractedListing) -> tuple[int, float]:
        lst = e.listing
        filled = sum(v is not None and v != "" for v in (lst.title, lst.price, lst.sold_date, lst.shipping, lst.condition))
        return filled, lst.confidence or 0.0

    kept: list[ExtractedListing] = []
    for e in sorted(items, key=completeness, reverse=True):
        if any(containment(e.box, k.box) >= threshold for k in kept):
            continue
        kept.append(e)
    return kept


def _finish(items: list[ExtractedListing]) -> list[ExtractedListing]:
    items = _dedupe(items)
    ordered = [items[i] for i in reading_order([e.box for e in items])]
    for pos, e in enumerate(ordered, start=1):
        e.listing.position = pos
    return ordered


# --- public API ---------------------------------------------------------------------------


def extract_listings_detailed(image: "ImageInput", *, detector: DetectorLike, ocr: "OcrEngine",
                              site: str = "www.ebay.com", today: date | None = None,
                              page_offset: Box | None = None, pad: int = 3) -> list[ExtractedListing]:
    """Like ``extract_listings`` but also returns boxes, raw OCR text and per-field confidence."""
    site = normalize_site(site)
    arr = load_image(image)
    h, w = arr.shape[:2]
    known = page_offset is not None and page_offset.w > 0
    scale = (w / page_offset.w) if known else 1.0
    dets = _detect(detector, arr, scale if known else None)
    offset = (page_offset.x, page_offset.y) if page_offset is not None else (0.0, 0.0)
    items = _extract(_Pixels([(0, arr)]), dets, ocr=ocr, site=site, today=today, scale=scale, offset=offset,
                     pad=pad, image_size=(w, h), allow_synthetic=True)
    return _finish(items)


def extract_listings(image: "ImageInput", *, detector: DetectorLike, ocr: "OcrEngine", site: str = "www.ebay.com",
                     today: date | None = None, page_offset: Box | None = None) -> list[Listing]:
    """Read every sold listing visible in one screenshot (a page tile, a full page or a single card).

    ``page_offset`` is where the image sits on the page in CSS pixels (as
    returned by ``capture.capture_tiles``); it gives the reported boxes of
    ``extract_listings_detailed`` page coordinates and tells the detector the
    image's scale (otherwise measured from the image). ``site`` takes any form
    ``urls.normalize_site`` accepts ("ebay.co.uk"). Results have ``extraction="vision"``,
    ``item_id=None`` and 1-based ``position`` in reading order.
    """
    return [e.listing for e in extract_listings_detailed(image, detector=detector, ocr=ocr, site=site,
                                                         today=today, page_offset=page_offset)]


def extract_page_detailed(tiles: Sequence[tuple[Path | str, Box]], *, detector: DetectorLike, ocr: "OcrEngine",
                          site: str = "www.ebay.com", today: date | None = None,
                          pad: int = 3) -> list[ExtractedListing]:
    """Like ``extract_page`` but returns boxes (page CSS px) and raw OCR text too."""
    site = normalize_site(site)
    if not tiles:
        return []
    strips: list[tuple[int, np.ndarray]] = []
    dets: list[Detection] = []
    scale = None
    page_bottom = max(b.y + b.h for _, b in tiles)
    for path, tile_box in tiles:
        arr = load_image(path)
        s = arr.shape[1] / tile_box.w if tile_box.w > 0 else 1.0
        scale = scale or s
        y_off = int(round(tile_box.y * scale))
        strips.append((y_off, arr))
        # Pieces of a card cut by this tile's top or bottom edge are re-joined with
        # the neighbouring tile's pieces by merge_detections.
        tol = max(2.0, 0.01 * arr.shape[0])
        first, last = tile_box.y <= 0, tile_box.y + tile_box.h >= page_bottom
        for d in _detect(detector, arr, scale):
            b = d.box
            dets.append(Detection(cls=d.cls, conf=d.conf, box=Box(x=b.x, y=b.y + y_off, w=b.w, h=b.h),
                                  cut_top=d.cut_top or (not first and b.y <= tol),
                                  cut_bottom=d.cut_bottom or (not last and b.y + b.h >= arr.shape[0] - tol)))
    merged = merge_detections(dets) if len(tiles) > 1 else dets
    pixels = _Pixels(strips)
    items = _extract(pixels, merged, ocr=ocr, site=site, today=today, scale=scale or 1.0,
                     offset=(tiles[0][1].x, 0.0), pad=pad, image_size=None, allow_synthetic=False)
    return _finish(items)


def extract_page(tiles: Sequence[tuple[Path | str, Box]], *, detector: DetectorLike, ocr: "OcrEngine",
                 site: str = "www.ebay.com", today: date | None = None) -> list[Listing]:
    """Read a page captured as overlapping tiles (``(png_path, tile_box_in_page_css_px)`` pairs).

    Cards seen in two tiles are reported once (the more complete reading).
    """
    return [e.listing for e in extract_page_detailed(tiles, detector=detector, ocr=ocr, site=site, today=today)]
