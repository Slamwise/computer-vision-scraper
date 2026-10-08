"""Score the vision pipeline against the DOM on rendered pages.

The DOM is exact, so it is the answer key: each page is rendered offline,
``capture.card_regions`` gives every card's box and visible texts, and the
same screenshot tiles go through ``extract_page``. Predicted cards are matched
to DOM cards by box IoU (or, for a card box of another shape, by where its
fields are); matched pairs are compared field by field after running *both*
sides through ``ebay_sold.normalize``, so formatting differences ("$65.00" vs
"65.00") do not count as errors but misreads do. Every field of every
matched card is scored: a value vision reports for a field the DOM card does
not show (a title line read as the condition) is an error too.

Use pages the detector was not trained on, and a viewport width it never saw,
to get honest numbers.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import BrowserSettings
from ..models import Box, CardRegion
from ..normalize import clean_text, parse_money, parse_shipping, parse_sold_date
from .dataset import page_stem, read_html
from .detector import iou
from .extract import TEXT_FIELDS, ExtractedListing, extract_page_detailed

log = logging.getLogger(__name__)

SCORED_FIELDS = ("price", "sold_date", "shipping", "condition", "title")


class FieldScore(BaseModel):
    total: int = 0  # matched cards (every card is scored; a field the DOM does not show must come out empty)
    correct: int = 0
    missing: int = 0  # the DOM shows the field, vision returned nothing
    extra: int = 0  # vision returned a value for a card whose DOM shows none (an invented field)

    @property
    def accuracy(self) -> float | None:
        return round(self.correct / self.total, 4) if self.total else None


class PageScore(BaseModel):
    page: str
    gt_listings: int
    pred_listings: int
    matched: int
    seconds: float


class EvalReport(BaseModel):
    weights: str
    ocr_backend: str
    viewport_width: int
    device_scale_factor: float = 1.0
    pages: list[PageScore] = Field(default_factory=list)
    gt_listings: int = 0
    pred_listings: int = 0
    matched: int = 0
    precision: float | None = None
    recall: float | None = None
    fields: dict[str, FieldScore] = Field(default_factory=dict)
    price_accuracy: float | None = None
    sold_date_accuracy: float | None = None
    shipping_accuracy: float | None = None
    condition_accuracy: float | None = None
    title_similarity_mean: float | None = None
    title_share_ge_0_9: float | None = None
    seconds: float = 0.0
    failures: list[dict[str, Any]] = Field(default_factory=list)  # examples of wrong fields / unmatched cards

    def summary(self) -> str:
        def pct(v: float | None) -> str:
            return "n/a" if v is None else f"{100 * v:.1f}%"

        def acc(name: str) -> str:
            fs = self.fields.get(name)
            if fs is None:
                return "n/a"
            return pct(fs.accuracy) + (f" ({fs.extra} invented)" if fs.extra else "")

        dsf = f" dsf={self.device_scale_factor:g}" if self.device_scale_factor != 1 else ""
        return (f"width={self.viewport_width}{dsf} ocr={self.ocr_backend}: cards P={pct(self.precision)} "
                f"R={pct(self.recall)} ({self.matched}/{self.gt_listings} matched, {self.pred_listings} predicted); "
                f"price {acc('price')}, sold_date {acc('sold_date')}, "
                f"shipping {acc('shipping')}, condition {acc('condition')}, "
                f"title sim {self.title_similarity_mean if self.title_similarity_mean is not None else 'n/a'} "
                f"(>=0.9: {pct(self.title_share_ge_0_9)})")


def match_cards(preds: list[ExtractedListing], regions: list[CardRegion], *,
                iou_threshold: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of predicted to DOM cards; ``(pred_idx, region_idx)`` pairs.

    A pair matches when the card boxes overlap with IoU >= ``iou_threshold``,
    or when the DOM card contains the centres of most of the predicted card's
    text fields. The second rule keeps the score meaningful on a layout the
    detector was not trained on, where it finds every card and field but draws
    the card box another shape (on the legacy layout it spans the filter
    sidebar and stops before the card's empty right half: IoU ~0.4). IoU
    matches are taken first.
    """
    def inside(x: float, y: float, b: Box) -> bool:
        return b.x <= x <= b.x + b.w and b.y <= y <= b.y + b.h

    pairs = []
    for i, p in enumerate(preds):
        centres = [(b.x + b.w / 2, b.y + b.h / 2) for k, b in p.fields.items() if k in TEXT_FIELDS]
        for j, r in enumerate(regions):
            v = iou(p.box, r.box)
            if v >= iou_threshold:
                pairs.append((1, v, i, j))
            elif centres:
                share = sum(inside(x, y, r.box) for x, y in centres) / len(centres)
                if share > 0.5:
                    pairs.append((0, share + v, i, j))
    used_p, used_r, out = set(), set(), []
    for _, _, i, j in sorted(pairs, reverse=True):
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        out.append((i, j))
    return out


_FOLD = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u201c": '"', "\u201d": '"',
                       "\u2018": "'", "\u2019": "'"})


def _fold(text: str | None) -> str:
    """Case-fold and unify dashes/quotes: OCR cannot tell an en dash from a hyphen at 13 px."""
    return clean_text(text).translate(_FOLD).lower()


def _title_similarity(a: str | None, b: str | None) -> float:
    a, b = _fold(a), _fold(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_pair(pred: ExtractedListing, region: CardRegion, *, today: date,
               currency: str | None = "USD") -> dict[str, tuple[bool, Any, Any]]:
    """Per-field ``(correct, expected, got)`` for one matched card.

    Every field is scored. Where the DOM card does not show a field the
    expected value is ``None``, so a value vision invents (a title line read as
    the condition, "Free returns" read as free shipping) counts as an error.
    """
    lst = pred.listing
    t = region.texts

    def money_ok(exp: float | None, got: float | None) -> bool:
        return (exp is None and got is None) or (exp is not None and got is not None and abs(got - exp) <= 0.01)

    m = parse_money(t.get("price"), currency)
    exp_price = m.amount if m else None
    exp_ship = parse_shipping(t.get("shipping"), currency)
    exp_date = parse_sold_date(t.get("sold_date"), today)
    exp_cond = clean_text(t.get("condition")) or None
    exp_title = clean_text(t.get("title"))
    return {
        "price": (money_ok(exp_price, lst.price), exp_price, lst.price),
        "sold_date": (exp_date == lst.sold_date, exp_date, lst.sold_date),
        "shipping": (money_ok(exp_ship, lst.shipping), exp_ship, lst.shipping),
        "condition": (_fold(exp_cond) == _fold(lst.condition), exp_cond, lst.condition),
        "title": (_title_similarity(exp_title, lst.title) >= 0.9, exp_title, lst.title),
    }


async def evaluate_async(weights: Path | str | None, sources: list[Path], *, viewport_width: int = 1280,
                         ocr_backend: str = "auto", device_scale_factor: float = 1.0, tile_height: int = 1280,
                         overlap: int = 160, load_images: bool = True, iou_threshold: float = 0.5,
                         imgsz: int = 1024, conf: float = 0.25, device: str = "cpu", today: date | None = None,
                         site: str = "www.ebay.com", detector: Any = None, ocr: Any = None,
                         settings: BrowserSettings | None = None, max_failures: int = 60) -> EvalReport:
    """Async version of ``evaluate`` (use it from inside an event loop)."""
    from ..capture import capture_tiles, card_regions, render_html
    from ..urls import SITE_CURRENCY
    from .detector import Detector
    from .ocr import get_ocr

    started = time.monotonic()
    today = today or date.today()
    if detector is None:
        if weights is None:
            raise ValueError("pass weights or a detector")
        detector = Detector(weights, imgsz=imgsz, conf=conf, device=device)
    ocr = ocr or get_ocr(ocr_backend)
    currency = SITE_CURRENCY.get(site)
    report = EvalReport(weights=str(weights or getattr(detector, "weights", "")), ocr_backend=getattr(ocr, "name", ocr_backend),
                        viewport_width=viewport_width, device_scale_factor=device_scale_factor,
                        fields={f: FieldScore() for f in SCORED_FIELDS})
    title_sims: list[float] = []
    settings = settings or BrowserSettings(headless=True)

    for src in sources:
        t0 = time.monotonic()
        name = page_stem(src)
        with tempfile.TemporaryDirectory(prefix="ebay-sold-eval-") as tmp:
            async with render_html(read_html(src), settings=settings, viewport_width=viewport_width,
                                   device_scale_factor=device_scale_factor, load_images=load_images) as page:
                regions = await card_regions(page)
                tiles = await capture_tiles(page, tmp, stem=name, tile_height=tile_height, overlap=overlap)
            preds = extract_page_detailed(tiles, detector=detector, ocr=ocr, site=site, today=today)
        pairs = match_cards(preds, regions, iou_threshold=iou_threshold)
        report.pages.append(PageScore(page=name, gt_listings=len(regions), pred_listings=len(preds),
                                      matched=len(pairs), seconds=round(time.monotonic() - t0, 1)))
        report.gt_listings += len(regions)
        report.pred_listings += len(preds)
        report.matched += len(pairs)
        matched_p = {i for i, _ in pairs}
        matched_r = {j for _, j in pairs}
        for i, j in pairs:
            for fname, (ok, exp, got) in score_pair(preds[i], regions[j], today=today, currency=currency).items():
                fs = report.fields[fname]
                fs.total += 1
                fs.correct += ok
                if exp not in (None, "") and got in (None, ""):
                    fs.missing += 1
                elif exp in (None, "") and got not in (None, ""):
                    fs.extra += 1
                if fname == "title":
                    title_sims.append(_title_similarity(exp, got))
                if not ok and len(report.failures) < max_failures:
                    report.failures.append({
                        "page": name, "field": fname, "expected": str(exp), "got": str(got),
                        "dom_text": regions[j].texts.get(fname), "ocr_text": preds[i].texts.get(fname),
                        "item_id": regions[j].item_id,
                    })
        for j, r in enumerate(regions):
            if j not in matched_r and len(report.failures) < max_failures:
                report.failures.append({"page": name, "field": "listing", "expected": "card", "got": "missed",
                                        "item_id": r.item_id, "box": r.box.model_dump()})
        for i, p in enumerate(preds):
            if i not in matched_p and len(report.failures) < max_failures:
                report.failures.append({"page": name, "field": "listing", "expected": "nothing",
                                        "got": "extra card", "box": p.box.model_dump(), "ocr_text": p.texts})

    report.precision = round(report.matched / report.pred_listings, 4) if report.pred_listings else None
    report.recall = round(report.matched / report.gt_listings, 4) if report.gt_listings else None
    report.price_accuracy = report.fields["price"].accuracy
    report.sold_date_accuracy = report.fields["sold_date"].accuracy
    report.shipping_accuracy = report.fields["shipping"].accuracy
    report.condition_accuracy = report.fields["condition"].accuracy
    if title_sims:
        report.title_similarity_mean = round(sum(title_sims) / len(title_sims), 4)
        report.title_share_ge_0_9 = round(sum(s >= 0.9 for s in title_sims) / len(title_sims), 4)
    report.seconds = round(time.monotonic() - started, 1)
    return report


def evaluate(weights: Path | str | None, sources: list[Path], *, viewport_width: int = 1280,
             ocr_backend: str = "auto", **kwargs: Any) -> EvalReport:
    """Render ``sources``, run detector + OCR on the screenshot tiles and score against the DOM.

    Reports card precision/recall (see ``match_cards``) and, for every matched
    card, price (+-0.01), sold date, shipping (exact), condition (exact up to
    case and dash/quote style) and title (mean similarity; share >= 0.9). A
    field the DOM card does not show must come out empty to count as correct.
    Other keyword arguments are those of ``evaluate_async``.
    """
    return asyncio.run(evaluate_async(weights, sources, viewport_width=viewport_width, ocr_backend=ocr_backend,
                                      **kwargs))
