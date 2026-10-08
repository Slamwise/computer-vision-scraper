"""Score the vision pipeline against the DOM on rendered pages.

The DOM is exact, so it is the answer key: each page is rendered offline,
``capture.card_regions`` gives every card's box and visible texts, and the
same screenshot tiles go through ``extract_page``. Predicted cards are matched
to DOM cards by box IoU; matched pairs are compared field by field after
running *both* sides through ``ebay_sold.normalize``, so formatting
differences ("$65.00" vs "65.00") do not count as errors but misreads do.

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
from ..models import CardRegion
from ..normalize import clean_text, parse_money, parse_shipping, parse_sold_date
from .dataset import page_stem, read_html
from .detector import iou
from .extract import ExtractedListing, extract_page_detailed

log = logging.getLogger(__name__)

SCORED_FIELDS = ("price", "sold_date", "shipping", "condition", "title")


class FieldScore(BaseModel):
    total: int = 0  # matched cards whose DOM shows this field
    correct: int = 0
    missing: int = 0  # field not detected / not read at all

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

        return (f"width={self.viewport_width} ocr={self.ocr_backend}: cards P={pct(self.precision)} "
                f"R={pct(self.recall)} ({self.matched}/{self.gt_listings} matched, {self.pred_listings} predicted); "
                f"price {pct(self.price_accuracy)}, sold_date {pct(self.sold_date_accuracy)}, "
                f"shipping {pct(self.shipping_accuracy)}, condition {pct(self.condition_accuracy)}, "
                f"title sim {self.title_similarity_mean if self.title_similarity_mean is not None else 'n/a'} "
                f"(>=0.9: {pct(self.title_share_ge_0_9)})")


def match_cards(preds: list[ExtractedListing], regions: list[CardRegion], *,
                iou_threshold: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of predicted to DOM cards by box IoU; ``(pred_idx, region_idx)`` pairs."""
    pairs = []
    for i, p in enumerate(preds):
        for j, r in enumerate(regions):
            v = iou(p.box, r.box)
            if v >= iou_threshold:
                pairs.append((v, i, j))
    used_p, used_r, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
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
    """Per-field ``(correct, expected, got)`` for one matched card (only fields the DOM shows)."""
    lst = pred.listing
    t = region.texts
    out: dict[str, tuple[bool, Any, Any]] = {}
    if "price" in t:
        m = parse_money(t["price"], currency)
        exp = m.amount if m else None
        ok = (exp is None and lst.price is None) or (
            exp is not None and lst.price is not None and abs(lst.price - exp) <= 0.01)
        out["price"] = (ok, exp, lst.price)
    if "sold_date" in t:
        exp = parse_sold_date(t["sold_date"], today)
        out["sold_date"] = (exp == lst.sold_date, exp, lst.sold_date)
    if "shipping" in t:
        exp = parse_shipping(t["shipping"], currency)
        ok = (exp is None and lst.shipping is None) or (
            exp is not None and lst.shipping is not None and abs(lst.shipping - exp) <= 0.01)
        out["shipping"] = (ok, exp, lst.shipping)
    if "condition" in t:
        exp = clean_text(t["condition"]) or None
        got = lst.condition
        out["condition"] = (_fold(exp) == _fold(got), exp, got)
    if "title" in t:
        exp = clean_text(t["title"])
        sim = _title_similarity(exp, lst.title)
        out["title"] = (sim >= 0.9, exp, lst.title)
    return out


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
                if fname not in preds[i].texts:
                    fs.missing += 1
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

    Reports card precision/recall (IoU >= 0.5) and, for matched cards, price
    (+-0.01), sold date, shipping (exact), condition (exact up to case and
    dash/quote style) and title (mean similarity; share >= 0.9). Other keyword arguments are those
    of ``evaluate_async``.
    """
    return asyncio.run(evaluate_async(weights, sources, viewport_width=viewport_width, ocr_backend=ocr_backend,
                                      **kwargs))
