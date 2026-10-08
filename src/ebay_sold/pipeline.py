"""End-to-end flows: search -> fetch -> extract -> store.

Extraction order for each page:

1. DOM parse of the HTML (exact, free).
2. If the DOM parser finds no cards on a page that is not a challenge, eBay
   probably changed its markup. Then the YOLO + OCR model reads screenshots.
   Screenshots come from re-rendering the *cached* HTML offline, so the
   fallback never costs another request to eBay.
3. Optionally, Claude reads the same screenshots if vision also finds nothing.
"""

from __future__ import annotations

import gzip
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .db import Database
from .models import Box, Listing, PageResult, SearchQuery
from .normalize import clean_text
from .parse import parse_search_page
from .urls import SITE_CURRENCY, normalize_site, search_url

log = logging.getLogger(__name__)

FallbackMode = Literal["off", "fallback", "always"]


class PageReport(BaseModel):
    page: int
    url: str
    from_cache: bool = False
    listings: int = 0
    exact_matches: int = 0
    extraction: str = "dom"
    new: int = 0
    updated: int = 0


class ScrapeReport(BaseModel):
    keywords: str
    search_id: int | None = None
    pages: list[PageReport] = Field(default_factory=list)
    total_results: int | None = None
    stopped_reason: str = "done"
    block_kind: str | None = None

    @property
    def listings(self) -> int:
        return sum(p.listings for p in self.pages)

    @property
    def new(self) -> int:
        return sum(p.new for p in self.pages)


async def scrape(
    queries: list[SearchQuery],
    *,
    settings: Settings,
    db: Database,
    pages: int = 1,
    use_cache: bool = True,
    screenshots: bool = False,
    vision: FallbackMode = "fallback",
    llm: FallbackMode = "off",
    on_page: Callable[[SearchQuery, PageReport], None] | None = None,
) -> list[ScrapeReport]:
    """Scrape sold listings for each query in one browser session.

    Stops a query early when there is no next page or when the page reached
    eBay's "Results matching fewer words" section (later pages hold only loose
    matches). Stops everything on a challenge the user did not solve, or when
    the per-run page budget is used up.
    """
    from .fetch import BlockedError, EbayFetcher, FetchError, PageBudgetExceeded

    settings.ensure_dirs()
    reports: list[ScrapeReport] = []
    async with EbayFetcher(settings) as fetcher:
        for query in queries:
            report = ScrapeReport(keywords=query.keywords)
            reports.append(report)
            report.search_id = db.record_search(query, search_url(query, 1))
            for page_no in range(1, pages + 1):
                url = search_url(query, page_no)
                shot_dir = settings.screenshot_dir / _slug(query.keywords) / f"p{page_no}" if screenshots else None
                try:
                    fetched = await fetcher.fetch(url, use_cache=use_cache, screenshot_dir=shot_dir)
                except BlockedError as exc:
                    report.stopped_reason = str(exc)
                    report.block_kind = exc.info.kind
                    log.warning("stopping: %s", exc)
                    return reports
                except PageBudgetExceeded:
                    report.stopped_reason = "page budget for this run used up"
                    return reports
                except FetchError as exc:
                    # This page failed after retries; the next search may still work.
                    report.stopped_reason = f"fetch failed: {exc}"
                    log.warning("%s: %s", url, exc)
                    break

                parsed = parse_search_page(fetched.html, site=query.site)
                try:
                    listings, how = await _extract(
                        parsed.listings, fetched.html, settings=settings, site=query.site,
                        vision=vision, llm=llm, tiles=[(Path(p), b) for p, b in fetched.tiles],
                    )
                except _ChallengeSeen as exc:
                    report.stopped_reason = str(exc)
                    report.block_kind = "captcha"
                    return reports
                result = PageResult(
                    url=url,
                    page=page_no,
                    fetched_at=fetched.fetched_at,
                    total_results=parsed.total_results,
                    listings=listings,
                    html_path=fetched.html_path,
                    screenshot_path=str(shot_dir) if shot_dir else None,
                    from_cache=fetched.from_cache,
                    has_next_page=parsed.has_next_page,
                )
                stats = db.save_page(report.search_id, result)
                if report.total_results is None:
                    report.total_results = parsed.total_results
                page_report = PageReport(
                    page=page_no, url=url, from_cache=fetched.from_cache, listings=len(listings),
                    exact_matches=sum(1 for item in listings if item.matches_query), extraction=how,
                    new=stats.new, updated=stats.updated,
                )
                report.pages.append(page_report)
                if on_page:
                    on_page(query, page_report)
                if not listings:
                    report.stopped_reason = (
                        "no listings on page" if fetched.has_results
                        else "eBay showed a page that is neither results nor a known challenge "
                             f"(saved at {fetched.html_path or 'n/a'})"
                    )
                    break
                if page_report.exact_matches < len(listings):
                    report.stopped_reason = "reached 'results matching fewer words'"
                    break
                if parsed.has_next_page is False:
                    report.stopped_reason = "last page"
                    break
    return reports


class _ChallengeSeen(RuntimeError):
    """A screenshot turned out to show a CAPTCHA: stop like a detected block."""


async def _extract(
    dom_listings: list[Listing],
    html: str,
    *,
    settings: Settings,
    site: str,
    vision: FallbackMode,
    llm: FallbackMode,
    tiles: list | None = None,
) -> tuple[list[Listing], str]:
    """Pick DOM results, or fall back to vision / Claude on screenshots of the page."""
    if dom_listings and vision != "always" and llm != "always":
        return dom_listings, "dom"
    if vision == "off" and llm == "off":
        return dom_listings, "dom"
    if tiles:
        return _extract_from_tiles(dom_listings, tiles, settings=settings, site=site, vision=vision, llm=llm)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ebay-sold-tiles-") as tmp:
        tiles = await render_tiles(html, settings=settings, out_dir=Path(tmp))
        return _extract_from_tiles(dom_listings, tiles, settings=settings, site=site, vision=vision, llm=llm)


def _extract_from_tiles(dom_listings: list[Listing], tiles: list, *, settings: Settings, site: str,
                        vision: FallbackMode, llm: FallbackMode) -> tuple[list[Listing], str]:
    if vision != "off" and (llm != "always"):
        found = vision_extract_tiles(tiles, settings=settings, site=site)
        if found or llm == "off":
            return (found or dom_listings), ("vision" if found else "dom")
    if llm == "off":
        return dom_listings, "dom"
    found = llm_extract_tiles(tiles, settings=settings, site=site)
    return (found or dom_listings), ("llm" if found else "dom")


async def render_tiles(html: str, *, settings: Settings, out_dir: Path) -> list[tuple[Path, Box]]:
    """Screenshot saved HTML offline (no request to eBay) as page tiles."""
    from .capture import capture_tiles, render_html

    async with render_html(html, settings=settings.browser.model_copy(update={"headless": True})) as page:
        return await capture_tiles(page, out_dir)


def vision_extract_tiles(tiles: list, *, settings: Settings, site: str) -> list[Listing]:
    """YOLO + OCR over page tiles; ``[]`` (with a warning) when no model or it fails."""
    weights = settings.weights_path
    if not weights.exists():
        log.warning("vision fallback skipped: no model at %s (see docs/vision.md to train one)", weights)
        return []
    try:
        from .vision.detector import Detector
        from .vision.extract import extract_page
        from .vision.ocr import get_ocr

        detector = Detector(weights, imgsz=settings.vision.imgsz, conf=settings.vision.conf)
        return extract_page(tiles, detector=detector, ocr=get_ocr(settings.vision.ocr_backend), site=site)
    except ImportError as exc:
        log.warning("vision fallback skipped: %s (pip install 'ebay-sold[vision]')", exc)
    except Exception:  # a broken model or OCR install must not lose the page
        log.exception("vision fallback failed")
    return []


def llm_extract_tiles(tiles: list, *, settings: Settings, site: str) -> list[Listing]:
    """Claude over page tiles; ``[]`` (with a warning) on failure. A CAPTCHA stops the run."""
    from .llm import ClaudeExtractor, LLMExtractionError

    try:
        return ClaudeExtractor(settings.llm).extract_tiles([(Path(p), b) for p, b in tiles], site=site)
    except LLMExtractionError as exc:
        if exc.reason == "captcha":
            raise _ChallengeSeen(f"the page screenshot shows a challenge: {exc}") from exc
        log.warning("Claude fallback failed (%s): %s", exc.reason, exc)
        return []


# --- saved pages and screenshots -----------------------------------------------


def read_html(path: str | Path) -> str:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="replace")


_TITLE_RE = re.compile(r"<title>(.*?)(?:\s+for sale)?\s*\|\s*eBay\s*</title>", re.IGNORECASE | re.DOTALL)
_SEARCH_BOX_RE = re.compile(r'<input[^>]*id="gh-ac"[^>]*value="([^"]*)"', re.IGNORECASE)


def guess_keywords(html: str) -> str | None:
    """Recover the search keywords from a saved results page."""
    import html as html_lib

    m = _SEARCH_BOX_RE.search(html)
    if m and m.group(1).strip():
        return clean_text(html_lib.unescape(m.group(1)))
    m = _TITLE_RE.search(html)
    return clean_text(html_lib.unescape(m.group(1))) if m else None


def import_html(
    paths: list[Path],
    *,
    db: Database,
    keywords: str | None = None,
    site: str = "www.ebay.com",
) -> list[ScrapeReport]:
    """Parse sold-search pages you saved from your own browser into the database."""
    site = normalize_site(site)
    reports: dict[str, ScrapeReport] = {}
    for i, path in enumerate(paths, start=1):
        html = read_html(path)
        kw = keywords or guess_keywords(html) or Path(path).stem
        if kw not in reports:
            query = SearchQuery(keywords=kw, site=site)
            reports[kw] = ScrapeReport(keywords=kw, search_id=db.record_search(query, f"file://{Path(path).resolve()}"))
        report = reports[kw]
        parsed = parse_search_page(html, site=site)
        result = PageResult(
            url=f"file://{Path(path).resolve()}",
            page=len(report.pages) + 1,
            fetched_at=datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc),
            total_results=parsed.total_results,
            listings=parsed.listings,
            html_path=str(path),
            has_next_page=parsed.has_next_page,
        )
        stats = db.save_page(report.search_id, result)
        report.total_results = report.total_results or parsed.total_results
        report.pages.append(PageReport(
            page=result.page, url=result.url, listings=len(parsed.listings),
            exact_matches=sum(1 for item in parsed.listings if item.matches_query),
            new=stats.new, updated=stats.updated,
        ))
        if not parsed.listings:
            log.warning("%s: no listings found (layout %s)", path, parsed.layout)
    return list(reports.values())


def default_currency(site: str) -> str:
    return SITE_CURRENCY[normalize_site(site)]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "query"
