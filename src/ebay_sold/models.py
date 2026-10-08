"""Data types shared by every stage: fetching, DOM parsing, vision, storage."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Extraction = Literal["dom", "vision", "llm"]
ListingFormat = Literal["auction", "buy_it_now", "best_offer", "unknown"]
Condition = Literal["new", "used", "open_box", "refurbished", "for_parts"]
SortOrder = Literal["ended_recently", "price_high", "price_low", "best_match"]
ListingType = Literal["all", "auction", "buy_it_now"]


class Listing(BaseModel):
    """One sold listing, however it was extracted.

    Money fields are floats in the listing's own currency. ``None`` means
    "not shown / could not be read"; it never means zero. ``shipping == 0.0``
    means free shipping.
    """

    item_id: str | None = None
    site: str = "www.ebay.com"
    title: str
    price: float | None = None
    price_max: float | None = None  # upper bound when eBay shows a range ("$3.75 to $23.95")
    original_price: float | None = None  # struck-through list price, when shown
    currency: str | None = None  # ISO 4217, e.g. "USD", "GBP"
    price_text: str | None = None
    shipping: float | None = None
    shipping_text: str | None = None
    sold_date: date | None = None
    sold_date_text: str | None = None
    condition: str | None = None
    listing_format: ListingFormat = "unknown"
    bids: int | None = None
    seller: str | None = None
    seller_feedback_pct: float | None = None
    seller_feedback_count: int | None = None
    location: str | None = None
    url: str | None = None  # canonical https://<site>/itm/<item_id>, tracking params stripped
    image_url: str | None = None
    sponsored: bool = False
    # False for cards eBay shows under "Results matching fewer words": they do
    # not contain every search keyword and usually should not be priced together.
    matches_query: bool = True
    position: int | None = None  # 1-based rank on the results page
    extraction: Extraction = "dom"
    confidence: float | None = None  # 0..1; DOM extraction uses None (exact)

    @property
    def total_price(self) -> float | None:
        """Price plus shipping, when both are known."""
        if self.price is None or self.shipping is None:
            return None
        return round(self.price + self.shipping, 2)


class SearchQuery(BaseModel):
    """A sold-listings search. ``ebay_sold.urls.search_url`` turns it into a URL."""

    keywords: str
    site: str = "www.ebay.com"
    category_id: int | None = None
    condition: Condition | None = None
    min_price: float | None = None
    max_price: float | None = None
    exclude: list[str] = Field(default_factory=list)  # words to exclude ("-word")
    listing_type: ListingType = "all"
    sort: SortOrder = "ended_recently"
    items_per_page: Literal[60, 120, 240] = 240  # bigger pages = fewer requests = fewer challenges


class PageResult(BaseModel):
    """Everything extracted from one search-results page."""

    url: str
    page: int = 1
    fetched_at: datetime
    total_results: int | None = None  # eBay's "N results for ..." count, when shown
    listings: list[Listing] = Field(default_factory=list)
    html_path: str | None = None
    screenshot_path: str | None = None
    from_cache: bool = False
    has_next_page: bool | None = None


class Box(BaseModel):
    """Axis-aligned box in CSS pixels, in page (document) coordinates."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def intersection(self, other: "Box") -> float:
        ix = max(0.0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0.0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        return ix * iy


# Detection classes, in YOLO class-index order. "listing" is the whole result
# card, so the vision pipeline can segment cards on a page without the DOM.
VISION_CLASSES: tuple[str, ...] = (
    "listing",
    "title",
    "price",
    "sold_date",
    "shipping",
    "condition",
    "image",
)


class CardRegion(BaseModel):
    """Where one result card and its fields are drawn on a rendered page.

    Produced from the live DOM by ``ebay_sold.capture.card_regions``. It is the
    ground truth for YOLO auto-labels and for scoring vision extraction.
    """

    item_id: str | None = None
    box: Box
    fields: dict[str, Box] = Field(default_factory=dict)  # VISION_CLASSES name -> box
    texts: dict[str, str] = Field(default_factory=dict)  # field name -> visible text
