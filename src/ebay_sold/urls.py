"""Build eBay sold-listing search URLs and canonical item URLs."""

from __future__ import annotations

import re
from urllib.parse import urlencode, urlsplit

from .models import SearchQuery

SUPPORTED_SITES = (
    "www.ebay.com",
    "www.ebay.co.uk",
    "www.ebay.ca",
    "www.ebay.com.au",
    "www.ebay.de",
    "www.ebay.fr",
    "www.ebay.it",
    "www.ebay.es",
)

SITE_CURRENCY = {
    "www.ebay.com": "USD",
    "www.ebay.co.uk": "GBP",
    "www.ebay.ca": "CAD",
    "www.ebay.com.au": "AUD",
    "www.ebay.de": "EUR",
    "www.ebay.fr": "EUR",
    "www.ebay.it": "EUR",
    "www.ebay.es": "EUR",
}

_SORT = {"best_match": 12, "ended_recently": 13, "price_low": 15, "price_high": 16}

_CONDITION = {
    "new": "1000",
    "open_box": "1500",
    "refurbished": "2000|2010|2020|2030|2500",
    "used": "3000",
    "for_parts": "7000",
}

_ITEM_ID_RE = re.compile(r"/itm/(?:[^/?#]+/)?(\d{9,15})")


def normalize_site(site: str) -> str:
    """Accept ``ebay.co.uk``, ``www.ebay.co.uk`` or a full URL; return the host."""
    host = urlsplit(site).netloc if "://" in site else site
    host = host.lower().strip().strip("/")
    if not host.startswith("www."):
        host = "www." + host
    if host not in SUPPORTED_SITES:
        raise ValueError(f"unsupported eBay site {site!r}; expected one of {', '.join(SUPPORTED_SITES)}")
    return host


def search_url(query: SearchQuery, page: int = 1) -> str:
    """Return the sold-and-completed search URL for ``query`` at ``page`` (1-based)."""
    if page < 1:
        raise ValueError("page is 1-based")
    keywords = " ".join([query.keywords.strip(), *(f"-{w.strip()}" for w in query.exclude if w.strip())])
    params: list[tuple[str, str | int]] = [
        ("_nkw", keywords),
        ("_sacat", query.category_id or 0),
        ("LH_Sold", 1),
        ("LH_Complete", 1),
        ("_sop", _SORT[query.sort]),
        ("_ipg", query.items_per_page),
    ]
    if query.condition:
        params.append(("LH_ItemCondition", _CONDITION[query.condition]))
    if query.min_price is not None:
        params.append(("_udlo", _fmt_price(query.min_price)))
    if query.max_price is not None:
        params.append(("_udhi", _fmt_price(query.max_price)))
    if query.listing_type == "auction":
        params.append(("LH_Auction", 1))
    elif query.listing_type == "buy_it_now":
        params.append(("LH_BIN", 1))
    params.append(("rt", "nc"))
    if page > 1:
        params.append(("_pgn", page))
    return f"https://{normalize_site(query.site)}/sch/i.html?{urlencode(params)}"


def item_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    m = _ITEM_ID_RE.search(url)
    return m.group(1) if m else None


def canonical_item_url(item_id: str, site: str = "www.ebay.com") -> str:
    return f"https://{normalize_site(site)}/itm/{item_id}"


def _fmt_price(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")
