"""Polite fetching of eBay result pages.

One persistent, human-paced browser (``browser``), an HTML cache so no page is
fetched twice (``cache``), pacing and a per-run page budget (``pacing``), and
challenge detection that stops a run instead of hammering (``blocks``).
"""

from .blocks import (
    BlockInfo,
    BlockKind,
    detect_block,
    has_captcha_widget,
    is_ebay_host,
    looks_like_results,
    page_title,
)
from .browser import BlockedError, EbayFetcher, FetchError, FetchResult
from .cache import CachedPage, HtmlCache, cache_key, page_stem
from .pacing import Pacer, PageBudgetExceeded

__all__ = [
    "BlockInfo",
    "BlockKind",
    "BlockedError",
    "CachedPage",
    "EbayFetcher",
    "FetchError",
    "FetchResult",
    "HtmlCache",
    "Pacer",
    "PageBudgetExceeded",
    "cache_key",
    "detect_block",
    "has_captcha_widget",
    "is_ebay_host",
    "looks_like_results",
    "page_stem",
    "page_title",
]
