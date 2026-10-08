"""Recognise bot-challenge and block pages, so the fetcher stops instead of hammering.

eBay protects search with Akamai's bot manager plus its own challenge pages.
When one of those answers instead of a results page, the worst thing a scraper
can do is retry: that is what turns a one-off challenge into an IP-level block.
So every page the fetcher loads goes through ``detect_block`` first.

Detection trusts only signals a real results page never carries:

* the final URL (``/splashui/challenge``, ``/splashui/captcha``, ``signin.ebay.*``),
* the HTTP status (403, 429),
* an actual CAPTCHA widget (hCaptcha / reCAPTCHA iframe, script or container),
* the exact titles and wording of known challenge pages.

Plain *mentions* are not signals: every results page links to signin.ebay.com
and a help footer may say "captcha". Titles and body wording are only checked
on pages that contain no search-results markup, and titles must match a whole
phrase ("Access Denied", not "Access Denied Poster for sale | eBay").

This module only recognises challenges. Nothing in this package solves them.
"""

from __future__ import annotations

import html as html_lib
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel

BlockKind = Literal["captcha", "interstitial", "access_denied", "rate_limited", "signin"]


class BlockInfo(BaseModel):
    """Why a page is a challenge or block rather than search results."""

    kind: BlockKind
    reason: str  # human-readable evidence, e.g. "title 'Security Measure'"
    url: str  # the final URL that showed it
    status: int | None = None
    title: str | None = None


# Matches eBay's own hosts: www.ebay.com, signin.ebay.co.uk, ebay.com.au, ...
_EBAY_HOST_RE = re.compile(
    r"(?:^|\.)ebay\.(?:com|co\.uk|ca|com\.au|de|fr|it|es|at|ch|be|nl|ie|pl|com\.hk|com\.sg|com\.my|ph|in)$"
)

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)

# Results markup: the s-card list (2025+) or the legacy s-item list.
_RESULTS_RE = re.compile(r"""\bclass\s*=\s*["']?[^"'>]*\b(?:srp-results|s-card|s-item)\b""", re.IGNORECASE)

# A CAPTCHA widget that is actually embedded, not a link that mentions one.
_WIDGET_RE = re.compile(
    r"""<(?:iframe|script)\b[^>]*?\bsrc\s*=\s*["']?[^"'>\s]*"""
    r"""(?:hcaptcha\.com|/recaptcha/|recaptcha\.net|arkoselabs\.com|funcaptcha\.com|captcha-delivery\.com)"""
    r"""|\bclass\s*=\s*["']?[^"'>]*\b(?:h-captcha|g-recaptcha)\b"""
    r"""|\bdata-hcaptcha-widget-id\b""",
    re.IGNORECASE,
)

# Whole-title phrases of known challenge pages, matched case-insensitively.
_TITLES: tuple[tuple[str, BlockKind], ...] = (
    ("security measure", "captcha"),
    ("please verify yourself", "captcha"),
    ("verify yourself", "captcha"),
    ("pardon our interruption", "interstitial"),
    ("access denied", "access_denied"),
)

# Body wording of the same pages (checked only when there are no results).
_PHRASES: tuple[tuple[str, BlockKind], ...] = (
    ("please verify yourself to continue", "captcha"),
    ("verify you are a human", "captcha"),
    ("made us think you were a bot", "interstitial"),
    ("checking your browser before you access", "interstitial"),
    ("you don't have permission to access", "access_denied"),
)

# Only the head of a big page is scanned for body wording: block pages are small.
_PHRASE_SCAN_CHARS = 300_000


def is_ebay_host(host: str | None) -> bool:
    """True for eBay's own sites (``www.ebay.com``, ``signin.ebay.de``...)."""
    return bool(host) and bool(_EBAY_HOST_RE.search(host.lower().rstrip(".")))


def page_title(html: str) -> str | None:
    m = _TITLE_RE.search(html)
    if not m:
        return None
    return re.sub(r"\s+", " ", html_lib.unescape(m.group(1))).strip() or None


def looks_like_results(html: str) -> bool:
    """True when the HTML contains a search-results list (s-card or legacy s-item)."""
    return bool(_RESULTS_RE.search(html))


def has_captcha_widget(html: str) -> bool:
    """True when a CAPTCHA widget is embedded (so only a person can get past the page)."""
    return bool(_WIDGET_RE.search(html))


def detect_block(*, status: int | None, url: str, html: str) -> BlockInfo | None:
    """Return why this page is a challenge/block, or ``None`` for an ordinary page."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = parts.path.lower()
    title = page_title(html)

    def hit(kind: BlockKind, reason: str) -> BlockInfo:
        return BlockInfo(kind=kind, reason=reason, url=url, status=status, title=title)

    if status == 429:
        return hit("rate_limited", "HTTP 429 Too Many Requests")
    if is_ebay_host(host) and (host.startswith("signin.") or path.startswith("/signin")):
        return hit("signin", f"redirected to sign-in ({host})")
    if path.startswith("/splashui/captcha") or path.startswith("/splashui/challenge"):
        return hit("captcha", f"challenge URL {parts.path}")
    if has_captcha_widget(html):
        return hit("captcha", "CAPTCHA widget (hCaptcha/reCAPTCHA) embedded in the page")

    if not looks_like_results(html):
        if title:
            for phrase, kind in _TITLES:
                if _title_is(title, phrase):
                    return hit(kind, f"title {title!r}")
        text = _visible_text(html[:_PHRASE_SCAN_CHARS])
        for phrase, kind in _PHRASES:
            if phrase in text:
                return hit(kind, f"page says {phrase!r}")

    if status == 403:
        if title and title.lower().startswith("error page"):
            return hit("access_denied", f"HTTP 403 {title!r} (Akamai edge block)")
        return hit("access_denied", "HTTP 403 Forbidden")
    return None


def _title_is(title: str, phrase: str) -> bool:
    """``title`` is ``phrase``, optionally followed by a separator ("| eBay", "...")."""
    t = title.lower()
    if not t.startswith(phrase):
        return False
    rest = t[len(phrase):]
    return not rest.strip() or bool(re.match(r"\s*[|\-–—:.…!]", rest))


def _visible_text(html: str) -> str:
    html = re.sub(r"<(script|style)\b.*?</\1\s*>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", html))
    text = text.replace("’", "'").replace(" ", " ")
    return re.sub(r"\s+", " ", text).lower()
