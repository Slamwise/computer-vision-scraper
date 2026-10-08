"""Turn the text eBay shows into typed values.

Shared by every extractor (DOM, YOLO + OCR, Claude), so it accepts both clean
DOM text ("Sold  Apr 25, 2026") and OCR output that lost its spaces or misread
a glyph ("SoldApr25,2026", "S65.00").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# eBay pads some strings with invisible separators (U+2063 etc.) to defeat scrapers.
_INVISIBLE_RE = re.compile("[​-‏⁠-⁤﻿­]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Drop invisible characters and collapse whitespace."""
    if not text:
        return ""
    return _WS_RE.sub(" ", _INVISIBLE_RE.sub("", text)).strip()


# --- money -------------------------------------------------------------------

# Longest prefixes first so "US $" wins over "$".
_CURRENCY_MARKERS: tuple[tuple[str, str | None], ...] = (
    ("US $", "USD"), ("US$", "USD"), ("USD", "USD"),
    ("C $", "CAD"), ("C$", "CAD"), ("CA $", "CAD"), ("CAD", "CAD"),
    ("AU $", "AUD"), ("AU$", "AUD"), ("A $", "AUD"), ("AUD", "AUD"),
    ("GBP", "GBP"), ("£", "GBP"),
    ("EUR", "EUR"), ("€", "EUR"),
    ("$", None),  # bare dollar: the site's own currency
)

_AMOUNT_RE = re.compile(r"(?<![\d])(\d{1,3}(?:[.,\u00a0 ]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)(?![\d])")
# What may sit between the two amounts of a price range: "$3.75 to $23.95", "3,75 EUR bis 23,95 EUR".
_RANGE_GAP_RE = re.compile(
    r"^\s*(?:US|C|CA|AU|A)?\s*(?:\$|£|€|USD|CAD|AUD|GBP|EUR)?\s*(?:to|bis|à|a|–|—|-)\s*(?:US|C|CA|AU|A)?\s*(?:\$|£|€|USD|CAD|AUD|GBP|EUR)?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Money:
    amount: float
    amount_max: float | None = None
    currency: str | None = None


def _detect_currency(text: str) -> str | None:
    """ISO code from a currency marker; ``None`` for a bare "$" or no marker."""
    upper = text.upper()
    for marker, code in _CURRENCY_MARKERS:
        if marker.upper() in upper:
            return code
    return None


def _parse_amount(raw: str) -> float | None:
    s = raw.replace(" ", "").replace("\u00a0", "")
    # The last separator is a decimal point iff 1-2 digits follow it; a
    # separator followed by exactly 3 digits groups thousands ("1,234" / "1.234").
    m = re.search(r"[.,](\d{1,2})$", s)
    if m:
        whole = re.sub(r"[.,]", "", s[: m.start()])
        s = f"{whole}.{m.group(1)}"
    else:
        s = re.sub(r"[.,]", "", s)
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _ocr_fix_money(text: str) -> str:
    # OCR commonly reads "$" as "S" and "O" as "0" next to digits.
    text = re.sub(r"(?<![A-Za-z])S(?=\s?\d)", "$", text)
    return re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", text)


def parse_money(text: str | None, default_currency: str | None = None) -> Money | None:
    """Parse "$1,234.56", "£9.99", "EUR 12,50", "$3.75 to $23.95"...

    Returns ``None`` when no amount is shown (e.g. "See price"). A bare "$"
    takes ``default_currency`` (the site's currency).
    """
    text = _ocr_fix_money(clean_text(text))
    if not text:
        return None
    matches = list(_AMOUNT_RE.finditer(text))
    if not matches:
        return None
    low = _parse_amount(matches[0].group(1))
    if low is None:
        return None
    high = None
    if len(matches) > 1 and _RANGE_GAP_RE.match(text[matches[0].end(): matches[1].start()]):
        second = _parse_amount(matches[1].group(1))
        if second is not None and second > low:
            high = second
    return Money(amount=low, amount_max=high, currency=_detect_currency(text) or default_currency)


_FREE_RE = re.compile(
    r"\bfree\b|kostenlos|gratuit|gratis|gratuita",
    re.IGNORECASE,
)


def parse_shipping(text: str | None, default_currency: str | None = None) -> float | None:
    """``0.0`` for free shipping, the amount for "+$9.45 delivery", else ``None``."""
    text = clean_text(text)
    if not text:
        return None
    if _FREE_RE.search(text):
        return 0.0
    money = parse_money(text, default_currency)
    return money.amount if money else None


# --- dates -------------------------------------------------------------------

_MONTHS: dict[str, int] = {}
for _n, _names in enumerate(
    (
        ("jan", "january", "januar", "janv", "janvier", "gen", "gennaio", "ene", "enero"),
        ("feb", "february", "februar", "févr", "fevr", "février", "febbraio", "febrero"),
        ("mar", "march", "mär", "märz", "maerz", "mars", "marzo"),
        ("apr", "april", "avr", "avril", "aprile", "abr", "abril"),
        ("may", "mai", "mag", "maggio", "mayo"),
        ("jun", "june", "juni", "juin", "giu", "giugno", "junio"),
        ("jul", "july", "juli", "juil", "juillet", "lug", "luglio", "julio"),
        ("aug", "august", "août", "aout", "ago", "agosto"),
        ("sep", "sept", "september", "septembre", "set", "settembre", "septiembre"),
        ("oct", "october", "okt", "oktober", "octobre", "ott", "ottobre", "octubre"),
        ("nov", "november", "novembre", "noviembre"),
        ("dec", "december", "dez", "dezember", "déc", "décembre", "dic", "dicembre", "diciembre"),
    ),
    start=1,
):
    for _name in _names:
        _MONTHS[_name] = _n

_MONTH_ALT = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))
# "Apr 25, 2026" / "Apr25,2026" / "April 25"
_MDY_RE = re.compile(rf"(?<![a-zà-ü])({_MONTH_ALT})\.?\s*(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?", re.IGNORECASE)
# "25 Apr 2026" / "25. Apr. 2026" / "25-Apr-26"
_DMY_RE = re.compile(rf"(\d{{1,2}})\.?[\s-]*({_MONTH_ALT})\.?(?![a-zà-ü])[\s-]*(\d{{4}}|\d{{2}})?", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})")
_SOLD_PREFIX_RE = re.compile(r"^\s*(sold|ended|verkauft|vendu(?:\s+le)?|venduto(?:\s+il)?|vendido(?:\s+el)?)\b[:\s]*", re.IGNORECASE)


def _mk_date(year: int | None, month: int, day: int, today: date) -> date | None:
    if year is not None and year < 100:
        year += 2000
    try:
        if year is None:
            d = date(today.year, month, day)
            return d if d <= today else date(today.year - 1, month, day)
        return date(year, month, day)
    except ValueError:
        return None


def parse_sold_date(text: str | None, today: date | None = None, *, day_first: bool = False) -> date | None:
    """Parse "Sold Apr 25, 2026", "SoldApr25,2026", "Sold 25 Apr 2026", "Verkauft 25. Apr. 2026".

    A missing year means the most recent such date not after ``today``.
    ``day_first`` decides all-numeric dates ("05/04/2026"): True for UK/EU sites.
    """
    text = clean_text(text)
    if not text:
        return None
    today = today or date.today()
    text = _SOLD_PREFIX_RE.sub("", text)
    if text.lower().startswith("sold"):  # OCR glued "Sold" to the month: "SoldApr25"
        text = text[4:]
    m = _MDY_RE.search(text)
    d_m = _DMY_RE.search(text)
    # Prefer whichever pattern starts first; "25 Apr 2026" also contains "Apr 2026" as a weak MDY match.
    if d_m and (not m or d_m.start() < m.start()):
        return _mk_date(int(d_m.group(3)) if d_m.group(3) else None, _MONTHS[d_m.group(2).lower()], int(d_m.group(1)), today)
    if m:
        return _mk_date(int(m.group(3)) if m.group(3) else None, _MONTHS[m.group(1).lower()], int(m.group(2)), today)
    n = _NUMERIC_RE.search(text)
    if n:
        a, b, y = int(n.group(1)), int(n.group(2)), int(n.group(3))
        day, month = (a, b) if day_first else (b, a)
        return _mk_date(y, month, day, today)
    return None


# --- counts, sellers, formats -----------------------------------------------

_COUNT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([KkMm])?")


def parse_count(text: str | None) -> int | None:
    """"267" -> 267, "2.1K" -> 2100, "1,234" -> 1234, "3 bids" -> 3."""
    text = clean_text(text)
    m = _COUNT_RE.search(text)
    if not m:
        return None
    num, suffix = m.group(1), (m.group(2) or "").upper()
    if suffix:
        value = float(num.replace(",", "."))
        return int(round(value * (1000 if suffix == "K" else 1_000_000)))
    return int(re.sub(r"[.,]", "", num))


_FEEDBACK_RE = re.compile(
    r"^(?P<seller>.*?)\s*(?P<pct>\d{1,3}(?:[.,]\d)?)\s*%\s*(?:positive|positiv|positif|positivo|positive feedback)?\s*\((?P<count>[\d.,]+\s*[KkMm]?)\)",
    re.IGNORECASE,
)


def parse_feedback(text: str | None) -> tuple[str | None, float | None, int | None]:
    """"mystuffnnowyours 100% positive (267)" -> ("mystuffnnowyours", 100.0, 267)."""
    text = clean_text(text)
    m = _FEEDBACK_RE.search(text)
    if not m:
        return (text.split(" ")[0] if text else None), None, None
    seller = m.group("seller").strip() or None
    return seller, float(m.group("pct").replace(",", ".")), parse_count(m.group("count"))


_BIDS_RE = re.compile(r"(\d+)\s*(?:bids?|gebote?|enchères?|offerte?|pujas?)\b", re.IGNORECASE)


def classify_format(texts: list[str]) -> tuple[str, int | None]:
    """Infer the sale format from a card's attribute rows.

    Returns ``(format, bids)`` where format is one of "auction", "best_offer",
    "buy_it_now", "unknown". "or Best Offer" means the seller accepted offers;
    the card does not say whether this sale was an accepted offer.
    """
    bids: int | None = None
    fmt = "unknown"
    for raw in texts:
        t = clean_text(raw).lower()
        m = _BIDS_RE.search(t)
        if m:
            bids = int(m.group(1))
            fmt = "auction"
        elif fmt != "auction" and ("best offer" in t or "preisvorschlag" in t or "offre directe" in t):
            fmt = "best_offer"
        elif fmt == "unknown" and ("buy it now" in t or "sofort-kaufen" in t or "achat immédiat" in t or "compralo subito" in t or "¡cómpralo ya!" in t):
            fmt = "buy_it_now"
        elif fmt == "unknown" and t in ("auction", "auktion", "enchères", "asta", "subasta"):
            fmt = "auction"
    return fmt, bids


_LOCATION_RE = re.compile(r"^(?:located in|from|aus|de|da|desde)\s+", re.IGNORECASE)


def parse_location(text: str | None) -> str | None:
    """"Located in United States" -> "United States"."""
    text = clean_text(text)
    if not text:
        return None
    return _LOCATION_RE.sub("", text) or None
