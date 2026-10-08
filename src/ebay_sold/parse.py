"""Read sold listings out of a saved eBay search-results page.

This is the primary extraction path: the HTML says exactly what each card
shows, so there is nothing to guess. The vision pipeline is the fallback for
when eBay changes its markup, and it is trained on boxes taken from this same
DOM, so the two agree on what "price" or "sold date" means.

Two layouts are understood:

* ``s-card`` (current since mid-2025): ``ul.srp-results > li.s-card``.
* ``s-item`` (2015 to mid-2025, may reappear on other sites/locales):
  ``ul.srp-results > li.s-item``.

The parser only locates text; every text -> value conversion goes through
:mod:`ebay_sold.normalize`, which the OCR and Claude extractors share.

A page that is not a results page (bot challenge, error page, empty string)
yields zero listings and ``layout="unknown"``; parsing never raises on page
content. Callers treat that combination as "fetch failed or markup changed".
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, TypeVar
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from . import normalize
from .models import Listing
from .urls import SITE_CURRENCY, canonical_item_url, item_id_from_url, normalize_site

log = logging.getLogger(__name__)

Layout = Literal["s-card", "s-item", "unknown"]

# eBay renders two hidden placeholder cards per page ("Shop on eBay", $20.00)
# that its client-side code clones. They are not listings.
TEMPLATE_LISTING_ID = "2500219655424533"
_TEMPLATE_TITLE = "shop on ebay"

# Same row rule as capture.card_regions, so DOM values and vision labels come
# from the same element.
_SHIPPING_ROW_RE = re.compile(r"delivery|shipping|postage|versand|livraison|spedizione|envío|envio", re.IGNORECASE)
_RETURNS_RE = re.compile(r"returns?", re.IGNORECASE)
# "Located in China" / "from China" / "aus China" / "da Cina" / "desde China" / "Provenance : Chine".
# normalize.parse_location strips every prefix but the French one, so that one
# is captured and removed here.
_LOCATION_ROW_RE = re.compile(
    r"^(?:(?:located in|from|aus|da|desde)\s+|(?P<fr>provenance\s*:\s*|provenance\s+))(?=\S)", re.IGNORECASE
)
# A shipping row with an amount always shows a currency marker ("+$9.45",
# "+EUR 4,90"); without one, digits are a delivery window ("in 2-4 days").
_MONEY_HINT_RE = re.compile(r"[$£€]|\b(?:USD|GBP|EUR|CAD|AUD)\b")
_COUNT_HEADING_NUM_RE = re.compile(r"\d[\d.,\s]*")
_MAX_COUNT_DIGITS = 12  # no search has a trillion results; longer digit runs are not a count

# eBay takes the condition line from a fixed list of labels, but a seller's own
# subtitle ("Brand New! Free Shipping!", "Extended 1 Year Warranty") uses the
# same markup. A subtitle is taken as the condition when every word in it is a
# condition-label word (en/de/fr/it/es sites) and nothing else but simple
# punctuation is left: no digits, "!" or "%".
_CONDITION_WORDS = frozenset(
    """
    brand new other see details open box with without tags tag defects pre-owned preowned used
    refurbished certified excellent very good fair seller manufacturer remanufactured
    parts only for or not working like acceptable ungraded graded
    neu brandneu neuwertig sonstige sonstiges siehe artikelbeschreibung mit ohne etikett etiketten
    originalverpackung ovp verpackung offene geöffnet gebraucht generalüberholt zertifiziert hervorragend
    sehr gut akzeptabel vom hersteller verkäufer als ersatzteil ersatzteile defekt nur wie fehlern mängeln
    neuf neuve tout autre avec sans étiquette étiquettes boîte boite défauts occasion d'occasion d’occasion
    ouvert ouverte jamais utilisé reconditionné certifié par le la fabricant vendeur pour pièces détachées
    uniquement ou ne fonctionne pas hors service comme très bon bonne état correct voir détails
    nuovo nuova altro con senza cartellino cartellini scatola difetti aperta aperto mai usato usata
    ricondizionato ricondizionata certificato dal produttore venditore solo ricambi per o non funzionante
    ottimo ottime condizioni buono buone accettabile eccellente vedi dettagli
    nuevo nueva totalmente otro sin etiquetas etiqueta caja defectos abierta abierto usado usada de
    segunda mano reacondicionado reacondicionada certificado el fabricante vendedor piezas para no
    funciona muy bueno buen estado aceptable excelente ver detalles usar
    """.split()
)
_CONDITION_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
_CONDITION_PUNCT = frozenset(" -–—()/:,.")
# Legacy seller line: "vintage_finds (1,234) 99.8%".
_LEGACY_SELLER_RE = re.compile(r"^(?P<seller>.*?)\s*\((?P<count>[\d.,]+\s*[KkMm]?)\)\s*(?P<pct>\d{1,3}(?:[.,]\d)?)\s*%")


class ParsedPage(BaseModel):
    """Everything the DOM parser read from one results page."""

    listings: list[Listing] = Field(default_factory=list)
    total_results: int | None = None  # the "N results for ..." heading ("6,200+" -> 6200)
    has_next_page: bool | None = None  # None: no pagination control found at all
    layout: Layout = "unknown"
    # Cards shown before eBay's "Results matching fewer words" divider; these
    # are the ones with matches_query=True.
    exact_match_count: int = 0


def detect_layout(html_or_soup: str | bytes | Tag) -> Layout:
    """Which results-page markup this is: ``"s-card"``, ``"s-item"`` or ``"unknown"``."""
    soup = html_or_soup if isinstance(html_or_soup, Tag) else _make_soup(html_or_soup)
    if soup is None:
        return "unknown"
    if soup.select_one("ul.srp-results > li.s-card, li.s-card[data-listingid]"):
        return "s-card"
    if soup.select_one("ul.srp-results > li.s-item, li.s-item .s-item__title"):
        return "s-item"
    return "unknown"


def parse_search_page(html: str, *, site: str = "www.ebay.com", today: date | None = None) -> ParsedPage:
    """Extract every sold listing on a search-results page, in page order.

    ``site`` sets the currency of bare "$" prices and whether all-numeric dates
    are day-first. ``today`` resolves sold dates shown without a year.
    Raises ``ValueError`` only for an unsupported ``site``.
    """
    site = normalize_site(site)
    soup = _make_soup(html)
    if soup is None:
        return ParsedPage()
    layout = detect_layout(soup)
    if layout == "unknown":
        return ParsedPage()

    sponsor_css = _guarded(_SponsorCss.from_soup, soup) if layout == "s-card" else None
    ctx = _Context(
        site=site,
        currency=SITE_CURRENCY[site],
        today=today or date.today(),
        day_first=site != "www.ebay.com",
        sponsor_css=sponsor_css or _SponsorCss(),
    )
    parse_card = _parse_s_card if layout == "s-card" else _parse_s_item
    listings: list[Listing] = []
    exact = True
    for kind, el in _river(soup, layout):
        if kind == "divider":
            exact = False
            continue
        try:
            listing = parse_card(el, ctx)
        except Exception:  # one odd card must not cost the whole page
            log.warning("skipping unparseable %s card %s", layout, el.get("data-listingid") or el.get("id"), exc_info=True)
            continue
        if listing is None:
            continue
        listing.position = len(listings) + 1
        listing.matches_query = exact
        listings.append(listing)

    return ParsedPage(
        listings=listings,
        total_results=_guarded(_total_results, soup),
        has_next_page=_guarded(_has_next_page, soup),
        layout=layout,
        exact_match_count=sum(1 for item in listings if item.matches_query),
    )


# --- page level ----------------------------------------------------------------


@dataclass
class _Context:
    site: str
    currency: str
    today: date
    day_first: bool
    sponsor_css: _SponsorCss


def _make_soup(html: str | bytes | None) -> BeautifulSoup | None:
    if not html or not html.strip():
        return None
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        log.warning("could not parse HTML", exc_info=True)
        return None


def _river(soup: BeautifulSoup, layout: Layout) -> Iterator[tuple[str, Tag]]:
    """Yield ``("card", li)`` and ``("divider", li)`` in page order."""
    card_cls = "s-card" if layout == "s-card" else "s-item"
    found = False
    for ul in soup.select("ul.srp-results"):
        for li in ul.find_all("li", recursive=False):
            classes = li.get("class") or []
            if card_cls in classes:
                found = True
                yield "card", li
            elif "srp-river-answer--REWRITE_START" in classes:
                yield "divider", li
    if found:
        return
    # Results list not where we expect it: fall back to every card in document order.
    for li in soup.select(f"li.{card_cls}, li.srp-river-answer--REWRITE_START"):
        yield ("divider" if "srp-river-answer--REWRITE_START" in li.get("class", []) else "card"), li


_T = TypeVar("_T")


def _guarded(fn: Callable[[BeautifulSoup], _T], soup: BeautifulSoup) -> _T | None:
    """Page-level fields are optional: a hostile or odd page loses the field, not the listings."""
    try:
        return fn(soup)
    except Exception:
        log.warning("could not read page field %s", fn.__name__, exc_info=True)
        return None


def _total_results(soup: BeautifulSoup) -> int | None:
    heading = soup.select_one(".srp-controls__count-heading")
    if heading is None:
        return None
    # The count leads the heading in every locale ("31 results for ...",
    # "1.234 Ergebnisse für ..."); digits further in belong to the search words.
    m = _COUNT_HEADING_NUM_RE.match(_text(heading))
    if not m:
        return None
    # The heading is a whole number with thousands separators ("1,234,567+",
    # "1 234 résultats"); normalize.parse_count alone would stop at the second
    # separator, so the separators go first.
    digits = re.sub(r"[.,\s]", "", m.group(0))
    if len(digits) > _MAX_COUNT_DIGITS:
        return None
    return normalize.parse_count(digits)


def _has_next_page(soup: BeautifulSoup) -> bool | None:
    nxt = soup.select_one(".pagination__next")
    if nxt is not None:
        disabled = nxt.get("aria-disabled") == "true" or nxt.name != "a" or not nxt.get("href")
        return not disabled
    if soup.select_one(".pagination, li.srp-river-answer--BASIC_PAGINATION_V2"):
        return False  # pagination area present, but no way forward: last (or only) page
    return None


# --- current "s-card" layout ---------------------------------------------------


def _parse_s_card(card: Tag, ctx: _Context) -> Listing | None:
    listing_id = card.get("data-listingid")
    if listing_id == TEMPLATE_LISTING_ID:
        return None
    title = _title(_first(card, ".s-card__title > .su-styled-text", ".s-card__title"))
    if not title or title.lower() == _TEMPLATE_TITLE:
        return None

    link = _first(card, "a.s-card__link[href*='/itm/']", "a[href*='/itm/']")
    item_id = listing_id if listing_id and listing_id.isdigit() else item_id_from_url(link.get("href") if link else None)

    primary = card.select_one(".su-card-container__attributes__primary")
    rows = primary.select(".s-card__attribute-row") if primary else []
    row_texts = [_text(r) for r in rows]

    listing = Listing(
        item_id=item_id,
        site=ctx.site,
        title=title,
        url=canonical_item_url(item_id, ctx.site) if item_id else None,
        image_url=_image_url(_first(card, "img.s-card__image", ".su-media-container img", ".su-image img"), ctx.site),
        sponsored=_is_sponsored(card, ctx.sponsor_css),
    )
    price_spans = card.select(".s-card__price")
    price_row = price_spans[0].find_parent(class_="s-card__attribute-row") if price_spans else None
    struck = price_row.select(".strikethrough:not(.s-card__price)") if price_row else []
    _set_price(listing, price_spans, struck[0] if struck else None, ctx)

    ship_text = next(
        (t for r, t in zip(rows, row_texts) if r is not price_row and _SHIPPING_ROW_RE.search(t) and not _RETURNS_RE.search(t)),
        None,
    )
    _set_shipping(listing, ship_text, ctx)
    _set_sold_date(listing, card.select_one(".s-card__caption"), ctx)

    listing.condition = _pick_condition([_text(s) for s in card.select(".s-card__subtitle")])
    listing.listing_format, listing.bids = normalize.classify_format(row_texts)  # type: ignore[assignment]
    listing.location = _location(row_texts)

    secondary = card.select_one(".su-card-container__attributes__secondary")
    if secondary is not None:
        for row in secondary.select(".s-card__attribute-row"):
            # Leaf text spans only: the Top Rated badge nests a tooltip in the row.
            spans = [s for s in row.select(".su-styled-text") if s.select_one(".su-styled-text") is None]
            seller, pct, count = normalize.parse_feedback(" ".join(_text(s) for s in spans))
            if pct is not None:
                listing.seller, listing.seller_feedback_pct, listing.seller_feedback_count = seller, pct, count
                break
    return listing


# --- legacy "s-item" layout ----------------------------------------------------


def _parse_s_item(card: Tag, ctx: _Context) -> Listing | None:
    title = _title(_first(card, ".s-item__title span[role=heading]", ".s-item__title"))
    if not title or title.lower() == _TEMPLATE_TITLE:
        return None
    link = _first(card, "a.s-item__link[href]", "a[href*='/itm/']")
    listing_id = card.get("data-listingid")
    item_id = listing_id if listing_id and listing_id.isdigit() else item_id_from_url(link.get("href") if link else None)
    if item_id == TEMPLATE_LISTING_ID:
        return None

    listing = Listing(
        item_id=item_id,
        site=ctx.site,
        title=title,
        url=canonical_item_url(item_id, ctx.site) if item_id else None,
        image_url=_image_url(_first(card, ".s-item__image-wrapper img", ".s-item__image img"), ctx.site),
    )
    price_el = card.select_one(".s-item__price")
    # A struck price outside .s-item__price is the original price ("Was: $39.95").
    struck = [s for s in card.select(".STRIKETHROUGH, .strikethrough") if not _within(s, price_el)]
    _set_price(listing, [price_el] if price_el else [], struck[0] if struck else None, ctx)

    ship = _first(card, ".s-item__shipping", ".s-item__logisticsCost", ".s-item__freeXDays")
    _set_shipping(listing, _text(ship) if ship else None, ctx)
    _set_sold_date(
        listing,
        _first(card, ".s-item__caption--signal", ".s-item__title--tag .POSITIVE", ".s-item__title--tagblock .POSITIVE",
               ".s-item__caption-section .POSITIVE", ".s-item__caption", ".s-item__ended-date", ".s-item__endedDate"),
        ctx,
    )
    cond = _first(card, ".s-item__subtitle .SECONDARY_INFO", ".SECONDARY_INFO")
    if cond is not None:
        listing.condition = _condition(_text(cond))

    detail_els = card.select(".s-item__detail") or card.select(
        ".s-item__purchase-options, .s-item__purchaseOptions, .s-item__bids, .s-item__bidCount, "
        ".s-item__formatBuyItNow, .s-item__formatBestOfferEnabled, .s-item__formatBestOfferAccepted"
    )
    listing.listing_format, listing.bids = normalize.classify_format([_text(e) for e in detail_els])  # type: ignore[assignment]

    loc = _first(card, ".s-item__location", ".s-item__itemLocation")
    listing.location = normalize.parse_location(_text(loc)) if loc else None

    seller_el = card.select_one(".s-item__seller-info-text")
    if seller_el is not None:
        text = _text(seller_el)
        # normalize.parse_feedback reads the current "name 99.8% positive (1,234)"
        # order; the legacy line puts the count first, so reorder it.
        m = _LEGACY_SELLER_RE.match(text)
        if m:
            text = f"{m.group('seller')} {m.group('pct')}% positive ({m.group('count')})"
        listing.seller, listing.seller_feedback_pct, listing.seller_feedback_count = normalize.parse_feedback(text)
    return listing


# --- shared field helpers ----------------------------------------------------------


def _first(card: Tag, *selectors: str) -> Tag | None:
    """First match of the first selector that matches (selectors in priority order)."""
    for sel in selectors:
        el = card.select_one(sel)
        if el is not None:
            return el
    return None


def _text(el: Tag | None) -> str:
    return normalize.clean_text(el.get_text(" ")) if el is not None else ""


def _title(el: Tag | None) -> str:
    if el is None:
        return ""
    # Screen-reader suffix ("Opens in a new window or tab") and the "New Listing" tag are not title text.
    for junk in el.select(".clipped, .s-card__new-listing, .LIGHT_HIGHLIGHT"):
        junk.decompose()
    return normalize.clean_text(el.get_text(""))


def _condition(text: str) -> str | None:
    # "Pre-Owned · Nintendo": the condition comes first.
    return text.split("·")[0].strip() or None


def _is_condition_label(text: str) -> bool:
    """True for eBay's own condition labels ("Pre-Owned", "New – Open box"), not seller taglines."""
    lower = text.lower()
    words = _CONDITION_WORD_RE.findall(lower)
    if not words:
        return False
    rest = set(_CONDITION_WORD_RE.sub("", lower))
    if words[0] in ("graded", "ungraded"):  # trading cards name the grader and grade: "Graded - PSA 10"
        return rest <= _CONDITION_PUNCT | set("0123456789")
    return all(w in _CONDITION_WORDS for w in words) and rest <= _CONDITION_PUNCT


def _pick_condition(subtitles: list[str]) -> str | None:
    """The condition among a card's subtitle lines (seller taglines share the markup)."""
    labels = [label for label in (_condition(s) for s in subtitles) if label]
    known = [label for label in labels if _is_condition_label(label)]
    if known:
        return known[-1]
    if len(labels) > 1:
        # Unfamiliar wording (a new label or locale): the condition line is
        # always drawn last, below any seller tagline.
        return labels[-1]
    if labels:
        log.debug("subtitle %r is not a condition label; condition left empty", labels[0])
    return None


def _location(row_texts: list[str]) -> str | None:
    for text in row_texts:
        m = _LOCATION_ROW_RE.match(text)
        if m and not _MONEY_HINT_RE.search(text):  # "from $9.99 shipping" is not a place
            return normalize.parse_location(text[m.end():] if m.group("fr") else text)
    return None


def _set_price(listing: Listing, price_els: list[Tag], struck_el: Tag | None, ctx: _Context) -> None:
    if not price_els:
        return
    listing.price_text = normalize.clean_text(" ".join(_text(e) for e in price_els)) or None
    # When a best offer was accepted eBay strikes through the asking price and
    # does not show what was paid: that number is an upper bound, not the price.
    shown = [e for e in price_els if not _is_struck(e)]
    asked = [e for e in price_els if _is_struck(e)]
    money = normalize.parse_money(" ".join(_text(e) for e in shown), ctx.currency) if shown else None
    if struck_el is None and asked:
        struck_el = asked[0]
    original = normalize.parse_money(_text(struck_el), ctx.currency) if struck_el is not None else None
    if money is not None:
        listing.price, listing.price_max = money.amount, money.amount_max
    if original is not None and (money is None or original.amount != money.amount):
        listing.original_price = original.amount
    listing.currency = (money or original).currency if (money or original) else ctx.currency


def _within(el: Tag, container: Tag | None) -> bool:
    return container is not None and (el is container or any(p is container for p in el.parents))


def _is_struck(el: Tag) -> bool:
    classes = el.get("class") or []
    return "strikethrough" in classes or "STRIKETHROUGH" in classes or el.select_one(".STRIKETHROUGH, .strikethrough") is not None


def _set_shipping(listing: Listing, text: str | None, ctx: _Context) -> None:
    if not text:
        return
    listing.shipping_text = text
    amount = normalize.parse_shipping(text, ctx.currency)
    listing.shipping = amount if amount == 0.0 or _MONEY_HINT_RE.search(text) else None


def _set_sold_date(listing: Listing, el: Tag | None, ctx: _Context) -> None:
    text = _text(el)
    if text:
        listing.sold_date_text = text
        listing.sold_date = normalize.parse_sold_date(text, ctx.today, day_first=ctx.day_first)


def _image_url(img: Tag | None, site: str) -> str | None:
    if img is None:
        return None
    # data-defer-load holds the full-size photo; src is often a small thumbnail or a placeholder.
    for attr in ("data-defer-load", "data-src", "src"):
        url = (img.get(attr) or "").strip()
        if not url or url.startswith("data:"):
            continue
        # Resolve "//host/x" and "/x" the way a browser on the eBay site would.
        try:
            url = urljoin(f"https://{site}/", url)
        except ValueError:  # e.g. a malformed IPv6 host
            continue
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        if url.startswith("https://"):
            return url
    return None


# --- the obfuscated "Sponsored" footer ---------------------------------------------
#
# Every card carries a footer (.s-card__sep) that spells "Sponsored" in a way
# meant to defeat scrapers; CSS decides whether a reader sees it. Verified by
# rendering all five fixtures in Chromium and comparing each footer with the
# raw markup (no real card in them is sponsored; the hidden template cards
# show the label, which gives a positive case):
#
# * 2026-04 pages: one span per character; inline <style> rules hide classes
#   via ``span.X {display: var(--pd-X)}`` + ``:root{--pd-X: none}`` or push
#   them off-screen (``left: -2000px``). Visible = spans with no hiding rule.
# * 2025-11 pages: a background SVG with the word; invisible copies use a
#   near-zero fill alpha, opacity or font-size (style overrides attributes).
# * 2026-02 pages: plain text in an obfuscated web font whose glyphs may be
#   blank. Telling them apart means decoding the font, so those cards stay
#   sponsored=False (sold searches rarely contain sponsored cards anyway).

_SPONSORED_WORDS = frozenset({"sponsored", "gesponsert", "sponsorisé", "sponsorise", "sponsorizzato", "patrocinado"})
# Cyrillic look-alikes eBay mixes into the label (U+0440 for "p", U+0455 for "s", ...).
_HOMOGLYPHS = str.maketrans({
    "\u0430": "a", "\u0441": "c", "\u0501": "d", "\u0435": "e", "\u0456": "i",
    "\u043e": "o", "\u0440": "p", "\u0455": "s", "\u0578": "n", "\u0443": "y",
})
_CSS_SPAN_CLASS_RE = re.compile(r"span\.([\w-]+)$")
_CSS_VAR_REF_RE = re.compile(r"var\(\s*--([\w-]+)\s*\)$")
_CSS_OFFSCREEN_RE = re.compile(r"-\d{3,}px$")
_SVG_URI_RE = re.compile(r"data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)")
_NUM_RE = re.compile(r"\s*(-?\d*\.?\d+)")


def _css_rules(css: str) -> Iterator[tuple[str, dict[str, str]]]:
    """``(selector, {property: value})`` for each innermost rule.

    Split on braces rather than matched with one big regex: patterns like
    ``span\\.x\\{[^}]*`` rescan to the end of the text for every unclosed rule,
    which is quadratic on a hostile page.
    """
    for chunk in css.split("}"):
        head, brace, body = chunk.rpartition("{")
        if not brace:
            continue
        decls: dict[str, str] = {}
        for decl in body.split(";"):
            prop, colon, value = decl.partition(":")
            if colon:
                prop = prop.strip()
                # Custom property names are case-sensitive; standard ones are not.
                decls[prop if prop.startswith("--") else prop.lower()] = value.replace("!important", "").strip()
        yield head.rpartition("{")[2].strip(), decls  # "@media x{span.a" -> "span.a"


@dataclass
class _SponsorCss:
    hidden_classes: set[str] = field(default_factory=set)

    @classmethod
    def from_soup(cls, soup: BeautifulSoup) -> _SponsorCss:
        css = "\n".join(s.get_text() for s in soup.find_all("style"))
        if "span." not in css:
            return cls()
        variables: dict[str, str] = {}
        display: dict[str, str] = {}
        hidden: set[str] = set()
        for selector, decls in _css_rules(css):
            variables.update((prop[2:], value) for prop, value in decls.items() if prop.startswith("--"))
            for sel in selector.split(","):
                m = _CSS_SPAN_CLASS_RE.search(sel.strip())
                if m is None:
                    continue
                if "display" in decls:
                    display[m.group(1)] = decls["display"]
                if _CSS_OFFSCREEN_RE.match(decls.get("left", "")):
                    hidden.add(m.group(1))
        for cls_name, value in display.items():
            ref = _CSS_VAR_REF_RE.match(value)
            if ((variables.get(ref.group(1)) if ref else value) or "").lower() == "none":
                hidden.add(cls_name)
        return cls(hidden_classes=hidden)


def _sponsored_label(card: Tag, css: _SponsorCss) -> str:
    """The footer text a sighted user would see, as far as raw HTML can tell ("" if none)."""
    sep = card.select_one(".s-card__sep")
    if sep is None:
        return ""
    spans = [s for s in sep.select("span[class]") if s.select_one("span") is None]
    if spans and css.hidden_classes:
        shown = (s.get_text() for s in spans if not css.hidden_classes.intersection(s.get("class") or []))
        return normalize.clean_text("".join(shown))
    styled = sep.select_one("[style*='svg+xml']")
    if styled is not None:
        m = _SVG_URI_RE.search(styled.get("style", ""))
        if m:
            try:
                return normalize.clean_text(_svg_visible_text(base64.b64decode(m.group(1)).decode("utf-8", "replace")))
            except (binascii.Error, ValueError):
                return ""
    return ""


def _is_sponsored(card: Tag, css: _SponsorCss) -> bool:
    letters = "".join(ch for ch in _sponsored_label(card, css).translate(_HOMOGLYPHS).lower() if ch.isalpha())
    return letters in _SPONSORED_WORDS


def _svg_visible_text(svg: str) -> str:
    out = []
    for t in BeautifulSoup(svg, "html.parser").find_all("text"):
        props = {k.lower(): str(v) for k, v in t.attrs.items()}
        for decl in (t.get("style") or "").split(";"):  # style beats attributes; later declarations win
            if ":" in decl:
                key, value = decl.split(":", 1)
                props[key.strip().lower()] = value.strip()
        if _svg_text_visible(props):
            out.append(t.get_text())
    return "".join(out)


def _svg_text_visible(props: dict[str, str]) -> bool:
    def num(key: str, default: float) -> float:
        m = _NUM_RE.match(props.get(key, ""))
        return float(m.group(1)) if m else default

    if props.get("display") == "none" or props.get("visibility") == "hidden":
        return False
    if num("opacity", 1.0) < 0.05 or num("fill-opacity", 1.0) < 0.05 or num("font-size", 16.0) < 4:
        return False
    fill = props.get("fill", "").strip().lower()
    if fill in ("none", "transparent"):
        return False
    alpha = re.match(r"rgba\([^)]*,\s*([\d.]+)\s*\)", fill)
    return not (alpha and float(alpha.group(1)) < 0.05)
