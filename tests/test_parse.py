"""DOM parser: archived eBay sold-search pages -> Listing rows.

Every expected value below was read off the fixture HTML by hand.
"""

from __future__ import annotations

import time
from datetime import date
from functools import cache

import pytest
from bs4 import BeautifulSoup

from ebay_sold import normalize
from ebay_sold import parse as parse_mod
from ebay_sold.models import Listing
from ebay_sold.parse import TEMPLATE_LISTING_ID, ParsedPage, detect_layout, parse_search_page

from conftest import fixture_names, load_fixture

XBOX = "sold_2025-11-02_xbox-one-controller"
SWITCH = "sold_2026-02-22_switch-pro-controller"
TA1 = "sold_2026-04-16_ta1-adapter"
SEARS = "sold_2026-04-24_sears-roebuck-magazine"
HOT_WHEELS = "sold_2026-04-26_hot-wheels-r34-zamac"
LEGACY = "legacy_s_item_synthetic"

# name -> (real cards, "N results" heading, has next page, cards before "Results matching fewer words")
PAGE_FACTS = {
    XBOX: (60, 6200, True, 60),  # "6,200 + results for xbox one controller"
    SWITCH: (90, 90, False, 90),
    TA1: (60, 67, True, 60),  # 67 results, 60 on page 1, link to page 2
    SEARS: (64, 6, False, 6),
    HOT_WHEELS: (70, 31, False, 31),
}
CAPTURED = {XBOX: date(2025, 11, 2), SWITCH: date(2026, 2, 22), TA1: date(2026, 4, 16),
            SEARS: date(2026, 4, 24), HOT_WHEELS: date(2026, 4, 26)}
# Cards with no sale price on the card: "See price" (price hidden until clicked)
# or "Best offer accepted" (asking price struck through, paid price not shown).
PRICELESS = {XBOX: {"204543799890", "196337548330", "196337547839", "376390868846"}}


@cache
def parsed(name: str) -> ParsedPage:
    return parse_search_page(load_fixture(name))


def by_id(name: str) -> dict[str, Listing]:
    return {item.item_id: item for item in parsed(name).listings}


# --- every fixture ---------------------------------------------------------------


def test_fixture_set_is_covered():
    assert set(fixture_names()) == set(PAGE_FACTS)


def test_one_listing_per_real_card(ebay_page):
    name, html = ebay_page
    soup = BeautifulSoup(html, "lxml")
    real_cards = [li["data-listingid"] for li in soup.select("li.s-card") if li.get("data-listingid") != TEMPLATE_LISTING_ID]
    page = parsed(name)
    assert page.layout == "s-card"
    assert [item.item_id for item in page.listings] == real_cards
    assert len(page.listings) == PAGE_FACTS[name][0]


@pytest.mark.parametrize("name", list(PAGE_FACTS))
def test_page_level_fields(name):
    count, total, has_next, exact = PAGE_FACTS[name]
    page = parsed(name)
    assert page.total_results == total
    assert page.has_next_page is has_next
    assert page.exact_match_count == exact
    assert [item.position for item in page.listings] == list(range(1, count + 1))


def test_every_listing_is_complete(ebay_page):
    name, _ = ebay_page
    for item in parsed(name).listings:
        assert item.title and "Opens in a new window" not in item.title and not item.title.lower().startswith("new listing")
        assert item.title.lower() != "shop on ebay"
        assert item.item_id and item.item_id.isdigit() and len(item.item_id) == 12
        assert item.url == f"https://www.ebay.com/itm/{item.item_id}"
        # One ta1-adapter card really did sell on Oct 28, 2024.
        assert date(2024, 10, 1) <= item.sold_date <= CAPTURED[name]
        assert item.sold_date_text.startswith("Sold ")
        assert item.currency == "USD"
        assert item.price_text
        if item.price is None:
            assert item.item_id in PRICELESS.get(name, set())
            assert item.price_text == "See price" or item.listing_format == "best_offer"
            assert item.original_price is not None
        assert item.price_max is None or item.price_max > item.price
        assert item.shipping is not None and item.shipping >= 0 and item.shipping_text
        assert item.listing_format in ("auction", "buy_it_now", "best_offer")
        assert (item.listing_format == "auction") == (item.bids is not None)
        assert item.seller and item.seller_feedback_pct is not None and item.seller_feedback_count is not None
        assert item.location
        assert item.image_url and item.image_url.startswith("https://")
        assert item.sponsored is False
        assert item.extraction == "dom" and item.confidence is None
    priceless = {item.item_id for item in parsed(name).listings if item.price is None}
    assert priceless == PRICELESS.get(name, set())


def test_parse_is_fast():
    html = load_fixture(XBOX)  # the largest fixture, ~890 KB
    assert len(html) > 850_000
    best = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        parse_search_page(html)
        best = min(best, time.perf_counter() - t0)
    assert best < 1.5


# --- specific cards ------------------------------------------------------------------


def test_hot_wheels_first_card_every_field():
    item = parsed(HOT_WHEELS).listings[0]
    assert item.model_dump() == {
        "item_id": "358473128518",
        "site": "www.ebay.com",
        "title": "2012 2013 Hot Wheels Zamac Nissan Skyline R34 H/T 2000GT - X lot of 2 Zamacs",
        "price": 65.0,
        "price_max": None,
        "original_price": None,
        "currency": "USD",
        "price_text": "$65.00",
        "shipping": 9.45,
        "shipping_text": "+$9.45 delivery",
        "sold_date": date(2026, 4, 25),
        "sold_date_text": "Sold Apr 25, 2026",
        "condition": "Brand New",
        "listing_format": "buy_it_now",
        "bids": None,
        "seller": "mystuffnnowyours",
        "seller_feedback_pct": 100.0,
        "seller_feedback_count": 267,
        "location": "United States",
        "url": "https://www.ebay.com/itm/358473128518",
        "image_url": "https://i.ebayimg.com/images/g/L-QAAeSwur5p5Uvf/s-l500.jpg",
        "sponsored": False,
        "matches_query": True,
        "position": 1,
        "extraction": "dom",
        "confidence": None,
    }
    assert item.total_price == 74.45


def test_price_range():
    item = by_id(HOT_WHEELS)["115585050069"]
    assert (item.price, item.price_max, item.price_text) == (3.75, 23.95, "$3.75 to $23.95")
    assert (item.shipping, item.shipping_text) == (5.60, "+$5.60 delivery in 2-4 days")
    assert item.position == 6


def test_strikethrough_original_price():
    item = by_id(SEARS)["336334905463"]
    assert (item.price, item.original_price, item.price_text) == (13.58, 39.95, "$13.58")
    assert item.listing_format == "best_offer"
    assert (item.seller, item.seller_feedback_pct, item.seller_feedback_count) == ("irvare", 99.8, 3600)


def test_see_price_card():
    item = by_id(XBOX)["204543799890"]
    assert item.price is None and item.price_max is None
    assert (item.price_text, item.original_price, item.currency) == ("See price", 64.99, "USD")
    assert (item.shipping, item.seller, item.seller_feedback_count) == (0.0, "best_buy", 888600)


def test_best_offer_accepted_struck_price_is_not_the_sale_price():
    item = by_id(XBOX)["376390868846"]
    assert item.price is None
    assert (item.original_price, item.price_text, item.listing_format) == (65.99, "$65.99", "best_offer")


def test_auctions_with_bids():
    item = by_id(HOT_WHEELS)["227199862285"]
    assert (item.listing_format, item.bids, item.price) == ("auction", 12, 48.58)
    # The seller row comes after a "Customs services ..." row; delivery is an estimate.
    assert (item.seller, item.seller_feedback_pct, item.seller_feedback_count) == ("smdiecasts", 99.9, 938)
    assert (item.shipping, item.location) == (24.38, "United Kingdom")
    auctions = [(i.item_id, i.bids, i.price) for i in parsed(XBOX).listings if i.listing_format == "auction"]
    assert auctions == [
        ("157429055837", 2, 5.5), ("317465028628", 1, 25.0), ("136667510147", 6, 41.0),
        ("205811664357", 30, 49.0), ("136667321838", 16, 14.5), ("365944647727", 14, 17.5),
        ("205811491019", 19, 31.0), ("136667063038", 26, 46.0), ("127456489953", 19, 43.0),
    ]
    assert sum(i.listing_format == "auction" for i in parsed(HOT_WHEELS).listings) == 13


@pytest.mark.parametrize(("name", "exact", "loose"), [(SEARS, 6, 58), (HOT_WHEELS, 31, 39)])
def test_matches_query_split_at_fewer_words_divider(name, exact, loose):
    page = parsed(name)
    flags = [item.matches_query for item in page.listings]
    assert flags == [True] * exact + [False] * loose
    # eBay's "N results" heading counts only the exact matches.
    assert page.exact_match_count == page.total_results == exact


def test_no_divider_means_everything_matches():
    for name in (XBOX, SWITCH, TA1):
        assert all(item.matches_query for item in parsed(name).listings)


def test_new_listing_tag_is_not_part_of_the_title():
    assert by_id(HOT_WHEELS)["147277980440"].title == "Hot Wheels Zamac Nissan Skyline H/T 2000GT-X"
    assert by_id(XBOX)["326845299627"].title == "Microsoft Xbox One Wireless Controller - Blue Vortex 1708 tested works free ship"


# (fixture, card) -> the seller-written subtitle drawn above the condition line.
SELLER_SUBTITLES = {
    (XBOX, "127367489079"): "Hybrid D-Pad, 40-Hour Battery Life, EP2-29935",
    (XBOX, "165457596316"): "Brand New! Free Shipping!",
    (XBOX, "386625264645"): "BRAND NEW IN BOX! SUPER FAST ON SHIPPING!",
    (TA1, "296754137988"): "Extended 1 Year Warranty",
    (TA1, "318020499598"): "USA Seller - Antigravity Gear Backed by Real Riders",
    (TA1, "317522053065"): "StarCycle - Your Antigravity Superstore on eBay!",
    (TA1, "326839196164"): "FREE AND FAST SHIPPING. FREE RETURNS. 100% AUTHENTIC!",
}


def _subtitles(name: str) -> dict[str, list[str]]:
    soup = BeautifulSoup(load_fixture(name), "lxml")
    return {
        li["data-listingid"]: [normalize.clean_text(s.get_text(" ")) for s in li.select(".s-card__subtitle")]
        for li in soup.select("ul.srp-results > li.s-card")
    }


def test_condition_is_the_last_subtitle():
    # A seller-written subtitle sits above the condition line on these cards.
    assert by_id(TA1)["296754137988"].condition == "Brand New"  # "Extended 1 Year Warranty"
    assert by_id(XBOX)["127367489079"].condition == "Open Box"  # "Hybrid D-Pad, 40-Hour Battery Life, ..."
    assert by_id(XBOX)["165457596316"].condition == "Brand New"  # "Brand New! Free Shipping!"
    assert by_id(TA1)["257396492770"].condition == "New – Open box"
    # 14 sears cards (magazines) show no condition at all.
    assert sum(item.condition is None for item in parsed(SEARS).listings) == 14


def test_condition_labels_are_told_apart_from_seller_subtitles():
    # Every subtitle line in the five fixtures: eBay's labels are recognised, the 7 seller lines are not.
    seen: dict[tuple[str, str], list[str]] = {}
    for name in PAGE_FACTS:
        for item_id, lines in _subtitles(name).items():
            seen[(name, item_id)] = lines
            labels = [line for line in lines if parse_mod._is_condition_label(line)]
            assert len(labels) == (1 if lines else 0), (name, item_id, lines)
            assert by_id(name)[item_id].condition == (labels[0] if labels else None)
    taglines = {key: lines[0] for key, lines in seen.items() if len(lines) > 1}
    assert taglines == SELLER_SUBTITLES
    assert not any(parse_mod._is_condition_label(t) for t in SELLER_SUBTITLES.values())
    labels = {line for lines in seen.values() for line in lines} - set(SELLER_SUBTITLES.values())
    assert labels == {"Brand New", "Pre-Owned", "New – Open box", "Open Box", "Parts Only", "Very Good - Refurbished",
                      "New (Other)", "Excellent - Refurbished"}


@pytest.mark.parametrize(
    "label",
    ["Pre-owned - Good", "New with tags", "New other (see details)", "For parts or not working", "Certified - Refurbished",
     "Used", "Like New", "Graded - PSA 10", "Gebraucht", "Neu: Sonstige (siehe Artikelbeschreibung)", "D'occasion",
     "Neuf avec étiquettes", "Usato", "Nuovo (altro)", "De segunda mano", "Reacondicionado"],
)
def test_condition_label_vocabulary(label):
    assert parse_mod._is_condition_label(label)


@pytest.mark.parametrize(
    ("subtitles", "expected"),
    [
        (["Rare 1935 issue - great ads!"], None),  # a seller line alone is not a condition
        (["Brand New! Free Shipping!"], None),
        (["Pre-Owned · Nintendo"], "Pre-Owned"),
        (["Extended 1 Year Warranty", "Brand New"], "Brand New"),
        (["Brand New", "Free shipping, fast!"], "Brand New"),  # a known label wins wherever it is
        (["Ships today from Ohio", "Wie neu, kaum benutzt"], "Wie neu, kaum benutzt"),  # unknown wording: the last line
    ],
)
def test_condition_from_subtitles(subtitles, expected):
    assert parse_search_page(_s_card(subtitles=subtitles)).listings[0].condition == expected


def test_top_rated_seller_badge_does_not_confuse_seller():
    item = by_id(XBOX)["135118584980"]
    assert (item.seller, item.seller_feedback_pct, item.seller_feedback_count) == ("goldstar_tech", 99.3, 291900)
    assert item.condition == "Very Good - Refurbished"


# --- the obfuscated "Sponsored" footer --------------------------------------------------


@pytest.mark.parametrize("name", [HOT_WHEELS, SEARS, TA1, XBOX])
def test_sponsored_rule_flags_only_the_card_chromium_shows_as_sponsored(name):
    # Rendering showed: in each of these pages exactly one hidden template card
    # draws "Sponsored" in its footer; every real card's footer is blank.
    soup = BeautifulSoup(load_fixture(name), "lxml")
    css = parse_mod._SponsorCss.from_soup(soup)
    flagged = [li for li in soup.select("li.s-card") if parse_mod._is_sponsored(li, css)]
    assert [li["data-listingid"] for li in flagged] == [TEMPLATE_LISTING_ID]
    assert flagged[0].find_parent("ul", class_="srp-results") is None


def test_font_obfuscated_footer_is_not_decided():
    # Feb 2026 pages draw the label in a per-page web font; raw HTML cannot tell blank glyphs apart.
    soup = BeautifulSoup(load_fixture(SWITCH), "lxml")
    css = parse_mod._SponsorCss.from_soup(soup)
    assert not any(parse_mod._is_sponsored(li, css) for li in soup.select("li.s-card"))


def test_sponsored_card_in_results_is_marked():
    # Move the template's footer (which renders "Sponsored") onto the first real card.
    soup = BeautifulSoup(load_fixture(HOT_WHEELS), "lxml")
    css = parse_mod._SponsorCss.from_soup(soup)
    template = next(li for li in soup.select("li.s-card") if parse_mod._is_sponsored(li, css))
    first = soup.select_one('ul.srp-results > li.s-card[data-listingid="358473128518"]')
    first.select_one(".s-card__sep").replace_with(template.select_one(".s-card__sep"))
    page = parse_search_page(str(soup))
    assert [item.sponsored for item in page.listings[:3]] == [True, False, False]
    assert sum(item.sponsored for item in page.listings) == 1


# --- legacy "s-item" layout -------------------------------------------------------------


def test_legacy_s_item_layout():
    page = parse_search_page(load_fixture(LEGACY), today=date(2024, 3, 10))
    assert (page.layout, page.total_results, page.has_next_page, page.exact_match_count) == ("s-item", 3, False, 3)
    ids = [item.item_id for item in page.listings]
    assert ids == ["185123456789", "266234567890", "394987654321", "126345678901"]  # template skipped
    assert [item.position for item in page.listings] == [1, 2, 3, 4]
    assert [item.matches_query for item in page.listings] == [True, True, True, False]

    auction, bin_, ranged, offer = page.listings
    assert auction.model_dump(exclude={"site", "extraction", "confidence"}) == {
        "item_id": "185123456789",
        "title": "Vintage Pyrex 403 Butterprint Mixing Bowl 2.5 Qt Turquoise on White",
        "price": 45.0, "price_max": None, "original_price": None, "currency": "USD", "price_text": "$45.00",
        "shipping": 12.85, "shipping_text": "+$12.85 shipping",
        "sold_date": date(2024, 3, 3), "sold_date_text": "Sold Mar 3, 2024",
        "condition": "Pre-Owned", "listing_format": "auction", "bids": 11,
        "seller": "kitchen_kollectibles", "seller_feedback_pct": 99.8, "seller_feedback_count": 2345,
        "location": "United States", "url": "https://www.ebay.com/itm/185123456789",
        "image_url": "https://i.ebayimg.com/thumbs/images/g/xYzAAOSwAbCdEfGh/s-l300.jpg",
        "sponsored": False, "matches_query": True, "position": 1,
    }
    assert bin_.title == "Pyrex Primary Colors Cinderella Bowl Set of 4 Complete"  # "New Listing" tag dropped
    assert (bin_.price, bin_.original_price, bin_.shipping, bin_.listing_format) == (89.99, 129.99, 0.0, "buy_it_now")
    assert (bin_.location, bin_.condition, bin_.seller_feedback_count) == ("Canada", "Brand New", 512)
    assert bin_.image_url == "https://i.ebayimg.com/thumbs/images/g/qRsAAOSwTuVwXyZa/s-l300.jpg"  # protocol-relative src
    assert (ranged.price, ranged.price_max, ranged.listing_format) == (3.75, 23.95, "best_offer")
    assert (ranged.condition, ranged.location, ranged.seller_feedback_count) == ("Brand New", "China", 12300)
    assert (offer.price, offer.original_price, offer.listing_format) == (None, 60.0, "best_offer")
    assert (offer.shipping, offer.shipping_text) == (None, "Shipping not specified")
    assert offer.image_url == "https://i.ebayimg.com/thumbs/images/g/AbCAAOSwDeFgHiJk/s-l300.jpg"  # data-src over placeholder


# --- layout detection and non-result pages --------------------------------------------


def test_detect_layout():
    assert detect_layout(load_fixture(HOT_WHEELS)) == "s-card"
    assert detect_layout(BeautifulSoup(load_fixture(XBOX), "lxml")) == "s-card"
    assert detect_layout(load_fixture(LEGACY)) == "s-item"
    assert detect_layout("<html><body><p>hello</p></body></html>") == "unknown"
    assert detect_layout("") == "unknown"


NON_RESULT_PAGES = {
    "empty": "",
    "whitespace": "  \n ",
    "akamai_block": "<html><head><title>Error Page | eBay</title></head><body><h1>Access Denied</h1>"
                    "<p>You don't have permission to access this page.</p><p>Reference #18.6f0e1002.1714000000.1a2b3c</p></body></html>",
    "challenge": "<html><head><title>Security Measure</title></head><body><h1>Please verify yourself to continue</h1>"
                 "<iframe src='https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html'></iframe></body></html>",
    "interstitial": "<html><head><title>Pardon Our Interruption...</title></head><body>As you were browsing something about "
                    "your browser made us think you were a bot.</body></html>",
    "signin": "<html><head><title>Sign in or Register | eBay</title></head><body><form id='signin-form'></form></body></html>",
    "json": '{"error": "rate limited", "status": 429}',
    "garbage": "<<<>>> \x00 not html",
}


@pytest.mark.parametrize("html", NON_RESULT_PAGES.values(), ids=NON_RESULT_PAGES.keys())
def test_non_result_pages_yield_nothing(html):
    page = parse_search_page(html)
    assert page == ParsedPage(listings=[], total_results=None, has_next_page=None, layout="unknown", exact_match_count=0)


def test_unsupported_site_is_rejected():
    with pytest.raises(ValueError):
        parse_search_page(load_fixture(HOT_WHEELS), site="www.ebay.example")


# --- site-specific conversions (minimal synthetic s-card markup) ------------------------


def _card(price: str, caption: str, shipping: str, listing_id: str = "123456789012") -> str:
    return f"""<ul class="srp-results"><li class="s-card" data-listingid="{listing_id}">
      <a class="s-card__link" href="/itm/{listing_id}?hash=x"><div class="s-card__title"><span class="su-styled-text">Widget</span></div></a>
      <div class="s-card__caption"><span class="su-styled-text">{caption}</span></div>
      <div class="su-card-container__attributes__primary">
        <div class="s-card__attribute-row"><span class="s-card__price">{price}</span></div>
        <div class="s-card__attribute-row"><span class="su-styled-text">{shipping}</span></div>
        <div class="s-card__attribute-row"><span class="su-styled-text">Delivery in 2-4 days</span></div>
      </div></li></ul>"""


def test_site_currency_and_day_first_dates():
    uk = parse_search_page(_card("£12.50", "Sold 05/04/2026", "+£3.20 postage"), site="ebay.co.uk").listings[0]
    assert (uk.price, uk.currency, uk.shipping, uk.sold_date) == (12.5, "GBP", 3.2, date(2026, 4, 5))
    assert uk.url == "https://www.ebay.co.uk/itm/123456789012"
    ca = parse_search_page(_card("$40.00", "Sold Apr 25, 2026", "Free shipping"), site="www.ebay.ca").listings[0]
    assert (ca.price, ca.currency, ca.shipping) == (40.0, "CAD", 0.0)
    us_marked = parse_search_page(_card("C $40.00", "Sold Apr 25, 2026", "Free delivery")).listings[0]
    assert us_marked.currency == "CAD"
    assert parse_search_page(_card("$1.00", "Sold 05/04/2026", "")).listings[0].sold_date == date(2026, 5, 4)


def test_year_less_sold_date_uses_today():
    item = parse_search_page(_card("$1.00", "Sold Dec 30", ""), today=date(2026, 1, 5)).listings[0]
    assert item.sold_date == date(2025, 12, 30)


def test_delivery_window_is_not_a_shipping_cost():
    # A row with digits but no currency ("Delivery in 2-4 days") must not become $2.00 shipping.
    item = parse_search_page(_card("$1.00", "Sold Apr 25, 2026", "Returns accepted")).listings[0]
    assert (item.shipping, item.shipping_text) == (None, "Delivery in 2-4 days")


def _s_card(*, rows: tuple[str, ...] = (), subtitles: tuple[str, ...] | list[str] = (), img: str = "",
            head: str = "", price: str = "$12.00") -> str:
    """One minimal s-card results page with the given attribute rows / subtitle lines."""
    row_html = "".join(f'<div class="s-card__attribute-row"><span class="su-styled-text">{r}</span></div>' for r in rows)
    sub_html = "".join(
        f'<div class="s-card__subtitle-row"><div class="s-card__subtitle"><span class="su-styled-text secondary default">'
        f"{s}</span></div></div>" for s in subtitles
    )
    return f"""{head}<ul class="srp-results"><li class="s-card" data-listingid="123456789012">
      <div class="s-card__title"><span class="su-styled-text">Widget</span></div>{sub_html}{img}
      <div class="s-card__caption"><span class="su-styled-text">Sold  Apr 25, 2026</span></div>
      <div class="su-card-container__attributes__primary">
        <div class="s-card__attribute-row"><span class="s-card__price">{price}</span></div>{row_html}
      </div></li></ul>"""


@pytest.mark.parametrize(
    ("site", "row", "expected"),
    [
        ("www.ebay.com", "Located in United States", "United States"),
        ("www.ebay.com", "from China", "China"),
        ("www.ebay.de", "aus China", "China"),
        ("www.ebay.it", "da Cina", "Cina"),
        ("www.ebay.fr", "Provenance : Chine", "Chine"),
        ("www.ebay.fr", "Provenance : Chine", "Chine"),
        ("www.ebay.fr", "Provenance: Royaume-Uni", "Royaume-Uni"),
        ("www.ebay.com", "from $9.99 shipping", None),  # a price, not a place
    ],
)
def test_location_rows(site, row, expected):
    item = parse_search_page(_s_card(rows=("Free delivery", row), price="EUR 12,00"), site=site).listings[0]
    assert item.location == expected


@pytest.mark.parametrize(
    ("site", "src", "expected"),
    [
        ("www.ebay.com", "/thumbs/x.jpg", "https://www.ebay.com/thumbs/x.jpg"),
        ("www.ebay.co.uk", "/thumbs/x.jpg", "https://www.ebay.co.uk/thumbs/x.jpg"),
        ("www.ebay.com", "//i.ebayimg.com/images/g/x/s-l500.jpg", "https://i.ebayimg.com/images/g/x/s-l500.jpg"),
        ("www.ebay.com", "http://i.ebayimg.com/images/g/x/s-l500.jpg", "https://i.ebayimg.com/images/g/x/s-l500.jpg"),
        ("www.ebay.com", "javascript:void(0)", None),
        ("www.ebay.com", "http://[::1/x.jpg", None),  # malformed: no image, but the card survives
    ],
)
def test_image_url_is_absolute_https(site, src, expected):
    page = parse_search_page(_s_card(img=f'<img class="s-card__image" src="{src}">', price="£1.00"), site=site)
    assert page.listings[0].image_url == expected


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("<span class='BOLD'>1,234,567+</span> results for <span class='BOLD'>iphone</span>", 1234567),
        ("1.234 Ergebnisse für <span>lego 42083</span>", 1234),
        ("1 234 résultats pour 12 pièces", 1234),
        ("0 results for xyzzy 2024", 0),
        ("No exact matches found for 123 widget", None),  # the digits are search words, not a count
        ("1" * 4400 + " results", None),  # longer than int() accepts: no count, but no exception either
    ],
)
def test_result_count_heading(heading, expected):
    page = parse_search_page(_s_card(head=f'<h1 class="srp-controls__count-heading">{heading}</h1>'))
    assert page.total_results == expected
    assert [item.item_id for item in page.listings] == ["123456789012"]


def test_sponsored_css_rules():
    css = """
      :root{--pd-a: none;} :root{--pd-b: block;}
      span.a {display: var(--pd-a);} span.b {display: var(--pd-b);} span.c {display: var(--pd-later);}
      span.d { position: absolute; left: -2000px; top: auto; }
      @media (min-width: 1px) { div span.e, span.f {display: NONE !important} }
      span.g {left: -20px} span.h {display: block}
      :root{--pd-later: none}
    """
    soup = BeautifulSoup(f"<style>{css}</style>", "lxml")
    assert parse_mod._SponsorCss.from_soup(soup).hidden_classes == {"a", "c", "d", "e", "f"}


@pytest.mark.parametrize(
    "css",
    ["span.a{" * 20000, "span.a{" + "x" * 300_000, "--a" * 100_000, "span.a" * 100_000 + "{display:none}"],
    ids=["unclosed-rules", "unclosed-body", "variable-run", "long-selector"],
)
def test_hostile_css_parses_in_linear_time(css):
    t0 = time.perf_counter()
    page = parse_search_page(f"<style>{css}</style>" + _s_card())
    assert time.perf_counter() - t0 < 1.0
    assert len(page.listings) == 1 and page.listings[0].sponsored is False


# --- the DOM labels used for vision training agree with the parser --------------------


@pytest.mark.browser
@pytest.mark.parametrize("name", list(PAGE_FACTS))
async def test_rendered_card_regions_agree_with_parser(require_browser, name):
    from ebay_sold.capture import card_regions, render_html

    page_result = parsed(name)
    listings = {item.item_id: item for item in page_result.listings}
    async with render_html(load_fixture(name), load_images=False) as page:
        regions = await card_regions(page)
        footers = await page.evaluate(
            "() => [...document.querySelectorAll('li.s-card')].map(c => "
            "[c.getAttribute('data-listingid'), (c.querySelector('.s-card__sep') || {innerText: ''}).innerText])"
        )

    assert [r.item_id for r in regions] == [item.item_id for item in page_result.listings]
    condition_mismatch = set()
    for region in regions:
        item = listings[region.item_id]
        texts = region.texts
        assert normalize.clean_text(texts["title"]) == item.title, region.item_id
        money = normalize.parse_money(texts.get("price"), "USD")
        if item.price is None:
            # "See price", or an accepted Best Offer whose struck asking price is not labelled
            assert money is None, region.item_id
        else:
            assert (money.amount, money.amount_max) == (item.price, item.price_max), region.item_id
        assert normalize.parse_sold_date(texts["sold_date"], CAPTURED[name]) == item.sold_date, region.item_id
        assert normalize.parse_shipping(texts.get("shipping"), "USD") == item.shipping, region.item_id
        assert normalize.clean_text(texts.get("shipping")).replace(" ", "") == item.shipping_text.replace(" ", "")
        if normalize.clean_text(texts.get("condition")) != (item.condition or ""):
            condition_mismatch.add(region.item_id)

    if name in (TA1, SEARS, HOT_WHEELS):
        # The raw-HTML "Sponsored" rule reproduces what Chromium draws in every
        # footer (templates included). Older pages draw it as an SVG or in a web
        # font, where innerText says nothing about what is visible.
        soup = BeautifulSoup(load_fixture(name), "lxml")
        css = parse_mod._SponsorCss.from_soup(soup)

        def letters(text: str) -> str:
            return "".join(ch for ch in text.translate(parse_mod._HOMOGLYPHS).lower() if ch.isalpha())

        raw = [letters(parse_mod._sponsored_label(li, css)) for li in soup.select("li.s-card")]
        assert raw == [letters(text) for _, text in footers]
        assert raw.count("sponsored") == 1

    # capture.card_regions picks the condition line with the parser's own rule,
    # so seller taglines never become "condition" labels.
    assert not condition_mismatch, sorted(condition_mismatch)
