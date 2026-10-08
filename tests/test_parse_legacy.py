"""Real 2024 result pages in eBay's older ``li.s-item`` layout (Wayback Machine captures)."""

from datetime import date

import pytest

from ebay_sold import normalize
from ebay_sold.parse import parse_search_page

from conftest import legacy_fixture_names, load_fixture

DRUMS = "legacy_2024-02-14_xbox-360-drums-guitar"
BUTTONS = "legacy_2024-03-06_vintage-lansing-buttons"
CAPTURED = {DRUMS: date(2024, 2, 14), BUTTONS: date(2024, 3, 6)}
# (cards, cards above "Results matching fewer words", eBay's result count)
FACTS = {DRUMS: (87, 29, 29), BUTTONS: (70, 17, 17)}


@pytest.mark.parametrize("name", legacy_fixture_names())
def test_every_legacy_card_is_read(name):
    page = parse_search_page(load_fixture(name), today=CAPTURED[name])
    cards, exact, total = FACTS[name]
    assert sum(item.condition is None for item in page.listings) == (1 if name == BUTTONS else 0)
    assert page.layout == "s-item"
    assert (len(page.listings), page.exact_match_count, page.total_results) == (cards, exact, total)
    assert [item.position for item in page.listings] == list(range(1, cards + 1))
    for item in page.listings:
        assert item.item_id and item.item_id.isdigit()
        assert item.title
        assert date(2023, 1, 1) <= item.sold_date <= CAPTURED[name]
        assert item.shipping is not None
        assert item.seller
        # A price is missing only where eBay shows just the struck-through asking price.
        assert (item.price is None) == (item.original_price is not None and item.price_text == f"${item.original_price:.2f}")


def test_known_legacy_card():
    page = parse_search_page(load_fixture(DRUMS), today=CAPTURED[DRUMS])
    item = next(i for i in page.listings if i.item_id == "325985482586")
    assert item.title == "Harmonix Wired XBox 360 Rock Band Guitar Hero Drum Set And Guitar"
    assert (item.price, item.currency, item.shipping) == (90.0, "USD", 11.97)
    assert item.sold_date == date(2024, 1, 28)
    assert (item.condition, item.listing_format, item.bids) == ("Pre-Owned", "auction", 1)
    assert (item.seller, item.seller_feedback_count) == ("needfulthangs444", 14)
    assert item.url == "https://www.ebay.com/itm/325985482586"


def test_accepted_best_offer_has_no_sale_price():
    page = parse_search_page(load_fixture(BUTTONS), today=CAPTURED[BUTTONS])
    item = next(i for i in page.listings if i.item_id == "225993244907")
    assert (item.price, item.original_price, item.listing_format) == (None, 9.99, "best_offer")


@pytest.mark.browser
@pytest.mark.parametrize("name", legacy_fixture_names())
async def test_rendered_legacy_regions_agree_with_parser(require_browser, name):
    from ebay_sold.capture import card_regions, render_html

    listings = {i.item_id: i for i in parse_search_page(load_fixture(name), today=CAPTURED[name]).listings}
    async with render_html(load_fixture(name), load_images=False) as page:
        regions = await card_regions(page)
    assert [r.item_id for r in regions] == list(listings)
    for region in regions:
        item = listings[region.item_id]
        assert normalize.clean_text(region.texts["title"]) == item.title
        assert normalize.parse_sold_date(region.texts["sold_date"], CAPTURED[name]) == item.sold_date
        if item.price is None:
            assert "price" not in region.fields
        else:
            assert normalize.parse_money(region.texts["price"], "USD").amount == item.price
        assert normalize.parse_shipping(region.texts.get("shipping"), "USD") == item.shipping
