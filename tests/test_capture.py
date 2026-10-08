"""Offline rendering of archived pages and DOM-derived card regions."""

import pytest

from ebay_sold.capture import capture_cards, capture_tiles, card_regions, render_html
from ebay_sold.models import VISION_CLASSES

from conftest import load_fixture

pytestmark = [pytest.mark.browser]

HOT_WHEELS = "sold_2026-04-26_hot-wheels-r34-zamac"


async def test_regions_cover_every_visible_card(require_browser, tmp_path):
    async with render_html(load_fixture(HOT_WHEELS), load_images=False) as page:
        regions = await card_regions(page)
        # 72 cards in the markup; two are hidden "Shop on eBay" templates.
        assert len(regions) == 70
        assert all(r.item_id and r.item_id.isdigit() for r in regions)
        for r in regions:
            assert set(r.fields) <= set(VISION_CLASSES) - {"listing"}
            for name, box in r.fields.items():
                # every field sits inside its card
                assert box.intersection(r.box) >= 0.95 * box.area, (r.item_id, name)
        first = regions[0]
        assert first.item_id == "358473128518"
        assert first.texts["price"] == "$65.00"
        assert first.texts["sold_date"] == "Sold Apr 25, 2026"
        assert first.texts["shipping"] == "+$9.45 delivery"
        assert first.texts["condition"] == "Brand New"

        tiles = await capture_tiles(page, tmp_path, tile_height=2000, overlap=200)
        assert len(tiles) > 5
        assert tiles[0][1].y == 0 and tiles[1][1].y == 1800

        shots = await capture_cards(page, tmp_path / "cards", regions=regions[:2])
        assert [p.exists() for _, p in shots] == [True, True]


async def test_labels_follow_parser_rules_for_condition_and_struck_prices(require_browser):
    from ebay_sold.parse import parse_search_page

    html = load_fixture("sold_2025-11-02_xbox-one-controller")
    parsed = {item.item_id: item for item in parse_search_page(html).listings}
    async with render_html(html, load_images=False) as page:
        regions = {r.item_id: r for r in await card_regions(page)}
    # 127367489079 has a seller tagline above its "Open Box" condition line.
    assert regions["127367489079"].texts["condition"] == parsed["127367489079"].condition == "Open Box"
    # 376390868846 sold via an accepted Best Offer: only the struck asking price is shown.
    assert parsed["376390868846"].price is None
    assert "price" not in regions["376390868846"].fields
