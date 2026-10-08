"""Claude vision extractor: request shape, normalization, error paths, slicing.

No network and no API key: a fake client records each request and replays
canned responses shaped like the SDK's (``stop_reason``, ``stop_details``,
``content`` blocks with ``.type`` / ``.text``).
"""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from ebay_sold.config import LLMSettings
from ebay_sold.llm import (
    CHUNK_OVERLAP,
    FALLBACK_BETA,
    MAX_CHUNK_HEIGHT,
    MAX_WIDTH,
    RESPONSE_SCHEMA,
    TEXT_FIELDS,
    ClaudeExtractor,
    LLMExtractionError,
    chunk_spans,
    estimate_cost,
    image_tokens,
    merge_chunks,
    split_screenshot,
)

TODAY = date(2026, 5, 1)

# The first card of the hot-wheels fixture, as the DOM parser reads it.
HOT_WHEELS = {
    "title": "2012 2013 Hot Wheels Zamac Nissan Skyline R34 H/T 2000GT - X lot of 2 Zamacs",
    "price_text": "$65.00",
    "original_price_text": None,
    "sold_date_text": "Sold  Apr 25, 2026",
    "shipping_text": "+$9.45 delivery",
    "condition": "Brand New",
    "format_text": "Buy It Now",
    "bids_text": None,
    "seller_text": "mystuffnnowyours 100% positive (267)",
    "location_text": "Located in United States",
    "below_fewer_words_heading": False,
}


def card(title: str, **fields: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {f: None for f in TEXT_FIELDS}
    entry["below_fewer_words_heading"] = False
    entry.update(title=title, **fields)
    return entry


def response(payload: Any = None, *, stop_reason: str = "end_turn", stop_details: Any = None,
             text: str | None = None, model: str = "claude-opus-5-5") -> SimpleNamespace:
    body = text if text is not None else json.dumps(payload)
    content = [] if stop_reason == "refusal" else [
        SimpleNamespace(type="thinking", thinking=""),  # thinking is always on for claude-opus-5-5
        SimpleNamespace(type="text", text=body),
    ]
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=stop_details,
        content=content,
        model=model,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )


def results(*cards: dict[str, Any], page_issue: str | None = None) -> SimpleNamespace:
    return response({"page_issue": page_issue, "listings": list(cards)})


class FakeMessages:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply(kwargs) if callable(reply) else reply


class FakeClient:
    def __init__(self, *replies: Any) -> None:
        self.messages = FakeMessages(list(replies))
        self.beta = SimpleNamespace(messages=self.messages)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


def png(width: int = 400, height: int = 300) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (width, height), "white")
    # Horizontal stripes so slices differ and can be located by colour.
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    for y in range(0, height, 100):
        draw.rectangle((0, y, width, y + 10), fill=(y // 100 % 256, 80, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sent_image(call: dict[str, Any]) -> tuple[str, bytes]:
    block = call["messages"][0]["content"][0]
    return block["source"]["media_type"], base64.b64decode(block["source"]["data"], validate=True)


def size_of(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return img.size


# --- request shape -------------------------------------------------------------


def _assert_strict(schema: dict[str, Any]) -> None:
    """Every object in the schema is closed and lists all its properties as required."""
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
    for value in schema.values():
        if isinstance(value, dict):
            _assert_strict(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _assert_strict(item)


def test_request_shape():
    image = png(400, 300)
    client = FakeClient(results(HOT_WHEELS))
    ClaudeExtractor(client=client).extract(image, today=TODAY)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == LLMSettings().max_tokens
    assert isinstance(call["system"], str) and call["system"]

    # One user turn; no assistant prefill (400 on this model).
    assert [m["role"] for m in call["messages"]] == ["user"]
    image_block, text_block = call["messages"][0]["content"]
    assert image_block["type"] == "image"
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    assert "\n" not in image_block["source"]["data"]
    assert base64.b64decode(image_block["source"]["data"]) == image  # small images pass through untouched
    assert text_block["type"] == "text"
    assert "www.ebay.com" in text_block["text"] and "null" in text_block["text"]
    assert "slice 1 of" not in text_block["text"]  # single image, no slice note

    out = call["output_config"]
    assert out["effort"] == "low"
    assert out["format"]["type"] == "json_schema"
    schema = out["format"]["schema"]
    assert schema is RESPONSE_SCHEMA
    _assert_strict(schema)
    item = schema["properties"]["listings"]["items"]
    for field in TEXT_FIELDS:
        assert item["properties"][field]["type"] == ["string", "null"]

    # Server-side refusal fallback, and nothing that 400s on claude-opus-5-5.
    assert call["fallbacks"] == "default"
    assert call["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    for forbidden in ("thinking", "tool_choice", "tools", "temperature"):
        assert forbidden not in call


def test_settings_are_honoured_and_fallback_only_where_supported():
    settings = LLMSettings(model="claude-haiku-5-5", effort="medium", max_tokens=2048)
    client = FakeClient(results())
    ClaudeExtractor(settings, client=client).extract(png(), today=TODAY)
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == 2048
    assert call["output_config"]["effort"] == "medium"
    assert "fallbacks" not in call and "betas" not in call

    client = FakeClient(results())
    ClaudeExtractor(client=client, server_fallback=False).extract(png(), today=TODAY)
    assert "fallbacks" not in client.calls[0]


# --- normalization ---------------------------------------------------------------


def test_normalizes_hot_wheels_card():
    client = FakeClient(results(HOT_WHEELS))
    [item] = ClaudeExtractor(client=client).extract(png(), today=TODAY)

    assert item.title == HOT_WHEELS["title"]
    assert item.price == 65.0 and item.price_max is None and item.currency == "USD"
    assert item.price_text == "$65.00"
    assert item.sold_date == date(2026, 4, 25)
    assert item.sold_date_text == "Sold Apr 25, 2026"
    assert item.shipping == 9.45 and item.total_price == 74.45
    assert item.condition == "Brand New"
    assert item.listing_format == "buy_it_now" and item.bids is None
    assert (item.seller, item.seller_feedback_pct, item.seller_feedback_count) == ("mystuffnnowyours", 100.0, 267)
    assert item.location == "United States"
    assert item.extraction == "llm" and item.confidence is None
    assert item.item_id is None and item.url is None
    assert item.position == 1 and item.matches_query is True


def test_agrees_with_dom_parser_on_the_same_card():
    from conftest import load_fixture

    from ebay_sold.parse import parse_search_page

    dom = parse_search_page(load_fixture("sold_2026-04-26_hot-wheels-r34-zamac"), site="www.ebay.com")
    dom_first = (dom.listings if hasattr(dom, "listings") else dom)[0]
    [llm_first] = ClaudeExtractor(client=FakeClient(results(HOT_WHEELS))).extract(png(), today=TODAY)
    fields = ("title", "price", "currency", "shipping", "sold_date", "condition", "listing_format",
              "seller", "seller_feedback_pct", "seller_feedback_count", "location", "price_text",
              "shipping_text", "sold_date_text")
    assert {f: getattr(llm_first, f) for f in fields} == {f: getattr(dom_first, f) for f in fields}


def test_auction_range_strikethrough_and_hidden_price():
    cards = [
        card("NEW LISTING Xbox One Controller", price_text="$23.50", sold_date_text="Sold  Apr 2, 2026",
             format_text="Auction", bids_text="11 bids", shipping_text="Free delivery"),
        card("Switch Pro Controller lot", price_text="$3.75 to $23.95", original_price_text="$39.99",
             format_text="or Best Offer", condition="Pre-Owned"),
        card("Sears Roebuck magazine", price_text="See price", sold_date_text="Sold  Mar 30, 2026"),
    ]
    a, b, c = ClaudeExtractor(client=FakeClient(results(*cards))).extract(png(), today=TODAY)
    assert a.title == "Xbox One Controller"  # tag dropped
    assert (a.listing_format, a.bids, a.price, a.shipping) == ("auction", 11, 23.5, 0.0)
    assert (b.price, b.price_max, b.original_price, b.listing_format) == (3.75, 23.95, 39.99, "best_offer")
    assert b.sold_date is None and b.sold_date_text is None  # not shown -> None, never guessed
    assert (c.price, c.currency, c.sold_date) == (None, "USD", date(2026, 3, 30))
    assert [x.position for x in (a, b, c)] == [1, 2, 3]


# Card 376390868846 of the xbox fixture: eBay strikes through the asking price
# of an accepted best offer and does not show what was paid.
BEST_OFFER_ACCEPTED = {
    "title": "Xbox Wireless Controller – DOOM The Dark Ages Limited Edition",
    "sold_date_text": "Sold  Nov 1, 2025",
    "format_text": "Best offer accepted",
    "shipping_text": "Free delivery",
    "condition": "Open Box",
}


@pytest.mark.parametrize("price_text,original_price_text", [
    (None, "$65.99"),      # as instructed: the struck price is not the sale price
    ("$65.99", "$65.99"),  # the same struck price reported in both fields
    ("$65.99", None),      # the strike line missed; "Best offer accepted" still says it is the asking price
])
def test_struck_through_asking_price_is_not_the_sale_price(price_text, original_price_text):
    raw = card(price_text=price_text, original_price_text=original_price_text, **BEST_OFFER_ACCEPTED)
    [item] = ClaudeExtractor(client=FakeClient(results(raw))).extract(png(), today=TODAY)
    assert (item.price, item.original_price, item.price_text) == (None, 65.99, "$65.99")
    assert (item.currency, item.listing_format, item.total_price) == ("USD", "best_offer", None)


def test_struck_price_agrees_with_dom_parser():
    from conftest import load_fixture

    from ebay_sold.parse import parse_search_page

    dom = parse_search_page(load_fixture("sold_2025-11-02_xbox-one-controller"), site="www.ebay.com")
    dom_card = next(x for x in dom.listings if x.item_id == "376390868846")
    raw = card(price_text=None, original_price_text="$65.99", **BEST_OFFER_ACCEPTED)
    [item] = ClaudeExtractor(client=FakeClient(results(raw))).extract(png(), today=TODAY)
    fields = ("title", "price", "price_max", "original_price", "currency", "price_text", "listing_format", "sold_date",
              "shipping", "condition")
    assert {f: getattr(item, f) for f in fields} == {f: getattr(dom_card, f) for f in fields}


def test_prompt_and_schema_say_where_a_struck_price_goes():
    client = FakeClient(results())
    ClaudeExtractor(client=client).extract(png())
    prompt = client.calls[0]["messages"][0]["content"][1]["text"]
    assert "line through it" in prompt and "price_text is null" in prompt
    hints = RESPONSE_SCHEMA["properties"]["listings"]["items"]["properties"]
    assert "struck through" in hints["price_text"]["description"]
    assert "struck through" in hints["original_price_text"]["description"]


def test_list_price_beside_a_sale_price_stays_the_original_price():
    raw = card("Switch Pro Controller", price_text="$22.57", original_price_text="$28.21", format_text="Buy It Now")
    [item] = ClaudeExtractor(client=FakeClient(results(raw))).extract(png(), today=TODAY)
    assert (item.price, item.original_price, item.price_text) == (22.57, 28.21, "$22.57")


def test_condition_keeps_only_the_label_before_the_brand():
    raw = card("Switch Pro Controller", price_text="$40.00", condition="Pre-Owned · Nintendo")
    [item] = ClaudeExtractor(client=FakeClient(results(raw))).extract(png(), today=TODAY)
    assert item.condition == "Pre-Owned"  # what parse.py stores for the same subtitle


def test_new_listing_tag_is_stripped_only_as_a_whole_word():
    cards = [card("New Listings Weekly magazine 1950", price_text="$5.00"),
             card("NEW LISTING  Sears Roebuck catalog", price_text="$6.00")]
    a, b = ClaudeExtractor(client=FakeClient(results(*cards))).extract(png(), today=TODAY)
    assert a.title == "New Listings Weekly magazine 1950"
    assert b.title == "Sears Roebuck catalog"


def test_non_us_site_uses_site_currency_and_day_first_dates():
    cards = [card("Hot Wheels R34", price_text="£12.50", sold_date_text="Sold  25 Apr 2026",
                  shipping_text="+£3.20 postage"),
             card("Hot Wheels R34 ii", price_text="$20.00", sold_date_text="Sold 05/04/2026")]
    a, b = ClaudeExtractor(client=FakeClient(results(*cards))).extract(png(), site="ebay.co.uk", today=TODAY)
    assert (a.site, a.currency, a.price, a.shipping, a.sold_date) == ("www.ebay.co.uk", "GBP", 12.5, 3.2,
                                                                      date(2026, 4, 25))
    assert (b.currency, b.sold_date) == ("GBP", date(2026, 4, 5))  # bare "$" takes the site currency


def test_cards_after_fewer_words_heading_do_not_match_query():
    cards = [card("exact one", price_text="$1.00"),
             card("loose one", price_text="$2.00", below_fewer_words_heading=True),
             card("loose two", price_text="$3.00")]
    out = ClaudeExtractor(client=FakeClient(results(*cards))).extract(png(), today=TODAY)
    assert [x.matches_query for x in out] == [True, False, False]


def test_untitled_cards_are_dropped():
    out = ClaudeExtractor(client=FakeClient(results(card("", price_text="$5.00"), card("kept")))).extract(
        png(), today=TODAY)
    assert [x.title for x in out] == ["kept"]


def test_accepts_paths_and_strings(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(png())
    client = FakeClient(results(HOT_WHEELS), results(HOT_WHEELS))
    extractor = ClaudeExtractor(client=client)
    assert len(extractor.extract(path, today=TODAY)) == 1
    assert len(extractor.extract(str(path), today=TODAY)) == 1
    assert extractor.usage == {"requests": 2, "input_tokens": 2400, "output_tokens": 600}

    with pytest.raises(LLMExtractionError) as err:
        extractor.extract(tmp_path / "missing.png")
    assert err.value.reason == "bad_image"


# --- stop reasons and page issues --------------------------------------------------


def test_refusal_raises_with_category():
    details = SimpleNamespace(type="refusal", category="cyber", explanation="policy")
    client = FakeClient(response(stop_reason="refusal", stop_details=details))
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract(png())
    assert err.value.reason == "refusal"
    assert err.value.category == "cyber"
    assert "cyber" in str(err.value)


def test_refusal_without_details():
    client = FakeClient(response(stop_reason="refusal", stop_details=None))
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract(png())
    assert (err.value.reason, err.value.category) == ("refusal", None)


def test_max_tokens_retries_once_with_a_larger_budget():
    truncated = response(stop_reason="max_tokens", text='{"page_issue": null, "listings": [{"title": "Hot')
    client = FakeClient(truncated, results(HOT_WHEELS))
    out = ClaudeExtractor(client=client).extract(png(), today=TODAY)
    assert len(out) == 1
    assert [c["max_tokens"] for c in client.calls] == [4096, 8192]


def test_max_tokens_twice_raises_truncated():
    truncated = response(stop_reason="max_tokens", text='{"listings": [')
    client = FakeClient(truncated, truncated)
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract(png())
    assert err.value.reason == "truncated"
    assert len(client.calls) == 2


def test_captcha_page_raises_and_stops():
    client = FakeClient(results(page_issue="captcha"), results(HOT_WHEELS))
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract_many([png(), png()])
    assert err.value.reason == "captcha"
    assert len(client.calls) == 1  # did not go on to the next screenshot


def _three_slices() -> bytes:
    image = png(1280, 3200)
    assert len(chunk_spans(3200)) == 3
    return image


REFUSAL = response(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber", explanation=None))
TRUNCATED = response(stop_reason="max_tokens", text='{"listings": [')


@pytest.mark.parametrize("bad_replies,reason", [
    ([REFUSAL], "refusal"),
    ([response(text="{not json")], "bad_output"),
    ([TRUNCATED, TRUNCATED], "truncated"),  # still cut off after the larger retry
])
def test_a_slice_that_fails_costs_that_slice_not_the_page(caplog, bad_replies, reason):
    client = FakeClient(results(card("Card A", price_text="$1.00")), *bad_replies,
                        results(card("Card C", price_text="$3.00")))
    extractor = ClaudeExtractor(client=client)
    with caplog.at_level("WARNING", logger="ebay_sold.llm"):
        out = extractor.extract(_three_slices(), today=TODAY)
    assert [x.title for x in out] == ["Card A", "Card C"]  # slices 1 and 3 were paid for and kept
    assert [x.position for x in out] == [1, 2]
    assert [e.reason for e in extractor.failures] == [reason]
    assert "slice 2/3" in str(extractor.failures[0])
    assert "1 of 3 slice(s) could not be read" in caplog.text
    assert extractor.usage["requests"] == len(client.calls) == 2 + len(bad_replies)

    # failures are per call
    extractor.client = FakeClient(results(HOT_WHEELS))
    extractor.extract(png(), today=TODAY)
    assert extractor.failures == []


def test_when_every_slice_fails_the_error_is_raised():
    client = FakeClient(REFUSAL, response(text="nope"), REFUSAL)
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract(_three_slices())
    assert (err.value.reason, err.value.category) == ("refusal", "cyber")
    assert len(client.calls) == 3


def test_captcha_or_api_errors_on_a_later_slice_still_stop_the_page():
    client = FakeClient(results(card("Card A", price_text="$1.00")), results(page_issue="captcha"), results())
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract(_three_slices())
    assert err.value.reason == "captcha" and len(client.calls) == 2

    auth_error, _ = next((e, r) for e, r in _sdk_errors() if r == "auth")
    client = FakeClient(results(card("Card A", price_text="$1.00")), auth_error, results())
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract(_three_slices())
    assert err.value.reason == "auth" and len(client.calls) == 2


@pytest.mark.parametrize("issue", ["not_search_results", "unreadable"])
def test_other_page_issues_yield_no_listings(issue):
    client = FakeClient(results(HOT_WHEELS, page_issue=issue))
    assert ClaudeExtractor(client=client).extract(png()) == []


def test_invalid_json_is_reported():
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=FakeClient(response(text="not json"))).extract(png())
    assert err.value.reason == "bad_output"


def test_served_by_fallback_model_is_accepted(caplog):
    reply = response({"page_issue": None, "listings": [HOT_WHEELS]}, model="claude-opus-4-8")
    with caplog.at_level("INFO", logger="ebay_sold.llm"):
        out = ClaudeExtractor(client=FakeClient(reply)).extract(png(), today=TODAY)
    assert len(out) == 1
    assert "served by claude-opus-4-8" in caplog.text


# --- SDK errors ----------------------------------------------------------------------


def _sdk_errors():
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return [
        (anthropic.RateLimitError("slow down", response=httpx2.Response(
            429, request=req, headers={"retry-after": "30"}), body=None), "rate_limited"),
        (anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=req), body=None), "auth"),
        (anthropic.BadRequestError("bad request", response=httpx2.Response(400, request=req), body=None),
         "api_error"),
        (anthropic.APIConnectionError(request=req), "connection"),
        (anthropic.APITimeoutError(request=req), "connection"),
    ]


def test_sdk_errors_become_extraction_errors():
    for exc, reason in _sdk_errors():
        with pytest.raises(LLMExtractionError) as err:
            ClaudeExtractor(client=FakeClient(exc)).extract(png())
        assert err.value.reason == reason, exc
        assert err.value.__cause__ is exc
    exc, _ = _sdk_errors()[0]
    with pytest.raises(LLMExtractionError, match="retry after 30s"):
        ClaudeExtractor(client=FakeClient(exc)).extract(png())


# --- tall / wide screenshots ------------------------------------------------------------


def test_chunk_spans_cover_the_page_with_overlap():
    assert chunk_spans(900) == [(0, 900)]
    for height in (1501, 3200, 9999, 60000):
        spans = chunk_spans(height)
        assert spans[0][0] == 0 and spans[-1][1] == height
        assert all(b - t <= MAX_CHUNK_HEIGHT for t, b in spans)
        assert all(prev[1] - nxt[0] >= CHUNK_OVERLAP - 1 for prev, nxt in zip(spans, spans[1:]))
        assert len({b - t for t, b in spans}) == 1  # equal heights, no sliver at the end
    with pytest.raises(ValueError):
        chunk_spans(5000, chunk_height=500, overlap=500)


def test_tall_screenshot_is_sliced_and_overlap_duplicates_merged():
    image = png(1280, 3200)
    a = card("Card A", price_text="$1.00", sold_date_text="Sold  Apr 1, 2026")
    b_partial = card("Card B", price_text="$2.00")  # cut off at the bottom of slice 1
    b_full = card("Card B", price_text="$2.00", sold_date_text="Sold  Apr 2, 2026", condition="Pre-Owned",
                  shipping_text="Free delivery")
    c = card("Card C", price_text="$3.00", sold_date_text="Sold  Apr 3, 2026")
    c_twin = card("Card C", price_text="$3.00", sold_date_text="Sold  Apr 3, 2026")  # a second, identical sale
    d = card("Card D", price_text="$4.00", sold_date_text="Sold  Apr 4, 2026")
    client = FakeClient(results(a, b_partial), results(b_full, c, c_twin), results(c, d))

    out = ClaudeExtractor(client=client).extract(image, today=TODAY)

    assert len(client.calls) == 3
    for i, call in enumerate(client.calls, 1):
        media_type, data = sent_image(call)
        assert media_type == "image/png"
        width, height = size_of(data)
        assert width == 1280 and height <= MAX_CHUNK_HEIGHT
        assert f"slice {i} of 3" in call["messages"][0]["content"][1]["text"]
    # B is merged across slices 1-2 (fields filled from the fuller sighting); C appears in
    # slices 2 and 3 and is merged once, but the identical twin inside slice 2 is kept.
    assert [x.title for x in out] == ["Card A", "Card B", "Card C", "Card C", "Card D"]
    b = out[1]
    assert (b.price, b.sold_date, b.condition, b.shipping) == (2.0, date(2026, 4, 2), "Pre-Owned", 0.0)
    assert [x.position for x in out] == [1, 2, 3, 4, 5]


def test_different_prices_are_not_merged():
    first = card("Same title", price_text="$10.00", sold_date_text="Sold  Apr 1, 2026")
    second = card("Same title", price_text="$12.00", sold_date_text="Sold  Apr 1, 2026")
    assert len(merge_chunks([[first], [second]])) == 2
    # Non-adjacent slices never overlap, so they are not merged either.
    assert len(merge_chunks([[first], [], [dict(first)]])) == 2
    assert len(merge_chunks([[first], [dict(first)]])) == 1


# Hot-wheels fixture, slices 10 and 11: the top edge of slice 11 cuts through
# this card's title, so slice 11 shows only its last line "R33" above the price.
R33_FULL = card("HOT WHEELS NISSAN SKYLINE GT-R (BCNR33) ZAMAC WALMART EXCLUSIVE 2019 R33",
                price_text="$17.75", sold_date_text="Sold  Apr 19, 2026", condition="Brand New",
                format_text="or Best Offer", shipping_text="+$5.83 delivery in 2-4 days",
                seller_text="hof-575 98.6% positive (142)")
R33_CUT_AT_TOP = card("R33", price_text="$17.75", condition="Brand New", format_text="or Best Offer",
                      shipping_text="+$5.83 delivery in 2-4 days", seller_text="hof-575 98.6% positive (142)",
                      location_text="Located in United States")
NEXT_CARD = card("Hot Wheels Nissan Skyline GT-R R33 Nismo Zamac - Tokyo Auto Salon Japan Exclusive",
                 price_text="$24.99", sold_date_text="Sold  Apr 19, 2026")


def test_card_whose_title_is_cut_at_the_top_of_a_slice_is_merged():
    merged = merge_chunks([[card("earlier card", price_text="$9.00"), R33_FULL], [R33_CUT_AT_TOP, NEXT_CARD]])
    assert [m["title"] for m in merged] == ["earlier card", R33_FULL["title"], NEXT_CARD["title"]]
    assert merged[1]["location_text"] == "Located in United States"  # filled from the cut sighting
    assert merged[1]["sold_date_text"] == "Sold  Apr 19, 2026"


def test_card_whose_title_is_cut_at_the_bottom_of_a_slice_is_merged():
    # Bottom of slice k: the sold date and the first title line, price below the edge.
    cut = card("Hot Wheels Nissan Skyline GT-R R33 Nismo Zamac -", sold_date_text="Sold  Apr 19, 2026")
    out = ClaudeExtractor(client=FakeClient(results(R33_FULL, cut), results(NEXT_CARD))).extract_many(
        [png(), png()], today=TODAY)
    assert [(x.title, x.price) for x in out] == [(R33_FULL["title"], 17.75), (NEXT_CARD["title"], 24.99)]


def test_partial_titles_need_agreeing_evidence():
    # Different price: a different sale whose title happens to end the same way.
    other = card("R33", price_text="$5.00", condition="Brand New")
    assert len(merge_chunks([[R33_FULL], [other]])) == 2
    # Nothing to corroborate the partial title (no price, no date).
    assert len(merge_chunks([[R33_FULL], [card("R33", condition="Brand New")]])) == 2
    # A word fragment is not a cut line ("33" is not a word of the title).
    assert len(merge_chunks([[R33_FULL], [card("33", price_text="$17.75")]])) == 2
    # A different seller on otherwise matching evidence.
    assert len(merge_chunks([[R33_FULL], [dict(R33_CUT_AT_TOP, seller_text="someone 99% positive (5)")]])) == 2


def test_identical_twin_cut_at_a_slice_edge_gets_its_own_price():
    # C and its identical twin C2 stacked at the bottom of slice k (C2 cut below its date); slice k+1
    # shows only C2 whole. The whole sighting belongs to the twin at the edge, not to C.
    c = card("Hot Wheels R34 Zamac", price_text="$3.00", sold_date_text="Sold  Apr 3, 2026",
             shipping_text="Free delivery")
    twin_cut = card("Hot Wheels R34 Zamac", sold_date_text="Sold  Apr 3, 2026")
    twin_whole = dict(c)
    out = ClaudeExtractor(client=FakeClient(results(c, twin_cut), results(twin_whole))).extract_many(
        [png(), png()], today=TODAY)
    assert [(x.title, x.price, x.shipping) for x in out] == [("Hot Wheels R34 Zamac", 3.0, 0.0)] * 2


def test_overlap_alignment_keeps_reading_order():
    a, b, c, d = (card(f"Card {x}", price_text=f"${i}.00") for i, x in enumerate("ABCD", 1))
    # B and C are in the overlap; the model skipped B in slice 2 but saw C.
    merged = merge_chunks([[a, b, c], [dict(c), d]])
    assert [m["title"] for m in merged] == ["Card A", "Card B", "Card C", "Card D"]


def test_wide_screenshot_is_downscaled():
    chunks = split_screenshot(png(2400, 1000))
    assert len(chunks) == 1
    assert (chunks[0].width, chunks[0].height) == (MAX_WIDTH, 653)
    assert size_of(chunks[0].data) == (MAX_WIDTH, 653)


def _text_image(width: int, height: int) -> Any:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 20):
        draw.text((10, y), "Sold  Apr 25, 2026   $65.00  +$9.45 delivery  mystuffnnowyours 100% positive (267)",
                  fill="black")
    return img


def _encoded(img: Any, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_palette_screenshot_is_downscaled_smoothly():
    pytest.importorskip("PIL")
    from PIL import Image

    rgb = _text_image(2560, 600)
    palette = rgb.quantize(16)  # what pngquant / TinyPNG produce
    sent = []
    for img in (rgb, palette):
        [chunk] = split_screenshot(_encoded(img, "PNG"))
        sent.append(Image.open(io.BytesIO(chunk.data)))
    assert sent[1].mode == "RGB" and sent[1].size == (MAX_WIDTH, 368)
    # Nearest-neighbour would keep the 16 palette colours; a proper resample anti-aliases the glyphs.
    assert len(set(sent[1].convert("L").tobytes())) > 64
    assert len(set(sent[0].convert("L").tobytes())) > 64


@pytest.mark.parametrize("size", [(2560, 1000), (800, 600)])
def test_cmyk_jpeg_is_converted_not_crashed_on(size):
    pytest.importorskip("PIL")
    from PIL import Image

    data = _encoded(_text_image(*size).convert("CMYK"), "JPEG")
    [chunk] = split_screenshot(data)
    sent = Image.open(io.BytesIO(chunk.data))
    assert sent.mode == "RGB" and chunk.media_type == "image/png"


def test_huge_screenshot_gives_a_clear_error(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image = png(400, 300)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # stands in for a 300 MP full-page shot
    with pytest.raises(LLMExtractionError, match="capture_tiles") as err:
        split_screenshot(image)
    assert err.value.reason == "bad_image"


def test_extract_many_merges_across_capture_tiles():
    client = FakeClient(results(card("Card A", price_text="$1.00"), card("Card B", price_text="$2.00")),
                        results(card("Card B", price_text="$2.00"), card("Card C", price_text="$3.00")))
    out = ClaudeExtractor(client=client).extract_many([png(), png()], today=TODAY)
    assert [x.title for x in out] == ["Card A", "Card B", "Card C"]


def _without_pillow(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)


def test_without_pillow_images_the_api_reads_whole_are_sent_whole(monkeypatch):
    tile = png(1280, 2000)  # a capture_tiles tile: read at full resolution, no slicing needed
    _without_pillow(monkeypatch)
    chunks = split_screenshot(tile)
    assert len(chunks) == 1 and chunks[0].data == tile and (chunks[0].width, chunks[0].height) == (1280, 2000)
    with pytest.raises(LLMExtractionError) as err:
        split_screenshot(b"definitely not an image")
    assert err.value.reason == "bad_image"


@pytest.mark.parametrize("size", [(1280, 3000), (1280, 20025), (2600, 900)])
def test_without_pillow_a_screenshot_needing_slices_is_refused_not_sent(monkeypatch, size):
    # Sent whole, a full page would be downscaled until unreadable (or rejected above 8000 px).
    image = png(*size)
    _without_pillow(monkeypatch)
    client = FakeClient(results(HOT_WHEELS))
    with pytest.raises(LLMExtractionError, match="pip install pillow") as err:
        ClaudeExtractor(client=client).extract(image)
    assert err.value.reason == "not_installed"
    assert client.calls == []  # nothing was paid for


@pytest.mark.parametrize("fmt,mode,kw", [("PNG", "RGB", {}), ("JPEG", "RGB", {}), ("JPEG", "CMYK", {}),
                                         ("JPEG", "RGB", {"progressive": True}), ("GIF", "P", {}),
                                         ("WEBP", "RGB", {}), ("WEBP", "RGB", {"lossless": True}),
                                         ("WEBP", "RGBA", {})])
def test_image_size_reads_headers_without_pillow(monkeypatch, fmt, mode, kw):
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new(mode, (1234, 4321)).save(buf, format=fmt, **kw)
    _without_pillow(monkeypatch)
    from ebay_sold.llm import image_size

    assert image_size(buf.getvalue()) == (1234, 4321)
    assert ClaudeExtractor(client=FakeClient()).estimate_cost([buf.getvalue()]).requests == 4


# --- cost ------------------------------------------------------------------------------


def test_estimate_cost():
    est = estimate_cost([(1280, 3200)])
    assert est.requests == 3
    per_slice = image_tokens(1280, 1334)
    assert per_slice == 2277
    assert est.input_tokens == 3 * (per_slice + 900)
    assert est.usd == pytest.approx((est.input_tokens * 4 + est.output_tokens * 20) / 1_000_000, abs=1e-4)
    assert estimate_cost([(800, 600)], model="some-unknown-model").usd is None

    extractor = ClaudeExtractor(client=FakeClient())
    assert extractor.estimate_cost([png(1280, 3200)]) == est
    assert extractor.estimate_cost([png(2400, 1000)]).requests == 1


# --- optional dependency -------------------------------------------------------------------


def test_missing_anthropic_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    with pytest.raises(LLMExtractionError, match=r"pip install 'ebay-sold\[llm\]'") as err:
        ClaudeExtractor()
    assert err.value.reason == "not_installed"
    # An injected client still works without the SDK installed.
    out = ClaudeExtractor(client=FakeClient(results(HOT_WHEELS))).extract(png(), today=TODAY)
    assert len(out) == 1


_CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
                   "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_IDENTITY_TOKEN_FILE",
                   "ANTHROPIC_IDENTITY_TOKEN")


def test_default_client_is_built_from_the_sdk(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    extractor = ClaudeExtractor()
    assert isinstance(extractor.client, anthropic.Anthropic)


def test_missing_credentials_fail_early_with_a_hint(monkeypatch, tmp_path):
    pytest.importorskip("anthropic")
    for name in _CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ANTHROPIC_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile on disk
    with pytest.raises(LLMExtractionError, match="ANTHROPIC_API_KEY") as err:
        ClaudeExtractor()
    assert err.value.reason == "auth"

    # A config dir without the profile makes the SDK itself raise while constructing.
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "anthropic"))
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor()
    assert err.value.reason == "auth"


# --- the real SDK, offline ------------------------------------------------------------------


def _sdk_client(reply: dict[str, Any], seen: list[Any]):
    """A real ``anthropic.Anthropic`` whose HTTP transport is a local stub (no network)."""
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    def handler(request: "httpx2.Request") -> "httpx2.Response":
        seen.append(request)
        return httpx2.Response(200, json=reply)

    return anthropic.Anthropic(
        api_key="test-key-not-real",
        base_url="https://api.anthropic.com",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def _message(text: str | None, *, stop_reason: str = "end_turn", stop_details: Any = None) -> dict[str, Any]:
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
            "usage": {"input_tokens": 2500, "output_tokens": 400}}


def test_real_sdk_serializes_the_request_and_parses_the_reply():
    seen: list[Any] = []
    client = _sdk_client(_message(json.dumps({"page_issue": None, "listings": [HOT_WHEELS]})), seen)
    [item] = ClaudeExtractor(client=client).extract(png(), today=TODAY)
    assert (item.price, item.sold_date, item.seller) == (65.0, date(2026, 4, 25), "mystuffnnowyours")

    [request] = seen
    assert request.url.path == "/v1/messages"
    assert request.headers["anthropic-beta"] == FALLBACK_BETA
    body = json.loads(request.content)
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["schema"] == RESPONSE_SCHEMA
    assert body["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert "thinking" not in body and "betas" not in body


def test_real_sdk_refusal_carries_category():
    seen: list[Any] = []
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    client = _sdk_client(_message(None, stop_reason="refusal", stop_details=details), seen)
    with pytest.raises(LLMExtractionError) as err:
        ClaudeExtractor(client=client).extract(png())
    assert (err.value.reason, err.value.category) == ("refusal", "cyber")


def test_max_tokens_above_the_non_streaming_limit_is_capped():
    # The SDK refuses a non-streaming request above ~21K max_tokens with a bare ValueError.
    seen: list[Any] = []
    client = _sdk_client(_message(json.dumps({"page_issue": None, "listings": [HOT_WHEELS]})), seen)
    out = ClaudeExtractor(LLMSettings(max_tokens=32000), client=client).extract(png(), today=TODAY)
    assert len(out) == 1
    assert json.loads(seen[0].content)["max_tokens"] == 16000


def test_client_side_value_errors_become_extraction_errors():
    pytest.importorskip("anthropic")
    client = FakeClient(ValueError("Streaming is required for operations that may take longer than 10 minutes."))
    with pytest.raises(LLMExtractionError, match="Streaming is required") as err:
        ClaudeExtractor(client=client).extract(png())
    assert err.value.reason == "client_error"


# --- capture_tiles output --------------------------------------------------------------------


def _tiles(tmp_path, page_png: bytes, tile_height: int, overlap: int, scale: float = 1.0):
    from PIL import Image

    from ebay_sold.models import Box

    page = Image.open(io.BytesIO(page_png))
    tiles, y, i = [], 0, 0
    css_h = page.height / scale
    while True:
        h = min(tile_height, css_h - y)
        path = tmp_path / f"tile{i}.png"
        page.crop((0, round(y * scale), page.width, round((y + h) * scale))).save(path)
        tiles.append((path, Box(x=0, y=y, w=page.width / scale, h=h)))
        if y + h >= css_h:
            return page, tiles
        y += tile_height - overlap
        i += 1


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_stitch_tiles_rebuilds_the_page(tmp_path, scale):
    from ebay_sold.llm import stitch_tiles

    page, tiles = _tiles(tmp_path, png(int(640 * scale), int(4500 * scale)), tile_height=2000, overlap=200,
                         scale=scale)
    assert len(tiles) == 3
    stitched = stitch_tiles(tiles)
    assert stitched.size == page.size
    assert stitched.tobytes() == page.convert("RGB").tobytes()


def test_extract_tiles_reslices_the_stitched_page(tmp_path):
    _, tiles = _tiles(tmp_path, png(1280, 4500), tile_height=2000, overlap=200)
    n = len(chunk_spans(4500))
    client = FakeClient(*[results(card(f"Card {i}", price_text=f"${i}.00")) for i in range(n)])
    out = ClaudeExtractor(client=client).extract_tiles(tiles, today=TODAY)
    assert len(client.calls) == n == 4  # vs 6 slices when each 2000 px tile is sliced on its own
    assert [x.title for x in out] == [f"Card {i}" for i in range(n)]
    for call in client.calls:
        assert size_of(sent_image(call)[1])[1] <= MAX_CHUNK_HEIGHT
