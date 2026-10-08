from datetime import date

import pytest

from ebay_sold.normalize import (
    Money,
    classify_format,
    clean_text,
    parse_count,
    parse_feedback,
    parse_location,
    parse_money,
    parse_shipping,
    parse_sold_date,
)

TODAY = date(2026, 5, 1)


@pytest.mark.parametrize(
    "text, default, expected",
    [
        ("$65.00", "USD", Money(65.0, None, "USD")),
        ("$1,234.56", "USD", Money(1234.56, None, "USD")),
        ("US $12.00", None, Money(12.0, None, "USD")),
        ("£9.99", None, Money(9.99, None, "GBP")),
        ("EUR 12,50", None, Money(12.5, None, "EUR")),
        ("1.234,56 €", None, Money(1234.56, None, "EUR")),
        ("C $15.00", None, Money(15.0, None, "CAD")),
        ("AU $7.25", None, Money(7.25, None, "AUD")),
        ("$3.75 to $23.95", "USD", Money(3.75, 23.95, "USD")),
        ("3,75 EUR bis 23,95 EUR", None, Money(3.75, 23.95, "EUR")),
        ("$1,200", "USD", Money(1200.0, None, "USD")),
        # OCR noise
        ("S65.00", "USD", Money(65.0, None, "USD")),
        ("$ 65.00", "USD", Money(65.0, None, "USD")),
        ("$6O.00", "USD", Money(60.0, None, "USD")),
    ],
)
def test_parse_money(text, default, expected):
    assert parse_money(text, default) == expected


@pytest.mark.parametrize("text", ["See price", "", None, "Buy It Now"])
def test_parse_money_none(text):
    assert parse_money(text, "USD") is None


def test_parse_money_not_a_range_when_digits_follow_later():
    # "2-4 days" must not turn a shipping price into a range.
    assert parse_money("+$1.50 delivery in 2-4 days", "USD") == Money(1.5, None, "USD")
    # strikethrough original next to the sale price is not a range either
    assert parse_money("$13.58 $39.95", "USD") == Money(13.58, None, "USD")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Free delivery", 0.0),
        ("Free shipping", 0.0),
        ("Free postage", 0.0),
        ("Kostenloser Versand", 0.0),
        ("+$9.45 delivery", 9.45),
        ("+$5.83 delivery in 2-4 days", 5.83),
        ("+£3.20 postage", 3.20),
        ("+$9.45delivery", 9.45),
        ("Shipping not specified", None),
        ("Freight", None),
        ("", None),
    ],
)
def test_parse_shipping(text, expected):
    assert parse_shipping(text, "USD") == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Sold  Apr 25, 2026", date(2026, 4, 25)),
        ("Sold Apr 25, 2026", date(2026, 4, 25)),
        ("SoldApr25,2026", date(2026, 4, 25)),  # OCR dropped the spaces
        ("Sold Jul 8, 2025", date(2025, 7, 8)),
        ("Sold 25 Apr 2026", date(2026, 4, 25)),  # UK order
        ("Verkauft 25. Apr. 2026", date(2026, 4, 25)),
        ("Sold Dec 30", date(2025, 12, 30)),  # no year: most recent past date
        ("Sold Apr 30", date(2026, 4, 30)),
        ("Ended Jan 3, 2026", date(2026, 1, 3)),
        ("Sold September 9, 2025", date(2025, 9, 9)),
        ("garbage", None),
        ("", None),
    ],
)
def test_parse_sold_date(text, expected):
    assert parse_sold_date(text, TODAY) == expected


def test_parse_sold_date_numeric_respects_day_first():
    assert parse_sold_date("Sold 05/04/2026", TODAY) == date(2026, 5, 4)
    assert parse_sold_date("Sold 05/04/2026", TODAY, day_first=True) == date(2026, 4, 5)


@pytest.mark.parametrize(
    "text, expected",
    [("267", 267), ("2.1K", 2100), ("12K", 12000), ("1,234", 1234), ("3 bids", 3), ("1.2M", 1_200_000), ("", None)],
)
def test_parse_count(text, expected):
    assert parse_count(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mystuffnnowyours 100% positive (267)", ("mystuffnnowyours", 100.0, 267)),
        ("goldstar_tech 99.8% positive (12K) Top Rated", ("goldstar_tech", 99.8, 12000)),
        ("hildrzac 100% positive (2.1K)", ("hildrzac", 100.0, 2100)),
    ],
)
def test_parse_feedback(text, expected):
    assert parse_feedback(text) == expected


@pytest.mark.parametrize(
    "rows, expected",
    [
        (["$65.00", "Buy It Now", "+$9.45 delivery"], ("buy_it_now", None)),
        (["$41.99", "or Best Offer"], ("best_offer", None)),
        (["$12.50", "11 bids", "+$5.00 delivery"], ("auction", 11)),
        (["$3.00", "1 bid"], ("auction", 1)),
        (["$3.00"], ("unknown", None)),
    ],
)
def test_classify_format(rows, expected):
    assert classify_format(rows) == expected


def test_clean_text_strips_invisible_separators():
    assert clean_text("S⁣p⁣o⁣n  sored\n") == "Spon sored"


def test_parse_location():
    assert parse_location("Located in United States") == "United States"
    assert parse_location("from Canada") == "Canada"
    assert parse_location("") is None
