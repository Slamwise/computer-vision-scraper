"""A block stops later runs too: the cooldown record and Retry-After (no browser)."""

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from ebay_sold.config import PacingSettings, Settings
from ebay_sold.fetch import BlockedError, BlockInfo, EbayFetcher
from ebay_sold.fetch.browser import MAX_COOLDOWN_S, _parse_retry_after

NOW = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
CAPTCHA = BlockInfo(kind="captcha", reason="challenge URL /splashui/challenge",
                    url="https://www.ebay.com/splashui/challenge?ru=x", status=200)
LIMITED = BlockInfo(kind="rate_limited", reason="HTTP 429 Too Many Requests",
                    url="https://www.ebay.com/sch/i.html?_nkw=x", status=429)


def fetcher(tmp_path, **pacing) -> EbayFetcher:
    return EbayFetcher(Settings(data_dir=tmp_path, pacing=PacingSettings(**pacing)), interactive=False)


def test_parse_retry_after():
    assert _parse_retry_after("120", NOW) == 120
    assert _parse_retry_after(" 3600 ", NOW) == 3600
    assert _parse_retry_after(format_datetime(NOW + timedelta(hours=2), usegmt=True), NOW) == 7200
    assert _parse_retry_after(format_datetime(NOW - timedelta(hours=2), usegmt=True), NOW) == 0
    for junk in (None, "", "soon", "-5", "1.5"):
        assert _parse_retry_after(junk, NOW) is None, junk


def test_a_stopped_run_makes_the_next_run_wait(tmp_path):
    first = fetcher(tmp_path, challenge_cooldown_s=900)
    with pytest.raises(BlockedError) as exc:
        first._stop(CAPTCHA, "not running interactively")
    assert exc.value.retry_after_s == 900
    record = json.loads(first.cooldown_path.read_text())
    assert record["block"]["kind"] == "captcha"

    second = fetcher(tmp_path, challenge_cooldown_s=900)
    assert 890 < second.cooldown_remaining() <= 900
    with pytest.raises(BlockedError) as exc:
        second._refuse_if_stopped()
    assert exc.value.info.kind == "captcha" and "cooling down until" in exc.value.info.reason
    assert "no request was sent" in str(exc.value) and str(second.cooldown_path) in str(exc.value)

    second.clear_cooldown()
    second._refuse_if_stopped()  # no longer raises
    second.clear_cooldown()  # idempotent


def test_retry_after_lengthens_but_never_shortens_the_cooldown(tmp_path):
    f = fetcher(tmp_path, challenge_cooldown_s=900)
    for header, expected in (("3600", 3600), ("60", 900), (str(10**9), MAX_COOLDOWN_S), (None, 900)):
        f._doc_retry_after = header
        with pytest.raises(BlockedError) as exc:
            f._stop(LIMITED, "rate limited")
        assert exc.value.retry_after_s == expected, header
    # Retry-After on a page that is not a 429 is not ours to honour
    f._doc_retry_after = "3600"
    with pytest.raises(BlockedError) as exc:
        f._stop(CAPTCHA, "x")
    assert exc.value.retry_after_s == 900


def test_expired_zero_or_unreadable_cooldown_is_ignored(tmp_path):
    f = fetcher(tmp_path, challenge_cooldown_s=0)
    with pytest.raises(BlockedError):
        f._stop(CAPTCHA, "x")
    assert not f.cooldown_path.exists()  # no cooldown configured: nothing recorded

    f.cooldown_path.parent.mkdir(parents=True, exist_ok=True)
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    f.cooldown_path.write_text(json.dumps({"since": (past - timedelta(hours=1)).isoformat(),
                                           "until": past.isoformat(), "block": CAPTCHA.model_dump()}))
    assert fetcher(tmp_path).cooldown_remaining() == 0
    fetcher(tmp_path)._refuse_if_stopped()

    for junk in ("{not json", "[]", "null", json.dumps({"since": "x", "until": "y", "block": {}})):
        f.cooldown_path.write_text(junk)
        fetcher(tmp_path)._refuse_if_stopped()  # a broken record never wedges the tool

    naive = (datetime.now(timezone.utc) + timedelta(minutes=5)).replace(tzinfo=None)
    f.cooldown_path.write_text(json.dumps({"since": naive.isoformat(), "until": naive.isoformat(),
                                           "block": CAPTCHA.model_dump()}))
    assert 0 < fetcher(tmp_path).cooldown_remaining() <= 300  # naive times are read as UTC
