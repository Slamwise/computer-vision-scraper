"""SQLite storage: dedup, extraction precedence, observations, filters, stats, export."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timezone

import pytest

from ebay_sold import db as dbmod
from ebay_sold.db import (
    MIGRATIONS,
    SCHEMA_VERSION,
    Database,
    PriceStats,
    UpsertStats,
    is_synthetic_id,
    normalize_title,
    synthetic_item_id,
)
from ebay_sold.models import Listing, PageResult, SearchQuery

T0 = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)


def L(item_id: str | None = "100000000001", **kw) -> Listing:
    data = {"title": "Hot Wheels Nissan Skyline GT-R R34 Zamac", "price": 10.0, "currency": "USD",
            "sold_date": date(2026, 4, 1)}
    data.update(kw)
    return Listing(item_id=item_id, **data)


def page(listings: list[Listing], n: int = 1, when: datetime = T0, **kw) -> PageResult:
    return PageResult(url=f"https://www.ebay.com/sch/i.html?_pgn={n}", page=n, fetched_at=when,
                      listings=listings, **kw)


@pytest.fixture
def db():
    with Database(":memory:") as d:
        yield d


def count(d: Database, table: str) -> int:
    return d.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def sources(d: Database, item_id: str, site: str = "www.ebay.com") -> dict[str, str]:
    row = d.connection.execute("SELECT field_sources FROM listings WHERE site = ? AND item_id = ?",
                               (site, item_id)).fetchone()
    return json.loads(row[0])


# --- schema, lifecycle, migrations ------------------------------------------------------


def test_file_db_created_with_wal_and_foreign_keys(tmp_path):
    path = tmp_path / "nested" / "dir" / "sold.sqlite"
    with Database(path) as d:
        assert path.exists()
        assert d.schema_version == SCHEMA_VERSION == len(MIGRATIONS)
        assert d.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert d.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        d.upsert_listings([L()])


def test_reopen_existing_file_is_idempotent(tmp_path):
    path = tmp_path / "sold.sqlite"
    with Database(path) as d:
        sid = d.record_search(SearchQuery(keywords="r34"), "u")
        d.save_page(sid, page([L()]))
        tables = d.connection.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall()
    for _ in range(2):
        with Database(str(path)) as d:
            assert d.schema_version == SCHEMA_VERSION
            assert d.connection.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall() == tables
            assert d.get_listing("100000000001").price == 10.0
            assert len(d.searches()) == 1


def test_memory_db_and_close():
    d = Database(":memory:")
    d.upsert_listings([L()])
    assert d.get_listing("100000000001") is not None
    d.close()
    d.close()  # closing twice is harmless
    with pytest.raises((sqlite3.ProgrammingError, AttributeError)):
        d.connection.execute("SELECT 1")


@pytest.fixture
def connections(monkeypatch):
    """Every sqlite3 connection opened during the test."""
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        opened.append(con)
        return con

    monkeypatch.setattr(dbmod.sqlite3, "connect", connect)
    return opened


def assert_closed(con: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        con.execute("SELECT 1")


def test_refuses_database_from_newer_version(tmp_path, connections):
    path = tmp_path / "future.sqlite"
    con = sqlite3.connect(path)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    con.close()
    with pytest.raises(RuntimeError, match="newer"):
        Database(path)
    # The refused file is left as it was (not switched to WAL) and the connection is closed.
    assert len(connections) == 2
    assert_closed(connections[1])
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    con.close()


def test_upgrades_a_version_1_file(tmp_path, monkeypatch):
    path = tmp_path / "v1.sqlite"
    with monkeypatch.context() as m:
        m.setattr(dbmod, "MIGRATIONS", MIGRATIONS[:1])
        m.setattr(dbmod, "SCHEMA_VERSION", 1)
        with Database(path) as d:
            d.upsert_listings([L()])
            assert d.schema_version == 1
    with Database(path) as d:
        assert d.schema_version == SCHEMA_VERSION
        indexes = {r[0] for r in d.connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert {"listings_match_price", "listings_match_date"} <= indexes and "listings_match_key" not in indexes
        d.upsert_listings([L(None, sold_date=None, extraction="vision")])
        assert count(d, "listings") == 1 and d.get_listing("100000000001").price == 10.0


def test_sql_statement_splitter():
    script = "CREATE TABLE a (x TEXT DEFAULT 'a;b'); -- c;\nCREATE TRIGGER t AFTER INSERT ON a BEGIN SELECT 1; END;\n"
    assert dbmod._sql_statements(script) == [
        "CREATE TABLE a (x TEXT DEFAULT 'a;b');",
        "-- c;\nCREATE TRIGGER t AFTER INSERT ON a BEGIN SELECT 1; END;",
    ]
    assert dbmod._sql_statements("SELECT 1; SELECT 2") == ["SELECT 1;", "SELECT 2"]


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_concurrent_first_open_migrates_once(tmp_path, journal_mode):
    """Another process creates the schema while this one is opening the same new file."""
    path = tmp_path / "race.sqlite"
    other = sqlite3.connect(path, isolation_level=None)
    other.execute(f"PRAGMA journal_mode = {journal_mode}")
    other.execute("BEGIN IMMEDIATE")  # the other process has started migrating
    result: dict[str, object] = {}

    def open_db() -> None:
        try:
            with Database(path) as d:
                result["version"] = d.schema_version
                d.upsert_listings([L()])
        except Exception as e:  # pragma: no cover - reported by the assert below
            result["error"] = e

    t = threading.Thread(target=open_db)
    t.start()
    time.sleep(0.5)  # it has seen schema version 0 and now waits for the write lock
    for script in MIGRATIONS:
        for statement in dbmod._sql_statements(script):
            other.execute(statement)
    other.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    other.execute("COMMIT")
    other.close()
    t.join(30)
    assert result == {"version": SCHEMA_VERSION}


_OPENER = """
import sys, time
from ebay_sold.db import Database
from ebay_sold.models import Listing
start, me, paths = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
for n, path in enumerate(paths):
    while time.time() < start + 0.4 * n:
        pass
    with Database(path) as d:
        d.upsert_listings([Listing(item_id=str(100000000000 + 1000 * me + i), title=f"t{i}", price=1.0)
                           for i in range(50)])
"""


def test_processes_racing_to_create_a_file(tmp_path):
    paths = [str(tmp_path / f"race{n}.sqlite") for n in range(4)]
    start = time.time() + 3  # after every interpreter has started
    procs = [subprocess.Popen([sys.executable, "-c", _OPENER, str(start), str(me), *paths],
                              stderr=subprocess.PIPE, text=True) for me in range(4)]
    errors = [p.communicate(timeout=60)[1] for p in procs]
    assert [p.returncode for p in procs] == [0] * 4, "\n".join(errors)
    for path in paths:
        with Database(path) as d:
            assert count(d, "listings") == 200


def test_new_migration_upgrades_existing_file_once(tmp_path, monkeypatch):
    path = tmp_path / "sold.sqlite"
    with Database(path) as d:
        d.upsert_listings([L()])
    v2 = "ALTER TABLE listings ADD COLUMN note TEXT;"
    monkeypatch.setattr(dbmod, "MIGRATIONS", (*MIGRATIONS, v2))
    monkeypatch.setattr(dbmod, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    for _ in range(2):  # a second open must not re-run ALTER TABLE (it would fail)
        with Database(path) as d:
            assert d.schema_version == SCHEMA_VERSION + 1
            cols = [r[1] for r in d.connection.execute("PRAGMA table_info(listings)")]
            assert "note" in cols
            assert d.get_listing("100000000001").title.startswith("Hot Wheels")


def test_failed_migration_rolls_back(tmp_path, monkeypatch, connections):
    path = tmp_path / "sold.sqlite"
    Database(path).close()
    monkeypatch.setattr(dbmod, "MIGRATIONS", (*MIGRATIONS, "CREATE TABLE extra (x); SELECT * FROM nope;"))
    monkeypatch.setattr(dbmod, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    with pytest.raises(sqlite3.OperationalError, match="nope"):
        Database(path)
    assert_closed(connections[-1])
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert con.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'extra'").fetchone()[0] == 0
    con.close()


def test_every_listing_field_is_stored_somewhere():
    stored = set(dbmod.MERGED_FIELDS + dbmod.IDENTITY_FIELDS + dbmod.OBSERVATION_FIELDS + dbmod.PROVENANCE_FIELDS)
    assert stored == set(Listing.model_fields), "Listing changed: add a migration and update ebay_sold.db field lists"
    assert dbmod.MERGED_FIELDS.index("price") < dbmod.MERGED_FIELDS.index("price_max")


# --- dedup and observations ------------------------------------------------------------


def test_dedup_across_pages_and_searches(db):
    a, b, c = L("1001", title="A item"), L("1002", title="B item"), L("1003", title="C item")
    s1 = db.record_search(SearchQuery(keywords="r34"), "u1")
    assert db.save_page(s1, page([a, b], 1)) == UpsertStats(new=2)
    assert db.save_page(s1, page([b, c], 2)) == UpsertStats(new=1, unchanged=1)
    s2 = db.record_search(SearchQuery(keywords="r34 zamac"), "u2")
    stats = db.save_page(s2, page([a, b, c], 1, when=T1))
    assert stats == UpsertStats(unchanged=3) and stats.total == 3
    assert count(db, "listings") == 3
    assert count(db, "observations") == 7  # 2 + 2 on search 1, 3 on search 2
    # Saving the same page of the same search again does not add observations.
    db.save_page(s2, page([a, b, c], 1, when=T1))
    assert count(db, "observations") == 7
    seen = db.connection.execute("SELECT first_seen_at, last_seen_at FROM listings WHERE item_id = '1001'").fetchone()
    assert seen[0] == "2026-04-01T12:00:00+00:00" and seen[1] == "2026-04-02T12:00:00+00:00"


def test_first_and_last_seen_follow_observation_time_not_call_order(db):
    db.upsert_listings([L()], observed_at=T1)
    db.upsert_listings([L()], observed_at=T0)  # an older cached page imported later
    row = db.connection.execute("SELECT first_seen_at, last_seen_at FROM listings").fetchone()
    assert (row[0], row[1]) == (T0.isoformat(), T1.isoformat())


def test_upsert_stats_count_real_changes(db):
    assert db.upsert_listings([L(), L("1002")]) == UpsertStats(new=2)
    assert db.upsert_listings([L()]) == UpsertStats(unchanged=1)
    assert db.upsert_listings([L(price=11.0)]) == UpsertStats(updated=1)
    assert db.upsert_listings([L(price=None)]) == UpsertStats(unchanged=1)  # None never erases
    assert (UpsertStats(new=1) + UpsertStats(updated=2)) == UpsertStats(new=1, updated=2)


def test_item_id_taken_from_url_and_site_normalized(db):
    db.upsert_listings([L(None, url="https://www.ebay.co.uk/itm/Some-Title/123456789012?hash=x", site="ebay.co.uk",
                          currency="GBP")])
    got = db.get_listing("123456789012", site="www.ebay.co.uk")
    assert got is not None and got.site == "www.ebay.co.uk" and got.currency == "GBP"
    assert db.get_listing("123456789012", site="ebay.co.uk") == got
    assert db.get_listing("123456789012") is None  # same id on another site is another row


def test_same_item_on_two_sites_is_two_rows(db):
    db.upsert_listings([L(site="www.ebay.com"), L(site="www.ebay.co.uk", currency="GBP", price=8.0)])
    assert count(db, "listings") == 2


def test_failed_page_is_rolled_back(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.save_page(999, page([L("1001"), L("1002")]))  # no such search
    assert count(db, "listings") == 0
    assert count(db, "search_pages") == 0


def test_disk_full_error_is_reported_not_masked(tmp_path):
    # SQLite rolls back by itself on SQLITE_FULL; the error must not turn into
    # "cannot rollback - no transaction is active".
    with Database(tmp_path / "full.sqlite") as d:
        d.connection.execute("PRAGMA max_page_count = 40")
        big = [L(str(100000000000 + i), title="x" * 300 + str(i)) for i in range(2000)]
        with pytest.raises(sqlite3.OperationalError, match="full"):
            d.upsert_listings(big)
        assert not d.connection.in_transaction and count(d, "listings") == 0
        d.upsert_listings([L()])  # the connection is still usable
        assert count(d, "listings") == 1


# --- extraction precedence -------------------------------------------------------------------


def test_vision_cannot_clobber_dom_but_fills_gaps(db):
    db.upsert_listings([L(price=10.0, condition="Pre-Owned", title="Exact DOM title")])
    stats = db.upsert_listings([L(price=11.0, condition="Used", title="Exact D0M title", seller="ocr_seller",
                                  shipping=4.5, extraction="vision", confidence=0.7)])
    assert stats == UpsertStats(updated=1)
    got = db.get_listing("100000000001")
    assert (got.price, got.condition, got.title) == (10.0, "Pre-Owned", "Exact DOM title")
    assert (got.seller, got.shipping) == ("ocr_seller", 4.5)  # gaps filled
    assert got.extraction == "dom" and got.confidence is None
    src = sources(db, "100000000001")
    assert src["price"] == "dom" and src["seller"] == "vision" and src["shipping"] == "vision"


def test_dom_overwrites_vision_and_takes_over_provenance(db):
    db.upsert_listings([L(price=11.0, shipping=None, seller="ocr_seller", extraction="vision", confidence=0.7)])
    db.upsert_listings([L(price=10.0, shipping=0.0, seller="real_seller")])
    got = db.get_listing("100000000001")
    assert (got.price, got.shipping, got.seller) == (10.0, 0.0, "real_seller")
    assert got.extraction == "dom" and got.confidence is None
    assert set(sources(db, "100000000001").values()) == {"dom"}


def test_llm_sits_between_vision_and_dom(db):
    db.upsert_listings([L(price=9.0, extraction="vision", confidence=0.5)])
    db.upsert_listings([L(price=9.5, extraction="llm", confidence=0.9)])
    assert db.get_listing("100000000001").price == 9.5
    db.upsert_listings([L(price=9.0, extraction="vision", confidence=0.99)])
    got = db.get_listing("100000000001")
    assert got.price == 9.5 and got.extraction == "llm" and got.confidence == 0.9
    db.upsert_listings([L(price=10.0)])
    assert db.get_listing("100000000001").price == 10.0


def test_equal_precedence_overwrites_and_none_never_erases(db):
    db.upsert_listings([L(price=5.0, seller="abc", location="United States", bids=3, listing_format="auction")])
    db.upsert_listings([L(price=6.0, seller=None, location="  ", bids=None, listing_format="unknown")])
    got = db.get_listing("100000000001")
    assert got.price == 6.0
    assert (got.seller, got.location, got.bids, got.listing_format) == ("abc", "United States", 3, "auction")


def test_same_extractor_confidence_updates_only_when_given(db):
    db.upsert_listings([L(extraction="vision", confidence=0.4)])
    db.upsert_listings([L(extraction="vision", confidence=None)])
    assert db.get_listing("100000000001").confidence == 0.4
    db.upsert_listings([L(extraction="vision", confidence=0.8)])
    assert db.get_listing("100000000001").confidence == 0.8


def test_price_max_travels_with_price(db):
    # Vision cannot bolt a (mis)read range onto an exact DOM price...
    db.upsert_listings([L(price=5.0)])
    db.upsert_listings([L(price=5.0, price_max=50.0, extraction="vision")])
    assert db.get_listing("100000000001").price_max is None
    # ...and a DOM single price replaces an earlier range, even though price_max is None.
    db.upsert_listings([L("1002", price=3.75, price_max=23.95)])
    assert db.get_listing("1002").price_max == 23.95
    db.upsert_listings([L("1002", price=4.0)])
    got = db.get_listing("1002")
    assert (got.price, got.price_max) == (4.0, None)
    assert "price_max" not in sources(db, "1002")


def test_sponsored_flag_follows_precedence(db):
    db.upsert_listings([L(sponsored=True)])
    db.upsert_listings([L(sponsored=False, extraction="vision")])
    assert db.get_listing("100000000001").sponsored is True
    db.upsert_listings([L(sponsored=False)])
    assert db.get_listing("100000000001").sponsored is False


# --- synthetic ids and merging -------------------------------------------------------------


def test_synthetic_id_is_deterministic_and_documented_formula():
    sid = synthetic_item_id("www.ebay.com", "Hot Wheels R34", 12.5, date(2026, 4, 1))
    expected = "x-" + hashlib.sha1(b"www.ebay.com|hot wheels r34|12.50|2026-04-01").hexdigest()[:16]
    assert sid == expected and is_synthetic_id(sid) and not is_synthetic_id("123456789012")
    # Normalisation: case, punctuation, spacing, site spelling, "New listing" prefix.
    assert synthetic_item_id("ebay.com", "  HOT wheels,  R34!! ", 12.50, "2026-04-01") == sid
    assert synthetic_item_id("www.ebay.com", "New Listing Hot Wheels R34", 12.5, date(2026, 4, 1)) == sid
    assert synthetic_item_id("www.ebay.com", "Hot Wheels R34", 12.51, date(2026, 4, 1)) != sid
    assert synthetic_item_id("www.ebay.com", "Hot Wheels R34", 12.5, date(2026, 4, 2)) != sid
    assert synthetic_item_id("www.ebay.co.uk", "Hot Wheels R34", 12.5, date(2026, 4, 1)) != sid
    assert synthetic_item_id("www.ebay.com", "Hot Wheels R34", None, None).startswith("x-")
    assert normalize_title("Ｈot  Wheels—R34_Zamac") == "hot wheels r34 zamac"


def test_vision_rows_without_id_get_stable_synthetic_id(db):
    v = L(None, extraction="vision", confidence=0.8)
    s = db.record_search(SearchQuery(keywords="r34"), "u")
    assert db.save_page(s, page([v])) == UpsertStats(new=1)
    assert db.save_page(s, page([v])) == UpsertStats(unchanged=1)
    # Same normalised title -> same id; the new spelling replaces the old one (equal precedence).
    assert db.upsert_listings([v.model_copy(update={"title": "hot wheels nissan skyline gt r r34 zamac"})]).updated == 1
    rows = db.connection.execute("SELECT item_id, has_real_id FROM listings").fetchall()
    assert len(rows) == 1
    expected = synthetic_item_id("www.ebay.com", v.title, v.price, v.sold_date)
    assert rows[0]["item_id"] == expected and rows[0]["has_real_id"] == 0
    got = db.get_listing(expected)
    assert got.item_id == expected and got.extraction == "vision" and got.title == "hot wheels nissan skyline gt r r34 zamac"


def test_vision_after_dom_lands_on_the_real_row(db):
    s = db.record_search(SearchQuery(keywords="r34"), "u")
    db.save_page(s, page([L(condition="Pre-Owned")]))
    stats = db.save_page(s, page([L(None, condition="Used", shipping=3.0, extraction="vision")], 2))
    assert stats == UpsertStats(updated=1)
    assert count(db, "listings") == 1
    got = db.get_listing("100000000001")
    assert got.condition == "Pre-Owned" and got.shipping == 3.0
    obs = db.connection.execute("SELECT item_id, page, extraction FROM observations ORDER BY page").fetchall()
    assert [tuple(o) for o in obs] == [("100000000001", 1, "dom"), ("100000000001", 2, "vision")]
    # The synthetic id now resolves to the real listing.
    synth = synthetic_item_id("www.ebay.com", got.title, 10.0, date(2026, 4, 1))
    assert db.get_listing(synth).item_id == "100000000001"


def test_dom_after_vision_absorbs_the_synthetic_row(db):
    s1 = db.record_search(SearchQuery(keywords="r34"), "u")
    v = L(None, seller="ocr_seller", price=10.0, extraction="vision", confidence=0.6, position=4)
    db.save_page(s1, page([v], when=T0))
    synth = synthetic_item_id("www.ebay.com", v.title, v.price, v.sold_date)
    assert db.get_listing(synth) is not None
    s2 = db.record_search(SearchQuery(keywords="r34"), "u")
    stats = db.save_page(s2, page([L(condition="Pre-Owned", position=2)], when=T1))
    assert stats == UpsertStats(updated=1)  # the sale was already known
    rows = db.connection.execute("SELECT item_id, has_real_id, first_seen_at, last_seen_at FROM listings").fetchall()
    assert len(rows) == 1
    assert (rows[0]["item_id"], rows[0]["has_real_id"]) == ("100000000001", 1)
    assert (rows[0]["first_seen_at"], rows[0]["last_seen_at"]) == (T0.isoformat(), T1.isoformat())
    got = db.get_listing("100000000001")
    assert (got.seller, got.condition, got.extraction, got.confidence) == ("ocr_seller", "Pre-Owned", "dom", None)
    assert db.get_listing(synth).item_id == "100000000001"
    obs = db.connection.execute("SELECT search_id, item_id FROM observations ORDER BY search_id").fetchall()
    assert [tuple(o) for o in obs] == [(s1, "100000000001"), (s2, "100000000001")]
    # Re-reading the old screenshot again does not resurrect the synthetic row.
    assert db.upsert_listings([v]).unchanged == 1
    assert count(db, "listings") == 1


def test_absorb_when_both_seen_on_same_search_page(db):
    s = db.record_search(SearchQuery(keywords="r34"), "u")
    db.save_page(s, page([L(None, extraction="vision", matches_query=False)]))
    db.save_page(s, page([L(matches_query=True)]))
    assert count(db, "listings") == 1
    obs = db.connection.execute("SELECT item_id, matches_query, extraction FROM observations").fetchall()
    assert [tuple(o) for o in obs] == [("100000000001", 1, "dom")]


def test_lower_precedence_reading_of_same_page_keeps_dom_observation(db):
    s = db.record_search(SearchQuery(keywords="r34"), "u")
    db.save_page(s, page([L(position=3, matches_query=True)]))
    db.upsert_listings([L(None, extraction="vision", matches_query=False, position=None)], search_id=s, page=1)
    obs = db.connection.execute("SELECT position, matches_query, extraction FROM observations").fetchall()
    assert [tuple(o) for o in obs] == [(3, 1, "dom")]
    # A later DOM reading of the same page does replace it.
    db.save_page(s, page([L(position=5, matches_query=False)]))
    obs = db.connection.execute("SELECT position, matches_query, extraction FROM observations").fetchall()
    assert [tuple(o) for o in obs] == [(5, 0, "dom")]


FULL_TITLE = "Hot Wheels 2024 Nissan Skyline GT-R R34 Zamac Walmart Exclusive Car Culture"


def test_truncated_and_misread_titles_still_merge(db):
    full = FULL_TITLE
    db.upsert_listings([L("2001", title=full, price=12.0)])
    db.upsert_listings([L(None, title="Hot Wheels 2024 Nissan Skyline GT-R R34 Zamac Wal...", price=12.0,
                          extraction="vision")])
    db.upsert_listings([L(None, title=full.replace("Skyline", "Skyiine"), price=12.0, extraction="llm")])
    # Truncated *and* misread: neither a prefix nor close to the whole title.
    db.upsert_listings([L(None, title="Hot Wheels 2024 Nissan Skyiine GT-R R34 Zamac Walmart Exc...", price=12.0,
                          extraction="vision")])
    assert count(db, "listings") == 1
    assert db.get_listing("2001").title == full


@pytest.mark.parametrize("title", [
    "Hot Wheels 2024 Nissan Skyline GT-R R34 Zamac Wal...",
    FULL_TITLE.replace("Skyline", "Skyiine"),
    "Hot Wheels 2024 Nissan Skyiine GT-R R34 Zamac Walmart Exc...",
])
def test_dom_page_absorbs_truncated_or_misread_reading(db, title):
    db.upsert_listings([L(None, title=title, price=12.0, extraction="vision")])
    assert db.upsert_listings([L("2001", title=FULL_TITLE, price=12.0)]) == UpsertStats(updated=1)
    assert count(db, "listings") == 1 and db.get_listing("2001").title == FULL_TITLE


def test_title_tiers():
    t = dbmod._title_tier
    a = normalize_title(FULL_TITLE)
    assert t(a, a) == dbmod.TITLE_EQUAL
    assert t(a[:30], a) == t(a, a[:30]) == dbmod.TITLE_TRUNCATED
    assert t(a.replace("skyline", "skyiine"), a) == dbmod.TITLE_MISREAD
    assert t(a[:50].replace("skyline", "skyiine"), a) == dbmod.TITLE_TRUNCATED_MISREAD
    assert t("hot wheels r34", "hot wheels r34 zamac walmart exclusive car culture") == 0  # prefix too short
    assert t(a, normalize_title("Matchbox 2024 Nissan Skyline GT-R R34 Blue Moving Parts Exclusive")) == 0


def test_partial_readings_of_one_card_are_one_sale():
    """A card cut off at a tile edge is read without its date (or price) beside a full reading of it."""
    full = L(None, price=49.99, sold_date=date(2026, 4, 25), shipping=5.0, extraction="llm", confidence=0.9)
    no_date = full.model_copy(update={"sold_date": None, "shipping": None})
    no_price = full.model_copy(update={"price": None})
    for batch in ([full, no_date, no_price], [no_date, full], [no_price, no_date, full], [no_date, no_price, full]):
        with Database(":memory:") as d:
            s = d.record_search(SearchQuery(keywords="r34"), "u")
            stats = d.save_page(s, page(batch))
            assert stats.new == 1 and stats.total == len(batch)
            assert count(d, "listings") == 1 and count(d, "observations") == 1
            got = d.query_listings(keywords="r34")[0]
            assert (got.price, got.sold_date, got.shipping) == (49.99, date(2026, 4, 25), 5.0)
            assert d.price_stats(keywords="r34")["USD"].count == 1
            # Each reading's id now resolves to the one row; re-reading changes nothing.
            for x in batch:
                assert d.get_listing(synthetic_item_id(x.site, x.title, x.price, x.sold_date)) == d.get_listing(
                    got.item_id)
            assert d.save_page(s, page(batch)).unchanged == len(batch)
            assert count(d, "listings") == 1


def test_partial_readings_merge_with_dom_in_either_order():
    dom = L(price=3.75, price_max=23.95)
    readings = [L(None, price=3.75, sold_date=None, extraction="vision", seller="ocr_seller"),
                L(None, price=None, extraction="vision")]
    for order in ("dom first", "readings first"):
        with Database(":memory:") as d:
            batches = [[dom], readings] if order == "dom first" else [readings, [dom]]
            for batch in batches:
                d.upsert_listings(batch)
            assert count(d, "listings") == 1, order
            got = d.get_listing("100000000001")
            assert (got.price, got.price_max, got.seller, got.extraction) == (3.75, 23.95, "ocr_seller", "dom")
            assert d.price_stats()["USD"].count == 1


def test_dom_absorbs_every_reading_of_a_sale_at_once(db):
    db.upsert_listings([L(None, title=FULL_TITLE, extraction="vision"),
                        L(None, title=FULL_TITLE.replace("Skyline", "Skyiine"), extraction="vision"),
                        L(None, title=FULL_TITLE, sold_date=None, extraction="vision")])
    assert count(db, "listings") == 2  # the misread one is not merged with the others: no ground truth
    dom = L(title=FULL_TITLE)
    assert db.upsert_listings([dom]) == UpsertStats(updated=1)
    assert count(db, "listings") == 1 and db.price_stats()["USD"].count == 1
    assert db.upsert_listings([dom]) == UpsertStats(unchanged=1)  # saving again changes nothing


def test_truncated_reading_does_not_replace_full_title(db):
    db.upsert_listings([L(None, title=FULL_TITLE, extraction="llm")])
    db.upsert_listings([L(None, title=FULL_TITLE[:45] + "...", sold_date=None, extraction="llm")])
    assert count(db, "listings") == 1
    assert db.query_listings()[0].title == FULL_TITLE


def test_different_price_or_date_is_a_different_sale(db):
    db.upsert_listings([L("2001")])
    db.upsert_listings([L(None, price=10.5, extraction="vision"), L(None, sold_date=date(2026, 4, 2),
                                                                      extraction="vision")])
    assert count(db, "listings") == 3


def test_ambiguous_match_is_left_separate(db):
    # Two different sales the (truncated) reading fits equally well.
    db.upsert_listings([L("2001", title="Hot Wheels Nissan Skyline GT-R R34 Zamac Walmart"),
                        L("2002", title="Hot Wheels Nissan Skyline GT-R R34 Zamac Target Exclusive")])
    db.upsert_listings([L(None, extraction="vision")])
    rows = db.connection.execute("SELECT has_real_id FROM listings ORDER BY has_real_id").fetchall()
    assert [r[0] for r in rows] == [0, 1, 1]
    # Re-reading it is still idempotent.
    assert db.upsert_listings([L(None, extraction="vision")]).unchanged == 1
    assert count(db, "listings") == 3


def test_partial_reading_that_fits_two_sales_is_ambiguous(db):
    db.upsert_listings([L("2001", sold_date=date(2026, 4, 1)), L("2002", sold_date=date(2026, 4, 2))])
    db.upsert_listings([L(None, sold_date=None, extraction="vision")])  # no date: could be either sale
    assert count(db, "listings") == 3
    # Nor does a DOM page with both sales absorb it into one of them.
    db.upsert_listings([L("2001", sold_date=date(2026, 4, 1)), L("2002", sold_date=date(2026, 4, 2))])
    assert count(db, "listings") == 3


def test_indistinguishable_sales_absorb_readings_in_any_order(db):
    # A seller's identical items sold the same day: any of the rows is the right home
    # for a reading, so it is not left over to be counted twice.
    dom = [L(str(3000 + i), position=i) for i in range(3)]
    readings = [L(None, extraction="vision", title=t) for t in (
        "Hot Wheels Nissan Skyline GT-R R34 Zamac",
        "Hot Wheels Nissan Skyiine GT-R R34 Zamac",
        "Hot Wheels Nissan Skyline GT-R R34 Za")]
    for order in ("vision first", "dom first"):
        with Database(":memory:") as d:
            s = d.record_search(SearchQuery(keywords="r34"), "u")
            pages = [(readings, 2), (dom, 1)] if order == "vision first" else [(dom, 1), (readings, 2)]
            for listings, n in pages:
                d.save_page(s, page(listings, n))
            rows = d.connection.execute("SELECT item_id FROM listings ORDER BY item_id").fetchall()
            assert [r[0] for r in rows] == ["3000", "3001", "3002"], order
            assert d.price_stats()["USD"].count == 3


def test_id_less_listing_without_price_or_date_never_merges(db):
    db.upsert_listings([L("2001", price=None, sold_date=None)])
    db.upsert_listings([L(None, price=None, sold_date=None, extraction="vision")])
    assert count(db, "listings") == 2


# --- observations and matches_query -------------------------------------------------------------


@pytest.fixture
def observed(db):
    """Search A sees 1 (match), 2 (loose); search B sees 2 (match); 3 seen only loose; 4 imported."""
    a = db.record_search(SearchQuery(keywords="Hot Wheels R34"), "ua")
    db.save_page(a, page([L("1", position=1, sold_date=date(2026, 4, 3)),
                          L("2", matches_query=False, position=2, sold_date=date(2026, 4, 2)),
                          L("3", matches_query=False, position=3, sold_date=date(2026, 4, 1))]))
    b = db.record_search(SearchQuery(keywords="r34 zamac"), "ub")
    db.save_page(b, page([L("2", position=7, sold_date=date(2026, 4, 2))], when=T1))
    db.upsert_listings([L("4", sold_date=date(2026, 3, 1))])
    return db


def ids(listings: list[Listing]) -> list[str | None]:
    return [x.item_id for x in listings]


def test_matches_query_is_per_search(observed):
    db = observed
    assert ids(db.query_listings(keywords="hot   wheels r34")) == ["1"]
    loose = db.query_listings(keywords="HOT WHEELS R34", exact_matches_only=False)
    assert ids(loose) == ["1", "2", "3"]
    assert [x.matches_query for x in loose] == [True, False, False]
    assert [x.position for x in loose] == [1, 2, 3]
    b = db.query_listings(keywords="R34 Zamac")
    assert ids(b) == ["2"] and b[0].matches_query is True and b[0].position == 7
    assert db.query_listings(keywords="never searched") == []


def test_without_keywords_exact_means_matched_somewhere_or_never_observed(observed):
    db = observed
    exact = db.query_listings()
    assert ids(exact) == ["1", "2", "4"]
    two = exact[1]
    assert two.matches_query is True and two.position == 7  # latest observation
    assert ids(db.query_listings(exact_matches_only=False)) == ["1", "2", "3", "4"]
    assert db.get_listing("3").matches_query is False
    assert db.get_listing("4").matches_query is True and db.get_listing("4").position is None


def test_searches_summary(observed):
    s = observed.searches()
    assert [x["keywords"] for x in s] == ["Hot Wheels R34", "r34 zamac"]
    assert (s[0]["pages"], s[0]["listings"], s[0]["exact_matches"]) == (1, 3, 1)
    assert (s[1]["pages"], s[1]["listings"], s[1]["exact_matches"]) == (1, 1, 1)
    assert s[0]["query"]["keywords"] == "Hot Wheels R34" and s[0]["url"] == "ua"
    summary = observed.summary()
    assert summary["listings"] == 4 and summary["observations"] == 4 and summary["searches"] == 2
    assert summary["real_ids"] == 4 and summary["synthetic_ids"] == 0
    assert summary["by_currency"] == {"USD": 4} and summary["last_sold"] == "2026-04-03"


def test_search_pages_are_recorded(db):
    s = db.record_search(SearchQuery(keywords="r34"), "u")
    db.save_page(s, page([L()], total_results=31, has_next_page=False, html_path="/x.html", from_cache=True))
    r = db.connection.execute("SELECT * FROM search_pages").fetchone()
    assert (r["total_results"], r["has_next_page"], r["html_path"], r["from_cache"], r["listings"]) == (
        31, 0, "/x.html", 1, 1)
    assert db.searches()[0]["total_results"] == 31


# --- filters ----------------------------------------------------------------------------------


@pytest.fixture
def mixed(db):
    db.upsert_listings([
        L("1", sold_date=date(2026, 4, 10), condition="Pre-Owned", price=10.0),
        L("2", sold_date=date(2026, 4, 5), condition="Brand New", price=20.0),
        L("3", sold_date=date(2026, 3, 31), condition="pre-owned", price=30.0),
        L("4", sold_date=None, price=40.0),
        L("5", sold_date=date(2026, 4, 7), price=7.0, currency="GBP", site="www.ebay.co.uk"),
        L("6", sold_date=date(2026, 4, 8), price=8.0, sponsored=True),
    ])
    return db


def test_order_newest_first_undated_last(mixed):
    assert ids(mixed.query_listings()) == ["1", "5", "2", "3", "4"]
    assert ids(mixed.query_listings(include_sponsored=True)) == ["1", "6", "5", "2", "3", "4"]
    assert ids(mixed.query_listings(limit=2)) == ["1", "5"]


def test_date_condition_currency_site_filters(mixed):
    assert ids(mixed.query_listings(since=date(2026, 4, 5))) == ["1", "5", "2"]
    assert ids(mixed.query_listings(since="2026-04-05", until="2026-04-07")) == ["5", "2"]
    assert ids(mixed.query_listings(until=date(2026, 3, 31))) == ["3"]
    assert ids(mixed.query_listings(condition="PRE-OWNED")) == ["1", "3"]
    assert ids(mixed.query_listings(currency="gbp")) == ["5"]
    assert ids(mixed.query_listings(site="ebay.co.uk")) == ["5"]
    assert ids(mixed.query_listings(site="www.ebay.com", currency="USD", since=date(2026, 4, 1))) == ["1", "2"]
    # Empty strings mean "no bound", like None (they used to crash).
    assert ids(mixed.query_listings(since="", until="")) == ids(mixed.query_listings())
    assert mixed.price_stats(since="", until="") == mixed.price_stats()


# --- price statistics ------------------------------------------------------------------------


def pct(values: list[float], p: float) -> float:
    """Independent linear-interpolation percentile (Hyndman-Fan type 7)."""
    xs = sorted(values)
    h = (len(xs) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def check_stats(s: PriceStats, values: list[float]) -> None:
    assert s.count == len(values)
    assert s.min == pytest.approx(min(values), abs=0.006)
    assert s.max == pytest.approx(max(values), abs=0.006)
    assert s.mean == pytest.approx(sum(values) / len(values), abs=0.006)
    for name, p in (("p10", 0.10), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.90)):
        assert getattr(s, name) == pytest.approx(pct(values, p), abs=0.006), name


PRICES = [12.5, 15.0, 9.99, 22.0, 18.75, 14.0, 11.25, 16.5, 13.0, 19.99, 10.0]


def test_stats_match_independent_calculation(db):
    db.upsert_listings([L(str(i), price=p, sold_date=date(2026, 4, 1 + i)) for i, p in enumerate(PRICES)])
    stats = db.price_stats()
    assert list(stats) == ["USD"]
    s = stats["USD"]
    assert s.outliers_removed == 0  # nothing outside 1.5 x IQR here
    check_stats(s, PRICES)
    assert (s.first_sold, s.last_sold) == (date(2026, 4, 1), date(2026, 4, 11))
    assert s.include_shipping is False and s.shipping_unknown == 0


def test_outlier_trimming_on_and_off(db):
    values = PRICES + [250.0, 60.0]
    db.upsert_listings([L(str(i), price=p) for i, p in enumerate(values)])
    q1, q3 = pct(values, 0.25), pct(values, 0.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    kept = [v for v in values if lo <= v <= hi]
    assert len(kept) == len(values) - 2  # both extremes fall outside the fences
    trimmed = db.price_stats()["USD"]
    assert trimmed.outliers_removed == 2
    check_stats(trimmed, kept)
    raw = db.price_stats(trim_outliers=False)["USD"]
    assert raw.outliers_removed == 0 and raw.max == 250.0
    check_stats(raw, values)


def test_no_trimming_below_eight_listings(db):
    values = [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 500.0]
    db.upsert_listings([L(str(i), price=p) for i, p in enumerate(values)])
    s = db.price_stats()["USD"]
    assert s.outliers_removed == 0 and s.max == 500.0
    check_stats(s, values)


def test_include_shipping(db):
    db.upsert_listings([
        L("1", price=10.0, shipping=5.0),
        L("2", price=20.0, shipping=0.0),
        L("3", price=30.0, shipping=None),  # unknown: counted at item price
        L("4", price=None, shipping=4.0),  # "See price": excluded
    ])
    plain = db.price_stats()["USD"]
    check_stats(plain, [10.0, 20.0, 30.0])
    with_ship = db.price_stats(include_shipping=True)["USD"]
    check_stats(with_ship, [15.0, 20.0, 30.0])
    assert with_ship.include_shipping is True and with_ship.shipping_unknown == 1


def test_price_range_counts_at_low_end(db):
    db.upsert_listings([L("1", price=3.75, price_max=23.95), L("2", price=5.0)])
    s = db.price_stats()["USD"]
    assert (s.min, s.max, s.count) == (3.75, 5.0, 2)


def test_currencies_are_never_mixed(db):
    usd = [10.0, 12.0, 14.0]
    gbp = [8.0, 9.0]
    db.upsert_listings([L(f"u{i}", price=p) for i, p in enumerate(usd)]
                       + [L(f"g{i}", price=p, currency="GBP") for i, p in enumerate(gbp)]
                       + [L("n0", price=99.0, currency=None, extraction="vision")])
    stats = db.price_stats()
    assert list(stats) == ["USD", "GBP", "UNKNOWN"]  # largest sample first
    check_stats(stats["USD"], usd)
    check_stats(stats["GBP"], gbp)
    assert stats["UNKNOWN"].count == 1 and stats["UNKNOWN"].median == 99.0
    assert list(db.price_stats(currency="GBP")) == ["GBP"]
    assert list(db.price_stats(currency="unknown")) == ["UNKNOWN"]


def test_stats_respect_query_scope(observed):
    stats = observed.price_stats(keywords="hot wheels r34")
    assert stats["USD"].count == 1
    assert observed.price_stats(keywords="hot wheels r34", exact_matches_only=False)["USD"].count == 3


def test_single_listing_and_empty_stats(db):
    assert db.price_stats() == {}
    assert db.price_stats(keywords="nothing") == {}
    db.upsert_listings([L(price=None)])
    assert db.price_stats() == {}
    db.upsert_listings([L("2", price=42.0)])
    s = db.price_stats()["USD"]
    assert s.count == 1 and s.min == s.max == s.median == s.p10 == s.p90 == 42.0


def test_price_series_by_month(db):
    db.upsert_listings([
        L("1", price=10.0, sold_date=date(2026, 3, 30)),
        L("2", price=20.0, sold_date=date(2026, 3, 2)),
        L("3", price=30.0, sold_date=date(2026, 4, 1)),
        L("4", price=50.0, sold_date=None),
    ])
    series = db.price_series(period="month")
    assert [(r["period"], r["currency"], r["count"], r["median"]) for r in series] == [
        ("2026-03", "USD", 2, 15.0), ("2026-04", "USD", 1, 30.0)]
    weeks = db.price_series(period="week")
    assert [r["period"] for r in weeks] == ["2026-W10", "2026-W14"]
    with pytest.raises(ValueError):
        db.price_series(period="decade")  # type: ignore[arg-type]
    assert [r["period"] for r in db.price_series(period="month", limit=2)] == ["2026-03", "2026-04"]


def test_price_series_rejects_bad_period_even_without_rows(db):
    with pytest.raises(ValueError, match="period"):
        db.price_series(period="decade")  # type: ignore[arg-type]
    db.upsert_listings([L(sold_date=None)])  # only undated rows
    with pytest.raises(ValueError, match="period"):
        db.price_series(period="decade")  # type: ignore[arg-type]


def test_price_stats_limit_takes_most_recent_priced_sales(db):
    db.upsert_listings([L(str(i), price=float(i), sold_date=date(2026, 4, i)) for i in range(1, 11)]
                       + [L("99", price=None, sold_date=date(2026, 4, 30))])  # newest, but no price
    s = db.price_stats(limit=3)["USD"]
    check_stats(s, [8.0, 9.0, 10.0])
    assert (s.first_sold, s.last_sold) == (date(2026, 4, 8), date(2026, 4, 10))
    assert db.price_stats(limit=0) == {}
    assert db.price_stats(limit=None)["USD"].count == 10


# --- export --------------------------------------------------------------------------------


@pytest.fixture
def exportable(db):
    s = db.record_search(SearchQuery(keywords="r34"), "u")
    db.save_page(s, page([
        L("1", title='Hot Wheels "R34", Zamac — 1:64 ünïcode', price=3.75, price_max=23.95, original_price=30.0,
          price_text="$3.75 to $23.95", shipping=4.5, shipping_text="+$4.50 delivery", sold_date=date(2026, 4, 25),
          sold_date_text="Sold  Apr 25, 2026", condition="Pre-Owned", listing_format="auction", bids=11,
          seller="seller,one", seller_feedback_pct=99.5, seller_feedback_count=1234, location="United States",
          url="https://www.ebay.com/itm/1", image_url="https://i.ebayimg.com/x.jpg", position=1),
        L("2", title="Line\nbreak title", price=None, shipping=0.0, position=2, matches_query=False),
    ]))
    db.upsert_listings([L(None, price=9.0, extraction="vision", confidence=0.75)])
    return db


def _from_csv(rec: dict[str, str]) -> Listing:
    return Listing.model_validate({k: (None if v == "" else v) for k, v in rec.items() if k in Listing.model_fields})


def test_csv_round_trip(exportable, tmp_path):
    out = tmp_path / "out" / "sold.csv"
    n = exportable.export_csv(out, exact_matches_only=False)
    expected = exportable.query_listings(exact_matches_only=False)
    assert n == len(expected) == 3
    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == Database.export_columns()
        rows = list(reader)
    assert [_from_csv(r) for r in rows] == expected
    first = rows[0]
    assert first["sold_date"] == "2026-04-25" and first["total_price"] == "8.25"
    assert first["has_real_id"] == "true" and first["first_seen_at"] == T0.isoformat()
    synthetic = next(r for r in rows if r["item_id"].startswith("x-"))
    assert synthetic["has_real_id"] == "false" and synthetic["extraction"] == "vision"


def test_json_round_trip(exportable, tmp_path):
    out = tmp_path / "sold.json"
    n = exportable.export_json(out, exact_matches_only=False)
    records = json.loads(out.read_text(encoding="utf-8"))
    expected = exportable.query_listings(exact_matches_only=False)
    assert n == len(records) == 3
    assert list(records[0]) == Database.export_columns()
    assert [Listing.model_validate({k: v for k, v in r.items() if k in Listing.model_fields}) for r in records] \
        == expected
    assert records[0]["sold_date"] == "2026-04-25" and records[0]["total_price"] == 8.25
    # total_price needs both price and shipping: "See price" + free shipping, and price + unknown shipping.
    totals = {r["item_id"]: (r["price"], r["shipping"], r["total_price"]) for r in records}
    synthetic = next(i for i in totals if i.startswith("x-"))
    assert totals == {"1": (3.75, 4.5, 8.25), "2": (None, 0.0, None), synthetic: (9.0, None, None)}


def test_export_applies_filters(exportable, tmp_path):
    assert exportable.export_json(tmp_path / "a.json") == 2  # listing 2 was a loose match
    assert exportable.export_csv(tmp_path / "b.csv", keywords="r34") == 1
    assert exportable.export_csv(tmp_path / "c.csv", keywords="unknown words") == 0
    with (tmp_path / "c.csv").open() as fh:
        assert fh.read().startswith("item_id,site,title")


# --- with the real DOM parser -------------------------------------------------------------------


def test_real_fixture_pages_dedup(db, ebay_page):
    parse = pytest.importorskip("ebay_sold.parse")
    name, html = ebay_page
    parsed = parse.parse_search_page(html, today=date(2026, 10, 8))
    if not parsed.listings:
        pytest.skip(f"{name}: parser found no listings")
    s = db.record_search(SearchQuery(keywords=name), "file://" + name)
    first = db.save_page(s, page(parsed.listings, total_results=parsed.total_results))
    unique = {(x.site, x.item_id) for x in parsed.listings}
    assert first.new == len(unique) == count(db, "listings")
    assert db.save_page(s, page(parsed.listings)) == UpsertStats(unchanged=len(parsed.listings))
    exact = [x for x in parsed.listings if x.matches_query and not x.sponsored]
    assert len(db.query_listings(keywords=name)) == len({x.item_id for x in exact})
    stats = db.price_stats(keywords=name, exact_matches_only=False, include_sponsored=True)
    assert sum(s.count + s.outliers_removed for s in stats.values()) == len(
        {x.item_id for x in parsed.listings if x.price is not None})


def _ocr_like(title: str) -> str:
    """Truncated to 60 characters (like a clipped card) with one letter misread."""
    t = title[:60] + "..." if len(title) > 63 else title
    letters = [i for i, c in enumerate(t) if c.isalpha()]
    i = letters[len(letters) // 2]
    return t[:i] + ("l" if t[i] != "l" else "i") + t[i + 1:]


@pytest.mark.parametrize("vision_first", [True, False], ids=["vision-first", "dom-first"])
def test_real_fixture_vision_readings_merge_with_dom(ebay_page, vision_first):
    """Screenshot readings (no ids, OCR-like titles) and the DOM page of the same results are one set of sales."""
    parse = pytest.importorskip("ebay_sold.parse")
    name, html = ebay_page
    parsed = parse.parse_search_page(html, today=date(2026, 10, 8))
    if not parsed.listings:
        pytest.skip(f"{name}: parser found no listings")
    readings = [x.model_copy(update={"item_id": None, "url": None, "extraction": "vision", "confidence": 0.9,
                                     "title": _ocr_like(x.title)}) for x in parsed.listings]
    # A card cut at a tile edge, read a second time without its sold date.
    readings.append(readings[len(readings) // 2].model_copy(update={"sold_date": None}))
    batches = [readings, parsed.listings] if vision_first else [parsed.listings, readings]
    with Database(":memory:") as d:
        s = d.record_search(SearchQuery(keywords=name), "u")
        for n, batch in enumerate(batches, 1):
            d.save_page(s, page(batch, n))
        assert count(d, "listings") == len({x.item_id for x in parsed.listings})
        assert d.summary()["synthetic_ids"] == 0
        stats = d.price_stats(keywords=name, exact_matches_only=False, include_sponsored=True, trim_outliers=False)
        assert sum(x.count for x in stats.values()) == len({x.item_id for x in parsed.listings if x.price is not None})
        # DOM values were never overwritten by the readings.
        for x in parsed.listings:
            got = d.get_listing(x.item_id)
            assert got.model_dump(exclude={"matches_query", "position"}) == x.model_dump(
                exclude={"matches_query", "position"})
