"""SQLite storage for sold listings: dedup, provenance, export and price statistics.

Standard-library ``sqlite3`` only. One file holds everything; it is opened in
WAL mode so ``ebay-sold stats`` can read while a scrape is writing.

Data model, and why:

* **Listings are keyed by (site, item_id).** eBay item ids are global, but the
  same item seen on www.ebay.com and www.ebay.co.uk is shown converted into a
  different currency, so each site keeps its own row.
* **Synthetic ids.** The vision and Claude extractors read screenshots and
  usually cannot see an item id. Such listings get the deterministic id
  ``"x-" + sha1(site|normalized title|price|sold_date)[:16]`` and
  ``has_real_id = 0``. Reading the same card again yields the same id. Real ids
  are all digits, so the two never clash.
* **Readings of one sale are matched, not just hashed.** A card cut off at a
  tile edge, or an OCR miss, gives a reading without its sold date or price,
  and so a different synthetic id. Two readings are taken to be the same sale
  when they are *compatible*: price and sold date each agree wherever both
  readings show them, at least one of the two is shown by both, and the titles
  agree at one of these tiers, best first: equal after normalisation; one a
  20+ character prefix of the other (truncated); near-identical (an OCR
  misread, difflib ratio >= 0.9); a 20+ character truncation with a misread
  (ratio >= 0.95 against the same-length prefix). Ties go to the candidate that
  shares more of price and date. A reading that fits several stored sales
  equally well is ambiguous and stays a row of its own, unless those sales are
  indistinguishable (same title, price and date: a seller's identical items
  sold the same day). Then any of them is equally right, it goes to the lowest
  id, and the sale is not counted twice. A reading with neither price nor date
  never merges: too little to tell sales of one title apart.
* **Merging.** An id-less listing is stored on the real-id row it fits best,
  else on the synthetic row holding another reading of it (synthetic to
  synthetic only on the two exact title tiers, since neither side is ground
  truth). Once a batch (one page) is stored, synthetic rows that are other
  readings of a sale it wrote, and that fit no third row as well (the batch's
  other listings included), are absorbed into it: fields, observations,
  first/last seen; then deleted. So a DOM page absorbs every earlier reading of
  each of its sales, and a full reading absorbs partial ones. A synthetic row
  keeps the id of its first reading; every absorbed or redirected synthetic id
  becomes an alias, so a later re-read of the same screenshot lands on the
  right row without guessing again.
* **Extraction precedence dom > llm > vision, per field.** ``field_sources``
  remembers which extractor set each stored field. A lower-precedence
  extraction never overwrites a field set by a higher one; equal or higher
  precedence overwrites with its non-null values; ``None`` (and an empty
  string, and ``listing_format == "unknown"``) never erases a stored value.
  One deliberate exception: ``price_max`` describes ``price`` (a "$3.75 to
  $23.95" range), so it is written together with ``price`` and only then,
  even when ``None``. Otherwise vision could bolt a misread range onto an
  exact DOM price. At equal precedence a title that merely truncates the stored
  one does not replace it. ``extraction`` on the row is the best extractor that
  contributed; ``confidence`` belongs to that extractor (``None`` = DOM, exact).
* **matches_query is per observation, not per listing.** A listing is "an exact
  match" for one search and a "results matching fewer words" filler for
  another. Every time a search shows a listing we store an observation
  ``(search_id, site, item_id, page, position, matches_query, sponsored,
  observed_at)``; ``page = 0`` means "not from a result page" (e.g. screenshots
  saved with ``upsert_listings``). Reading the same page again replaces its
  observation unless the earlier reading came from a more reliable extractor.
  ``query_listings(keywords=...)`` looks only
  at observations from searches whose keywords are equal ignoring case and
  spacing, and ``exact_matches_only`` keeps listings observed as a match in at
  least one of them. Without keywords, ``exact_matches_only`` keeps listings
  matched in any observation, plus listings with no observations at all
  (imported directly, nothing says they are loose matches).
* **Sponsored** is stored on the listing (latest value under the precedence
  rules) and, for the record, on each observation. ``include_sponsored=False``
  drops listings whose stored flag is set.
* **Schema versions** live in ``PRAGMA user_version``; ``MIGRATIONS`` is an
  append-only list of SQL scripts, each applied once in its own write
  transaction. The version is read again after taking the write lock, so two
  processes opening a new (or old) file at once migrate it once. A file from a
  newer ebay-sold is refused before anything is written to it. To add a
  column, append ``ALTER TABLE ...`` to the list and extend the field tuples below.

Price statistics never mix currencies: they are computed per currency, and
listings whose currency is unknown are reported under ``"UNKNOWN"`` rather than
guessed. Listings without a price ("See price") are left out; a price range
counts at its low end. With ``include_shipping``, shipping is added when it is
known (free shipping adds 0); a listing whose shipping is unknown counts at its
item price alone and is tallied in ``PriceStats.shipping_unknown``, so you can
tell how many totals are understated. ``limit`` on statistics means the most
recently sold N *priced* listings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sqlite3
import statistics
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from .models import Listing, PageResult, SearchQuery
from .normalize import clean_text
from .urls import item_id_from_url, normalize_site

if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import Self

log = logging.getLogger(__name__)

PRECEDENCE: dict[str, int] = {"vision": 0, "llm": 1, "dom": 2}
_RANK_SQL = "(CASE {0}.extraction " + " ".join(f"WHEN '{k}' THEN {v}" for k, v in PRECEDENCE.items()) + " ELSE -1 END)"
UNKNOWN_CURRENCY = "UNKNOWN"

# Listing fields stored on the listings row and merged under the precedence
# rules. Order matters only in that "price" comes before "price_max".
MERGED_FIELDS: tuple[str, ...] = (
    "title",
    "price",
    "price_max",
    "original_price",
    "currency",
    "price_text",
    "shipping",
    "shipping_text",
    "sold_date",
    "sold_date_text",
    "condition",
    "listing_format",
    "bids",
    "seller",
    "seller_feedback_pct",
    "seller_feedback_count",
    "location",
    "url",
    "image_url",
    "sponsored",
)
# Listing fields that are not merged columns: identity, per-observation, and provenance.
IDENTITY_FIELDS: tuple[str, ...] = ("item_id", "site")
OBSERVATION_FIELDS: tuple[str, ...] = ("matches_query", "position")
PROVENANCE_FIELDS: tuple[str, ...] = ("extraction", "confidence")
EXPORT_EXTRA_FIELDS: tuple[str, ...] = ("total_price", "has_real_id", "first_seen_at", "last_seen_at")

_V1 = """
CREATE TABLE searches (
    id INTEGER PRIMARY KEY,
    keywords TEXT NOT NULL,
    keywords_norm TEXT NOT NULL,
    site TEXT NOT NULL,
    url TEXT NOT NULL,
    query_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX searches_keywords ON searches (keywords_norm);

CREATE TABLE search_pages (
    search_id INTEGER NOT NULL REFERENCES searches (id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    url TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    total_results INTEGER,
    listings INTEGER NOT NULL,
    html_path TEXT,
    screenshot_path TEXT,
    from_cache INTEGER NOT NULL DEFAULT 0,
    has_next_page INTEGER,
    PRIMARY KEY (search_id, page)
);

CREATE TABLE listings (
    site TEXT NOT NULL,
    item_id TEXT NOT NULL,
    has_real_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    price REAL,
    price_max REAL,
    original_price REAL,
    currency TEXT,
    price_text TEXT,
    shipping REAL,
    shipping_text TEXT,
    sold_date TEXT,
    sold_date_text TEXT,
    condition TEXT,
    listing_format TEXT,
    bids INTEGER,
    seller TEXT,
    seller_feedback_pct REAL,
    seller_feedback_count INTEGER,
    location TEXT,
    url TEXT,
    image_url TEXT,
    sponsored INTEGER NOT NULL DEFAULT 0,
    extraction TEXT NOT NULL,
    confidence REAL,
    field_sources TEXT NOT NULL DEFAULT '{}',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (site, item_id)
);
CREATE INDEX listings_sold_date ON listings (sold_date);
CREATE INDEX listings_match_key ON listings (site, sold_date, price);

CREATE TABLE observations (
    id INTEGER PRIMARY KEY,
    search_id INTEGER NOT NULL REFERENCES searches (id) ON DELETE CASCADE,
    site TEXT NOT NULL,
    item_id TEXT NOT NULL,
    page INTEGER NOT NULL DEFAULT 0,
    position INTEGER,
    matches_query INTEGER NOT NULL,
    sponsored INTEGER NOT NULL DEFAULT 0,
    extraction TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    UNIQUE (search_id, site, item_id, page),
    FOREIGN KEY (site, item_id) REFERENCES listings (site, item_id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX observations_listing ON observations (site, item_id);

CREATE TABLE listing_aliases (
    site TEXT NOT NULL,
    alias_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    PRIMARY KEY (site, alias_id),
    FOREIGN KEY (site, item_id) REFERENCES listings (site, item_id) ON DELETE CASCADE ON UPDATE CASCADE
);
"""

# Matching an id-less reading looks rows up by price or by sold date, among
# real-id or among synthetic rows; has_real_id leads so a DOM write finds the
# (usually few) synthetic rows without scanning the real ones.
_V2 = """
DROP INDEX listings_match_key;
CREATE INDEX listings_match_price ON listings (site, has_real_id, price, sold_date);
CREATE INDEX listings_match_date ON listings (site, has_real_id, sold_date, price);
"""

# Append-only. Entry N (0-based) upgrades a database from user_version N to N + 1.
MIGRATIONS: tuple[str, ...] = (_V1, _V2)
SCHEMA_VERSION = len(MIGRATIONS)


def _sql_statements(script: str) -> list[str]:
    """Split a migration script into single statements (``execute`` takes one at a time)."""
    out, start = [], 0
    for i, ch in enumerate(script):
        if ch == ";" and sqlite3.complete_statement(script[start : i + 1]):
            out.append(script[start : i + 1].strip())
            start = i + 1
    out.append(script[start:].strip())  # a trailing statement without ";"
    return [s for s in out if s.strip(";").strip()]


class UpsertStats(BaseModel):
    """What an upsert did: ``new`` sales, ``updated`` rows, ``unchanged`` re-sightings."""

    new: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def total(self) -> int:
        return self.new + self.updated + self.unchanged

    def __add__(self, other: UpsertStats) -> UpsertStats:
        return UpsertStats(new=self.new + other.new, updated=self.updated + other.updated,
                           unchanged=self.unchanged + other.unchanged)


class PriceStats(BaseModel):
    """Sold-price distribution for one currency. Money values are rounded to cents."""

    currency: str
    count: int
    outliers_removed: int = 0
    min: float
    max: float
    mean: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float
    first_sold: date | None = None
    last_sold: date | None = None
    include_shipping: bool = False
    shipping_unknown: int = 0  # with include_shipping: listings counted at item price only


# --- ids and normalisation ---------------------------------------------------------

_NEW_LISTING_RE = re.compile(r"^new listing\s+")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_title(title: str | None) -> str:
    """Case-, width- and punctuation-insensitive form of a title, for matching."""
    text = unicodedata.normalize("NFKC", clean_text(title)).casefold()
    text = _NON_WORD_RE.sub(" ", text).strip()
    return _NEW_LISTING_RE.sub("", text)


def normalize_keywords(keywords: str) -> str:
    return " ".join(clean_text(keywords).casefold().split())


def synthetic_item_id(site: str, title: str | None, price: float | None, sold_date: date | str | None) -> str:
    """Deterministic id for a listing whose eBay item id was not visible."""
    price_key = "" if price is None else f"{price:.2f}"
    date_key = "" if sold_date is None else (sold_date if isinstance(sold_date, str) else sold_date.isoformat())
    key = f"{_site(site)}|{normalize_title(title)}|{price_key}|{date_key}"
    return "x-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def is_synthetic_id(item_id: str | None) -> bool:
    return bool(item_id and item_id.startswith("x-"))


def _site(site: str | None) -> str:
    site = site or "www.ebay.com"
    try:
        return normalize_site(site)
    except ValueError:
        return site.strip().lower()


def _iso_utc(when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    if when.tzinfo is None:  # naive datetimes are taken to be UTC
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).isoformat(timespec="seconds")


def _as_date(value: date | datetime | str | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


# --- merging -----------------------------------------------------------------------


def _db_values(item: Listing) -> dict[str, Any]:
    """Listing -> column values, with every "not shown" spelling turned into None."""
    out: dict[str, Any] = {}
    for f in MERGED_FIELDS:
        v = getattr(item, f)
        if isinstance(v, str):
            v = v.strip() or None
        if f == "listing_format" and v == "unknown":
            v = None
        elif f == "sold_date" and v is not None:
            v = v.isoformat()
        elif f == "sponsored":
            v = int(bool(v))
        out[f] = v
    return out


@dataclass
class _Row:
    """The mergeable part of a stored listing."""

    values: dict[str, Any]
    sources: dict[str, str]
    extraction: str
    confidence: float | None
    first_seen_at: str
    last_seen_at: str


def _truncates(title: str, stored: str | None) -> bool:
    """True when ``title`` is ``stored`` cut short (a clipped or cut-off card)."""
    short, full = normalize_title(title), normalize_title(stored)
    return len(short) < len(full) and full.startswith(short)


def _merge(stored: _Row, incoming: _Row) -> _Row:
    """Apply ``incoming`` on top of ``stored`` under the per-field precedence rules."""
    values, sources = dict(stored.values), dict(stored.sources)
    for f in MERGED_FIELDS:
        if f == "price_max":
            continue  # travels with price
        v = incoming.values.get(f)
        src = incoming.sources.get(f)
        if v is None or src is None:
            continue
        current = PRECEDENCE[sources[f]] if values.get(f) is not None and f in sources else -1
        if PRECEDENCE[src] < current:
            continue
        if f == "title" and PRECEDENCE[src] == current and _truncates(v, values[f]):
            continue
        values[f], sources[f] = v, src
        if f == "price":
            pmax = incoming.values.get("price_max")
            values["price_max"] = pmax
            if pmax is None:
                sources.pop("price_max", None)
            else:
                sources["price_max"] = src
    extraction, confidence = stored.extraction, stored.confidence
    if PRECEDENCE[incoming.extraction] > PRECEDENCE[stored.extraction]:
        extraction, confidence = incoming.extraction, incoming.confidence
    elif incoming.extraction == stored.extraction and incoming.confidence is not None:
        confidence = incoming.confidence
    return _Row(
        values=values,
        sources=sources,
        extraction=extraction,
        confidence=confidence,
        first_seen_at=min(stored.first_seen_at, incoming.first_seen_at),
        last_seen_at=max(stored.last_seen_at, incoming.last_seen_at),
    )


def _content(row: _Row) -> tuple[Any, ...]:
    return row.values, row.extraction, row.confidence


# How well two normalised titles agree, best first (see the module docstring).
TITLE_EQUAL, TITLE_TRUNCATED, TITLE_MISREAD, TITLE_TRUNCATED_MISREAD = 4, 3, 2, 1


def _close(a: str, b: str, threshold: float) -> bool:
    m = SequenceMatcher(None, a, b, autojunk=False)
    # The quick upper bounds skip the quadratic ratio() for clearly different titles.
    return m.real_quick_ratio() >= threshold and m.quick_ratio() >= threshold and m.ratio() >= threshold


def _title_tier(a: str, b: str, *, fuzzy: bool = True) -> int:
    """How well two normalised titles agree: one of the TITLE_* tiers, or 0 = different.

    ``fuzzy=False`` checks only the cheap exact tiers (equal, truncated).
    """
    if not a or not b:
        return 0
    if a == b:
        return TITLE_EQUAL
    short, long_ = sorted((a, b), key=lambda s: (len(s), s))
    if len(short) >= 20 and long_.startswith(short):
        return TITLE_TRUNCATED
    if not fuzzy:
        return 0
    if len(short) >= 12 and _close(short, long_, 0.9):
        return TITLE_MISREAD
    if len(short) >= 20 and _close(short, long_[: len(short)], 0.95):
        return TITLE_TRUNCATED_MISREAD
    return 0


def _shared_keys(price_a: float | None, date_a: str | None, price_b: float | None, date_b: str | None) -> int:
    """How many of price and sold date both readings show and agree on; -1 if one disagrees."""
    shared = 0
    for x, y in ((price_a, price_b), (date_a, date_b)):
        if x is None or y is None:
            continue
        if x != y:
            return -1
        shared += 1
    return shared


Score = tuple[int, int]  # (title tier, shared keys): how well a stored row fits a reading
_Key = tuple[str, "float | None", "str | None"]  # (title_norm, price, sold_date) of a stored row


def _best_fit(scored: list[tuple[Score, str, _Key]]) -> tuple[str | None, Score]:
    """The best-fitting id and its score; the id is None when nothing fits or the fit is ambiguous.

    A tie is ambiguous unless the tied rows are indistinguishable (same title,
    price and date: e.g. a seller's identical items sold the same day). Then
    any of them is equally right, and the lowest id is taken so the outcome
    does not depend on the order things were stored in.
    """
    if not scored:
        return None, (0, 0)
    best = max(s for s, _, _ in scored)
    tied = [(i, k) for s, i, k in scored if s == best]
    if len({k for _, k in tied}) > 1:
        return None, best
    return min(i for i, _ in tied), best


# --- statistics ----------------------------------------------------------------------


def _percentiles(values: list[float]) -> tuple[float, float, float, float, float]:
    """(p10, p25, median, p75, p90) using ``statistics.quantiles(method="inclusive")``."""
    if len(values) == 1:
        return (values[0],) * 5  # quantiles() needs two points before Python 3.13
    q = statistics.quantiles(values, n=100, method="inclusive")
    return q[9], q[24], statistics.median(values), q[74], q[89]


def compute_price_stats(
    values: list[float],
    *,
    currency: str,
    sold_dates: list[date | None] | None = None,
    trim_outliers: bool = True,
    include_shipping: bool = False,
    shipping_unknown: list[bool] | None = None,
) -> PriceStats | None:
    """Stats over ``values`` (parallel lists). Outliers outside 1.5 x IQR are dropped
    only when ``trim_outliers`` and there are at least 8 values."""
    if not values:
        return None
    n = len(values)
    dates = sold_dates if sold_dates is not None else [None] * n
    unknown = shipping_unknown if shipping_unknown is not None else [False] * n
    keep = list(range(n))
    if trim_outliers and n >= 8:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        keep = [i for i in keep if lo <= values[i] <= hi]
    kept = sorted(values[i] for i in keep)
    kept_dates = [dates[i] for i in keep if dates[i] is not None]
    p10, p25, median, p75, p90 = _percentiles(kept)
    return PriceStats(
        currency=currency,
        count=len(kept),
        outliers_removed=n - len(kept),
        min=round(kept[0], 2),
        max=round(kept[-1], 2),
        mean=round(statistics.fmean(kept), 2),
        median=round(median, 2),
        p10=round(p10, 2),
        p25=round(p25, 2),
        p75=round(p75, 2),
        p90=round(p90, 2),
        first_sold=min(kept_dates) if kept_dates else None,
        last_sold=max(kept_dates) if kept_dates else None,
        include_shipping=include_shipping,
        shipping_unknown=sum(1 for i in keep if unknown[i]),
    )


# --- the database -----------------------------------------------------------------------


@dataclass
class _Found:
    listing: Listing
    has_real_id: bool
    first_seen_at: str
    last_seen_at: str


class Database:
    """SQLite store of searches, listings and observations. Use as a context manager."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:" and not self.path.startswith("file:"):
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # Autocommit mode: transactions are explicit (see _tx), so a crash mid-page
        # never leaves half a page of listings behind.
        self._conn = sqlite3.connect(self.path, isolation_level=None, uri=self.path.startswith("file:"))
        self._tx_depth = 0
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 10000")
            self._check_not_newer()  # before WAL: a refused file is left exactly as it was
            self._enable_wal()
            self._migrate()
        except BaseException:
            self.close()
            raise

    # -- lifecycle -------------------------------------------------------------------

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, for ad-hoc SQL."""
        return self._conn

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _check_not_newer(self) -> None:
        version = self.schema_version
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} has schema version {version}, newer than this ebay-sold ({SCHEMA_VERSION}); upgrade ebay-sold"
            )

    def _enable_wal(self, timeout: float = 10.0) -> None:
        # Switching a file to WAL needs it to itself for a moment, and SQLite
        # reports "locked" at once instead of waiting for busy_timeout.
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or time.monotonic() > deadline:
                    raise
                time.sleep(0.02)

    def _migrate(self) -> None:
        while self.schema_version < SCHEMA_VERSION:
            with self._tx() as c:
                # Read again under the write lock: another process may have just migrated.
                version = self.schema_version
                if version >= SCHEMA_VERSION:
                    self._check_not_newer()  # a newer ebay-sold may have been the one
                    break
                (log.debug if version == 0 else log.info)("migrating %s to schema version %d", self.path, version + 1)
                for statement in _sql_statements(MIGRATIONS[version]):
                    c.execute(statement)
                c.execute(f"PRAGMA user_version = {version + 1}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        if self._tx_depth:
            self._tx_depth += 1
            try:
                yield self._conn
            finally:
                self._tx_depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._tx_depth = 1
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except BaseException:
            # Some errors (disk full, I/O) end the transaction inside SQLite already;
            # a second ROLLBACK would then replace the real error with "no transaction".
            if self._conn.in_transaction:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    log.warning("rollback failed after an error", exc_info=True)
            raise
        finally:
            self._tx_depth = 0

    # -- writing -----------------------------------------------------------------------

    def record_search(self, query: SearchQuery, url: str) -> int:
        """Store one run of a search; observations from its pages point here."""
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO searches (keywords, keywords_norm, site, url, query_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (query.keywords, normalize_keywords(query.keywords), _site(query.site), url,
                 query.model_dump_json(), _iso_utc()),
            )
            return int(cur.lastrowid)

    def save_page(self, search_id: int, result: PageResult) -> UpsertStats:
        """Store a parsed results page: its listings, one observation each, and the page itself."""
        with self._tx() as c:
            stats = self.upsert_listings(result.listings, search_id=search_id, page=result.page,
                                         observed_at=result.fetched_at)
            c.execute(
                "INSERT OR REPLACE INTO search_pages (search_id, page, url, fetched_at, total_results, listings,"
                " html_path, screenshot_path, from_cache, has_next_page) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (search_id, result.page, result.url, _iso_utc(result.fetched_at), result.total_results,
                 len(result.listings), result.html_path, result.screenshot_path, int(result.from_cache),
                 None if result.has_next_page is None else int(result.has_next_page)),
            )
        return stats

    def upsert_listings(
        self,
        listings: Iterable[Listing],
        *,
        search_id: int | None = None,
        page: int | None = None,
        observed_at: datetime | None = None,
    ) -> UpsertStats:
        """Insert or merge listings; with ``search_id``, also record that the search showed them."""
        seen_at = _iso_utc(observed_at)
        written: list[list[str]] = []  # [site, item_id, outcome] per listing
        first: dict[tuple[str, str], int] = {}  # row -> index of the first listing written to it
        with self._tx():
            for item in listings:
                site, item_id, outcome = self._upsert_one(item, seen_at)
                first.setdefault((site, item_id), len(written))
                written.append([site, item_id, outcome])
                if search_id is not None:
                    self._observe(search_id, site, item_id, item, page, seen_at)
            # Earlier readings are folded in only once the whole batch is stored, so
            # one that fits several listings of this page is seen to be ambiguous.
            created = {(s, i) for s, i, o in written if o == "new"}
            gone: set[tuple[str, str]] = set()
            for key, index in first.items():
                if key in gone:
                    continue
                absorbed = {(key[0], other) for other in self._absorb_other_readings(*key)}
                gone |= absorbed
                if absorbed - created:  # the sale was already stored, under another reading's id
                    written[index][2] = "updated"
            stats = UpsertStats()
            for s, i, outcome in written:
                if (s, i) in gone and outcome == "new":
                    outcome = "updated"  # another reading of a sale this batch stored under another id
                setattr(stats, outcome, getattr(stats, outcome) + 1)
        return stats

    def _upsert_one(self, item: Listing, seen_at: str) -> tuple[str, str, str]:
        site = _site(item.site)
        values = _db_values(item)
        incoming = _Row(
            values=values,
            sources={f: item.extraction for f, v in values.items() if v is not None},
            extraction=item.extraction,
            confidence=item.confidence,
            first_seen_at=seen_at,
            last_seen_at=seen_at,
        )
        given = (item.item_id or "").strip() or None
        real_id = given if given and not is_synthetic_id(given) else item_id_from_url(item.url)
        if real_id:
            target = real_id
        else:
            synth_id = given or synthetic_item_id(site, values["title"], values["price"], values["sold_date"])
            target = self._resolve_alias(site, synth_id)
            if target is None and not self._exists(site, synth_id):
                target = self._match_reading(site, synth_id, values)
                if target:
                    self._add_alias(site, synth_id, target)
            target = target or synth_id
        return site, target, self._write(site, target, not is_synthetic_id(target), incoming)

    def _write(self, site: str, item_id: str, has_real_id: bool, incoming: _Row) -> str:
        stored = self._load(site, item_id)
        if stored is None:
            self._insert(site, item_id, has_real_id, incoming)
            return "new"
        merged = _merge(stored, incoming)
        if merged != stored:
            self._update(site, item_id, merged)
        # Only a change of content counts as an update, not merely being seen again.
        return "updated" if _content(merged) != _content(stored) else "unchanged"

    def _exists(self, site: str, item_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM listings WHERE site = ? AND item_id = ?", (site, item_id)
        ).fetchone() is not None

    def _load(self, site: str, item_id: str) -> _Row | None:
        r = self._conn.execute("SELECT * FROM listings WHERE site = ? AND item_id = ?", (site, item_id)).fetchone()
        if r is None:
            return None
        return _Row(
            values={f: r[f] for f in MERGED_FIELDS},
            sources=json.loads(r["field_sources"]),
            extraction=r["extraction"],
            confidence=r["confidence"],
            first_seen_at=r["first_seen_at"],
            last_seen_at=r["last_seen_at"],
        )

    def _insert(self, site: str, item_id: str, has_real_id: bool, row: _Row) -> None:
        values = dict(row.values)
        values["title"] = values["title"] or ""
        cols = ["site", "item_id", "has_real_id", "title_norm", *MERGED_FIELDS,
                "extraction", "confidence", "field_sources", "first_seen_at", "last_seen_at"]
        params = [site, item_id, int(has_real_id), normalize_title(values["title"]), *(values[f] for f in MERGED_FIELDS),
                  row.extraction, row.confidence, json.dumps(row.sources, sort_keys=True),
                  row.first_seen_at, row.last_seen_at]
        self._conn.execute(
            f"INSERT INTO listings ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", params
        )

    def _update(self, site: str, item_id: str, row: _Row) -> None:
        values = dict(row.values)
        values["title"] = values["title"] or ""
        sets = [f"{f} = ?" for f in MERGED_FIELDS]
        params = [*(values[f] for f in MERGED_FIELDS), normalize_title(values["title"]), row.extraction,
                  row.confidence, json.dumps(row.sources, sort_keys=True), row.first_seen_at, row.last_seen_at,
                  site, item_id]
        self._conn.execute(
            f"UPDATE listings SET {', '.join(sets)}, title_norm = ?, extraction = ?, confidence = ?, field_sources = ?,"
            " first_seen_at = ?, last_seen_at = ? WHERE site = ? AND item_id = ?",
            params,
        )

    def _observe(self, search_id: int, site: str, item_id: str, item: Listing, page: int | None, seen_at: str) -> None:
        # Seeing the same listing on the same page again replaces the observation,
        # unless the earlier one came from a more reliable extractor.
        self._conn.execute(
            "INSERT INTO observations (search_id, site, item_id, page, position, matches_query, sponsored, extraction,"
            " observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (search_id, site, item_id, page) DO UPDATE SET"
            " position = COALESCE(excluded.position, observations.position),"
            " matches_query = excluded.matches_query, sponsored = excluded.sponsored,"
            " extraction = excluded.extraction, observed_at = excluded.observed_at"
            f" WHERE {_RANK_SQL.format('excluded')} >= {_RANK_SQL.format('observations')}",
            (search_id, site, item_id, page or 0, item.position, int(item.matches_query), int(item.sponsored),
             item.extraction, seen_at),
        )

    # -- synthetic <-> real matching ------------------------------------------------------

    def _like(
        self,
        site: str,
        title_norm: str,
        price: float | None,
        sold_date: str | None,
        *,
        real: bool,
        min_tier: int = TITLE_TRUNCATED_MISREAD,
        exclude: Iterable[str] = (),
    ) -> list[tuple[Score, str, _Key]]:
        """Scored real-id (or synthetic) rows that could be the sale this reading shows."""
        if price is None and sold_date is None:
            return []  # too little to tell sales of the same title apart
        # Each branch is an index lookup; the rest of the compatibility test is in Python.
        branches = []
        if price is not None:
            branches.append("SELECT item_id, title_norm, price, sold_date FROM listings"
                            " WHERE site = :site AND has_real_id = :real AND price = :price"
                            " AND (sold_date IS NULL OR :sold IS NULL OR sold_date = :sold)")
        if sold_date is not None:
            branches.append("SELECT item_id, title_norm, price, sold_date FROM listings"
                            " WHERE site = :site AND has_real_id = :real AND sold_date = :sold"
                            " AND (price IS NULL OR :price IS NULL OR price = :price)")
        rows = self._conn.execute(" UNION ".join(branches),
                                  {"site": site, "real": int(real), "price": price, "sold": sold_date})
        skip = set(exclude)
        found = []
        for r in rows:
            shared = _shared_keys(price, sold_date, r["price"], r["sold_date"])
            if shared > 0 and r["item_id"] not in skip:
                found.append((r, shared))
        tiers = [_title_tier(title_norm, r["title_norm"], fuzzy=False) for r, _ in found]
        # Fuzzy tiers rank below the exact ones, so they only matter when no title is
        # an exact fit; skipping them keeps a lookup among many same-day sales cheap.
        if min_tier < TITLE_TRUNCATED and not any(tiers):
            tiers = [_title_tier(title_norm, r["title_norm"]) for r, _ in found]
        return [((tier, shared), r["item_id"], (r["title_norm"], r["price"], r["sold_date"]))
                for (r, shared), tier in zip(found, tiers) if tier >= min_tier]

    def _match_reading(self, site: str, synth_id: str, values: dict[str, Any]) -> str | None:
        """The stored row an id-less listing is another reading of: the real-id row it
        fits best, else the synthetic row (exact title tiers only) it fits best."""
        title_norm, price, sold = normalize_title(values["title"]), values["price"], values["sold_date"]
        real, score = _best_fit(self._like(site, title_norm, price, sold, real=True))
        if real is not None:
            return real
        if score[0]:
            log.debug("%s: several real listings fit equally well; not merging it into one", synth_id)
        synthetic, _ = _best_fit(self._like(site, title_norm, price, sold, real=False, min_tier=TITLE_TRUNCATED,
                                            exclude=(synth_id,)))
        return synthetic

    def _absorb_other_readings(self, site: str, item_id: str) -> list[str]:
        """Fold synthetic rows that are other readings of the sale at ``item_id`` into it.

        A synthetic row is absorbed only when this row is the one it fits best
        (see ``_best_fit``), so an ambiguous reading is never pinned to one sale.
        Returns the absorbed (now deleted) ids.
        """
        absorbed = []
        while (other := self._next_absorbable(site, item_id)) is not None:
            self._absorb(site, other, item_id)
            absorbed.append(other)
        return absorbed

    def _next_absorbable(self, site: str, item_id: str) -> str | None:
        me = self._conn.execute(
            "SELECT has_real_id, title_norm, price, sold_date FROM listings WHERE site = ? AND item_id = ?",
            (site, item_id),
        ).fetchone()
        if me is None:
            return None
        real = bool(me["has_real_id"])
        # Between two synthetic rows neither title is ground truth: exact tiers only.
        min_tier = TITLE_TRUNCATED_MISREAD if real else TITLE_TRUNCATED
        candidates = self._like(site, me["title_norm"], me["price"], me["sold_date"], real=False, min_tier=min_tier,
                                exclude=(item_id,))
        for _, other, key in sorted(candidates, reverse=True):
            # Every row the candidate fits (this one included, the scores are symmetric).
            fits = self._like(site, *key, real=True)
            if not real:
                fits += self._like(site, *key, real=False, min_tier=TITLE_TRUNCATED, exclude=(other,))
            if _best_fit(fits)[0] == item_id:
                return other
        return None

    def _absorb(self, site: str, other: str, item_id: str) -> None:
        """Fold synthetic row ``other`` into row ``item_id`` (the same sale), then delete it."""
        stored, extra = self._load(site, item_id), self._load(site, other)
        assert stored is not None and extra is not None
        # The row just written is the newer reading: it goes on top, under the precedence rules.
        self._update(site, item_id, _merge(extra, stored))
        # This row's own observation wins where both were seen by the same search page.
        self._conn.execute("UPDATE OR IGNORE observations SET item_id = ? WHERE site = ? AND item_id = ?",
                           (item_id, site, other))
        self._conn.execute("UPDATE listing_aliases SET item_id = ? WHERE site = ? AND item_id = ?",
                           (item_id, site, other))
        self._conn.execute("DELETE FROM listings WHERE site = ? AND item_id = ?", (site, other))
        self._add_alias(site, other, item_id)
        log.debug("merged %s into %s", other, item_id)

    def _add_alias(self, site: str, alias_id: str, item_id: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO listing_aliases (site, alias_id, item_id) VALUES (?, ?, ?)", (site, alias_id, item_id)
        )

    def _resolve_alias(self, site: str, item_id: str) -> str | None:
        r = self._conn.execute(
            "SELECT item_id FROM listing_aliases WHERE site = ? AND alias_id = ?", (site, item_id)
        ).fetchone()
        return r["item_id"] if r else None

    # -- reading -----------------------------------------------------------------------

    def get_listing(self, item_id: str, site: str = "www.ebay.com") -> Listing | None:
        """One stored listing by real or synthetic id (a merged synthetic id resolves to its real row)."""
        site = _site(site)
        item_id = self._resolve_alias(site, item_id) or item_id
        found = self._query(site=site, item_id=item_id, exact_matches_only=False, include_sponsored=True)
        return found[0].listing if found else None

    def query_listings(
        self,
        *,
        keywords: str | None = None,
        site: str | None = None,
        since: date | str | None = None,
        until: date | str | None = None,
        condition: str | None = None,
        currency: str | None = None,
        exact_matches_only: bool = True,
        include_sponsored: bool = False,
        limit: int | None = None,
    ) -> list[Listing]:
        """Stored listings matching the filters, newest sold date first (undated last).

        ``matches_query`` and ``position`` on the returned listings come from the
        observations in scope (the ``keywords`` searches, or all of them).
        """
        return [f.listing for f in self._query(
            keywords=keywords, site=site, since=since, until=until, condition=condition, currency=currency,
            exact_matches_only=exact_matches_only, include_sponsored=include_sponsored, limit=limit,
        )]

    def _query(self, *, limit: int | None = None, **filters: Any) -> list[_Found]:
        sql, params = self._select_sql(**filters)
        sql += " ORDER BY l.sold_date IS NULL, l.sold_date DESC, l.last_seen_at DESC, l.item_id"
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = int(limit)
        return [self._to_found(r) for r in self._conn.execute(sql, params)]

    def _select_sql(
        self,
        *,
        keywords: str | None = None,
        site: str | None = None,
        since: date | str | None = None,
        until: date | str | None = None,
        condition: str | None = None,
        currency: str | None = None,
        exact_matches_only: bool = True,
        include_sponsored: bool = False,
        item_id: str | None = None,
        priced_only: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        scope = "o.site = l.site AND o.item_id = l.item_id"
        if keywords is not None and keywords.strip():
            scope += " AND o.search_id IN (SELECT id FROM searches WHERE keywords_norm = :kw)"
            params["kw"] = normalize_keywords(keywords)
            keyed = True
        else:
            keyed = False
        where = ["1 = 1"]
        if keyed:
            match = " AND o.matches_query = 1" if exact_matches_only else ""
            where.append(f"EXISTS (SELECT 1 FROM observations o WHERE {scope}{match})")
        elif exact_matches_only:
            where.append(f"(EXISTS (SELECT 1 FROM observations o WHERE {scope} AND o.matches_query = 1)"
                         f" OR NOT EXISTS (SELECT 1 FROM observations o WHERE {scope}))")
        if site:
            where.append("l.site = :site")
            params["site"] = _site(site)
        if item_id is not None:
            where.append("l.item_id = :item_id")
            params["item_id"] = item_id
        if priced_only:
            where.append("l.price IS NOT NULL")
        since_d, until_d = _as_date(since), _as_date(until)  # "" means no bound, like None
        if since_d is not None:
            where.append("l.sold_date >= :since")
            params["since"] = since_d.isoformat()
        if until_d is not None:
            where.append("l.sold_date <= :until")
            params["until"] = until_d.isoformat()
        if condition:
            where.append("l.condition = :condition COLLATE NOCASE")
            params["condition"] = clean_text(condition)
        if currency:
            if currency.strip().upper() == UNKNOWN_CURRENCY:
                where.append("l.currency IS NULL")
            else:
                where.append("l.currency = :currency COLLATE NOCASE")
                params["currency"] = currency.strip()
        if not include_sponsored:
            where.append("l.sponsored = 0")
        sql = (
            "SELECT l.*,"
            f" (SELECT MAX(o.matches_query) FROM observations o WHERE {scope}) AS obs_match,"
            f" (SELECT o.position FROM observations o WHERE {scope} ORDER BY o.observed_at DESC, o.id DESC LIMIT 1)"
            " AS obs_position"
            " FROM listings l WHERE " + " AND ".join(where)
        )
        return sql, params

    @staticmethod
    def _to_found(r: sqlite3.Row) -> _Found:
        data: dict[str, Any] = {f: r[f] for f in MERGED_FIELDS}
        data["listing_format"] = data["listing_format"] or "unknown"
        data["sponsored"] = bool(data["sponsored"])
        listing = Listing(
            item_id=r["item_id"],
            site=r["site"],
            matches_query=True if r["obs_match"] is None else bool(r["obs_match"]),
            position=r["obs_position"],
            extraction=r["extraction"],
            confidence=r["confidence"],
            **data,
        )
        return _Found(listing, bool(r["has_real_id"]), r["first_seen_at"], r["last_seen_at"])

    def searches(self) -> list[dict[str, Any]]:
        """Every recorded search run with page and listing counts, oldest first."""
        rows = self._conn.execute(
            """
            SELECT s.id, s.keywords, s.site, s.url, s.created_at, s.query_json,
                   (SELECT COUNT(*) FROM search_pages p WHERE p.search_id = s.id) AS pages,
                   (SELECT MAX(p.total_results) FROM search_pages p WHERE p.search_id = s.id) AS total_results,
                   (SELECT COUNT(DISTINCT o.site || '|' || o.item_id) FROM observations o WHERE o.search_id = s.id)
                       AS listings,
                   (SELECT COUNT(DISTINCT o.site || '|' || o.item_id) FROM observations o
                     WHERE o.search_id = s.id AND o.matches_query = 1) AS exact_matches
            FROM searches s ORDER BY s.id
            """
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["query"] = json.loads(d.pop("query_json"))
            out.append(d)
        return out

    def summary(self) -> dict[str, Any]:
        """Row counts and coverage, for a quick look at what the database holds."""
        c = self._conn

        def one(sql: str) -> Any:
            return c.execute(sql).fetchone()[0]

        return {
            "schema_version": self.schema_version,
            "listings": one("SELECT COUNT(*) FROM listings"),
            "real_ids": one("SELECT COUNT(*) FROM listings WHERE has_real_id = 1"),
            "synthetic_ids": one("SELECT COUNT(*) FROM listings WHERE has_real_id = 0"),
            "searches": one("SELECT COUNT(*) FROM searches"),
            "observations": one("SELECT COUNT(*) FROM observations"),
            "by_extraction": dict(c.execute("SELECT extraction, COUNT(*) FROM listings GROUP BY 1 ORDER BY 1").fetchall()),
            "by_currency": dict(c.execute(
                f"SELECT COALESCE(currency, '{UNKNOWN_CURRENCY}'), COUNT(*) FROM listings GROUP BY 1 ORDER BY 1"
            ).fetchall()),
            "first_sold": one("SELECT MIN(sold_date) FROM listings"),
            "last_sold": one("SELECT MAX(sold_date) FROM listings"),
        }

    # -- statistics ----------------------------------------------------------------------

    def _priced(self, include_shipping: bool, **filters: Any) -> dict[str, list[tuple[float, date | None, bool]]]:
        groups: dict[str, list[tuple[float, date | None, bool]]] = defaultdict(list)
        for f in self._query(priced_only=True, **filters):
            item = f.listing
            if item.price is None:  # excluded by priced_only; kept for the type checker
                continue
            value, unknown = item.price, False
            if include_shipping:
                if item.shipping is None:
                    unknown = True
                else:
                    value = round(item.price + item.shipping, 2)
            groups[(item.currency or UNKNOWN_CURRENCY).upper()].append((value, item.sold_date, unknown))
        return groups

    def price_stats(
        self,
        *,
        keywords: str | None = None,
        site: str | None = None,
        since: date | str | None = None,
        until: date | str | None = None,
        condition: str | None = None,
        currency: str | None = None,
        exact_matches_only: bool = True,
        include_sponsored: bool = False,
        limit: int | None = None,
        include_shipping: bool = False,
        trim_outliers: bool = True,
    ) -> dict[str, PriceStats]:
        """Sold-price statistics per currency (largest sample first); ``{}`` when nothing matches.

        Takes the filters of ``query_listings``; ``limit`` keeps the N most recently
        sold priced listings (across currencies) before outliers are trimmed.
        """
        groups = self._priced(
            include_shipping, keywords=keywords, site=site, since=since, until=until, condition=condition,
            currency=currency, exact_matches_only=exact_matches_only, include_sponsored=include_sponsored,
            limit=limit,
        )
        out: dict[str, PriceStats] = {}
        for cur, rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            stats = compute_price_stats(
                [v for v, _, _ in rows], currency=cur, sold_dates=[d for _, d, _ in rows],
                trim_outliers=trim_outliers, include_shipping=include_shipping,
                shipping_unknown=[u for _, _, u in rows],
            )
            if stats is not None:
                out[cur] = stats
        return out

    def price_series(
        self,
        *,
        period: Literal["day", "week", "month"] = "month",
        include_shipping: bool = False,
        trim_outliers: bool = True,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Price stats per currency per sold-date period ("2026-04", "2026-W17", "2026-04-25"), oldest first.

        Takes the same filters as ``query_listings`` (``limit`` as in ``price_stats``).
        Undated listings are left out.
        """
        if period not in ("day", "week", "month"):
            raise ValueError(f"unknown period {period!r}; expected 'day', 'week' or 'month'")
        buckets: dict[tuple[str, str], list[tuple[float, date | None, bool]]] = defaultdict(list)
        for cur, rows in self._priced(include_shipping, **filters).items():
            for v, d, u in rows:
                if d is None:
                    continue
                if period == "day":
                    key = d.isoformat()
                elif period == "week":
                    iso = d.isocalendar()
                    key = f"{iso[0]}-W{iso[1]:02d}"
                else:
                    key = f"{d.year}-{d.month:02d}"
                buckets[(key, cur)].append((v, d, u))
        out = []
        for (key, cur), rows in sorted(buckets.items()):
            stats = compute_price_stats(
                [v for v, _, _ in rows], currency=cur, sold_dates=[d for _, d, _ in rows],
                trim_outliers=trim_outliers, include_shipping=include_shipping,
                shipping_unknown=[u for _, _, u in rows],
            )
            out.append({"period": key, **stats.model_dump()})  # type: ignore[union-attr]
        return out

    # -- export ------------------------------------------------------------------------

    @staticmethod
    def export_columns() -> list[str]:
        return [*Listing.model_fields, *EXPORT_EXTRA_FIELDS]

    def _export_records(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        records = []
        for f in self._query(**filters):
            rec = f.listing.model_dump(mode="json")
            rec["total_price"] = f.listing.total_price
            rec["has_real_id"] = f.has_real_id
            rec["first_seen_at"] = f.first_seen_at
            rec["last_seen_at"] = f.last_seen_at
            records.append({k: rec[k] for k in self.export_columns()})
        return records

    def export_csv(self, path: str | Path, **filters: Any) -> int:
        """Write matching listings to CSV (empty cell = unknown, dates ISO). Returns the row count."""
        records = self._export_records(filters)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.export_columns())
            writer.writeheader()
            for rec in records:
                writer.writerow({k: "" if v is None else str(v).lower() if isinstance(v, bool) else v
                                 for k, v in rec.items()})
        return len(records)

    def export_json(self, path: str | Path, **filters: Any) -> int:
        """Write matching listings to a JSON array (null = unknown, dates ISO). Returns the row count."""
        records = self._export_records(filters)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return len(records)


__all__ = [
    "MIGRATIONS",
    "PRECEDENCE",
    "SCHEMA_VERSION",
    "Database",
    "PriceStats",
    "UpsertStats",
    "compute_price_stats",
    "is_synthetic_id",
    "normalize_keywords",
    "normalize_title",
    "synthetic_item_id",
]
