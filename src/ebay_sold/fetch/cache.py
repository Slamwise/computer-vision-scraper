"""On-disk cache of fetched result pages, so the same page is never fetched twice.

Every request to eBay is a chance to be challenged, so a page that was already
fetched is served from disk: re-running a scrape, re-parsing after a parser fix,
or rendering screenshots for the vision path costs no requests at all.

Two URLs that would return the same results share an entry. ``cache_key``
lowercases the host, sorts the query and drops parameters that only track how
you got there (``_trksid``, ``_from``, ``_odkw``, the ``mk*`` marketing
parameters...), but keeps every parameter that changes the results (keywords,
filters, sort, page, page size).

Each entry is a plain ``.html`` file (openable in a browser, importable with
``ebay-sold import-html``) next to a ``.json`` file holding the metadata and a
SHA-1 of the HTML. Both are written atomically, the ``.json`` last, so a crash
mid-write never leaves an entry that looks complete; an entry whose HTML no
longer matches its SHA-1 (truncated, edited) is a miss, like any unreadable one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel

log = logging.getLogger(__name__)

# Parameters that record navigation history or campaign tracking, never what is shown.
TRACKING_PARAMS = frozenset({
    "_trksid", "_trkparms", "_from", "_odkw", "_osacat", "itmmeta", "hash",
    "campid", "customid", "toolid",
})
# eBay's marketing / affiliate parameters all start with "mk" (mkevt, mkcid, mkrid, mksid, mktype...).
_TRACKING_PREFIXES = ("mk",)

# Only a plain page number goes into a file name; anything else is left to the hash.
_PAGE_NO_RE = re.compile(r"\d{1,6}")

_DEFAULT_PORTS = {("http", 80), ("https", 443)}


class CachedPage(BaseModel):
    url: str
    final_url: str
    status: int | None
    html: str
    fetched_at: datetime
    path: Path  # the .html file


def cache_key(url: str) -> str:
    """Normalised URL: same key for URLs that return the same results."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    netloc = host if parts.port is None or (scheme, parts.port) in _DEFAULT_PORTS else f"{host}:{parts.port}"
    params = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k))
    return urlunsplit((scheme, netloc, parts.path or "/", urlencode(params), ""))


def _is_tracking(param: str) -> bool:
    name = param.lower()
    return name in TRACKING_PARAMS or name.startswith(_TRACKING_PREFIXES)


def page_stem(url: str) -> str:
    """Readable, collision-safe file stem for a page: ``hot-wheels-r34-p2-1a2b3c4d5e6f``."""
    key = cache_key(url)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    query = dict(parse_qsl(urlsplit(key).query))
    words = query.get("_nkw") or urlsplit(key).path
    slug = re.sub(r"[^a-z0-9]+", "-", words.lower()).strip("-")[:48].strip("-") or "page"
    page_no = query.get("_pgn", "1")
    if page_no != "1" and _PAGE_NO_RE.fullmatch(page_no):
        slug += f"-p{page_no}"
    return f"{slug}-{digest}"


class HtmlCache:
    """``<dir>/<stem>.html`` + ``<dir>/<stem>.json``; entries older than ``ttl_hours`` are ignored."""

    def __init__(self, dir: str | Path, ttl_hours: float = 24.0, *,
                 now: Callable[[], datetime] | None = None):
        self.dir = Path(dir)
        self.ttl_hours = ttl_hours  # <= 0: entries never expire
        self._now = now or (lambda: datetime.now(timezone.utc))

    def key(self, url: str) -> str:
        return cache_key(url)

    def path_for(self, url: str) -> Path:
        return self.dir / f"{page_stem(url)}.html"

    def get(self, url: str) -> CachedPage | None:
        path = self.path_for(url)
        meta_path = path.with_suffix(".json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError(f"metadata is a {type(meta).__name__}, not an object")
            if meta.get("key") != self.key(url):
                return None
            fetched_at = datetime.fromisoformat(meta["fetched_at"])
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            if self.ttl_hours > 0 and self._now() - fetched_at > timedelta(hours=self.ttl_hours):
                log.debug("cache entry expired: %s", path.name)
                return None
            data = path.read_bytes()
            if meta.get("sha1") is not None and hashlib.sha1(data).hexdigest() != meta["sha1"]:
                raise ValueError("the HTML does not match its recorded SHA-1 (truncated or edited)")
            html = data.decode("utf-8")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("ignoring unreadable cache entry %s: %s", meta_path, exc)
            return None
        return CachedPage(
            url=meta.get("url", url),
            final_url=meta.get("final_url") or url,
            status=meta.get("status"),
            html=html,
            fetched_at=fetched_at,
            path=path,
        )

    def put(self, url: str, *, final_url: str, status: int | None, html: str) -> Path:
        """Store a page; returns the path of the ``.html`` file."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(url)
        data = html.encode("utf-8", errors="replace")
        meta = {
            "url": url,
            "key": self.key(url),
            "final_url": final_url,
            "status": status,
            "fetched_at": self._now().isoformat(),
            "bytes": len(data),
            "sha1": hashlib.sha1(data).hexdigest(),
        }
        _atomic_write(path, data)
        _atomic_write(path.with_suffix(".json"), json.dumps(meta, indent=2).encode("utf-8"))
        return path


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
