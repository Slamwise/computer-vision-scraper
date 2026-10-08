"""End-to-end flows: saved pages and the CLI, and scraping against a local stand-in for eBay."""

from __future__ import annotations

import functools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import pytest

from ebay_sold import pipeline
from ebay_sold.cli import main
from ebay_sold.config import BrowserSettings, PacingSettings, Settings
from ebay_sold.db import Database
from ebay_sold.models import Listing, SearchQuery

from conftest import EBAY_FIXTURES, FIXTURES, fixture_names

HOT_WHEELS = EBAY_FIXTURES / "sold_2026-04-26_hot-wheels-r34-zamac.html.gz"


# --- saved pages and the CLI (no browser) -------------------------------------


def test_import_html_then_stats_list_and_export(tmp_path, capsys):
    data = tmp_path / "data"
    files = [str(EBAY_FIXTURES / f"{name}.html.gz") for name in fixture_names()]
    assert main(["--data-dir", str(data), "import-html", *files]) == 0
    out = capsys.readouterr().out
    assert "Hot Wheels R34 Nissan Skyline GT-R ZAMAC: 70 listings from 1 page(s), 70 new" in out

    with Database(data / "ebay_sold.sqlite") as db:
        exact = db.query_listings(keywords="hot wheels r34 nissan skyline gt-r zamac")
        loose = db.query_listings(keywords="hot wheels r34 nissan skyline gt-r zamac", exact_matches_only=False)
    # 31 cards sit above eBay's "Results matching fewer words" divider
    assert len(exact) == 31 and len(loose) == 70

    assert main(["--data-dir", str(data), "stats", "Hot Wheels R34 Nissan Skyline GT-R ZAMAC", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)["USD"]
    assert stats["count"] + stats["outliers_removed"] <= 31
    assert stats["p25"] <= stats["median"] <= stats["p75"]

    assert main(["--data-dir", str(data), "stats", "Hot Wheels R34 Nissan Skyline GT-R ZAMAC"]) == 0
    assert "median" in capsys.readouterr().out

    assert main(["--data-dir", str(data), "stats", "ta1 adapter", "--by", "month"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[:3] == ["period", "cur", "sold"] and len(lines) >= 3
    assert all(line.split()[0] < later.split()[0] for line, later in zip(lines[1:], lines[2:]))  # oldest first

    assert main(["--data-dir", str(data), "list", "ta1 adapter", "--limit", "3"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3

    out_csv = tmp_path / "sold.csv"
    assert main(["--data-dir", str(data), "export", "--format", "csv", "--out", str(out_csv), "--all-matches"]) == 0
    assert len(out_csv.read_text().strip().splitlines()) == 1 + 344  # header + every real card


def test_import_html_is_idempotent(tmp_path):
    with Database(tmp_path / "db.sqlite") as db:
        first = pipeline.import_html([HOT_WHEELS], db=db)
        again = pipeline.import_html([HOT_WHEELS], db=db)
    assert first[0].new == 70 and again[0].new == 0


def test_cli_reports_missing_files(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "import-html", str(tmp_path / "nope.html")]) == 2
    assert "not found" in capsys.readouterr().err


async def test_extract_prefers_dom_and_falls_back_to_vision(tmp_path, monkeypatch):
    dom = [Listing(title="from dom", price=1.0)]
    seen: list[list] = []

    def fake_vision(tiles, *, settings, site):
        seen.append(tiles)
        return [Listing(title="from vision", price=2.0, extraction="vision")]

    monkeypatch.setattr(pipeline, "vision_extract_tiles", fake_vision)
    settings = Settings(data_dir=tmp_path)
    tiles = [(tmp_path / "t.png", None)]
    assert await pipeline._extract(dom, "", settings=settings, site="www.ebay.com", vision="fallback",
                                   llm="off", tiles=tiles) == (dom, "dom")
    assert not seen
    listings, how = await pipeline._extract([], "", settings=settings, site="www.ebay.com", vision="fallback",
                                            llm="off", tiles=tiles)
    assert (how, listings[0].title) == ("vision", "from vision")
    listings, how = await pipeline._extract(dom, "", settings=settings, site="www.ebay.com", vision="always",
                                            llm="off", tiles=tiles)
    assert how == "vision"
    assert await pipeline._extract([], "", settings=settings, site="www.ebay.com", vision="off",
                                   llm="off", tiles=tiles) == ([], "dom")


async def test_llm_captcha_stops_like_a_block(tmp_path, monkeypatch):
    from ebay_sold import llm

    class FakeExtractor:
        def __init__(self, *_a, **_k):
            pass

        def extract_tiles(self, tiles, *, site):
            raise llm.LLMExtractionError("captcha on screen", reason="captcha")

    monkeypatch.setattr(llm, "ClaudeExtractor", FakeExtractor)
    with pytest.raises(pipeline._ChallengeSeen):
        pipeline.llm_extract_tiles([(tmp_path / "t.png", None)], settings=Settings(data_dir=tmp_path),
                                   site="www.ebay.com")


def test_vision_fallback_without_a_model_is_a_warning(tmp_path, caplog):
    assert pipeline.vision_extract_tiles([], settings=Settings(data_dir=tmp_path), site="www.ebay.com") == []
    assert "no model" in caplog.text


# --- scraping against a local server (browser) ----------------------------------


def _results_page(keywords: str, start: int, count: int, *, has_next: bool, card_class: str = "s-card") -> bytes:
    cards = "".join(
        f"""<li class="{card_class}" data-listingid="3000000{start + i:05d}">
  <div class="s-card__caption"><span class="su-styled-text">Sold  Apr {1 + (start + i) % 28}, 2026</span></div>
  <div class="s-card__title"><span class="su-styled-text">{keywords} item {start + i}</span></div>
  <div class="s-card__subtitle"><span class="su-styled-text">Pre-Owned</span></div>
  <div class="su-card-container__attributes__primary">
    <div class="s-card__attribute-row"><span class="su-styled-text s-card__price">${10 + i}.00</span></div>
    <div class="s-card__attribute-row"><span class="su-styled-text">+$4.50 delivery</span></div>
  </div>
  <a class="s-card__link" href="/itm/3000000{start + i:05d}">view</a>
</li>"""
        for i in range(count)
    )
    nxt = '<a class="pagination__next" href="?next">next</a>' if has_next else ""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{keywords} for sale | eBay</title>
<style>li {{ height: 120px; }}</style></head>
<body><ul class="srp-results srp-list">{cards}
<li class="srp-river-answer srp-river-answer--BASIC_PAGINATION_V2">{nxt}</li></ul></body></html>""".encode()


@pytest.fixture
def local_ebay():
    challenge = (FIXTURES / "blocks" / "splashui_challenge.html").read_bytes()
    hits: list[str] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, status: int, body: bytes, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            with lock:
                hits.append(self.path)
            parts = urlsplit(self.path)
            q = parse_qs(parts.query)
            if parts.path.startswith("/splashui/"):
                return self._send(200, challenge)
            if parts.path != "/sch/i.html":
                return self._send(404, b"not found")
            kw, page = q.get("_nkw", [""])[0], int(q.get("_pgn", ["1"])[0])
            if kw == "widget":
                return self._send(200, _results_page(kw, (page - 1) * 12, 12 if page == 1 else 5, has_next=page == 1))
            if kw == "blocked":
                return self._send(302, b"", {"Location": f"/splashui/challenge?ru={quote(self.path)}"})
            if kw == "broken":
                return self._send(500, b"<html><head><title>Oops</title></head><body>error</body></html>")
            if kw == "newlayout":  # eBay renamed every result class: the DOM parser finds nothing
                return self._send(200, _results_page(kw, 0, 3, has_next=False, card_class="x-result"))
            return self._send(404, b"no such search")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", hits
    server.shutdown()


@pytest.fixture
def scrape_env(tmp_path, monkeypatch, local_ebay):
    base, hits = local_ebay
    monkeypatch.setattr(pipeline, "search_url",
                        lambda query, page=1: f"{base}/sch/i.html?_nkw={quote(query.keywords)}&_pgn={page}")
    from ebay_sold import fetch

    # Short waits for pages that never show results.
    monkeypatch.setattr(fetch, "EbayFetcher", functools.partial(
        fetch.EbayFetcher, scroll=False, results_timeout_s=2.0, other_page_grace_s=0.5, self_clear_s=1.0))
    settings = Settings(
        data_dir=tmp_path / "data",
        browser=BrowserSettings(headless=True),
        pacing=PacingSettings(min_delay_s=0, max_delay_s=0, long_pause_every=0, warmup=False, max_retries=1,
                              manual_solve_timeout_s=2),
    )
    return settings, hits


def _queries(*keywords: str) -> list[SearchQuery]:
    return [SearchQuery(keywords=k) for k in keywords]


@pytest.mark.browser
async def test_scrape_follows_pages_then_serves_from_cache(require_browser, scrape_env):
    settings, hits = scrape_env
    with Database(settings.db_path) as db:
        [report] = await pipeline.scrape(_queries("widget"), settings=settings, db=db, pages=5)
        assert [p.listings for p in report.pages] == [12, 5]
        assert report.stopped_reason == "last page" and report.new == 17
        assert len(db.query_listings(keywords="widget")) == 17
        before = len(hits)
        [again] = await pipeline.scrape(_queries("widget"), settings=settings, db=db, pages=5)
    assert all(p.from_cache for p in again.pages) and again.new == 0
    assert len(hits) == before  # nothing re-fetched


@pytest.mark.browser
async def test_challenge_stops_the_run_and_the_next_run_waits(require_browser, scrape_env):
    settings, hits = scrape_env
    with Database(settings.db_path) as db:
        reports = await pipeline.scrape(_queries("blocked", "widget"), settings=settings, db=db)
        assert len(reports) == 1 and reports[0].block_kind == "captcha"
        assert "Stopped instead of retrying" in reports[0].stopped_reason
        assert not any("widget" in h for h in hits)  # the second search was never started
        before = len(hits)
        [later] = await pipeline.scrape(_queries("widget"), settings=settings, db=db)
    assert later.block_kind is not None and len(hits) == before  # cooldown: no request at all


@pytest.mark.browser
async def test_failed_page_moves_on_to_the_next_search(require_browser, scrape_env):
    settings, _ = scrape_env
    with Database(settings.db_path) as db:
        broken, widget = await pipeline.scrape(_queries("broken", "widget"), settings=settings, db=db)
    assert broken.stopped_reason.startswith("fetch failed") and not broken.pages
    assert widget.listings == 12


@pytest.mark.browser
async def test_unknown_layout_falls_back_to_vision_on_offline_render(require_browser, scrape_env, monkeypatch):
    settings, hits = scrape_env
    tiles_seen: list = []

    def fake_vision(tiles, *, settings, site):
        tiles_seen.extend(tiles)
        assert all(Path(p).exists() for p, _ in tiles)
        return [Listing(title="read from pixels", price=12.0, extraction="vision", confidence=0.9)]

    monkeypatch.setattr(pipeline, "vision_extract_tiles", fake_vision)
    with Database(settings.db_path) as db:
        [report] = await pipeline.scrape(_queries("newlayout"), settings=settings, db=db)
        assert report.pages[0].extraction == "vision" and report.listings == 1
        assert db.query_listings(keywords="newlayout")[0].title == "read from pixels"
    assert tiles_seen  # rendered from the fetched HTML ...
    assert sum("newlayout" in h for h in hits) == 1  # ... without asking the server again
