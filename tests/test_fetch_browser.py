"""EbayFetcher end to end, against a local HTTP server (never eBay).

Every test routes the browser so any request that is not to the local server is
aborted: fixture pages link to eBay's CDNs and challenge pages load hCaptcha.
"""

from __future__ import annotations

import asyncio
import random
import socket
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import pytest

from ebay_sold.config import BrowserSettings, PacingSettings, Settings
from ebay_sold.fetch import BlockedError, EbayFetcher, FetchError, HtmlCache, Pacer, PageBudgetExceeded

from conftest import FIXTURES, load_fixture

pytestmark = [pytest.mark.browser]

RESULTS = load_fixture("sold_2026-04-16_ta1-adapter").encode("utf-8")
CHALLENGE = (FIXTURES / "blocks" / "splashui_challenge.html").read_bytes()
AKAMAI = (FIXTURES / "blocks" / "akamai_403.html").read_bytes()
# Challenge pages send the browser back to their ``ru`` parameter once passed;
# /splashui/verify is where the page reports that it was passed.
_BACK_TO_RU = (b"fetch('/splashui/verify').then(() => "
               b"location.replace(new URLSearchParams(location.search).get('ru')))")
# A CAPTCHA that "a human solves" 1.5 s after it appears.
SOLVE = CHALLENGE.replace(b"</body>", b"<script>setTimeout(() => " + _BACK_TO_RU + b", 1500);</script></body>")
# eBay's JavaScript browser check: no CAPTCHA, an ordinary browser passes it by itself.
_CHECK = (b"<!DOCTYPE html><html><head><title>Pardon Our Interruption...</title></head><body>"
          b"<h1>Pardon Our Interruption...</h1><p>Checking your browser before you access eBay.</p>"
          b"<p>Your browser will redirect to your requested content shortly.</p>%s</body></html>")
CHECK_PASSES = _CHECK % (b"<script>setTimeout(() => " + _BACK_TO_RU + b", 1000);</script>")
CHECK_STUCK = _CHECK % b""
CHECK_THEN_CAPTCHA = _CHECK % (
    b"<script>setTimeout(() => { const d = document.createElement('div'); d.className = 'h-captcha';"
    b" document.body.appendChild(d); }, 500);</script>")
CHALLENGE_PAGES = {"solve": SOLVE, "check": CHECK_PASSES, "stuck": CHECK_STUCK, "widget": CHECK_THEN_CAPTCHA}
HOME = b"<!DOCTYPE html><html><head><title>Electronics, Cars, Fashion | eBay</title></head><body>home</body></html>"
TOO_MANY = b"<html><head><title>Too Many Requests</title></head><body>Too many requests</body></html>"
# An Akamai sensor page: answered 200, then its script reloads the same URL (here: into the 403 edge block).
_SCRIPTED = (b"<!DOCTYPE html><html><head><title>eBay</title></head>"
             b"<body><script>setTimeout(() => %s, 300);</script></body></html>")
SENSOR = _SCRIPTED % b"location.reload()"
JS_TO_AKAMAI = _SCRIPTED % b"{ location.href = '/akamai'; }"
PLAIN = b"<!DOCTYPE html><html><head><title>Hello</title></head><body><p>Nothing to see here.</p></body></html>"


def results_for(nkw: str) -> bytes:
    """The results fixture, its title naming the search, so a test can tell which search it shows."""
    return RESULTS.replace(b"<title>", f"<title>{nkw} | ".encode(), 1)


def _small_results(n: int = 12) -> bytes:
    """A styled s-card results page with no external resources (fast, stable screenshots)."""
    cards = "".join(
        f"""<li class="s-card s-card--horizontal" data-listingid="2000000000{i:02d}">
  <img class="s-card__image" alt="" width="180" height="180">
  <div class="body">
    <div class="s-card__caption"><span class="su-styled-text">Sold  Apr {i + 1}, 2026</span></div>
    <div class="s-card__title"><span class="su-styled-text">Test widget number {i}</span></div>
    <div class="s-card__subtitle"><span class="su-styled-text">Pre-Owned</span></div>
    <div class="su-card-container__attributes__primary">
      <div class="s-card__attribute-row"><span class="su-styled-text s-card__price">${10 + i}.99</span></div>
      <div class="s-card__attribute-row"><span class="su-styled-text">+$4.50 delivery</span></div>
    </div>
    <a class="s-card__link" href="/itm/2000000000{i:02d}">view</a>
  </div>
</li>"""
        for i in range(n)
    )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Test Widget for sale | eBay</title>
<style>
  body {{ font-family: sans-serif; margin: 0; }}
  ul.srp-results {{ list-style: none; margin: 0; padding: 16px; }}
  li.s-card {{ display: flex; gap: 16px; height: 200px; padding: 10px; border-bottom: 1px solid #ddd; }}
  .s-card__image {{ background: #ccc; }}
</style></head>
<body><ul class="srp-results srp-list">{cards}</ul></body></html>""".encode()


SMALL = _small_results()


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str]] = []  # (path?query, Cookie header)
        self.flaky_failures = 1
        self.gate_open = False  # /gated shows results once a challenge was passed
        self.sensor_hits = 0
        self.retry_after = "120"
        self.base = ""

    def record(self, path: str, cookie: str) -> None:
        with self.lock:
            self.requests.append((path, cookie))

    def count(self, path: str) -> int:
        with self.lock:
            return sum(1 for p, _ in self.requests if urlsplit(p).path == path)

    def cookies_for(self, path_and_query: str) -> list[str]:
        with self.lock:
            return [c for p, c in self.requests if p == path_and_query]


@pytest.fixture
def server():
    state = ServerState()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/favicon.ico":
                self._send(404, b"")
                return
            state.record(self.path, self.headers.get("Cookie", ""))
            query = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
            headers: dict[str, str] = {}
            if path == "/sch/i.html":
                headers["Set-Cookie"] = "visitor=returning; Max-Age=86400; Path=/"
                status, body = 200, RESULTS
            elif path == "/small":
                status, body = 200, SMALL
            elif path == "/gated":
                # Like eBay: redirect to a challenge whose ``ru`` is the search (or ``land``, to
                # simulate a challenge that lets the browser through onto some other page).
                if state.gate_open:
                    status, body = 200, results_for(query.get("_nkw", ""))
                else:
                    ru = query.get("land", self.path)
                    headers["Location"] = f"/splashui/challenge?mode={query.get('via', '')}&ru={quote(ru, safe='')}"
                    status, body = 302, b""
            elif path == "/splashui/challenge":
                status, body = 200, CHALLENGE_PAGES.get(query.get("mode", ""), CHALLENGE)
            elif path == "/splashui/verify":
                state.gate_open = True
                status, body = 200, b"ok"
            elif path == "/akamai":
                status, body = 403, AKAMAI
            elif path == "/sensor":
                with state.lock:
                    state.sensor_hits += 1
                    first = state.sensor_hits == 1
                status, body = (200, SENSOR) if first else (403, AKAMAI)
            elif path == "/jsredirect":
                status, body = 200, JS_TO_AKAMAI
            elif path == "/softerror":  # eBay's error page, but served with 200
                status, body = 200, AKAMAI
            elif path == "/plain":
                status, body = 200, PLAIN
            elif path == "/ratelimit":
                headers["Retry-After"] = state.retry_after
                status, body = 429, TOO_MANY
            elif path == "/flaky":
                with state.lock:
                    fail = state.flaky_failures > 0
                    state.flaky_failures -= 1
                status, body = (500, b"<html><body>Internal Server Error</body></html>") if fail else (200, RESULTS)
            elif path == "/":
                status, body = 200, HOME
            else:
                status, body = 404, b"<html><body>not found</body></html>"
            self._send(status, body, headers)

        def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state
    httpd.shutdown()
    httpd.server_close()


FAST = dict(min_delay_s=0, max_delay_s=0, long_pause_every=0, warmup=False, manual_solve_timeout_s=10,
            max_retries=2, max_pages_per_run=25, max_challenges_per_run=2)


class Sleeps(list):
    async def __call__(self, seconds: float) -> None:
        self.append(seconds)


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


@pytest.fixture
def make_fetcher(tmp_path, sleeps, require_browser):
    """Build a fetcher with instant pacing (sleeps are recorded, not slept)."""

    def factory(*, data_dir: Path | None = None, pacing: dict | None = None, **kwargs) -> EbayFetcher:
        settings = Settings(
            data_dir=data_dir or tmp_path / "data",
            browser=BrowserSettings(headless=True, nav_timeout_s=20),
            pacing=PacingSettings(**{**FAST, **(pacing or {})}),
        )
        pacer = Pacer(settings.pacing, sleep=sleeps, rng=random.Random(7))
        kwargs.setdefault("interactive", False)
        return EbayFetcher(settings, pacer=pacer, **kwargs)

    return factory


@asynccontextmanager
async def running(fetcher: EbayFetcher, server: ServerState):
    """Start the fetcher with everything but the local server blocked, and scroll tracking."""
    async with fetcher:
        async def abort(route):
            await route.abort()

        await fetcher.context.route(lambda url: not url.startswith(server.base), abort)
        await fetcher.context.add_init_script(
            "window.__wheels = 0; window.__maxY = 0;"
            "addEventListener('wheel', () => { window.__wheels++; }, {passive: true});"
            "addEventListener('scroll', () => { window.__maxY = Math.max(window.__maxY, scrollY); }, {passive: true});"
        )
        yield fetcher


async def test_fetch_caches_and_second_fetch_is_a_cache_hit(make_fetcher, server):
    url = server.base + "/sch/i.html?_nkw=ta1+adapter&LH_Sold=1&LH_Complete=1&_ipg=240"
    async with running(make_fetcher(), server) as f:
        r = await f.fetch(url)
        assert r.status == 200 and not r.from_cache and r.block is None
        assert r.final_url == url
        assert 'data-listingid="' in r.html or "data-listingid=" in r.html
        assert r.html_path and Path(r.html_path).is_file() and Path(r.html_path).with_suffix(".json").is_file()
        assert Path(r.html_path).parent == f.settings.cache_dir
        # read like a person: wheel-scrolled all the way down, then back to the top
        wheels, max_y, bottom, y = await f.page.evaluate(
            "() => [window.__wheels, window.__maxY, document.documentElement.scrollHeight - innerHeight, scrollY]")
        assert wheels > 3 and max_y >= bottom - 2 and y == 0
        assert server.count("/sch/i.html") == 1

        hit = await f.fetch(url + "&_trksid=p2334524.m570.l1313")  # same results, tracking param differs
        assert hit.from_cache and hit.html == r.html and hit.html_path == r.html_path
        assert server.count("/sch/i.html") == 1  # no new request
        assert f.pages_loaded == 1

        fresh = await f.fetch(url, use_cache=False)
        assert not fresh.from_cache
        assert server.count("/sch/i.html") == 2


async def test_non_interactive_challenge_stops_after_one_request(make_fetcher, server):
    async with running(make_fetcher(), server) as f:
        with pytest.raises(BlockedError) as exc:
            await f.fetch(server.base + "/splashui/challenge?ap=1&ru=x")
        assert exc.value.info.kind == "captcha"
        assert exc.value.challenges_seen == 1
        assert exc.value.retry_after_s == f.settings.pacing.challenge_cooldown_s
        assert server.count("/splashui/challenge") == 1  # never retried
        assert not list(f.settings.cache_dir.glob("*.html"))  # a challenge is never cached
        saved = list((f.settings.data_dir / "blocked").glob("*-captcha.html"))
        assert len(saved) == 1 and "Security Measure" in saved[0].read_text()  # kept for diagnosis

        # the run has stopped: nothing else is requested
        with pytest.raises(BlockedError):
            await f.fetch(server.base + "/sch/i.html?_nkw=after")
        assert server.count("/sch/i.html") == 0
        assert len(server.requests) == 1


def gated(server: ServerState, nkw: str, via: str, **extra: str) -> str:
    return f"{server.base}/gated?" + urlencode({"_nkw": nkw, "via": via, **extra})


async def test_interactive_challenge_waits_for_the_human(make_fetcher, server, sleeps):
    seen = []
    url = gated(server, "A", "solve")
    async with running(make_fetcher(interactive=True, on_block=seen.append), server) as f:
        r = await f.fetch(url)
        assert [b.kind for b in seen] == ["captcha"]
        assert r.block is not None and r.block.kind == "captcha"
        assert r.status == 200 and r.has_results
        assert r.final_url == url
        assert "<title>A | " in r.html and "Security Measure" not in r.html
        assert server.count("/splashui/challenge") == 1  # polled, never reloaded
        assert server.count("/gated") == 2  # the search, then the navigation the "human" caused
        assert f.challenges_seen == 1 and f.blocked is None
        assert f.cache.get(url).html == r.html  # cached under the URL asked for

        # after a solved challenge the next page load waits a long break first
        before = len(sleeps)
        await f.fetch(server.base + "/small")
        assert any(s >= f.settings.pacing.long_pause_min_s for s in sleeps[before:])


async def test_challenge_that_clears_onto_another_search_loads_the_one_asked_for(make_fetcher, server):
    # The challenge lets the browser through, but onto a different search: that page
    # must not be cached (or returned) as the search that was asked for.
    url = gated(server, "A", "solve", land="/gated?_nkw=OTHER")
    async with running(make_fetcher(interactive=True, on_block=lambda info: None), server) as f:
        r = await f.fetch(url)
        assert r.final_url == url and "<title>A | " in r.html and "OTHER" not in r.html
        assert f.cache.get(url).html == r.html
        assert server.cookies_for("/gated?_nkw=OTHER")  # the browser did land there first
        assert server.count("/gated") == 3  # search, the other search, then the search again
        assert f.challenges_seen == 1


@pytest.mark.parametrize("interactive", [False, True])
async def test_browser_check_without_captcha_passes_by_itself(make_fetcher, server, interactive):
    seen = []
    url = gated(server, "B", "check")
    async with running(make_fetcher(interactive=interactive, on_block=seen.append), server) as f:
        r = await f.fetch(url)
        assert r.final_url == url and r.has_results and r.html_path
        assert r.block is not None and r.block.kind == "captcha"  # the /splashui/challenge URL
        assert seen == []  # nobody was asked to do anything
        assert f.challenges_seen == 1 and f.blocked is None
        assert server.count("/splashui/challenge") == 1


async def test_browser_check_that_never_passes_stops(make_fetcher, server):
    async with running(make_fetcher(self_clear_s=2), server) as f:
        t0 = time.monotonic()
        with pytest.raises(BlockedError, match="not running interactively") as exc:
            await f.fetch(gated(server, "C", "stuck"))
        assert time.monotonic() - t0 >= 2
        assert exc.value.info.kind == "captcha"
        assert server.count("/splashui/challenge") == 1 and server.count("/gated") == 1


async def test_browser_check_that_turns_into_a_captcha_stops_at_once(make_fetcher, server):
    async with running(make_fetcher(self_clear_s=60), server) as f:
        t0 = time.monotonic()
        with pytest.raises(BlockedError, match="not running interactively"):
            await f.fetch(gated(server, "D", "widget"))
        assert time.monotonic() - t0 < 20  # gave up when the CAPTCHA appeared, not after 60 s
        assert server.count("/splashui/challenge") == 1


@pytest.mark.parametrize("path", ["/sensor", "/jsredirect"])
async def test_block_after_a_script_reload_or_redirect_is_seen(make_fetcher, server, path):
    # goto() answered 200, then the page replaced itself with a 403 edge block.
    async with running(make_fetcher(), server) as f:
        with pytest.raises(BlockedError) as exc:
            await f.fetch(server.base + path + "?_nkw=x")
        assert exc.value.info.kind == "access_denied" and exc.value.info.status == 403
        assert not list(f.settings.cache_dir.glob("*.html"))
        requests = len(server.requests)
        with pytest.raises(BlockedError):  # the run stays stopped
            await f.fetch(server.base + path + "?_nkw=x")
        assert len(server.requests) == requests


@pytest.mark.parametrize("path", ["/softerror", "/plain"])
async def test_page_without_results_is_returned_but_never_cached(make_fetcher, server, path):
    url = server.base + path + "?_nkw=x"
    async with running(make_fetcher(results_timeout_s=30, other_page_grace_s=1), server) as f:
        t0 = time.monotonic()
        r = await f.fetch(url)
        assert time.monotonic() - t0 < 15  # did not sit out results_timeout_s
        assert r.status == 200 and r.block is None and not r.has_results
        assert r.html_path is None and not list(f.settings.cache_dir.glob("*.html"))
        again = await f.fetch(url)
        assert not again.from_cache and server.count(path) == 2

        # such a page cached by an older version is not served either
        f.cache.put(url, final_url=url, status=200, html=r.html)
        third = await f.fetch(url)
        assert not third.from_cache and server.count(path) == 3


async def test_interactive_challenge_not_solved_in_time(make_fetcher, server):
    seen = []

    async def on_block(info):  # async callbacks are awaited
        seen.append(info)

    fetcher = make_fetcher(interactive=True, on_block=on_block, pacing={"manual_solve_timeout_s": 2.5})
    async with running(fetcher, server) as f:
        with pytest.raises(BlockedError, match="not solved within"):
            await f.fetch(server.base + "/splashui/challenge")
        assert len(seen) == 1
        assert server.count("/splashui/challenge") == 1


async def test_challenge_limit_per_run(make_fetcher, server):
    seen = []
    fetcher = make_fetcher(interactive=True, on_block=seen.append, pacing={"max_challenges_per_run": 0})
    async with running(fetcher, server) as f:
        with pytest.raises(BlockedError, match="more than 0 challenges"):
            await f.fetch(gated(server, "A", "solve"))
        assert seen == []


async def test_server_error_is_retried_with_backoff(make_fetcher, server, sleeps):
    async with running(make_fetcher(), server) as f:
        r = await f.fetch(server.base + "/flaky")
        assert r.status == 200 and "srp-results" in r.html
        assert server.count("/flaky") == 2
        assert any(4.0 <= s <= 6.25 for s in sleeps)  # first backoff step
        assert f.pages_loaded == 2  # a retry is a page load too


async def test_persistent_server_error_gives_fetch_error(make_fetcher, server, sleeps):
    server.flaky_failures = 99
    async with running(make_fetcher(pacing={"max_retries": 2}), server) as f:
        with pytest.raises(FetchError) as exc:
            await f.fetch(server.base + "/flaky")
        assert exc.value.status == 500
        assert server.count("/flaky") == 3
        assert f.blocked is None


async def test_network_error_is_retried_then_fails(make_fetcher, server, sleeps):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    async with make_fetcher(pacing={"max_retries": 2}) as f:  # nothing listens there: connection refused
        with pytest.raises(FetchError, match="ERR_CONNECTION_REFUSED"):
            await f.fetch(f"http://127.0.0.1:{port}/sch/i.html")
        assert len([s for s in sleeps if s >= 4.0]) == 2


async def test_akamai_block_is_not_retried_even_with_a_human(make_fetcher, server):
    seen = []
    async with running(make_fetcher(interactive=True, on_block=seen.append), server) as f:
        with pytest.raises(BlockedError) as exc:
            await f.fetch(server.base + "/akamai")
        assert exc.value.info.kind == "access_denied" and exc.value.info.status == 403
        assert server.count("/akamai") == 1
        assert seen == []  # nothing a person could solve


async def test_rate_limited(make_fetcher, server):
    async with running(make_fetcher(), server) as f:
        with pytest.raises(BlockedError) as exc:
            await f.fetch(server.base + "/ratelimit")
        assert exc.value.info.kind == "rate_limited" and exc.value.info.status == 429
        assert exc.value.retry_after_s == f.settings.pacing.challenge_cooldown_s  # Retry-After: 120 is shorter
        assert server.count("/ratelimit") == 1


async def test_block_cooldown_carries_over_to_the_next_run(make_fetcher, server, tmp_path):
    server.retry_after = "3600"
    data_dir = tmp_path / "shared"
    async with running(make_fetcher(data_dir=data_dir), server) as f:
        await f.fetch(server.base + "/sch/i.html?_nkw=cached")
        with pytest.raises(BlockedError) as exc:
            await f.fetch(server.base + "/ratelimit")
        assert exc.value.retry_after_s == 3600  # the server asked for longer than challenge_cooldown_s
        assert (data_dir / "blocked" / "cooldown.json").is_file()

    async with running(make_fetcher(data_dir=data_dir), server) as f:
        assert 3500 < f.cooldown_remaining() <= 3600
        with pytest.raises(BlockedError) as exc:
            await f.fetch(server.base + "/sch/i.html?_nkw=next")
        assert exc.value.info.kind == "rate_limited" and "cooling down until" in exc.value.info.reason
        assert 3500 < exc.value.retry_after_s <= 3600 and "no request was sent" in str(exc.value)
        assert f.pages_loaded == 0 and server.count("/sch/i.html") == 1  # nothing new requested
        assert (await f.fetch(server.base + "/sch/i.html?_nkw=cached")).from_cache  # the cache still works

        f.clear_cooldown()
        assert f.cooldown_remaining() == 0
        assert (await f.fetch(server.base + "/sch/i.html?_nkw=next")).status == 200


async def test_screenshots_live_and_from_cache(make_fetcher, server, tmp_path):
    url = server.base + "/small"
    async with running(make_fetcher(), server) as f:
        r = await f.fetch(url, screenshot_dir=tmp_path / "shots", card_shots=True)
        assert len(r.tiles) >= 2
        assert all(Path(p).is_file() and Path(p).stat().st_size > 1000 for p, _ in r.tiles)
        first, second = r.tiles[0][1], r.tiles[1][1]
        assert first.y == 0 and 0 < second.y < first.h  # consecutive tiles overlap
        assert len(r.card_shots) == 12
        region, path = r.card_shots[0]
        assert region.item_id == "200000000000" and Path(path).is_file()
        assert region.texts["price"] == "$10.99"

        # a cache hit is rendered offline for screenshots: still no new request
        again = await f.fetch(url, screenshot_dir=tmp_path / "shots2")
        assert again.from_cache and len(again.tiles) >= 2
        assert all(Path(p).is_file() for p, _ in again.tiles)
        assert server.count("/small") == 1


async def test_page_budget(make_fetcher, server):
    async with running(make_fetcher(pacing={"max_pages_per_run": 1}), server) as f:
        await f.fetch(server.base + "/sch/i.html?_nkw=a")
        with pytest.raises(PageBudgetExceeded):
            await f.fetch(server.base + "/sch/i.html?_nkw=b")
        assert server.count("/sch/i.html") == 1
        assert (await f.fetch(server.base + "/sch/i.html?_nkw=a")).from_cache  # cache still served


async def test_profile_keeps_cookies_between_runs(make_fetcher, server, tmp_path):
    data_dir = tmp_path / "shared"
    async with running(make_fetcher(data_dir=data_dir), server) as f:
        await f.fetch(server.base + "/sch/i.html?_nkw=first")
    assert (data_dir / "browser-profile").is_dir()
    async with running(make_fetcher(data_dir=data_dir), server) as f:
        await f.fetch(server.base + "/sch/i.html?_nkw=second")
    assert "visitor=returning" not in server.cookies_for("/sch/i.html?_nkw=first")[0]
    assert "visitor=returning" in server.cookies_for("/sch/i.html?_nkw=second")[0]


async def test_warmup_only_for_ebay_hosts(make_fetcher, server):
    async with running(make_fetcher(pacing={"warmup": True}), server) as f:
        await f.fetch(server.base + "/sch/i.html?_nkw=x")
        assert server.count("/") == 0
        assert f.pages_loaded == 1


async def test_fetch_requires_start(make_fetcher):
    with pytest.raises(RuntimeError, match="not started"):
        await make_fetcher(scroll=False).fetch("http://127.0.0.1:9/x", use_cache=False)


async def test_concurrent_fetches_are_serialised(make_fetcher, server):
    async with running(make_fetcher(), server) as f:
        a, b = await asyncio.gather(f.fetch(server.base + "/small"), f.fetch(server.base + "/sch/i.html?_nkw=c"))
        assert a.status == b.status == 200
        assert "Test widget number 0" in a.html and "srp-results" in b.html


async def test_concurrent_fetches_of_the_same_page_load_it_once(make_fetcher, server):
    url = server.base + "/small"
    async with running(make_fetcher(scroll=False), server) as f:
        a, b = await asyncio.gather(f.fetch(url), f.fetch(url))
        assert sorted([a.from_cache, b.from_cache]) == [False, True]
        assert a.html == b.html and server.count("/small") == 1 and f.pages_loaded == 1


async def test_failed_warmup_costs_one_page_load(make_fetcher, server, sleeps, monkeypatch):
    from ebay_sold.fetch import browser

    monkeypatch.setattr(browser, "is_ebay_host", lambda host: True)  # so 127.0.0.1 gets a warm-up visit
    home_hits = []

    async def broken_home(route):
        home_hits.append(route.request.url)
        await route.fulfill(status=503, body="<html><body>Service Unavailable</body></html>", content_type="text/html")

    fetcher = make_fetcher(pacing={"warmup": True, "max_pages_per_run": 3, "max_retries": 3})
    async with running(fetcher, server) as f:
        await f.context.route("https://127.0.0.1/", broken_home)
        r = await f.fetch(server.base + "/sch/i.html?_nkw=warm")
        assert r.status == 200 and r.has_results
        assert home_hits == ["https://127.0.0.1/"]  # one attempt, no retries
        assert f.pages_loaded == 2 and not [s for s in sleeps if s >= 4]  # no backoff spent on it


async def test_cache_write_failure_keeps_the_page(make_fetcher, server, tmp_path):
    not_a_dir = tmp_path / "cache-is-a-file"
    not_a_dir.write_text("x")
    async with running(make_fetcher(cache=HtmlCache(not_a_dir, 24)), server) as f:
        r = await f.fetch(server.base + "/sch/i.html?_nkw=x")
        assert r.status == 200 and r.has_results and r.html_path is None


async def test_warmup_visits_the_ebay_home_page_once(make_fetcher, monkeypatch):
    # No browser and no eBay: record what _load would have navigated to.
    from ebay_sold.fetch import browser

    fetcher = make_fetcher(pacing={"warmup": True})
    visited = []

    async def fake_load(url, *, expect_results, retries=None):
        visited.append((url, expect_results, retries))
        return browser._Loaded(url=url, final_url=url, status=200, html="<html></html>", block=None)

    monkeypatch.setattr(fetcher, "_load", fake_load)
    await fetcher._maybe_warmup("https://www.ebay.co.uk/sch/i.html?_nkw=a&LH_Sold=1")
    await fetcher._maybe_warmup("https://www.ebay.co.uk/sch/i.html?_nkw=b&LH_Sold=1")
    await fetcher._maybe_warmup("https://www.ebay.de/sch/i.html?_nkw=a&LH_Sold=1")
    await fetcher._maybe_warmup("http://127.0.0.1:8000/sch/i.html")
    # one attempt each (retries=0): an optional visit must not eat the search's page budget
    assert visited == [("https://www.ebay.co.uk/", False, 0), ("https://www.ebay.de/", False, 0)]

    off = make_fetcher(pacing={"warmup": False})
    monkeypatch.setattr(off, "_load", fake_load)
    await off._maybe_warmup("https://www.ebay.com/sch/i.html?_nkw=a")
    assert len(visited) == 2
