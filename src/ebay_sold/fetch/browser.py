"""Fetch eBay result pages with one real, persistent, human-paced browser.

Why the 2021 scraper kept hitting CAPTCHAs, and the stance this module takes
instead: we do not try to *defeat* bot detection, we try not to look like a bot
in the first place, by behaving like ONE consistent, ordinary browser user.

* **One persistent profile** (``settings.profile_dir``). Cookies and local
  storage survive between runs, so we are a returning visitor, not a brand-new
  one deep-linking straight into search results on every request.
* **The browser's real user agent.** No UA override: the 2021 code put a random
  Firefox UA on Chromium, an instant fingerprint mismatch. No stealth or
  fingerprint-spoofing plugins either; every patched property is one more
  inconsistency to catch. ``browser.channel = "chrome"`` drives your installed Chrome.
* **Fixed locale and viewport** (``capture.context_options``): a stable
  fingerprint, and reproducible screenshots as a bonus.
* **Human pacing** (``Pacer``): jittered delays, longer breaks, a page budget
  per run, one home-page visit before the first search, and gradual scrolling
  with variable steps and short pauses (which also loads lazy content).
* **Fewer requests.** 240 results per page (``urls.search_url``) and an HTML
  cache (``HtmlCache``), so the same page is never fetched twice.
* **Stop when challenged.** Every page goes through ``blocks.detect_block``,
  judged by the status of the document actually on screen (a sensor page that
  reloads itself into a 403 is a 403). A challenge is never retried
  automatically: hammering a server that has started to doubt you is what turns
  a challenge into a block. Without a human the fetcher raises
  ``BlockedError``. After a block it refuses to load more pages, in this run and
  in later runs on the same ``data_dir`` until ``challenge_cooldown_s`` (or a
  longer ``Retry-After``) has passed. Only timeouts, network errors and HTTP
  5xx are retried.
* **Let an ordinary browser check pass.** A challenge page with no CAPTCHA on
  it is usually a JavaScript check that a normal browser passes by itself in a
  few seconds, so the fetcher first watches it (no reload) for ``self_clear_s``.
* **A human solves the rare CAPTCHA.** In headed (interactive) mode the fetcher
  pauses, says so, and polls the *same* page every couple of seconds, without
  reloading, until you have solved it in the window. Automated CAPTCHA solving
  is never integrated. An Akamai 403 or an HTTP 429 has nothing to solve, so
  those stop the run even when a human is present.
* **Never keep junk.** Only pages that show search results are cached; anything
  else (an unrecognised interstitial, a soft error page) is returned but not
  stored, so it is not replayed from the cache.

Playwright's automation flag is left as it is (``navigator.webdriver`` is true,
and headless Chromium says ``HeadlessChrome`` in its user agent). Hiding those
would be fingerprint spoofing, which this module deliberately does not do.

What none of this can fix is a flagged IP address: from datacenter IPs eBay
answers every request with an Akamai 403. Run it from a home connection
(see docs/anti-bot.md).
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NoReturn
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from .. import capture
from ..config import Settings
from ..models import Box, CardRegion
from .blocks import BlockInfo, detect_block, has_captcha_widget, is_ebay_host, looks_like_results, page_title
from .cache import HtmlCache, cache_key, page_stem
from .pacing import Pacer

if TYPE_CHECKING:  # pragma: no cover
    from playwright.async_api import BrowserContext, Page, Playwright, Response

log = logging.getLogger(__name__)

# Challenge kinds a person can get past in the window. A 403 edge block or a
# 429 has nothing to click; waiting on it would only add requests.
SOLVABLE_KINDS = frozenset({"captcha", "interstitial", "signin"})
# Kinds that can be a JavaScript browser check, which passes without anyone
# clicking anything (when no CAPTCHA widget is on the page).
SELF_CLEAR_KINDS = frozenset({"captcha", "interstitial"})
SOLVE_POLL_S = 2.0
SELF_CLEAR_POLL_S = 1.0
# An absurd Retry-After should not lock the tool for weeks.
MAX_COOLDOWN_S = 24 * 3600.0
COOLDOWN_FILE = "cooldown.json"

# Resolves once the page shows results or looks like a challenge, whichever
# comes first; a page that is neither gets ``graceMs`` (counted from the start
# of its navigation, so a reload restarts it) for its scripts to move on.
_READY_JS = """
(graceMs) => {
  if (document.querySelector('ul.srp-results, li.s-card, li.s-item')) return 'results';
  if (location.pathname.toLowerCase().startsWith('/splashui/')) return 'block';
  const t = (document.title || '').toLowerCase();
  if (/security measure|verify yourself|pardon our interruption|access denied|error page/.test(t)) return 'block';
  if (document.querySelector('iframe[src*="captcha"], .h-captcha, .g-recaptcha')) return 'block';
  if (document.readyState === 'complete' && performance.now() > graceMs) return 'other';
  return false;
}
"""

# The document on screen, read in one step: its HTML (as page.content() builds
# it), URL and HTTP status. The status comes from the document itself
# (Navigation Timing), so it is right even after a JS reload or redirect.
_SNAPSHOT_JS = """
() => {
  let html = '';
  if (document.doctype) html = new XMLSerializer().serializeToString(document.doctype);
  if (document.documentElement) html += document.documentElement.outerHTML;
  let status = 0;
  try {
    const nav = performance.getEntriesByType('navigation')[0];
    status = (nav && nav.responseStatus) || 0;
  } catch (e) {}
  return [html, location.href, status];
}
"""

_SCROLL_STATE_JS = """
() => [window.scrollY,
       Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0),
       window.innerHeight]
"""

# net::ERR_* failures that retrying cannot fix.
_PERMANENT_NET_ERRORS = (
    "ERR_ABORTED", "ERR_BLOCKED_BY_CLIENT", "ERR_INVALID_URL", "ERR_UNSAFE_PORT", "ERR_CERT_",
    "ERR_SSL_", "ERR_BAD_SSL", "ERR_TOO_MANY_REDIRECTS", "ERR_UNKNOWN_URL_SCHEME",
)


_KIND_PHRASE = {
    "captcha": "a CAPTCHA",
    "interstitial": "a browser check",
    "access_denied": "an access-denied page",
    "rate_limited": "a too-many-requests page",
    "signin": "a sign-in page",
}


class BlockedError(RuntimeError):
    """eBay answered with a challenge or block that was not (or cannot be) solved. Stop the run."""

    def __init__(self, info: BlockInfo, challenges_seen: int, *, detail: str = "",
                 retry_after_s: float | None = None):
        self.info = info
        self.challenges_seen = challenges_seen
        self.retry_after_s = retry_after_s  # suggested pause before the next run
        msg = f"eBay served {_KIND_PHRASE.get(info.kind, 'a ' + info.kind + ' page')} ({info.reason}) at {info.url}"
        if detail:
            msg += f": {detail}"
        msg += ". Stopped instead of retrying"
        if retry_after_s:
            msg += f"; wait at least {retry_after_s / 60:.0f} min before the next run"
        super().__init__(msg + " (see docs/anti-bot.md)")


class FetchError(RuntimeError):
    """A page could not be loaded (network error, timeout, HTTP 5xx) even after retries."""

    def __init__(self, message: str, *, url: str, status: int | None = None):
        super().__init__(message)
        self.url = url
        self.status = status


class FetchResult(BaseModel):
    url: str  # what was asked for
    final_url: str  # where the browser ended up
    status: int | None
    html: str
    html_path: str | None = None  # the HtmlCache file, when caching is on
    from_cache: bool = False
    fetched_at: datetime
    block: BlockInfo | None = None  # a challenge that appeared and was cleared (by a person or by itself)
    # False when the page shows neither results nor a known challenge (soft
    # error page, unrecognised interstitial): returned for inspection, never cached.
    has_results: bool = True
    tiles: list[tuple[str, Box]] = Field(default_factory=list)  # (png path, area in page coords)
    card_shots: list[tuple[CardRegion, str]] = Field(default_factory=list)


@dataclass
class _Loaded:
    url: str
    final_url: str
    status: int | None
    html: str
    block: BlockInfo | None


class EbayFetcher:
    """Loads result pages in one persistent browser session. Use as ``async with``.

    ``interactive`` (default: the browser is headed) lets a person solve a
    challenge in the window; otherwise a challenge raises ``BlockedError``.
    A challenge page without a CAPTCHA is first watched for ``self_clear_s``
    in case it is a browser check that passes by itself. ``on_block`` (sync or
    async) is called when the fetcher starts waiting for a person; the default
    logs what to do. Pages eBay served instead of results are kept in
    ``<data_dir>/blocked/`` for diagnosis, and a block that stopped a run is
    recorded in ``<data_dir>/blocked/cooldown.json`` so the next run waits too.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        interactive: bool | None = None,
        pacer: Pacer | None = None,
        cache: HtmlCache | None = None,
        on_block: Callable[[BlockInfo], Any] | None = None,
        scroll: bool = True,
        max_scroll_s: float = 60.0,
        results_timeout_s: float = 20.0,
        other_page_grace_s: float = 6.0,
        self_clear_s: float = 20.0,
    ):
        self.settings = settings
        self.interactive = (not settings.browser.headless) if interactive is None else interactive
        self.pacer = pacer or Pacer(settings.pacing)
        if cache is None and settings.cache.enabled:
            cache = HtmlCache(settings.cache_dir, settings.cache.ttl_hours)
        self.cache = cache
        self.on_block = on_block or self._announce_block
        self.scroll = scroll
        self.max_scroll_s = max_scroll_s
        self.results_timeout_s = results_timeout_s
        self.other_page_grace_s = other_page_grace_s
        self.self_clear_s = self_clear_s
        self.cooldown_path = settings.data_dir / "blocked" / COOLDOWN_FILE
        self.challenges_seen = 0
        self.blocked: BlockInfo | None = None  # set once the run has stopped at a block
        self._retry_after_s: float | None = None
        self._warmed: set[str] = set()
        self._lock = asyncio.Lock()
        self._pw: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        # The latest main-frame document response; a fallback when the page
        # cannot report its own status, and the source of Retry-After.
        self._doc_status: int | None = None
        self._doc_retry_after: str | None = None

    # --- lifecycle ---------------------------------------------------------------

    async def __aenter__(self) -> "EbayFetcher":
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def start(self) -> "EbayFetcher":
        if self._context is not None:
            return self
        from playwright.async_api import async_playwright

        b = self.settings.browser
        profile = self.settings.profile_dir
        profile.mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        try:
            self._context = await self._pw.chromium.launch_persistent_context(
                str(profile), **capture.chromium_launch_options(b), **capture.context_options(b)
            )
        except BaseException:
            log.error("could not start the browser on profile %s (is it open in another window?)", profile)
            await self._pw.stop()
            self._pw = None
            raise
        self._context.set_default_timeout(b.nav_timeout_s * 1000)
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        self._page.on("response", self._on_response)
        return self

    async def close(self) -> None:
        context, pw = self._context, self._pw
        self._context = self._page = self._pw = None
        if context is not None:
            with contextlib.suppress(Exception):
                await context.close()  # flushes cookies to the profile
        if pw is not None:
            with contextlib.suppress(Exception):
                await pw.stop()

    @property
    def context(self) -> "BrowserContext":
        if self._context is None:
            raise RuntimeError("EbayFetcher is not started; use 'async with EbayFetcher(settings) as fetcher:'")
        return self._context

    @property
    def page(self) -> "Page":
        if self._page is None:
            raise RuntimeError("EbayFetcher is not started; use 'async with EbayFetcher(settings) as fetcher:'")
        return self._page

    @property
    def pages_loaded(self) -> int:
        return self.pacer.pages_loaded

    # --- public API ----------------------------------------------------------------

    async def fetch(self, url: str, *, use_cache: bool = True, screenshot_dir: Path | None = None,
                    card_shots: bool = False) -> FetchResult:
        """Return the page at ``url``: from the cache if fresh, else loaded politely.

        ``use_cache=False`` skips the cache lookup but still stores the fresh page.
        With ``screenshot_dir`` the page is also captured as tiles (and per-card
        shots with ``card_shots``); a cached page is re-rendered offline for
        that, so a cache hit never touches eBay.

        Raises ``BlockedError`` (stop the run), ``PageBudgetExceeded`` (stop the
        run) or ``FetchError`` (this page failed after retries).
        """
        if use_cache:
            hit = await self._from_cache(url, screenshot_dir, card_shots)
            if hit is not None:
                return hit

        async with self._lock:
            if use_cache:
                # Another fetch of the same page may have stored it while we waited.
                hit = await self._from_cache(url, screenshot_dir, card_shots)
                if hit is not None:
                    return hit
            page = self.page
            self._refuse_if_stopped()
            await self._maybe_warmup(url)
            loaded = await self._load(url, expect_results=True)
            loaded, solved = await self._resolve_blocks(loaded, url, expect_results=True)

            has_results = looks_like_results(loaded.html)
            html_path = None
            if self.cache is not None and (loaded.status is None or 200 <= loaded.status < 300):
                if has_results:
                    html_path = self._store(url, loaded)
                else:
                    log.warning("%s shows neither search results nor a known challenge (title %r, HTTP %s); "
                                "returning it without caching", loaded.final_url, page_title(loaded.html),
                                loaded.status)
            result = FetchResult(url=url, final_url=loaded.final_url, status=loaded.status, html=loaded.html,
                                 html_path=html_path, from_cache=False, fetched_at=datetime.now(timezone.utc),
                                 block=solved, has_results=has_results)
            if screenshot_dir is not None:
                await capture.stabilize_page(page)
                result.tiles, result.card_shots = await self._capture(page, url, Path(screenshot_dir), card_shots)
            return result

    def cooldown_remaining(self) -> float:
        """Seconds left of the cooldown an earlier block imposed on this ``data_dir`` (0 when none)."""
        found = self._read_cooldown()
        return (found[2] - datetime.now(timezone.utc)).total_seconds() if found else 0.0

    def clear_cooldown(self) -> None:
        """Forget the recorded block, so the next fetch may load pages again."""
        with contextlib.suppress(FileNotFoundError):
            self.cooldown_path.unlink()

    async def _from_cache(self, url: str, screenshot_dir: Path | None, card_shots: bool) -> FetchResult | None:
        if self.cache is None:
            return None
        hit = self.cache.get(url)
        if hit is None:
            return None
        if not looks_like_results(hit.html):  # stored by an older version that cached any 2xx page
            log.info("ignoring cached page without search results: %s", hit.path.name)
            return None
        log.info("cache hit: %s", url)
        result = FetchResult(url=url, final_url=hit.final_url, status=hit.status, html=hit.html,
                             html_path=str(hit.path), from_cache=True, fetched_at=hit.fetched_at)
        if screenshot_dir is not None:
            result.tiles, result.card_shots = await self._render_cached(hit.html, url, Path(screenshot_dir), card_shots)
        return result

    def _store(self, url: str, loaded: "_Loaded") -> str | None:
        try:
            return str(self.cache.put(url, final_url=loaded.final_url, status=loaded.status, html=loaded.html))
        except OSError as exc:  # the page was fetched fine; losing the cache entry is not worth failing for
            log.warning("could not cache %s: %s", url, exc)
            return None

    # --- loading -------------------------------------------------------------------

    async def _maybe_warmup(self, url: str) -> None:
        """Visit the site's home page once before the first search, like a person arriving."""
        if not self.settings.pacing.warmup:
            return
        host = (urlsplit(url).hostname or "").lower()
        if host in self._warmed or not is_ebay_host(host) or host.startswith("signin."):
            return
        self._warmed.add(host)
        if self.pacer.budget_left < 2:  # the search itself matters more
            return
        home = f"https://{host}/"
        log.info("warm-up: visiting %s before the first search", home)
        try:
            # One attempt: retrying an optional visit would spend the page budget the search needs.
            loaded = await self._load(home, expect_results=False, retries=0)
        except FetchError as exc:
            log.warning("warm-up visit failed (%s); going straight to the search", exc)
            return
        await self._resolve_blocks(loaded, home, expect_results=False)

    async def _load(self, url: str, *, expect_results: bool, retries: int | None = None) -> _Loaded:
        """Paced navigation with retries for transient failures. Blocks are returned, never retried."""
        from playwright.async_api import Error as PlaywrightError

        page = self.page
        max_retries = max(0, self.settings.pacing.max_retries if retries is None else retries)
        timeout_ms = self.settings.browser.nav_timeout_s * 1000
        attempt = 0
        while True:
            await self.pacer.wait()
            log.info("loading %s", url)
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                status = response.status if response is not None else self._doc_status
            except PlaywrightError as exc:
                msg = _first_line(exc)
                if "interrupted by another navigation" in msg:
                    # The page sent us elsewhere (often to a challenge): inspect where we landed.
                    await self._quietly(page.wait_for_load_state("domcontentloaded", timeout=timeout_ms))
                    status = self._doc_status
                elif _is_transient(exc) and attempt < max_retries:
                    attempt += 1
                    delay = self.pacer.backoff(attempt)
                    log.warning("loading %s failed (%s); retry %d/%d in %.0f s", url, msg, attempt, max_retries, delay)
                    await self.pacer.cooldown(delay)
                    continue
                else:
                    raise FetchError(f"could not load {url}: {msg}", url=url) from exc

            loaded = await self._inspect(url, status, expect_results=expect_results)
            if loaded.block is None and loaded.status is not None and loaded.status >= 500:
                if attempt < max_retries:
                    attempt += 1
                    delay = self.pacer.backoff(attempt)
                    log.warning("HTTP %d from %s; retry %d/%d in %.0f s",
                                loaded.status, url, attempt, max_retries, delay)
                    await self.pacer.cooldown(delay)
                    continue
                raise FetchError(f"HTTP {loaded.status} from {url} after {attempt} retries", url=url,
                                 status=loaded.status)
            return loaded

    async def _inspect(self, url: str, status: int | None, *, expect_results: bool) -> _Loaded:
        """Wait for the page to show something, check for a block, then read it like a person would.

        ``status`` (from ``goto``) only decides whether to wait; every verdict
        uses the status of the document read with it, because a page can reload
        or redirect itself (e.g. into a 403) after ``goto`` returned.
        """
        if _ok(status):
            await self._settle(expect_results)
        html, final_url, status = await self._snapshot(status)
        block = detect_block(status=status, url=final_url, html=html)
        if block is None and _ok(status) and self.scroll:
            await self._human_scroll(full=expect_results)
            html, final_url, status = await self._snapshot(status)
            block = detect_block(status=status, url=final_url, html=html)
        return _Loaded(url=url, final_url=final_url, status=status, html=html, block=block)

    async def _settle(self, expect_results: bool) -> None:
        from playwright.async_api import Error as PlaywrightError

        page = self.page
        try:
            if expect_results:
                await page.wait_for_function(_READY_JS, arg=self.other_page_grace_s * 1000,
                                             timeout=self.results_timeout_s * 1000, polling=250)
            else:
                await page.wait_for_load_state("load", timeout=min(15.0, self.settings.browser.nav_timeout_s) * 1000)
        except PlaywrightError as exc:
            log.debug("page did not settle: %s", _first_line(exc))

    async def _read_page(self, fallback_status: int | None = None) -> tuple[str, str, int | None]:
        """``(html, url, status)`` of the document on screen, in one step. Raises mid-navigation."""
        html, url, status = await self.page.evaluate(_SNAPSHOT_JS)
        if status:
            return html, url, int(status)
        # The browser could not say (very old Chromium): best guess from the network events.
        return html, url, self._doc_status if self._doc_status is not None else fallback_status

    async def _snapshot(self, fallback_status: int | None = None) -> tuple[str, str, int | None]:
        """Like ``_read_page``, but waits out a navigation in progress."""
        from playwright.async_api import Error as PlaywrightError

        page = self.page
        for _ in range(20):
            try:
                return await self._read_page(fallback_status)
            except PlaywrightError:
                await asyncio.sleep(0.25)
                await self._quietly(page.wait_for_load_state("domcontentloaded", timeout=5000))
        return await self._read_page(fallback_status)

    async def _human_scroll(self, *, full: bool) -> None:
        """Scroll down in uneven wheel steps with short pauses, then back to the top."""
        from playwright.async_api import Error as PlaywrightError

        page, rng, b = self.page, self.pacer.rng, self.settings.browser
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_scroll_s
        try:
            await page.mouse.move(b.viewport_width * rng.uniform(0.3, 0.7), b.viewport_height * rng.uniform(0.35, 0.65),
                                  steps=rng.randint(4, 12))
            for _ in range(400 if full else rng.randint(2, 4)):
                y, height, inner = await page.evaluate(_SCROLL_STATE_JS)
                if y + inner >= height - 2:
                    break
                dy = round(b.viewport_height * rng.uniform(0.4, 0.95))
                await page.mouse.wheel(0, dy)
                await self.pacer.pause(0.15, 0.7)
                if rng.random() < 0.08:
                    await self.pacer.pause(0.8, 2.5)  # stop to read a card
                if await page.evaluate("() => window.scrollY") <= y:
                    # The wheel landed on something that does not scroll the page.
                    await page.evaluate("(dy) => window.scrollBy(0, dy)", dy)
                if loop.time() > deadline:
                    await page.evaluate("() => window.scrollTo(0, document.documentElement.scrollHeight)")
                    break
            await self.pacer.pause(0.3, 1.2)
            await page.evaluate("() => window.scrollTo(0, 0)")
        except PlaywrightError as exc:
            log.debug("scrolling stopped: %s", _first_line(exc))

    # --- challenges ----------------------------------------------------------------

    async def _resolve_blocks(self, loaded: _Loaded, target_url: str, *,
                              expect_results: bool) -> tuple[_Loaded, BlockInfo | None]:
        """Return ``loaded`` once it is not a block, waiting for it to clear if it can; else raise."""
        solved: BlockInfo | None = None
        pacing = self.settings.pacing
        while loaded.block is not None:
            info = loaded.block
            self.challenges_seen += 1
            saved = self._save_block_page(info, loaded.html)
            log.warning("challenge %d this run: %s (%s) at %s%s", self.challenges_seen, info.kind, info.reason,
                        info.url, f" [page saved to {saved}]" if saved else "")
            if info.kind not in SOLVABLE_KINDS:
                self._stop(info, f"there is nothing to solve on a {info.kind.replace('_', ' ')} page")
            if self.challenges_seen > pacing.max_challenges_per_run:
                self._stop(info, f"more than {pacing.max_challenges_per_run} challenges this run")
            after: _Loaded | None = None
            if info.kind in SELF_CLEAR_KINDS and self.self_clear_s > 0 and not has_captcha_widget(loaded.html):
                log.info("no CAPTCHA on the challenge page; giving the browser check up to %.0f s to pass",
                         self.self_clear_s)
                after = await self._await_clear(target_url, self.self_clear_s, human=False,
                                                expect_results=expect_results)
            if after is None:
                if not self.interactive:
                    self._stop(info, "not running interactively, so no one can solve it; "
                                     "run with a visible browser (headless = false) to solve challenges by hand")
                pending = self.on_block(info)
                if inspect.isawaitable(pending):
                    await pending
                after = await self._await_clear(target_url, pacing.manual_solve_timeout_s, human=True,
                                                expect_results=expect_results)
                if after is None:
                    self._stop(info, f"not solved within {pacing.manual_solve_timeout_s:.0f} s")
            if after.block is None:
                log.info("challenge cleared; taking a break before the next page")
                self.pacer.note_challenge()
                solved = info
            loaded = after  # still a block: it turned into another one, handled by the next pass
        return loaded, solved

    async def _await_clear(self, target_url: str, timeout_s: float, *, human: bool,
                           expect_results: bool) -> _Loaded | None:
        """Watch the challenge page, never reloading it, until it is gone.

        ``human=False`` is the short wait for a JavaScript browser check to pass
        by itself; it gives up as soon as the page needs a person (a CAPTCHA
        appears). Returns the page asked for once the browser shows it; if the
        challenge clears onto some other page (another search, other filters),
        the page asked for is loaded normally, after a break, rather than
        caching the wrong one. Returns a page whose ``block`` is set when the
        challenge turned into one nobody can solve, and ``None`` on timeout.
        """
        from playwright.async_api import Error as PlaywrightError

        page = self.page
        if human:
            with contextlib.suppress(PlaywrightError):
                await page.bring_to_front()
        poll_s = SOLVE_POLL_S if human else SELF_CLEAR_POLL_S
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_s)
        target_key, target_path = cache_key(target_url), urlsplit(target_url).path
        while (remaining := deadline - loop.time()) > 0:
            await asyncio.sleep(min(poll_s, remaining))
            if page.is_closed():
                return None
            try:
                html, url, status = await self._read_page()
            except PlaywrightError:
                continue  # mid-navigation; look again at the next poll
            block = detect_block(status=status, url=url, html=html)
            if block is not None:
                if block.kind not in SOLVABLE_KINDS:
                    return _Loaded(url=target_url, final_url=url, status=status, html=html, block=block)
                if not human and (block.kind not in SELF_CLEAR_KINDS or has_captcha_widget(html)):
                    return None  # this one needs a person
                continue
            if cache_key(url) == target_key:
                return await self._inspect(target_url, status, expect_results=expect_results)
            if looks_like_results(html) or urlsplit(url).path == target_path:
                log.info("challenge cleared, but the browser is on %s; loading the page asked for", url)
                self.pacer.note_challenge()
                return await self._load(target_url, expect_results=expect_results)
        return None

    def _save_block_page(self, info: BlockInfo, html: str) -> Path | None:
        """Keep what eBay served instead of results, for diagnosing why (best effort)."""
        try:
            out = self.settings.data_dir / "blocked"
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{info.kind}.html"
            path.write_text(html, encoding="utf-8", errors="replace")
            return path
        except OSError as exc:
            log.debug("could not save the block page: %s", exc)
            return None

    def _stop(self, info: BlockInfo, detail: str) -> NoReturn:
        """Stop this run, and record the block so the next run on this ``data_dir`` waits too."""
        cooldown = max(0.0, self.settings.pacing.challenge_cooldown_s)
        if info.status == 429 or info.kind == "rate_limited":
            asked = _parse_retry_after(self._doc_retry_after, datetime.now(timezone.utc))
            if asked is not None:
                cooldown = max(cooldown, min(asked, MAX_COOLDOWN_S))
        self.blocked, self._retry_after_s = info, cooldown
        self._write_cooldown(info, cooldown)
        raise BlockedError(info, self.challenges_seen, detail=detail, retry_after_s=cooldown)

    def _refuse_if_stopped(self) -> None:
        if self.blocked is not None:
            raise BlockedError(self.blocked, self.challenges_seen, detail="this run already stopped at a challenge",
                               retry_after_s=self._retry_after_s)
        found = self._read_cooldown()
        if found is not None:
            info, since, until = found
            info = info.model_copy(update={"reason": f"{info.reason}; cooling down until {until:%H:%M} UTC"})
            raise BlockedError(
                info, self.challenges_seen, retry_after_s=(until - datetime.now(timezone.utc)).total_seconds(),
                detail=(f"that stopped an earlier run at {since:%Y-%m-%d %H:%M} UTC, so no request was sent "
                        f"(delete {self.cooldown_path} to end the cooldown early)"))

    def _write_cooldown(self, info: BlockInfo, seconds: float) -> None:
        if seconds <= 0:
            return
        now = datetime.now(timezone.utc)
        record = {"since": now.isoformat(), "until": (now + timedelta(seconds=seconds)).isoformat(),
                  "block": info.model_dump()}
        try:
            self.cooldown_path.parent.mkdir(parents=True, exist_ok=True)
            self.cooldown_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("could not record the cooldown in %s: %s", self.cooldown_path, exc)

    def _read_cooldown(self) -> tuple[BlockInfo, datetime, datetime] | None:
        """``(block, since, until)`` of a cooldown still running, else ``None``."""
        try:
            record = json.loads(self.cooldown_path.read_text(encoding="utf-8"))
            since, until = (_aware(datetime.fromisoformat(record[k])) for k in ("since", "until"))
            info = BlockInfo.model_validate(record["block"])
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:  # pydantic's ValidationError is a ValueError
            log.warning("ignoring unreadable cooldown record %s: %s", self.cooldown_path, exc)
            return None
        return (info, since, until) if until > datetime.now(timezone.utc) else None

    def _announce_block(self, info: BlockInfo) -> None:
        log.warning(
            "eBay wants to check that you are human (%s: %s). Solve it in the browser window; "
            "the run continues by itself once search results show (waiting up to %.0f min). "
            "Nothing is solved automatically.",
            info.kind, info.reason, self.settings.pacing.manual_solve_timeout_s / 60,
        )

    def _on_response(self, response: "Response") -> None:
        try:
            if self._page is None or response.frame != self._page.main_frame:
                return
            if not response.request.is_navigation_request() or 300 <= response.status < 400:
                return
        except Exception:  # the frame may already be gone
            return
        self._doc_status = response.status
        self._doc_retry_after = response.headers.get("retry-after")

    # --- screenshots ---------------------------------------------------------------

    async def _capture(self, page: "Page", url: str, out_dir: Path,
                       card_shots: bool) -> tuple[list[tuple[str, Box]], list[tuple[CardRegion, str]]]:
        stem = page_stem(url)
        tiles = await capture.capture_tiles(page, out_dir, stem=stem)
        shots: list[tuple[CardRegion, Path]] = []
        if card_shots:
            regions = await capture.card_regions(page)
            shots = await capture.capture_cards(page, out_dir / f"{stem}_cards", stem="card", regions=regions)
        return [(str(p), box) for p, box in tiles], [(region, str(p)) for region, p in shots]

    async def _render_cached(self, html: str, url: str, out_dir: Path,
                             card_shots: bool) -> tuple[list[tuple[str, Box]], list[tuple[CardRegion, str]]]:
        """Screenshots of a cached page, rendered offline (scripts off, no eBay page requests)."""
        b = self.settings.browser
        async with capture.render_html(html, settings=b.model_copy(update={"headless": True}),
                                       viewport_width=b.viewport_width,
                                       device_scale_factor=b.device_scale_factor) as page:
            return await self._capture(page, url, out_dir, card_shots)

    @staticmethod
    async def _quietly(awaitable: Any) -> None:
        from playwright.async_api import Error as PlaywrightError

        with contextlib.suppress(PlaywrightError):
            await awaitable


def _aware(when: datetime) -> datetime:
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _ok(status: int | None) -> bool:
    return status is None or status < 400


def _parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Seconds asked for by a ``Retry-After`` header (delta-seconds or an HTTP date)."""
    if not value or not value.strip():
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (_aware(when) - now).total_seconds())


def _is_transient(exc: BaseException) -> bool:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    if isinstance(exc, PlaywrightTimeoutError):
        return True
    msg = str(exc)
    return "net::ERR_" in msg and not any(code in msg for code in _PERMANENT_NET_ERRORS)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
