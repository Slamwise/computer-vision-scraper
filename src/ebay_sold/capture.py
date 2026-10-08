"""Deterministic rendering, screenshots, and DOM-derived card regions.

Why screenshots used to be inconsistent, and what this module does instead:

* No fixed viewport / scale factor -> every run rendered a different layout.
  We render at a fixed viewport width and ``device_scale_factor``.
* ``waitForTimeout(5000)`` then shoot -> lazy images and web fonts were sometimes
  loaded, sometimes not. We wait for fonts, force lazy images to load, and wait
  for every image to settle.
* Animations, carets, sticky headers and floating "Feedback" buttons landed in
  random places. We disable animations and hide fixed/sticky overlays with
  ``visibility: hidden`` (layout does not move).
* Very tall full-page shots (240 results is ~60k px) exceed GPU texture limits
  and come back corrupted. We capture fixed-height tiles instead.
* The old ``resize.py`` squashed every image to the same height, distorting
  text differently per image. Nothing here resizes; YOLO letterboxes itself.

``card_regions`` reads where each result card and its fields are drawn. That is
the ground truth used to auto-label YOLO training data and to score the vision
pipeline against the DOM parser.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

from .config import BrowserSettings
from .models import Box, CardRegion

if TYPE_CHECKING:  # pragma: no cover
    from playwright.async_api import Page, Route

STABILIZE_CSS = """
*, *::before, *::after {
  animation: none !important;
  transition: none !important;
  caret-color: transparent !important;
  scroll-behavior: auto !important;
}
"""

# Kept synchronous: on pages rendered with JavaScript disabled, timers and
# load events never fire inside evaluate(), so all waiting happens in Python.
_STABILIZE_JS = """
() => {
  // Hide anything pinned to the viewport: it would be stamped onto every tile.
  for (const el of document.querySelectorAll('body *')) {
    const pos = getComputedStyle(el).position;
    if (pos === 'fixed' || pos === 'sticky') el.style.setProperty('visibility', 'hidden', 'important');
  }
  // Force lazy images to load now rather than when scrolled into view.
  for (const img of document.querySelectorAll('img')) {
    img.loading = 'eager';
    const deferred = img.getAttribute('data-defer-load') || img.getAttribute('data-src');
    if (deferred && img.getAttribute('src') !== deferred) img.setAttribute('src', deferred);
  }
  window.scrollTo(0, 0);
  return document.images.length;
}
"""

_PENDING_JS = """
() => ({
  images: [...document.images].filter((i) => !i.complete).length,
  fonts: document.fonts ? document.fonts.status : 'loaded',
})
"""

_REGIONS_JS = """
() => {
  const sx = window.scrollX, sy = window.scrollY;
  const box = (r) => ({ x: r.left + sx, y: r.top + sy, w: r.width, h: r.height });
  const rectOf = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return r.width >= 2 && r.height >= 2 ? box(r) : null;
  };
  // Tight box around the rendered text, not the (often full-width) block element.
  const textRect = (els) => {
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const el of els) {
      if (!el) continue;
      const range = document.createRange();
      range.selectNodeContents(el);
      for (const r of range.getClientRects()) {
        if (r.width < 2 || r.height < 2) continue;
        x1 = Math.min(x1, r.left); y1 = Math.min(y1, r.top);
        x2 = Math.max(x2, r.right); y2 = Math.max(y2, r.bottom);
      }
    }
    return x2 > x1 ? box({ left: x1, top: y1, width: x2 - x1, height: y2 - y1 }) : null;
  };
  const text = (els) => els.filter(Boolean).map((e) => e.innerText).join(' ').replace(/\\s+/g, ' ').trim();
  const shown = (el) => {
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 50 && r.height > 20;
  };

  let cards = [...document.querySelectorAll('ul.srp-results > li.s-card, ul.srp-results > li.s-item')];
  if (!cards.length) cards = [...document.querySelectorAll('li.s-card, li.s-item')];
  const out = [];
  for (const card of cards) {
    if (!shown(card)) continue;
    const modern = card.classList.contains('s-card');
    const one = (sels) => { for (const s of sels) { const e = card.querySelector(s); if (e && e.innerText.trim()) return e; } return null; };
    const fields = {}, texts = {};
    const put = (name, els, tight = true) => {
      els = els.filter(Boolean);
      if (!els.length) return;
      const r = tight ? textRect(els) : rectOf(els[0]);
      if (!r) return;
      fields[name] = r;
      texts[name] = text(els);
    };

    const title = modern
      ? one(['.s-card__title > .su-styled-text', '.s-card__title'])
      : one(['.s-item__title > span[role=heading]', '.s-item__title > span', '.s-item__title']);
    if (title && /^shop on ebay$/i.test(title.innerText.trim())) continue;  // hidden template card
    put('title', [title]);

    const prices = modern ? [...card.querySelectorAll('.s-card__price')] : [one(['.s-item__price'])];
    put('price', prices);

    put('sold_date', [modern
      ? one(['.s-card__caption .su-styled-text', '.s-card__caption'])
      : one(['.s-item__caption--signal', '.s-item__title--tagblock .POSITIVE', '.s-item__caption'])]);

    let shipping = null;
    if (modern) {
      for (const row of card.querySelectorAll('.su-card-container__attributes__primary .s-card__attribute-row')) {
        const t = row.innerText;
        if (/(delivery|shipping|postage|versand|livraison|spedizione|envío)/i.test(t) && !/returns?/i.test(t)) {
          shipping = row;  // whole row: "+$5.83" and "delivery in 2-4 days" are separate spans
          break;
        }
      }
    } else {
      shipping = one(['.s-item__shipping', '.s-item__logisticsCost', '.s-item__freeXDays']);
    }
    put('shipping', [shipping]);

    put('condition', [modern
      ? one(['.s-card__subtitle > .su-styled-text', '.s-card__subtitle'])
      : one(['.s-item__subtitle .SECONDARY_INFO', '.SECONDARY_INFO'])]);

    const img = modern
      ? card.querySelector('img.s-card__image, .su-media-container img, .su-image img')
      : card.querySelector('.s-item__image-wrapper img, .s-item__image img');
    const ir = rectOf(img);
    if (ir) fields.image = ir;

    let itemId = card.getAttribute('data-listingid');
    if (!itemId) {
      const a = card.querySelector('a[href*="/itm/"]');
      const m = a && a.href.match(/\\/itm\\/(?:[^/?#]+\\/)?(\\d{9,15})/);
      itemId = m ? m[1] : null;
    }
    out.push({ item_id: itemId, box: rectOf(card), fields, texts });
  }
  return out.filter((c) => c.box);
}
"""


def find_chromium_executable() -> str | None:
    """Return an explicit Chromium path if one is configured or pre-installed.

    ``None`` lets Playwright use its own managed browser (``playwright install chromium``).
    """
    env = os.environ.get("EBAY_SOLD_BROWSER_EXECUTABLE")
    if env:
        return env
    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        # Containers sometimes ship a Chromium build older than the installed
        # Playwright expects; fall back to whatever is there.
        root = os.environ["PLAYWRIGHT_BROWSERS_PATH"]
        hits = sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")))
        if hits:
            return hits[-1]
    return None


def chromium_launch_options(settings: BrowserSettings, *, headless: bool | None = None) -> dict[str, Any]:
    """Keyword arguments for ``chromium.launch`` / ``launch_persistent_context``."""
    opts: dict[str, Any] = {"headless": settings.headless if headless is None else headless}
    if settings.executable_path:
        opts["executable_path"] = settings.executable_path
    elif settings.channel:
        opts["channel"] = settings.channel
    else:
        exe = find_chromium_executable()
        if exe:
            opts["executable_path"] = exe
    if settings.proxy_server:
        proxy = {"server": settings.proxy_server}
        if settings.proxy_username:
            proxy["username"] = settings.proxy_username
        if settings.proxy_password:
            proxy["password"] = settings.proxy_password
        opts["proxy"] = proxy
    return opts


def context_options(settings: BrowserSettings, *, viewport_width: int | None = None,
                    device_scale_factor: float | None = None) -> dict[str, Any]:
    """Keyword arguments for a browser context with fixed, reproducible geometry."""
    opts: dict[str, Any] = {
        "viewport": {"width": viewport_width or settings.viewport_width, "height": settings.viewport_height},
        "device_scale_factor": device_scale_factor or settings.device_scale_factor,
        "locale": settings.locale,
        "color_scheme": "light",
        "reduced_motion": "reduce",
    }
    if settings.timezone_id:
        opts["timezone_id"] = settings.timezone_id
    return opts


async def stabilize_page(page: "Page", *, timeout_ms: int = 15000) -> dict[str, int]:
    """Freeze a loaded page into a reproducible state before screenshots.

    Returns ``{"images": total, "pending": still_loading_at_timeout}``.
    """
    # Not page.add_style_tag(): it waits for the style's onload event, which
    # never fires on pages rendered with JavaScript disabled.
    await page.evaluate(
        "(css) => { const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s); }",
        STABILIZE_CSS,
    )
    total = await page.evaluate(_STABILIZE_JS)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_ms / 1000
    state = await page.evaluate(_PENDING_JS)
    while (state["images"] or state["fonts"] != "loaded") and loop.time() < deadline:
        await asyncio.sleep(0.1)
        state = await page.evaluate(_PENDING_JS)
    return {"images": int(total), "pending": int(state["images"])}


async def card_regions(page: "Page") -> list[CardRegion]:
    """Locate every visible result card and its fields, in page coordinates."""
    raw = await page.evaluate(_REGIONS_JS)
    return [CardRegion.model_validate(r) for r in raw]


async def page_size(page: "Page") -> tuple[int, int]:
    w, h = await page.evaluate(
        "() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight]"
    )
    return int(w), int(h)


async def capture_tiles(page: "Page", out_dir: str | Path, *, stem: str = "page",
                        tile_height: int = 2000, overlap: int = 200) -> list[tuple[Path, Box]]:
    """Screenshot the whole page as overlapping horizontal tiles.

    Returns ``(png_path, tile_box)`` pairs; ``tile_box`` is the tile's area in
    page coordinates, so detections can be mapped back onto the page.
    """
    if overlap >= tile_height:
        raise ValueError("overlap must be smaller than tile_height")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width, height = await page_size(page)
    tiles: list[tuple[Path, Box]] = []
    y, i = 0, 0
    while True:
        h = min(tile_height, height - y)
        clip = {"x": 0, "y": y, "width": width, "height": h}
        path = out / f"{stem}_tile{i:03d}.png"
        await page.screenshot(path=str(path), clip=clip, full_page=True, animations="disabled", caret="hide")
        tiles.append((path, Box(x=0, y=y, w=width, h=h)))
        if y + h >= height:
            return tiles
        y += tile_height - overlap
        i += 1


async def capture_cards(page: "Page", out_dir: str | Path, *, stem: str = "card",
                        regions: list[CardRegion] | None = None,
                        pad: int = 4) -> list[tuple[CardRegion, Path]]:
    """Screenshot each result card separately (uniform, self-contained crops)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    regions = regions if regions is not None else await card_regions(page)
    width, height = await page_size(page)
    shots: list[tuple[CardRegion, Path]] = []
    for i, region in enumerate(regions):
        b = region.box
        x0, y0 = max(0.0, b.x - pad), max(0.0, b.y - pad)
        clip = {"x": x0, "y": y0, "width": min(width - x0, b.w + 2 * pad), "height": min(height - y0, b.h + 2 * pad)}
        name = f"{stem}_{i:03d}_{region.item_id or 'noid'}.png"
        path = out / name
        await page.screenshot(path=str(path), clip=clip, full_page=True, animations="disabled", caret="hide")
        shots.append((region, path))
    return shots


def asset_cache_dir() -> Path:
    base = os.environ.get("EBAY_SOLD_ASSET_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "ebay-sold", "assets"
    )
    return Path(base)


@asynccontextmanager
async def render_html(html: str, *, settings: BrowserSettings | None = None, viewport_width: int = 1280,
                      device_scale_factor: float = 1.0, load_images: bool = True,
                      timeout_ms: int = 60000) -> AsyncIterator["Page"]:
    """Render saved eBay HTML offline (no eBay page requests) and yield the stabilized page.

    Scripts are disabled and navigations to eBay pages are blocked; stylesheets
    and product images are fetched from eBay's static CDNs and cached on disk,
    so re-rendering the same page is fast and works offline after the first time.
    With ``load_images=False`` product photos are replaced by a flat gray image.
    """
    from playwright.async_api import async_playwright

    settings = settings or BrowserSettings(headless=True)
    cache = asset_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    async def route(route: "Route") -> None:
        req = route.request
        kind = req.resource_type
        if kind not in ("stylesheet", "image", "font"):
            await route.abort()
            return
        if kind == "image" and not load_images:
            await route.fulfill(status=200, content_type="image/png", body=_GRAY_PNG)
            return
        key = cache / hashlib.sha1(req.url.encode()).hexdigest()
        if key.exists():
            meta = key.with_suffix(".type")
            ctype = meta.read_text() if meta.exists() else "application/octet-stream"
            await route.fulfill(status=200, content_type=ctype, body=key.read_bytes())
            return
        try:
            resp = await route.fetch(timeout=20000)
        except Exception:
            await route.abort()
            return
        body = await resp.body()
        if resp.ok:
            key.write_bytes(body)
            key.with_suffix(".type").write_text(resp.headers.get("content-type", "application/octet-stream"))
        await route.fulfill(response=resp, body=body)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**chromium_launch_options(settings, headless=True))
        try:
            ctx = await browser.new_context(java_script_enabled=False,
                                            **context_options(settings, viewport_width=viewport_width,
                                                              device_scale_factor=device_scale_factor))
            page = await ctx.new_page()
            await page.route("**/*", route)
            await page.set_content(html, wait_until="load", timeout=timeout_ms)
            # Scripts are off, so evaluate() still works (it runs in the page's isolated world).
            await stabilize_page(page)
            yield page
        finally:
            await browser.close()


def _solid_png(width: int = 8, height: int = 8, gray: int = 200) -> bytes:
    """A valid solid-gray PNG, built with the standard library."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([gray]) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


# Stand-in for product photos when images are not loaded.
_GRAY_PNG = _solid_png()
