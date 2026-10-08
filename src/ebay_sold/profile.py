"""Open the scraper's persistent browser profile by hand.

Useful before the first scrape (accept cookies, browse a little, optionally
sign in) and after a challenge (check whether eBay still asks for verification).
Whatever you do here, the scraper sees the same cookies next time.
"""

from __future__ import annotations

from .capture import chromium_launch_options, context_options
from .config import Settings


async def open_profile_browser(settings: Settings, url: str = "https://www.ebay.com/") -> None:
    """Show a headed browser on the scraper's profile and wait until it is closed."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            str(settings.profile_dir),
            **chromium_launch_options(settings.browser, headless=False),
            **context_options(settings.browser),
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(url)
        # Closing the last window closes the context.
        await ctx.wait_for_event("close", timeout=0)
