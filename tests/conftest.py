"""Shared test helpers: archived eBay pages and environment capability checks."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
EBAY_FIXTURES = FIXTURES / "ebay"


def fixture_names() -> list[str]:
    """Archived sold-search pages, e.g. ``sold_2026-04-26_hot-wheels-r34-zamac``."""
    manifest = json.loads((EBAY_FIXTURES / "manifest.json").read_text())
    return [f["file"].removesuffix(".html.gz") for f in manifest["fixtures"]]


def legacy_fixture_names() -> list[str]:
    """Real 2024 pages in eBay's older ``li.s-item`` layout."""
    manifest = json.loads((EBAY_FIXTURES / "manifest.json").read_text())
    return [f["file"].removesuffix(".html.gz") for f in manifest.get("legacy", [])]


def load_fixture(name: str) -> str:
    path = EBAY_FIXTURES / f"{name}.html.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return (EBAY_FIXTURES / f"{name}.html").read_text(encoding="utf-8")


def browser_available() -> bool:
    """True when Playwright can launch Chromium here."""
    if importlib.util.find_spec("playwright") is None:
        return False
    from playwright.sync_api import sync_playwright

    from ebay_sold.capture import chromium_launch_options
    from ebay_sold.config import BrowserSettings

    try:
        with sync_playwright() as pw:
            pw.chromium.launch(**chromium_launch_options(BrowserSettings(headless=True))).close()
        return True
    except Exception:
        return False


def vision_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("ultralytics", "cv2", "numpy"))


_BROWSER_OK: bool | None = None


@pytest.fixture(scope="session")
def require_browser() -> None:
    global _BROWSER_OK
    if _BROWSER_OK is None:
        _BROWSER_OK = browser_available()
    if not _BROWSER_OK:
        pytest.skip("Chromium is not available to Playwright (run `playwright install chromium`)")


@pytest.fixture(scope="session")
def require_vision() -> None:
    if not vision_available():
        pytest.skip("vision extra not installed (pip install -e '.[vision]')")


@pytest.fixture(params=fixture_names())
def ebay_page(request) -> tuple[str, str]:
    """Each archived page as ``(name, html)``."""
    return request.param, load_fixture(request.param)
