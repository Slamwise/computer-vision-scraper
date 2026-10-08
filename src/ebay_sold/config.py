"""Runtime settings, loaded from defaults < ``ebay-sold.toml`` < environment variables.

Example ``ebay-sold.toml``::

    data_dir = "data"

    [browser]
    headless = false
    channel = "chrome"            # use your installed Google Chrome instead of bundled Chromium
    proxy_server = "http://user:pass@host:port"   # optional; see docs/anti-bot.md

    [pacing]
    min_delay_s = 8
    max_delay_s = 20
    max_pages_per_run = 20
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

CONFIG_FILENAME = "ebay-sold.toml"


class BrowserSettings(BaseModel):
    # A visible window is both less likely to be challenged and lets you solve a
    # challenge by hand when one appears. Use headless only on machines without a display.
    headless: bool = False
    channel: str | None = None  # "chrome" / "msedge" to drive an installed browser
    executable_path: str | None = None
    # Fixed geometry is what makes screenshots reproducible run to run.
    viewport_width: int = 1366
    viewport_height: int = 900
    device_scale_factor: float = 1.0
    locale: str = "en-US"
    timezone_id: str | None = None  # None = the machine's own zone (a mismatch with your IP is a bot signal)
    proxy_server: str | None = None
    proxy_username: str | None = None
    proxy_password: str | None = None
    nav_timeout_s: float = 45.0


class PacingSettings(BaseModel):
    min_delay_s: float = 6.0
    max_delay_s: float = 18.0
    long_pause_every: int = 8  # take a longer break after this many page loads
    long_pause_min_s: float = 45.0
    long_pause_max_s: float = 120.0
    max_pages_per_run: int = 25
    max_retries: int = 3  # for timeouts / 5xx, with exponential backoff
    challenge_cooldown_s: float = 900.0  # wait this long after a challenge before trying again
    max_challenges_per_run: int = 2  # stop the run rather than keep poking a suspicious server
    manual_solve_timeout_s: float = 300.0  # headed mode: how long to wait for you to solve a challenge
    warmup: bool = True  # visit the home page once before the first search


class CacheSettings(BaseModel):
    enabled: bool = True
    ttl_hours: float = 24.0


class VisionSettings(BaseModel):
    weights: Path | None = None  # default: <data_dir>/models/ebay-sold-yolo.pt
    imgsz: int = 1024
    conf: float = 0.25
    ocr_backend: str = "auto"  # "auto" | "rapidocr" | "tesseract"


class LLMSettings(BaseModel):
    model: str = "claude-opus-5-5"
    effort: str = "low"  # reading a price off a card is a simple task
    max_tokens: int = 4096


class Settings(BaseModel):
    data_dir: Path = Path("data")
    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    pacing: PacingSettings = Field(default_factory=PacingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ebay_sold.sqlite"

    @property
    def profile_dir(self) -> Path:
        return self.data_dir / "browser-profile"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "html-cache"

    @property
    def screenshot_dir(self) -> Path:
        return self.data_dir / "screenshots"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def weights_path(self) -> Path:
        return self.vision.weights or (self.models_dir / "ebay-sold-yolo.pt")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.profile_dir, self.cache_dir, self.screenshot_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


_ENV_MAP: dict[str, tuple[str, ...]] = {
    "EBAY_SOLD_DATA_DIR": ("data_dir",),
    "EBAY_SOLD_HEADLESS": ("browser", "headless"),
    "EBAY_SOLD_BROWSER_CHANNEL": ("browser", "channel"),
    "EBAY_SOLD_BROWSER_EXECUTABLE": ("browser", "executable_path"),
    "EBAY_SOLD_PROXY": ("browser", "proxy_server"),
    "EBAY_SOLD_PROXY_USERNAME": ("browser", "proxy_username"),
    "EBAY_SOLD_PROXY_PASSWORD": ("browser", "proxy_password"),
    "EBAY_SOLD_LLM_MODEL": ("llm", "model"),
}


def load_settings(config_path: str | Path | None = None, **overrides: Any) -> Settings:
    """Load settings. ``overrides`` are top-level or dotted keys, e.g. ``{"browser.headless": True}``."""
    data: dict[str, Any] = {}
    path = Path(config_path) if config_path else Path(CONFIG_FILENAME)
    if path.is_file():
        data = _read_toml(path)
    elif config_path:
        raise FileNotFoundError(path)
    for env, keys in _ENV_MAP.items():
        if env in os.environ:
            _set(data, keys, os.environ[env])
    for dotted, value in overrides.items():
        if value is not None:
            _set(data, tuple(dotted.split(".")), value)
    return Settings.model_validate(data)


def _set(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    for k in keys[:-1]:
        data = data.setdefault(k, {})
    data[keys[-1]] = value


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as f:
        return tomllib.load(f)
