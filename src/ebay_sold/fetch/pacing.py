"""Human-paced navigation: jittered delays, occasional breaks, a page budget per run.

A fast, regular, unbounded stream of page loads is the clearest behavioural
bot signal there is, and the 2021 scraper produced exactly that (a fixed
``waitForTimeout`` and no limit). ``Pacer`` spaces navigations the way a person
reading results would:

* every gap is a fresh random draw in ``[min_delay_s, max_delay_s]``, skewed
  toward the low-middle of the range with an occasional long one (a beta
  distribution), so there is no fixed rhythm to spot;
* the gap is measured from the start of the previous navigation, so time spent
  loading and scrolling the page counts as reading time;
* every ``long_pause_every`` loads it takes a longer break;
* after ``max_pages_per_run`` loads it refuses (``PageBudgetExceeded``): spread
  big jobs over several runs instead of one long session.

``sleep``, ``clock`` and ``rng`` are injectable so tests run instantly and
deterministically.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable

from ..config import PacingSettings

log = logging.getLogger(__name__)

SleepFn = Callable[[float], Awaitable[Any]]


class PageBudgetExceeded(RuntimeError):
    """The per-run page budget (``PacingSettings.max_pages_per_run``) is used up."""

    def __init__(self, limit: int):
        super().__init__(
            f"page budget for this run used up ({limit} page loads); "
            "continue in a later run (raise pacing.max_pages_per_run to change the limit)"
        )
        self.limit = limit


class Pacer:
    """Decides how long to wait before each navigation, and stops the run at the budget."""

    def __init__(
        self,
        settings: PacingSettings,
        *,
        sleep: SleepFn = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        backoff_base_s: float = 5.0,
        backoff_cap_s: float = 300.0,
    ):
        self.settings = settings
        self.rng = rng or random.Random()
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.pages_loaded = 0
        self._sleep = sleep
        self._clock = clock
        self._last_nav: float | None = None
        self._break_pending = False

    @property
    def budget_left(self) -> int:
        return max(0, self.settings.max_pages_per_run - self.pages_loaded)

    def next_delay(self) -> float:
        """A fresh gap between navigations, in seconds."""
        lo, hi = sorted((max(0.0, self.settings.min_delay_s), max(0.0, self.settings.max_delay_s)))
        # Beta(2, 3): mode at 1/3, mean 0.4 of the range, long tail toward the max.
        return lo + (hi - lo) * self.rng.betavariate(2.0, 3.0)

    def long_pause(self) -> float:
        lo, hi = sorted((max(0.0, self.settings.long_pause_min_s), max(0.0, self.settings.long_pause_max_s)))
        return self.rng.uniform(lo, hi)

    def note_challenge(self) -> None:
        """Take a long break before the next navigation (e.g. after a challenge was solved)."""
        self._break_pending = True

    async def wait(self) -> float:
        """Call before every navigation. Sleeps as needed; returns the seconds slept.

        Raises ``PageBudgetExceeded`` instead of allowing another load past the budget.
        """
        if self.pages_loaded >= self.settings.max_pages_per_run:
            raise PageBudgetExceeded(self.settings.max_pages_per_run)
        slept = 0.0
        if self._last_nav is not None:
            every = self.settings.long_pause_every
            if self._break_pending or (every > 0 and self.pages_loaded % every == 0):
                self._break_pending = False
                target, what = self.long_pause(), "break"
            else:
                target, what = self.next_delay(), "delay"
            remaining = target - (self._clock() - self._last_nav)
            if remaining > 0:
                log.debug("pacing %s: sleeping %.1f s before page load %d", what, remaining, self.pages_loaded + 1)
                await self._sleep(remaining)
                slept = remaining
        self.pages_loaded += 1
        self._last_nav = self._clock()
        return slept

    def backoff(self, attempt: int) -> float:
        """Seconds to wait before retry number ``attempt`` (1-based): exponential, jittered, capped."""
        base = self.backoff_base_s * (2 ** max(0, attempt - 1))
        return min(self.backoff_cap_s, base * self.rng.uniform(0.8, 1.25))

    async def cooldown(self, seconds: float) -> None:
        """Sleep for ``seconds`` (backoff after an error, or a cooldown chosen by the caller)."""
        if seconds > 0:
            log.info("cooling down for %.0f s", seconds)
            await self._sleep(seconds)

    async def pause(self, lo: float, hi: float) -> float:
        """A short in-page pause (between scroll steps). Does not count as a page load."""
        seconds = lo + (hi - lo) * self.rng.betavariate(2.0, 3.0)
        if seconds > 0:
            await self._sleep(seconds)
        return seconds
