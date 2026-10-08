"""Navigation pacing, breaks, page budget and backoff (deterministic, no real sleeping)."""

import random
import statistics

import pytest

from ebay_sold.config import PacingSettings
from ebay_sold.fetch.pacing import Pacer, PageBudgetExceeded


class FakeTime:
    """A clock that only moves when the pacer sleeps (or the test advances it)."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make(seed=1, **overrides):
    defaults = dict(min_delay_s=6.0, max_delay_s=18.0, long_pause_every=0, long_pause_min_s=45.0,
                    long_pause_max_s=120.0, max_pages_per_run=1000)
    settings = PacingSettings(**{**defaults, **overrides})
    t = FakeTime()
    return Pacer(settings, sleep=t.sleep, clock=t.clock, rng=random.Random(seed)), t


async def test_first_load_is_immediate_then_delays_stay_in_range():
    pacer, t = make()
    assert await pacer.wait() == 0.0
    assert t.sleeps == [] and pacer.pages_loaded == 1
    for _ in range(300):
        await pacer.wait()
    assert len(t.sleeps) == 300
    assert all(6.0 <= s <= 18.0 for s in t.sleeps)
    # skewed toward the low-middle, with some long gaps
    assert statistics.median(t.sleeps) < 12.0
    assert max(t.sleeps) > 14.0
    assert len({round(s, 3) for s in t.sleeps}) > 250  # no fixed rhythm


async def test_time_spent_on_the_page_counts_toward_the_delay():
    pacer, t = make()
    await pacer.wait()
    t.now += 4.0  # loading + scrolling took 4 s
    slept = await pacer.wait()
    assert 2.0 <= slept <= 14.0
    t.now += 60.0  # a long read: no extra wait needed
    assert await pacer.wait() == 0.0
    assert pacer.pages_loaded == 3


async def test_long_pause_every_n_loads():
    pacer, t = make(long_pause_every=3)
    for _ in range(7):
        await pacer.wait()
    # loads 2,3 short; before load 4 a break; 5,6 short; before load 7 a break
    assert len(t.sleeps) == 6
    breaks = [i for i, s in enumerate(t.sleeps) if s >= 45.0]
    assert breaks == [2, 5]
    assert all(45.0 <= t.sleeps[i] <= 120.0 for i in breaks)


async def test_long_pause_disabled_with_zero():
    pacer, t = make(long_pause_every=0)
    for _ in range(50):
        await pacer.wait()
    assert max(t.sleeps) <= 18.0


async def test_note_challenge_forces_a_break():
    pacer, t = make()
    await pacer.wait()
    pacer.note_challenge()
    await pacer.wait()
    assert 45.0 <= t.sleeps[-1] <= 120.0
    await pacer.wait()
    assert t.sleeps[-1] <= 18.0


async def test_page_budget():
    pacer, t = make(max_pages_per_run=3)
    assert pacer.budget_left == 3
    for left in (2, 1, 0):
        await pacer.wait()
        assert pacer.budget_left == left
    with pytest.raises(PageBudgetExceeded) as exc:
        await pacer.wait()
    assert exc.value.limit == 3
    assert pacer.pages_loaded == 3
    assert len(t.sleeps) == 2  # no sleeping for a load that never happens


async def test_zero_delay_settings_never_sleep():
    pacer, t = make(min_delay_s=0, max_delay_s=0)
    for _ in range(5):
        await pacer.wait()
    assert t.sleeps == []


def test_backoff_grows_with_jitter_and_is_capped():
    pacer, _ = make()
    values = [pacer.backoff(n) for n in range(1, 12)]
    assert 4.0 <= values[0] <= 6.25
    assert 8.0 <= values[1] <= 12.5
    assert 16.0 <= values[2] <= 25.0
    assert all(b > a for a, b in zip(values[:5], values[1:6], strict=True))
    assert max(values) <= 300.0
    assert values[-1] == 300.0


async def test_cooldown_and_pause():
    pacer, t = make()
    await pacer.cooldown(900)
    assert t.sleeps == [900]
    await pacer.cooldown(0)
    assert t.sleeps == [900]
    p = await pacer.pause(0.2, 0.6)
    assert 0.2 <= p <= 0.6 and t.sleeps[-1] == p
    assert pacer.pages_loaded == 0  # neither counts as a page load


async def test_same_seed_same_schedule():
    a, ta = make(seed=42, long_pause_every=4)
    b, tb = make(seed=42, long_pause_every=4)
    for _ in range(20):
        await a.wait()
        await b.wait()
    assert ta.sleeps == tb.sleeps
    assert [a.backoff(3) for _ in range(3)] == [b.backoff(3) for _ in range(3)]


async def test_reversed_min_max_is_tolerated():
    pacer, t = make(min_delay_s=10, max_delay_s=2)
    await pacer.wait()
    await pacer.wait()
    assert 2 <= t.sleeps[0] <= 10
