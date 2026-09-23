"""Regression tests for the important email watcher's notification window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from server.services.gmail.importance_watcher import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    ImportantEmailWatcher,
)

EASTERN = timezone(timedelta(hours=-4))


def at(hour: int, minute: int, second: int) -> datetime:
    """A 2026-09-20 timestamp in the user's recorded timezone."""
    return datetime(2026, 9, 20, hour, minute, second, tzinfo=EASTERN)


@pytest.fixture
def watcher() -> ImportantEmailWatcher:
    return ImportantEmailWatcher()


@pytest.mark.parametrize(
    "arrived, previous_poll, now",
    [
        # The incident: delivered 01:04:23, missed by the 01:04:31 poll because Gmail's
        # search index had not caught up, then dropped by the 01:05:32 poll as 9s too old.
        (at(1, 4, 23), at(1, 4, 31), at(1, 5, 32)),
        # Normal cadence: polls sit 61s apart (60s sleep + ~1s of work), so a window
        # anchored to "now - 60s" loses the overshoot every cycle.
        (at(1, 5, 33), at(1, 5, 32), at(1, 6, 33)),
        # Slow poll: classifier round-trips stretch the gap further.
        (at(1, 5, 37), at(1, 5, 32), at(1, 6, 47)),
    ],
)
def test_mail_arriving_since_the_previous_poll_is_eligible(watcher, arrived, previous_poll, now):
    assert watcher._compute_cutoff(now, previous_poll) < arrived


def test_backlog_older_than_the_gap_is_still_suppressed(watcher):
    """The gate must still reject the 10-minute fetch backlog it exists to filter."""
    previous_poll = at(1, 5, 32)
    cutoff = watcher._compute_cutoff(previous_poll + timedelta(seconds=61), previous_poll)

    assert cutoff > previous_poll - timedelta(minutes=8)


def test_first_poll_falls_back_to_one_interval(watcher):
    """With no previous poll recorded, fall back to one interval plus grace."""
    now = at(1, 5, 32)
    cutoff = watcher._compute_cutoff(now, None)

    assert cutoff < now - timedelta(seconds=DEFAULT_POLL_INTERVAL_SECONDS)
