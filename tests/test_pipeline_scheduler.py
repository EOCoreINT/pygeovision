"""Cron matching and the once-per-minute guarantee of ``PipelineScheduler``."""
from datetime import datetime, timezone

import pytest

from pygeovision.pipelines.scheduler import PipelineScheduler, cron_matches


def at(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_daily_at_six_does_not_fire_every_hour():
    # the old fallback only compared the minute field, so "0 6 * * *" fired at every full hour
    assert cron_matches("0 6 * * *", at(2026, 10, 12, 6, 0))
    assert not cron_matches("0 6 * * *", at(2026, 10, 12, 7, 0))
    assert not cron_matches("0 6 * * *", at(2026, 10, 12, 6, 1))


def test_weekday_monday_is_1_and_sunday_is_0_or_7():
    monday = at(2026, 10, 12, 6, 0)           # a Monday
    sunday = at(2026, 10, 11, 6, 0)
    assert cron_matches("0 6 * * 1", monday) and not cron_matches("0 6 * * 1", sunday)
    assert cron_matches("0 6 * * 0", sunday) and cron_matches("0 6 * * 7", sunday)


def test_steps_ranges_and_lists():
    assert cron_matches("*/15 * * * *", at(2026, 1, 1, 3, 45))
    assert not cron_matches("*/15 * * * *", at(2026, 1, 1, 3, 40))
    assert cron_matches("0 9-17 * * *", at(2026, 1, 1, 17, 0)) and not cron_matches("0 9-17 * * *", at(2026, 1, 1, 18, 0))
    assert cron_matches("0 0 1,15 * *", at(2026, 3, 15, 0, 0))
    assert cron_matches("10-30/10 * * * *", at(2026, 1, 1, 0, 20))


def test_day_of_month_or_weekday_when_both_given():
    # classic cron: the 13th OR a Friday
    assert cron_matches("0 0 13 * 5", at(2026, 10, 13, 0, 0))     # Tuesday the 13th
    assert cron_matches("0 0 13 * 5", at(2026, 10, 16, 0, 0))     # Friday the 16th
    assert not cron_matches("0 0 13 * 5", at(2026, 10, 14, 0, 0))


@pytest.mark.parametrize("bad", ["", "* * * *", "61 * * * *", "* 24 * * *", "* * 0 * *", "*/0 * * * *", "a * * * *"])
def test_invalid_expressions_are_rejected(bad):
    with pytest.raises(ValueError):
        cron_matches(bad, at(2026, 1, 1, 0, 0))


def test_add_job_validates_the_expression():
    with pytest.raises(ValueError):
        PipelineScheduler().add_job("x", lambda: None, cron="not a cron")


def test_tick_runs_once_per_minute_and_survives_a_failing_job():
    calls = []
    s = PipelineScheduler()
    s.add_job("boom", lambda: 1 / 0, cron="* * * * *")
    s.add_job("ok", lambda: calls.append(1), cron="* * * * *")
    now = at(2026, 10, 12, 6, 0)
    s._tick(now)
    s._tick(now)                                  # same minute: must not run again
    assert calls == [1]
    s._tick(at(2026, 10, 12, 6, 1))
    assert calls == [1, 1]
    status = {j["name"]: j for j in s.status()}
    assert status["boom"]["errors"] == 2 and status["ok"]["n_runs"] == 2


def test_stop_returns_promptly():
    import time
    s = PipelineScheduler().start()
    t = time.time()
    s.stop()
    assert time.time() - t < 3
