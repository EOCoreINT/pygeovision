"""Cron-based scheduling of Python callables (for example a function that runs a pipeline).

Times are **UTC**. A job fires at most once per matching minute. The scheduler runs inside your Python process: it
stops when the process stops. For a schedule that must survive a reboot use the operating system's cron (see the
scheduling section of the YAML pipeline guide).
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# (lowest, highest) value of each cron field: minute, hour, day of month, month, day of week
_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


def _field_values(field: str, low: int, high: int) -> set[int]:
    """Expand one cron field (``*``, ``*/5``, ``1-5``, ``1-10/3``, ``1,15,30``) into the set of values it allows."""
    values: set[int] = set()
    for part in field.split(","):
        if not part:
            raise ValueError("empty item in cron field")
        rng, _, step_text = part.partition("/")
        step = int(step_text) if step_text else 1
        if step < 1:
            raise ValueError(f"step must be at least 1 in {part!r}")
        if rng == "*":
            start, stop = low, high
        elif "-" in rng:
            a, _, b = rng.partition("-")
            start, stop = int(a), int(b)
        else:
            start = int(rng)
            stop = high if step_text else start
        if start < low or stop > high or start > stop:
            raise ValueError(f"{part!r} is outside {low}-{high}")
        values.update(range(start, stop + 1, step))
    return values


def cron_matches(expression: str, when: datetime) -> bool:
    """Whether the five-field cron ``expression`` ("minute hour day-of-month month day-of-week") matches ``when``.

    Supports ``*``, lists, ranges and steps. Day of week is 0-6 from Sunday (7 also means Sunday). As in classic
    cron, when both day-of-month and day-of-week are restricted, either one matching is enough.

    Raises:
        ValueError: if the expression does not have five valid fields.
    """
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError(f"a cron expression has five fields (minute hour day month weekday), got {expression!r}")
    minute, hour, dom, month, dow = (_field_values(f, lo, hi) for f, (lo, hi) in zip(fields, _RANGES))
    if 7 in dow:
        dow = (dow - {7}) | {0}
    weekday = (when.weekday() + 1) % 7                     # Python: Monday=0 -> cron: Sunday=0
    dom_free, dow_free = fields[2] == "*", fields[4] == "*"
    if dom_free or dow_free:
        day_ok = when.day in dom and weekday in dow
    else:
        day_ok = when.day in dom or weekday in dow
    return when.minute in minute and when.hour in hour and when.month in month and day_ok


class PipelineScheduler:
    """Run callables on cron schedules, in a background thread.

    Example::

        scheduler = PipelineScheduler()
        scheduler.add_job("buildings", fn=run_my_pipeline, cron="0 6 * * 1")   # Mondays 06:00 UTC
        scheduler.start()
        ...
        scheduler.stop()
    """

    def __init__(self) -> None:
        self._jobs: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()

    def add_job(self, name: str, fn: Callable, cron: str,
                args: tuple = (), kwargs: dict | None = None) -> PipelineScheduler:
        """Add a scheduled job.

        Args:
            name: Job identifier.
            fn: Callable to execute.
            cron: Cron expression ("minute hour day month weekday"), validated now so a typo fails here and not
                silently at 06:00.

        Raises:
            ValueError: if ``cron`` is not a valid five-field expression.
        """
        cron_matches(cron, datetime.now(timezone.utc))          # validate
        self._jobs.append({
            "name": name, "fn": fn, "cron": cron,
            "args": args, "kwargs": kwargs or {},
            "last_run": None, "n_runs": 0, "errors": 0, "_last_minute": None,
        })
        logger.info("Scheduled: '%s' @ '%s' (UTC)", name, cron)
        return self

    def _parse_cron(self, cron: str, now: datetime | None = None) -> bool:
        """Whether ``cron`` matches the current minute (UTC)."""
        return cron_matches(cron, now or datetime.now(timezone.utc))

    def _tick(self, now: datetime | None = None) -> None:
        """Run every job whose schedule matches this minute and has not already run in it."""
        now = now or datetime.now(timezone.utc)
        minute_key = int(now.timestamp() // 60)
        for job in self._jobs:
            if job["_last_minute"] == minute_key or not self._parse_cron(job["cron"], now):
                continue
            job["_last_minute"] = minute_key
            try:
                job["fn"](*job["args"], **job["kwargs"])
                job["n_runs"] += 1
                job["last_run"] = time.time()
            except Exception as exc:  # noqa: BLE001 - one failing job must not stop the others
                job["errors"] += 1
                logger.error("Job '%s' failed: %s", job["name"], exc)

    def _run_loop(self) -> None:
        while self._running:
            self._tick()
            self._wake.wait(timeout=60 - time.time() % 60 + 0.5)     # to just after the next minute starts
            self._wake.clear()

    def start(self) -> PipelineScheduler:
        """Start the scheduler in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started with %d jobs", len(self._jobs))
        return self

    def stop(self) -> None:
        """Stop the background thread (returns promptly)."""
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def status(self) -> list[dict]:
        """One dict per job: name, cron, number of runs, number of errors and the time of the last run."""
        return [{"name": j["name"], "cron": j["cron"],
                 "n_runs": j["n_runs"], "errors": j["errors"],
                 "last_run": j["last_run"]} for j in self._jobs]

    def run_now(self, name: str) -> Any:
        """Run a scheduled job immediately and return what it returns."""
        for job in self._jobs:
            if job["name"] == name:
                return job["fn"](*job["args"], **job["kwargs"])
        raise KeyError(f"Job '{name}' not found")
