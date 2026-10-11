"""Date-window helpers shared by the pipelines and the CLI."""
from __future__ import annotations

import calendar


def date_window(value: str) -> tuple[str, str]:
    """``YYYY-MM`` -> (first day, last day) of that month; ``YYYY-MM-DD`` -> (day, day);
    ``YYYY-MM-DD/YYYY-MM-DD`` -> that range (a flood event is a matter of days, not a month).

    The window used to be hard-coded as ``YYYY-MM-01`` to ``YYYY-MM-28``, so the
    last 1 to 3 days of every month were never searched (the 29th-31st of a
    31-day month, which also hid the freshest scene of a "latest" request).
    """
    value = str(value).strip()
    if "/" in value:
        start, _, end = (part.strip() for part in value.partition("/"))
        if len(start) != 10 or len(end) != 10 or start > end:
            raise ValueError(f"date range {value!r} must be 'YYYY-MM-DD/YYYY-MM-DD' with start <= end")
        return start, end
    if len(value) == 7:
        year, month = int(value[:4]), int(value[5:7])
        last = calendar.monthrange(year, month)[1]
        return f"{value}-01", f"{value}-{last:02d}"
    return value, value