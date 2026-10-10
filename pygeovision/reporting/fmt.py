"""Number formatting shared by the PDF, Word and Excel writers."""
from __future__ import annotations

import math
from typing import Any

NA = "n/a"

#: format code -> Excel number format (fractions are stored as 0-1, so "0.0%" shows 12.3 %)
EXCEL_FORMATS = {
    "frac": "0.0%", "ha": "#,##0.0", "km2": "#,##0.00", "int": "#,##0",
    "num1": "0.0", "num2": "0.00", "num3": "0.000", "num4": "0.0000",
    "c": "0.0", "m": "#,##0.0", "deg": "0.0", "text": "@", "bool": "@",
}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return isinstance(value, float) and not math.isfinite(value)
    except TypeError:
        return False


def _unneg(text: str) -> str:
    """'-0.00' -> '0.00' (a tiny negative that rounds to zero)."""
    if text.startswith("-") and not any(ch in "123456789" for ch in text):
        return text[1:]
    return text


def fmt_value(value: Any, code: str = "text") -> str:
    """Render ``value`` for people. ``code`` is one of EXCEL_FORMATS' keys."""
    if is_missing(value):
        return NA
    return _unneg(_fmt(value, code))


def _fmt(value: Any, code: str) -> str:
    if code == "frac":
        return f"{float(value) * 100:.1f} %"
    if code == "ha":
        return f"{float(value):,.1f} ha"
    if code == "km2":
        return f"{float(value):,.2f} km²"
    if code == "int":
        return f"{int(round(float(value))):,}"
    if code in ("num1", "num2", "num3", "num4"):
        return f"{float(value):,.{int(code[-1])}f}"
    if code == "c":
        return f"{float(value):.1f} °C"
    if code == "m":
        return f"{float(value):,.1f} m"
    if code == "deg":
        return f"{float(value):.1f}°"
    if code == "bool":
        return "yes" if value else "no"
    return str(value)


def guess_code(key: str, value: Any) -> str:
    """Pick a format for a statistic that has no explicit spec."""
    k = key.lower()
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        if any(t in k for t in ("fraction", "coverage", "pct_", "share")) and 0.0 <= value <= 1.0:
            return "frac"
        if k.endswith("_ha") or "_ha_" in k:
            return "ha"
        if k.endswith("_celsius") or k.endswith("_c"):
            return "c"
        return "num3"
    return "text"


def slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "x"
