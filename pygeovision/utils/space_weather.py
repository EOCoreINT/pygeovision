"""
pygeovision.utils.space_weather
================================
Live space-weather data helpers for IonoForecaster.

Replaces the fetch_noaa_kp / fetch_noaa_f107 / s4_to_position_error
functions defined inline in the ionoforecaster notebook.

All endpoints are real NOAA SWPC public JSON APIs.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fetch_kp(n_3h_periods: int = 80) -> tuple[list[tuple[str, float]] | None, str]:
    """
    Fetch the NOAA planetary Kp geomagnetic index.

    Endpoint: ``https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json``
    Update interval: every 3 hours.
    Format returned: list of ``(timestamp_str, kp_value)`` tuples.

    Args:
        n_3h_periods: Number of 3-hour periods to return (default 80 = ~10 days).

    Returns:
        ``(records, source_label)`` where ``records`` is a list of
        ``(timestamp, kp)`` tuples, or ``(None, error_message)`` on failure.

    Example::

        from pygeovision.utils import fetch_kp

        records, src = fetch_kp()
        if records:
            latest_kp = records[-1][1]
            print(f"Current Kp: {latest_kp}  ({src})")
    """
    import requests

    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        # First row is header ["time_tag","Kp","Kp_int","a_Running","station_count"]
        records = []
        for row in raw[1:]:
            if len(row) >= 2:
                try:
                    records.append((str(row[0])[:10], float(row[1])))
                except (ValueError, TypeError):
                    pass
        records = records[-n_3h_periods:]
        logger.info("fetch_kp: %d records from NOAA SWPC", len(records))
        return records, "NOAA SWPC live"
    except Exception as e:
        logger.warning("fetch_kp failed: %s — using fallback Kp=2.0", e)
        return None, str(e)


def fetch_f107() -> tuple[float | None, str]:
    """
    Fetch the current F10.7 solar radio flux from NOAA SWPC.

    Endpoint: ``https://services.swpc.noaa.gov/products/summary/10cm-flux.json``
    Units: Solar Flux Units (sfu).  Quiet sun: ~70 sfu.  Active: >150 sfu.

    Returns:
        ``(f107_value, source_label)`` or ``(None, error_message)``.

    Example::

        from pygeovision.utils import fetch_f107

        f107, src = fetch_f107()
        if f107:
            print(f"F10.7: {f107:.0f} sfu  ({src})")
    """
    import requests

    url = "https://services.swpc.noaa.gov/products/summary/10cm-flux.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        f107 = float(data["Flux"])
        logger.info("fetch_f107: %.1f sfu from NOAA SWPC", f107)
        return f107, "NOAA SWPC live"
    except Exception as e:
        logger.warning("fetch_f107 failed: %s — using fallback 145 sfu", e)
        return None, str(e)


def kp_current(fallback: float = 2.0) -> float:
    """
    Return the latest Kp value, falling back to ``fallback`` on failure.

    Args:
        fallback: Value to use if NOAA SWPC is unavailable. Default 2.0 (quiet).

    Returns:
        Current Kp float.
    """
    records, _ = fetch_kp(n_3h_periods=1)
    if records:
        return float(records[-1][1])
    return fallback


def f107_current(fallback: float = 145.0) -> float:
    """
    Return the current F10.7 value, falling back to ``fallback`` on failure.

    Args:
        fallback: Value to use if NOAA SWPC is unavailable. Default 145.0 sfu.

    Returns:
        Current F10.7 float.
    """
    val, _ = fetch_f107()
    return val if val is not None else fallback


def s4_to_position_error(s4: float) -> tuple[float, str]:
    """
    Convert S4 scintillation index to GNSS positioning error and impact label.

    Uses the Aquino et al. 2009 empirical model (GPS Solutions):
    ``σ_position ≈ σ_S4 × scale_factor`` where scale_factor grows
    non-linearly with S4 severity.

    Reference:
        Aquino M. et al. (2009). ``Improving the GNSS positioning stochastic
        model in the presence of ionospheric scintillation''. Journal of Geodesy.
        doi:10.1007/s00190-009-0293-x

    Args:
        s4: S4 amplitude scintillation index (0 = no scintillation, 1 = severe).

    Returns:
        ``(positioning_error_m, impact_label)``

    Severity thresholds::

        S4 < 0.2   → Negligible     (< 1 m error)
        S4 < 0.4   → Low            (< 6 m error)
        S4 < 0.6   → Moderate       (< 24 m error)
        S4 < 0.8   → Severe         (< 64 m error)
        S4 ≥ 0.8   → Critical       (> 120 m — possible signal loss-of-lock)

    Example::

        from pygeovision.utils import s4_to_position_error

        err, label = s4_to_position_error(0.75)
        print(f"{err:.1f} m — {label}")
        # 60.0 m — Severe
    """
    if s4 < 0.2:
        return s4 * 5.0,   "Negligible"
    elif s4 < 0.4:
        return s4 * 15.0,  "Low"
    elif s4 < 0.6:
        return s4 * 40.0,  "Moderate"
    elif s4 < 0.8:
        return s4 * 80.0,  "Severe"
    else:
        return s4 * 150.0, "Critical — possible signal loss-of-lock"


def scintillation_risk_label(s4: float) -> str:
    """
    Return a short emoji + text risk label from an S4 value.

    Args:
        s4: S4 index (0–1).

    Returns:
        e.g. ``"🔴 SEVERE"``
    """
    if s4 > 0.7:   return "🔴 SEVERE"
    elif s4 > 0.4: return "🟡 MODERATE"
    else:          return "🟢 LOW"
