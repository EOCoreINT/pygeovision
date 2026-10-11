"""Alert management for drift detection and performance monitoring."""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class AlertManager:
    """Manage and route drift + performance alerts."""

    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels = channels or ["log"]
        self._handlers: dict[str, Callable] = {"log": self._log_alert}
        self._history: list[dict] = []

    def register_handler(self, name: str, fn: Callable) -> None:
        self._handlers[name] = fn

    def trigger(self, severity: str, message: str, data: dict | None = None) -> None:
        entry = {
            "timestamp": _utcnow(),
            "severity": severity,
            "message": message,
            "data": data or {},
        }
        self._history.append(entry)
        for channel in self.channels:
            handler = self._handlers.get(channel)
            if handler:
                handler(entry)
            else:
                logger.warning("[Alert] channel %r has no handler (register_handler); alert not delivered there: %s",
                               channel, message)

    def _log_alert(self, entry: dict) -> None:
        level = {"info": logger.info, "warning": logger.warning, "critical": logger.critical}
        fn = level.get(entry["severity"], logger.warning)
        fn("[Alert] %s: %s", entry["severity"].upper(), entry["message"])

    def check_drift_report(self, report: dict) -> None:
        if report.get("drift_detected"):
            for alert in report.get("performance_drift", {}).get("alerts", []):
                self.trigger("warning", f"Performance drift: {alert}", report)
            dd = report.get("data_drift", {})
            if dd.get("drift_level") == "major":
                self.trigger("critical", "Major data distribution shift detected!", dd)
            elif dd.get("drift_level") == "minor":
                self.trigger("warning", "Minor data drift detected.", dd)

    @property
    def history(self) -> list[dict]:
        return self._history
