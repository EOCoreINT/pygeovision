"""Alert handlers for ``AlertManager``: a JSON-lines file, a Slack webhook, e-mail, and "only once per distinct alert".

A handler is any function that takes the alert entry ``{"timestamp", "severity", "message", "data"}``. Nothing here is part of the
package: copy what you need. Each handler is written so that a failure to deliver is logged instead of raised, because an exception in
a handler would stop the handlers after it and the code that called ``trigger``.
"""
from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger("alert_handlers")


def _plain(value):
    """Make an entry JSON-safe (drift reports can hold NumPy numbers and other odd types)."""
    return json.loads(json.dumps(value, default=str))


def jsonl_handler(path: str):
    """Append every alert as one JSON line. Survives restarts, unlike ``AlertManager.history``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def handle(entry: dict) -> None:
        try:
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_plain(entry)) + "\n")
        except OSError as exc:
            logger.error("could not write the alert to %s: %s", target, exc)

    return handle


def slack_handler(webhook_url: str, timeout: float = 5.0):
    """Post ``"<SEVERITY>: <message>"`` to a Slack incoming webhook (a URL you create in Slack)."""

    def handle(entry: dict) -> None:
        body = json.dumps({"text": f"{entry['severity'].upper()}: {entry['message']}"}).encode("utf-8")
        request = urllib.request.Request(webhook_url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                if reply.status >= 300:
                    logger.error("Slack answered %s", reply.status)
        except Exception as exc:                     # network down, bad URL, timeout
            logger.error("could not post the alert to Slack: %s", exc)

    return handle


def email_handler(host: str, sender: str, recipients: list[str], port: int = 587, username: str | None = None,
                  password: str | None = None, only: tuple[str, ...] = ("critical",)):
    """E-mail alerts of the listed severities only (a mailbox full of warnings is a mailbox nobody reads)."""

    def handle(entry: dict) -> None:
        if entry["severity"] not in only:
            return
        message = EmailMessage()
        message["Subject"] = f"[{entry['severity'].upper()}] {entry['message']}"
        message["From"], message["To"] = sender, ", ".join(recipients)
        message.set_content(json.dumps(_plain(entry), indent=2))
        try:
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.starttls()
                if username:
                    server.login(username, password or "")
                server.send_message(message)
        except Exception as exc:
            logger.error("could not e-mail the alert: %s", exc)

    return handle


class OnChange:
    """Wrap a handler so each distinct ``(severity, message)`` reaches it only once, until ``reset()``.

    A nightly check that finds the same major drift every night then alerts once, not every night. One drift report can raise
    several alerts, so a "same as the last alert" rule would fire every night as they alternate; remembering every key does not.
    Call ``reset()`` when the problem is fixed so a recurrence alerts again. (The set lives in memory: after a restart the
    first night alerts again, which is the safe direction.)
    """

    def __init__(self, handler):
        self._handler = handler
        self._seen: set[tuple[str, str]] = set()

    def __call__(self, entry: dict) -> None:
        key = (entry["severity"], entry["message"])
        if key not in self._seen:
            self._seen.add(key)
            self._handler(entry)

    def reset(self) -> None:
        self._seen.clear()
