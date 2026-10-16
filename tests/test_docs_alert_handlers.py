# ruff: noqa: E702, N802, C408, E731
"""docs/_examples/monitoring/alert_handlers.py with the real AlertManager (local HTTP server, patched SMTP)."""
import importlib.util
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from pygeovision.monitoring import AlertManager

spec = importlib.util.spec_from_file_location("alert_handlers", Path(__file__).resolve().parents[1] / "docs/_examples/monitoring/alert_handlers.py")
ah = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ah)

REPORT = {"drift_detected": True, "incomplete": False,
          "performance_drift": {"alerts": ["mean_iou dropped by 8.0% (threshold: 5.0%)"]},
          "data_drift": {"drift_level": "major", "psi": np.float64(2.4)}}


def test_jsonl_handler_writes_one_line_per_alert_and_copes_with_numpy_values(tmp_path):
    am = AlertManager(channels=["log", "file"])
    am.register_handler("file", ah.jsonl_handler(str(tmp_path / "out" / "alerts.jsonl")))
    am.check_drift_report(REPORT)
    lines = (tmp_path / "out" / "alerts.jsonl").read_text().splitlines()
    assert len(lines) == 2 == len(am.history)
    first, second = map(json.loads, lines)
    assert first["severity"] == "warning" and second["severity"] == "critical" and second["data"]["psi"] == 2.4


def test_slack_handler_posts_json_text_and_never_raises_on_failure(caplog):
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        def log_message(self, *a): pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        am = AlertManager(channels=["slack"]); am.register_handler("slack", ah.slack_handler(f"http://127.0.0.1:{srv.server_port}/hook"))
        am.trigger("critical", "Nightly job failed")
    finally:
        srv.shutdown()
    assert got == [{"text": "CRITICAL: Nightly job failed"}]
    dead = AlertManager(channels=["slack"]); dead.register_handler("slack", ah.slack_handler("http://127.0.0.1:9/hook", timeout=1))
    with caplog.at_level(logging.ERROR):
        dead.trigger("critical", "x")                                       # must not raise
    assert "could not post the alert to Slack" in caplog.text and len(dead.history) == 1


def test_email_handler_sends_only_the_chosen_severities(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None): sent.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): sent.append(("tls",))
        def login(self, u, p): sent.append(("login", u))
        def send_message(self, m): sent.append(("msg", m["Subject"], m["To"]))

    monkeypatch.setattr(ah.smtplib, "SMTP", FakeSMTP)
    am = AlertManager(channels=["mail"])
    am.register_handler("mail", ah.email_handler("smtp.example.org", "pgv@example.org", ["a@x.org", "b@x.org"], username="u", password="p"))
    am.trigger("warning", "minor"); am.trigger("critical", "Major data distribution shift detected!")
    assert ("msg", "[CRITICAL] Major data distribution shift detected!", "a@x.org, b@x.org") in sent
    assert sum(1 for s in sent if s[0] == "msg") == 1 and ("login", "u") in sent


def test_onchange_alerts_once_until_reset():
    seen = []
    am = AlertManager(channels=["x"]); wrapped = ah.OnChange(seen.append); am.register_handler("x", wrapped)
    for _ in range(3):
        am.check_drift_report(REPORT)
    assert len(seen) == 2 and len(am.history) == 6          # the two distinct alerts reached the handler once each
    wrapped.reset(); am.check_drift_report(REPORT)
    assert len(seen) == 4
