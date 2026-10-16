# Alerts

**Class:** `AlertManager` (`pygeovision.monitoring`) · **Python:** `client.monitoring.alert_manager(channels)`

## Why

A drift report in a JSON file nobody opens protects nobody. `AlertManager` turns a report into **alerts with a severity**
and sends each one to the channels you configure: the log by default, plus anything you register (a Slack message, an
e-mail, a ticket).

```{figure} ../_static/img/flow-alerts.svg
:alt: From drift report to a person: check_drift_report turns a report into alerts with a severity; trigger stores each in history and calls the handler of every channel; handlers write a file, post to chat or send e-mail
:width: 100%
:class: pgv-figure

The manager decides severity and routes; what a channel *does* is a function you supply.
```

## How to use

```python
from pygeovision.monitoring import AlertManager

am = AlertManager(channels=["log", "slack"])
am.register_handler("slack", lambda entry: post_to_slack(f"{entry['severity']}: {entry['message']}"))

am.check_drift_report(report)               # a report from DriftDetector.check
am.trigger("critical", "Nightly job failed", {"job": "tiles"})
print(am.history)
```

| Member | Meaning |
|---|---|
| `AlertManager(channels=None)` | Names of the channels to use. Default `["log"]` |
| `register_handler(name, fn)` | `fn(entry)` is called for every alert on channel `name` |
| `trigger(severity, message, data=None)` | Creates an entry `{timestamp, severity, message, data}`, stores it in `history`, and calls the handler of every channel |
| `check_drift_report(report)` | Creates alerts from a `DriftDetector.check` report |
| `history` | All entries so far (in memory only) |

### What `check_drift_report` raises

It does nothing when `report["drift_detected"]` is false. Otherwise, one alert per entry:

| Condition | Severity | Message |
|---|---|---|
| Each text in `performance_drift["alerts"]` | `warning` | `Performance drift: <text>` |
| `data_drift["drift_level"] == "major"` | `critical` | `Major data distribution shift detected!` |
| `data_drift["drift_level"] == "minor"` | `warning` | `Minor data drift detected.` |

Real output for a report with a performance alert and major data drift:

```text
WARNING pygeovision.monitoring.alerts: [Alert] WARNING: Performance drift: mean_iou dropped by 8.0% (threshold: 5.0%)
CRITICAL pygeovision.monitoring.alerts: [Alert] CRITICAL: Major data distribution shift detected!
```

The `log` channel maps severity to log level: `info`, `warning`, `critical`; anything else is logged as a warning.

**It does not alert on an incomplete report.** If the data comparison failed (`report["incomplete"]` is true),
`drift_detected` is false and nothing is sent. Handle it yourself:

```python
if report["incomplete"]:
    am.trigger("critical", "Drift check could not run", {"errors": report["errors"]})
else:
    am.check_drift_report(report)
```

### Channels without a handler

A channel name with no registered handler (for example `"email"` without `register_handler("email", ...)`) used to drop
the alert silently. It now logs a warning:
`channel 'email' has no handler (register_handler); alert not delivered there: hello`. The alert is still stored in `history`.
There are no built-in e-mail or Slack senders: you supply the function.

A handler that raises an exception propagates to the caller of `trigger`, and the channels after it in the list are **skipped** (the alert is
already in `history`). So wrap network calls in your own `try/except`, as the handlers below do.

## Ready-made handlers you can copy

These four are a documented example ({download}`alert_handlers.py <../_examples/monitoring/alert_handlers.py>`), not part of the package. They were
run here with the real `AlertManager`: the JSON-lines file, the Slack post to a local HTTP server, e-mail through a stand-in SMTP object, and the
once-only wrapper (`tests/test_docs_alert_handlers.py`). **Not run:** a real Slack workspace or mail server.

```{literalinclude} ../_examples/monitoring/alert_handlers.py
:language: python
:caption: alert_handlers.py
:linenos:
```

```python
from pygeovision.monitoring import AlertManager
from alert_handlers import jsonl_handler, slack_handler, email_handler, OnChange

am = AlertManager(channels=["log", "file", "slack", "mail"])
am.register_handler("file", jsonl_handler("monitoring/alerts.jsonl"))
am.register_handler("slack", OnChange(slack_handler("https://hooks.slack.com/services/...")))
am.register_handler("mail", email_handler("smtp.example.org", "pgv@example.org", ["team@example.org"], username="pgv", password=PASSWORD))
```

| Handler | What it does | Why it is written this way |
|---|---|---|
| `jsonl_handler(path)` | Appends one JSON object per line | Survives a restart (`history` does not). Values that are not JSON (NumPy numbers) are turned into text instead of failing |
| `slack_handler(url, timeout=5)` | Posts `"CRITICAL: <message>"` to a Slack incoming webhook | A dead network or a bad URL is logged, not raised, so the other channels still run |
| `email_handler(host, sender, recipients, ..., only=("critical",))` | Sends an e-mail for the severities you list | A mailbox of warnings is a mailbox nobody reads; the default is critical only |
| `OnChange(handler)` | Wraps any handler so each distinct `(severity, message)` reaches it once until `reset()` | One drift report raises several alerts, so "same as last time" would fire every night as they alternate; remembering each key does not. The memory is in the process, so after a restart the first alert goes out again |

Keep secrets (the webhook URL, the SMTP password) in environment variables or a secrets store, not in the script.

## A nightly job, end to end

```python
import logging, sys
from pygeovision.monitoring import AlertManager, DriftDetector
from alert_handlers import jsonl_handler, slack_handler, OnChange

am = AlertManager(channels=["log", "file", "slack"])
am.register_handler("file", jsonl_handler("monitoring/alerts.jsonl"))
am.register_handler("slack", OnChange(slack_handler(WEBHOOK)))

det = DriftDetector(storage_dir="./monitoring/")
report = det.check(todays_tifs, current_metrics=None)

if report["incomplete"]:
    am.trigger("critical", "Drift check could not run", {"errors": report["errors"]})
    sys.exit(1)                                  # the scheduler sees the failure too
am.check_drift_report(report)
```

Two details matter. The incomplete branch is **yours** to write (the manager sends nothing for an incomplete report). And `check_drift_report` hands the
*whole report* to the performance alerts as `data`, so the entry in a file can be large; if that is a problem, trigger your own shorter alerts from the fields you need.

## Severity

`trigger` accepts any text as the severity. The `log` channel understands three: `info`, `warning` and `critical` (anything else is logged as a warning). Your own handlers
can use any scheme (`page` for pager-worthy, say), but keep one: a handler like `email_handler` compares the text exactly.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| No alert although drift happened | `report["drift_detected"]` false (including `incomplete`) | Check `incomplete` and `errors` |
| Alert only in the log | Only the `log` channel is configured | Register more handlers and list them in `channels` |
| `channel 'x' has no handler` warning | Channel named but not registered | `register_handler("x", fn)` |
| Handler failure stops later channels | An exception in a handler | Catch exceptions inside the handler |
| `history` empty after a restart | It is in memory | Write alerts to your own store in a handler |
| Too many alerts | Every check above the threshold alerts | Wrap the handler in `OnChange`, or raise the thresholds |
| Later channels never fire | An earlier handler raised | Catch exceptions inside handlers (the examples do) |
| The file has no alerts after a restart you did not expect | `history` is in memory only | Use `jsonl_handler` |

## Related

- [Monitoring overview](index.md), [Data drift](drift.md), [Performance tracking](performance.md), [Site tracking](site-tracking.md)
- [`pygeovision.monitoring` in the Python API](../reference/api/monitoring.md)
