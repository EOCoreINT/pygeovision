# Alerts

**Class:** `AlertManager` (`pygeovision.monitoring`) · **Python:** `client.monitoring.alert_manager(channels)`

## Why

A drift report in a JSON file nobody opens protects nobody. `AlertManager` turns a report into **alerts with a severity**
and sends each one to the channels you configure: the log by default, plus anything you register (a Slack message, an
e-mail, a ticket).

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

A handler that raises an exception propagates to the caller of `trigger`, so wrap network calls in your own `try/except`.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| No alert although drift happened | `report["drift_detected"]` false (including `incomplete`) | Check `incomplete` and `errors` |
| Alert only in the log | Only the `log` channel is configured | Register more handlers and list them in `channels` |
| `channel 'x' has no handler` warning | Channel named but not registered | `register_handler("x", fn)` |
| Handler failure stops later channels | An exception in a handler | Catch exceptions inside the handler |
| `history` empty after a restart | It is in memory | Write alerts to your own store in a handler |
| Too many alerts | Every check above the threshold alerts | Alert on a change of state yourself, or raise `recent_n`/thresholds |

```{seealso}
[Data drift](drift.md) · [Performance tracking](performance.md)
```
