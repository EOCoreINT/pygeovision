"""``pygeovision data pipeline|auth`` wiring to the real pygeofetch API (these used to raise AttributeError)."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from pygeovision.cli.main import cli

GOOD = """name: demo
schedule: "0 6 * * 1"
steps:
  - filter:
      expression: "data.cloud_cover < 20"
"""


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from pathlib import Path
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def run(*args):
    return CliRunner().invoke(cli, ["data", "pipeline", *map(str, args)])


@pytest.fixture()
def yml(tmp_path):
    p = tmp_path / "p.yaml"; p.write_text(GOOD)
    return p


def test_validate_accepts_good_and_rejects_bad(home, yml, tmp_path):
    assert run("validate", yml).exit_code == 0
    bad = tmp_path / "bad.yaml"; bad.write_text("name: x\nsteps:\n  - search\n")
    assert run("validate", bad).exit_code == 1
    noname = tmp_path / "n.yaml"; noname.write_text("steps: []\n")
    assert run("validate", noname).exit_code == 1
    empty = tmp_path / "e.yaml"; empty.write_text("name: e\nsteps: []\n")
    assert run("validate", empty).exit_code == 1


def test_schedule_list_roundtrip_uses_pygeofetch_registry(home, yml):
    assert run("list").output.strip().endswith("No scheduled pipelines.")
    r = run("schedule", yml, "--name", "nightly", "--cron", "0 2 * * *")
    assert r.exit_code == 0, r.output
    store = json.loads((home / ".pygeofetch" / "scheduled_pipelines.json").read_text())
    assert store["nightly"]["schedule"] == "0 2 * * *"           # same file `pygeofetch pipeline schedule` writes
    out = run("list").output
    assert "nightly" in out and "0 2 * * *" in out


def test_run_records_history_and_unknown_step_fails(home, yml):
    r = run("run", yml)
    assert r.exit_code == 0, r.output
    assert "filter: ok" in r.output
    h = run("history").output
    assert "demo" in h
    assert run("run", yml, "--step", "nosuch").exit_code != 0


def test_failed_step_exits_nonzero_and_is_logged(home, tmp_path):
    p = tmp_path / "f.yaml"
    p.write_text("name: fails\nsteps:\n  - no_such_step:\n      x: 1\n")
    r = run("run", p)
    assert r.exit_code == 1 and "failed" in r.output.lower()
    assert "✗" in run("history").output


def test_auth_test_open_provider_checks_endpoint_and_closed_needs_credentials(home, monkeypatch):
    import requests
    seen = {}

    class R:
        status_code = 200

    monkeypatch.setattr(requests, "get", lambda url, timeout: seen.setdefault("url", url) and R())
    r = CliRunner().invoke(cli, ["data", "auth", "test", "planetary_computer"])
    assert r.exit_code == 0 and "planetarycomputer" in seen["url"]
    r = CliRunner().invoke(cli, ["data", "auth", "test", "usgs"])         # needs credentials, none stored
    assert r.exit_code == 1


def test_auth_test_open_provider_unreachable_is_a_failure(home, monkeypatch):
    import requests

    def boom(url, timeout):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", boom)
    assert CliRunner().invoke(cli, ["data", "auth", "test", "planetary_computer"]).exit_code == 1
