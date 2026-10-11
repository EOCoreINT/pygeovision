"""doctor / status / datasets report what is true and exit non-zero when something is wrong."""
from __future__ import annotations

import json

from click.testing import CliRunner

from pygeovision.cli.main import cli


def run(*a):
    return CliRunner().invoke(cli, list(a))


def test_status_counts_open_providers_from_the_catalog():
    from pygeovision import PyGeoVision
    d = json.loads(run("status", "--json").output[run("status", "--json").output.index("{"):])
    c = PyGeoVision()
    assert d["pygeofetch"]["providers"] == len(c.list_providers())
    assert d["pygeofetch"]["open_providers"] == len(c.list_providers(open_only=True)) > 0     # used to say 0


def test_doctor_marks_every_check_and_counts_them():
    r = run("doctor")
    assert r.exit_code == 0, r.output
    assert "✗" not in r.output
    n = int(r.output.split("→ ")[1].split("/")[0]); m = int(r.output.split("→ ")[1].split("/")[1].split()[0])
    assert n == m >= 6                      # credentials + cache are counted, not shown as ✗ and ignored


def test_doctor_exits_nonzero_when_a_check_fails(monkeypatch):
    from pygeovision.data.fetch import SatelliteFetcher
    monkeypatch.setattr(SatelliteFetcher, "cache_stats", lambda self: (_ for _ in ()).throw(OSError("disk")))
    r = run("doctor")
    assert r.exit_code == 1 and "disk" in r.output


def test_datasets_unknown_name_is_a_clean_error_and_download_never_pretends():
    for cmd in (["datasets", "info", "NoSuch"], ["datasets", "similar", "NoSuch"], ["datasets", "download", "NoSuch"]):
        r = run(*cmd)
        assert r.exit_code == 1 and "Traceback" not in r.output and "not found" in r.output
    r = run("datasets", "download", "BigEarthNet")
    assert r.exit_code == 1 and "no automated download path" in r.output


def test_list_templates_finds_the_packaged_templates():
    assert "agriculture" in run("pipeline", "list-templates").output
