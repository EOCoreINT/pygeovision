"""`download(bands=...)` must not report success for a scene that is missing a
band the caller explicitly asked for (#3).

`resolve_band_keys()` silently drops requested bands an item does not carry, so
`bands=["B03", "B11", "SCL"]` against an item that only carries B03 and B11
used to download two assets and return success=True; the caller found out only
by inspecting the output folder. The direct (pystac-fallback) path now records
per-band status, and a scene missing a requested band comes back success=False
with `partial=True` and the missing band named in `error`.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pygeovision.data.fetch import SatelliteFetcher, SearchResult


def _item(asset_keys):
    item = MagicMock(spec=SearchResult)
    item.id = "S2A_TEST_SCENE"
    item.provider = "planetary_computer"
    item.assets = {k: {"href": f"https://example.test/{k}.tif"} for k in asset_keys}
    item.satellite_data = None
    return item


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    f = SatelliteFetcher.__new__(SatelliteFetcher)
    f._credentials = {}

    monkeypatch.setattr(SatelliteFetcher, "_use_cli", lambda self: False)

    def fake_stream(self, session, url, out_path, provider, item_id, headers, bw):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x" * 128)
        return 128, None

    monkeypatch.setattr(SatelliteFetcher, "_stream_download_asset", fake_stream)
    monkeypatch.setattr(
        SatelliteFetcher,
        "_pick_stac_download_urls",
        lambda self, assets, provider: [
            (assets[k]["href"], k) for k in assets
        ],
    )
    return f


def test_missing_requested_band_fails_the_scene(tmp_path, monkeypatch, fetcher):
    results = fetcher.download(
        _item(["B03", "B11"]),
        output_dir=tmp_path,
        bands=["B03", "B11", "SCL"],
    )
    r = results[0]
    assert r.success is False
    assert r.partial is True
    assert r.asset_status == {
        "B03": "downloaded",
        "B11": "downloaded",
        "SCL": "missing",
    }
    assert "SCL" in r.error


def test_all_requested_bands_present_stays_success(tmp_path, monkeypatch, fetcher):
    results = fetcher.download(
        _item(["B03", "B11"]),
        output_dir=tmp_path,
        bands=["B03", "B11"],
    )
    r = results[0]
    assert r.success is True
    assert r.partial is False
    assert r.asset_status == {"B03": "downloaded", "B11": "downloaded"}


def test_no_bands_requested_leaves_accounting_alone(tmp_path, monkeypatch, fetcher):
    results = fetcher.download(_item(["B03", "B11"]), output_dir=tmp_path)
    r = results[0]
    assert r.success is True
    assert r.asset_status == {}
    assert r.partial is False


def test_plain_install_keeps_band_files_instead_of_failing(
    tmp_path, monkeypatch, fetcher
):
    import pygeovision.preprocess.core as core

    class _NoRasterio:
        def stack_bands(self, *a, **k):
            raise ImportError("rasterio is required for stacking")

    monkeypatch.setattr(core, "Preprocessor", _NoRasterio)
    results = fetcher.download(
        _item(["B03", "B11"]),
        output_dir=tmp_path,
        bands=["B03", "B11"],
    )
    r = results[0]
    assert r.success is True
    assert r.partial is False
    # The first band file is kept as the scene's output rather than a stacked
    # GeoTIFF that was never written.
    assert r.path is not None and Path(r.path).exists()
    assert "_B03" in Path(r.path).name
