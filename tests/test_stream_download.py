"""Acquiring imagery: stream the area of interest, or download, with a fallback that lives in PyGeoVision.

pygeofetch is used as it is (``engine.stream`` and ``engine.download``). ``pygeovision.data._stream.fetch_scenes`` is the one
place that decides: ``stream=True`` (or ``None`` with an area known) streams and downloads every scene the stream could not
deliver; ``stream=False`` always downloads. Everything runs offline: the "remote" rasters are local GeoTIFFs, the real
``PyGeoFetch.stream`` reads them, and only the engine's *full* download is replaced by a recorder.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeofetch.core.engine import PyGeoFetch  # noqa: E402
from pygeofetch.models.download_task import (  # noqa: E402
    DownloadOptions,
    DownloadResult as PgfResult,
    DownloadStatus,
)
from pygeofetch.models.satellite_data import SatelliteAsset, SatelliteData  # noqa: E402
from pygeofetch.streaming import scene as scene_mod  # noqa: E402

from pygeovision.data import _stream  # noqa: E402
from pygeovision.data._stream import (  # noqa: E402
    fetch_scenes,
    reproject_file_inplace,
    streaming_enabled,
    want_stream,
)
from pygeovision.data.fetch import DownloadResult, SatelliteFetcher, SearchResult  # noqa: E402

BBOX = (10.02, 50.02, 10.06, 50.06)  # inside the 0.1 deg test rasters


def geotiff(path: Path, crs="EPSG:4326", dtype="uint16", value=None) -> str:
    """100 x 100 px raster over lon 10.0-10.1, lat 50.0-50.1 (or the same size in metres for a projected CRS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        data = (np.arange(100 * 100) % 200).astype(dtype).reshape(1, 100, 100)
    else:
        data = np.full((1, 100, 100), value, dtype=dtype)
    geographic = crs == "EPSG:4326"
    tf = from_origin(10.0, 50.1, 0.001, 0.001) if geographic else from_origin(800000, 620000, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=1, dtype=dtype, crs=crs, transform=tf
    ) as dst:
        dst.write(data)
    return str(path)


def scene(tmp: Path, sid: str, provider="planetary_computer") -> SatelliteData:
    assets = {
        key: SatelliteAsset(
            key=key,
            href=geotiff(tmp / "remote" / f"{sid}_{key}.tif", dtype=dt, value=val),
            media_type="image/tiff",
            roles=["data"],
        )
        for key, dt, val in (("B04", "uint16", None), ("SCL", "uint8", 4))
    }
    return SatelliteData(id=sid, provider=provider, assets=assets)


def archive_scene(sid="S2_ARCHIVE") -> SatelliteData:
    """A scene whose only asset is a ``.zip`` archive: there is no COG in the stack to stream."""
    return SatelliteData(
        id=sid,
        provider="copernicus",
        assets={
            "PRODUCT": SatelliteAsset(
                key="PRODUCT", href=f"https://example.invalid/{sid}.SAFE.zip", roles=["data"]
            )
        },
    )


def search_result(sd: SatelliteData) -> SearchResult:
    return SearchResult(
        id=sd.id,
        provider=sd.provider,
        satellite="sentinel-2",
        datetime="2026-06-30T10:30:00Z",
        cloud_cover=5.0,
        bbox=BBOX,
        satellite_data=sd,
        assets={k: a.href for k, a in sd.assets.items()},
    )


class _Provider:
    @staticmethod
    def sign_href(href, data):
        return href


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(scene_mod, "_get_provider", lambda data, provider=None: _Provider())
    monkeypatch.delenv("PYGEOVISION_STREAM", raising=False)


class FullDownloads:
    """Replaces the engine's full ``download``: records what was asked for, writes a marker file."""

    def __init__(self, fail=(), raises=False):
        self.calls: list[tuple[list[str], DownloadOptions]] = []
        self.fail, self.raises = set(fail), raises

    def __call__(self, items, destination=None, options=None, **kw):
        self.calls.append(([i.id for i in items], options))
        if self.raises:
            raise RuntimeError("network down")
        out = []
        for i in items:
            if i.id in self.fail:
                out.append(
                    PgfResult(status=DownloadStatus.FAILED, data_id=i.id, provider=i.provider, error="HTTP 503")
                )
                continue
            p = Path(destination) / i.provider / f"{i.id}_FULL.tif"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"whole scene")
            out.append(
                PgfResult(
                    status=DownloadStatus.COMPLETED,
                    data_id=i.id,
                    provider=i.provider,
                    output_path=p,
                    output_paths=[p],
                    bytes_downloaded=11,
                )
            )
        return out


def engine(full: FullDownloads) -> PyGeoFetch:
    """The real pygeofetch engine (its real ``stream``), with only the full download replaced."""
    eng = PyGeoFetch.__new__(PyGeoFetch)
    eng.download = full  # type: ignore[method-assign]
    return eng


def fetcher(full: FullDownloads) -> SatelliteFetcher:
    f = SatelliteFetcher.__new__(SatelliteFetcher)
    f._engine = engine(full)
    f._credentials = {}
    f._use_cli = lambda: False  # type: ignore[method-assign]
    return f


def spy_on_stream(eng: PyGeoFetch, monkeypatch):
    calls = []
    real = eng.stream
    monkeypatch.setattr(eng, "stream", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


# ---------------------------------------------------------------------------------------- the user's choice
@pytest.mark.parametrize(
    "stream,bbox,env,expected",
    [
        (None, BBOX, None, True),  # automatic: stream when an area is known
        (None, None, None, False),  # no area: nothing to read
        (True, BBOX, None, True),
        (True, None, None, False),  # asked to stream but there is no area: download, no error
        (False, BBOX, None, False),  # asked for whole scenes
        (None, BBOX, "0", False),  # PYGEOVISION_STREAM=0 turns the automatic choice off
        (True, BBOX, "0", True),  # an explicit choice wins over the environment default
        (False, BBOX, "1", False),
    ],
)
def test_who_streams(stream, bbox, env, expected, monkeypatch):
    if env is not None:
        monkeypatch.setenv("PYGEOVISION_STREAM", env)
    assert want_stream(stream, bbox) is expected


def test_the_environment_switch_values(monkeypatch):
    assert streaming_enabled()
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("PYGEOVISION_STREAM", off)
        assert not streaming_enabled()


def test_stream_false_never_touches_the_stream_layer(tmp_path, monkeypatch):
    full = FullDownloads()
    eng = engine(full)
    streamed = spy_on_stream(eng, monkeypatch)
    res = fetch_scenes(eng, [scene(tmp_path, "S2_A")], tmp_path / "o", DownloadOptions(), stream_bbox=BBOX, stream=False)
    assert streamed == [] and [c[0] for c in full.calls] == [["S2_A"]]
    assert res[0].acquisition == "download" and res[0].stream_error == ""


def test_without_an_area_scenes_are_downloaded_even_when_stream_is_true(tmp_path, monkeypatch):
    full = FullDownloads()
    eng = engine(full)
    streamed = spy_on_stream(eng, monkeypatch)
    res = fetch_scenes(eng, [scene(tmp_path, "S2_A")], tmp_path / "o", DownloadOptions(), stream=True)
    assert streamed == [] and res[0].acquisition == "download" and res[0].result.success


# --------------------------------------------------------------------------------- stream, with the fallback
def test_a_cog_scene_is_streamed_as_an_aoi_window_without_any_full_download(tmp_path):
    full = FullDownloads()
    res = fetch_scenes(
        engine(full), [scene(tmp_path, "S2_A")], tmp_path / "out", DownloadOptions(), stream_bbox=BBOX, stream=True
    )
    assert full.calls == []
    assert res[0].result.success and res[0].acquisition == "stream" and res[0].stream_error == ""
    with rasterio.open(res[0].result.output_paths[0]) as src:
        assert src.width < 100 and src.height < 100


def test_a_stack_without_a_cog_is_downloaded_not_raised_and_order_is_kept(tmp_path):
    full = FullDownloads()
    opts = DownloadOptions(retry_attempts=9, verify_checksum=False)
    res = fetch_scenes(
        engine(full),
        [scene(tmp_path, "S2_A"), archive_scene("S2_B"), scene(tmp_path, "S2_C")],
        tmp_path / "out",
        opts,
        stream_bbox=BBOX,
        stream=True,
    )
    assert [r.acquisition for r in res] == ["stream", "download", "stream"]
    assert all(r.result.success for r in res)
    assert [c[0] for c in full.calls] == [["S2_B"]]  # only the scene that could not be streamed
    assert full.calls[0][1].retry_attempts == 9  # with the caller's download options
    assert "archive" in res[1].stream_error


def test_a_half_streamed_scene_is_downloaded_and_its_aoi_files_are_removed(tmp_path):
    full = FullDownloads()
    sd = scene(tmp_path, "S2_P")
    Path(sd.assets["SCL"].href).unlink()  # one band cannot be read: B04 streams, SCL fails
    out = tmp_path / "out"
    res = fetch_scenes(engine(full), [sd], out, DownloadOptions(), stream_bbox=BBOX, stream=True)
    assert res[0].acquisition == "download" and res[0].result.success
    assert "SCL" in res[0].stream_error
    assert full.calls and not list(out.glob("*/S2_P_B04.tif"))  # no AOI/whole mix left behind


def test_a_failing_stream_layer_means_download_for_every_scene(tmp_path):
    class Boom:
        def __init__(self, full):
            self.download = full

        def stream(self, *a, **k):
            raise ImportError("no rasterio")

    full = FullDownloads()
    res = fetch_scenes(
        Boom(full), [scene(tmp_path, "S2_A"), scene(tmp_path, "S2_B")], tmp_path / "o", DownloadOptions(), stream_bbox=BBOX, stream=True
    )
    assert [c[0] for c in full.calls] == [["S2_A", "S2_B"]]
    assert all(r.acquisition == "download" and r.result.success for r in res)
    assert "no rasterio" in res[0].stream_error


def test_an_engine_without_stream_downloads(tmp_path):
    full = FullDownloads()
    eng = SimpleNamespace(download=full)
    res = fetch_scenes(eng, [scene(tmp_path, "S2_A")], tmp_path / "o", DownloadOptions(), stream_bbox=BBOX, stream=True)
    assert res[0].acquisition == "download" and res[0].result.success and "no stream" in res[0].stream_error


def test_when_the_download_fails_too_both_reasons_are_reported(tmp_path):
    res = fetch_scenes(
        engine(FullDownloads(fail={"S2_B"})), [archive_scene("S2_B")], tmp_path / "o", DownloadOptions(), stream_bbox=BBOX, stream=True
    )
    r = res[0].result
    assert r.status == DownloadStatus.FAILED
    assert "stream failed" in r.error and "HTTP 503" in r.error


def test_a_download_that_raises_becomes_a_failed_result_for_that_scene_only(tmp_path):
    res = fetch_scenes(
        engine(FullDownloads(raises=True)),
        [archive_scene("S2_B"), scene(tmp_path, "S2_A")],
        tmp_path / "o",
        DownloadOptions(),
        stream_bbox=BBOX,
        stream=True,
    )
    assert res[0].result.status == DownloadStatus.FAILED and "network down" in res[0].result.error
    assert res[1].result.success and res[1].acquisition == "stream"


def test_empty_input_is_a_noop(tmp_path):
    assert fetch_scenes(engine(FullDownloads()), [], tmp_path, DownloadOptions(), stream_bbox=BBOX) == []


# ------------------------------------------------------------------------------------------------- reprojection
def test_streamed_files_get_the_reproject_step_like_a_full_download(tmp_path):
    full = FullDownloads()
    res = fetch_scenes(
        engine(full),
        [scene(tmp_path, "S2_A")],
        tmp_path / "out",
        DownloadOptions(),
        stream_bbox=BBOX,
        stream=True,
        post_process=["reproject:EPSG:3857"],
    )
    assert res[0].acquisition == "stream" and full.calls == []
    for pth in res[0].result.output_paths:
        with rasterio.open(pth) as src:
            assert str(src.crs).upper() == "EPSG:3857"


def test_a_streamed_file_that_cannot_be_reprojected_is_downloaded_instead(tmp_path):
    full = FullDownloads()
    res = fetch_scenes(
        engine(full),
        [scene(tmp_path, "S2_A")],
        tmp_path / "out",
        DownloadOptions(),
        stream_bbox=BBOX,
        stream=True,
        post_process=["reproject:EPSG:99999999"],
    )
    assert res[0].acquisition == "download" and "reproject" in res[0].stream_error
    assert full.calls and res[0].result.success


def test_reproject_changes_crs_but_not_categorical_values(tmp_path):
    p = Path(geotiff(tmp_path / "scl.tif", crs="EPSG:32630", dtype="uint8", value=4))
    reproject_file_inplace(p, "EPSG:4326")
    with rasterio.open(p) as src:
        assert str(src.crs).upper() == "EPSG:4326"
        assert set(np.unique(src.read(1)).tolist()) <= {0, 4}  # nearest neighbour: no invented class values


def test_reprojecting_an_already_matching_file_changes_nothing(tmp_path):
    p = Path(geotiff(tmp_path / "a.tif"))
    before = p.read_bytes()
    reproject_file_inplace(p, "EPSG:4326")
    assert p.read_bytes() == before


# ------------------------------------------------------------------------ SatelliteFetcher.download(stream=...)
def test_fetcher_streams_by_default_when_an_area_is_known_and_tags_each_result(tmp_path):
    full = FullDownloads()
    f = fetcher(full)
    items = [search_result(scene(tmp_path, "S2_1")), search_result(archive_scene("S2_2"))]
    out = f.download(items, tmp_path / "d", stream_bbox=BBOX, verify_checksum=False)
    assert [r.scene_id for r in out] == ["S2_1", "S2_2"] and all(r.success for r in out)
    assert [r.acquisition for r in out] == ["stream", "download"]
    assert "archive" in out[1].stream_error and out[0].stream_error == ""
    assert [c[0] for c in full.calls] == [["S2_2"]]


def test_fetcher_stream_false_downloads_whole_scenes(tmp_path, monkeypatch):
    full = FullDownloads()
    f = fetcher(full)
    streamed = spy_on_stream(f._engine, monkeypatch)
    out = f.download([search_result(scene(tmp_path, "S2_1"))], tmp_path / "d", stream_bbox=BBOX, stream=False)
    assert streamed == [] and out[0].acquisition == "download" and full.calls


def test_fetcher_without_an_area_downloads(tmp_path, monkeypatch):
    full = FullDownloads()
    f = fetcher(full)
    streamed = spy_on_stream(f._engine, monkeypatch)
    out = f.download([search_result(archive_scene("S2_2"))], tmp_path / "d", verify_checksum=False)
    assert streamed == [] and out[0].success and out[0].acquisition == "download"


def test_fetcher_stream_true_that_fails_still_returns_the_scene(tmp_path):
    full = FullDownloads()
    f = fetcher(full)
    out = f.download([search_result(archive_scene("S2_2"))], tmp_path / "d", stream=True, stream_bbox=BBOX)
    assert out[0].success and out[0].acquisition == "download" and out[0].stream_error


def test_the_environment_switch_only_changes_the_automatic_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("PYGEOVISION_STREAM", "0")
    f = fetcher(FullDownloads())
    auto = f.download([search_result(scene(tmp_path, "S2_1"))], tmp_path / "a", stream_bbox=BBOX)
    forced = f.download([search_result(scene(tmp_path, "S2_2"))], tmp_path / "b", stream_bbox=BBOX, stream=True)
    assert auto[0].acquisition == "download" and forced[0].acquisition == "stream"


def test_band_selection_narrows_the_assets_before_streaming(tmp_path):
    full = FullDownloads()
    f = fetcher(full)
    out = f.download([search_result(scene(tmp_path, "S2_1"))], tmp_path / "d", stream_bbox=BBOX, bands=["B04"], verify_checksum=False)
    assert out[0].success and out[0].acquisition == "stream"
    assert set(out[0].asset_paths) == {"B04"} and full.calls == []


# ---------------------------------------------------------------------------- the other acquisition layers
def test_every_acquisition_layer_takes_the_stream_choice():
    import inspect

    from pygeovision import PyGeoVision
    from pygeovision.data.acquire import SatelliteAcquirer
    from pygeovision.data.ai_prep import EOPipeline
    from pygeovision.data.pgf_bridge import PyGeoFetchBridge

    for fn in (
        SatelliteFetcher.download,
        PyGeoVision.download,
        SatelliteAcquirer.download,
        PyGeoFetchBridge.download,
        EOPipeline.download,
    ):
        assert "stream" in inspect.signature(fn).parameters, fn
    for fn in (
        SatelliteFetcher.download,
        PyGeoVision.download,
        SatelliteAcquirer.download,
        PyGeoFetchBridge.download,
    ):
        assert "stream_bbox" in inspect.signature(fn).parameters, fn


def test_client_stream_is_download_with_stream_true(monkeypatch):
    from pygeovision import PyGeoVision

    seen = {}
    client = PyGeoVision.__new__(PyGeoVision)
    client.data = SimpleNamespace(download=lambda **kw: seen.update(kw) or [])
    client.stream(["x"], (1, 2, 3, 4), retry_attempts=2)
    assert seen["stream"] is True and seen["stream_bbox"] == (1.0, 2.0, 3.0, 4.0) and seen["retry_attempts"] == 2
    seen.clear()
    client.download(["x"])
    assert seen["stream"] is None and seen["stream_bbox"] is None  # automatic, and no area: a plain download


def test_no_second_fallback_or_provider_list_is_left_in_the_fetcher():
    import pygeovision.data.fetch as fetch_mod

    assert not hasattr(SatelliteFetcher, "_stream_items")
    assert not hasattr(fetch_mod, "STREAMABLE_PROVIDERS")
    assert fetch_mod.fetch_scenes is _stream.fetch_scenes  # one decision point


# ----------------------------------------------------------------------------------------------- pipelines
def test_pipelines_acquire_every_scene_through_one_call_with_the_users_choice(tmp_path):
    from pygeovision.ai.pipelines import BasePipeline

    seen = []

    class Pgv:
        def download(self, items, **kw):
            seen.append(kw)
            return [DownloadResult(scene_id="s", success=True, path=tmp_path / "a.tif")]

    class P(BasePipeline):
        name = "stub"

        def run(self, bbox, output_dir, **kw): ...

    p = P(Pgv())
    p._stream_bbox = BBOX
    for pref in (None, True, False):
        p._stream_pref = pref
        p._download_scene_files(SimpleNamespace(id="s", provider="x"), tmp_path, ("red",), False, None, "s")
        assert seen[-1]["stream"] is pref and seen[-1]["stream_bbox"] == BBOX
    p._stream_pref = True  # the retry after an incomplete scene downloads the whole scene, whatever was chosen
    p._download_scene_files(
        SimpleNamespace(id="s", provider="x"), tmp_path, ("red",), False, None, "s", fresh=True
    )
    assert seen[-1]["stream"] is False and seen[-1]["overwrite"] is True and seen[-1]["resume"] is False
    p._stream_bbox = None  # a caller that never went through _search_and_download: no area, a plain download
    p._download_scene_files(SimpleNamespace(id="s", provider="x"), tmp_path, ("red",), False, None, "s")
    assert seen[-1]["stream_bbox"] is None


def test_client_pipeline_hands_the_choice_to_the_pipeline_and_does_not_warn_about_it(monkeypatch, caplog):
    import pygeovision.ai.pipelines as pl
    from pygeovision import PyGeoVision

    class Stub:
        name = "stub"
        _stream_pref = "unset"
        got = None

        def run(self, bbox, output_dir, **kw):
            Stub.got = kw
            return SimpleNamespace(success=False, error="x", output_path=None, stats={}, metadata={})

    stub = Stub()
    monkeypatch.setattr(pl, "get_pipeline", lambda name, pgv_client=None: stub)
    client = PyGeoVision.__new__(PyGeoVision)
    with caplog.at_level("WARNING"):
        client.pipeline("stub", (0, 1, 2, 3), "o", stream=False, report=False)
    assert stub._stream_pref is False and "stream" not in Stub.got
    assert "no argument" not in caplog.text


# ------------------------------------------------------------------------------------------ dataloader
def test_dataloader_streams_only_when_an_area_is_given(tmp_path):
    from pygeovision.ai.data.dataloader import GeoDataLoader

    seen = []

    class Fetch:
        def download(self, results, **kw):
            seen.append(kw)
            return []

    loader = GeoDataLoader(data_pipeline=Fetch())
    loader._download_via_pygeofetch([object()], tmp_path)
    assert "stream_bbox" not in seen[-1] and "stream" not in seen[-1]  # as before
    loader._download_via_pygeofetch([object()], tmp_path, BBOX, None)
    assert seen[-1]["stream_bbox"] == BBOX and seen[-1]["stream"] is None
    loader._download_via_pygeofetch([object()], tmp_path, None, False)
    assert seen[-1]["stream"] is False


# ------------------------------------------------------------------------------------------ CLI and YAML steps
class _Recorder:
    """Stands in for PyGeoVision: records how the scenes were requested."""

    def __init__(self):
        self.calls = []

    def search(self, **kw):
        return [SimpleNamespace(id="s1")]

    def download(self, items, **kw):
        self.calls.append(kw)
        return [DownloadResult(scene_id="s1", provider="planetary_computer", path=Path("a.tif"), success=True)]


def _run_cli(monkeypatch, extra):
    from click.testing import CliRunner

    import pygeovision
    from pygeovision.cli.main import cli

    rec = _Recorder()
    monkeypatch.setattr(pygeovision, "PyGeoVision", lambda *a, **k: rec)
    args = ["data", "download", "--bbox", "-0.35", "5.45", "0.05", "5.75", "--date", "2026-06"] + extra
    return rec, CliRunner().invoke(cli, args)


def test_cli_download_streams_the_bbox_by_default(monkeypatch):
    rec, res = _run_cli(monkeypatch, [])
    assert res.exit_code == 0, res.output
    assert rec.calls[0]["stream"] is None and rec.calls[0]["stream_bbox"] == (-0.35, 5.45, 0.05, 5.75)


def test_cli_stream_and_no_stream_flags(monkeypatch):
    rec, res = _run_cli(monkeypatch, ["--stream"])
    assert res.exit_code == 0, res.output
    assert rec.calls[0]["stream"] is True
    rec, res = _run_cli(monkeypatch, ["--no-stream"])
    assert res.exit_code == 0, res.output
    assert rec.calls[0]["stream"] is False
    assert "Downloading" in res.output and "Streaming" not in res.output


def test_cli_passes_the_download_options_for_the_fallback(monkeypatch):
    rec, res = _run_cli(monkeypatch, ["--retry", "7", "--on-failure", "retry"])
    assert res.exit_code == 0, res.output
    assert rec.calls[0]["retry_attempts"] == 7 and rec.calls[0]["on_failure"] == "retry"


def test_channel_command_passes_the_stream_choice_to_the_pipeline(monkeypatch):
    from click.testing import CliRunner

    import pygeovision
    from pygeovision.cli.main import cli

    seen = []

    class Client:
        def pipeline(self, name, **kw):
            seen.append(kw)
            return SimpleNamespace(success=False, error="stop", output_path=None, stats={}, metadata={})

    monkeypatch.setattr(pygeovision, "PyGeoVision", lambda *a, **k: Client())
    base = ["channel", "water_bodies", "--bbox", "0", "1", "2", "3", "--date", "2024-06"]
    CliRunner().invoke(cli, base)
    CliRunner().invoke(cli, base + ["--no-stream"])
    CliRunner().invoke(cli, base + ["--stream"])
    assert "stream" not in seen[0] and seen[1]["stream"] is False and seen[2]["stream"] is True


def test_yaml_download_step_follows_the_stream_key_and_the_search_box():
    from pygeovision.pipelines.steps import DownloadStep

    rec = _Recorder()
    step = DownloadStep(name="d", params={"output_dir": "./x", "parallel": 2, "verify_checksum": False})
    step.run({"client": rec, "search_results": [object()], "bbox": (0, 1, 2, 3)})
    kw = rec.calls[0]
    assert kw["stream"] is None and kw["stream_bbox"] == (0, 1, 2, 3)
    assert kw["output_dir"] == "./x" and kw["parallel"] == 2
    DownloadStep(name="d", params={"stream": False}).run({"client": rec, "search_results": [1], "bbox": (0, 1, 2, 3)})
    assert rec.calls[-1]["stream"] is False
    DownloadStep(name="d", params={"stream": True}).run({"client": rec, "search_results": [1], "bbox": (0, 1, 2, 3)})
    assert rec.calls[-1]["stream"] is True
    DownloadStep(name="d", params={}).run({"client": rec, "search_results": [1]})  # no box known
    assert rec.calls[-1]["stream_bbox"] is None
