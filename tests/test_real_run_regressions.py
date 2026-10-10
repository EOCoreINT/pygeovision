"""Regression tests for defects found in the first real-network runs.

Each class reproduces one problem seen in the logs of ``pygeovision channel
change_detection|water_bodies|flood_mapping`` on live Sentinel-2 data, using
synthetic tiles and a fake download client (no network):

* every scene of a multi-tile AOI was stacked into the same ``stack.tif``, so the
  mosaic held the last tile only and the result silently covered ~23% of the bbox;
* the search kept 5 candidates, mostly the same scene listed by three providers;
* scenes from 2022 onward were scaled without their +1000 DN offset;
* ``PyGeoVision.download(bands=...)`` never forwarded ``bands``, so every scene
  fetched all 15 assets;
* month windows ended on the 28th; GDAL TILED warnings; every pygeofetch log line
  printed twice; "0.0 m/px" in the reproject log.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from rasterio.transform import from_bounds  # noqa: E402

from pygeovision.ai.pipelines import BasePipeline  # noqa: E402
from pygeovision.data.fetch import DownloadResult, SearchResult  # noqa: E402

PX = 40   # pixels per tile edge


def _id(tile: str, date: str = "20220925", processed: str = "20220925T223429") -> str:
    return f"S2B_MSIL2A_{date}T055639_R091_T{tile}_{processed}"


class Scene:
    """A synthetic tile: footprint, reflectance, and whether its files are complete."""

    def __init__(self, tile, bbox, reflectance, *, date="20220925", cloud=1.0, provider="planetary_computer",
                 sid=None, complete=True, scl=4, fail=False):
        self.tile, self.bbox, self.reflectance = tile, bbox, reflectance
        self.id = sid or _id(tile, date)
        self.provider, self.cloud, self.complete, self.scl = provider, cloud, complete, scl
        self.fail = fail

    def result(self) -> SearchResult:
        return SearchResult(
            id=self.id, provider=self.provider, satellite="Sentinel-2B", datetime="2022-09-25T05:56:39Z",
            cloud_cover=self.cloud, bbox=self.bbox, satellite_data=object(),
        )


class FakePgv:
    """Stands in for PyGeoVision: serves scenes and writes per-asset GeoTIFFs."""

    def __init__(self, scenes, root: Path):
        self.scenes = {s.id: s for s in scenes}
        self.ordered = list(scenes)
        self.root = root
        self.search_calls: list[dict] = []
        self.download_calls: list[dict] = []

    def search(self, **kw):
        self.search_calls.append(kw)
        return [s.result() for s in self.ordered][: kw["max_results"]]

    def download(self, items, output_dir, **kw):
        self.download_calls.append({"ids": [i.id for i in items], **kw})
        out = []
        for item in items:
            sc = self.scenes[item.id]
            if sc.fail:
                out.append(DownloadResult(scene_id=item.id, provider=sc.provider, success=False, error="HTTP 503"))
                continue
            wanted = [b for b in (kw.get("bands") or []) if b.startswith("B") or b == "SCL"]
            keys = list(dict.fromkeys(wanted))
            # an interrupted earlier download that resume mistook for complete: one file only
            if not sc.complete and kw.get("resume", True):
                keys = keys[:1]
            d = Path(output_dir) / sc.provider
            d.mkdir(parents=True, exist_ok=True)
            west, south, east, north = sc.bbox
            tr = from_bounds(west, south, east, north, PX, PX)
            paths = {}
            for k in keys:
                dn = sc.scl if k == "SCL" else int(round(1000 + sc.reflectance * 10000))
                p = d / f"{item.id}_{k}.tif"
                with rasterio.open(p, "w", driver="GTiff", height=PX, width=PX, count=1, dtype="uint16",
                                   crs="EPSG:4326", transform=tr) as dst:
                    dst.write(np.full((PX, PX), dn, dtype="uint16"), 1)
                paths[k] = p
            out.append(DownloadResult(scene_id=item.id, provider=sc.provider, path=next(iter(paths.values())),
                                      asset_paths=paths, success=True))
        return out


class _Pipe(BasePipeline):
    name = "probe"

    def run(self, bbox, output_dir, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError


# Two tiles stacked north-south, like 42RUP / 42RUQ; the AOI straddles the seam.
TILE_S = (0.0, 0.0, 1.0, 1.0)
TILE_N = (0.0, 1.0, 1.0, 2.0)
AOI = (0.2, 0.6, 0.8, 1.4)


def _run(tmp_path, scenes, bands=("green", "swir1"), **kw):
    pgv = FakePgv(scenes, tmp_path)
    pipe = _Pipe(pgv)
    out = pipe._search_and_download(AOI, "2022-09", tmp_path / "after", bands=bands, **kw)
    return pgv, pipe, out


class TestMultiTileMosaic:
    def test_mosaic_holds_both_tiles_not_just_the_last(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10)
        n = Scene("42RUQ", TILE_N, 0.30)
        _, _, out = _run(tmp_path, [s, n])
        assert out is not None
        with rasterio.open(out) as src:
            band = src.read(1)
        vals = np.unique(np.round(band[np.isfinite(band)], 3))
        assert 0.1 in vals and 0.3 in vals, f"mosaic lost a tile, values: {vals}"

    def test_each_scene_gets_its_own_stack_file(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10)
        n = Scene("42RUQ", TILE_N, 0.30)
        _run(tmp_path, [s, n])
        stacks = sorted(p.name for p in (tmp_path / "after" / "imagery").glob("stack_*.tif")
                        if p.name != "stack_clipped.tif")
        assert len(stacks) == 2 and len(set(stacks)) == 2
        assert not (tmp_path / "after" / "imagery" / "stack.tif").exists()

    def test_mosaic_refuses_the_same_file_twice(self, tmp_path):
        p = tmp_path / "a.tif"
        p.write_bytes(b"x")
        with pytest.raises(ValueError, match="more than once"):
            _Pipe(None)._mosaic_scenes([p, p], tmp_path / "m.tif")


class TestSceneSelection:
    def test_search_asks_for_a_big_pool_and_bypasses_the_cache(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10)
        n = Scene("42RUQ", TILE_N, 0.30)
        pgv, _, _ = _run(tmp_path, [s, n])
        call = pgv.search_calls[0]
        assert call["max_results"] >= 60
        assert call["use_cache"] is False
        assert call["date_range"] == ("2022-09-01", "2022-09-30")

    def test_repeat_listings_do_not_crowd_out_the_other_tile(self, tmp_path):
        # Same southern scene listed by three providers ranks first (lowest cloud),
        # then the northern tile. With a pool of 5 the old code saw 3 copies + 1.
        dupes = [Scene("42RUP", TILE_S, 0.10, cloud=0.0, provider=p, sid=_id("42RUP"))
                 for p in ("planetary_computer", "aws_earth", "element84")]
        extra = [Scene("42RUP", TILE_S, 0.10, cloud=0.5, date=f"2022090{i}") for i in range(1, 6)]
        n = Scene("42RUQ", TILE_N, 0.30, cloud=3.0)
        _, _, out = _run(tmp_path, dupes + extra + [n])
        with rasterio.open(out) as src:
            band = src.read(1)
        assert np.isclose(np.nanmax(band), 0.3, atol=1e-3)

    def test_dedupe_keys_match_across_providers(self):
        p = _Pipe(None)
        pc = SimpleNamespace(id="S2B_MSIL2A_20220925T055639_R091_T42RUQ_20220925T223429", bbox=None, datetime="")
        e84 = SimpleNamespace(id="S2B_42RUQ_20220925_0_L2A", bbox=None, datetime="")
        other_date = SimpleNamespace(id="S2A_42RUQ_20220923_0_L2A", bbox=None, datetime="")
        other_tile = SimpleNamespace(id="S2B_42RUP_20220925_0_L2A", bbox=None, datetime="")
        assert p._scene_key(pc) == p._scene_key(e84)
        assert len(p._dedupe_scenes([pc, e84, other_date, other_tile])) == 3

    def test_a_sliver_of_extra_coverage_does_not_buy_another_scene(self):
        # `a` leaves a 0.001-degree strip (0.1% of the AOI) uncovered; `sliver` would fill only that.
        a = SimpleNamespace(id="a", bbox=(0.0, 0.0, 1.0, 1.399), cloud_cover=1)
        sliver = SimpleNamespace(id="b", bbox=(0.0, 1.3, 1.0, 1.4), cloud_cover=2)
        sel, full = _Pipe(None)._check_aoi_coverage((0.2, 0.6, 0.8, 1.4), [a, sliver])
        assert [r.id for r in sel] == ["a"] and not full

    def test_clearer_scene_wins_a_near_tie(self):
        clear = SimpleNamespace(id="clear", bbox=(0.0, 0.0, 1.0, 1.0), cloud_cover=0)
        bigger = SimpleNamespace(id="bigger", bbox=(0.0, 0.0, 1.0, 1.01), cloud_cover=9)
        sel, _ = _Pipe(None)._check_aoi_coverage((0.1, 0.1, 0.9, 0.9), [clear, bigger])
        assert sel[0].id == "clear"


class TestFailureHandling:
    def test_failed_scene_is_replaced_by_another_covering_the_same_area(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10, cloud=0.0, fail=True)                  # download keeps failing
        s_alt = Scene("42RUP", TILE_S, 0.12, cloud=2.0, date="20220923")        # same tile, other date
        n = Scene("42RUQ", TILE_N, 0.30, cloud=1.0)
        pgv, pipe, out = _run(tmp_path, [s, n, s_alt])
        assert out is not None
        entry = pipe._coverage_log[-1]
        assert s.id in entry["scenes_failed"]
        assert s_alt.id in entry["scenes_used"]
        assert entry["aoi_coverage_fraction"] >= 0.99

    def test_interrupted_download_is_redone_without_resume(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10, complete=False)
        pgv = FakePgv([s], tmp_path)
        out = _Pipe(pgv)._download_and_prepare_scene(
            s.result(), tmp_path / "x", ("green", "swir1"), True, True, None,
        )
        assert out is not None
        first, second = pgv.download_calls
        assert "resume" not in first
        assert second["resume"] is False and second["overwrite"] is True


class TestCoverageReporting:
    def test_partial_coverage_is_in_the_pipeline_result(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10)      # the northern tile is not available
        _, pipe, out = _run(tmp_path, [s])
        assert out is not None
        res = pipe._ok(out, {"flooded_area_ha": 1.0})
        frac = res.stats["aoi_coverage_fraction"]
        assert 0.4 < frac < 0.6               # AOI is 0.6..1.4 in latitude, tile covers 0.6..1.0
        assert res.metadata["input_coverage"][0]["scenes_used"] == [s.id]

    def test_full_coverage_is_one(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10)
        n = Scene("42RUQ", TILE_N, 0.30)
        _, pipe, out = _run(tmp_path, [s, n])
        assert pipe._ok(out).stats["aoi_coverage_fraction"] == pytest.approx(1.0, abs=1e-3)

    def test_failed_run_clears_stale_coverage(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10)
        _, pipe, _ = _run(tmp_path, [s])
        pipe._fail("boom")
        assert pipe._coverage_log == []


class TestDownloadsOnlyWhatIsNeeded:
    def test_requested_assets_are_the_bands_plus_cloud_mask(self, tmp_path):
        s = Scene("42RUP", TILE_S, 0.10)
        n = Scene("42RUQ", TILE_N, 0.30)
        pgv, _, _ = _run(tmp_path, [s, n], bands=("green", "swir1"))
        req = set(pgv.download_calls[0]["bands"])
        assert {"B03", "B11", "SCL"} <= req
        assert not ({"B02", "B04", "B08", "B12", "AOT", "WVP"} & req)

    def test_facade_forwards_bands_to_the_data_layer(self):
        from pygeovision import PyGeoVision

        client = PyGeoVision.__new__(PyGeoVision)
        client.data = MagicMock()
        item = SearchResult(id="x", provider="p", satellite="s", datetime="", cloud_cover=0, bbox=None)
        client.download([item], bands=["B03", "B11"])
        assert client.data.download.call_args.kwargs["bands"] == ["B03", "B11"]


class TestSentinel2Offset:
    @pytest.mark.parametrize("sid,expected", [
        ("S2B_MSIL2A_20240626T105619_R094_T30UXC_20240626T151111", 1000),     # Planetary Computer, 2024
        ("S2A_MSIL2A_20220622T105631_R094_T30UXC_20240718T031007", 1000),     # 2022 scene reprocessed in 2024
        ("S2B_30UXC_20240626_0_L2A", 1000),                                    # Earth Search
        ("S2A_MSIL2A_20240626T105631_N0510_R094_T30UXC_20240626T151111", 1000),
        ("S2A_MSIL2A_20210626T105631_N0300_R094_T30UXC_20210626T151111", 0),
        ("S2A_MSIL2A_20210626T105631_R094_T30UXC_20210626T151111", 0),
        ("S2A_MSIL2A_20220120T105631_R094_T30UXC_20220120T151111", 0),         # before 2022-01-25
        ("S2A_MSIL2A_20220126T105631_R094_T30UXC_20220126T151111", 1000),
        ("S2A_30UXC_20180626_0_L2A", 0),
        ("", 0),
    ])
    def test_offset_from_scene_id(self, sid, expected):
        from pygeovision.data.radiometric import sentinel2_boa_add_offset

        assert sentinel2_boa_add_offset(sid) == expected

    def test_water_stays_water_after_the_offset_fix(self):
        """Why it matters: Thames-like water, green 0.08 / NIR 0.03 reflectance. Without the
        offset NDWI falls from 0.45 to 0.07, below the 0.3 water threshold."""
        from pygeovision.data.radiometric import dn_to_reflectance

        sid = "S2B_MSIL2A_20240626T105619_R094_T30UXC_20240626T151111"
        dn = np.array([[[1800]], [[1300]]], dtype="uint16")             # green 0.08, nir 0.03
        refl = dn_to_reflectance(dn, sid, is_landsat=False)
        g, n = refl[0, 0, 0], refl[1, 0, 0]
        assert (g - n) / (g + n) == pytest.approx(0.4545, abs=1e-3)
        naive = dn.astype("float32") / 10000
        assert (naive[0, 0, 0] - naive[1, 0, 0]) / (naive[0, 0, 0] + naive[1, 0, 0]) < 0.3

    def test_single_file_scene_is_offset_corrected_too(self, tmp_path):
        """The cached-search path delivers one pre-stacked file; it must scale the same way."""
        sid = _id("42RUP", "20240626", "20240626T151111")
        d = tmp_path / "planetary_computer"
        d.mkdir()
        p = d / f"{sid}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=2, dtype="uint16",
                           crs="EPSG:4326", transform=from_bounds(0, 0, 1, 1, 4, 4)) as dst:
            dst.write(np.full((2, 4, 4), 1500, dtype="uint16"))
        dl = DownloadResult(scene_id=sid, provider="planetary_computer", path=p, asset_paths={"x": p}, success=True)
        out = _Pipe(None)._prepare_downloaded_scene(dl, tmp_path, ("green", "swir1"), True, True, "imagery")
        with rasterio.open(out) as src:
            assert src.read(1)[0, 0] == pytest.approx(0.05, abs=1e-4)


class TestMonthWindow:
    @pytest.mark.parametrize("value,expected", [
        ("2024-06", ("2024-06-01", "2024-06-30")),
        ("2022-05", ("2022-05-01", "2022-05-31")),
        ("2024-02", ("2024-02-01", "2024-02-29")),
        ("2023-02", ("2023-02-01", "2023-02-28")),
        ("2022-09-15", ("2022-09-15", "2022-09-15")),
    ])
    def test_date_window(self, value, expected):
        from pygeovision.data._dates import date_window

        assert date_window(value) == expected

    def test_cli_download_uses_real_month_end(self):
        src = Path(__file__).resolve().parents[1].joinpath("pygeovision/cli/main.py").read_text()
        assert '-28"' not in src


class TestSearchDefaults:
    def _fetcher(self, stored=()):
        from pygeovision.data.fetch import SatelliteFetcher

        f = SatelliteFetcher.__new__(SatelliteFetcher)
        f.list_credentials = lambda: list(stored)
        return f

    def test_unusable_defaults_are_dropped(self):
        f = self._fetcher()
        out = f._drop_unusable_default_providers(
            ["planetary_computer", "copernicus", "aws_earth", "element84", "sentinel_hub"], ["sentinel-2-l2a"])
        assert out == ["planetary_computer", "aws_earth", "element84"]

    def test_provider_with_credentials_is_kept(self):
        f = self._fetcher(stored=["sentinel_hub"])
        out = f._drop_unusable_default_providers(["planetary_computer", "sentinel_hub"], ["sentinel-2-l2a"])
        assert out == ["planetary_computer", "sentinel_hub"]

    def test_never_returns_an_empty_list(self):
        f = self._fetcher()
        assert f._drop_unusable_default_providers(["sentinel_hub"], ["sentinel-2-l2a"]) == ["sentinel_hub"]

    def test_cache_key_depends_on_result_cap_and_sort(self):
        f = self._fetcher()
        a = f._cache_key((0, 0, 1, 1), ("a", "b"), ["p"], 10, ["c"], 5, "cloud_cover", "asc")
        b = f._cache_key((0, 0, 1, 1), ("a", "b"), ["p"], 10, ["c"], 60, "cloud_cover", "asc")
        c = f._cache_key((0, 0, 1, 1), ("a", "b"), ["p"], 10, ["c"], 5, "datetime", "desc")
        assert len({a, b, c}) == 3


class TestLogging:
    def _capture(self):
        root = logging.getLogger()
        saved = (list(root.handlers), root.level)
        for h in list(root.handlers):
            root.removeHandler(h)
        pgf = logging.getLogger("pygeofetch")
        saved_pgf = list(pgf.handlers)
        for h in saved_pgf:
            pgf.removeHandler(h)
        return root, pgf, saved, saved_pgf

    def _restore(self, root, pgf, saved, saved_pgf):
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in list(pgf.handlers):
            pgf.removeHandler(h)
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])
        for h in saved_pgf:
            pgf.addHandler(h)

    def test_pygeofetch_message_is_printed_once(self):
        from pygeovision._logging import configure_logging

        root, pgf, saved, saved_pgf = self._capture()
        try:
            configure_logging(logging.INFO)
            root_buf = io.StringIO()
            root.handlers[0].setStream(root_buf)
            pgf_buf = io.StringIO()
            own = logging.StreamHandler(pgf_buf)           # what pygeofetch installs on its logger
            pgf.addHandler(own)
            logging.getLogger("pygeofetch.core.engine").info("PyGeoFetch ready")
            logging.getLogger("pygeovision.data.fetch").info("Searching")
            assert pgf_buf.getvalue().count("PyGeoFetch ready") == 1
            assert "PyGeoFetch ready" not in root_buf.getvalue()
            assert "Searching" in root_buf.getvalue()
        finally:
            self._restore(root, pgf, saved, saved_pgf)

    def test_pygeofetch_message_still_shown_when_it_has_no_handler(self):
        from pygeovision._logging import configure_logging

        root, pgf, saved, saved_pgf = self._capture()
        try:
            configure_logging(logging.INFO)
            buf = io.StringIO()
            root.handlers[0].setStream(buf)
            logging.getLogger("pygeofetch.core.engine").info("PyGeoFetch ready")
            assert buf.getvalue().count("PyGeoFetch ready") == 1
        finally:
            self._restore(root, pgf, saved, saved_pgf)


class TestGdalWarnings:
    def test_writable_profile_drops_block_sizes_of_striped_sources_only(self):
        from pygeovision.data._profile import writable_profile

        striped = {"tiled": False, "blockxsize": 100, "blockysize": 1, "count": 1}
        tiled = {"tiled": True, "blockxsize": 256, "blockysize": 256, "count": 1}
        assert "blockxsize" not in writable_profile(striped, count=3)
        assert writable_profile(striped, count=3)["count"] == 3
        assert writable_profile(tiled)["blockxsize"] == 256

    def test_stacking_and_cropping_a_striped_source_logs_no_gdal_warning(self, tmp_path, caplog):
        from pygeovision.data.radiometric import crop_to_bbox, stack_and_prepare_bands

        tr = from_bounds(0, 0, 1, 1, 64, 64)
        assets = {}
        for k in ("B03", "B11"):
            p = tmp_path / f"{k}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint16",
                               crs="EPSG:4326", transform=tr, tiled=False, blockysize=8) as dst:
                dst.write(np.full((64, 64), 1500, dtype="uint16"), 1)
            assets[k] = p
        with caplog.at_level(logging.WARNING):
            st = stack_and_prepare_bands(assets, tmp_path / "s.tif", bands=("green", "swir1"),
                                         scene_id=_id("42RUP"), apply_cloud_mask=False)
            crop_to_bbox(st, (0.1, 0.1, 0.9, 0.9), tmp_path / "c.tif")
        assert not [r for r in caplog.records if "BLOCKXSIZE" in r.getMessage()]


class TestReprojectLog:
    def test_degrees_are_not_reported_as_metres(self, tmp_path, caplog):
        from rasterio.crs import CRS

        from pygeovision.data.fetch import SatelliteFetcher

        src = tmp_path / "scene.tif"
        tr = rasterio.transform.from_origin(500000, 5700000, 10, 10)
        with rasterio.open(src, "w", driver="GTiff", height=50, width=50, count=1, dtype="uint16",
                           crs=CRS.from_epsg(32630), transform=tr) as dst:
            dst.write(np.full((50, 50), 1500, dtype="uint16"), 1)
        f = SatelliteFetcher.__new__(SatelliteFetcher)
        with caplog.at_level(logging.INFO, logger="pygeovision.data.fetch"):
            out = f._apply_post_process(src, ["reproject:EPSG:4326"])
        assert out is not None
        msg = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("reproject:"))
        assert "deg/px" in msg and "0.0 m/px" not in msg


class TestVerifyInstallTool:
    def _tool(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "tools" / "verify_install.py"
        spec = importlib.util.spec_from_file_location("verify_install", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_reports_missing_and_stale_files_and_ignores_line_endings(self, tmp_path):
        tool = self._tool()
        (tmp_path / "a.py").write_bytes(b"print(1)\r\n")            # CRLF checkout of an LF file
        (tmp_path / "b.py").write_text("old")
        manifest = tmp_path / "MANIFEST.sha256"
        manifest.write_text(
            "# comment\n"
            f"{tool.digest(tmp_path / 'a.py')}  a.py\n"
            f"{__import__('hashlib').sha256(b'new').hexdigest()}  b.py\n"
            f"{'0' * 64}  gone.py\n"
        )
        assert tool.main(["--root", str(tmp_path), "--manifest", str(manifest)]) == 1
        (tmp_path / "b.py").write_text("new")
        manifest.write_text(f"{tool.digest(tmp_path / 'a.py')}  a.py\n{tool.digest(tmp_path / 'b.py')}  b.py\n")
        assert tool.main(["--root", str(tmp_path), "--manifest", str(manifest)]) == 0

    def test_missing_manifest_is_an_error(self, tmp_path):
        assert self._tool().main(["--root", str(tmp_path)]) == 2


class TestOnlyNeededAssetsReachTheProvider:
    """With pygeofetch's real models: 15 Sentinel-2 assets in, 3 out (the 15-asset downloads in the logs)."""

    def test_flood_mapping_bands_download_three_of_fifteen_assets(self, tmp_path):
        pytest.importorskip("pygeofetch")
        from pygeofetch.models.satellite_data import (
            SatelliteAsset,
            SatelliteData,
            resolve_band_keys,
        )

        from pygeovision import PyGeoVision
        from pygeovision.data.fetch import SatelliteFetcher

        keys = ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A", "SCL", "WVP"]
        sid = "S2B_MSIL2A_20240626T105619_R094_T30UXC_20240626T151111"
        sd = SatelliteData(
            id=sid, provider="planetary_computer", satellite="Sentinel-2B", collection="sentinel-2-l2a",
            assets={k: SatelliteAsset(key=k, href=f"https://example.invalid/{k}.tif",
                                      type="image/tiff; application=geotiff", roles=["data"]) for k in keys},
        )
        item = SearchResult(id=sid, provider="planetary_computer", satellite="Sentinel-2B",
                            datetime="2024-06-26T10:56:19Z", cloud_cover=0, bbox=(-1.5, 51.3, 0.1, 52.3),
                            assets={k: {"href": f"https://x/{k}.tif"} for k in keys}, satellite_data=sd)
        seen: dict = {}

        class StubEngine:
            def download(self, satellite_data_list, destination, options):
                data = satellite_data_list[0]
                pool = data.data_assets or data.assets
                matched = resolve_band_keys(list(options.bands or []), list(pool))
                seen["downloaded"] = sorted(k for k in pool if k in matched)
                return []

        fetcher = SatelliteFetcher.__new__(SatelliteFetcher)
        fetcher._engine, fetcher._credentials = StubEngine(), {}
        fetcher._use_cache = False
        fetcher._use_cli = lambda: False
        client = PyGeoVision.__new__(PyGeoVision)
        client.data = fetcher

        pipe = _Pipe(client)
        # the exact request _download_scene_files builds for flood_mapping (green + swir1, cloud mask on)
        pipe._download_scene_files(item, tmp_path, ("green", "swir1"), True, None, "imagery")
        assert seen["downloaded"] == ["B03", "B11", "SCL"]
