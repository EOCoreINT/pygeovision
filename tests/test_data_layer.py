"""
Tests for PyGeoVision data layer — pygeofetch + pystac_client integration.
All network calls are mocked; no real satellite APIs required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from pygeovision.data.fetch import DownloadResult, SatelliteFetcher, SearchResult
from pygeovision.data.pipeline import DataPipeline
from pygeovision.data.providers import (
    DEFAULT_SEARCH_PROVIDERS,
    OPEN_PROVIDERS,
    PROVIDERS,
    SATELLITE_SHORTCUTS,
    STAC_PROVIDERS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher():
    return SatelliteFetcher()

@pytest.fixture
def fetcher_no_cli():
    """Fetcher configured to simulate no pygeofetch CLI available."""
    f = SatelliteFetcher()
    f._pygeofetch_available = False
    return f

@pytest.fixture
def sample_results():
    return [
        SearchResult(
            id="S2C_MSIL2A_20260516T153811_R01",
            provider="planetary_computer",
            satellite="Sentinel-2C",
            datetime="2026-05-16T15:38:11",
            cloud_cover=0.0,
            bbox=(-74.1, 40.6, -73.7, 40.9),
            score=0.99,
            collection="sentinel-2-l2a",
            properties={"eo:cloud_cover": 0, "platform": "Sentinel-2C"},
        ),
        SearchResult(
            id="LC09_L2SP_013032_20260505_02_T",
            provider="planetary_computer",
            satellite="Landsat-9",
            datetime="2026-05-05T14:30:00",
            cloud_cover=1.0,
            bbox=(-74.1, 40.6, -73.7, 40.9),
            score=0.98,
            collection="landsat-c2-l2",
        ),
    ]

@pytest.fixture
def sample_geojson(tmp_path):
    """Write a sample pygeofetch-style GeoJSON and return its path."""
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "S2C_MSIL2A_20260516T153811_R01",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-74.1, 40.6], [-73.7, 40.6],
                        [-73.7, 40.9], [-74.1, 40.9], [-74.1, 40.6],
                    ]],
                },
                "properties": {
                    "id": "S2C_MSIL2A_20260516T153811_R01",
                    "provider": "planetary_computer",
                    "satellite": "Sentinel-2C",
                    "datetime": "2026-05-16T15:38:11",
                    "eo:cloud_cover": 0,
                    "score": 0.99,
                    "collection": "sentinel-2-l2a",
                },
            },
            {
                "type": "Feature",
                "id": "LC09_L2SP_013032_20260505_02_T",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-74.1, 40.6], [-73.7, 40.6],
                        [-73.7, 40.9], [-74.1, 40.9], [-74.1, 40.6],
                    ]],
                },
                "properties": {
                    "id": "LC09_L2SP_013032_20260505_02_T",
                    "provider": "planetary_computer",
                    "satellite": "landsat-9",
                    "datetime": "2026-05-05T14:30:00",
                    "eo:cloud_cover": 1,
                    "score": 0.98,
                    "collection": "landsat-c2-l2",
                },
            },
        ],
    }
    path = tmp_path / "results.geojson"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Providers registry
# ---------------------------------------------------------------------------

class TestProviders:
    def test_22_providers_defined(self):
        assert len(PROVIDERS) == 22

    def test_all_have_required_fields(self):
        for pid, pinfo in PROVIDERS.items():
            assert "name" in pinfo, f"{pid} missing name"
            assert "satellites" in pinfo, f"{pid} missing satellites"
            assert "open" in pinfo, f"{pid} missing open flag"

    def test_open_providers_identified(self):
        assert "planetary_computer" in OPEN_PROVIDERS
        assert "aws_earth" in OPEN_PROVIDERS
        assert "noaa_big_data" in OPEN_PROVIDERS
        assert "usgs" not in OPEN_PROVIDERS
        assert "planet" not in OPEN_PROVIDERS

    def test_stac_providers_have_endpoints(self):
        for pid, url in STAC_PROVIDERS.items():
            assert url.startswith("https://"), f"{pid} has invalid endpoint"

    def test_satellite_shortcuts(self):
        assert "planetary_computer" in SATELLITE_SHORTCUTS["sentinel-2"]
        assert "planet" in SATELLITE_SHORTCUTS["planetscope"]
        assert "usgs" in SATELLITE_SHORTCUTS["landsat"]
        assert "opentopography" in SATELLITE_SHORTCUTS["dem"]

    def test_default_providers_are_open(self):
        for p in DEFAULT_SEARCH_PROVIDERS:
            assert p in OPEN_PROVIDERS, f"{p} is in defaults but requires auth"

    def test_sar_providers(self):
        sar = {k for k, v in PROVIDERS.items() if v.get("sar")}
        assert "copernicus" in sar
        assert "alaska_satellite_facility" in sar
        assert "sentinel_hub" in sar

    def test_sub_meter_providers(self):
        sub = {k for k, v in PROVIDERS.items() if v.get("sub_meter")}
        assert "planet" in sub
        assert "maxar_gbdx" in sub
        assert "airbus_oneatlas" in sub


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_basic_creation(self):
        r = SearchResult(
            id="scene-001",
            provider="planetary_computer",
            satellite="Sentinel-2C",
            datetime="2026-05-16T15:38:11",
            cloud_cover=5.0,
            bbox=(-74.1, 40.6, -73.7, 40.9),
            score=0.95,
            collection="sentinel-2-l2a",
        )
        assert r.id == "scene-001"
        assert r.date == "2026-05-16"
        assert not r.is_sar
        assert r.resolution_m == 10.0  # Sentinel-2 default

    def test_sar_detection(self):
        r = SearchResult(
            id="s1-001", provider="copernicus", satellite="Sentinel-1A",
            datetime="2026-01-01", cloud_cover=None,
            bbox=(-74.1, 40.6, -73.7, 40.9),
        )
        assert r.is_sar

    def test_landsat_resolution(self):
        r = SearchResult(
            id="l9-001", provider="usgs", satellite="Landsat-9",
            datetime="2026-01-01", cloud_cover=2.0,
            bbox=(-74.1, 40.6, -73.7, 40.9),
        )
        assert r.resolution_m == 30.0

    def test_planet_resolution(self):
        r = SearchResult(
            id="ps-001", provider="planet", satellite="PlanetScope",
            datetime="2026-01-01", cloud_cover=0.0,
            bbox=(-74.1, 40.6, -73.7, 40.9),
        )
        assert r.resolution_m == 3.0

    def test_to_dict(self, sample_results):
        d = sample_results[0].to_dict()
        assert d["id"] == "S2C_MSIL2A_20260516T153811_R01"
        assert d["provider"] == "planetary_computer"
        assert d["cloud_cover"] == 0.0
        assert d["collection"] == "sentinel-2-l2a"
        assert isinstance(d["bbox"], list)

    def test_str_representation(self, sample_results):
        s = str(sample_results[0])
        assert "planetary_computer" in s
        assert "Sentinel-2C" in s
        assert "0%" in s

    def test_none_cloud_cover(self):
        r = SearchResult(
            id="x", provider="usgs", satellite="Landsat",
            datetime="2026-01-01", cloud_cover=None, bbox=None,
        )
        assert r.cloud_cover is None
        assert "N/A" in str(r)


# ---------------------------------------------------------------------------
# DownloadResult
# ---------------------------------------------------------------------------

class TestDownloadResult:
    def test_success_result(self, tmp_path):
        r = DownloadResult(
            scene_id="scene-001",
            provider="planetary_computer",
            path=tmp_path / "scene.tif",
            success=True,
            bytes_downloaded=50 * 1024 * 1024,  # 50MB
            duration_seconds=12.5,
        )
        assert r.size_mb == pytest.approx(50.0)
        assert "✓" in str(r)
        assert "50.0 MB" in str(r)

    def test_failure_result(self):
        r = DownloadResult(
            scene_id="scene-002",
            provider="usgs",
            success=False,
            error="Authentication failed",
        )
        assert "✗" in str(r)
        assert "Authentication failed" in str(r)


# ---------------------------------------------------------------------------
# SatelliteFetcher — CLI mode
# ---------------------------------------------------------------------------

class TestSatelliteFetcherCLI:
    def test_has_pygeofetch_detection(self, fetcher):
        # Just tests that detection works (not that pygeofetch is installed)
        result = fetcher._has_pygeofetch()
        assert isinstance(result, bool)

    def test_resolve_providers_explicit(self, fetcher):
        providers = fetcher._resolve_providers(
            ["planetary_computer", "usgs"], None, None
        )
        assert providers == ["planetary_computer", "usgs"]

    def test_resolve_providers_from_satellite(self, fetcher):
        providers = fetcher._resolve_providers(None, "sentinel-2", None)
        assert "planetary_computer" in providers

    def test_resolve_providers_from_collections(self, fetcher):
        providers = fetcher._resolve_providers(
            None, None, ["sentinel-2-l2a"]
        )
        assert "planetary_computer" in providers

    def test_resolve_providers_landsat(self, fetcher):
        providers = fetcher._resolve_providers(
            None, None, ["landsat-c2-l2"]
        )
        assert "planetary_computer" in providers or "usgs" in providers

    def test_resolve_providers_default(self, fetcher):
        providers = fetcher._resolve_providers(None, None, None)
        assert providers == DEFAULT_SEARCH_PROVIDERS

    def test_parse_geojson_results(self, fetcher, sample_geojson):
        results = fetcher._parse_stac_geojson_file(sample_geojson)
        assert len(results) == 2
        assert results[0].id == "S2C_MSIL2A_20260516T153811_R01"
        assert results[0].provider == "planetary_computer"
        assert results[0].satellite == "Sentinel-2C"
        assert results[0].cloud_cover == 0.0
        assert results[0].score == 0.99
        assert results[0].bbox is not None

    def test_parse_geojson_extracts_bbox(self, fetcher, sample_geojson):
        results = fetcher._parse_stac_geojson_file(sample_geojson)
        bbox = results[0].bbox
        assert bbox is not None
        assert len(bbox) == 4
        assert bbox[0] == pytest.approx(-74.1)

    def test_cache_key_deterministic(self, fetcher):
        k1 = fetcher._cache_key(
            (-74.1, 40.6, -73.7, 40.9), ("2024-01-01", "2024-06-01"),
            ["planetary_computer"], 15.0, ["sentinel-2-l2a"],
        )
        k2 = fetcher._cache_key(
            (-74.1, 40.6, -73.7, 40.9), ("2024-01-01", "2024-06-01"),
            ["planetary_computer"], 15.0, ["sentinel-2-l2a"],
        )
        assert k1 == k2

    def test_cache_key_different_for_different_params(self, fetcher):
        k1 = fetcher._cache_key(
            (-74.1, 40.6, -73.7, 40.9), ("2024-01-01", "2024-06-01"),
            ["planetary_computer"], 15.0, None,
        )
        k2 = fetcher._cache_key(
            (-0.15, 51.47, -0.10, 51.52), ("2024-01-01", "2024-06-01"),
            ["planetary_computer"], 15.0, None,
        )
        assert k1 != k2

    def test_cache_save_and_load(self, fetcher, sample_results):
        key = "test_cache_key_xyz"
        fetcher._save_cache(key, sample_results)
        loaded = fetcher._load_cache(key)
        assert loaded is not None
        assert len(loaded) == len(sample_results)
        assert loaded[0].id == sample_results[0].id

    def test_cache_returns_none_if_not_found(self, fetcher):
        result = fetcher._load_cache("nonexistent_key_abc123")
        assert result is None

    def test_add_credentials_stores_in_memory(self, fetcher):
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = None; _fetch._PYGEOFETCH_PY_AVAILABLE = False  # Simulate no CLI
        fetcher.add_credentials("usgs", username="user", password="pass")
        assert "usgs" in fetcher._credentials
        assert fetcher._credentials["usgs"]["username"] == "user"

    def test_add_credentials_chaining(self, fetcher):
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = None; _fetch._PYGEOFETCH_PY_AVAILABLE = False
        result = (
            fetcher
            .add_credentials("usgs", username="user", password="pass")
            .add_credentials("planet", api_key="PL_KEY")
        )
        assert result is fetcher  # Returns self
        assert "usgs" in fetcher._credentials
        assert "planet" in fetcher._credentials

    def test_collection_to_satellite_mapping(self, fetcher):
        assert fetcher._collection_to_satellite("sentinel-2-l2a") == "Sentinel-2"
        assert fetcher._collection_to_satellite("sentinel-1-rtc") == "Sentinel-1"
        assert fetcher._collection_to_satellite("landsat-c2-l2") == "Landsat"
        assert fetcher._collection_to_satellite("naip") == "NAIP"

    def test_pick_best_asset(self, fetcher):
        result = SearchResult(
            id="x", provider="p", satellite="s", datetime="", cloud_cover=None, bbox=None,
            assets={"visual": {"href": "https://example.com/visual.tif"}, "B04": {"href": "..."}}
        )
        best = fetcher._pick_best_asset(result)
        assert best in ("B04", "visual")

    @patch("subprocess.run")
    def test_search_via_cli_calls_pygeofetch(self, mock_run, fetcher, sample_geojson):
        """Test that search() invokes pygeofetch search run."""
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = "pygeofetch"; _fetch._PYGEOFETCH_PY_AVAILABLE = False; fetcher._pgf_engine = None

        # Mock subprocess to write geojson to output file
        def side_effect(cmd, **kwargs):
            # Find --output argument and write our sample there
            if "--output" in cmd:
                idx = cmd.index("--output")
                out_path = Path(cmd[idx + 1])
                out_path.write_text(sample_geojson.read_text())
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        results = fetcher.search(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            date_range=("2024-01-01", "2024-06-01"),
            providers=["planetary_computer"],
            cloud_cover_max=15.0,
            use_cache=False,
        )

        assert len(results) == 2
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "search" in call_args
        assert "run" in call_args
        assert "--bbox" in call_args

    @patch("subprocess.run")
    def test_download_via_cli_calls_pygeofetch(self, mock_run, fetcher, sample_results, tmp_path):
        """Test that download() invokes pygeofetch download run."""
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = "pygeofetch"; _fetch._PYGEOFETCH_PY_AVAILABLE = False; fetcher._pgf_engine = None
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Create a fake output file
        (tmp_path / f"{sample_results[0].id[:20]}_visual.tif").touch()

        fetcher.download(
            sample_results[:1],
            output_dir=tmp_path,
            parallel=2,
            post_process=["unzip", "cog"],
        )

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "download" in call_args
        assert "run" in call_args
        assert "--from-search" in call_args
        assert "--parallel" in call_args

    @patch("subprocess.run")
    def test_auth_add_calls_cli(self, mock_run, fetcher):
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = "pygeofetch"; _fetch._PYGEOFETCH_PY_AVAILABLE = False; fetcher._pgf_engine = None
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        fetcher.add_credentials("usgs", username="user", password="pass")

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "auth" in call_args
        assert "add" in call_args
        assert "usgs" in call_args
        assert "--username" in call_args

    @patch("subprocess.run")
    def test_pipeline_run_calls_cli(self, mock_run, fetcher, tmp_path):
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = "pygeofetch"; _fetch._PYGEOFETCH_PY_AVAILABLE = False; fetcher._pgf_engine = None
        mock_run.return_value = MagicMock(returncode=0, stdout="Pipeline complete", stderr="")

        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text("name: test\nsteps:\n  - search:\n      providers: [planetary_computer]\n")

        result = fetcher.run_pipeline(yaml_file)
        assert result["success"]

        call_args = mock_run.call_args[0][0]
        assert "pipeline" in call_args
        assert "run" in call_args

    @patch("subprocess.run")
    def test_cache_stats_cli(self, mock_run, fetcher):
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = "pygeofetch"; _fetch._PYGEOFETCH_PY_AVAILABLE = False; fetcher._pgf_engine = None
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"entries": 42, "size_bytes": 1048576}',
            stderr="",
        )
        fetcher.cache_stats()
        assert mock_run.called

    @patch("subprocess.run")
    def test_clear_cache_cli(self, mock_run, fetcher):
        import pygeovision.data.fetch as _fetch; _fetch._PYGEOFETCH_CLI_CHECKED = True; _fetch._PYGEOFETCH_CLI_EXE = "pygeofetch"; _fetch._PYGEOFETCH_PY_AVAILABLE = False; fetcher._pgf_engine = None
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        fetcher.clear_cache(provider="planetary_computer", older_than="7d")
        call_args = mock_run.call_args[0][0]
        assert "cache" in call_args
        assert "clear" in call_args
        assert "--provider" in call_args


# ---------------------------------------------------------------------------
# SatelliteFetcher — STAC fallback mode
# ---------------------------------------------------------------------------

class TestSatelliteFetcherSTAC:
    def test_search_stac_fallback_no_pygeofetch(self, fetcher_no_cli):
        """When CLI unavailable, uses pystac_client for STAC providers."""
        import pygeovision.data.fetch as _fetch
        pytest.importorskip("pystac_client")
        # Force both Python API and CLI unavailable so pystac_client fallback runs
        orig_py = _fetch._PYGEOFETCH_PY_AVAILABLE
        orig_cli = _fetch._PYGEOFETCH_CLI_EXE
        orig_checked = _fetch._PYGEOFETCH_CLI_CHECKED
        _fetch._PYGEOFETCH_PY_AVAILABLE = False
        _fetch._PYGEOFETCH_CLI_EXE = None
        _fetch._PYGEOFETCH_CLI_CHECKED = True

        mock_item = MagicMock()
        mock_item.id = "S2A_MSIL2A_20240601T154811"
        mock_item.datetime = MagicMock()
        mock_item.datetime.isoformat.return_value = "2024-06-01T15:48:11"
        mock_item.bbox = [-0.15, 51.47, -0.10, 51.52]
        mock_item.properties = {"eo:cloud_cover": 3.0, "platform": "Sentinel-2A"}
        mock_item.assets = {}
        mock_item.collection_id = "sentinel-2-l2a"

        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]

        mock_catalog = MagicMock()
        mock_catalog.search.return_value = mock_search

        with patch("pystac_client.Client.open", return_value=mock_catalog):
            results = fetcher_no_cli.search(
                bbox=(-0.15, 51.47, -0.10, 51.52),
                date_range=("2024-06-01", "2024-06-30"),
                providers=["planetary_computer"],
                cloud_cover_max=15,
                use_cache=False,
            )

        assert len(results) >= 1
        assert results[0].id == "S2A_MSIL2A_20240601T154811"
        assert results[0].cloud_cover == pytest.approx(3.0)
        # Restore
        _fetch._PYGEOFETCH_PY_AVAILABLE = orig_py
        _fetch._PYGEOFETCH_CLI_EXE = orig_cli
        _fetch._PYGEOFETCH_CLI_CHECKED = orig_checked


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

class TestDataPipeline:
    def test_create_basic_pipeline(self):
        p = DataPipeline("test-pipeline", description="Test")
        assert p.name == "test-pipeline"
        assert len(p.steps) == 0

    def test_search_step(self):
        p = DataPipeline("test")
        p.search(
            providers=["planetary_computer"],
            bbox=(-74.1, 40.6, -73.7, 40.9),
            date_range="last_7_days",
            cloud_cover="0-10",
        )
        assert len(p.steps) == 1
        assert p.steps[0].type == "search"
        assert p.steps[0].config["cloud_cover"] == "0-10"
        assert "planetary_computer" in p.steps[0].config["providers"]

    def test_filter_step(self):
        p = DataPipeline("test")
        p.filter("data.cloud_cover < 5")
        assert len(p.steps) == 1
        assert p.steps[0].type == "filter"
        assert p.steps[0].config["expression"] == "data.cloud_cover < 5"

    def test_download_step(self):
        p = DataPipeline("test")
        p.download(
            output="./raw/",
            parallel=4,
            post_process=["unzip", "reproject:EPSG:4326", "cog"],
        )
        assert p.steps[0].type == "download"
        assert p.steps[0].config["parallel"] == 4
        assert "unzip" in p.steps[0].config["post_process"]

    def test_export_step(self):
        p = DataPipeline("test")
        p.export(format="cloud_optimized_geotiff", destination="s3://bucket/")
        assert p.steps[0].type == "export"
        assert p.steps[0].config["destination"] == "s3://bucket/"

    def test_ai_step(self):
        p = DataPipeline("test")
        p.ai_process(model="unet_resnet50", task="segmentation", num_classes=5)
        assert p.steps[0].type == "ai"
        assert p.steps[0].config["num_classes"] == 5

    def test_chaining(self):
        p = (
            DataPipeline("weekly-sentinel2")
            .search(providers=["planetary_computer"], bbox=(-74.1, 40.6, -73.7, 40.9))
            .filter("data.cloud_cover < 5")
            .download(parallel=4, output="./raw/")
            .export(destination="s3://bucket/")
        )
        assert len(p.steps) == 4
        assert p.steps[0].type == "search"
        assert p.steps[3].type == "export"

    def test_schedule_setting(self):
        p = DataPipeline("test")
        p.set_schedule("0 6 * * 1")
        assert p.schedule == "0 6 * * 1"

    def test_schedule_via_chaining(self):
        p = DataPipeline("test").search().set_schedule("0 6 * * 1")
        assert p.schedule == "0 6 * * 1"

    def test_to_yaml(self):
        p = (
            DataPipeline("weekly-s2", description="Weekly Sentinel-2")
            .search(providers=["planetary_computer"])
            .download(output="./raw/")
            .set_schedule("0 6 * * 1")
        )
        yaml_str = p.to_yaml()
        assert "weekly-s2" in yaml_str
        assert "search" in yaml_str
        assert "download" in yaml_str
        assert "0 6 * * 1" in yaml_str

    def test_save_and_load(self, tmp_path):
        p = (
            DataPipeline("test-save")
            .search(providers=["planetary_computer"], bbox=(-74.1, 40.6, -73.7, 40.9))
            .download(output="./raw/")
        )
        path = p.save(tmp_path / "pipeline.yaml")
        assert path.exists()

        loaded = DataPipeline.from_yaml(path)
        assert loaded.name == "test-save"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].type == "search"

    def test_validate_empty_pipeline(self):
        p = DataPipeline("empty")
        assert not p.validate()

    def test_validate_nonempty_pipeline(self):
        p = DataPipeline("test").search()
        assert p.validate()

    def test_repr(self):
        p = DataPipeline("my-pipe").search().download()
        r = repr(p)
        assert "my-pipe" in r
        assert "steps=2" in r

    def test_tuple_date_range(self):
        p = DataPipeline("test").search(date_range=("2024-01-01", "2024-06-30"))
        config = p.steps[0].config
        assert "start_date" in config
        assert config["start_date"] == "2024-01-01"

    def test_tuple_bbox_converted(self):
        p = DataPipeline("test").search(bbox=(-74.1, 40.6, -73.7, 40.9))
        config = p.steps[0].config
        assert "bbox" in config
        assert isinstance(config["bbox"], str)
        assert "-74.1" in config["bbox"]


# ---------------------------------------------------------------------------
# PyGeoVision main client
# ---------------------------------------------------------------------------

class TestPyGeoVisionClient:
    def test_client_creation(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        assert client is not None
        assert hasattr(client, "data")
        assert hasattr(client, "segmentation")
        assert isinstance(client.data, SatelliteFetcher)

    def test_status_structure(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        status = client.status()
        assert "pygeovision_version" in status
        assert "pygeofetch" in status
        assert "torch" in status
        assert isinstance(status["pygeofetch"]["providers"], int)
        assert status["pygeofetch"]["providers"] == 22

    def test_list_providers_all(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        providers = client.list_providers()
        assert len(providers) == 22

    def test_list_providers_open_only(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        providers = client.list_providers(open_only=True)
        for p in providers.values():
            assert p.get("open"), f"Provider {p['name']} not open"

    def test_list_providers_sar(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        providers = client.list_providers(capabilities=["sar"])
        for p in providers.values():
            assert p.get("sar")

    def test_add_credentials_chaining(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        client.data._pygeofetch_available = False  # No CLI in test env
        result = (
            client
            .add_credentials("usgs", username="u", password="p")
            .add_credentials("planet", api_key="PL_KEY")
        )
        assert result is client
        assert "usgs" in client.data._credentials
        assert "planet" in client.data._credentials

    def test_native_ai_layers_present(self):
        from pygeovision import (
            PyGeoVision,
            _ChangeDetectionClientProxy,
            _ClassificationClientProxy,
            _DetectionClientProxy,
            _SegmentationClientProxy,
        )
        client = PyGeoVision()
        assert isinstance(client.segmentation, _SegmentationClientProxy)
        assert isinstance(client.detection, _DetectionClientProxy)
        assert isinstance(client.change, _ChangeDetectionClientProxy)
        assert isinstance(client.classification, _ClassificationClientProxy)

    def test_native_ai_layers_stable(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        assert client.segmentation is client.segmentation  # Same instance

    def test_create_pipeline(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        p = client.create_pipeline("my-pipeline", description="Test pipeline")
        assert isinstance(p, DataPipeline)
        assert p.name == "my-pipeline"

    def test_repr(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        r = repr(client)
        assert "PyGeoVision" in r
        assert "pygeofetch=" in r
        assert "ai=" in r

    def test_cache_stats_returns_dict(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        client.data._pygeofetch_available = False
        stats = client.cache_stats()
        assert isinstance(stats, dict)
        assert isinstance(stats, dict) and len(stats) > 0  # pygeofetch or local cache stats

    def test_provider_info(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        info = client.data.provider_info("planetary_computer")
        assert info["name"] == "Microsoft Planetary Computer"
        assert info["open"] is True
        assert info["stac"] is True


class TestPlanetaryComputerDirectDownloadSigning:
    """Regression tests for a real bug: _download_stac_item_direct fetched
    Planetary Computer asset URLs directly, without SAS-token signing.
    PC's Blob Storage assets are private and always require signing
    (`planetary_computer.sign()`) — a raw href 404s/409s
    ('Public access is not permitted on this storage account')."""

    def _make_item(self, url="https://sentinel2l2a01.blob.core.windows.net/x/B04.tif"):
        return SearchResult(
            id="S2A_MSIL2A_20240128T101301_test",
            provider="planetary_computer",
            satellite="sentinel-2",
            datetime="2024-01-28T10:13:01Z",
            cloud_cover=0.0,
            bbox=(-0.25, 5.52, -0.20, 5.60),
            assets={"B04": {"href": url}},
        )

    def test_download_signs_pc_url_before_requesting(self, tmp_path, monkeypatch):
        fetcher = SatelliteFetcher()
        item = self._make_item()

        pc = pytest.importorskip("planetary_computer", reason="planetary-computer not installed")
        signed = "https://sentinel2l2a01.blob.core.windows.net/x/B04.tif?st=SIGNED&sig=abc"
        monkeypatch.setattr(pc, "sign", lambda u: signed)

        captured = {}

        class _FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def iter_content(self, chunk_size):
                return [b"fake-tif-bytes"]

        class _FakeSession:
            def __init__(self):
                self.headers = MagicMock()
            def get(self, url, stream=None, timeout=None, allow_redirects=None):
                captured["url"] = url
                return _FakeResp()

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "Session", lambda: _FakeSession())

        result = fetcher._download_stac_item_direct(
            item, output_dir=tmp_path, post_process=None, bandwidth_limit_mb=None,
        )

        assert captured["url"] == signed
        assert result.success is True

    def test_download_fails_gracefully_without_planetary_computer_installed(
        self, tmp_path, monkeypatch,
    ):
        """If the optional `planetary-computer` package isn't installed, the
        download should still proceed (with a clear warning) rather than
        crashing with an unrelated ImportError deep in the call stack."""
        fetcher = SatelliteFetcher()
        item = self._make_item()

        import sys
        monkeypatch.setitem(sys.modules, "planetary_computer", None)

        class _FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def iter_content(self, chunk_size):
                return [b"fake-tif-bytes"]

        class _FakeSession:
            def __init__(self):
                self.headers = MagicMock()
            def get(self, url, stream=None, timeout=None, allow_redirects=None):
                return _FakeResp()

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "Session", lambda: _FakeSession())

        result = fetcher._download_stac_item_direct(
            item, output_dir=tmp_path, post_process=None, bandwidth_limit_mb=None,
        )
        assert result.success is True  # doesn't crash even without the optional package

    def _make_multiband_item(self, bands=("B02", "B03", "B04", "B08")):
        base = "https://sentinel2l2a01.blob.core.windows.net/x"
        return SearchResult(
            id="S2A_MSIL2A_20240128T101301_multiband",
            provider="planetary_computer",
            satellite="sentinel-2",
            datetime="2024-01-28T10:13:01Z",
            cloud_cover=0.0,
            bbox=(-0.25, 5.52, -0.20, 5.60),
            assets={b: {"href": f"{base}/{b}.tif"} for b in bands},
        )

    def test_all_requested_bands_are_downloaded_not_just_one(self, tmp_path, monkeypatch):
        """Regression test for a real bug: only the first-priority band
        (always B04) was ever downloaded, silently ignoring every other
        band the caller requested via client.download(bands=[...])."""
        import rasterio

        fetcher = SatelliteFetcher()
        item = self._make_multiband_item(("B02", "B03", "B04", "B08"))

        pc = pytest.importorskip("planetary_computer", reason="planetary-computer not installed")
        monkeypatch.setattr(pc, "sign", lambda u: u + "?st=SIGNED")

        requested_urls = []

        def _make_fake_tif(path):
            from rasterio.transform import from_bounds
            transform = from_bounds(-0.25, 5.52, -0.20, 5.60, 8, 8)
            with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((1, 8, 8), 100, dtype="uint16"))

        class _FakeSession:
            def __init__(self):
                self.headers = MagicMock()

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "Session", lambda: _FakeSession())

        # Patch the streaming helper to just materialize a real tiny GeoTIFF
        # and record the (already-signed) URL it was asked to fetch, since
        # constructing real streamed TIF bytes through a fake HTTP response
        # isn't practical here.
        def fake_stream(self, session, url, out_path, provider, item_id, headers, bw):
            requested_urls.append(url)
            _make_fake_tif(out_path)
            return 128, None
        monkeypatch.setattr(SatelliteFetcher, "_stream_download_asset", fake_stream)

        result = fetcher._download_stac_item_direct(
            item, output_dir=tmp_path, post_process=None, bandwidth_limit_mb=None,
        )

        assert result.success is True
        # All 4 bands must have been requested, not just B04
        assert len(requested_urls) == 4
        assert all("?st=SIGNED" in u for u in requested_urls)

        # The final file must be a stacked 4-band raster, not a single band
        with rasterio.open(result.path) as src:
            assert src.count == 4

    def test_single_band_request_skips_stacking(self, tmp_path, monkeypatch):
        """A single requested band shouldn't go through the stack_bands path
        — it should be used directly as the final output."""
        import rasterio
        from rasterio.transform import from_bounds

        fetcher = SatelliteFetcher()
        item = self._make_multiband_item(("B04",))

        pc = pytest.importorskip("planetary_computer", reason="planetary-computer not installed")
        monkeypatch.setattr(pc, "sign", lambda u: u + "?st=SIGNED")

        def fake_stream(self, session, url, out_path, provider, item_id, headers, bw):
            transform = from_bounds(-0.25, 5.52, -0.20, 5.60, 8, 8)
            with rasterio.open(out_path, "w", driver="GTiff", height=8, width=8, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((1, 8, 8), 100, dtype="uint16"))
            return 64, None
        monkeypatch.setattr(SatelliteFetcher, "_stream_download_asset", fake_stream)

        class _FakeSession:
            def __init__(self):
                self.headers = MagicMock()
            def get(self, *a, **kw):
                raise AssertionError("session.get should not be called directly when _stream_download_asset is patched")

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "Session", lambda: _FakeSession())

        result = fetcher._download_stac_item_direct(
            item, output_dir=tmp_path, post_process=None, bandwidth_limit_mb=None,
        )
        assert result.success is True
        assert "_B04.tif" in str(result.path)
        with rasterio.open(result.path) as src:
            assert src.count == 1

    def test_extension_detected_from_url_path_not_raw_signed_url(self):
        """Regression test for a real bug: extension was checked via
        `url.endswith('.tif')` AFTER SAS-signing appended `?st=...&sig=...`,
        so every signed .tif asset was misnamed with a .zip extension."""
        signed_tif_url = "https://sentinel2l2a01.blob.core.windows.net/x/B04.tif?st=2024&se=2025&sig=abc123"
        url_path = signed_tif_url.split("?", 1)[0]
        assert url_path.endswith(".tif")
        # The old buggy check would have failed here:
        assert not signed_tif_url.endswith(".tif")

    def test_post_process_step_warns_if_it_silently_drops_bands(self, tmp_path, caplog):
        """If a post-process step (e.g. cog) unexpectedly collapses a
        multi-band integer raster to fewer bands or a float dtype, that
        must be surfaced immediately with a clear warning naming the step
        responsible — not left to surface only much later in a downstream
        validation report, far from the actual cause."""
        import logging
        import rasterio
        from rasterio.transform import from_bounds

        fetcher = SatelliteFetcher()

        src_path = tmp_path / "scene.tif"
        transform = from_bounds(818000, 615000, 818640, 615640, 16, 16)
        with rasterio.open(src_path, "w", driver="GTiff", height=16, width=16, count=4,
                            dtype="uint16", crs="EPSG:32630", transform=transform) as dst:
            dst.write(np.full((4, 16, 16), 1000, dtype="uint16"))

        with caplog.at_level(logging.WARNING):
            fetcher._warn_if_bands_or_dtype_changed(
                "cog", src_path, expected_count=4, expected_dtype="uint16",
            )
        # Same file, same shape — must NOT warn when nothing actually changed
        assert not any("changed the raster unexpectedly" in r.message for r in caplog.records)

        # Now simulate the step having collapsed it to 1 band / float32
        corrupted_path = tmp_path / "scene_corrupted.tif"
        with rasterio.open(corrupted_path, "w", driver="GTiff", height=16, width=16, count=1,
                            dtype="float32", crs="EPSG:32630", transform=transform) as dst:
            dst.write(np.full((1, 16, 16), 1000.0, dtype="float32"))

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            fetcher._warn_if_bands_or_dtype_changed(
                "cog", corrupted_path, expected_count=4, expected_dtype="uint16",
            )
        assert any("changed the raster unexpectedly" in r.message for r in caplog.records)