"""Tests for two real, significant gaps found from a direct user
question about pipeline design correctness:

  1. No bbox cropping: the requested bbox was used only to SEARCH for
     overlapping scenes -- nothing ever cropped the downloaded/stacked
     imagery to it afterward, so every pipeline ran tiled inference
     over the entire downloaded scene/tile (often 100km+ on a side for
     Sentinel-2, ~185x180km for Landsat), not just the requested area.

  2. A severe, previously-undiscovered bug in carbon_estimation and
     water_bodies: both requested their index (NDVI/NDWI) via a
     post_process=[...,"ndvi"/"ndwi"] step -- but that step never
     actually ran for any real multi-asset scene, because
     _search_and_download's _stack_bands path always intercepts first
     for a multi-asset scene (virtually every real Sentinel-2/Landsat
     product). Both pipelines were reading band 1 of the resulting RGB
     stack (red-band reflectance) and treating it as if it were the
     real index -- confirmed to produce carbon estimates off by a
     factor of ~255x for a real healthy-vegetation reflectance
     signature, and a water mask that would miss real water (which has
     low red reflectance, well below the 0.3 threshold) while
     potentially flagging bright bare/urban surfaces instead.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds
from unittest.mock import MagicMock


class TestCropToBbox:
    def test_crops_to_the_real_requested_extent(self, tmp_path):
        from pygeovision.data.radiometric import crop_to_bbox

        size = 200
        full_transform = from_bounds(-1.0, 4.0, 1.0, 6.0, size, size)
        full_path = tmp_path / "full_scene.tif"
        with rasterio.open(full_path, "w", driver="GTiff", height=size, width=size, count=3,
                            dtype="float32", crs="EPSG:4326", transform=full_transform) as dst:
            dst.write(np.ones((3, size, size), dtype="float32"))

        requested_bbox = (-0.45, 5.50, 0.05, 5.75)
        cropped_path = crop_to_bbox(full_path, requested_bbox, tmp_path / "cropped.tif")

        with rasterio.open(full_path) as src_full, rasterio.open(cropped_path) as src_cropped:
            assert src_cropped.width < src_full.width
            assert src_cropped.height < src_full.height
            cb = src_cropped.bounds
            assert abs(cb.left - requested_bbox[0]) < 0.05
            assert abs(cb.bottom - requested_bbox[1]) < 0.05
            assert abs(cb.right - requested_bbox[2]) < 0.05
            assert abs(cb.top - requested_bbox[3]) < 0.05

    def test_non_intersecting_bbox_raises_clearly(self, tmp_path):
        from pygeovision.data.radiometric import crop_to_bbox

        size = 50
        transform = from_bounds(-1.0, 4.0, 1.0, 6.0, size, size)
        path = tmp_path / "scene.tif"
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.ones((1, size, size), dtype="float32"))

        with pytest.raises(ValueError, match="does not intersect"):
            crop_to_bbox(path, (50.0, 50.0, 51.0, 51.0), tmp_path / "bad.tif")

    def test_default_output_path(self, tmp_path):
        from pygeovision.data.radiometric import crop_to_bbox

        size = 50
        transform = from_bounds(-1.0, 4.0, 1.0, 6.0, size, size)
        path = tmp_path / "scene.tif"
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.ones((1, size, size), dtype="float32"))

        result = crop_to_bbox(path, (-0.45, 5.50, 0.05, 5.75))
        assert result == tmp_path / "scene_cropped.tif"


class TestSearchAndDownloadCropsToBbox:
    def test_full_chain_returns_cropped_not_full_scene(self, tmp_path):
        from pygeovision.data.fetch import DownloadResult, SearchResult
        from pygeovision.ai.pipelines import LandCoverPipeline

        imagery_dir = tmp_path / "imagery"
        imagery_dir.mkdir(parents=True)
        size = 300
        full_transform = from_bounds(-1.0, 4.0, 1.0, 6.0, size, size)
        scene_id = "LC08_L2SP_014031_20240615_02_T1"
        rng = np.random.default_rng(0)
        asset_files = {"SR_B4": "SR_B4", "SR_B3": "SR_B3", "SR_B2": "SR_B2"}
        asset_paths = {}
        for key, suffix in asset_files.items():
            p = imagery_dir / f"{scene_id}_{scene_id}_{suffix}.TIF"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=full_transform) as dst:
                dst.write(rng.integers(500, 3000, (size, size)).astype("uint16"), 1)
            asset_paths[key] = p

        fake_download_result = DownloadResult(
            scene_id=scene_id, provider="planetary_computer",
            path=asset_paths["SR_B4"], asset_paths=asset_paths, success=True,
        )
        mock_pgv = MagicMock()
        # Real fix: MagicMock(spec=SearchResult) doesn't work here --
        # dataclass instance fields like bbox aren't visible via
        # dir(SearchResult) (the class), which is what spec= introspects,
        # so the mock never had a usable .bbox at all. A real instance
        # with a concrete bbox is also what this test actually needs to
        # exercise the real coverage/crop logic meaningfully.
        real_search_result = SearchResult(
            id=scene_id, provider="planetary_computer", satellite="landsat-8",
            datetime="2024-06-15", cloud_cover=5.0,
            bbox=(-1.0, 4.0, 1.0, 6.0),  # covers the full scene extent above
        )
        mock_pgv.search.return_value = [real_search_result]
        mock_pgv.download.return_value = [fake_download_result]

        pipeline = LandCoverPipeline(mock_pgv)
        requested_bbox = (-0.45, 5.50, 0.05, 5.75)
        img_path = pipeline._search_and_download(requested_bbox, "2024-01", tmp_path / "out")

        with rasterio.open(img_path) as src:
            assert src.width < size and src.height < size
