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
        mock_pgv.search.return_value = [MagicMock(spec=SearchResult)]
        mock_pgv.download.return_value = [fake_download_result]

        pipeline = LandCoverPipeline(mock_pgv)
        requested_bbox = (-0.45, 5.50, 0.05, 5.75)
        img_path = pipeline._search_and_download(requested_bbox, "2024-01", tmp_path / "out")

        with rasterio.open(img_path) as src:
            assert src.width < size and src.height < size


class TestCarbonEstimationRealNDVI:
    def test_requests_red_and_nir_bands_not_rgb(self, tmp_path):
        from pygeovision.ai.pipelines import CarbonEstimationPipeline

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        stack_path = tmp_path / "stack.tif"
        with rasterio.open(stack_path, "w", driver="GTiff", height=size, width=size, count=2,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.full((size, size), 0.05, dtype="float32"), 1)
            dst.write(np.full((size, size), 0.45, dtype="float32"), 2)

        pipeline = CarbonEstimationPipeline.__new__(CarbonEstimationPipeline)
        pipeline.pgv = MagicMock()
        pipeline._search_and_download = MagicMock(return_value=stack_path)

        pipeline.run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir=str(tmp_path / "out"), date="2024-01")

        call_kwargs = pipeline._search_and_download.call_args.kwargs
        assert call_kwargs.get("bands") == ("red", "nir")

    def test_carbon_value_matches_real_ndvi_formula_not_red_reflectance(self, tmp_path):
        from pygeovision.ai.pipelines import CarbonEstimationPipeline

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        red_val, nir_val = 0.05, 0.45
        expected_ndvi = (nir_val - red_val) / (nir_val + red_val)
        expected_carbon = np.clip(50.0 * expected_ndvi ** 2, 0, 500) * 0.47

        stack_path = tmp_path / "stack.tif"
        with rasterio.open(stack_path, "w", driver="GTiff", height=size, width=size, count=2,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.full((size, size), red_val, dtype="float32"), 1)
            dst.write(np.full((size, size), nir_val, dtype="float32"), 2)

        pipeline = CarbonEstimationPipeline.__new__(CarbonEstimationPipeline)
        pipeline.pgv = MagicMock()
        pipeline._search_and_download = MagicMock(return_value=stack_path)

        result = pipeline.run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir=str(tmp_path / "out"), date="2024-01")

        with rasterio.open(result.output_path) as src:
            carbon = src.read(1)
        assert abs(carbon[0, 0] - expected_carbon) < 0.01

        # The old, buggy behavior would have used red_val directly as "ndvi"
        buggy_carbon = np.clip(50.0 * red_val ** 2, 0, 500) * 0.47
        assert abs(carbon[0, 0] - buggy_carbon) > 1.0, "result should differ substantially from the old buggy formula"


class TestWaterBodiesRealNDWI:
    def test_requests_green_and_nir_bands_not_rgb(self, tmp_path):
        from pygeovision.ai.pipelines import WaterBodiesPipeline

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        stack_path = tmp_path / "stack.tif"
        with rasterio.open(stack_path, "w", driver="GTiff", height=size, width=size, count=2,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.full((size, size), 0.08, dtype="float32"), 1)
            dst.write(np.full((size, size), 0.02, dtype="float32"), 2)

        pipeline = WaterBodiesPipeline.__new__(WaterBodiesPipeline)
        pipeline.pgv = MagicMock()
        pipeline._search_and_download = MagicMock(return_value=stack_path)

        pipeline.run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir=str(tmp_path / "out"),
                     date="2024-01", method="ndwi")

        call_kwargs = pipeline._search_and_download.call_args.kwargs
        assert call_kwargs.get("bands") == ("green", "nir")

    def test_real_water_signature_is_correctly_detected(self, tmp_path):
        """Real water: moderate green, very low NIR (strong NIR
        absorption) -- must produce a real NDWI > 0.3 and be detected."""
        from pygeovision.ai.pipelines import WaterBodiesPipeline

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        green_val, nir_val = 0.08, 0.02
        expected_ndwi = (green_val - nir_val) / (green_val + nir_val)

        stack_path = tmp_path / "stack.tif"
        with rasterio.open(stack_path, "w", driver="GTiff", height=size, width=size, count=2,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.full((size, size), green_val, dtype="float32"), 1)
            dst.write(np.full((size, size), nir_val, dtype="float32"), 2)

        pipeline = WaterBodiesPipeline.__new__(WaterBodiesPipeline)
        pipeline.pgv = MagicMock()
        pipeline._search_and_download = MagicMock(return_value=stack_path)

        result = pipeline.run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir=str(tmp_path / "out"),
                               date="2024-01", method="ndwi")

        assert result.stats["water_coverage"] == 1.0
        with rasterio.open(result.output_path) as src:
            ndwi_out = src.read(1)
        assert abs(ndwi_out[0, 0] - expected_ndwi) < 0.001
