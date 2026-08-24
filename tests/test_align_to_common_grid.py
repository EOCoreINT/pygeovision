"""Tests for BasePipeline._align_to_common_grid().

Real gap found and fixed, confirmed from a production failure log:
run_pair()'s pixel-alignment check correctly caught two different
Landsat scenes (different dates, different acquisition footprints)
ending up at different pixel dimensions after being independently
stacked -- but nothing upstream ever resolved this. Both scenes were
individually correct (their own bands were internally aligned), but the
PAIR was never aligned to each other.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds


def _make_pipeline():
    from pygeovision.ai.pipelines import BasePipeline

    class FakePipeline(BasePipeline):
        def run(self, bbox, output_dir, **kwargs):
            pass

    return FakePipeline.__new__(FakePipeline)


class TestAlignToCommonGrid:
    def test_two_different_sized_overlapping_scenes_end_up_on_identical_grid(self, tmp_path):
        """The core regression test: reproduces the real mismatch from
        the production log (different dimensions, same real bbox)."""
        pipeline = _make_pipeline()
        size_a, size_b = (60, 50), (65, 55)  # (height, width), genuinely different
        transform_a = from_bounds(-74.15, 40.55, -73.65, 40.95, size_a[1], size_a[0])
        transform_b = from_bounds(-74.12, 40.58, -73.68, 40.92, size_b[1], size_b[0])

        rng = np.random.default_rng(0)
        path_a, path_b = tmp_path / "before.tif", tmp_path / "after.tif"
        for p, size, transform in [(path_a, size_a, transform_a), (path_b, size_b, transform_b)]:
            data = rng.integers(500, 3000, (3, *size)).astype("uint16")
            with rasterio.open(p, "w", driver="GTiff", height=size[0], width=size[1], count=3,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(data)

        aligned_a, aligned_b = pipeline._align_to_common_grid(path_a, path_b, tmp_path / "aligned")

        with rasterio.open(aligned_a) as src_a, rasterio.open(aligned_b) as src_b:
            assert src_a.shape == src_b.shape
            assert src_a.transform == src_b.transform
            assert src_a.crs == src_b.crs

    def test_aligned_pair_is_accepted_by_run_pair_without_crashing(self, tmp_path):
        """End-to-end: the actual consumer of this alignment (siamese
        change-detection inference) must accept the result cleanly."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        pytest.importorskip("segmentation_models_pytorch", reason="smp not installed")
        from pygeovision.ai.models.architectures.change_detection import build_siamese_unet
        from pygeovision.ai.inference.tiled_inference import TiledInference

        pipeline = _make_pipeline()
        size_a, size_b = (60, 50), (65, 55)
        transform_a = from_bounds(-74.15, 40.55, -73.65, 40.95, size_a[1], size_a[0])
        transform_b = from_bounds(-74.12, 40.58, -73.68, 40.92, size_b[1], size_b[0])
        rng = np.random.default_rng(0)
        path_a, path_b = tmp_path / "before.tif", tmp_path / "after.tif"
        for p, size, transform in [(path_a, size_a, transform_a), (path_b, size_b, transform_b)]:
            data = rng.integers(500, 3000, (3, *size)).astype("uint16")
            with rasterio.open(p, "w", driver="GTiff", height=size[0], width=size[1], count=3,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(data)

        aligned_a, aligned_b = pipeline._align_to_common_grid(path_a, path_b, tmp_path / "aligned")

        model = build_siamese_unet(in_channels=3, num_classes=2, encoder="resnet18", pretrained=False)
        engine = TiledInference(model, tile_size=64, overlap=16)
        pred = engine.run_pair(str(aligned_a), str(aligned_b), str(tmp_path / "change.tif"), num_classes=2)
        assert pred is not None

    def test_non_overlapping_images_are_left_unchanged_not_corrupted(self, tmp_path):
        """A real, honest edge case: if the two images genuinely don't
        overlap at all, don't silently produce a degenerate result --
        leave them unchanged so the existing mismatch check reports it."""
        pipeline = _make_pipeline()
        size = (30, 30)
        transform_a = from_bounds(-80.0, 30.0, -79.0, 31.0, *size[::-1])
        transform_b = from_bounds(0.0, 0.0, 1.0, 1.0, *size[::-1])

        rng = np.random.default_rng(1)
        path_a, path_b = tmp_path / "a.tif", tmp_path / "b.tif"
        for p, transform in [(path_a, transform_a), (path_b, transform_b)]:
            data = rng.integers(500, 3000, (3, *size)).astype("uint16")
            with rasterio.open(p, "w", driver="GTiff", height=size[0], width=size[1], count=3,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(data)

        result_a, result_b = pipeline._align_to_common_grid(path_a, path_b, tmp_path / "aligned")
        assert result_a == path_a
        assert result_b == path_b

    def test_finer_native_resolution_is_preserved_not_downsampled(self, tmp_path):
        """The common grid should use the finer of the two native
        resolutions, so neither image loses real detail. Uses a square
        bbox so both images have realistic square pixels, matching how
        real reprojected satellite imagery actually looks."""
        pipeline = _make_pipeline()
        size_a, size_b = (100, 100), (40, 40)
        transform_a = from_bounds(-74.0, 40.6, -73.6, 41.0, size_a[1], size_a[0])
        transform_b = from_bounds(-74.0, 40.6, -73.6, 41.0, size_b[1], size_b[0])

        rng = np.random.default_rng(2)
        path_a, path_b = tmp_path / "hires.tif", tmp_path / "lores.tif"
        for p, size, transform in [(path_a, size_a, transform_a), (path_b, size_b, transform_b)]:
            data = rng.integers(500, 3000, (1, *size)).astype("uint16")
            with rasterio.open(p, "w", driver="GTiff", height=size[0], width=size[1], count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(data)

        aligned_a, aligned_b = pipeline._align_to_common_grid(path_a, path_b, tmp_path / "aligned")
        with rasterio.open(aligned_a) as src:
            # Should be close to the higher-resolution input's pixel count,
            # not collapsed down to the coarser image's resolution.
            assert src.height >= 90
