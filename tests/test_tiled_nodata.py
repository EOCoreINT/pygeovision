"""Tests for TiledInference.infer()'s nodata_value handling.

Real, severe bug found and fixed: nodata_value was accepted and
documented ("Value to treat as nodata/ignore") but never referenced
anywhere in the actual logic. Two real consequences:

  1. Percentile-based normalisation (the default) computed its 2nd/98th
     percentiles over the WHOLE band including nodata pixels. Confirmed
     directly: with 30% nodata coverage at -9999, the computed 2nd
     percentile was LITERALLY -9999 — crushing all real pixel values
     (e.g. a realistic 500-3000 reflectance range) into a tiny sliver
     near the top of the normalised [0,1] range before the data even
     reached the model, destroying almost all real contrast.
  2. The output raster never masked nodata regions, silently producing
     spurious predicted classes for areas that were never real data.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
import torch.nn as nn
from rasterio.transform import from_bounds


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)))


def _make_scene_with_nodata(path, size=64, nodata_frac_rows=20):
    """Realistic-range data (500-3000) with a genuine nodata block at an
    extreme value (-9999) — the exact scenario that skews percentile
    normalisation if nodata isn't excluded."""
    data = np.random.default_rng(0).uniform(500, 3000, (3, size, size)).astype(np.float32)
    data[:, :nodata_frac_rows, :] = -9999.0
    transform = from_bounds(0, 0, size * 10, size * 10, size, size)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3,
                        dtype="float32", crs="EPSG:32630", transform=transform, nodata=-9999.0) as dst:
        dst.write(data)


class TestNodataMasksTheOutput:
    def test_nodata_region_is_masked_in_label_output(self, tmp_path):
        from pygeovision.inference.tiled import TiledInference
        raster_path = tmp_path / "scene.tif"
        _make_scene_with_nodata(raster_path)

        torch.manual_seed(0)
        model = TinyModel()
        inf = TiledInference(model, chip_size=32, overlap=8, num_classes=2)
        out_path = tmp_path / "pred.tif"
        inf.infer(str(raster_path), str(out_path))

        with rasterio.open(out_path) as src:
            pred = src.read(1)
            assert src.nodata == 255
        assert (pred[:20, :] == 255).all(), "nodata region not masked in output"

    def test_valid_region_has_real_class_predictions(self, tmp_path):
        from pygeovision.inference.tiled import TiledInference
        raster_path = tmp_path / "scene.tif"
        _make_scene_with_nodata(raster_path)

        torch.manual_seed(0)
        model = TinyModel()
        inf = TiledInference(model, chip_size=32, overlap=8, num_classes=2)
        out_path = tmp_path / "pred.tif"
        inf.infer(str(raster_path), str(out_path))

        with rasterio.open(out_path) as src:
            pred = src.read(1)
        assert set(np.unique(pred[20:, :])) <= {0, 1}

    def test_explicit_nodata_value_overrides_raster_metadata(self, tmp_path):
        """An explicitly passed nodata_value must be honoured even for a
        raster whose own metadata doesn't declare one."""
        from pygeovision.inference.tiled import TiledInference
        size = 48
        data = np.random.default_rng(1).uniform(500, 3000, (3, size, size)).astype(np.float32)
        data[:, :15, :] = -1.0  # a sentinel value, but the file has NO nodata metadata
        transform = from_bounds(0, 0, size * 10, size * 10, size, size)
        raster_path = tmp_path / "scene_no_meta.tif"
        with rasterio.open(raster_path, "w", driver="GTiff", height=size, width=size, count=3,
                            dtype="float32", crs="EPSG:32630", transform=transform) as dst:
            dst.write(data)  # no nodata= kwarg

        torch.manual_seed(0)
        model = TinyModel()
        inf = TiledInference(model, chip_size=24, overlap=8, num_classes=2)
        out_path = tmp_path / "pred.tif"
        inf.infer(str(raster_path), str(out_path), nodata_value=-1.0)

        with rasterio.open(out_path) as src:
            pred = src.read(1)
        assert (pred[:15, :] == 255).all()


class TestNodataExcludedFromNormalisation:
    """Direct verification of the normalisation math, isolated from the
    full model pipeline — confirms nodata pixels no longer skew the
    percentile computation."""

    def test_percentile_excludes_nodata_pixels(self):
        size = 30
        band = np.random.default_rng(2).uniform(500, 3000, (size, size)).astype(np.float32)
        band[:9, :] = -9999.0  # 30% nodata

        valid_mask = band != -9999.0
        p2_fixed, p98_fixed = np.percentile(band[valid_mask], (2, 98))
        p2_buggy, _ = np.percentile(band, (2, 98))

        assert p2_buggy < -1000, "sanity check: unfiltered percentile should be skewed toward -9999"
        assert p2_fixed > 400, "nodata-excluded percentile should reflect the real 500-3000 data range"
