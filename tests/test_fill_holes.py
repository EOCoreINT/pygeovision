"""Tests for PostProcessor.fill_holes().

Real bug found and fixed: method="bilinear" was documented as a distinct
algorithm requiring scipy, but rio_fill() (rasterio's fillnodata, an
inverse-distance-weighted method) was called unconditionally regardless
of what method was requested — "bilinear" had zero effect.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
scipy = pytest.importorskip("scipy", reason="scipy not installed")

import numpy as np
from rasterio.transform import from_bounds


def _make_holed_gradient(path, size=30):
    """A smooth linear gradient with a real hole punched in the middle —
    linear interpolation should reconstruct this near-exactly, which is
    a real, checkable property, not just 'produces some output'."""
    x, y = np.meshgrid(np.linspace(0, 10, size), np.linspace(0, 10, size))
    true_data = (x + y).astype(np.float32)
    holed = true_data.copy()
    holed[10:20, 10:20] = -9999

    transform = from_bounds(0, 0, size * 10, size * 10, size, size)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1,
                        dtype="float32", crs="EPSG:32630", transform=transform, nodata=-9999) as dst:
        dst.write(holed, 1)
    return true_data


class TestFillHolesMethodsAreGenuinelyDifferent:
    def test_bilinear_differs_from_nearest(self, tmp_path):
        from pygeovision.data.postprocess import PostProcessor
        raster_path = tmp_path / "holed.tif"
        _make_holed_gradient(raster_path)

        post = PostProcessor()
        out_nearest = tmp_path / "nearest.tif"
        out_bilinear = tmp_path / "bilinear.tif"
        post.fill_holes(str(raster_path), str(out_nearest), method="nearest")
        post.fill_holes(str(raster_path), str(out_bilinear), method="bilinear")

        with rasterio.open(out_nearest) as src:
            nearest = src.read(1)[10:20, 10:20]
        with rasterio.open(out_bilinear) as src:
            bilinear = src.read(1)[10:20, 10:20]

        assert np.abs(nearest - bilinear).mean() > 1e-4, (
            "bilinear produces identical output to nearest — method has no effect"
        )

    def test_bilinear_accurately_reconstructs_a_linear_gradient(self, tmp_path):
        """Linear interpolation should be near-exact on a genuinely
        linear function — a real, checkable correctness property."""
        from pygeovision.data.postprocess import PostProcessor
        raster_path = tmp_path / "holed.tif"
        true_data = _make_holed_gradient(raster_path)

        post = PostProcessor()
        out_path = tmp_path / "bilinear.tif"
        post.fill_holes(str(raster_path), str(out_path), method="bilinear")

        with rasterio.open(out_path) as src:
            filled = src.read(1)

        error = np.abs(filled[10:20, 10:20] - true_data[10:20, 10:20]).mean()
        assert error < 0.1, f"bilinear reconstruction of a linear gradient has real error: {error}"

    def test_no_holes_returns_data_unchanged(self, tmp_path):
        from pygeovision.data.postprocess import PostProcessor
        size = 10
        data = np.random.default_rng(0).random((size, size)).astype(np.float32)
        raster_path = tmp_path / "no_holes.tif"
        transform = from_bounds(0, 0, size * 10, size * 10, size, size)
        with rasterio.open(raster_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="float32", crs="EPSG:32630", transform=transform, nodata=-9999) as dst:
            dst.write(data, 1)

        post = PostProcessor()
        out_path = tmp_path / "filled.tif"
        post.fill_holes(str(raster_path), str(out_path), method="bilinear")

        with rasterio.open(out_path) as src:
            result = src.read(1)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_missing_scipy_gives_clear_error_for_bilinear(self, tmp_path, monkeypatch):
        from pygeovision.data.postprocess import PostProcessor
        raster_path = tmp_path / "holed.tif"
        _make_holed_gradient(raster_path)

        import builtins
        real_import = builtins.__import__
        def blocking_import(name, *args, **kwargs):
            if name == "scipy.interpolate":
                raise ImportError("simulated missing scipy")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", blocking_import)

        post = PostProcessor()
        with pytest.raises(ImportError, match="scipy"):
            post.fill_holes(str(raster_path), str(tmp_path / "out.tif"), method="bilinear")
