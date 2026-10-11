"""The AI-readiness gate must accept lon/lat rasters (pixel size in degrees) and still reject broken transforms."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine, from_origin  # noqa: E402

from pygeovision.data.ai_prep import DataValidator  # noqa: E402


def _write(path, crs, transform, n=32):
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=2, dtype="float32",
                       crs=crs, transform=transform) as d:
        d.write(np.random.default_rng(0).random((2, n, n), dtype="float32"))
    return str(path)


def test_lonlat_raster_is_accepted(tmp_path):
    p = _write(tmp_path / "a.tif", "EPSG:4326", from_origin(-0.2, 5.6, 0.0001, 0.0001))
    arr, rep = DataValidator().validate_for_ai(p, expected_bands=2)
    assert rep["valid"] and arr is not None
    assert 10 < rep["pixel_size_m"] < 12


def test_identity_transform_is_rejected(tmp_path):
    p = _write(tmp_path / "b.tif", "EPSG:4326", Affine.identity())
    arr, rep = DataValidator().validate_for_ai(p)
    assert arr is None and any("plausible" in e for e in rep["errors"])


def test_projected_raster_is_accepted(tmp_path):
    p = _write(tmp_path / "c.tif", "EPSG:32630", from_origin(500000, 600000, 10, 10))
    arr, rep = DataValidator().validate_for_ai(p, expected_bands=2)
    assert rep["valid"] and rep["pixel_size_m"] == 10
