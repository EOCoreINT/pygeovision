"""zonal_statistics must return real values for integer rasters, nodata=0 and zones in another CRS."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.data.postprocess import PostProcessor  # noqa: E402


def _write(path, arr, crs, transform, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype=str(arr.dtype), crs=crs, transform=transform, nodata=nodata) as d:
        d.write(arr, 1)


def _zone(tmp_path, ring):
    p = tmp_path / "z.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": 1}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
    return str(p)


def test_integer_class_raster_gives_values(tmp_path):
    arr = np.zeros((20, 20), "uint8")
    arr[5:15, 5:15] = 3
    _write(tmp_path / "c.tif", arr, "EPSG:4326", from_origin(0, 0.002, 0.0001, 0.0001), nodata=255)
    ring = [[0.0005, 0.0015], [0.0015, 0.0015], [0.0015, 0.0005], [0.0005, 0.0005], [0.0005, 0.0015]]
    out = PostProcessor().zonal_statistics(str(tmp_path / "c.tif"), _zone(tmp_path, ring), stats=["mean", "count"])
    assert out[0]["properties"]["stat_count"] == 100
    assert out[0]["properties"]["stat_mean"] == pytest.approx(3.0)


def test_nodata_zero_is_excluded(tmp_path):
    arr = np.zeros((10, 10), "float32")
    arr[:, :5] = 2.0
    _write(tmp_path / "f.tif", arr, "EPSG:4326", from_origin(0, 0.001, 0.0001, 0.0001), nodata=0.0)
    ring = [[0, 0.001], [0.001, 0.001], [0.001, 0], [0, 0], [0, 0.001]]
    out = PostProcessor().zonal_statistics(str(tmp_path / "f.tif"), _zone(tmp_path, ring), stats=["mean", "count"])
    assert out[0]["properties"]["stat_mean"] == pytest.approx(2.0)
    assert out[0]["properties"]["stat_count"] == 50


def test_lonlat_zone_over_projected_raster(tmp_path):
    from pyproj import Transformer
    x0, y0 = Transformer.from_crs(4326, 32630, always_xy=True).transform(-0.2, 5.6)
    arr = np.full((50, 50), 7.0, "float32")
    _write(tmp_path / "u.tif", arr, "EPSG:32630", from_origin(x0, y0, 10, 10), nodata=-9999.0)
    lo, la = Transformer.from_crs(32630, 4326, always_xy=True).transform(x0 + 100, y0 - 100)
    lo2, la2 = Transformer.from_crs(32630, 4326, always_xy=True).transform(x0 + 300, y0 - 300)
    ring = [[lo, la], [lo2, la], [lo2, la2], [lo, la2], [lo, la]]
    out = PostProcessor().zonal_statistics(str(tmp_path / "u.tif"), _zone(tmp_path, ring), stats=["mean", "count"])
    assert out[0]["properties"]["stat_mean"] == pytest.approx(7.0)
    assert out[0]["properties"]["stat_count"] > 100
