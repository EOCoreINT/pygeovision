"""Areas from a lon/lat (EPSG:4326) raster must be true areas, not degrees**2 labelled as m2."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.data.postprocess import PostProcessor  # noqa: E402


@pytest.fixture()
def pred(tmp_path):
    p = tmp_path / "pred.tif"
    arr = np.zeros((40, 40), dtype="uint8")
    arr[10:20, 10:20] = 1                      # 100 pixels of class 1
    arr[0:2, :] = 255                          # nodata
    with rasterio.open(p, "w", driver="GTiff", height=40, width=40, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(-0.15, 51.50, 0.0001, 0.0001), nodata=255) as dst:
        dst.write(arr, 1)
    return str(p)


def test_class_statistics_uses_true_area(pred):
    stats = PostProcessor().class_statistics(pred)
    ha = stats[1]["area_ha"]
    # 0.0001 deg at 51.5 N is ~11.12 m x ~6.92 m = ~77 m2 per pixel -> 100 pixels ~ 0.77 ha (not 1e-6 ha)
    assert 0.70 < ha < 0.85
    assert stats[1]["pixels"] == 100 and 255 not in stats


def test_class_statistics_projected_unchanged(tmp_path):
    p = tmp_path / "m.tif"
    arr = np.ones((10, 10), dtype="uint8")
    with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8", crs="EPSG:32630",
                       transform=from_origin(500000, 5700000, 10, 10)) as dst:
        dst.write(arr, 1)
    assert PostProcessor().class_statistics(str(p))[1]["area_ha"] == 1.0     # 100 px x 100 m2


def test_vectorise_area_and_min_area_are_in_square_metres(pred, tmp_path):
    out = tmp_path / "v.geojson"
    PostProcessor().vectorise(pred, str(out), target_class=1, min_area_m2=0)
    feats = json.loads(out.read_text())["features"]
    assert len(feats) == 1
    assert 7000 < feats[0]["properties"]["area_m2"] < 8500
    PostProcessor().vectorise(pred, str(out), target_class=1, min_area_m2=9000)    # larger than the polygon
    assert json.loads(out.read_text())["features"] == []


def test_geojson_unit_detection():
    from pygeovision.data.postprocess import _geojson_is_geographic
    deg = {"features": [{"geometry": {"type": "Polygon", "coordinates": [[[-0.1, 51.5], [-0.09, 51.5], [-0.09, 51.51], [-0.1, 51.5]]]}}]}
    metres = {"features": [{"geometry": {"type": "Polygon", "coordinates": [[[500000, 5700000], [500010, 5700000], [500010, 5700010], [500000, 5700000]]]}}]}
    local = {"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [20, 0], [20, 10], [0, 0]]]}}]}
    assert _geojson_is_geographic(deg) is True
    assert _geojson_is_geographic(metres) is False
    assert _geojson_is_geographic(local) is False


def test_vectorise_target_class_property_is_the_class(tmp_path):
    import json

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from pygeovision.data.postprocess import PostProcessor

    arr = np.zeros((20, 20), "uint8")
    arr[2:8, 2:8] = 2
    arr[10:16, 10:16] = 3
    src = tmp_path / "c.tif"
    with rasterio.open(src, "w", driver="GTiff", width=20, height=20, count=1, dtype="uint8", crs="EPSG:32630",
                       transform=from_origin(500000, 600000, 10, 10)) as d:
        d.write(arr, 1)
    out = tmp_path / "v.geojson"
    PostProcessor().vectorise(str(src), str(out), target_class=3)
    feats = json.loads(out.read_text())["features"]
    assert [f["properties"]["class"] for f in feats] == [3]
    assert feats[0]["properties"]["area_m2"] == pytest.approx(6 * 6 * 100)
