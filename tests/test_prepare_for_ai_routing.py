"""prepare_for_ai must not route a call to the pygeofetch engine when that engine would drop an argument."""
from unittest import mock

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.data.pgf_bridge import PyGeoFetchBridge  # noqa: E402


@pytest.fixture()
def tif(tmp_path):
    p = tmp_path / "scene.tif"
    with rasterio.open(p, "w", driver="GTiff", width=8, height=8, count=2, dtype="uint16",
                       crs="EPSG:4326", transform=from_origin(0, 1, 0.01, 0.01)) as dst:
        dst.write(np.full((2, 8, 8), 1000, "uint16"))
    return str(p)


def _bridge():
    b = PyGeoFetchBridge()
    b._pgf_v2 = True
    return b


@pytest.mark.parametrize("kw", [
    {"clip_geojson": "aoi.geojson"},
    {"cloud_mask_path": "mask.tif"},
    {"scl_keep_classes": (4, 5)},
    {"stack_bands": ["B02", "B03"]},
])
def test_unsupported_arguments_use_native_route(tif, tmp_path, kw):
    b = _bridge()
    b._pre = mock.Mock()
    b._pre.pipeline.return_value = {"steps_applied": ["x"], "shape": (2, 8, 8), "resolution_m": 1.0}
    with mock.patch.object(b, "_prepare_via_pgf_v2") as v2:
        b.prepare_for_ai(tif, output_path=str(tmp_path / "out.tif"), **kw)
    v2.assert_not_called()
    assert b._pre.pipeline.called


def test_plain_call_uses_pygeofetch_route(tif, tmp_path):
    b = _bridge()
    with mock.patch.object(b, "_prepare_via_pgf_v2", return_value="ok") as v2:
        out = b.prepare_for_ai(tif, bbox=(0, 0.9, 0.05, 1.0), output_path=str(tmp_path / "out.tif"))
    v2.assert_called_once()
    assert out == "ok"
