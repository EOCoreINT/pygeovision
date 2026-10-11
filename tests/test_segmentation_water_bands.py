"""client.segmentation.water must not guess bands for images that are not a B-G-R-NIR stack."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeovision import PyGeoVision


def _write(path, bands):
    arr = np.stack(bands).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs="EPSG:32630", transform=from_origin(0, 100, 10, 10)) as d:
        d.write(arr)


def test_rgb_image_is_refused_not_misread(tmp_path):
    p = tmp_path / "rgb.tif"
    _write(p, [np.full((10, 10), 0.3)] * 3)
    with pytest.raises(ValueError, match="green_band"):
        PyGeoVision().segmentation.water(str(p), output_path=str(tmp_path / "w.tif"))


def test_explicit_bands_work_on_rgb_plus_nir(tmp_path):
    green, nir = np.full((10, 10), 0.4), np.full((10, 10), 0.1)
    p = tmp_path / "x.tif"
    _write(p, [np.full((10, 10), 0.2), green, nir])          # R, G, NIR
    out = PyGeoVision().segmentation.water(str(p), output_path=str(tmp_path / "w.tif"), green_band=2, nir_band=3)
    assert out["water_fraction"] == 1.0


def test_default_is_band2_and_band4_of_a_four_band_stack(tmp_path):
    p = tmp_path / "s.tif"
    _write(p, [np.full((10, 10), 0.1), np.full((10, 10), 0.4), np.full((10, 10), 0.1), np.full((10, 10), 0.1)])
    assert PyGeoVision().segmentation.water(str(p), output_path=str(tmp_path / "w.tif"))["water_fraction"] == 1.0


def test_out_of_range_band_is_reported(tmp_path):
    p = tmp_path / "s.tif"
    _write(p, [np.zeros((4, 4))] * 4)
    with pytest.raises(ValueError, match="outside"):
        PyGeoVision().segmentation.water(str(p), output_path=str(tmp_path / "w.tif"), green_band=2, nir_band=9)
