"""The tile Microsoft Buildings labeler must burn footprints on the tile's own (projected) grid.

Defect fixed: footprints were burned on a lon/lat envelope of the tile, so on a UTM tile the labels were
skewed against the imagery.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
gpd = pytest.importorskip("geopandas", reason="geopandas not installed")
shapely = pytest.importorskip("shapely", reason="shapely not installed")

from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402
from shapely.geometry import box  # noqa: E402

from pygeovision.ai.data.dataset import TileMetadata  # noqa: E402
from pygeovision.ai.labeling.microsoft_buildings import MicrosoftBuildingsLabeler  # noqa: E402


def test_footprint_lands_on_the_tile_grid(tmp_path, monkeypatch):
    (x0,), (y0,) = warp_transform("EPSG:4326", "EPSG:32632", [10.0], [50.01])
    tf = from_origin(x0, y0, 10.0, 10.0)
    tile = TileMetadata(
        tile_id="t0", source_file=tmp_path / "s.tif", bounds=(x0, y0 - 1000, x0 + 1000, y0), crs="EPSG:32632",
        transform=list(tf)[:6], row_off=0, col_off=0, height=100, width=100, bands=[1], extra={},
    )
    gdf = gpd.GeoDataFrame({"geometry": [box(10.003, 50.004, 10.004, 50.005)]}, crs="EPSG:4326")
    from pygeovision.data import footprints as fp

    monkeypatch.setattr(fp, "microsoft_footprints", lambda bbox, cache_dir=None, **k: gdf)
    lab = MicrosoftBuildingsLabeler(cache_dir=tmp_path / "cache")
    res = lab.label_tile(tile, tmp_path, "segmentation")
    assert res.success, res.error
    with rasterio.open(res.label_path) as src:
        arr = src.read(1)
        assert str(src.crs).endswith("32632")
        (cx,), (cy,) = warp_transform("EPSG:4326", "EPSG:32632", [10.0035], [50.0045])
        r, c = src.index(cx, cy)
    assert arr[r, c] == 1
    assert arr.sum() > 20
