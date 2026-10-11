"""building_footprints (published-dataset mode) and road_network: counts, areas, clipping, density rasters."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, box

from pygeovision.ai.pipelines import BuildingFootprintsPipeline
from pygeovision.ai.pipelines.vector_products import RoadNetworkPipeline, length_per_cell
from pygeovision.data import footprints as fp
from pygeovision.data.reference import make_grid

BBOX = (0.45, 5.90, 0.47, 5.92)          # about 2.2 km square
D = 0.0001                               # degrees; about 11.07 m x 11.06 m at this latitude = about 122.4 m2
ONE_M2 = 122.4


def bldg(lon, lat, k=1):
    return box(lon, lat, lon + k * D, lat + k * D)


def frame(geoms, **cols):
    return gpd.GeoDataFrame(cols or {"id": list(range(len(geoms)))}, geometry=geoms, crs="EPSG:4326")


class TestBuildings:
    @pytest.fixture
    def raw(self):
        # place the buildings relative to the middle of a real 100 m grid cell so none sits on a cell edge
        from pyproj import Transformer
        grid = make_grid(BBOX, 100.0)
        to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(grid.transform.c + 10.5 * 100, grid.transform.f - 10.5 * 100)
        lon2, lat2 = to_ll.transform(grid.transform.c + 5.5 * 100, grid.transform.f - 12.5 * 100)
        g = [bldg(lon - 3 * D, lat - 3 * D), bldg(lon + D, lat + D, 2),     # two in the same cell (1 + 4 units)
             bldg(lon2, lat2),                                              # another cell
             bldg(0.5000, 5.9100)]                                          # well outside the box
        return frame(g)

    def test_microsoft_mode(self, tmp_path, monkeypatch, raw):
        monkeypatch.setattr(fp, "microsoft_footprints", lambda bbox, **kw: raw)
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path, source="microsoft", cell_m=100.0)
        assert r.success, r.error
        s = r.stats
        assert s["building_count"] == 3 and s["source"] == "microsoft" and "Microsoft" in s["source_note"]
        assert s["building_area_ha"] * 1e4 == pytest.approx(ONE_M2 * (1 + 4 + 1), rel=0.02)
        assert s["buildings_per_km2"] == pytest.approx(3 / s["aoi_area_km2"])
        with rasterio.open(tmp_path / "building_count.tif") as src:
            cnt = src.read(1)
        assert np.nansum(cnt) == 3 and np.nanmax(cnt) == 2
        with rasterio.open(tmp_path / "building_density.tif") as src:
            dens = src.read(1)
        assert np.nanmax(dens) == pytest.approx(5 * ONE_M2 / 1e4, rel=0.02)
        assert s["max_cell_coverage"] == pytest.approx(np.nanmax(dens))
        layer = gpd.read_file(tmp_path / "building_footprints.gpkg", layer="buildings")
        assert len(layer) == 3 and (layer["source"] == "microsoft").all()
        assert layer["area_m2"].sum() == pytest.approx(s["building_area_ha"] * 1e4, rel=1e-3)
        assert r.metadata["vector_files"][0]["name"] == "buildings"

    def test_building_straddling_the_box_edge_is_clipped(self, tmp_path, monkeypatch):
        edge = box(BBOX[0] - 2 * D, 5.9101, BBOX[0] + 2 * D, 5.9101 + 2 * D)         # half outside
        monkeypatch.setattr(fp, "microsoft_footprints", lambda bbox, **kw: frame([edge]))
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path, cell_m=100.0)
        assert r.stats["building_count"] == 1
        assert r.stats["building_area_ha"] * 1e4 == pytest.approx(ONE_M2 * 4, rel=0.02)         # only the 2 x 2 unit half

    def test_google_passes_the_confidence_threshold(self, tmp_path, monkeypatch, raw):
        seen = {}

        def g(bbox, **kw):
            seen.update(kw)
            return raw
        monkeypatch.setattr(fp, "google_footprints", g)
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path, source="google", min_confidence=0.8)
        assert r.success and seen["min_confidence"] == 0.8 and r.stats["min_confidence"] == 0.8

    def test_osm_mode_and_heights(self, tmp_path, monkeypatch):
        g = frame([bldg(0.4601, 5.9101)], height_m=[7.5], name=["x"])
        monkeypatch.setattr(fp, "osm_features", lambda bbox, kind, **kw: g if kind == "building" else None)
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path, source="osm")
        assert r.success and r.stats["mean_height_m"] == 7.5 and "volunteered" in r.stats["source_note"]

    def test_no_buildings_is_a_valid_empty_result_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fp, "microsoft_footprints", lambda bbox, **kw: frame([]))
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path)
        assert r.success and r.stats["building_count"] == 0 and r.stats["footprints_file"] is None
        assert "vector_files" not in r.metadata
        with rasterio.open(tmp_path / "building_density.tif") as src:
            assert np.nanmax(src.read(1)) == 0.0

    def test_source_errors_become_a_failed_result_with_the_reason(self, tmp_path, monkeypatch):
        def boom(bbox, **kw):
            raise fp.FootprintSourceError("no partition covers this box")
        monkeypatch.setattr(fp, "microsoft_footprints", boom)
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path)
        assert not r.success and "no partition" in r.error

    def test_bad_source_and_cell(self, tmp_path):
        p = BuildingFootprintsPipeline(object())
        assert "source must be" in p.run(BBOX, tmp_path, source="bing").error
        assert "cell_m" in p.run(BBOX, tmp_path, cell_m=0).error

    def test_with_a_checkpoint_it_is_the_network_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fp, "microsoft_footprints", lambda *a, **k: pytest.fail("must not use the dataset when a checkpoint is given"))
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path, checkpoint=str(tmp_path / "missing.pth"))
        assert not r.success and "missing.pth" in r.error


class TestLengthPerCell:
    def test_conserves_length_and_lands_in_the_right_cells(self):
        grid = make_grid(BBOX, 500.0)
        y = grid.transform.f - 2.5 * 500                      # middle of the third row
        x0 = grid.transform.c + 100.0
        line = gpd.GeoSeries([LineString([(x0, y), (x0 + 1900, y)])], crs=grid.crs)
        acc = length_per_cell(gpd.GeoDataFrame(geometry=line), grid, 100.0)
        assert acc.sum() == pytest.approx(1900.0)
        assert (acc[2] > 0).sum() == 4 and acc[2, 0] == pytest.approx(400.0) and acc[2, 1] == pytest.approx(500.0)
        assert acc[[0, 1, 3]].sum() == 0

    def test_empty(self):
        grid = make_grid(BBOX, 500.0)
        assert length_per_cell(gpd.GeoDataFrame(geometry=[], crs=grid.crs), grid, 100.0).sum() == 0


class TestRoadNetwork:
    LAT = 5.91
    KM_PER_DEG = 111.3195 * np.cos(np.radians(5.91))                      # km per degree of longitude

    @pytest.fixture
    def osm(self):
        lines = [LineString([(0.455, self.LAT), (0.465, self.LAT)]),                 # primary, 0.01 deg
                 LineString([(0.455, 5.915), (0.460, 5.915)]),                       # residential, 0.005 deg
                 LineString([(0.44, 5.905), (0.46, 5.905)]),                         # track, crosses the west edge: 0.01 deg inside
                 LineString([(0.455, 5.902), (0.465, 5.902)])]                       # proposed: excluded
        return frame(lines, highway=["primary", "residential", "track", "proposed"], name=["Main", None, None, None],
                     surface=[None] * 4, oneway=[None] * 4, osm_id=[1, 2, 3, 4])

    def run(self, tmp_path, monkeypatch, osm, **kw):
        monkeypatch.setattr(fp, "osm_features", lambda bbox, kind, **k: osm)
        return RoadNetworkPipeline(object()).run(BBOX, tmp_path, **kw)

    def test_lengths_by_class(self, tmp_path, monkeypatch, osm):
        r = self.run(tmp_path, monkeypatch, osm)
        assert r.success, r.error
        s = r.stats
        assert s["km_by_highway"]["primary"] == pytest.approx(0.01 * self.KM_PER_DEG, rel=0.01)
        assert s["km_by_highway"]["residential"] == pytest.approx(0.005 * self.KM_PER_DEG, rel=0.01)
        assert s["km_by_highway"]["track"] == pytest.approx(0.01 * self.KM_PER_DEG, rel=0.01)    # clipped at the box edge
        assert "proposed" not in s["km_by_highway"] and s["segments"] == 3
        assert set(s["km_by_group"]) == {"major roads", "local roads", "tracks and paths"}
        assert s["total_length_km"] == pytest.approx(0.025 * self.KM_PER_DEG, rel=0.01)
        assert s["road_density_km_per_km2"] == pytest.approx(s["total_length_km"] / s["aoi_area_km2"])
        assert s["unnamed_share_of_length"] == pytest.approx(0.015 / 0.025, rel=0.01)

    def test_density_raster_conserves_length(self, tmp_path, monkeypatch, osm):
        r = self.run(tmp_path, monkeypatch, osm, cell_m=500.0)
        with rasterio.open(tmp_path / "road_density.tif") as src:
            d = src.read(1)
        assert np.nansum(d) * 0.25 == pytest.approx(r.stats["total_length_km"], rel=0.02)     # km/km2 x km2 per cell

    def test_gpkg_has_length_and_group(self, tmp_path, monkeypatch, osm):
        self.run(tmp_path, monkeypatch, osm)
        roads = gpd.read_file(tmp_path / "road_network.gpkg", layer="roads")
        assert len(roads) == 3 and {"length_m", "group", "highway"} <= set(roads.columns)
        assert roads.loc[roads.highway == "primary", "group"].iloc[0] == "major roads"

    def test_exclude_can_be_changed(self, tmp_path, monkeypatch, osm):
        r = self.run(tmp_path, monkeypatch, osm, exclude=())
        assert r.stats["segments"] == 4 and "proposed" in r.stats["km_by_highway"]

    def test_no_roads(self, tmp_path, monkeypatch):
        empty = frame([], highway=[], name=[], surface=[], oneway=[], osm_id=[])
        r = self.run(tmp_path, monkeypatch, empty)
        assert r.success and r.stats["total_length_km"] == 0 and r.stats["unnamed_share_of_length"] is None
        assert "vector_files" not in r.metadata

    def test_overpass_failure_is_reported(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise fp.FootprintSourceError("Overpass API request failed")
        monkeypatch.setattr(fp, "osm_features", boom)
        r = RoadNetworkPipeline(object()).run(BBOX, tmp_path)
        assert not r.success and "Overpass" in r.error


class TestLabelersShareTheFootprintReader:
    """The old labelers built a wrong download URL and swallowed the failure; they now use data.footprints."""

    def _tile(self, tmp_path):
        from pygeovision.ai.data.dataset import TileMetadata
        return TileMetadata(tile_id="t0", source_file=tmp_path / "s.tif", bounds=BBOX, crs="EPSG:4326",
                            transform=[0.0002, 0, BBOX[0], 0, -0.0002, BBOX[3], 0, 0, 1], row_off=0, col_off=0,
                            height=100, width=100, bands=[1])

    def test_microsoft_labeler_reports_the_source_error_instead_of_an_empty_tile(self, tmp_path, monkeypatch):
        from pygeovision.ai.labeling.microsoft_buildings import MicrosoftBuildingsLabeler

        def boom(bbox, **kw):
            raise fp.FootprintSourceError("no partition covers this box")
        monkeypatch.setattr(fp, "microsoft_footprints", boom)
        r = MicrosoftBuildingsLabeler(cache_dir=tmp_path / "c").label_tile(self._tile(tmp_path), tmp_path / "labels")
        assert r.error and "no partition" in r.error and not r.skipped

    def test_microsoft_labeler_rasterises_what_the_reader_returns(self, tmp_path, monkeypatch):
        from pygeovision.ai.labeling.microsoft_buildings import MicrosoftBuildingsLabeler
        monkeypatch.setattr(fp, "microsoft_footprints", lambda bbox, **kw: frame([box(0.455, 5.905, 0.465, 5.915)]))
        (tmp_path / "labels").mkdir()
        r = MicrosoftBuildingsLabeler(cache_dir=tmp_path / "c").label_tile(self._tile(tmp_path), tmp_path / "labels")
        assert r.error is None and r.label_path is not None
        with rasterio.open(r.label_path) as src:
            m = src.read(1)
        assert 0.15 < (m == 1).mean() < 0.35                  # the box is a quarter of the tile
