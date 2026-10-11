"""Pipelines that read published products (annual land cover, forest, surface water, ET, fire).

The "remote" tiles are small GeoTIFFs written on the exact grid the pipeline builds, so every expected number below
is computed by hand from the pixels that were written (pixel count x pixel area), not taken from the pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio

from pygeovision.ai.pipelines.reference_products import (
    ActiveFirePipeline, EvapotranspirationPipeline, ForestChangePipeline, LandCoverChangePipeline,
    SurfaceWaterDynamicsPipeline, UrbanExpansionPipeline,
)
from pygeovision.data.reference import make_grid

BBOX = (0.45, 5.90, 0.52, 5.97)             # about 8 km square near the lower Volta
COARSE = (0.40, 5.90, 0.60, 6.10)           # 22 km, for the 500 m and 1 km products
H = 0.01                                    # hectares per 10 m pixel


class FakePC:
    """Stands in for PlanetaryComputer: items are given per collection; search filters by year/date like STAC."""

    def __init__(self, collections):
        self.collections = collections
        self.searches = []

    def search(self, collection, bbox, datetime=None, **kw):
        self.searches.append((collection, datetime))
        items = self.collections.get(collection, [])
        if datetime and "/" in datetime:
            lo, hi = datetime.split("/")
            def inside(it):
                p = it["properties"]
                a = (p.get("start_datetime") or p.get("datetime"))[:10]
                b = (p.get("end_datetime") or p.get("start_datetime") or p.get("datetime"))[:10]
                return a <= hi and b >= lo
            items = [it for it in items if inside(it)]
        return items

    def asset_urls(self, collection, items, asset):
        return [it["assets"][asset]["href"] for it in items if asset in it["assets"]]


def write_tile(path, arr, grid, dtype="uint8", nodata=0):
    prof = grid.profile()
    prof.update(count=1, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype(dtype), 1)
    return str(path)


def item(id_, path, start, end=None, asset="data", **extra):
    props = {"start_datetime": start, "datetime": None}
    if end:
        props["end_datetime"] = end
    return {"id": id_, "properties": props, "assets": {asset: {"href": str(path)}}, **extra}


def grid_for(res, bbox=BBOX):
    return make_grid(bbox, res)


# ---------------------------------------------------------------------------------------------------------------------
class TestLandCoverChange:
    def _world(self, tmp_path):
        g = grid_for(10)
        h, w = g.height, g.width
        a = np.full((h, w), 2, np.uint8)                       # trees everywhere in 2018
        b = a.copy()
        b[10:30, 10:30] = 7                                    # 400 px trees -> built
        b[40:50, 40:50] = 5                                    # 100 px trees -> crops
        b[60, 60] = 8                                          # one isolated px trees -> bare
        b[70:75, 70:75] = 10                                   # 25 px cloud in 2022: must be nodata
        a[80:90, 80:90] = 5
        b[80:90, 80:90] = 5                                    # unchanged crops
        ta, tb = write_tile(tmp_path / "a.tif", a, g), write_tile(tmp_path / "b.tif", b, g)
        pc = FakePC({"io-lulc-annual-v02": [
            item("31N-2018", ta, "2018-01-01T00:00:00Z", "2019-01-01T00:00:00Z"),
            item("31N-2022", tb, "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z")]})
        return g, pc

    def _run(self, tmp_path, **kw):
        g, pc = self._world(tmp_path)
        p = LandCoverChangePipeline(None)
        p.pc = pc
        kw.setdefault("date_before", 2018)
        kw.setdefault("date_after", "2022-06")
        return g, pc, p.run(BBOX, tmp_path / "out", **kw)

    def test_areas_match_the_pixels_that_were_written(self, tmp_path):
        g, pc, r = self._run(tmp_path)
        assert r.success, r.error
        s = r.stats
        assert s["changed_area_ha"] == pytest.approx((400 + 100 + 1) * H)
        n_valid = g.width * g.height - 25
        assert s["change_fraction"] == pytest.approx(501 / n_valid)
        assert s["valid_fraction"] == pytest.approx(n_valid / (g.width * g.height))
        tr = {(t["from"], t["to"]): t["area_ha"] for t in s["transitions"]}
        assert tr[("Trees", "Built area")] == pytest.approx(4.0)
        assert tr[("Trees", "Crops")] == pytest.approx(1.0)
        assert tr[("Trees", "Bare ground")] == pytest.approx(0.01)
        assert s["transitions"][0]["from"] == "Trees" and s["transitions"][0]["to"] == "Built area"   # sorted by area

    def test_both_searches_used_a_window_that_cannot_match_another_year(self, tmp_path):
        g, pc, r = self._run(tmp_path)
        assert [d for _, d in pc.searches] == ["2018-03-01/2018-09-30", "2022-03-01/2022-09-30"]

    def test_cloud_pixels_are_nodata_not_unchanged(self, tmp_path):
        g, pc, r = self._run(tmp_path)
        with rasterio.open(tmp_path / "out" / "change_mask.tif") as src:
            m = src.read(1)
        assert (m[70:75, 70:75] == 255).all()
        assert m[0, 0] == 0 and m[20, 20] == 1
        assert (m == 255).sum() == 25

    def test_transition_raster_encodes_from_and_to(self, tmp_path):
        g, pc, r = self._run(tmp_path)
        with rasterio.open(tmp_path / "out" / "transition.tif") as src:
            t = src.read(1)
        order = [1, 2, 4, 5, 7, 8, 9, 11]
        assert t[20, 20] == 1 + 8 * order.index(2) + order.index(7)
        assert t[45, 45] == 1 + 8 * order.index(2) + order.index(5)
        assert t[85, 85] == 0

    def test_area_tables_per_year(self, tmp_path):
        g, pc, r = self._run(tmp_path)
        assert r.stats["area_ha_after"]["Built area"] == pytest.approx(4.0)
        assert "Built area" not in r.stats["area_ha_before"]

    def test_min_change_removes_the_isolated_pixel_only(self, tmp_path):
        g, pc, r = self._run(tmp_path, min_change_ha=0.05)
        assert r.stats["pixels_removed_by_min_change"] == 1
        assert r.stats["changed_area_ha"] == pytest.approx(500 * H)
        assert r.stats["changed_in_patches_of_3px_or_less"] == 0.0          # of the mask that was kept

    def test_noise_indicator_without_a_minimum_patch(self, tmp_path):
        g, pc, r = self._run(tmp_path)
        assert r.stats["pixels_removed_by_min_change"] == 0
        assert r.stats["changed_in_patches_of_3px_or_less"] == pytest.approx(1 / 501)

    @pytest.mark.parametrize("kw,msg", [({"date_before": 2022, "date_after": 2018}, "earlier"),
                                        ({"date_before": 2015}, "outside"),
                                        ({"date_after": 2030}, "outside"),
                                        ({"date_before": None}, "needs date_before"),
                                        ({"date_before": "June"}, "year")])
    def test_bad_dates_are_explained(self, tmp_path, kw, msg):
        g, pc = self._world(tmp_path)
        p = LandCoverChangePipeline(None)
        p.pc = pc
        args = {"date_before": 2018, "date_after": 2022}
        args.update(kw)
        r = p.run(BBOX, tmp_path / "out", **args)
        assert not r.success and msg in r.error

    def test_year_without_a_tile_is_a_failure_not_a_zero(self, tmp_path):
        g, pc = self._world(tmp_path)
        p = LandCoverChangePipeline(None)
        p.pc = pc
        r = p.run(BBOX, tmp_path / "out", date_before=2019, date_after=2022)
        assert not r.success and "2019" in r.error

    def test_identical_years_report_zero_change(self, tmp_path):
        g = grid_for(10)
        a = np.full((g.height, g.width), 2, np.uint8)
        t = write_tile(tmp_path / "a.tif", a, g)
        pc = FakePC({"io-lulc-annual-v02": [item("x-2018", t, "2018-01-01T00:00:00Z", "2019-01-01T00:00:00Z"),
                                             item("x-2022", t, "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z")]})
        p = LandCoverChangePipeline(None)
        p.pc = pc
        r = p.run(BBOX, tmp_path / "out", date_before=2018, date_after=2022)
        assert r.success and r.stats["changed_area_ha"] == 0 and r.stats["transitions"] == []

    def test_resolution_is_coarsened_for_huge_areas_and_says_so(self, tmp_path, monkeypatch):
        import pygeovision.ai.pipelines.reference_products as rp
        monkeypatch.setattr(rp, "AUTO_MAX_PIXELS", 5_000)
        g, pc, r = self._run(tmp_path)
        assert r.success and r.stats["resolution_coarsened"] and r.stats["resolution_m"] > 10


class TestUrbanExpansion:
    def test_gain_loss_net_and_rate(self, tmp_path):
        g = grid_for(10)
        a = np.full((g.height, g.width), 5, np.uint8)
        a[0:20, 0:20] = 7                                       # 400 px built in 2018
        b = a.copy()
        b[0:10, 0:20] = 8                                       # 200 px of it lost
        b[50:80, 50:60] = 7                                     # 300 px new
        b[100:105, 100:105] = 10                                # cloud: nodata
        ta, tb = write_tile(tmp_path / "a.tif", a, g), write_tile(tmp_path / "b.tif", b, g)
        pc = FakePC({"io-lulc-annual-v02": [item("a", ta, "2018-01-01T00:00:00Z", "2019-01-01T00:00:00Z"),
                                             item("b", tb, "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z")]})
        p = UrbanExpansionPipeline(None)
        p.pc = pc
        r = p.run(BBOX, tmp_path / "out", date_before=2018, date_after=2023)
        assert r.success, r.error
        s = r.stats
        assert s["built_area_ha_before"] == pytest.approx(4.0)
        assert s["built_area_ha_after"] == pytest.approx(5.0)
        assert s["gain_ha"] == pytest.approx(3.0) and s["loss_ha"] == pytest.approx(2.0) and s["net_change_ha"] == pytest.approx(1.0)
        assert s["annual_growth_rate"] == pytest.approx((5.0 / 4.0) ** (1 / 5) - 1)
        with rasterio.open(tmp_path / "out" / "built_area_change.tif") as src:
            c = src.read(1)
        assert c[15, 5] == 1 and c[5, 5] == 3 and c[60, 55] == 2 and c[200, 200] == 0 and c[102, 102] == 255

    def test_no_built_area_gives_no_rate(self, tmp_path):
        g = grid_for(10)
        a = np.full((g.height, g.width), 5, np.uint8)
        t = write_tile(tmp_path / "a.tif", a, g)
        pc = FakePC({"io-lulc-annual-v02": [item("a", t, "2018-01-01T00:00:00Z", "2019-01-01T00:00:00Z"),
                                             item("b", t, "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z")]})
        p = UrbanExpansionPipeline(None)
        p.pc = pc
        r = p.run(BBOX, tmp_path / "out", date_before=2018, date_after=2023)
        assert r.success and r.stats["annual_growth_rate"] is None and r.stats["gain_ha"] == 0


class TestForestChange:
    def _run(self, tmp_path, **kw):
        g = grid_for(25)
        a = np.full((g.height, g.width), 3, np.uint8)         # non-forest
        a[0:20, 0:20] = 1                                     # 400 px dense forest
        a[30:40, 0:20] = 2                                    # 200 px open forest
        a[60:70, 60:70] = 4                                   # water in 2015
        b = a.copy()
        b[0:10, 0:20] = 3                                     # 200 px dense forest lost
        b[30:40, 0:10] = 3                                    # 100 px open forest lost
        b[100:110, 100:110] = 1                               # 100 px gain
        b[60:70, 60:70] = 1                                   # water -> forest: product disagreement, nodata
        ta, tb = write_tile(tmp_path / "a.tif", a, g), write_tile(tmp_path / "b.tif", b, g)
        pc = FakePC({"alos-fnf-mosaic": [item("N06E000_15_FNF", ta, "2015-01-01T00:00:00Z", "2015-12-31T23:59:59Z", asset="C"),
                                         item("N06E000_20_FNF", tb, "2020-01-01T00:00:00Z", "2020-12-31T23:59:59Z", asset="C")]})
        p = ForestChangePipeline(None)
        p.pc = pc
        return p.run(BBOX, tmp_path / "out", date_before=2015, date_after=2020, **kw), g

    def test_dense_forest_definition(self, tmp_path):
        r, g = self._run(tmp_path)
        assert r.success, r.error
        ha = 25 * 25 / 10_000
        s = r.stats
        assert s["forest_area_ha_before"] == pytest.approx(400 * ha)
        assert s["forest_loss_ha"] == pytest.approx(200 * ha) and s["forest_gain_ha"] == pytest.approx(100 * ha)
        assert s["net_change_ha"] == pytest.approx(-100 * ha)
        assert s["loss_ha_per_year"] == pytest.approx(200 * ha / 5)
        assert s["loss_share_of_initial_forest"] == pytest.approx(0.5)
        with rasterio.open(tmp_path / "out" / "forest_change.tif") as src:
            c = src.read(1)
        assert c[65, 65] == 255                         # water in one year is not called gain
        assert c[5, 5] == 2 and c[15, 5] == 1 and c[105, 105] == 3 and c[35, 5] == 0

    def test_open_forest_switch_changes_the_answer(self, tmp_path):
        r, g = self._run(tmp_path, include_open_forest=True)
        ha = 25 * 25 / 10_000
        assert r.stats["forest_area_ha_before"] == pytest.approx(600 * ha)
        assert r.stats["forest_loss_ha"] == pytest.approx(300 * ha)

    def test_years_outside_the_product_are_refused(self, tmp_path):
        p = ForestChangePipeline(None)
        p.pc = FakePC({})
        r = p.run(BBOX, tmp_path, date_before=2010, date_after=2020)
        assert not r.success and "2015-2020" in r.error


class TestSurfaceWater:
    def test_classes_groups_and_nodata(self, tmp_path):
        g = grid_for(30)
        h, w = g.height, g.width
        tr = np.zeros((h, w), np.uint8)
        tr[0:10, 0:10] = 1; tr[10:20, 0:10] = 2; tr[20:30, 0:10] = 3; tr[30:40, 0:10] = 4
        tr[40:50, 0:10] = 5; tr[50:60, 0:10] = 6; tr[60:70, 0:10] = 7; tr[70:80, 0:10] = 8
        tr[80:90, 0:10] = 9; tr[90:100, 0:10] = 10
        tr[:, w - 5:] = 255                                       # outside the product
        occ = np.where(tr > 0, 80, 0).astype(np.uint8); occ[tr == 255] = 255
        sea = np.where(tr == 4, 12, 0).astype(np.uint8); sea[tr == 255] = 255
        files = {n: write_tile(tmp_path / f"{n}.tif", a, g, nodata=None) for n, a in
                 (("transitions", tr), ("occurrence", occ), ("seasonality", sea))}
        it = {"id": "0E_10Nv1_3_2020", "properties": {"datetime": "2020-07-01T00:00:00Z"},
              "assets": {k: {"href": v} for k, v in files.items()}}
        p = SurfaceWaterDynamicsPipeline(None)
        p.pc = FakePC({"jrc-gsw": [it]})
        r = p.run(BBOX, tmp_path / "out")
        assert r.success, r.error
        s, ha = r.stats, 0.09
        assert s["permanent_now_ha"] == pytest.approx(300 * ha)       # classes 1, 2, 7
        assert s["seasonal_now_ha"] == pytest.approx(300 * ha)        # 4, 5, 8
        assert s["lost_ha"] == pytest.approx(200 * ha)                # 3, 6
        assert s["ephemeral_ha"] == pytest.approx(200 * ha)           # 9, 10
        assert s["ever_water_ha"] == pytest.approx(1000 * ha)
        assert s["valid_fraction"] == pytest.approx((w - 5) / w)
        assert s["area_ha_by_class"]["Lost permanent"] == pytest.approx(100 * ha)
        with rasterio.open(tmp_path / "out" / "water_transitions.tif") as src:
            o = src.read(1)
        assert (o[:, w - 5:] == 255).all() and o[0, 0] == 1
        with rasterio.open(tmp_path / "out" / "water_occurrence.tif") as src:
            oc = src.read(1)
        assert oc[5, 5] == 80 and np.isnan(oc[0, w - 1]) and oc[200, 200] == 0

    def test_no_tile_is_a_failure(self, tmp_path):
        p = SurfaceWaterDynamicsPipeline(None)
        p.pc = FakePC({"jrc-gsw": []})
        r = p.run(BBOX, tmp_path)
        assert not r.success and "no JRC" in r.error


class TestEvapotranspiration:
    def _items(self, tmp_path, et, pet, g, extra=()):
        te, tp = write_tile(tmp_path / "et.tif", et, g, "uint16", nodata=None), write_tile(tmp_path / "pet.tif", pet, g, "uint16", nodata=None)
        it = {"id": "MOD16A3GF.A2022001.h18v08.061.2023", "properties": {"start_datetime": "2022-01-01T00:00:00Z", "end_datetime": "2022-12-31T23:59:59Z", "datetime": None},
              "assets": {"ET_500m": {"href": te}, "PET_500m": {"href": tp}}}
        return [it, *extra]

    def test_scaling_flags_ratio_and_volume(self, tmp_path):
        g = grid_for(500, COARSE)
        et = np.full((g.height, g.width), 12345, np.uint16)       # 1234.5 mm
        pet = np.full((g.height, g.width), 20000, np.uint16)      # 2000 mm
        et[0, 0] = 65535                                          # fill value
        et[0, 1] = 65531                                          # land-class flag (urban)
        p = EvapotranspirationPipeline(None)
        p.pc = FakePC({"modis-16A3GF-061": self._items(tmp_path, et, pet, g)})
        r = p.run(COARSE, tmp_path / "out", date="2022-06", resolution_m=500)
        assert r.success, r.error
        s = r.stats
        n = g.width * g.height
        assert s["mean_et_mm"] == pytest.approx(1234.5) and s["mean_pet_mm"] == pytest.approx(2000.0)
        assert s["mean_et_pet_ratio"] == pytest.approx(1234.5 / 2000.0)
        assert s["valid_fraction"] == pytest.approx((n - 2) / n)
        assert s["total_et_volume_million_m3"] == pytest.approx(1234.5 / 1000 * 500 * 500 * (n - 2) / 1e6, rel=1e-6)
        with rasterio.open(tmp_path / "out" / "et_mm_per_year.tif") as src:
            a = src.read(1)
        assert np.isnan(a[0, 0]) and np.isnan(a[0, 1]) and a[5, 5] == pytest.approx(1234.5)

    def test_terra_preferred_over_aqua_for_the_same_tile(self, tmp_path):
        g = grid_for(500, COARSE)
        et = np.full((g.height, g.width), 10000, np.uint16)
        et2 = np.full((g.height, g.width), 20000, np.uint16)
        pet = np.full((g.height, g.width), 30000, np.uint16)
        items = self._items(tmp_path, et2, pet, g)            # id MOD16...: 2000 mm
        aq = write_tile(tmp_path / "aq.tif", et, g, "uint16", nodata=None)
        aqua = {"id": "MYD16A3GF.A2022001.h18v08.061.2023", "properties": {"start_datetime": "2022-01-01T00:00:00Z", "end_datetime": "2022-12-31T23:59:59Z"},
                "assets": {"ET_500m": {"href": aq}, "PET_500m": {"href": aq}}}
        p = EvapotranspirationPipeline(None)
        p.pc = FakePC({"modis-16A3GF-061": [aqua, *items]})
        r = p.run(COARSE, tmp_path / "out", date=2022)
        assert r.success and r.stats["mean_et_mm"] == pytest.approx(2000.0)
        assert r.metadata["tiles"] == ["MOD16A3GF.A2022001.h18v08.061.2023"]

    def test_all_flag_area_is_a_failure_with_the_reason(self, tmp_path):
        g = grid_for(500, COARSE)
        et = np.full((g.height, g.width), 65531, np.uint16)
        p = EvapotranspirationPipeline(None)
        p.pc = FakePC({"modis-16A3GF-061": self._items(tmp_path, et, et, g)})
        r = p.run(COARSE, tmp_path / "out", date=2022)
        assert not r.success and "fill" in r.error

    def test_needs_a_year(self, tmp_path):
        p = EvapotranspirationPipeline(None)
        p.pc = FakePC({})
        assert "needs date" in p.run(COARSE, tmp_path).error
        assert "2001" in p.run(COARSE, tmp_path, date=1999).error


class TestActiveFire:
    def _world(self, tmp_path):
        g = grid_for(926.6254331, COARSE)
        h, w = g.height, g.width
        d1 = np.full((h, w), 5, np.uint8)                       # clear land
        d1[1, 1] = 9; d1[2, 2] = 7; d1[3, 3] = 4                # high, low, cloud
        d2 = np.full((h, w), 5, np.uint8)
        d2[1, 1] = 8; d2[3, 3] = 8                              # nominal fire again at (1,1); fire at (3,3)
        d2[0, 0] = 4                                            # cloud: not observed on day 2
        frp1 = np.zeros((h, w), np.uint32); frp1[1, 1] = 250    # 25.0 MW
        frp2 = np.zeros((h, w), np.uint32); frp2[1, 1] = 100; frp2[3, 3] = 40
        items = []
        for i, (fm, frp, day) in enumerate(((d1, frp1, "2024-02-10"), (d2, frp2, "2024-02-11"))):
            a = write_tile(tmp_path / f"fm{i}.tif", fm, g, nodata=None)
            b = write_tile(tmp_path / f"frp{i}.tif", frp, g, "uint32", nodata=None)
            items.append({"id": f"MOD14A1.A{i}", "properties": {"start_datetime": f"{day}T00:00:00Z", "datetime": None},
                          "assets": {"FireMask": {"href": a}, "MaxFRP": {"href": b}}})
        return g, FakePC({"modis-14A1-061": items})

    def test_counts_days_points_and_observation_days(self, tmp_path):
        g, pc = self._world(tmp_path)
        p = ActiveFirePipeline(None)
        p.pc = pc
        r = p.run(COARSE, tmp_path / "out", date="2024-02", min_confidence="nominal")
        assert r.success, r.error
        s = r.stats
        assert s["days_with_data"] == 2 and s["min_confidence"] == "nominal"
        assert s["detections_pixel_days"] == 3                  # day1: high; day2: two nominal (low is below threshold)
        assert s["pixels_with_fire"] == 2 and s["max_days_with_fire_in_a_pixel"] == 2
        assert s["per_day_detections"] == {"2024-02-10": 1, "2024-02-11": 2}
        assert s["max_frp_mw"] == pytest.approx(25.0)
        with rasterio.open(tmp_path / "out" / "fire_days.tif") as src:
            fd = src.read(1)
        with rasterio.open(tmp_path / "out" / "observed_days.tif") as src:
            od = src.read(1)
        assert fd[1, 1] == 2 and fd[3, 3] == 1 and fd[2, 2] == 0
        assert od[0, 0] == 1 and od[3, 3] == 1 and od[5, 5] == 2   # clouded days are not "observed"
        import geopandas as gpd
        pts = gpd.read_file(tmp_path / "out" / "fire_detections.gpkg")
        assert len(pts) == 3 and set(pts.confidence) == {"high", "nominal"}
        assert str(pts.crs).endswith("4326") and pts.geometry.within(__import__("shapely.geometry", fromlist=["box"]).box(*COARSE).buffer(0.02)).all()

    def test_lower_threshold_counts_the_low_confidence_pixel(self, tmp_path):
        g, pc = self._world(tmp_path)
        p = ActiveFirePipeline(None)
        p.pc = pc
        r = p.run(COARSE, tmp_path / "out", date_before="2024-02-10", date_after="2024-02-11", min_confidence="low")
        assert r.success and r.stats["detections_pixel_days"] == 4 and r.stats["pixels_with_fire"] == 3
        assert r.stats["start"] == "2024-02-10" and r.stats["end"] == "2024-02-11"

    def test_no_fire_is_a_valid_result_not_a_failure(self, tmp_path):
        g = grid_for(926.6254331, COARSE)
        fm = np.full((g.height, g.width), 5, np.uint8)
        a, b = write_tile(tmp_path / "fm.tif", fm, g, nodata=None), write_tile(tmp_path / "frp.tif", np.zeros_like(fm), g, "uint32", nodata=None)
        it = {"id": "x", "properties": {"start_datetime": "2024-02-10T00:00:00Z"}, "assets": {"FireMask": {"href": a}, "MaxFRP": {"href": b}}}
        p = ActiveFirePipeline(None)
        p.pc = FakePC({"modis-14A1-061": [it]})
        r = p.run(COARSE, tmp_path / "out", date="2024-02")
        assert r.success and r.stats["detections_pixel_days"] == 0 and r.stats["max_frp_mw"] is None
        assert not (tmp_path / "out" / "fire_detections.gpkg").exists()

    def test_argument_errors(self, tmp_path):
        p = ActiveFirePipeline(None)
        p.pc = FakePC({"modis-14A1-061": []})
        assert "needs date" in p.run(COARSE, tmp_path).error
        assert "min_confidence" in p.run(COARSE, tmp_path, date="2024-02", min_confidence="x").error
        assert "no MOD14A1" in p.run(COARSE, tmp_path, date="2024-02").error
