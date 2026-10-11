"""sar_flood_mapping on Sentinel-1-like scenes whose answer is known by construction (local GeoTIFFs, no network)."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeovision.ai.pipelines.sar_flood import SarFloodMappingPipeline, otsu_threshold, speckle_mean

BBOX = (0.45, 5.90, 0.4725, 5.9225)             # about 2.5 km square
RASTER = (0.44, 5.89, 0.4825, 5.9325)            # a scene extends beyond the requested box
RES = 0.0001
H = int(round((RASTER[3] - RASTER[1]) / RES))
W = int(round((RASTER[2] - RASTER[0]) / RES))
# a block that floods: lon 0.455-0.462, lat 5.905-5.912 (about 60 ha)
FLOOD = (0.455, 5.905, 0.462, 5.912)
FLOOD_HA = 0.007 * 111_320 * np.cos(np.radians(5.9)) * 0.007 * 110_574 / 1e4


def lonlat_mask(box):
    lon = RASTER[0] + (np.arange(W) + 0.5) * RES
    lat = RASTER[3] - (np.arange(H) + 0.5) * RES
    LON, LAT = np.meshgrid(lon, lat)
    return (box[0] <= LON) & (box[2] >= LON) & (box[1] <= LAT) & (box[3] >= LAT)


def scene(path, db, nodata_mask=None):
    lin = (10.0 ** (np.asarray(db, dtype="float64") / 10.0)).astype("float32")
    if nodata_mask is not None:
        lin[nodata_mask] = -32768.0
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(RASTER[0], RASTER[3], RES, RES), nodata=-32768.0) as d:
        d.write(lin, 1)
    return str(path)


def land(db=-10.0):
    return np.full((H, W), db, dtype="float64")


def item(i, day, path, state="descending", orbit=1, pol="vv"):
    w, s, e, n = RASTER
    return {"id": f"S1_{i}", "bbox": [w, s, e, n],
            "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
            "properties": {"datetime": f"{day}T05:30:00Z", "sat:orbit_state": state, "sat:relative_orbit": orbit},
            "assets": {pol: {"href": path}}}


class FakePC:
    def __init__(self, items):
        self.items = items
        self.searches = []

    def search(self, collection, bbox, datetime=None, **kw):
        assert collection == "sentinel-1-rtc"
        self.searches.append(datetime)
        lo, hi = (datetime or "0000/9999").split("/")
        return [it for it in self.items if lo[:10] <= it["properties"]["datetime"][:10] <= hi[:10]]

    def asset_urls(self, collection, items, asset):
        return [it["assets"][asset]["href"] for it in items]


def pipeline(items):
    p = SarFloodMappingPipeline(None)
    p.pc = FakePC(items)
    return p


def run(p, tmp_path, **kw):
    kw.setdefault("speckle_filter", False)
    kw.setdefault("date_before", "2026-06-01/2026-06-20")
    kw.setdefault("date_after", "2026-06-27/2026-07-03")
    return p.run(BBOX, tmp_path / "out", **kw)


def classes(tmp_path):
    with rasterio.open(tmp_path / "out" / "flood_extent.tif") as s:
        return s.read(1), s


@pytest.fixture
def basic(tmp_path):
    """Dry reference: land -10 dB with a permanent lake. Flood: the same plus a flooded block."""
    lake = lonlat_mask((0.4510, 5.9150, 0.4560, 5.9200))
    before = land()
    before[lake] = -24.0
    after = before.copy()
    after[lonlat_mask(FLOOD)] = -22.0
    # darker, but still above the water threshold: not flooded
    after[lonlat_mask((0.465, 5.905, 0.470, 5.910))] = -13.5
    # dark in both (radar shadow / permanent dark surface): permanent, never "flooded"
    shadow = lonlat_mask((0.4650, 5.9150, 0.4700, 5.9200))
    before[shadow] = -22.0
    after[shadow] = -22.0
    items = [item(1, "2026-06-10", scene(tmp_path / "b1.tif", before)),
             item(2, "2026-06-29", scene(tmp_path / "a1.tif", after))]
    return items, lake, shadow


class TestFloodMapping:
    def test_classes_and_areas_follow_the_construction(self, tmp_path, basic):
        items, lake, shadow = basic
        r = run(pipeline(items), tmp_path)
        assert r.success, r.error
        cls, src = classes(tmp_path)
        assert src.nodata == 255
        s = r.stats
        assert s["flooded_area_ha"] == pytest.approx(FLOOD_HA, rel=0.06)
        # lake (5 ha-ish blocks) and shadow are permanent water, the shallow-dark block is neither
        perm_expected = (0.005 * 111_320 * np.cos(np.radians(5.9)) * 0.005 * 110_574 / 1e4) * 2
        assert s["permanent_water_area_ha"] == pytest.approx(perm_expected, rel=0.1)
        assert s["water_db"] == -15.0 and s["water_db_source"] == "conventional default"
        assert s["scenes_reference"] == 1 and s["scenes_flood"] == 1
        assert (s["orbit_state"], s["relative_orbit"]) == ("descending", 1)
        # the flooded pixels lie inside the flood block
        grid_t = src.transform
        rows, cols = np.nonzero(cls == 1)
        from pyproj import Transformer
        lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(
            grid_t.c + (cols + 0.5) * grid_t.a, grid_t.f + (rows + 0.5) * grid_t.e)
        lon, lat = np.asarray(lon), np.asarray(lat)
        assert lon.min() > FLOOD[0] - 0.0005 and lon.max() < FLOOD[2] + 0.0005
        assert lat.min() > FLOOD[1] - 0.0005 and lat.max() < FLOOD[3] + 0.0005
        for name in ("change_db.tif", "before_db.tif", "after_db.tif"):
            assert (tmp_path / "out" / name).is_file()

    def test_change_db_is_flood_minus_reference(self, tmp_path, basic):
        items, *_ = basic
        r = run(pipeline(items), tmp_path)
        assert r.success
        with rasterio.open(tmp_path / "out" / "change_db.tif") as s:
            ch = s.read(1)
            cls = rasterio.open(tmp_path / "out" / "flood_extent.tif").read(1)
        assert np.nanmedian(ch[cls == 1]) == pytest.approx(-12.0, abs=0.3)      # -22 minus -10

    def test_requiring_more_darkening_removes_a_marginal_flood(self, tmp_path, basic):
        items, *_ = basic
        assert run(pipeline(items), tmp_path / "a", change_db=3.0).stats["flooded_area_ha"] > 50
        assert run(pipeline(items), tmp_path / "b", change_db=15.0).stats["flooded_area_ha"] == 0

    def test_nodata_pixels_are_nodata_not_dry(self, tmp_path):
        before, after = land(), land()
        after[lonlat_mask(FLOOD)] = -22.0
        hole = lonlat_mask((0.4560, 5.9060, 0.4590, 5.9100))                     # inside the flood block, no data
        items = [item(1, "2026-06-10", scene(tmp_path / "b.tif", before)),
                 item(2, "2026-06-29", scene(tmp_path / "a.tif", after, nodata_mask=hole))]
        r = run(pipeline(items), tmp_path)
        cls, _ = classes(tmp_path)
        assert r.success and (cls == 255).sum() > 0 and r.stats["valid_fraction"] < 1.0
        assert r.stats["flooded_area_ha"] < FLOOD_HA * 0.8

    def test_median_ignores_a_one_scene_artefact(self, tmp_path):
        base_after = land()
        base_after[lonlat_mask(FLOOD)] = -22.0
        artefact = base_after.copy()
        artefact[lonlat_mask((0.4660, 5.9100, 0.4700, 5.9140))] = -25.0           # dark in one pass only
        items = [item(1, "2026-06-10", scene(tmp_path / "b.tif", land())),
                 item(2, "2026-06-28", scene(tmp_path / "a1.tif", base_after)),
                 item(3, "2026-06-29", scene(tmp_path / "a2.tif", artefact)),
                 item(4, "2026-06-30", scene(tmp_path / "a3.tif", base_after))]
        r = run(pipeline(items), tmp_path)
        assert r.success and r.stats["scenes_flood"] == 3
        assert r.stats["flooded_area_ha"] == pytest.approx(FLOOD_HA, rel=0.06)

    def test_small_patches_are_removed_and_reported(self, tmp_path):
        after = land()
        after[lonlat_mask(FLOOD)] = -22.0
        after[lonlat_mask((0.4700, 5.9200, 0.47025, 5.92025))] = -22.0            # a few pixels
        items = [item(1, "2026-06-10", scene(tmp_path / "b.tif", land())), item(2, "2026-06-29", scene(tmp_path / "a.tif", after))]
        kept = run(pipeline(items), tmp_path / "k", min_patch_ha=0)
        cut = run(pipeline(items), tmp_path / "c", min_patch_ha=0.5)
        assert kept.stats["flooded_area_ha"] > cut.stats["flooded_area_ha"]
        assert cut.stats["removed_small_patch_ha"] > 0
        assert cut.stats["flooded_area_ha"] == pytest.approx(FLOOD_HA, rel=0.06)

    def test_other_orbit_tracks_are_not_mixed_in(self, tmp_path, basic):
        items, *_ = basic
        everything_dark = scene(tmp_path / "other.tif", land(-30.0))
        items = items + [item(9, "2026-06-28", everything_dark, state="ascending", orbit=99)]
        r = run(pipeline(items), tmp_path)
        assert r.success
        assert r.stats["scenes_flood"] == 1 and r.stats["relative_orbit"] == 1
        assert r.stats["flooded_area_ha"] == pytest.approx(FLOOD_HA, rel=0.06)

    def test_no_common_track_is_refused_with_the_reason(self, tmp_path):
        items = [item(1, "2026-06-10", scene(tmp_path / "b.tif", land()), orbit=1),
                 item(2, "2026-06-29", scene(tmp_path / "a.tif", land()), state="ascending", orbit=2)]
        r = run(pipeline(items), tmp_path)
        assert not r.success and "no Sentinel-1 orbit track in common" in r.error and "viewing geometry" in r.error

    def test_empty_window_lists_nearby_acquisitions(self, tmp_path):
        items = [item(1, "2026-06-10", scene(tmp_path / "b.tif", land())),
                 item(2, "2026-07-12", scene(tmp_path / "a.tif", land()))]
        r = run(pipeline(items), tmp_path, date_after="2026-06-29")
        assert not r.success and "flood window" in r.error and "2026-07-12" in r.error
        r2 = run(pipeline(items), tmp_path / "x", date_before="2026-05-20")
        assert not r2.success and "reference window" in r2.error and "2026-06-10" in r2.error

    def test_windows_are_enforced_even_if_the_server_ignores_the_datetime_filter(self, tmp_path, basic):
        """A search that returns every scene regardless of the window must not make both windows use the same scenes."""
        items, *_ = basic

        class Sloppy(FakePC):
            def search(self, collection, bbox, datetime=None, **kw):
                self.searches.append(datetime)
                return self.items
        p = SarFloodMappingPipeline(None)
        p.pc = Sloppy(items)
        r = run(p, tmp_path)
        assert r.success, r.error
        assert r.stats["reference_dates"] == ["2026-06-10"] and r.stats["flood_dates"] == ["2026-06-29"]
        assert r.metadata["scene_ids"] == {"reference": ["S1_1"], "flood": ["S1_2"]}
        assert r.stats["flooded_area_ha"] == pytest.approx(FLOOD_HA, rel=0.06)

    def test_polarization_without_the_asset_has_no_scene(self, tmp_path, basic):
        items, *_ = basic
        r = run(pipeline(items), tmp_path, polarization="vh")
        assert not r.success and "VH" in r.error

    def test_bad_arguments_are_clear(self, tmp_path, basic):
        items, *_ = basic
        p = pipeline(items)
        assert "date_before" in run(p, tmp_path, date_before=None).error
        assert "must end before" in run(p, tmp_path, date_before="2026-06-30", date_after="2026-06-29").error
        assert "polarization" in run(p, tmp_path, polarization="xx").error
        assert "water_db" in run(p, tmp_path, water_db="deep").error
        assert "negative" in run(p, tmp_path, change_db=-1).error

    def test_month_windows_are_accepted(self, tmp_path, basic):
        items, *_ = basic
        assert run(pipeline(items), tmp_path, date_before="2026-06", date_after="2026-07").success is False   # overlap-free but no scene in July
        items2 = [items[0], item(5, "2026-07-03", items[1]["assets"]["vv"]["href"])]
        assert run(pipeline(items2), tmp_path / "y", date_before="2026-06", date_after="2026-07").success

    def test_speckle_filter_keeps_the_flood_and_fills_edges(self, tmp_path, basic):
        items, *_ = basic
        r = run(pipeline(items), tmp_path, speckle_filter=True)
        assert r.success and r.stats["flooded_area_ha"] == pytest.approx(FLOOD_HA, rel=0.08)


class TestThreshold:
    def test_default_threshold_depends_on_polarisation(self, tmp_path, basic):
        items, *_ = basic
        vh_items = [{**it, "assets": {"vh": it["assets"]["vv"]}} for it in items]
        r = run(pipeline(vh_items), tmp_path, polarization="vh")
        assert r.success and r.stats["water_db"] == -20.0

    def test_explicit_threshold_and_otsu(self, tmp_path, basic):
        items, *_ = basic
        assert run(pipeline(items), tmp_path / "u", water_db=-18).stats["water_db_source"] == "user"
        r = run(pipeline(items), tmp_path / "o", water_db="otsu")
        assert r.success and r.stats["water_db_source"] == "otsu"
        assert -22 < r.stats["water_db"] < -10 and 0 < r.stats["otsu_separability"] <= 1

    def test_otsu_splits_two_clusters(self):
        rng = np.random.RandomState(1)
        v = np.concatenate([rng.normal(-24, 1.5, 4000), rng.normal(-9, 2.0, 6000)])
        t, sep = otsu_threshold(v)
        assert -20 < t < -12 and sep > 0.8

    def test_otsu_needs_enough_pixels(self):
        from pygeovision.data.reference import ReferenceDataError
        with pytest.raises(ReferenceDataError):
            otsu_threshold(np.array([-10.0] * 10))


def test_speckle_mean_ignores_invalid_neighbours():
    lin = np.full((5, 5), 1.0)
    lin[2, 2] = 10.0
    valid = np.ones((5, 5), bool)
    valid[2, 3] = False
    lin[2, 3] = 1000.0                                  # invalid: must not leak into the mean
    out = speckle_mean(lin, valid)
    assert out[2, 2] == pytest.approx((10.0 + 7 * 1.0) / 8.0)
    assert np.isnan(out[2, 3])
