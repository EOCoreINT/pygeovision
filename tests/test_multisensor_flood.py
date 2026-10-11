"""multisensor_flood_mapping on radar + optical + DEM data whose answer is known by construction (local GeoTIFFs)."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from test_sar_flood import BBOX, FLOOD, RASTER, RES, H, W, item, land, lonlat_mask, scene
from test_terrain_pipelines import expanded, write_dem

from pygeovision.ai.pipelines.multisensor_flood import (
    MultiSensorFloodPipeline,
    _scale_offset,
    agreement,
    calibrate_water_db,
    mndwi,
)

LAKE = (0.4510, 5.9150, 0.4540, 5.9200)          # water in both windows, both sensors
CLOUDED = (0.4650, 5.9050, 0.4700, 5.9100)       # radar flooded, optical cloud-covered
OPT_ONLY = (0.4650, 5.9110, 0.4700, 5.9140)      # optical water, radar did not change
DISPUTED = (0.4560, 5.9125, 0.4600, 5.9150)      # radar flooded, clear optical view says dry
HILL = (0.4550, 5.9195, 0.4620, 5.9215)          # both sensors flood, but high above any drainage
SHRINK = 0.0004                                  # stay clear of the blocks' edges (resampling mixes them)


def s2_scene(path, water, cloud=None, shift=0):
    """Green (B03) and SWIR1 (B11) digital numbers: water has MNDWI +0.6, land -0.25 (``shift`` = processing offset)."""
    w = lonlat_mask_any(water)
    green = np.where(w, 1200.0, 1500.0) + shift
    swir = np.where(w, 300.0, 2500.0) + shift
    scl = np.full((H, W), 4.0)
    if cloud is not None:
        scl[lonlat_mask_any(cloud)] = 9.0
    out = {}
    for name, arr in (("B03", green), ("B11", swir), ("SCL", scl)):
        p = f"{path}_{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(RASTER[0], RASTER[3], RES, RES)) as d:
            d.write(arr.astype("float32"), 1)
        out[name] = {"href": p}
    return out


def lonlat_mask_any(boxes):
    boxes = [boxes] if isinstance(boxes[0], float) else boxes
    m = np.zeros((H, W), bool)
    for b in boxes:
        m |= lonlat_mask(b)
    return m


def s2_item(i, day, assets, cloud=10.0, baseline=None, raster_bands=None):
    props = {"datetime": f"{day}T10:30:00Z", "eo:cloud_cover": cloud}
    if baseline:
        props["s2:processing_baseline"] = baseline
    assets = {k: dict(v) for k, v in assets.items()}
    if raster_bands:
        assets["B03"]["raster:bands"] = raster_bands
        assets["B11"]["raster:bands"] = raster_bands
    return {"id": f"S2_{i}", "properties": props, "assets": assets}


class FakePC:
    def __init__(self, s1=(), s2=(), dem=()):
        self.data = {"sentinel-1-rtc": list(s1), "sentinel-2-l2a": list(s2), "cop-dem-glo-30": list(dem)}

    def search(self, collection, bbox, datetime=None, **kw):
        items = self.data[collection]
        if collection == "cop-dem-glo-30" or not datetime:
            return items
        lo, hi = datetime.split("/")
        return [it for it in items if lo[:10] <= it["properties"]["datetime"][:10] <= hi[:10]]

    def asset_urls(self, collection, items, asset):
        return [it["assets"][asset]["href"] for it in items if asset in it["assets"]]


def block_classes(cls, src, box):
    """Distinct classes inside ``box`` (shrunk by SHRINK degrees)."""
    t = src.transform
    rows, cols = np.mgrid[0:cls.shape[0], 0:cls.shape[1]]
    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(
        t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e)
    m = (lon > box[0] + SHRINK) & (lon < box[2] - SHRINK) & (lat > box[1] + SHRINK) & (lat < box[3] - SHRINK)
    assert m.sum() > 20, "the block is too small to test"
    return set(np.unique(cls[m]).tolist())


@pytest.fixture
def world(tmp_path):
    """Radar before/after scenes, optical before/after scenes (one cloud-covered block) and a DEM."""
    flooded = [FLOOD, CLOUDED, DISPUTED, HILL]
    sb, sa = land(), land()
    sb[lonlat_mask(LAKE)] = -24.0
    sa[lonlat_mask(LAKE)] = -24.0
    for b in flooded:
        sa[lonlat_mask(b)] = -22.0
    s1 = [item(1, "2026-06-10", scene(tmp_path / "rb.tif", sb)), item(2, "2026-06-29", scene(tmp_path / "ra.tif", sa))]

    ob = s2_scene(tmp_path / "ob", [LAKE])
    oa = s2_scene(tmp_path / "oa", [LAKE, FLOOD, OPT_ONLY, HILL, CLOUDED], cloud=CLOUDED)
    s2 = [s2_item(1, "2026-06-12", ob), s2_item(2, "2026-06-30", oa)]

    p = MultiSensorFloodPipeline(None)
    grid, _ = p._grid(BBOX, 20.0)
    z = np.repeat(np.arange(grid.height + 6 - 1, -1, -1, dtype=float)[:, None], grid.width + 6, axis=1)   # slopes to the south, 1 m/cell
    dem = [write_dem(tmp_path / "dem.tif", z, expanded(grid, 3))]
    return tmp_path, s1, s2, dem


def run(tmp_path, pc, **kw):
    p = MultiSensorFloodPipeline(None)
    p.pc = pc
    kw.setdefault("speckle_filter", False)
    kw.setdefault("date_before", "2026-06-01/2026-06-20")
    kw.setdefault("date_after", "2026-06-27/2026-07-03")
    kw.setdefault("stream_threshold_km2", 0.02)
    return p.run(BBOX, tmp_path / "out", **kw)


def read(tmp_path, name="flood_extent.tif"):
    with rasterio.open(tmp_path / "out" / name) as s:
        return s.read(1), s


class TestAgreement:
    def test_each_block_gets_the_class_its_evidence_implies(self, world):
        tmp_path, s1, s2, dem = world
        r = run(tmp_path, FakePC(s1, s2, dem))
        assert r.success, r.error
        cls, src = read(tmp_path)
        assert src.nodata == 255
        assert block_classes(cls, src, FLOOD) == {1}               # radar and optical agree
        assert block_classes(cls, src, CLOUDED) == {2}             # optical could not see it
        assert block_classes(cls, src, OPT_ONLY) == {3}            # radar did not change
        assert block_classes(cls, src, DISPUTED) == {4}            # radar flooded, optical clear and dry
        assert block_classes(cls, src, LAKE) == {5}                # water before the event
        assert block_classes(cls, src, HILL) == {0}                # removed by terrain
        s = r.stats
        assert s["sensors_used"] == ["sentinel-1", "sentinel-2", "copernicus-dem"]
        assert s["hand_applied"] is True and s["excluded_by_hand_ha"] > 0
        assert s["flooded_area_ha"] == pytest.approx(s["flooded_confirmed_ha"] + s["flooded_radar_only_ha"]
                                                     + s["flooded_optical_only_ha"])
        assert s["flooded_upper_ha"] == pytest.approx(s["flooded_area_ha"] + s["flooded_disputed_ha"])
        assert s["flooded_disputed_ha"] > 0 and s["permanent_water_area_ha"] > 0
        assert 0.5 < s["optical_coverage"] < 1.0                   # the clouded block is not covered
        assert s["water_db_source"].startswith("calibrated on Sentinel-2") and -22 <= s["water_db"] <= -12
        assert s["radar_optical_f1_reference"] > 0.95              # radar and optical agree about the lake before the event
        assert s["land_area_ha"] > 0 and s["flooded_fraction"] == pytest.approx(s["flooded_area_ha"] / s["land_area_ha"])
        assert s["optical_scenes_reference"] == 1 and s["optical_scenes_flood"] == 1
        for name in ("change_db.tif", "mndwi_after.tif", "hand_m.tif"):
            assert (tmp_path / "out" / name).is_file()

    def test_the_hill_is_excluded_because_it_is_high_not_by_accident(self, world):
        tmp_path, s1, s2, dem = world
        run(tmp_path, FakePC(s1, s2, dem))
        hand, src = read(tmp_path, "hand_m.tif")
        t = src.transform
        lon, lat = (HILL[0] + HILL[2]) / 2, (HILL[1] + HILL[3]) / 2
        x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
        row, col = int((y - t.f) / t.e), int((x - t.c) / t.a)
        assert hand[row, col] > 15.0
        fr, fc = [int((v - o) / d) for v, o, d in zip(
            Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform((FLOOD[0] + FLOOD[2]) / 2, (FLOOD[1] + FLOOD[3]) / 2)[::-1],
            (t.f, t.c), (t.e, t.a))]
        assert hand[fr, fc] < 15.0

    def test_raising_the_hand_limit_keeps_the_hill(self, world):
        tmp_path, s1, s2, dem = world
        r = run(tmp_path, FakePC(s1, s2, dem), max_hand_m=500.0)
        assert r.success
        cls, src = read(tmp_path)
        assert block_classes(cls, src, HILL) == {1}
        assert r.stats["excluded_by_hand_ha"] == 0


class TestMissingData:
    def test_cloudy_flood_window_gives_radar_and_terrain_only_and_says_so(self, world):
        tmp_path, s1, s2, dem = world
        s2[1]["properties"]["eo:cloud_cover"] = 99.0
        r = run(tmp_path, FakePC(s1, s2, dem))
        assert r.success, r.error
        cls, src = read(tmp_path)
        s = r.stats
        assert s["sensors_used"] == ["sentinel-1", "copernicus-dem"]
        assert "no Sentinel-2 scene" in s["data_notes"] and "flood window" in s["data_notes"]
        assert s["optical_coverage"] == 0.0
        assert block_classes(cls, src, FLOOD) == {2}               # no optical view: radar only, not "confirmed"
        assert not np.isin(cls, (1, 3, 4)).any()

    def test_no_dem_means_no_terrain_filter_and_a_note(self, world):
        tmp_path, s1, s2, _ = world
        r = run(tmp_path, FakePC(s1, s2, []))
        assert r.success, r.error
        cls, src = read(tmp_path)
        assert r.stats["hand_applied"] is False and "terrain" in r.stats["data_notes"]
        assert "copernicus-dem" not in r.stats["sensors_used"]
        assert block_classes(cls, src, HILL) == {1}
        assert not (tmp_path / "out" / "hand_m.tif").exists()

    def test_sensors_can_be_switched_off(self, world):
        tmp_path, s1, s2, dem = world
        r = run(tmp_path, FakePC(s1, s2, dem), use_optical=False, use_terrain=False)
        assert r.success, r.error
        assert r.stats["sensors_used"] == ["sentinel-1"]
        cls, src = read(tmp_path)
        assert block_classes(cls, src, HILL) == {2}
        assert block_classes(cls, src, OPT_ONLY) == {0}

    def test_scene_filter_uses_the_window_even_if_the_server_ignores_it(self, world):
        tmp_path, s1, s2, dem = world

        class Sloppy(FakePC):
            def search(self, collection, bbox, datetime=None, **kw):
                return self.data[collection]
        r = run(tmp_path, Sloppy(s1, s2, dem))
        assert r.success, r.error
        assert r.stats["reference_dates"] == ["2026-06-10"] and r.stats["flood_dates"] == ["2026-06-29"]
        assert r.stats["optical_flood_dates"] == ["2026-06-30"]


class TestSensorsTalk:
    def test_radar_threshold_is_calibrated_on_optical_so_dark_land_is_not_water(self, world):
        """Dark but dry ground (-17 dB) lies under the conventional -15 dB threshold; optical knows it is land."""
        tmp_path, _, s2, dem = world
        DARK = (0.4625, 5.9120, 0.4645, 5.9150)
        sb, sa = land(), land()
        sb[lonlat_mask(LAKE)] = -24.0
        sa[lonlat_mask(LAKE)] = -24.0
        sb[lonlat_mask(DARK)] = -17.0
        sa[lonlat_mask(DARK)] = -17.0
        sa[lonlat_mask(FLOOD)] = -22.0
        s1 = [item(1, "2026-06-10", scene(tmp_path / "db.tif", sb)), item(2, "2026-06-29", scene(tmp_path / "da.tif", sa))]
        r_auto = run(tmp_path / "auto", FakePC(s1, s2, dem))
        r_fix = run(tmp_path / "fix", FakePC(s1, s2, dem), water_db=-15.0)
        assert r_auto.success and r_fix.success
        assert r_auto.stats["water_db"] < -17.0                    # calibrated below the dark land
        assert r_fix.stats["water_db"] == -15.0 and r_fix.stats["water_db_source"] == "user"
        cls_auto, src = read(tmp_path / "auto")
        assert block_classes(cls_auto, src, DARK) == {0}           # dark dry ground: not water
        assert r_auto.stats["permanent_water_area_ha"] < r_fix.stats["permanent_water_area_ha"]   # -17 dB land is not "permanent water"

    def test_a_clear_flood_view_confirms_radar_even_without_an_optical_reference(self, world):
        tmp_path, s1, s2, dem = world
        r = run(tmp_path, FakePC(s1, s2[1:], dem))                 # no Sentinel-2 scene in the reference window
        assert r.success, r.error
        cls, src = read(tmp_path)
        assert block_classes(cls, src, FLOOD) == {1}               # radar flood + optical water in the flood window
        assert block_classes(cls, src, DISPUTED) == {4}
        assert block_classes(cls, src, OPT_ONLY) == {0}            # optical-only flooding needs an optical reference
        assert "no clear Sentinel-2 view in the reference window" in r.stats["data_notes"]
        assert r.stats["water_db_source"] == "conventional default" and r.stats["radar_optical_f1_reference"] is None

    def test_reference_optical_scenes_are_not_read_when_the_flood_window_has_no_optical_view(self, world):
        tmp_path, s1, s2, dem = world
        s2[1]["properties"]["eo:cloud_cover"] = 99.0

        class Spy(FakePC):
            def search(self, collection, bbox, datetime=None, **kw):
                self.searched = getattr(self, "searched", []) + [(collection, datetime)]
                return super().search(collection, bbox, datetime, **kw)
        pc = Spy(s1, s2, dem)
        r = run(tmp_path, pc)
        assert r.success
        assert not any(c == "sentinel-2-l2a" and d.startswith("2026-06-01") for c, d in pc.searched)
        assert not (tmp_path / "out" / "mndwi_after.tif").exists()     # nothing to show, so no blank layer

    def test_offset_between_the_two_windows_is_removed_before_the_change_test(self, world):
        tmp_path, _, _, dem = world
        MARGINAL = (0.4650, 5.9050, 0.4700, 5.9100)
        sb, sa = land(), land(-12.0)                               # the whole flood scene is 2 dB darker (calibration, wet soil)
        sb[lonlat_mask(MARGINAL)] = -11.0
        sa[lonlat_mask(MARGINAL)] = -15.5                          # 4.5 dB darker than before, only 2.5 beyond the offset
        sb[lonlat_mask(FLOOD)] = -10.0
        sa[lonlat_mask(FLOOD)] = -22.0
        s1 = [item(1, "2026-06-10", scene(tmp_path / "ob_.tif", sb)), item(2, "2026-06-29", scene(tmp_path / "oa_.tif", sa))]
        r = run(tmp_path, FakePC(s1, [], dem), use_optical=False, water_db=-15.0)
        assert r.success, r.error
        assert r.stats["radiometric_offset_db"] == pytest.approx(-2.0, abs=0.2)
        cls, src = read(tmp_path)
        assert block_classes(cls, src, FLOOD) == {2}
        assert block_classes(cls, src, MARGINAL) == {0}


class TestCalibration:
    def test_plateau_middle_and_too_little_to_calibrate(self):
        rng = np.random.default_rng(3)
        water = np.r_[rng.normal(-24, 1.0, 1000), rng.normal(-10, 1.0, 3000)]
        opt = np.r_[np.ones(1000, bool), np.zeros(3000, bool)]
        thr, f1 = calibrate_water_db(water, opt, np.ones(4000, bool))
        assert -19 < thr < -14 and f1 > 0.98
        assert calibrate_water_db(water[:100], opt[:100], np.ones(100, bool)) is None        # fewer than 200 of a class
        assert calibrate_water_db(water[1000:], opt[1000:], np.ones(3000, bool)) is None      # no water at all

    def test_agreement_scores(self):
        r = np.array([1, 1, 0, 0], bool)
        o = np.array([1, 0, 1, 0], bool)
        a = agreement(r, o, np.ones(4, bool))
        assert a["f1"] == pytest.approx(0.5) and a["iou"] == pytest.approx(1 / 3)


class TestOpticalReflectance:
    def test_scale_offset_follows_metadata_then_baseline(self):
        base5 = {"properties": {"s2:processing_baseline": "05.10"}, "assets": {"B03": {}}}
        base3 = {"properties": {"s2:processing_baseline": "03.00"}, "assets": {"B03": {}}}
        meta = {"properties": {"s2:processing_baseline": "05.10"},
                "assets": {"B03": {"raster:bands": [{"scale": 0.0001, "offset": 0.0}]}}}
        assert _scale_offset(base5, "B03") == (1e-4, -0.1)
        assert _scale_offset(base3, "B03") == (1e-4, 0.0)
        assert _scale_offset(meta, "B03") == (1e-4, 0.0)
        assert _scale_offset({"properties": {}, "assets": {}}, "B03") == (1e-4, 0.0)

    def test_mndwi_formula(self):
        g, s = np.array([0.12, 0.15, 0.0]), np.array([0.03, 0.25, 0.0])
        out = mndwi(g, s)
        assert out[0] == pytest.approx(0.6) and out[1] == pytest.approx(-0.25) and np.isnan(out[2])

    def test_processing_offset_matters_when_the_threshold_is_strict(self, world):
        """With baseline 05.10 the DNs carry a +1000 offset. Corrected, water has MNDWI 0.6 (> 0.5); read as plain
        reflectance it would be about 0.26 and be missed."""
        tmp_path, s1, _, dem = world
        ob = s2_scene(tmp_path / "ob2", [LAKE], shift=1000)
        oa = s2_scene(tmp_path / "oa2", [LAKE, FLOOD], shift=1000)
        s2 = [s2_item(1, "2026-06-12", ob, baseline="05.10"), s2_item(2, "2026-06-30", oa, baseline="05.10")]
        r = run(tmp_path, FakePC(s1, s2, dem), mndwi_threshold=0.5)
        assert r.success, r.error
        cls, src = read(tmp_path)
        assert block_classes(cls, src, FLOOD) == {1}
        assert block_classes(cls, src, LAKE) == {5}


class TestArguments:
    def test_both_dates_are_required(self, world):
        tmp_path, s1, s2, dem = world
        r = run(tmp_path, FakePC(s1, s2, dem), date_before=None)
        assert not r.success and "date_before" in r.error

    def test_reference_window_must_precede_the_flood(self, world):
        tmp_path, s1, s2, dem = world
        r = run(tmp_path, FakePC(s1, s2, dem), date_before="2026-07-01", date_after="2026-06-27/2026-07-03")
        assert not r.success and "before" in r.error

    def test_no_radar_scenes_is_an_error_not_an_empty_map(self, world):
        tmp_path, _, s2, dem = world
        r = run(tmp_path, FakePC([], s2, dem))
        assert not r.success
        assert not (tmp_path / "out" / "flood_extent.tif").exists()

    def test_registered_with_a_report_spec(self):
        from pygeovision.ai.pipelines import pipeline_catalogue
        from pygeovision.reporting.specs import SPECS
        assert "multisensor_flood_mapping" in pipeline_catalogue()
        assert pipeline_catalogue()["multisensor_flood_mapping"].kind == "ready"
        assert SPECS["multisensor_flood_mapping"].layers[0].file == "flood_extent.tif"


class TestReport:
    def test_report_is_built_with_ranges_land_percentages_and_no_blank_layers(self, world):
        from pygeovision.reporting import generate_deliverables
        tmp_path, s1, s2, dem = world
        r = run(tmp_path, FakePC(s1, s2, dem))
        assert r.success, r.error
        d = generate_deliverables(r, BBOX, tmp_path / "out", report_formats="pdf,docx")
        assert not [w for w in d.warnings if "fail" in w.lower() or "error" in w.lower()], d.warnings
        assert d.maps and d.charts and d.reports.get("pdf") and d.reports["pdf"].exists()
        import json
        period = json.loads(d.summary_json.read_text(encoding="utf-8"))["period"]
        assert period == "2026-06-10 to 2026-06-29"

    def test_a_date_range_on_either_side_reads_as_an_arrow(self):
        from pygeovision.reporting.build import _period
        assert _period({"date_before": "2026-06-02 \u2013 2026-06-09", "date_after": "2026-06-27 \u2013 2026-07-03"}, {}) == \
            "2026-06-02 \u2013 2026-06-09 \u2192 2026-06-27 \u2013 2026-07-03"
        assert _period({"date_before": "2020-01", "date_after": "2024-01"}, {}) == "2020-01 to 2024-01"
