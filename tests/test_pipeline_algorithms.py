"""Correctness tests for the pipelines that run without trained weights.

Each test builds a scene whose right answer is known by construction
(hand-computed index values, known regions, known cloud holes), runs the
real pipeline code on it, and checks the raster and the statistics. Several
tests pin specific defects found in the pre-launch audit; those are named
in the docstrings.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from pipeline_synth import BBOX, const, patch_downloads, read, write_raster  # noqa: E402

from pygeovision.data.radiometric import real_pixel_area_ha  # noqa: E402

SHAPE = (40, 40)
NODATA = 255


class FakeClient:
    """Pipelines only touch the client through search/download; patched per test."""


@pytest.fixture
def client():
    return FakeClient()


def px_ha(path) -> float:
    with rasterio.open(path) as src:
        return real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)


def band_stack(*bands):
    return np.stack(bands).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# crop_health
# ──────────────────────────────────────────────────────────────────────────
class TestCropHealth:
    def _scene(self, tmp_path):
        before = write_raster(tmp_path / "b.tif", band_stack(const(0.5), const(0.1)))      # nir, red -> NDVI 0.667
        nir_a, red_a = const(0.5), const(0.1)
        nir_a[:, :20], red_a[:, :20] = 0.3, 0.2                                            # NDVI 0.2: browned
        nir_a[:, 35:] = red_a[:, 35:] = np.nan                                             # cloud
        after = write_raster(tmp_path / "a.tif", band_stack(nir_a, red_a))
        return before, after

    def test_flags_only_the_browned_area_and_excludes_cloud(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines.domains import CropHealthPipeline
        patch_downloads(monkeypatch, pair=self._scene(tmp_path))
        r = CropHealthPipeline(client).run(BBOX, tmp_path / "o", date_before="2024-05", date_after="2024-08")
        assert r.success, r.error
        m = read(r.output_path)[0]
        assert (m[:, :20] == 1).all() and (m[:, 20:35] == 0).all() and (m[:, 35:] == NODATA).all()
        # 20 flagged of 35 valid columns; the old code divided by all 40 (0.5)
        assert r.stats["pct_flagged"] == pytest.approx(20 / 35)
        assert r.stats["valid_fraction"] == pytest.approx(35 / 40)
        assert r.stats["mean_ndvi_before"] == pytest.approx(0.4 / 0.6, abs=1e-4)

    def test_requires_both_dates(self, tmp_path, client):
        from pygeovision.ai.pipelines.domains import CropHealthPipeline
        r = CropHealthPipeline(client).run(BBOX, tmp_path, date_before="2024-05")
        assert not r.success and "date_after" in r.error


# ──────────────────────────────────────────────────────────────────────────
# vegetation_indices
# ──────────────────────────────────────────────────────────────────────────
class TestVegetationIndices:
    def test_exact_index_values_and_missing_dates(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines.domains import VegetationIndicesPipeline
        # bands requested: blue, red, nir, green
        img = write_raster(tmp_path / "s.tif", band_stack(const(0.05), const(0.1), const(0.5), const(0.08)))
        patch_downloads(monkeypatch, single={"2024-05": img, "2024-06": None})
        r = VegetationIndicesPipeline(client).run(BBOX, tmp_path / "o", dates=["2024-05", "2024-06"])
        assert r.success, r.error
        assert r.stats["n_dates_used"] == 1 and list(r.stats["per_date"]) == ["2024-05"]
        d = r.stats["per_date"]["2024-05"]
        assert d["mean_ndvi"] == pytest.approx(0.4 / 0.6, abs=1e-4)
        assert d["mean_evi"] == pytest.approx(2.5 * 0.4 / (0.5 + 6 * 0.1 - 7.5 * 0.05 + 1), abs=1e-4)
        assert d["mean_ndwi"] == pytest.approx((0.08 - 0.5) / (0.08 + 0.5), abs=1e-4)
        ndvi_band = read(d["indices_path"])[0]
        assert np.allclose(ndvi_band, 0.4 / 0.6, atol=1e-4)

    def test_needs_dates(self, tmp_path, client):
        from pygeovision.ai.pipelines.domains import VegetationIndicesPipeline
        assert not VegetationIndicesPipeline(client).run(BBOX, tmp_path).success


# ──────────────────────────────────────────────────────────────────────────
# water_quality
# ──────────────────────────────────────────────────────────────────────────
class TestWaterQuality:
    def test_ndci_only_inside_water_and_cloud_excluded(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines.domains import WaterQualityPipeline
        green, red, edge, nir = const(0.08), const(0.2), const(0.25), const(0.4)   # land
        green[:, :20], nir[:, :20], red[:, :20], edge[:, :20] = 0.10, 0.03, 0.04, 0.06   # water, NDCI 0.2
        for b in (green, red, edge, nir):
            b[0, :] = np.nan                                                           # cloud row
        img = write_raster(tmp_path / "s.tif", band_stack(green, red, edge, nir))
        patch_downloads(monkeypatch, single=img)
        r = WaterQualityPipeline(client).run(BBOX, tmp_path / "o")
        assert r.success, r.error
        out = read(r.output_path)
        assert np.allclose(out[0][1:, :20], 0.2, atol=1e-5)            # NDCI in water
        assert np.allclose(out[1][1:, :20], 0.04, atol=1e-6)           # red reflectance indicator
        assert np.isnan(out[0][1:, 20:]).all()                         # nothing outside water
        assert r.stats["pct_water"] == pytest.approx(0.5)              # of the 39 valid rows
        assert r.stats["mean_ndci_in_water"] == pytest.approx(0.2, abs=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# flood_mapping / coastal_monitoring (MNDWI)
# ──────────────────────────────────────────────────────────────────────────
def mndwi_scene(water_cols=None, water_mask=None, cloud=None):
    """(green, swir1) with MNDWI +0.667 on water and -0.51 on land."""
    g, s = const(0.08), const(0.25)
    if water_mask is None:
        water_mask = np.zeros(SHAPE, bool)
        for lo, hi in water_cols:
            water_mask[:, lo:hi] = True
    g[water_mask], s[water_mask] = 0.10, 0.02
    if cloud is not None:
        g[cloud] = s[cloud] = np.nan
    return band_stack(g, s)


class TestFloodMapping:
    def test_flood_is_new_water_and_cloud_is_nodata(self, tmp_path, monkeypatch, client):
        """Pre-audit, flood_mapping requested RGB (no NIR) and read the wrong
        bands as NDWI; it also could not separate permanent water."""
        from pygeovision.ai.pipelines.domains import FloodMappingPipeline
        cloud = np.zeros(SHAPE, bool); cloud[:4, :] = True
        before = write_raster(tmp_path / "b.tif", mndwi_scene([(20, 25)]))
        after = write_raster(tmp_path / "a.tif", mndwi_scene([(10, 35)], cloud=cloud))
        rec: dict = {}
        patch_downloads(monkeypatch, pair=(before, after), record=rec)
        r = FloodMappingPipeline(client).run(BBOX, tmp_path / "o", date_before="2024-01", date_after="2024-02")
        assert r.success, r.error
        assert rec["calls"][0]["bands"] == ("green", "swir1")
        c = read(r.output_path)[0]
        assert (c[:4] == NODATA).all()
        assert (c[4:, 10:20] == 1).all() and (c[4:, 25:35] == 1).all()     # flooded
        assert (c[4:, 20:25] == 2).all()                                    # permanent water
        assert (c[4:, :10] == 0).all() and (c[4:, 35:] == 0).all()          # dry
        ha = px_ha(r.output_path)
        assert r.stats["flooded_area_ha"] == pytest.approx(20 * 36 * ha)
        assert r.stats["permanent_water_area_ha"] == pytest.approx(5 * 36 * ha)
        assert r.stats["flooded_fraction"] == pytest.approx(20 / 40)

    def test_requires_a_dry_reference(self, tmp_path, client):
        from pygeovision.ai.pipelines.domains import FloodMappingPipeline
        r = FloodMappingPipeline(client).run(BBOX, tmp_path, date_after="2024-02")
        assert not r.success and "date_before" in r.error


class TestCoastalMonitoring:
    def test_retreat_and_advance(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines.domains import CoastalMonitoringPipeline
        wb = np.zeros(SHAPE, bool); wb[:, 25:] = True
        wa = np.zeros(SHAPE, bool); wa[:, 30:] = True; wa[:10, :5] = True
        before = write_raster(tmp_path / "b.tif", mndwi_scene(water_mask=wb))
        after = write_raster(tmp_path / "a.tif", mndwi_scene(water_mask=wa))
        patch_downloads(monkeypatch, pair=(before, after))
        r = CoastalMonitoringPipeline(client).run(BBOX, tmp_path / "o", date_before="2020-01", date_after="2024-01")
        assert r.success, r.error
        c = read(r.output_path)[0]
        assert (c[:, 30:] == 1).all()                    # water both dates
        assert (c[10:, :25] == 0).all()                  # land both dates
        assert (c[:, 25:30] == 3).all()                  # water -> land (advance), 5 cols x 40 rows
        assert (c[:10, :5] == 2).all()                   # land -> water, 10 x 5
        ha = px_ha(r.output_path)
        assert r.stats["water_to_land_ha"] == pytest.approx(200 * ha)
        assert r.stats["land_to_water_ha"] == pytest.approx(50 * ha)
        assert r.stats["net_water_change_ha"] == pytest.approx(-150 * ha)


# ──────────────────────────────────────────────────────────────────────────
# wetland_mapping
# ──────────────────────────────────────────────────────────────────────────
def test_wetland_separates_open_water_wet_vegetation_and_upland(tmp_path, monkeypatch, client):
    from pygeovision.ai.pipelines.domains import WetlandMappingPipeline
    blue, red, nir, green, swir = const(0.05), const(0.05), const(0.3), const(0.08), const(0.25)   # upland
    blue[:, :10], red[:, :10], nir[:, :10], green[:, :10], swir[:, :10] = .05, .04, .03, .10, .02  # open water
    blue[:, 10:20], red[:, 10:20], nir[:, 10:20], green[:, 10:20], swir[:, 10:20] = .04, .05, .40, .10, .02  # wet veg
    img = write_raster(tmp_path / "s.tif", band_stack(blue, red, nir, green, swir))
    patch_downloads(monkeypatch, single=img)
    r = WetlandMappingPipeline(client).run(BBOX, tmp_path / "o")
    assert r.success, r.error
    c = read(r.output_path)[0]
    assert (c[:, :10] == 1).all() and (c[:, 10:20] == 2).all() and (c[:, 20:] == 0).all()
    assert r.stats["pct_wet_vegetation"] == pytest.approx(0.25)
    assert r.stats["pct_open_water"] == pytest.approx(0.25)


# ──────────────────────────────────────────────────────────────────────────
# wildfire_severity
# ──────────────────────────────────────────────────────────────────────────
class TestWildfireSeverity:
    def test_classes_and_cloud_is_not_unburned(self, tmp_path, monkeypatch, client):
        """Pre-audit, cloud-masked (NaN) pixels fell into class 0 'unburned'
        and counted in every percentage."""
        from pygeovision.ai.pipelines.domains import WildfireSeverityPipeline
        pre = write_raster(tmp_path / "pre.tif", band_stack(const(0.4), const(0.1)))     # NBR 0.6
        nir, sw = const(0.4), const(0.1)
        # dNBR: 0.08 unburned | 0.267 low | 0.60 moderate | 0.933 high
        for (lo, hi), (n, s) in {(0, 10): (.38, .12), (10, 20): (.30, .15), (20, 30): (.25, .25), (30, 40): (.15, .30)}.items():
            nir[:, lo:hi], sw[:, lo:hi] = n, s
        nir[:10, :] = sw[:10, :] = np.nan                                                # cloud
        post = write_raster(tmp_path / "post.tif", band_stack(nir, sw))
        patch_downloads(monkeypatch, pair=(pre, post))
        r = WildfireSeverityPipeline(client).run(BBOX, tmp_path / "o", date_before="2018-10", date_after="2018-12")
        assert r.success, r.error
        c = read(r.output_path)[0]
        assert (c[:10] == NODATA).all()
        assert (c[10:, :10] == 0).all() and (c[10:, 10:20] == 1).all()
        assert (c[10:, 20:30] == 2).all() and (c[10:, 30:] == 3).all()
        for k in ("pct_unburned", "pct_low_severity", "pct_moderate_severity", "pct_high_severity"):
            assert r.stats[k] == pytest.approx(0.25)          # old code gave 0.4375 for unburned
        assert r.stats["burned_area_ha"] == pytest.approx(0.75 * 30 * 40 * px_ha(r.output_path))
        dnbr = read(r.output_path.parent / "dnbr.tif")[0]
        assert dnbr[20, 25] == pytest.approx(0.6, abs=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# snow_cover / glacier_monitoring
# ──────────────────────────────────────────────────────────────────────────
def ndsi_scene(snow_cols, cloud_cols=None):
    g, s = const(0.1), const(0.2)                                  # NDSI -0.33
    g[:, snow_cols[0]:snow_cols[1]], s[:, snow_cols[0]:snow_cols[1]] = 0.8, 0.1   # NDSI +0.78
    if cloud_cols:
        g[:, cloud_cols[0]:cloud_cols[1]] = s[:, cloud_cols[0]:cloud_cols[1]] = np.nan
    return band_stack(g, s)


def test_snow_cover_fraction_is_over_valid_pixels(tmp_path, monkeypatch, client):
    from pygeovision.ai.pipelines.domains import SnowCoverPipeline
    img = write_raster(tmp_path / "s.tif", ndsi_scene((0, 20), cloud_cols=(36, 40)))
    patch_downloads(monkeypatch, single=img)
    r = SnowCoverPipeline(client).run(BBOX, tmp_path / "o")
    assert r.success, r.error
    assert r.stats["snow_cover_fraction"] == pytest.approx(20 / 36)
    assert r.stats["snow_area_ha"] == pytest.approx(20 * 40 * px_ha(r.output_path))
    assert (read(r.output_path)[0][:, 36:] == NODATA).all()


class TestGlacier:
    def test_cloud_over_ice_is_not_reported_as_retreat(self, tmp_path, monkeypatch, client):
        """Ice is unchanged, but the later image has cloud over part of it.
        Pre-audit this read as a 33% retreat."""
        from pygeovision.ai.pipelines.domains import GlacierMonitoringPipeline
        before = write_raster(tmp_path / "b.tif", ndsi_scene((0, 30)))
        after = write_raster(tmp_path / "a.tif", ndsi_scene((0, 30), cloud_cols=(20, 30)))
        patch_downloads(monkeypatch, pair=(before, after))
        r = GlacierMonitoringPipeline(client).run(BBOX, tmp_path / "o", date_before="2023-09", date_after="2024-09")
        assert r.success, r.error
        assert r.stats["area_change_ha"] == pytest.approx(0.0)
        assert r.stats["area_ha_before"] == pytest.approx(20 * 40 * px_ha(r.output_path))   # only commonly valid pixels

    def test_real_retreat_is_measured(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines.domains import GlacierMonitoringPipeline
        before = write_raster(tmp_path / "b.tif", ndsi_scene((0, 30)))
        after = write_raster(tmp_path / "a.tif", ndsi_scene((0, 24)))
        patch_downloads(monkeypatch, pair=(before, after))
        r = GlacierMonitoringPipeline(client).run(BBOX, tmp_path / "o", date_before="2023-09", date_after="2024-09")
        assert r.stats["area_change_pct"] == pytest.approx(-20.0)


# ──────────────────────────────────────────────────────────────────────────
# Landsat thermal: urban_heat_island / land_surface_temperature / volcano
# ──────────────────────────────────────────────────────────────────────────
def to_dn(temp_c: np.ndarray) -> np.ndarray:
    return np.rint((temp_c + 273.15 - 149.0) / 0.00341802).astype(np.uint16)


@pytest.fixture
def thermal_scene(tmp_path):
    t = np.full(SHAPE, 25.0)
    t[5:13, 5:13] = 30.0          # 64 px, +5 C
    t[30:34, 30:34] = 80.0        # 16 px, +55 C
    dn = to_dn(t)
    dn[0, :] = 0                  # fill / cloud
    return write_raster(tmp_path / "t.tif", dn[None], dtype="uint16"), t, dn


class TestThermal:
    def test_lst_conversion_and_heat_island_mask(self, tmp_path, monkeypatch, client, thermal_scene):
        from pygeovision.ai.pipelines.domains import UrbanHeatIslandPipeline
        path, t, dn = thermal_scene
        rec: dict = {}
        patch_downloads(monkeypatch, single=path, record=rec)
        r = UrbanHeatIslandPipeline(client).run(BBOX, tmp_path / "o", threshold_c=2.0)
        assert r.success, r.error
        assert rec["calls"][0]["apply_scale"] is False and rec["calls"][0]["bands"] == ("thermal",)
        lst = read(r.output_path)[0]
        assert lst[10, 10] == pytest.approx(30.0, abs=0.01) and lst[20, 20] == pytest.approx(25.0, abs=0.01)
        assert np.isnan(lst[0]).all()                                  # DN 0 is fill, not 0 K
        valid = dn > 0
        mean_c = (dn[valid] * 0.00341802 + 149.0 - 273.15).mean()
        assert r.stats["mean_temp_celsius"] == pytest.approx(mean_c, abs=1e-3)
        hot = (((dn * 0.00341802 + 149.0 - 273.15) - mean_c) > 2.0) & valid
        assert r.stats["pct_heat_island"] == pytest.approx(hot.sum() / valid.sum())
        assert read(r.stats["mask_path"])[0][10, 10] == 1

    def test_threshold_is_configurable(self, tmp_path, monkeypatch, client, thermal_scene):
        from pygeovision.ai.pipelines.domains import UrbanHeatIslandPipeline
        patch_downloads(monkeypatch, single=thermal_scene[0])
        r = UrbanHeatIslandPipeline(client).run(BBOX, tmp_path / "o", threshold_c=100.0)
        assert r.stats["pct_heat_island"] == 0.0

    def test_volcano_flags_only_the_extreme_patch(self, tmp_path, monkeypatch, client, thermal_scene):
        from pygeovision.ai.pipelines.domains import VolcanoMonitoringPipeline
        patch_downloads(monkeypatch, single=thermal_scene[0])
        r = VolcanoMonitoringPipeline(client).run(BBOX, tmp_path / "o")
        assert r.success, r.error
        mask = read(r.stats["mask_path"])[0]
        assert (mask[30:34, 30:34] == 1).all()                         # 80 C
        assert (mask[5:13, 5:13] == 0).all()                           # 30 C is only +5 C
        assert r.stats["pct_thermal_anomaly"] == pytest.approx(16 / (SHAPE[0] * SHAPE[1] - SHAPE[1]))

    def test_lst_alias_reports_same_temperatures(self, tmp_path, monkeypatch, client, thermal_scene):
        from pygeovision.ai.pipelines.domains import LandSurfaceTemperaturePipeline
        patch_downloads(monkeypatch, single=thermal_scene[0])
        r = LandSurfaceTemperaturePipeline(client).run(BBOX, tmp_path / "o")
        assert r.success and read(r.output_path)[0][10, 10] == pytest.approx(30.0, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────
# water_bodies / change_detection (model-free paths)
# ──────────────────────────────────────────────────────────────────────────
class TestWaterBodies:
    def test_ndwi_threshold_and_nodata(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines import WaterBodiesPipeline
        g, n = const(0.08), const(0.4)                                    # land: NDWI -0.67
        g[:, :10], n[:, :10] = 0.10, 0.03                                 # water: +0.54
        g[:, 10:15], n[:, 10:15] = 0.08, 0.05                             # marginal: +0.23
        g[:, 36:] = n[:, 36:] = np.nan                                    # cloud
        img = write_raster(tmp_path / "s.tif", band_stack(g, n))
        patch_downloads(monkeypatch, single=img)
        r = WaterBodiesPipeline(client).run(BBOX, tmp_path / "o")
        assert r.success, r.error
        assert r.stats["water_coverage"] == pytest.approx(10 / 36)         # default threshold 0.3
        mask = read(r.metadata["mask_path"])[0]
        assert (mask[:, 36:] == NODATA).all() and (mask[:, 10:15] == 0).all()
        r2 = WaterBodiesPipeline(client).run(BBOX, tmp_path / "o2", ndwi_threshold=0.0)
        assert r2.stats["water_coverage"] == pytest.approx(15 / 36)        # marginal pixels now count

    def test_unknown_method_rejected(self, tmp_path, client):
        from pygeovision.ai.pipelines import WaterBodiesPipeline
        assert not WaterBodiesPipeline(client).run(BBOX, tmp_path, method="magic").success


class TestChangeDetectionCVA:
    BANDS = (0.1, 0.2, 0.15, 0.4)   # blue, green, red, nir

    def _pair(self, tmp_path, change_cols=None, cloud_cols=None, seed=0):
        rs = np.random.RandomState(seed)
        base = np.stack([const(v) for v in self.BANDS])
        a = base + rs.normal(0, 0.002, base.shape).astype(np.float32)
        b = base + rs.normal(0, 0.002, base.shape).astype(np.float32)
        if change_cols:
            b[:, :, change_cols[0]:change_cols[1]] += np.array([0.1, 0.1, 0.15, -0.25], dtype=np.float32)[:, None, None]
        if cloud_cols:
            b[:, :, cloud_cols[0]:cloud_cols[1]] = np.nan
        return write_raster(tmp_path / "a.tif", a), write_raster(tmp_path / "b.tif", b)

    def test_unchanged_scene_reports_no_change(self, tmp_path, monkeypatch, client):
        """Otsu alone splits pure noise into two classes; the magnitude floor stops that."""
        from pygeovision.ai.pipelines import ChangeDetectionPipeline
        patch_downloads(monkeypatch, pair=self._pair(tmp_path))
        r = ChangeDetectionPipeline(client).run(BBOX, tmp_path / "o", date_before="2022-06", date_after="2024-06")
        assert r.success, r.error
        assert r.stats["change_fraction"] == 0.0

    def test_changed_half_is_found(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines import ChangeDetectionPipeline
        patch_downloads(monkeypatch, pair=self._pair(tmp_path, change_cols=(0, 20)))
        r = ChangeDetectionPipeline(client).run(BBOX, tmp_path / "o", date_before="2022-06", date_after="2024-06")
        assert r.stats["change_fraction"] == pytest.approx(0.5)
        m = read(r.output_path)[0]
        assert (m[:, :20] == 1).all() and (m[:, 20:] == 0).all()
        assert r.stats["changed_area_ha"] == pytest.approx(20 * 40 * px_ha(r.output_path))

    def test_cloud_in_either_date_is_nodata(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines import ChangeDetectionPipeline
        patch_downloads(monkeypatch, pair=self._pair(tmp_path, change_cols=(0, 20), cloud_cols=(36, 40)))
        r = ChangeDetectionPipeline(client).run(BBOX, tmp_path / "o", date_before="2022-06", date_after="2024-06")
        assert r.stats["change_fraction"] == pytest.approx(20 / 36)
        assert (read(r.output_path)[0][:, 36:] == NODATA).all()

    def test_requested_bands_are_the_four_spectral_bands(self, tmp_path, monkeypatch, client):
        from pygeovision.ai.pipelines import ChangeDetectionPipeline
        rec: dict = {}
        patch_downloads(monkeypatch, pair=self._pair(tmp_path), record=rec)
        ChangeDetectionPipeline(client).run(BBOX, tmp_path / "o", date_before="2022-06", date_after="2024-06")
        assert rec["calls"][0]["bands"] == ("blue", "green", "red", "nir")


# ──────────────────────────────────────────────────────────────────────────
# Land cover: the WorldCover reference map
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture
def worldcover_home(tmp_path, monkeypatch):
    """A synthetic WorldCover 2021 tile (N51 W003, 0.01 deg/px): trees west of lon -0.12, built-up east."""
    home = tmp_path / "home"
    cache = home / ".pygeovision" / "cache" / "esa_worldcover"
    cache.mkdir(parents=True)
    classes = np.full((300, 300), 10, np.uint8)
    classes[:, 288:] = 50                                  # column 288 is lon -3 + 288*0.01 = -0.12
    from rasterio.transform import from_origin
    with rasterio.open(cache / "ESA_WorldCover_10m_2021_v200_N51W003_Map.tif", "w", driver="GTiff", height=300, width=300,
                       count=1, dtype="uint8", crs="EPSG:4326", transform=from_origin(-3, 54, 0.01, 0.01)) as dst:
        dst.write(classes[None])
    monkeypatch.setenv("HOME", str(home))
    return home


class TestLandCoverWorldCover:
    def test_returns_the_reference_map_for_the_bbox(self, tmp_path, worldcover_home, client):
        from pygeovision.ai.pipelines import LandCoverPipeline
        r = LandCoverPipeline(client).run(BBOX, tmp_path / "o")
        assert r.success, r.error
        lc = read(r.output_path)[0]
        assert set(np.unique(lc)) == {10, 50}
        assert lc[:, 0].max() == 10 and lc[:, -1].min() == 50
        assert r.metadata["worldcover_year"] == 2021 and r.metadata["date_used"] is False

    def test_date_does_not_change_the_result(self, tmp_path, worldcover_home, client):
        """It is a static 2021 map, so it says so instead of pretending to follow the date."""
        from pygeovision.ai.pipelines import LandCoverPipeline
        a = LandCoverPipeline(client).run(BBOX, tmp_path / "a", date="2018-06")
        b = LandCoverPipeline(client).run(BBOX, tmp_path / "b", date="2024-06")
        assert np.array_equal(read(a.output_path), read(b.output_path))

    def test_needs_no_imagery(self, tmp_path, worldcover_home, monkeypatch, client):
        from pipeline_synth import no_network
        from pygeovision.ai.pipelines import LandCoverPipeline
        hit = no_network(monkeypatch)
        assert LandCoverPipeline(client).run(BBOX, tmp_path / "o").success and hit["n"] == 0

    def test_year_selects_the_right_edition_url(self, tmp_path, monkeypatch):
        """2020 is WorldCover v100, 2021 is v200; the labeler used v200 for both."""
        from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
        monkeypatch.setenv("HOME", str(tmp_path))
        seen = []

        class Resp:
            status_code = 404
            def raise_for_status(self):
                import requests
                raise requests.HTTPError(response=self)

        import requests
        monkeypatch.setattr(requests, "get", lambda url, **kw: seen.append(url) or Resp())
        ESAWorldCoverLabeler(year=2020)._download_tile(51, -3)
        ESAWorldCoverLabeler(year=2021)._download_tile(51, -3)
        assert "/v100/2020/map/ESA_WorldCover_10m_2020_v100_N51W003_Map.tif" in seen[0]
        assert "/v200/2021/map/ESA_WorldCover_10m_2021_v200_N51W003_Map.tif" in seen[1]


# ──────────────────────────────────────────────────────────────────────────
# biodiversity_hotspot
# ──────────────────────────────────────────────────────────────────────────
def brute_force_shannon(classes, window, nodata=0):
    h, w = classes.shape
    r = window // 2
    out = np.full((h, w), np.nan)
    for i in range(h):
        for j in range(w):
            if classes[i, j] == nodata:
                continue
            win = classes[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
            win = win[win != nodata]
            _, counts = np.unique(win, return_counts=True)
            p = counts / counts.sum()
            out[i, j] = -(p * np.log(p)).sum()
    return out


class TestLocalShannon:
    @pytest.mark.parametrize("window", [3, 5, 9])
    def test_matches_brute_force_with_nodata(self, window):
        from pygeovision.ai.pipelines.domains import local_shannon
        rs = np.random.RandomState(3)
        classes = rs.choice([10, 30, 40, 50], size=(25, 31)).astype(np.uint8)
        classes[rs.rand(25, 31) < 0.1] = 0
        got = local_shannon(classes, window)
        want = brute_force_shannon(classes, window)
        assert np.allclose(got, want, atol=1e-5, equal_nan=True)

    def test_known_values(self):
        from pygeovision.ai.pipelines.domains import local_shannon
        assert np.allclose(local_shannon(np.full((9, 9), 10, np.uint8), 5), 0.0)
        two = np.tile(np.array([[10, 20], [20, 10]], np.uint8), (6, 6))
        assert local_shannon(two, 3)[5, 5] == pytest.approx(np.log(2), abs=0.07)
        with pytest.raises(ValueError):
            local_shannon(two, 4)


def test_biodiversity_pipeline_outputs(tmp_path, monkeypatch, client):
    from pygeovision.ai.pipelines import LandCoverPipeline, PipelineResult
    from pygeovision.ai.pipelines.domains import BiodiversityHotspotPipeline
    classes = np.full(SHAPE, 10, np.uint8)
    classes[:20, 20:] = 40
    classes[20:, :20] = 50
    classes[20:, 20:] = 80                                      # four equal quadrants

    def fake_run(self, bbox, output_dir, **kw):
        path = write_raster(tmp_path / "lc" / "land_cover.tif", classes[None], dtype="uint8", nodata=0)
        return PipelineResult(pipeline="land_cover", output_path=path)

    monkeypatch.setattr(LandCoverPipeline, "run", fake_run)
    r = BiodiversityHotspotPipeline(client).run(BBOX, tmp_path / "o", window_px=5)
    assert r.success, r.error
    h = read(r.output_path)[0]
    assert h[5, 5] == pytest.approx(0.0, abs=1e-6)              # inside one quadrant
    assert h[19, 19] > 1.0                                      # corner where four classes meet (max ln4 = 1.386)
    assert r.stats["region_shannon_index"] == pytest.approx(np.log(4), abs=1e-6)
    assert r.stats["region_evenness"] == pytest.approx(1.0, abs=1e-6)
    assert r.stats["n_habitat_classes"] == 4
    hot = read(r.output_path.parent / "hotspot_mask.tif")[0]
    assert hot[19, 19] == 1 and hot[5, 5] == 0


def test_biodiversity_uniform_area_has_no_hotspots(tmp_path, monkeypatch, client):
    """One land-cover class everywhere: the 90th percentile of diversity is 0, and
    ``h >= 0`` used to flag every pixel as a hotspot (100 % 'hotspot')."""
    from pygeovision.ai.pipelines import LandCoverPipeline, PipelineResult
    from pygeovision.ai.pipelines.domains import BiodiversityHotspotPipeline
    classes = np.full(SHAPE, 30, np.uint8)
    classes[0, 0] = 10                                          # a single odd pixel so the region has 2 classes

    def fake_run(self, bbox, output_dir, **kw):
        path = write_raster(tmp_path / "lc" / "land_cover.tif", classes[None], dtype="uint8", nodata=0)
        return PipelineResult(pipeline="land_cover", output_path=path)

    monkeypatch.setattr(LandCoverPipeline, "run", fake_run)
    r = BiodiversityHotspotPipeline(client).run(BBOX, tmp_path / "o", window_px=5)
    assert r.success, r.error
    hot = read(r.output_path.parent / "hotspot_mask.tif")[0]
    assert hot[20, 20] == 0, "a pixel with zero local diversity can never be a hotspot"
    assert r.stats["hotspot_fraction"] < 0.05


# ──────────────────────────────────────────────────────────────────────────
# DEM pipelines: solar_potential / archaeological_site / wind_farm_siting
# ──────────────────────────────────────────────────────────────────────────
class FakeResult:
    def __init__(self, code):
        self.id = f"opentopo_{code}_-0.15_51.47"
        self.properties = {"dem_type": code}


class FakeDownload:
    def __init__(self, path):
        self.success, self.path = True, path


class DemClient:
    """OpenTopography stand-in: offers several products, returns a different DEM per product."""

    CODES = ["SRTMGL3", "SRTMGL1", "COP30", "NASADEM"]

    def __init__(self, dems: dict[str, str]):
        self.dems, self.downloaded = dems, []

    def search(self, **kw):
        return [FakeResult(c) for c in self.CODES]

    def download(self, results, out_dir):
        code = results[0].id.split("_")[1]
        self.downloaded.append(code)
        return [FakeDownload(self.dems[code])]


def plane(rows_per_m=0.2, shape=(60, 60), north_high=True):
    """DEM rising toward the north (south-facing slope) or the south (north-facing).

    Rise per row = gradient x the real cell height (0.05 deg / 60 rows ~ 92.6 m).
    """
    dy = 0.05 * 111_200 / shape[0]
    rows = np.arange(shape[0], dtype=np.float64)[:, None] * np.ones((1, shape[1]))
    z = rows_per_m * dy * (shape[0] - 1 - rows if north_high else rows)
    return z


@pytest.fixture
def dem_client(tmp_path):
    files = {c: write_raster(tmp_path / f"{c}.tif", plane(0.2) + i * 100.0, dtype="float32")
             for i, c in enumerate(DemClient.CODES)}
    return DemClient({c: str(p) for c, p in files.items()})


class TestDemSelection:
    def test_default_is_srtm_30m_not_whichever_product_comes_first(self, tmp_path, dem_client):
        """The provider lists SRTM 90 m first; dem_type was ignored so that is what ran."""
        from pygeovision.ai.pipelines.domains import SolarPotentialPipeline
        r = SolarPotentialPipeline(dem_client).run(BBOX, tmp_path / "o")
        assert r.success, r.error
        assert dem_client.downloaded == ["SRTMGL1"]

    def test_explicit_product_and_api_code(self, tmp_path, dem_client):
        from pygeovision.ai.pipelines.domains import SolarPotentialPipeline
        SolarPotentialPipeline(dem_client).run(BBOX, tmp_path / "a", dem_type="cop30")
        SolarPotentialPipeline(dem_client).run(BBOX, tmp_path / "b", dem_type="NASADEM")
        assert dem_client.downloaded == ["COP30", "NASADEM"]

    def test_unavailable_product_lists_what_was_offered(self, tmp_path, dem_client):
        from pygeovision.ai.pipelines.domains import SolarPotentialPipeline
        r = SolarPotentialPipeline(dem_client).run(BBOX, tmp_path / "o", dem_type="alos")
        assert not r.success and "SRTMGL1" in r.error

    def test_missing_api_key_message(self, tmp_path):
        from pygeovision.ai.pipelines.domains import SolarPotentialPipeline

        class Empty:
            def search(self, **kw): return []

        r = SolarPotentialPipeline(Empty()).run(BBOX, tmp_path)
        assert not r.success and "API key" in r.error


class TestSolarPotential:
    def _run(self, tmp_path, north_high, **kw):
        f = write_raster(tmp_path / "d.tif", plane(0.2, north_high=north_high), dtype="float32")
        client = DemClient({"SRTMGL1": str(f)}); client.CODES = ["SRTMGL1"]
        from pygeovision.ai.pipelines.domains import SolarPotentialPipeline
        return SolarPotentialPipeline(client).run(BBOX, tmp_path / "o", **kw)

    def test_south_facing_slope_gets_more_sun_than_north_facing_at_noon(self, tmp_path):
        slope = np.arctan(0.2)
        lat, decl = np.radians(51.495), np.radians(23.44)
        zen = lat - decl                                                   # sun due south at noon
        south = self._run(tmp_path / "s", True, date="2024-06-21")
        north = self._run(tmp_path / "n", False, date="2024-06-21")
        assert south.success and north.success
        s_val = read(south.output_path)[0][30, 30]
        n_val = read(north.output_path)[0][30, 30]
        assert s_val == pytest.approx(np.cos(zen - slope), abs=0.01)       # sun 16.7 deg off the slope normal
        assert n_val == pytest.approx(np.cos(zen + slope), abs=0.01)
        assert s_val > n_val + 0.15
        assert south.stats["sun_zenith_deg"] == pytest.approx(np.degrees(zen), abs=0.3)

    def test_sun_below_the_horizon_gives_zero(self, tmp_path):
        r = self._run(tmp_path, True, date="2024-06-21", hour=0.0)
        assert r.success and np.nanmax(read(r.output_path)[0]) == 0.0

    def test_bad_date_is_a_clear_error(self, tmp_path):
        r = self._run(tmp_path, True, date="2024-06")
        assert not r.success and "YYYY-MM-DD" in r.error


class TestArchaeologicalSite:
    def test_local_relief_model_finds_a_mound_and_handles_nodata(self, tmp_path):
        from pygeovision.ai.pipelines.domains import ArchaeologicalSitePipeline
        yy, xx = np.mgrid[0:60, 0:60]
        dem = 100.0 + 5.0 * np.exp(-(((yy - 30) ** 2 + (xx - 30) ** 2) / (2 * 2.5 ** 2)))   # 5 m mound
        dem[5:8, 5:8] = -9999.0                                                          # void
        f = write_raster(tmp_path / "d.tif", dem, dtype="float32", nodata=-9999.0)
        client = DemClient({"SRTMGL1": str(f)}); client.CODES = ["SRTMGL1"]
        r = ArchaeologicalSitePipeline(client).run(BBOX, tmp_path / "o", smoothing_radius_px=10)
        assert r.success, r.error
        lrm = read(r.output_path)[0]
        assert np.unravel_index(np.nanargmax(lrm), lrm.shape) == (30, 30)
        assert np.nanmax(lrm) > 3.5
        assert np.isnan(lrm[5:8, 5:8]).all() and np.isfinite(lrm[30, 40])
        hs = read(r.stats["hillshade_path"])[0]
        assert np.nanmin(hs) >= 0.0 and np.nanmax(hs) <= 1.0
        assert r.stats["dem_cell_size_m"][0] > 0


class TestWindFarmSiting:
    def _client(self, tmp_path, gradient):
        f = write_raster(tmp_path / "d.tif", plane(gradient), dtype="float32")
        c = DemClient({"SRTMGL1": str(f)}); c.CODES = ["SRTMGL1"]
        return c

    def test_slope_constraint(self, tmp_path):
        from pygeovision.ai.pipelines.domains import WindFarmSitingPipeline
        c = self._client(tmp_path, 0.2)                     # 11.3 deg
        ok = WindFarmSitingPipeline(c).run(BBOX, tmp_path / "a", max_slope_deg=15.0, use_landcover=False)
        no = WindFarmSitingPipeline(c).run(BBOX, tmp_path / "b", max_slope_deg=10.0, use_landcover=False)
        assert ok.stats["pct_area_suitable_by_slope"] == pytest.approx(1.0)
        assert no.stats["pct_area_suitable_by_slope"] < 0.1
        assert ok.stats["landcover_exclusion_applied"] is False

    def test_landcover_is_resampled_onto_the_dem_grid(self, tmp_path, monkeypatch):
        """Pre-audit the WorldCover mask came from a Sentinel-2 grid and was dropped
        whenever its shape differed from the DEM's (always), so water was never excluded."""
        from pygeovision.ai.labeling.base_labeler import LabelingResult
        from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
        from pygeovision.ai.pipelines.domains import WindFarmSitingPipeline
        c = self._client(tmp_path, 0.2)
        grids = {}

        def fake_label(self, tile_path, meta, out_path):
            with rasterio.open(tile_path) as src:
                grids["shape"] = (src.height, src.width)
            lc = np.full(grids["shape"], 30, np.uint8)
            lc[:, :15] = 80                                  # open water
            write_raster(out_path, lc[None], dtype="uint8")
            return LabelingResult(tile_id="t", label_path=out_path, source="esa_worldcover")

        monkeypatch.setattr(ESAWorldCoverLabeler, "label_tile", fake_label)
        r = WindFarmSitingPipeline(c).run(BBOX, tmp_path / "o")
        assert r.success, r.error
        assert grids["shape"] == (60, 60) and r.stats["landcover_exclusion_applied"] is True
        score = read(r.output_path)[0]
        assert np.isnan(score[:, :15]).all() and np.isfinite(score[10:50, 20:50]).all()

    def test_unreachable_landcover_fails_instead_of_silently_skipping(self, tmp_path, monkeypatch):
        from pygeovision.ai.labeling.base_labeler import LabelingResult
        from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
        from pygeovision.ai.pipelines.domains import WindFarmSitingPipeline
        monkeypatch.setattr(ESAWorldCoverLabeler, "label_tile",
                            lambda self, *a: LabelingResult(tile_id="t", label_path=None, source="x", error="offline"))
        r = WindFarmSitingPipeline(self._client(tmp_path, 0.2)).run(BBOX, tmp_path / "o")
        assert not r.success and "use_landcover=False" in r.error
