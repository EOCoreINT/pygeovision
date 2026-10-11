"""vegetation_anomaly and water_extent_timeseries on scenes whose answer is known by construction."""
from __future__ import annotations

import csv

import numpy as np
import pytest
import rasterio

from pygeovision.ai.pipelines.timeseries import VegetationAnomalyPipeline, WaterExtentTimeseriesPipeline
from tests.pipeline_synth import BBOX, patch_downloads, write_raster

N = 40


def red_nir(ndvi: np.ndarray):
    """red/nir that give exactly ``ndvi`` (NaN stays NaN in both bands)."""
    red = np.full(ndvi.shape, 0.1, "float32")
    nir = (red * (1 + ndvi) / (1 - ndvi)).astype("float32")
    red = np.where(np.isfinite(ndvi), red, np.nan).astype("float32")
    return np.stack([red, nir])


def ndvi_field(left: float, right: float) -> np.ndarray:
    a = np.full((N, N), right, "float32")
    a[:, : N // 2] = left
    return a


def tif(tmp_path, name, bands, **kw):
    return write_raster(tmp_path / "src" / f"{name}.tif", bands, bounds=BBOX, nodata=np.nan, **kw)


class TestVegetationAnomaly:
    def scenes(self, tmp_path):
        base = [0.5, 0.6, 0.5, 0.6, 0.5]
        s = {f"{2019 + i}-06": tif(tmp_path, f"b{2019 + i}", red_nir(ndvi_field(v, v))) for i, v in enumerate(base)}
        s["2024-06"] = tif(tmp_path, "t", red_nir(ndvi_field(0.2, 0.55)))
        return s

    def test_known_zscore_and_classes(self, tmp_path, monkeypatch):
        patch_downloads(monkeypatch, self.scenes(tmp_path))
        r = VegetationAnomalyPipeline(object()).run(BBOX, tmp_path / "o", date="2024-06")
        assert r.success, r.error
        with rasterio.open(tmp_path / "o" / "ndvi_zscore.tif") as s:
            z = s.read(1)
        std = np.std([0.5, 0.6, 0.5, 0.6, 0.5], ddof=1)
        assert z[0, 0] == pytest.approx((0.2 - 0.54) / std, rel=1e-3)
        assert z[0, -1] == pytest.approx((0.55 - 0.54) / std, rel=1e-2)
        with rasterio.open(tmp_path / "o" / "anomaly_class.tif") as s:
            c = s.read(1)
        assert (c[:, : N // 2] == 0).all() and (c[:, N // 2:] == 2).all()
        assert r.stats["pct_below_normal"] == pytest.approx(0.5)
        assert r.stats["pct_above_normal"] == pytest.approx(0.0)
        assert r.stats["mean_anomaly"] == pytest.approx(0.5 * (0.2 - 0.54) + 0.5 * (0.55 - 0.54), abs=1e-4)
        assert r.stats["baseline_years_used"] == [2019, 2020, 2021, 2022, 2023]

    def test_vci_is_0_for_the_lowest_on_record_and_100_for_the_highest(self, tmp_path, monkeypatch):
        s = self.scenes(tmp_path)
        s["2024-06"] = tif(tmp_path, "t2", red_nir(ndvi_field(0.2, 0.7)))      # below / above every baseline year
        patch_downloads(monkeypatch, s)
        r = VegetationAnomalyPipeline(object()).run(BBOX, tmp_path / "o", date="2024-06")
        with rasterio.open(tmp_path / "o" / "vci.tif") as src:
            v = src.read(1)
        assert v[0, 0] == pytest.approx(0.0, abs=1e-3) and v[0, -1] == pytest.approx(100.0, abs=1e-3)
        assert r.success

    def test_cloud_in_the_target_is_nodata_not_normal(self, tmp_path, monkeypatch):
        s = self.scenes(tmp_path)
        t = ndvi_field(0.2, 0.55)
        t[:10, :] = np.nan
        s["2024-06"] = tif(tmp_path, "tc", red_nir(t))
        patch_downloads(monkeypatch, s)
        r = VegetationAnomalyPipeline(object()).run(BBOX, tmp_path / "o", date="2024-06")
        with rasterio.open(tmp_path / "o" / "anomaly_class.tif") as src:
            c = src.read(1)
        assert (c[:10] == 255).all() and (c[10:] != 255).all()
        assert r.stats["valid_fraction"] == pytest.approx(0.75)
        assert r.stats["pct_below_normal"] == pytest.approx(0.5)            # over observed pixels only

    def test_a_pixel_clouded_in_most_baseline_years_is_nodata(self, tmp_path, monkeypatch):
        s = self.scenes(tmp_path)
        for i in range(2019, 2022):                                         # 3 of 5 years clouded at the corner
            a = ndvi_field(0.5, 0.5)
            a[0, 0] = np.nan
            s[f"{i}-06"] = tif(tmp_path, f"c{i}", red_nir(a))
        patch_downloads(monkeypatch, s)
        r = VegetationAnomalyPipeline(object()).run(BBOX, tmp_path / "o", date="2024-06")
        with rasterio.open(tmp_path / "o" / "ndvi_zscore.tif") as src:
            z = src.read(1)
        assert np.isnan(z[0, 0]) and np.isfinite(z[0, 1])
        assert r.stats["min_baseline_observations"] >= 3

    def test_too_few_baseline_years_fails_and_names_the_missing_ones(self, tmp_path, monkeypatch):
        s = self.scenes(tmp_path)
        for y in (2019, 2020, 2021):
            s.pop(f"{y}-06")
        patch_downloads(monkeypatch, s)
        r = VegetationAnomalyPipeline(object()).run(BBOX, tmp_path / "o", date="2024-06")
        assert not r.success and "2019" in r.error and "at least 3" in r.error

    def test_input_validation(self, tmp_path):
        p = VegetationAnomalyPipeline(object())
        assert "YYYY-MM" in p.run(BBOX, tmp_path, date=None).error
        assert "earlier" in p.run(BBOX, tmp_path, date="2024-06", baseline_years=[2024, 2020, 2021]).error
        assert "at least 3" in p.run(BBOX, tmp_path, date="2024-06", baseline_years=2).error

    def test_asks_for_red_and_nir_only(self, tmp_path, monkeypatch):
        rec = {}
        patch_downloads(monkeypatch, self.scenes(tmp_path), record=rec)
        VegetationAnomalyPipeline(object()).run(BBOX, tmp_path / "o", date="2024-06")
        assert all(tuple(c["bands"]) == ("red", "nir") for c in rec["calls"])


def mndwi_bands(water: np.ndarray):
    """green/swir1 whose normalised difference is +0.5 on water and -0.5 elsewhere; NaN where water is NaN."""
    green = np.where(water > 0.5, 0.3, 0.1).astype("float32")
    swir = np.where(water > 0.5, 0.1, 0.3).astype("float32")
    bad = np.isnan(water)
    green[bad] = np.nan
    swir[bad] = np.nan
    return np.stack([green, swir])


class TestWaterTimeseries:
    def field(self, cols_water: int, cloud_rows: int = 0) -> np.ndarray:
        a = np.zeros((N, N), "float32")
        a[:, :cols_water] = 1
        if cloud_rows:
            a[:cloud_rows] = np.nan
        return a

    def run(self, tmp_path, monkeypatch, scenes, **kw):
        src = {d: tif(tmp_path, f"w{d}", mndwi_bands(f)) for d, f in scenes.items()}
        patch_downloads(monkeypatch, src)
        return WaterExtentTimeseriesPipeline(object()).run(BBOX, tmp_path / "o", dates=list(scenes), **kw)

    def test_areas_and_frequency(self, tmp_path, monkeypatch):
        r = self.run(tmp_path, monkeypatch, {"2024-01": self.field(10), "2024-02": self.field(20), "2024-03": self.field(10), "2024-04": self.field(0)})
        assert r.success, r.error
        with rasterio.open(tmp_path / "o" / "water_frequency.tif") as s:
            f = s.read(1)
        assert f[0, 5] == pytest.approx(0.75) and f[0, 15] == pytest.approx(0.25) and f[0, 30] == 0.0
        series = {x["date"]: x for x in r.stats["series"]}
        assert series["2024-02"]["water_fraction_of_valid"] == pytest.approx(0.5)
        assert series["2024-04"]["water_area_ha"] == 0.0
        total = r.stats["ever_water_ha"]
        assert r.stats["always_water_ha"] == 0.0                              # April had no water at all
        assert total == pytest.approx(r.stats["max_water_area_ha"], rel=1e-6)
        assert r.stats["variable_water_ha"] == pytest.approx(total)
        rows = list(csv.DictReader((tmp_path / "o" / "water_timeseries.csv").open()))
        assert [x["date"] for x in rows] == ["2024-01", "2024-02", "2024-03", "2024-04"]

    def test_always_water(self, tmp_path, monkeypatch):
        r = self.run(tmp_path, monkeypatch, {"2024-01": self.field(10), "2024-02": self.field(20)})
        assert r.stats["always_water_ha"] == pytest.approx(r.stats["min_water_area_ha"])
        assert r.stats["variable_water_ha"] == pytest.approx(r.stats["max_water_area_ha"] - r.stats["min_water_area_ha"])

    def test_cloud_is_not_dry(self, tmp_path, monkeypatch):
        # Feb: clouds over the top 4 rows. Observed pixels still show water; the cloudy area must not count as dry.
        r = self.run(tmp_path, monkeypatch, {"2024-01": self.field(10), "2024-02": self.field(10, cloud_rows=4), "2024-03": self.field(10)})
        series = {x["date"]: x for x in r.stats["series"]}
        assert series["2024-02"]["valid_fraction"] == pytest.approx(0.9)
        assert series["2024-02"]["water_fraction_of_valid"] == pytest.approx(0.25)
        assert series["2024-02"]["water_area_ha"] < series["2024-01"]["water_area_ha"]      # less ground seen, less water seen
        with rasterio.open(tmp_path / "o" / "water_frequency.tif") as s:
            f = s.read(1)
        assert f[0, 5] == 1.0                                                  # 2 of 2 observations are water; the cloudy date is not counted as dry

    def test_mostly_cloudy_date_is_listed_but_left_out(self, tmp_path, monkeypatch):
        r = self.run(tmp_path, monkeypatch, {"2024-01": self.field(10), "2024-02": self.field(10, cloud_rows=30), "2024-03": self.field(10)})
        series = {x["date"]: x for x in r.stats["series"]}
        assert series["2024-02"]["usable"] is False and "25%" in series["2024-02"]["note"]
        assert r.stats["n_dates_usable"] == 2

    def test_a_date_without_imagery_is_reported(self, tmp_path, monkeypatch):
        src = {d: tif(tmp_path, f"w{d}", mndwi_bands(self.field(10))) for d in ("2024-01", "2024-03")}
        patch_downloads(monkeypatch, src)
        r = WaterExtentTimeseriesPipeline(object()).run(BBOX, tmp_path / "o", dates=["2024-01", "2024-02", "2024-03"])
        assert r.success
        assert [x["usable"] for x in r.stats["series"]] == [True, False, True]

    def test_fails_with_fewer_than_two_usable_dates(self, tmp_path, monkeypatch):
        patch_downloads(monkeypatch, {})
        r = WaterExtentTimeseriesPipeline(object()).run(BBOX, tmp_path / "o", dates=["2024-01", "2024-02"])
        assert not r.success and "two dates" in r.error

    def test_needs_dates(self, tmp_path):
        assert "dates" in WaterExtentTimeseriesPipeline(object()).run(BBOX, tmp_path, dates=["2024-01"]).error
