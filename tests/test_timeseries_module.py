"""GeoTimeSeries on rasters whose index is known by construction."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.advanced.timeseries import GeoTimeSeries  # noqa: E402


def stack(path, red, nir, *, blue=None, dtype="uint16", nodata=None, scale=10000.0, size=8):
    """4-band file (blue, green, red, nir) with constant red/nir (reflectance) per pixel."""
    arr = np.zeros((4, size, size), dtype="float64")
    arr[0] = 0.05 if blue is None else blue
    arr[1] = 0.1
    arr[2] = red
    arr[3] = nir
    if np.issubdtype(np.dtype(dtype), np.integer):
        arr = np.round(arr * scale)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=4, dtype=dtype, crs="EPSG:4326",
                       transform=from_origin(10.0, 50.0, 0.001, 0.001), nodata=nodata) as dst:
        dst.write(arr.astype(dtype))
    return str(path)


def ndvi(r, n):
    return (n - r) / (n + r)


class TestIndexSeries:
    def test_ndvi_value_and_scaling_for_ints_and_floats(self, tmp_path):
        a = stack(tmp_path / "a.tif", 0.1, 0.5)
        b = stack(tmp_path / "b.tif", 0.1, 0.5, dtype="float32")
        s = GeoTimeSeries().compute_index_series([a, b], "ndvi", ["2024-01", "2024-02"])
        assert s["mean"] == [pytest.approx(ndvi(0.1, 0.5), abs=1e-3)] * 2
        assert s["dates"] == ["2024-01", "2024-02"] and s["errors"] == {} and s["valid_fraction"] == [1.0, 1.0]

    def test_evi_uses_the_blue_band(self, tmp_path):
        p1 = stack(tmp_path / "a.tif", 0.1, 0.5, blue=0.02)
        p2 = stack(tmp_path / "b.tif", 0.1, 0.5, blue=0.10)
        s = GeoTimeSeries().compute_index_series([p1, p2], "evi")
        exp = lambda b: 2.5 * (0.5 - 0.1) / (0.5 + 0.6 - 7.5 * b + 1)  # noqa: E731
        assert s["mean"][0] == pytest.approx(exp(0.02), abs=2e-3)
        assert s["mean"][1] == pytest.approx(exp(0.10), abs=2e-3)
        assert s["mean"][0] != s["mean"][1]

    def test_nodata_pixels_are_ignored(self, tmp_path):
        p = tmp_path / "n.tif"
        arr = np.zeros((4, 8, 8), "uint16")
        arr[2], arr[3] = 1000, 5000
        arr[:, :4, :] = 0                        # empty half, nodata=0 declared
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8, count=4, dtype="uint16", crs="EPSG:4326",
                           transform=from_origin(0, 1, 0.1, 0.1), nodata=0) as d:
            d.write(arr)
        s = GeoTimeSeries().compute_index_series([str(p)], "ndvi")
        assert s["mean"][0] == pytest.approx(ndvi(0.1, 0.5), abs=1e-3)
        assert s["valid_fraction"][0] == 0.5

    def test_all_zero_pixels_without_a_nodata_flag_are_still_ignored(self, tmp_path):
        p = tmp_path / "z.tif"
        arr = np.zeros((4, 8, 8), "uint16")
        arr[2, 4:], arr[3, 4:] = 1000, 5000
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8, count=4, dtype="uint16", crs="EPSG:4326",
                           transform=from_origin(0, 1, 0.1, 0.1)) as d:
            d.write(arr)
        s = GeoTimeSeries().compute_index_series([str(p)], "ndvi")
        assert s["mean"][0] == pytest.approx(ndvi(0.1, 0.5), abs=1e-3)

    def test_mask_paths_exclude_cloud(self, tmp_path):
        p = tmp_path / "c.tif"
        arr = np.zeros((4, 8, 8), "float32")
        arr[2], arr[3] = 0.1, 0.5
        arr[3, :4] = 0.12                       # cloudy half: bright, low ndvi
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8, count=4, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(0, 1, 0.1, 0.1)) as d:
            d.write(arr)
        m = tmp_path / "m.tif"
        mk = np.zeros((1, 8, 8), "uint8")
        mk[:, :4] = 1
        with rasterio.open(m, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint8", crs="EPSG:4326",
                           transform=from_origin(0, 1, 0.1, 0.1)) as d:
            d.write(mk)
        ts = GeoTimeSeries()
        assert ts.compute_index_series([str(p)], "ndvi")["mean"][0] < 0.5
        s = ts.compute_index_series([str(p)], "ndvi", mask_paths=[str(m)])
        assert s["mean"][0] == pytest.approx(ndvi(0.1, 0.5), abs=1e-3) and s["valid_fraction"][0] == 0.5

    def test_offset_for_sentinel2_baseline_04(self, tmp_path):
        p = tmp_path / "o.tif"
        arr = np.zeros((4, 8, 8), "uint16")
        arr[2], arr[3] = 1000 + 1000, 1000 + 5000      # +1000 processing offset
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8, count=4, dtype="uint16", crs="EPSG:4326",
                           transform=from_origin(0, 1, 0.1, 0.1)) as d:
            d.write(arr)
        s = GeoTimeSeries().compute_index_series([str(p)], "savi", reflectance_offset=-1000)
        assert s["mean"][0] == pytest.approx(1.5 * 0.4 / (0.6 + 0.5), abs=1e-3)

    def test_bad_file_recorded_not_fatal(self, tmp_path):
        good = stack(tmp_path / "g.tif", 0.1, 0.5)
        s = GeoTimeSeries().compute_index_series([good, str(tmp_path / "nope.tif")], "ndvi")
        assert s["mean"][1] is None and len(s["errors"]) == 1 and "nope.tif" in next(iter(s["errors"]))

    def test_too_few_bands_recorded(self, tmp_path):
        p = tmp_path / "two.tif"
        with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=2, dtype="uint16", crs="EPSG:4326",
                           transform=from_origin(0, 1, 0.1, 0.1)) as d:
            d.write(np.ones((2, 4, 4), "uint16"))
        s = GeoTimeSeries().compute_index_series([str(p)], "ndvi")
        assert s["mean"] == [None] and s["errors"]

    def test_validation(self, tmp_path):
        ts = GeoTimeSeries()
        with pytest.raises(ValueError, match="unknown index"):
            ts.compute_index_series([], "nope")
        with pytest.raises(ValueError, match="date_strings has 1"):
            ts.compute_index_series([], "ndvi", ["a"])
        with pytest.raises(ValueError, match="unknown sensor"):
            GeoTimeSeries("sentinel9")
        with pytest.raises(ValueError, match="needs band"):
            GeoTimeSeries("planet").compute_index_series([], "ndbi")        # planet preset has no swir
        with pytest.warns(DeprecationWarning):
            ts.compute_index_series([], "ndvi", cloud_mask=True)

    def test_band_map_override(self, tmp_path):
        p = tmp_path / "bm.tif"
        arr = np.zeros((2, 4, 4), "float32")
        arr[0], arr[1] = 0.5, 0.1                       # nir first, red second
        with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=2, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(0, 1, 0.1, 0.1)) as d:
            d.write(arr)
        s = GeoTimeSeries(band_map={"nir": 1, "red": 2}).compute_index_series([str(p)], "ndvi")
        assert s["mean"][0] == pytest.approx(ndvi(0.1, 0.5), abs=1e-3)


class TestAnomalies:
    ts = GeoTimeSeries()

    def test_dates_stay_aligned_when_a_scene_failed(self):
        """A None used to shift every later date by one."""
        mean = [0.6, None, 0.61, 0.62, 0.6, 0.61, 0.62, 0.6, 0.61, 0.1]
        dates = [f"d{i}" for i in range(10)]
        a = self.ts.detect_anomalies({"mean": mean, "dates": dates}, threshold=2.0)
        assert [x["date"] for x in a] == ["d9"] and a[0]["type"] == "low"

    def test_short_series_zscore_cannot_flag_but_mad_can(self):
        mean = [0.6, 0.61, 0.6, 0.62, 0.1, 0.6]
        s = {"mean": mean, "dates": list(range(6))}
        assert self.ts.detect_anomalies(s) == []                       # z of the outlier is only 2.04 < 2.5
        assert [x["date"] for x in self.ts.detect_anomalies(s, method="mad")] == [4]

    def test_period_stops_seasonality_being_flagged(self):
        cycle = [0.2, 0.3, 0.5, 0.7, 0.8, 0.7, 0.5, 0.3]
        mean = cycle * 4
        s = {"mean": list(mean), "dates": list(range(32))}
        assert self.ts.detect_anomalies(s, method="mad", threshold=2.0, period=8) == []
        mean[20] = 0.1                                                  # the same cycle position, much lower
        s = {"mean": mean, "dates": list(range(32))}
        out = self.ts.detect_anomalies(s, method="mad", threshold=3.5, period=8)
        assert [x["date"] for x in out] == [20] and out[0]["expected"] == pytest.approx(0.8, abs=0.3)

    def test_errors(self):
        with pytest.raises(ValueError, match="at least 5"):
            self.ts.detect_anomalies({"mean": [0.1, 0.2, None], "dates": [1, 2, 3]})
        with pytest.raises(ValueError, match="unknown method"):
            self.ts.detect_anomalies({"mean": [0.1] * 6}, method="iqr")
        with pytest.raises(ValueError, match="two full cycles"):
            self.ts.detect_anomalies({"mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, period=4)

    def test_constant_series_has_no_anomalies(self):
        assert self.ts.detect_anomalies({"mean": [0.5] * 8, "dates": list(range(8))}, method="mad") == []


class TestTrend:
    ts = GeoTimeSeries()

    def test_slope_per_step_and_per_year(self):
        s = {"mean": [0.5 + 0.01 * i for i in range(12)], "dates": [f"2020-{m:02d}" for m in range(1, 13)]}
        t = self.ts.compute_trend(s)
        assert t["slope"] == pytest.approx(0.01, abs=1e-6) and t["direction"] == "increasing"
        assert t["significant"] is True and t["r_squared"] == pytest.approx(1.0)
        assert t["slope_per_year"] == pytest.approx(0.12, rel=0.05)

    def test_missing_dates_leave_gaps_not_squeeze(self):
        m = [0.5 + 0.01 * i for i in range(10)]
        m[3] = m[4] = None
        assert self.ts.compute_trend({"mean": m, "dates": list(range(10))})["slope"] == pytest.approx(0.01, abs=1e-6)

    def test_flat_noise_not_significant_and_constant_r2_zero(self):
        rng = np.random.default_rng(0)
        t = self.ts.compute_trend({"mean": list(0.5 + rng.normal(0, 0.05, 30)), "dates": list(range(30))})
        assert t["significant"] is False
        c = self.ts.compute_trend({"mean": [0.5] * 6, "dates": list(range(6))})
        assert c["r_squared"] == 0.0 and c["direction"] == "stable" and c["p_value"] is None

    def test_insufficient(self):
        t = self.ts.compute_trend({"mean": [0.1, None, 0.3], "dates": [1, 2, 3]})
        assert t["trend"] == "insufficient_data" and t["direction"] == "insufficient_data"


class TestPointAndPlot:
    def test_extract_point(self, tmp_path):
        a = stack(tmp_path / "a.tif", 0.1, 0.5)
        b = stack(tmp_path / "b.tif", 0.2, 0.4)
        ts = GeoTimeSeries()
        s = ts.extract_point([a, b], lon=10.004, lat=49.996, index="ndvi", date_strings=["x", "y"])
        assert s["mean"] == [pytest.approx(ndvi(0.1, 0.5), abs=1e-3), pytest.approx(ndvi(0.2, 0.4), abs=1e-3)]
        out = ts.extract_point([a], lon=11.0, lat=49.996)
        assert out["mean"] == [None]                                    # outside the image

    def test_plot_saves_and_closes(self, tmp_path):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        s = {"index": "ndvi", "dates": ["a", "b", "c"], "mean": [0.4, None, 0.6], "p10": [0.3, None, 0.5],
             "p90": [0.5, None, 0.7]}
        out = tmp_path / "p.png"
        GeoTimeSeries().plot(s, save_path=str(out))
        assert out.stat().st_size > 1000
        with pytest.raises(ValueError, match="nothing to plot"):
            GeoTimeSeries().plot({"mean": [None], "dates": ["a"]})


class TestCli:
    def test_analyze_and_anomaly(self, tmp_path):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        vals = [0.5, 0.52, 0.5, 0.51, 0.5, 0.52, 0.5, 0.51, 0.5, 0.1]
        files = [stack(tmp_path / f"s{i}.tif", 0.1, 0.1 * (1 + v) / (1 - v)) for i, v in enumerate(vals)]
        r = CliRunner().invoke(cli, ["timeseries", "analyze", *files, "--dates", ",".join(f"m{i}" for i in range(10)),
                                     "-o", str(tmp_path / "o.json")])
        assert r.exit_code == 0, r.output
        assert "m9" in r.output and "Trend:" in r.output
        assert len(json.loads((tmp_path / "o.json").read_text())["mean"]) == 10
        r = CliRunner().invoke(cli, ["timeseries", "anomaly", *files, "-m", "mad"])
        assert r.exit_code == 0 and "low" in r.output

    def test_too_few_images_is_a_clean_error(self, tmp_path):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        f = stack(tmp_path / "a.tif", 0.1, 0.5)
        r = CliRunner().invoke(cli, ["timeseries", "anomaly", f, f])
        assert r.exit_code != 0 and "at least 5" in r.output and "Traceback" not in r.output

    def test_bad_dates_count(self, tmp_path):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        f = stack(tmp_path / "a.tif", 0.1, 0.5)
        r = CliRunner().invoke(cli, ["timeseries", "analyze", f, "--dates", "a,b"])
        assert r.exit_code != 0 and "date_strings has 2" in r.output


class TestSeasonalTrend:
    ts = GeoTimeSeries()

    def series(self, slope=0.002, noise=0.0, years=3, seed=0):
        rng = np.random.default_rng(seed)
        n = 12 * years
        mean = [0.5 + 0.25 * np.sin(2 * np.pi * (i % 12 - 3) / 12) + slope * i + rng.normal(0, noise) for i in range(n)]
        return {"mean": [round(float(v), 5) for v in mean], "dates": [f"{2021 + i // 12}-{i % 12 + 1:02d}" for i in range(n)]}

    def test_period_recovers_the_true_slope_that_the_plain_fit_misses(self):
        s = self.series(slope=0.002, noise=0.01)
        plain = self.ts.compute_trend(s)
        seasonal = self.ts.compute_trend(s, period=12)
        assert plain["significant"] is False                   # swamped by the seasonal swing
        assert seasonal["slope"] == pytest.approx(0.002, abs=0.0005)
        assert seasonal["significant"] is True and seasonal["period"] == 12

    def test_flat_seasonal_series_not_significant(self):
        s = self.series(slope=0.0, noise=0.02, seed=3)
        assert self.ts.compute_trend(s, period=12)["significant"] is False

    def test_period_validation(self):
        with pytest.raises(ValueError, match="two full cycles"):
            self.ts.compute_trend({"mean": [0.1, 0.2, 0.3, 0.4, 0.5], "dates": list(range(5))}, period=4)
        s = self.series()
        s["mean"][0] = s["mean"][12] = None                    # cycle position 0 left with one observation
        with pytest.raises(ValueError, match="at least 2 observations"):
            self.ts.compute_trend(s, period=12)
