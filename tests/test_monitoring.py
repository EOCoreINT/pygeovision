"""Tests for drift detection and monitoring (Phase 2+)."""
import json
from pathlib import Path


class TestDistributionDrift:
    def test_compute_psi_identical(self):
        import numpy as np

        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        data = np.random.rand(1000)
        psi = d.compute_psi(data, data)
        assert abs(psi) < 0.05  # identical distributions → PSI ≈ 0

    def test_compute_psi_different(self):
        import numpy as np

        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000) + 5   # shifted distribution
        psi = d.compute_psi(data1, data2)
        assert psi > 0.2  # major drift

    def test_compute_kl_identical(self):
        import numpy as np

        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        p = np.ones(10) / 10
        kl = d.compute_kl_divergence(p, p)
        assert abs(kl) < 1e-6

    def test_psi_thresholds(self):
        from pygeovision.monitoring.drift import DistributionDrift
        assert DistributionDrift.PSI_THRESHOLDS["low"] == 0.1
        assert DistributionDrift.PSI_THRESHOLDS["medium"] == 0.2

    def test_recommendation_no_drift(self):
        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        rec = d._recommendation(0.05)
        assert "No action" in rec

    def test_recommendation_minor(self):
        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        rec = d._recommendation(0.15)
        assert "Minor" in rec

    def test_recommendation_major(self):
        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        rec = d._recommendation(0.30)
        assert "Major" in rec or "Retrain" in rec


class TestPerformanceDrift:
    def test_init_creates_storage(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        storage = str(tmp_path / "perf.jsonl")
        PerformanceDrift(storage_path=storage)
        assert Path(storage).parent.exists()

    def test_log_appends(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        storage = str(tmp_path / "perf.jsonl")
        pd = PerformanceDrift(storage_path=storage)
        pd.log({"mean_iou": 0.85, "accuracy": 0.92})
        pd.log({"mean_iou": 0.83, "accuracy": 0.91})
        with open(storage) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2

    def test_set_baseline_stores(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        pd.set_baseline({"mean_iou": 0.85, "accuracy": 0.92})
        assert pd._baseline["mean_iou"] == 0.85

    def test_detect_no_baseline(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        result = pd.detect()
        assert "error" in result

    def test_detect_no_drift(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        pd.set_baseline({"mean_iou": 0.85})
        for _ in range(5):
            pd.log({"mean_iou": 0.85})  # stable
        result = pd.detect()
        assert result.get("drift_detected") is False

    def test_detect_drift_found(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"),
                               alert_thresholds={"mean_iou": 0.05})
        pd.set_baseline({"mean_iou": 0.85})
        for _ in range(5):
            pd.log({"mean_iou": 0.70})  # -15pp drop >> 5pp threshold
        result = pd.detect()
        assert result.get("drift_detected") is True
        assert len(result.get("alerts", [])) > 0

    def test_performance_recommendation_no_alerts(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        rec = pd._performance_recommendation([])
        assert "stable" in rec.lower() or "No action" in rec


class TestDriftDetector:
    def test_init(self, tmp_path):
        from pygeovision.monitoring.drift import DriftDetector
        d = DriftDetector(storage_dir=str(tmp_path))
        assert d.data_drift is not None
        assert d.perf_drift is not None

    def test_fit_no_images(self, tmp_path):
        from pygeovision.monitoring.drift import DriftDetector
        d = DriftDetector(storage_dir=str(tmp_path))
        stats = d.fit([])   # empty list
        # Should not crash
        assert stats is d or stats is None or isinstance(stats, dict)

    def test_save_report(self, tmp_path):
        from pygeovision.monitoring.drift import DriftDetector
        d = DriftDetector(storage_dir=str(tmp_path))
        report = {"drift_detected": False, "data_drift": {}, "performance_drift": {}}
        path = d.save_report(report, str(tmp_path / "report.json"))
        assert Path(path).exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["drift_detected"] is False


class TestModelPerformanceTracker:
    def test_record_and_summary(self, tmp_path):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        tracker = ModelPerformanceTracker("test_model", str(tmp_path / "tracker.json"))
        tracker.record({"mean_iou": 0.82, "accuracy": 0.91}, split="val", n_samples=500)
        tracker.record({"mean_iou": 0.84, "accuracy": 0.93}, split="val", n_samples=600)
        summary = tracker.summary()
        assert summary["n_records"] == 2
        assert summary["model"] == "test_model"
        assert "mean_iou" in summary["latest_metrics"]

    def test_trend_improving(self, tmp_path):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        tracker = ModelPerformanceTracker("m", str(tmp_path / "t.json"))
        for v in [0.70, 0.75, 0.80, 0.82]:
            tracker.record({"mean_iou": v})
        trend = tracker.trend("mean_iou")
        assert trend["direction"] == "improving"

    def test_trend_declining(self, tmp_path):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        tracker = ModelPerformanceTracker("m", str(tmp_path / "t.json"))
        for v in [0.85, 0.82, 0.78, 0.75]:
            tracker.record({"mean_iou": v})
        trend = tracker.trend("mean_iou")
        assert trend["direction"] == "declining"


class TestAlertManager:
    def test_init_default(self):
        from pygeovision.monitoring.alerts import AlertManager
        am = AlertManager()
        assert "log" in am.channels

    def test_trigger_logs(self):
        from pygeovision.monitoring.alerts import AlertManager
        am = AlertManager(channels=["log"])
        am.trigger("warning", "Test alert")
        assert len(am.history) == 1
        assert am.history[0]["message"] == "Test alert"

    def test_register_custom_handler(self):
        from pygeovision.monitoring.alerts import AlertManager
        received = []
        am = AlertManager(channels=["custom"])
        am.register_handler("custom", lambda entry: received.append(entry))
        am.trigger("info", "Custom test")
        assert len(received) == 1
        assert received[0]["severity"] == "info"

    def test_check_drift_report_no_drift(self):
        from pygeovision.monitoring.alerts import AlertManager
        am = AlertManager()
        report = {"drift_detected": False, "data_drift": {"drift_level": "none"},
                   "performance_drift": {"drift_detected": False, "alerts": []}}
        am.check_drift_report(report)
        # No alerts should be triggered
        assert len(am.history) == 0

    def test_check_drift_report_major_drift(self):
        from pygeovision.monitoring.alerts import AlertManager
        received = []
        am = AlertManager(channels=["custom"])
        am.register_handler("custom", lambda e: received.append(e))
        report = {
            "drift_detected": True,
            "data_drift": {"drift_level": "major"},
            "performance_drift": {"drift_detected": True, "alerts": ["mean_iou dropped"]},
        }
        am.check_drift_report(report)
        severities = [r["severity"] for r in received]
        assert "critical" in severities or "warning" in severities


# --------------------------------------------------------------------------------------
# Regression tests added with the monitoring documentation pass
# --------------------------------------------------------------------------------------
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")


def _tif(path, arr, nodata=None):
    from rasterio.transform import from_origin
    arr = np.asarray(arr, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs="EPSG:32633", transform=from_origin(0, 100, 1, 1), nodata=nodata) as dst:
        dst.write(arr)


def _scene(seed, scale=1.0, offset=0.0, bands=3, size=64):
    rng = np.random.default_rng(seed)
    return rng.uniform(20, 120, (bands, size, size)) * scale + offset


class TestDataDriftDetectsShifts:
    def test_uniform_brightness_shift_is_seen(self, tmp_path):
        """Histograms used to be built on each image set's own range, so a shift of every pixel by a
        constant gave PSI ~ 0 ('no drift')."""
        from pygeovision.monitoring.drift import DistributionDrift
        ref = [tmp_path / f"r{i}.tif" for i in range(3)]
        cur = [tmp_path / f"c{i}.tif" for i in range(3)]
        for i, p in enumerate(ref):
            _tif(p, _scene(i))
        for i, p in enumerate(cur):
            _tif(p, _scene(10 + i, offset=60))
        d = DistributionDrift()
        d.fit_reference([str(p) for p in ref])
        rep = d.detect([str(p) for p in cur])
        assert rep["drift_level"] == "major" and rep["overall_psi"] > 0.5

    def test_same_distribution_is_stable(self, tmp_path):
        from pygeovision.monitoring.drift import DistributionDrift
        ref = [str(tmp_path / f"r{i}.tif") for i in range(3)]
        cur = [str(tmp_path / f"c{i}.tif") for i in range(2)]
        for i, p in enumerate(ref):
            _tif(p, _scene(i))
        for i, p in enumerate(cur):
            _tif(p, _scene(20 + i))
        d = DistributionDrift()
        d.fit_reference(ref)
        rep = d.detect(cur)
        assert rep["drift_level"] == "none"
        assert rep["n_reference"] == 3 and rep["n_current"] == 2
        assert set(rep["per_band"]) == {"band_1", "band_2", "band_3"}
        assert rep["worst_band"] in rep["per_band"] and rep["max_psi"] >= rep["overall_psi"]

    def test_reference_keeps_bin_edges(self, tmp_path):
        from pygeovision.monitoring.drift import DistributionDrift
        p = str(tmp_path / "r.tif")
        _tif(p, _scene(0))
        d = DistributionDrift(n_bins=8)
        stats = d.fit_reference([p])
        assert len(stats["band_1"]["bin_edges"]) == 9 and len(stats["band_1"]["histogram"]) == 8

    def test_nodata_pixels_are_ignored(self, tmp_path):
        from pygeovision.monitoring.drift import DistributionDrift
        arr = _scene(0)
        arr[:, :, :32] = 0
        tagged, untagged = str(tmp_path / "t.tif"), str(tmp_path / "u.tif")
        _tif(tagged, arr, nodata=0)
        _tif(untagged, arr)
        d = DistributionDrift()
        s = d.fit_reference([tagged])
        assert s["band_1"]["mean"] > 40                      # zeros left out
        s = d.fit_reference([untagged])
        assert s["band_1"]["mean"] < 40                      # zeros counted

    def test_unreadable_files_are_skipped_with_count(self, tmp_path):
        from pygeovision.monitoring.drift import DistributionDrift
        good, bad = str(tmp_path / "g.tif"), str(tmp_path / "bad.tif")
        _tif(good, _scene(0))
        (tmp_path / "bad.tif").write_text("not a tiff")
        d = DistributionDrift()
        d.fit_reference([good, bad])
        assert d.n_reference_images == 1
        rep = d.detect([good, bad])
        assert rep["n_current"] == 1

    def test_nothing_comparable_is_an_error_not_no_drift(self, tmp_path):
        from pygeovision.monitoring.drift import DistributionDrift, DriftDetector
        good = str(tmp_path / "g.tif")
        _tif(good, _scene(0))
        d = DistributionDrift()
        d.fit_reference([good])
        assert "error" in d.detect([str(tmp_path / "missing.tif")])
        empty = DistributionDrift()
        empty.fit_reference([])
        assert "error" in empty.detect([good])
        det = DriftDetector(storage_dir=str(tmp_path / "m"))
        det.fit([])
        rep = det.check([good])
        assert rep["incomplete"] is True and rep["drift_detected"] is False and rep["errors"]

    def test_psi_counts_values_outside_reference_range(self):
        from pygeovision.monitoring.drift import DistributionDrift
        a = np.random.default_rng(0).normal(0, 1, 2000)
        assert DistributionDrift().compute_psi(a, a + 6) > 1.0


class TestPerformanceWindow:
    def test_recent_n_reacts_to_a_sudden_drop(self, tmp_path):
        from pygeovision.monitoring.drift import DriftDetector
        good = str(tmp_path / "g.tif")
        _tif(good, _scene(0))
        d = DriftDetector(storage_dir=str(tmp_path / "m"))
        d.fit([good], reference_metrics={"mean_iou": 0.80})
        for v in (0.80, 0.80, 0.80, 0.80, 0.60):
            rep_window = d.check([good], {"mean_iou": v})
        assert rep_window["performance_drift"]["drift_detected"] is False     # averaged over all 5
        rep_last = d.check([good], recent_n=1)
        assert rep_last["performance_drift"]["drift_detected"] is True


class TestAlertChannels:
    def test_channel_without_handler_is_reported(self, caplog):
        import logging

        from pygeovision.monitoring.alerts import AlertManager
        am = AlertManager(channels=["log", "email"])
        with caplog.at_level(logging.WARNING):
            am.trigger("warning", "x")
        assert "email" in caplog.text and "no handler" in caplog.text


class TestAIMonitoringHardening:
    def test_fit_reference_accepts_lists(self):
        from pygeovision.ai.monitoring import DriftDetector
        d = DriftDetector()
        d.fit_reference([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        assert d.check([[0.5, 1.5], [1.5, 2.5]]).drift_score >= 0

    def test_column_mismatch_and_bad_method(self):
        pytest.importorskip("scipy")
        from pygeovision.ai.monitoring import DriftDetector
        d = DriftDetector()
        d.fit_reference(np.random.randn(50, 3))
        with pytest.raises(ValueError, match="columns"):
            d.check(np.random.randn(10, 2))
        e = DriftDetector(method="nope")
        e.fit_reference(np.random.randn(50, 3))
        with pytest.raises(ValueError, match="unknown method"):
            e.check(np.random.randn(10, 3))

    def test_corrupt_history_is_kept_aside(self, tmp_path):
        from pygeovision.ai.monitoring import PerformanceTracker
        p = tmp_path / "h.json"
        p.write_text("{broken")
        t = PerformanceTracker(storage_path=p)
        t.log({"miou": 0.5})
        assert (tmp_path / "h.corrupt.json").read_text() == "{broken"
        assert len(t.get_trend("miou")) == 1


class TestMonitorCLI:
    def test_drift_command_reports_incomplete(self, tmp_path):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        res = CliRunner().invoke(cli, ["monitor", "drift", str(a), str(b), "-o", str(tmp_path / "r.json")])
        assert res.exit_code == 1 and "could not be assessed" in res.output

    def test_drift_command_ok(self, tmp_path):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        _tif(a / "1.tif", _scene(1)); _tif(b / "1.tif", _scene(2, offset=80))
        res = CliRunner().invoke(cli, ["monitor", "drift", str(a), str(b), "-o", str(tmp_path / "r.json")])
        assert res.exit_code == 0 and "major" in res.output
