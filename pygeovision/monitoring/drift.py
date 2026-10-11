"""
Model drift detection for production geospatial AI systems (G7).

Monitors:
    - Data distribution shift (input imagery statistics)
    - Prediction distribution shift (output class probabilities)
    - Model performance degradation (mIoU/mAP over time)
    - Spatial distribution shift (which regions are being processed)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """UTC timestamp as an ISO string without a zone suffix (same format as the old ``utcnow``)."""
    from datetime import timezone
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _read_band_samples(path: str, per_band: int, rng: Any) -> dict[str, Any]:
    """Random sample of valid pixel values per band of one GeoTIFF.

    NaN, infinite and nodata pixels are skipped. Returns ``{"band_1": array, ...}``.
    """
    import numpy as np
    import rasterio

    out = {}
    with rasterio.open(path) as src:
        nodata = src.nodata
        data = src.read().astype(np.float32)
    for b in range(data.shape[0]):
        flat = data[b].ravel()
        ok = np.isfinite(flat)
        if nodata is not None and np.isfinite(nodata):
            ok &= flat != np.float32(nodata)
        vals = flat[ok]
        if vals.size > per_band:
            vals = vals[rng.choice(vals.size, per_band, replace=False)]
        out[f"band_{b+1}"] = vals
    return out


class DistributionDrift:
    """Detect input data distribution shift using statistical tests.

    Uses the Population Stability Index (PSI) and KL divergence
    to compare reference (training) distribution against production data.

    The reference is stored as per-band histograms with **fixed bin edges**;
    current data is counted into those same bins (values outside the
    reference range land in the first/last bin), so a shift in brightness or
    contrast is visible, not only a change of shape.
    """

    PSI_THRESHOLDS = {"low": 0.1, "medium": 0.2, "high": 0.25}
    MAX_REFERENCE_IMAGES = 100
    MAX_CURRENT_IMAGES = 50
    SAMPLES_PER_BAND_PER_IMAGE = 10_000

    def __init__(self, reference_stats: dict | None = None,
                 n_bins: int = 10) -> None:
        self.reference_stats = reference_stats
        self.n_bins = n_bins
        self.n_reference_images = 0

    def compute_psi(self, expected: Any, actual: Any) -> float:
        """Population Stability Index (PSI) between two samples.

        Bins come from ``expected``; values of ``actual`` outside its range are
        counted in the outer bins.
        PSI < 0.1: No drift | 0.1-0.2: Minor | >0.2: Major drift.
        """
        import numpy as np
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)
        edges = np.histogram_bin_edges(expected, bins=self.n_bins)
        exp_counts, _ = np.histogram(expected, bins=edges)
        act_counts, _ = np.histogram(np.clip(actual, edges[0], edges[-1]), bins=edges)
        return self._psi_from_counts(exp_counts, act_counts)

    @staticmethod
    def _psi_from_counts(exp_counts: Any, act_counts: Any) -> float:
        import numpy as np
        exp = np.asarray(exp_counts, dtype=float)
        act = np.asarray(act_counts, dtype=float)
        # Proportions per bin; empty bins are floored at 1e-4 (the usual PSI convention) so
        # that a bin empty on one side contributes a large but finite amount.
        exp = np.clip(exp / exp.sum(), 1e-4, None)
        act = np.clip(act / act.sum(), 1e-4, None)
        exp, act = exp / exp.sum(), act / act.sum()
        return float(np.sum((act - exp) * np.log(act / exp)))

    def compute_kl_divergence(self, p: Any, q: Any) -> float:
        """KL divergence — measures distribution distance."""
        import numpy as np
        p = np.clip(np.asarray(p, dtype=float), 1e-10, None)
        q = np.clip(np.asarray(q, dtype=float), 1e-10, None)
        p = p / p.sum(); q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def fit_reference(self, images: list[str]) -> dict[str, Any]:
        """Compute reference statistics from a list of GeoTIFFs.

        Up to ``MAX_REFERENCE_IMAGES`` images are used; from each, up to
        ``SAMPLES_PER_BAND_PER_IMAGE`` random valid pixels per band
        (reproducible: fixed seed). Unreadable files are skipped with a
        warning. Returns the statistics (empty if nothing could be read).
        """
        try:
            import numpy as np
            import rasterio  # noqa: F401
        except ImportError:
            return {}

        rng = np.random.default_rng(0)
        pooled: dict[str, list] = {}
        used = 0
        for path in list(images)[: self.MAX_REFERENCE_IMAGES]:
            try:
                samples = _read_band_samples(path, self.SAMPLES_PER_BAND_PER_IMAGE, rng)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fit_reference: skipping unreadable image %s (%s)", path, exc)
                continue
            used += 1
            for band, vals in samples.items():
                pooled.setdefault(band, []).append(vals)

        self.n_reference_images = used
        self.reference_stats = {}
        for band, chunks in pooled.items():
            values = np.concatenate(chunks)
            if values.size == 0:
                continue
            edges = np.histogram_bin_edges(values, bins=self.n_bins)
            self.reference_stats[band] = {
                "mean": float(values.mean()),
                "std":  float(values.std()),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
                "histogram": np.histogram(values, bins=edges)[0].tolist(),
                "bin_edges": edges.tolist(),
                "n_samples": int(values.size),
            }
        if not self.reference_stats:
            logger.warning("fit_reference: no readable reference images, so no reference statistics")
        return self.reference_stats

    def detect(self, current_images: list[str]) -> dict[str, Any]:
        """Detect drift in current production images vs reference.

        Compares up to ``MAX_CURRENT_IMAGES`` images per call. Returns
        ``{"error": ...}`` instead of a "no drift" answer whenever nothing
        could be compared.
        """
        if self.reference_stats is None:
            return {"error": "No reference stats. Call fit_reference() first."}
        if not self.reference_stats:
            return {"error": "Reference statistics are empty (no readable reference images)."}
        if any("bin_edges" not in v for v in self.reference_stats.values()):
            return {"error": "Reference statistics have no bin edges (made by an older version). "
                             "Call fit_reference() again."}

        try:
            import numpy as np
            import rasterio  # noqa: F401
        except ImportError:
            return {"error": "rasterio required"}

        rng = np.random.default_rng(1)
        current_values: dict[str, list] = {}
        n_read = 0
        for path in list(current_images)[: self.MAX_CURRENT_IMAGES]:
            try:
                samples = _read_band_samples(path, self.SAMPLES_PER_BAND_PER_IMAGE, rng)
            except Exception as exc:  # noqa: BLE001
                logger.warning("detect: skipping unreadable image %s (%s)", path, exc)
                continue
            n_read += 1
            for band, vals in samples.items():
                current_values.setdefault(band, []).append(vals)

        drift_report = {}
        for band, ref_stats in self.reference_stats.items():
            if band not in current_values:
                continue
            edges = np.asarray(ref_stats["bin_edges"], dtype=float)
            ref = np.asarray(ref_stats["histogram"], dtype=float)
            vals = np.concatenate(current_values[band])
            if vals.size == 0 or ref.sum() == 0:
                continue
            curr = np.histogram(np.clip(vals, edges[0], edges[-1]), bins=edges)[0].astype(float)
            psi = self._psi_from_counts(ref, curr)
            kl = self.compute_kl_divergence(curr, ref)
            level = "none" if psi < 0.1 else "minor" if psi < 0.2 else "major"
            drift_report[band] = {
                "psi": round(psi, 4), "kl": round(kl, 4), "level": level,
                "mean_reference": round(ref_stats["mean"], 4),
                "mean_current": round(float(vals.mean()), 4),
            }

        if not drift_report:
            return {"error": "Nothing could be compared: no readable current images, or "
                             "their bands do not match the reference."}

        overall_psi = sum(b["psi"] for b in drift_report.values()) / len(drift_report)
        worst = max(drift_report, key=lambda k: drift_report[k]["psi"])
        return {
            "overall_psi": round(overall_psi, 4),
            "drift_level": "none" if overall_psi < 0.1 else "minor" if overall_psi < 0.2 else "major",
            "per_band": drift_report,
            "max_psi": drift_report[worst]["psi"],
            "worst_band": worst,
            "n_reference": self.n_reference_images,
            "n_current": n_read,
            "recommendation": self._recommendation(overall_psi),
        }

    def _recommendation(self, psi: float) -> str:
        if psi < 0.1:
            return "No action needed. Distribution stable."
        elif psi < 0.2:
            return "Minor drift detected. Monitor closely and consider recalibrating."
        else:
            return "Major distribution shift! Retrain model on recent data immediately."


class PerformanceDrift:
    """Detect model performance degradation over time (G7).

    Tracks mIoU, F1, and accuracy over rolling windows and
    triggers alerts when performance drops below thresholds.
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_thresholds: dict[str, float] | None = None,
        storage_path: str = "./monitoring/performance_log.jsonl",
    ) -> None:
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            "mean_iou":  0.05,   # Alert if mIoU drops by 5pp
            "accuracy":  0.03,
            "mean_f1":   0.05,
        }
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._baseline: dict[str, float] | None = None
        self._history: list[dict] = []
        self._load_history()

    def _load_history(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                self._history = [json.loads(l) for l in f if l.strip()]

    def log(self, metrics: dict[str, float], timestamp: str | None = None) -> None:
        """Log a new metric measurement."""
        entry = {
            "timestamp": timestamp or _utcnow(),
            **metrics,
        }
        self._history.append(entry)
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def set_baseline(self, metrics: dict[str, float]) -> None:
        """Set the reference performance (from validation set)."""
        self._baseline = metrics
        logger.info("Baseline set: %s", metrics)

    def detect(self, recent_n: int | None = None) -> dict[str, Any]:
        """Detect performance drift vs baseline."""
        if self._baseline is None:
            return {"error": "No baseline. Call set_baseline() with validation metrics."}

        n = recent_n or self.window_size
        recent = self._history[-n:]
        if not recent:
            return {"error": "No performance history logged yet."}

        recent_avg = {}
        for metric in self.alert_thresholds:
            vals = [e[metric] for e in recent if metric in e]
            if vals:
                recent_avg[metric] = sum(vals) / len(vals)

        alerts = []
        drift_report = {}
        for metric, threshold in self.alert_thresholds.items():
            if metric not in recent_avg or metric not in self._baseline:
                continue
            drop = self._baseline[metric] - recent_avg[metric]
            drifted = drop > threshold
            drift_report[metric] = {
                "baseline": round(self._baseline[metric], 4),
                "recent":   round(recent_avg[metric], 4),
                "drop":     round(drop, 4),
                "threshold": threshold,
                "alert": drifted,
            }
            if drifted:
                alerts.append(f"{metric} dropped by {drop:.1%} (threshold: {threshold:.1%})")

        return {
            "n_measurements": len(recent),
            "drift_detected": len(alerts) > 0,
            "alerts": alerts,
            "metrics": drift_report,
            "recommendation": self._performance_recommendation(alerts),
        }

    def _performance_recommendation(self, alerts: list[str]) -> str:
        if not alerts:
            return "Model performance stable. No action required."
        n = len(alerts)
        if n == 1:
            return f"Minor performance drop in {alerts[0]}. Consider re-calibration."
        return f"{n} metrics degraded. Schedule fine-tuning on recent labelled data."

    def plot_history(self, metric: str = "mean_iou", save_path: str | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
            vals = [(e["timestamp"][:10], e[metric]) for e in self._history if metric in e]
            if not vals: return
            dates, scores = zip(*vals)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(scores, "b-", linewidth=1.5, label=metric)
            if self._baseline and metric in self._baseline:
                ax.axhline(self._baseline[metric], color="g", linestyle="--", label="Baseline")
                threshold = self.alert_thresholds.get(metric, 0.05)
                ax.axhline(self._baseline[metric] - threshold, color="r", linestyle="--",
                            label="Alert threshold")
            ax.set_xlabel("Measurement"); ax.set_ylabel(metric)
            ax.set_title(f"Model Performance History — {metric}")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=120)
            else: plt.show()
        except ImportError:
            logger.warning("plot_history() requires matplotlib: pip install matplotlib")


class DriftDetector:
    """Unified drift detector — monitors data + prediction + performance drift."""

    def __init__(self, model: Any | None = None,
                 storage_dir: str = "./monitoring/") -> None:
        self.model = model
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.data_drift   = DistributionDrift()
        self.perf_drift   = PerformanceDrift(storage_path=str(self.storage_dir / "perf.jsonl"))
        self._reference_images: list[str] = []

    def fit(self, reference_images: list[str],
             reference_metrics: dict[str, float] | None = None) -> DriftDetector:
        """Set the reference distribution and performance baseline."""
        self._reference_images = reference_images
        self.data_drift.fit_reference(reference_images)
        if reference_metrics:
            self.perf_drift.set_baseline(reference_metrics)
        logger.info("DriftDetector: reference set (%d images)", len(reference_images))
        return self

    def check(self, current_images: list[str],
               current_metrics: dict[str, float] | None = None,
               recent_n: int | None = None) -> dict[str, Any]:
        """Run full drift check and return combined report.

        ``recent_n`` is how many of the latest logged measurements are averaged for
        the performance comparison (default: the whole window, 100). A small value
        reacts faster to a sudden drop; ``1`` compares only the newest measurement.
        """
        data_report = self.data_drift.detect(current_images)
        if current_metrics:
            self.perf_drift.log(current_metrics)
        perf_report = self.perf_drift.detect(recent_n)

        any_drift = (
            data_report.get("drift_level") in ("minor", "major") or
            perf_report.get("drift_detected", False)
        )
        errors = [r["error"] for r in (data_report, perf_report) if "error" in r]
        return {
            "timestamp": _utcnow(),
            "drift_detected": any_drift,
            "data_drift": data_report,
            "performance_drift": perf_report,
            "action_required": data_report.get("drift_level") == "major" or perf_report.get("drift_detected"),
            # True when the data comparison could not be made: "drift_detected: False" then means
            # "unknown", not "no drift".
            "incomplete": "error" in data_report,
            "errors": errors,
        }

    def save_report(self, report: dict, path: str | None = None) -> str:
        p = path or str(self.storage_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        with open(p, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return p