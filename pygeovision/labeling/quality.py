"""
Label quality assessment -- cheap, automatic sanity checks for a label raster, plus a conservative cleaner.

These are *heuristics about the raster itself* (class mix, specks, coverage, shapes). They cannot tell you
whether a label is **right**: for that, compare against a surveyed reference or look at it. A grade of "A"
means "no obvious defects", not "accurate".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AVAILABLE_CHECKS = ("class_balance", "slivers", "coverage", "connectivity", "boundaries")
DEFAULT_CHECKS = ("class_balance", "slivers", "coverage", "connectivity")


def pixel_area_m2(src: Any) -> float:
    """Ground area of one pixel in square metres (geographic CRS -> converted at the raster's centre latitude)."""
    import math

    tf = src.transform
    crs = src.crs
    base = abs(tf.determinant)
    if crs is not None and getattr(crs, "is_geographic", False):
        lat = tf.f + tf.e * (src.height / 2.0)
        m_lat = 111_320.0
        m_lon = 111_320.0 * math.cos(math.radians(lat))
        return base * m_lat * m_lon
    if crs is not None and getattr(crs, "is_projected", False):
        try:
            f = float(crs.linear_units_factor[1])
        except Exception:  # noqa: BLE001
            f = 1.0
        return base * f * f
    return base


class LabelQualityAssessor:
    """Automatic quality checks and cleaning for geospatial label rasters.

    Checks (all on the raster only):

    * ``class_balance`` -- pixels per class and the max/min ratio. Imbalance is often *true* of the world
      (5% of a city is buildings); it matters for training, not as a defect, so it lowers the score gently.
    * ``slivers`` -- connected specks of fewer than ``max(4 px, 1 m2)``.
    * ``coverage`` -- fraction of pixels that are not ``ignore_index`` / nodata.
    * ``connectivity`` -- the share of foreground pixels that sit in components of at least 16 pixels
      (a low share means dust, not objects). Many large objects score well.
    * ``boundaries`` (opt-in) -- mean isoperimetric quotient of the foreground shapes (1 = circle,
      ~0.1 = very ragged).

    Args:
        num_classes: Number of classes (values ``0..num_classes-1``). ``None`` (default) infers the classes
            from the data. Values outside ``0..num_classes-1`` (other than ``ignore_index``) are reported as
            ``unexpected_values`` instead of being silently ignored.
        ignore_index: Value meaning "no label / no data" (default 255).

    Example::

        qa = LabelQualityAssessor()
        report = qa.assess("./labels/buildings.tif")
        if report["quality_score"] < 0.7:
            qa.clean("./labels/buildings.tif", "./labels/buildings_clean.tif")
    """

    def __init__(self, num_classes: int | None = None, ignore_index: int = 255) -> None:
        if num_classes is not None and num_classes < 1:
            raise ValueError("num_classes must be >= 1 or None")
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    # ------------------------------------------------------------------ helpers
    def _classes(self, label) -> tuple[list[int], list[int]]:
        """``(classes, unexpected)`` for a label array."""
        import numpy as np
        present = sorted(int(v) for v in np.unique(label) if int(v) != self.ignore_index)
        if self.num_classes is None:
            return present, []
        classes = list(range(self.num_classes))
        return classes, [v for v in present if v >= self.num_classes or v < 0]

    # ------------------------------------------------------------------ assess
    def assess(
        self,
        label_path: str | Path,
        image_path: str | None = None,
        checks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run quality checks on a label raster.

        Args:
            label_path: Label GeoTIFF (first band).
            image_path: Reserved. No check uses an image yet; passing one logs a warning.
            checks: Subset of ``AVAILABLE_CHECKS`` (default: all except ``boundaries``). An unknown name
                raises ``ValueError``.

        Returns:
            ``quality_score`` (0-1, mean of the check scores), ``quality_grade`` (A-F), ``checks``,
            ``classes``, ``unexpected_values``, ``pixel_area_m2``, ``recommendations``.
            An unreadable file gives ``{"error": ..., "quality_score": 0.0}``.
        """
        checks = list(checks) if checks else list(DEFAULT_CHECKS)
        unknown = [c for c in checks if c not in AVAILABLE_CHECKS]
        if unknown:
            raise ValueError(f"Unknown check(s) {unknown}. Available: {list(AVAILABLE_CHECKS)}")
        if image_path:
            logger.warning("LabelQualityAssessor.assess: image_path is not used by any check yet; ignoring it.")
        label_path = Path(label_path)
        results: dict[str, Any] = {"label_path": str(label_path), "checks": {}}

        try:
            import numpy as np
            import rasterio
            with rasterio.open(str(label_path)) as src:
                label = src.read(1).astype(np.int64)
                area_px = pixel_area_m2(src)
                nodata = src.nodata
        except Exception as exc:
            return {"error": str(exc), "quality_score": 0.0}

        invalid = label == self.ignore_index
        if nodata is not None and int(nodata) != self.ignore_index:
            invalid |= label == int(nodata)
        label = np.where(invalid, self.ignore_index, label)
        total_px = label.size
        valid_px = int((~invalid).sum())
        classes, unexpected = self._classes(label)
        results["classes"] = classes
        results["unexpected_values"] = unexpected
        results["pixel_area_m2"] = round(area_px, 4)
        fg = [c for c in classes if c != 0]

        scores = []

        if "class_balance" in checks:
            class_counts = {c: int((label == c).sum()) for c in classes}
            n_vals = [v for v in class_counts.values() if v > 0]
            if len(n_vals) > 1:
                imbalance_ratio = max(n_vals) / (min(n_vals) + 1)
                balance_score = float(1.0 / (1 + imbalance_ratio / 10))
            else:
                imbalance_ratio = float("inf")
                balance_score = 0.0
            results["checks"]["class_balance"] = {
                "class_counts": class_counts,
                "class_pct": {c: round(n / max(valid_px, 1) * 100, 2) for c, n in class_counts.items()},
                "imbalance_ratio": round(imbalance_ratio, 1) if imbalance_ratio != float("inf") else None,
                "score": round(balance_score, 3),
                "status": "ok" if imbalance_ratio < 20 else "warning" if imbalance_ratio < 100 else "critical",
            }
            scores.append(balance_score)

        if "coverage" in checks:
            valid_fraction = float(valid_px / max(total_px, 1))
            coverage_score = min(1.0, valid_fraction * 2)
            results["checks"]["coverage"] = {
                "valid_pixels": valid_px,
                "total_pixels": int(total_px),
                "valid_fraction": round(valid_fraction, 3),
                "score": round(coverage_score, 3),
                "status": "ok" if valid_fraction > 0.5 else "warning",
            }
            scores.append(coverage_score)

        need_scipy = {"slivers", "connectivity"} & set(checks)
        ndimage = None
        if need_scipy:
            try:
                from scipy import ndimage
            except ImportError:
                for name in need_scipy:
                    results["checks"][name] = {"status": "skipped", "reason": "pip install scipy"}

        if ndimage is not None and "slivers" in checks:
            sliver_px = max(4, int(np.ceil(1.0 / max(area_px, 1e-12))))
            n_slivers = 0
            for c in classes:
                lab, n_comp = ndimage.label(label == c)
                if n_comp:
                    sizes = np.bincount(lab.ravel())[1:]
                    n_slivers += int((sizes < sliver_px).sum())
            sliver_score = float(1.0 - min(1.0, n_slivers / max(valid_px * 0.001, 1)))
            results["checks"]["slivers"] = {
                "n_slivers": n_slivers,
                "sliver_threshold_px": sliver_px,
                "score": round(sliver_score, 3),
                "status": "ok" if n_slivers < 100 else "warning",
            }
            scores.append(sliver_score)

        if ndimage is not None and "connectivity" in checks:
            min_obj_px = 16
            n_comp_total, fg_px, fg_px_in_objects = 0, 0, 0
            for c in fg:
                lab, n_comp = ndimage.label(label == c)
                n_comp_total += n_comp
                if n_comp:
                    sizes = np.bincount(lab.ravel())[1:]
                    fg_px += int(sizes.sum())
                    fg_px_in_objects += int(sizes[sizes >= min_obj_px].sum())
            conn_score = (fg_px_in_objects / fg_px) if fg_px else 0.0
            results["checks"]["connectivity"] = {
                "n_components": n_comp_total,
                "foreground_pixels": fg_px,
                "share_in_objects": round(conn_score, 3),
                "min_object_px": min_obj_px,
                "score": round(conn_score, 3),
                "status": "ok" if conn_score >= 0.8 else "warning",
            }
            scores.append(conn_score)

        if "boundaries" in checks:
            try:
                import cv2
                tot_a, tot_iq = 0.0, 0.0
                for c in fg:
                    mask = (label == c).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        a, p = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
                        if a >= 4 and p > 0:
                            tot_a += a
                            tot_iq += a * min(1.0, 4 * np.pi * a / (p * p))
                iq = tot_iq / tot_a if tot_a else 0.0
                b_score = min(1.0, iq * 2)
                results["checks"]["boundaries"] = {
                    "isoperimetric_quotient": round(float(iq), 3),
                    "score": round(float(b_score), 3),
                    "status": "ok" if iq > 0.05 else "warning",
                }
                scores.append(b_score)
            except ImportError:
                results["checks"]["boundaries"] = {"status": "skipped", "reason": "pip install opencv-python"}

        quality_score = float(sum(scores) / len(scores)) if scores else 0.0
        n_foreground = int(((label != 0) & ~invalid).sum())
        results["foreground_pixels"] = n_foreground
        if n_foreground == 0:
            quality_score = 0.0          # a label with nothing on it cannot be "mostly fine"
        results["quality_score"] = round(quality_score, 3)
        results["quality_grade"] = (
            "A" if quality_score >= 0.9 else
            "B" if quality_score >= 0.75 else
            "C" if quality_score >= 0.6 else
            "D" if quality_score >= 0.4 else "F"
        )
        results["recommendations"] = self._recommendations(results)
        return results

    def _recommendations(self, results: dict) -> list[str]:
        recs = []
        checks = results.get("checks", {})

        if results.get("unexpected_values"):
            recs.append(f"Values {results['unexpected_values']} are outside 0..{(self.num_classes or 1) - 1} and were "
                        "not counted: raise num_classes or leave it as None")

        cb = checks.get("class_balance", {})
        if cb.get("status") in ("warning", "critical"):
            ratio = cb.get("imbalance_ratio")
            if ratio is not None and ratio > 0:
                recs.append(f"Class imbalance ratio {ratio:.0f}x -- use a weighted/overlap loss "
                            "(see docs/training/losses.md) or oversample the minority class")
            else:
                recs.append("Only one class has pixels -- the label raster carries no information to learn from")

        slivers = checks.get("slivers", {})
        if slivers.get("status") == "warning":
            recs.append(f"{slivers.get('n_slivers', 0)} specks smaller than {slivers.get('sliver_threshold_px')} px "
                        "-- run clean() to remove them")

        conn = checks.get("connectivity", {})
        if conn.get("status") == "warning":
            recs.append(f"Only {conn.get('share_in_objects', 0):.0%} of the foreground sits in objects of "
                        f">= {conn.get('min_object_px')} px -- the label is mostly dust; run clean() or check the source")

        cov = checks.get("coverage", {})
        if cov.get("valid_fraction", 1.0) < 0.5:
            recs.append("Less than 50% of pixels carry a label (the rest are no-data) -- expand coverage or "
                        "choose chips with more labelled area")

        if results.get("foreground_pixels", 1) == 0:
            recs.append("The raster has no foreground pixels (everything is background or no-data) -- "
                        "the source returned nothing for this area")

        if results.get("quality_score", 1.0) < 0.7:
            recs.append("Quality score < 0.7 -- inspect the label over the imagery before training")

        return recs

    # ------------------------------------------------------------------ clean
    def clean(
        self,
        label_path: str | Path,
        output_path: str | Path,
        remove_slivers: bool = True,
        min_area_m2: float = 5.0,
        fill_holes: bool = True,
        smooth: bool = True,
    ) -> dict[str, Any]:
        """Remove specks, fill holes and close small gaps in the foreground classes.

        Class 0 (background) and ``ignore_index`` are never edited except that holes / gaps are filled
        *from* background: a foreground class can grow into background pixels, never into another class
        or into no-data. ``min_area_m2`` is real ground area (a geographic CRS is converted to metres).

        Note ``fill_holes`` also fills genuine courtyards (a ring-shaped building becomes a solid one), and
        ``smooth`` (a 3x3 closing) merges objects closer than ~2 pixels. Turn them off for such labels.

        Returns:
            ``success``, ``output_path``, ``n_components_removed``, ``min_component_px``,
            ``before_label_pixels``, ``after_label_pixels`` (foreground pixels, no-data excluded),
            ``holes_filled_px``, ``smoothed_px``.
        """
        try:
            import numpy as np
            import rasterio
            from scipy import ndimage
        except ImportError as exc:
            raise ImportError(f"scipy + rasterio required: {exc}")

        label_path = Path(label_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(str(label_path)) as src:
            label = src.read(1).astype(np.int64)
            profile = src.profile.copy()
            area_px = pixel_area_m2(src)

        classes, _ = self._classes(label)
        fg = [c for c in classes if c != 0]
        min_px = max(1, int(np.ceil(min_area_m2 / max(area_px, 1e-12))))
        n_removed = holes_px = smooth_px = 0
        cleaned = label.copy()
        kernel = np.ones((3, 3), np.uint8)

        for c in fg:
            if remove_slivers:
                lab, n_comp = ndimage.label(cleaned == c)
                if n_comp:
                    sizes = np.bincount(lab.ravel())
                    small = np.nonzero(sizes[1:] < min_px)[0] + 1
                    if small.size:
                        cleaned[np.isin(lab, small)] = 0
                        n_removed += int(small.size)
            if fill_holes:
                filled = ndimage.binary_fill_holes(cleaned == c)
                add = filled & (cleaned == 0)
                holes_px += int(add.sum())
                cleaned[add] = c
            if smooth:
                try:
                    import cv2
                    closed = cv2.morphologyEx((cleaned == c).astype(np.uint8), cv2.MORPH_CLOSE, kernel) == 1
                    add = closed & (cleaned == 0)
                    smooth_px += int(add.sum())
                    cleaned[add] = c
                except ImportError:
                    pass

        def count_fg(a):
            return int(((a > 0) & (a != self.ignore_index)).sum())

        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(cleaned.astype(profile["dtype"])[np.newaxis])
            dst.update_tags(cleaned_by="LabelQualityAssessor", min_area_m2=str(min_area_m2))

        return {
            "success": True,
            "output_path": str(output_path),
            "n_components_removed": n_removed,
            "min_component_px": min_px,
            "before_label_pixels": count_fg(label),
            "after_label_pixels": count_fg(cleaned),
            "holes_filled_px": holes_px,
            "smoothed_px": smooth_px,
        }

    # ------------------------------------------------------------------ report
    def report_html(self, results: dict) -> str:
        """Generate a small standalone HTML report."""
        from html import escape

        grade = str(results.get("quality_grade", "?"))
        score = results.get("quality_score", 0)
        color = "#22c55e" if grade in ("A", "B") else "#f59e0b" if grade == "C" else "#ef4444"
        recs = results.get("recommendations", [])

        checks_html = ""
        for check_name, check_data in results.get("checks", {}).items():
            status = check_data.get("status", "unknown")
            icon = "✓" if status == "ok" else "⚠" if status == "warning" else "✗"
            checks_html += (f'<tr><td>{escape(str(check_name))}</td><td>{icon} {escape(str(status))}</td>'
                            f'<td>{escape(str(check_data.get("score", "—")))}</td></tr>')

        recs_html = "".join(f"<li>{escape(str(r))}</li>" for r in recs) if recs else "<li>No issues found</li>"

        return f"""<html><body style="font-family:sans-serif;padding:20px">
<h2>Label Quality Report</h2>
<p>File: <code>{escape(str(results.get("label_path", "?")))}</code></p>
<h3>Grade: <span style="color:{color};font-size:2em">{escape(grade)}</span> ({score:.2%})</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse">
<tr><th>Check</th><th>Status</th><th>Score</th></tr>
{checks_html}
</table>
<h3>Recommendations:</h3><ul>{recs_html}</ul>
</body></html>"""
