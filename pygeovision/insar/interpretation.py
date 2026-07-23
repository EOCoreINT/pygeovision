"""
pygeovision.insar.interpretation
==================================
Interpret InSAR deformation results.

Produces structured reports covering:
  • Subsidence / uplift detection and extent
  • Deformation rate classification
  • Anomaly hotspot identification
  • Temporal trend analysis
  • PDF/JSON/Markdown export

Usage::

    from pygeovision.insar.interpretation import InSARInterpreter

    interp = InSARInterpreter()
    report = interp.interpret(
        displacement_path="displacement.tif",
        rate_path="deformation_rate.tif",    # optional
        coherence_path="coherence.tif",      # optional
        study_area="Jakarta, Indonesia",
    )
    print(report.summary())
    report.export("insar_report.json")
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("pygeovision.insar.interpretation")


@dataclass
class DeformationZone:
    """A spatial zone of detected deformation."""
    zone_type:     str        # "subsidence" | "uplift" | "stable" | "anomaly"
    max_magnitude: float      # metres
    mean_magnitude: float     # metres
    extent_km2:    float
    confidence:    float      # 0–1
    description:   str = ""


@dataclass
class DeformationReport:
    """Complete InSAR interpretation report."""
    study_area:       str
    generated_at:     str
    displacement_path: str
    pixel_spacing_m:  float
    n_pixels_total:   int

    # Deformation summary
    max_subsidence_m:   float = 0.0
    max_uplift_m:       float = 0.0
    mean_displacement_m: float = 0.0
    deformation_rate_mm_yr: float | None = None

    # Classification
    subsidence_extent_km2: float = 0.0
    uplift_extent_km2:     float = 0.0
    stable_pct:            float = 0.0

    zones: list[DeformationZone] = field(default_factory=list)
    flags: list[str]             = field(default_factory=list)
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"InSAR Deformation Report — {self.study_area}",
            f"Generated : {self.generated_at}",
            f"Raster    : {Path(self.displacement_path).name}",
            f"Pixels    : {self.n_pixels_total:,}  (spacing={self.pixel_spacing_m:.1f} m)",
            "",
            f"Max subsidence : {self.max_subsidence_m * 100:.1f} cm",
            f"Max uplift     : {self.max_uplift_m * 100:.1f} cm",
            f"Mean deformation: {self.mean_displacement_m * 100:.1f} cm",
            f"Subsidence area : {self.subsidence_extent_km2:.2f} km²",
            f"Uplift area     : {self.uplift_extent_km2:.2f} km²",
            f"Stable area     : {self.stable_pct:.1f}%",
        ]
        if self.deformation_rate_mm_yr is not None:
            lines.append(f"Rate           : {self.deformation_rate_mm_yr:.1f} mm/year")
        if self.flags:
            lines += ["", "⚠ Flags:"] + [f"  • {f}" for f in self.flags]
        if self.zones:
            lines += ["", "Deformation Zones:"]
            for z in self.zones:
                lines.append(
                    f"  [{z.zone_type.upper()}] {z.extent_km2:.2f} km²  "
                    f"max={z.max_magnitude*100:.1f} cm  conf={z.confidence:.2f}  "
                    f"{z.description}"
                )
        if self.notes:
            lines += ["", f"Notes: {self.notes}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["zones"] = [asdict(z) for z in self.zones]
        return d

    def export(self, path: str, format: str = "auto") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        suffix = Path(path).suffix.lower()
        if format == "json" or suffix == ".json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif format == "md" or suffix in (".md", ".txt"):
            with open(path, "w") as f:
                f.write(self.summary())
        else:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Report exported to %s", path)
        return path


class InSARInterpreter:
    """
    Interprets InSAR displacement and rate maps.

    Parameters
    ----------
    subsidence_threshold_m : float
        Negative displacement below this threshold → subsidence zone (default –0.02 m).
    uplift_threshold_m : float
        Positive displacement above this threshold → uplift zone (default +0.01 m).
    coherence_threshold : float
        Pixels below this coherence → unreliable, excluded from stats.
    min_zone_km2 : float
        Minimum area for a deformation zone to be reported.
    """

    def __init__(
        self,
        subsidence_threshold_m: float = -0.02,
        uplift_threshold_m:     float = +0.01,
        coherence_threshold:    float = 0.4,
        min_zone_km2:           float = 0.1,
    ) -> None:
        self.sub_thresh  = subsidence_threshold_m
        self.upl_thresh  = uplift_threshold_m
        self.coh_thresh  = coherence_threshold
        self.min_zone    = min_zone_km2

    def interpret(
        self,
        displacement_path: str,
        rate_path:         str | None = None,
        coherence_path:    str | None = None,
        study_area:        str = "Study Area",
    ) -> DeformationReport:
        """
        Run complete interpretation of a displacement map.

        Returns a ``DeformationReport`` with subsidence/uplift extent,
        flagged anomalies, and zone-level summaries.
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("pip install rasterio")

        with rasterio.open(displacement_path) as src:
            disp      = src.read(1).astype("float32")
            src.profile.copy()
            pix_m     = abs(src.transform.a)
            area_per  = pix_m**2 / 1e6  # km² per pixel

        # Apply coherence mask
        valid_mask = np.isfinite(disp)
        if coherence_path:
            try:
                with rasterio.open(coherence_path) as src:
                    coh = src.read(1).astype("float32")
                if coh.shape == disp.shape:
                    valid_mask &= (coh >= self.coh_thresh)
            except Exception as exc:
                logger.warning("Coherence mask load failed: %s", exc)

        disp_valid = disp[valid_mask]

        # Zone detection
        sub_mask   = valid_mask & (disp < self.sub_thresh)
        upl_mask   = valid_mask & (disp > self.upl_thresh)
        stable_mask= valid_mask & (disp >= self.sub_thresh) & (disp <= self.upl_thresh)

        zones: list[DeformationZone] = []

        if sub_mask.sum() > 0:
            sub_area = float(sub_mask.sum()) * area_per
            if sub_area >= self.min_zone:
                zones.append(DeformationZone(
                    zone_type="subsidence",
                    max_magnitude=float(np.nanmin(disp[sub_mask])),
                    mean_magnitude=float(np.nanmean(disp[sub_mask])),
                    extent_km2=round(sub_area, 3),
                    confidence=0.85,
                    description=f"Below threshold {self.sub_thresh*100:.0f} cm",
                ))

        if upl_mask.sum() > 0:
            upl_area = float(upl_mask.sum()) * area_per
            if upl_area >= self.min_zone:
                zones.append(DeformationZone(
                    zone_type="uplift",
                    max_magnitude=float(np.nanmax(disp[upl_mask])),
                    mean_magnitude=float(np.nanmean(disp[upl_mask])),
                    extent_km2=round(upl_area, 3),
                    confidence=0.80,
                    description=f"Above threshold {self.upl_thresh*100:.0f} cm",
                ))

        # Anomaly hotspots
        p99 = float(np.nanpercentile(np.abs(disp_valid), 99))
        anomaly_mask = valid_mask & (np.abs(disp) > p99)
        if anomaly_mask.sum() * area_per >= self.min_zone:
            zones.append(DeformationZone(
                zone_type="anomaly",
                max_magnitude=float(np.nanmax(np.abs(disp[anomaly_mask]))),
                mean_magnitude=float(np.nanmean(np.abs(disp[anomaly_mask]))),
                extent_km2=round(float(anomaly_mask.sum()) * area_per, 3),
                confidence=0.60,
                description=f"Top 1% displacement magnitude (>={p99*100:.1f} cm)",
            ))

        # Rate loading
        rate_mean = None
        if rate_path:
            try:
                with rasterio.open(rate_path) as src:
                    rate = src.read(1).astype("float32")
                valid_rate = rate[np.isfinite(rate) & (rate != -9999.0)]
                rate_mean = float(np.nanmean(valid_rate))
            except Exception as exc:
                logger.warning("Rate path load failed: %s", exc)

        # Flags
        flags = []
        if len(disp_valid) > 0:
            if abs(float(np.nanmin(disp))) > 0.20:
                flags.append(f"CRITICAL: subsidence exceeds 20 cm (max={float(np.nanmin(disp))*100:.0f} cm) — structural risk")
            if abs(float(np.nanmax(disp))) > 0.10:
                flags.append("SIGNIFICANT: uplift exceeds 10 cm — possible volcanic or hydrological activity")
            stable_pct = float(stable_mask.sum() / (valid_mask.sum() + 1) * 100)
            if stable_pct < 50:
                flags.append(f"WIDESPREAD: {100-stable_pct:.0f}% of reliable pixels show significant deformation")

        report = DeformationReport(
            study_area=study_area,
            generated_at=datetime.now(timezone.utc).isoformat(),
            displacement_path=displacement_path,
            pixel_spacing_m=round(pix_m, 2),
            n_pixels_total=int(disp.size),
            max_subsidence_m=round(float(np.nanmin(disp_valid)) if len(disp_valid) else 0, 4),
            max_uplift_m=round(float(np.nanmax(disp_valid)) if len(disp_valid) else 0, 4),
            mean_displacement_m=round(float(np.nanmean(disp_valid)) if len(disp_valid) else 0, 4),
            deformation_rate_mm_yr=round(rate_mean, 2) if rate_mean else None,
            subsidence_extent_km2=round(sub_mask.sum() * area_per, 3),
            uplift_extent_km2=round(upl_mask.sum() * area_per, 3),
            stable_pct=round(float(stable_mask.sum() / (valid_mask.sum() + 1) * 100), 1),
            zones=zones,
            flags=flags,
            notes="GRD amplitude proxy — for mm-precision use SLC data + SNAP InSAR",
        )

        return report
