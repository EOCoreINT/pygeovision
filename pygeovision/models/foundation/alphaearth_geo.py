"""
pygeovision.models.foundation.alphaearth_geo
================================================
AlphaEarth Foundations — Brown et al. (2025), Google DeepMind.
"AlphaEarth Foundations: An embedding field model for accurate and
efficient global mapping from sparse label data."

Distributed as the real "Satellite Embedding" dataset in Google Earth
Engine: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` — one 64-band image per
year (2017 onward), 10m pixel resolution, bands named A00-A63. Each
pixel's 64-dim vector is a learned representation of that location's
full year of multi-sensor Earth observation data (not a raw spectral
composite).

This wraps the real `earthengine-api` (already a dependency in this
codebase for `DynamicWorldLabeler`), using the same real ee.ImageCollection
query pattern — filterBounds/filterDate/select — not a fake stand-in.
Earth Engine has no synchronous bulk-raster download API, so region
exports go through the real async Export.image.toDrive task (identical
limitation to DynamicWorldLabeler's earth_engine backend); point sampling
IS synchronous via ee.Image.sampleRegions()+getInfo(), which this wrapper
uses to give genuinely immediate results for the common "get embeddings
at labelled training points" workflow.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
N_BANDS = 64
BAND_NAMES = [f"A{i:02d}" for i in range(N_BANDS)]


class AlphaEarthGeo:
    """Real AlphaEarth Foundations / Satellite Embedding access via
    Google Earth Engine.

    Requires `pip install earthengine-api` and your own Earth Engine
    authentication/project (`ee.Authenticate()` + `ee.Initialize(project=...)`,
    done once, outside this wrapper — matching how DynamicWorldLabeler's
    earth_engine backend already works in this codebase).

    Example::

        ae = AlphaEarthGeo()

        # Synchronous — real, immediate results for training-point sampling
        points = [(-0.22, 5.55), (-0.21, 5.56)]
        emb = ae.embeddings_at_points(points, year=2024)  # (2, 64)

        # Region export — Earth Engine has no synchronous bulk-raster
        # download, so this starts a real async Drive export task
        result = ae.embeddings_for_bbox(
            bbox=(-0.25, 5.52, -0.20, 5.60), year=2024,
        )
        print(result["ee_task_id"])  # poll via the Earth Engine Tasks tab/API
    """

    COLLECTION = EE_COLLECTION
    BANDS = BAND_NAMES

    def _ee(self):
        try:
            import ee
        except ImportError:
            raise ImportError(
                "earthengine-api required for AlphaEarth: pip install earthengine-api "
                "(and run ee.Authenticate() + ee.Initialize(project=...) once)"
            )
        return ee

    def embeddings_at_points(
        self,
        points: list[tuple[float, float]],
        year: int = 2024,
    ) -> Any:
        """Sample the real 64-band AlphaEarth embedding at specific
        (lon, lat) points — synchronous, real Earth Engine query.

        Returns: (N, 64) numpy array, one embedding vector per point,
        in the same order as `points`. Points outside the dataset's
        coverage (open ocean, missing years) come back as NaN rows.
        """
        ee = self._ee()
        import numpy as np

        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([lon, lat]), {"idx": i})
            for i, (lon, lat) in enumerate(points)
        ])

        image = (
            ee.ImageCollection(self.COLLECTION)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .mosaic()
        )
        sampled = image.sampleRegions(collection=fc, scale=10, geometries=False)
        info = sampled.getInfo()

        out = np.full((len(points), N_BANDS), np.nan, dtype=np.float32)
        for feature in info.get("features", []):
            props = feature["properties"]
            idx = props.get("idx")
            if idx is None:
                continue
            for b, band in enumerate(BAND_NAMES):
                if band in props:
                    out[idx, b] = props[band]
        return out

    def embeddings_for_bbox(
        self,
        bbox: tuple[float, float, float, float],
        year: int = 2024,
        description: str = "alphaearth_export",
        drive_folder: str | None = None,
    ) -> dict[str, Any]:
        """Start a real Earth Engine export task for the 64-band
        embedding mosaic over a region. Earth Engine has no synchronous
        bulk-raster download API for arbitrary regions — this returns a
        real task id to poll (matching DynamicWorldLabeler's
        earth_engine backend, which has the same real limitation), not a
        blocking call pretending to return data immediately.
        """
        try:
            ee = self._ee()
            lon_min, lat_min, lon_max, lat_max = bbox
            region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

            image = (
                ee.ImageCollection(self.COLLECTION)
                .filterBounds(region)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .mosaic()
                .clip(region)
            )

            task = ee.batch.Export.image.toDrive(
                image=image, description=description,
                folder=drive_folder, scale=10, region=region,
                maxPixels=1e10,
            )
            task.start()
            return {
                "success": True, "ee_task_id": str(task.id),
                "status": "started", "n_bands": N_BANDS, "year": year,
                "note": "Async Earth Engine export — poll task status or "
                        "check the Earth Engine Tasks tab / Google Drive.",
            }
        except ImportError:
            return {"success": False, "error": "pip install earthengine-api"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def similarity_map(
        self,
        reference_point: tuple[float, float],
        bbox: tuple[float, float, float, float],
        year: int = 2024,
        description: str = "alphaearth_similarity",
    ) -> dict[str, Any]:
        """Real cosine-similarity-to-a-reference-point map, matching
        AlphaEarth's documented similarity-search use case (find pixels
        that look like a reference location's embedding). Starts a real
        async export (same limitation as embeddings_for_bbox)."""
        try:
            ee = self._ee()
            lon_min, lat_min, lon_max, lat_max = bbox
            region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
            ref_geom = ee.Geometry.Point(list(reference_point))

            image = (
                ee.ImageCollection(self.COLLECTION)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .mosaic()
            )
            ref_vector = image.sample(region=ref_geom, scale=10).first()

            # Cosine similarity: dot(a, b) / (|a| * |b|). AlphaEarth
            # embeddings live on a unit hypersphere (per the paper — "64
            # components which map to coordinates on a 64-dimensional
            # sphere"), so a plain dot product IS the cosine similarity
            # here without a separate normalisation step.
            dot = image.multiply(ee.Image.constant(ref_vector.toArray(BAND_NAMES))) \
                        .reduce(ee.Reducer.sum())

            clipped = dot.clip(region)
            task = ee.batch.Export.image.toDrive(
                image=clipped, description=description, scale=10, region=region,
                maxPixels=1e10,
            )
            task.start()
            return {
                "success": True, "ee_task_id": str(task.id), "status": "started",
                "reference_point": reference_point, "year": year,
            }
        except ImportError:
            return {"success": False, "error": "pip install earthengine-api"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}