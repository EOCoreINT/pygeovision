"""LiDAR / point-cloud processing: read LAS/LAZ, canopy height model, per-point classification, building footprints.

Needs ``laspy`` (``pip install laspy``; add ``lazrs`` for ``.laz``). ``classify_points`` also needs torch.
All coordinates are used in the file's own units; the functions that report metres or square metres assume a
projected CRS in metres (the norm for airborne LiDAR) and refuse a geographic one where it would give wrong areas.
"""
from __future__ import annotations

import logging
import math
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

#: LAS standard classes that are noise (low/high points, "bird" returns): left out of surface models.
NOISE_CLASSES = (7, 18)


def _require_laspy():
    try:
        import laspy
    except ImportError as exc:
        raise ImportError("laspy is required for point clouds: pip install laspy (and lazrs for .laz files)") from exc
    return laspy


def _grid(xmin: float, ymin: float, xmax: float, ymax: float, resolution: float):
    """Cell grid with exactly ``resolution``-sized cells covering the extent (north-up)."""
    from rasterio.transform import from_origin
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    w = max(1, math.ceil((xmax - xmin) / resolution))
    h = max(1, math.ceil((ymax - ymin) / resolution))
    return w, h, from_origin(xmin, ymax, resolution, resolution)


def _to_crs(crs_str: str):
    if not crs_str or crs_str == "unknown":
        return None
    try:
        import pyproj
        return pyproj.CRS.from_user_input(crs_str)
    except Exception:
        return None


class PointCloudProcessor:
    """Read LiDAR files and derive a canopy height model, per-point classes and building footprints.

    What it does, and does not do:

    * :meth:`read` - LAS/LAZ into numpy arrays.
    * :meth:`canopy_height_model` - gridded surface height above ground, from a *classified* file (ground =
      LAS class 2) or, as a rough fallback, from the lowest point per cell.
    * :meth:`classify_points` - runs a PointNet++ or RandLA-Net **you trained** over the file in blocks and
      writes the predicted class into the LAS classification field. No pretrained weights ship with the
      package and there is no point-cloud training loop in ``Trainer``; with no checkpoint the output is noise.
    * :meth:`extract_buildings` - building polygons from points already classified as buildings (class 6).
      The polygons follow the raster grid (blocky edges); they are not regularised outlines.

    Example::

        proc = PointCloudProcessor()
        proc.canopy_height_model("forest.las", "chm.tif", resolution=1.0)
        proc.extract_buildings("urban_classified.las", "buildings.geojson")
    """

    def __init__(self) -> None:
        pass

    # ── reading ─────────────────────────────────────────────────────────────

    def read(self, path: str) -> dict[str, Any]:
        """Load a LAS/LAZ file.

        Returns ``{"n_points", "x", "y", "z", "intensity", "classification", "crs", "bounds"}`` where the
        coordinate arrays are float64 in the file's units, ``intensity`` and ``classification`` are the raw
        integer arrays, ``crs`` is a string such as ``"EPSG:32630"`` (``"unknown"`` if the header has none) and
        ``bounds`` is ``(xmin, ymin, xmax, ymax)``.

        Raises:
            ImportError: laspy is not installed.
            FileNotFoundError: no such file.
            ValueError: the file has no points, or cannot be parsed as LAS/LAZ.
        """
        laspy = _require_laspy()
        import numpy as np

        if not pathlib.Path(path).is_file():
            raise FileNotFoundError(f"point cloud not found: {path}")
        try:
            las = laspy.read(path)
        except Exception as exc:
            raise ValueError(f"could not read {path} as LAS/LAZ ({type(exc).__name__}: {exc}); "
                             "for .laz files install lazrs or laszip") from exc
        if len(las.x) == 0:
            raise ValueError(f"{path} contains no points")
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        crs = None
        try:
            parsed = las.header.parse_crs()
            if parsed is not None:
                crs = parsed.to_string()
        except Exception:
            crs = None
        return {
            "n_points": int(len(x)),
            "x": x, "y": y, "z": z,
            "intensity": np.asarray(las.intensity) if "intensity" in las.point_format.dimension_names else None,
            "classification": np.asarray(las.classification),
            "crs": crs or "unknown",
            "bounds": (float(x.min()), float(y.min()), float(x.max()), float(y.max())),
        }

    # ── canopy height model ─────────────────────────────────────────────────

    def canopy_height_model(
        self,
        las_path: str,
        output_path: str,
        resolution: float = 1.0,
        filter_ground: bool = True,
        exclude_noise: bool = True,
    ) -> dict[str, Any]:
        """Canopy height model, CHM = highest point per cell (DSM) minus ground height (DTM).

        Args:
            las_path: input LAS/LAZ.
            output_path: GeoTIFF to write (float32, one band, LZW, nodata -9999, the file's CRS).
            resolution: cell size in the file's units (metres for projected data). The raster covers the
                points' bounding box with cells of exactly this size.
            filter_ground: ``True`` (default) builds the DTM from points classified as ground (LAS class 2) -
                the file must have a ground classification. ``False`` takes the lowest point of every cell as
                ground; use it for unclassified files, but cells fully covered by canopy then show a too-low
                height because their lowest return is vegetation.
            exclude_noise: ignore points in LAS noise classes 7 and 18 (they would create tall spikes).

        Cells with no points at all are nodata (-9999). Cells with points but no ground take the DTM of the
        nearest cell that has ground (nearest-neighbour fill: fine for flat or gently sloped terrain, blocky on
        steep slopes). Negative heights are set to 0.

        Returns:
            ``{"success", "output_path", "n_points", "resolution", "crs", "grid": {"width","height"},
            "empty_cells_fraction", "height_stats": {"max_m","mean_m","coverage","canopy_cover_2m"}}``;
            ``mean_m`` averages cells above 0, ``coverage`` is the share of data cells above 0 and
            ``canopy_cover_2m`` the share at or above 2 m.

        Raises:
            ValueError: ``filter_ground=True`` but the file has no ground points (try ``filter_ground=False``).
        """
        import numpy as np
        import rasterio

        pc = self.read(las_path)
        x, y, z, cls = pc["x"], pc["y"], pc["z"], pc["classification"]
        if exclude_noise:
            keep = ~np.isin(cls, NOISE_CLASSES)
            if not keep.all():
                x, y, z, cls = x[keep], y[keep], z[keep], cls[keep]
        if len(x) == 0:
            raise ValueError("every point is in a noise class; nothing to model")

        ground = cls == 2 if filter_ground else np.ones(len(cls), dtype=bool)
        if filter_ground and not ground.any():
            raise ValueError(f"{las_path} has no ground points (LAS class 2), so no terrain model can be built. "
                             "Classify the ground first, or pass filter_ground=False to use the lowest point per cell.")

        xmin, ymin, xmax, ymax = float(x.min()), float(y.min()), float(x.max()), float(y.max())
        W, H, transform = _grid(xmin, ymin, xmax, ymax, resolution)

        def cells(xs, ys):
            return (np.clip(((ymax - ys) / resolution).astype(int), 0, H - 1),
                    np.clip(((xs - xmin) / resolution).astype(int), 0, W - 1))

        dsm = np.full((H, W), -np.inf)
        r, c = cells(x, y)
        np.maximum.at(dsm, (r, c), z)
        has_points = np.isfinite(dsm)

        if filter_ground:
            gsum = np.zeros((H, W))
            gcnt = np.zeros((H, W), dtype=np.int64)
            gr, gc = cells(x[ground], y[ground])
            np.add.at(gsum, (gr, gc), z[ground])
            np.add.at(gcnt, (gr, gc), 1)
            dtm_valid = gcnt > 0
            dtm = np.where(dtm_valid, gsum / np.maximum(gcnt, 1), 0.0)
        else:
            dtm = np.full((H, W), np.inf)
            np.minimum.at(dtm, (r, c), z)
            dtm_valid = np.isfinite(dtm)
            dtm = np.where(dtm_valid, dtm, 0.0)

        if not dtm_valid.all():
            from scipy.ndimage import distance_transform_edt
            _, nearest = distance_transform_edt(~dtm_valid, return_indices=True)
            dtm = dtm[tuple(nearest)]

        chm = np.where(has_points, np.maximum(0.0, dsm - dtm), np.nan)

        crs_str = pc["crs"]
        out_crs = _to_crs(crs_str)
        if out_crs is None:
            logger.warning("%s: no usable CRS in the file header (%r); the GeoTIFF is written without a CRS. "
                           "Set it yourself once you know the EPSG code.", las_path, crs_str)

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        nodata = -9999.0
        with rasterio.open(output_path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                           crs=out_crs, transform=transform, compress="lzw", nodata=nodata) as dst:
            dst.write(np.where(np.isnan(chm), nodata, chm).astype("float32")[np.newaxis])
            dst.update_tags(source="LiDAR_CHM", resolution=str(resolution))

        valid = ~np.isnan(chm)
        vals = chm[valid]
        return {
            "success": True, "output_path": output_path,
            "n_points": int(len(x)), "resolution": resolution,
            "crs": str(out_crs) if out_crs else "unset (unknown source CRS)",
            "grid": {"width": W, "height": H},
            "empty_cells_fraction": round(float(1 - valid.mean()), 3),
            "height_stats": {
                "max_m": round(float(vals.max()), 1),
                "mean_m": round(float(vals[vals > 0].mean()), 1) if (vals > 0).any() else 0,
                "coverage": round(float((vals > 0).mean()), 3),
                "canopy_cover_2m": round(float((vals >= 2.0).mean()), 3),
            },
        }

    # ── per-point classification ────────────────────────────────────────────

    def classify_points(
        self,
        las_path: str,
        output_path: str,
        num_classes: int = 3,
        class_names: list[str] | None = None,
        model: str = "randlanet",
        checkpoint_path: str | None = None,
        device: str | None = None,
        block_size: float = 50.0,
        max_points: int = 8192,
        min_points: int = 1024,
        coord_scale: float = 1.0,
        intensity_scale: float = 65535.0,
        class_codes: list[int] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Label every point with a PointNet++ or RandLA-Net you trained, and write a new LAS/LAZ.

        The cloud is cut into ``block_size`` x ``block_size`` squares (file units); a square with more than
        ``max_points`` points is split at random into several chunks, and a chunk with fewer than
        ``min_points`` is padded with repeated points (the padding is discarded). Every point is predicted
        exactly once. Network input per chunk is 4 channels: x, y, z minus the chunk's centroid, divided by
        ``coord_scale``, and intensity divided by ``intensity_scale`` (0 if the file has none). **Use the same
        scaling when you train**, otherwise the checkpoint sees inputs it never learned.

        Args:
            num_classes: output classes of the network (must equal the checkpoint's).
            class_names: one name per class, for the count report.
            model: ``"randlanet"`` or ``"pointnet2"``.
            checkpoint_path: a ``state_dict`` saved with ``torch.save`` from a model built by
                ``get_model('randlanet' | 'pointnet2-ssg', num_classes=..., in_channels=4)``. Without it the
                weights are random and the labels are meaningless; this is allowed only to test plumbing and
                the result says so (``"meaningful": False``).
            class_codes: LAS classification code written for each network class (default: the class index).
                Map to standard codes (2 ground, 5 high vegetation, 6 building...) if other software will read it.
            seed: fixes the random sampling inside the networks and the chunk split, so a run is repeatable.

        Returns:
            ``{"success", "output_path", "n_points", "model", "used_pretrained_checkpoint", "meaningful",
            "n_chunks", "class_counts": {name: n}}`` (plus ``"warning"`` when no checkpoint was given).

        Raises:
            ImportError: torch or laspy missing. ValueError: bad arguments, or the checkpoint does not fit the
            network (message names the likely cause).
        """
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for classify_points: pip install torch") from exc
        if model not in ("randlanet", "pointnet2"):
            raise ValueError(f"model must be 'randlanet' or 'pointnet2', got {model!r}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        names = class_names or [f"class_{i}" for i in range(num_classes)]
        if len(names) != num_classes:
            raise ValueError(f"class_names has {len(names)} entries for num_classes={num_classes}")
        codes = list(class_codes) if class_codes is not None else list(range(num_classes))
        if len(codes) != num_classes:
            raise ValueError(f"class_codes has {len(codes)} entries for num_classes={num_classes}")
        if block_size <= 0 or max_points < 16 or min_points < 1 or coord_scale <= 0 or intensity_scale <= 0:
            raise ValueError("block_size, coord_scale and intensity_scale must be > 0, max_points >= 16, min_points >= 1")
        min_points = min(min_points, max_points)

        laspy = _require_laspy()
        pc = self.read(las_path)
        las = laspy.read(las_path)
        max_code = 31 if las.header.point_format.id <= 5 else 255
        if min(codes) < 0 or max(codes) > max_code:
            raise ValueError(f"class_codes must be within 0..{max_code} for LAS point format {las.header.point_format.id}")

        from pygeovision.models.registry import get_model
        registry_name = "randlanet" if model == "randlanet" else "pointnet2-ssg"
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)            # also fixes the random initial weights when no checkpoint is given
        net = get_model(registry_name, num_classes=num_classes, in_channels=4, pretrained=False)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device).eval()
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if isinstance(state, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            try:
                net.load_state_dict(state)
            except RuntimeError as exc:
                raise ValueError(f"checkpoint {checkpoint_path} does not fit a {registry_name} with "
                                 f"num_classes={num_classes} and 4 input channels (xyz + intensity): {exc}") from exc
            warning = None
        else:
            warning = ("no checkpoint_path: the network has random weights, so the classes are meaningless "
                       "(plumbing test only). Train a model and pass its state_dict.")
            logger.warning("classify_points: %s", warning)

        x, y, z = pc["x"], pc["y"], pc["z"]
        n = len(x)
        inten = pc["intensity"].astype("float64") / intensity_scale if pc["intensity"] is not None else np.zeros(n)
        rng = np.random.default_rng(seed)

        ix = np.floor((x - x.min()) / block_size).astype(np.int64)
        iy = np.floor((y - y.min()) / block_size).astype(np.int64)
        key = ix * (iy.max() + 1) + iy
        order = np.argsort(key, kind="stable")
        _, starts = np.unique(key[order], return_index=True)
        chunks: list[Any] = []
        for blk in np.split(order, starts[1:]):
            if len(blk) > max_points:
                perm = rng.permutation(blk)
                chunks.extend(np.array_split(perm, math.ceil(len(blk) / max_points)))
            else:
                chunks.append(blk)

        pred = np.zeros(n, dtype=np.int64)
        try:
            with torch.no_grad():
                for idx in chunks:
                    use = idx
                    if len(use) < min_points:
                        use = np.concatenate([idx, rng.choice(idx, min_points - len(idx), replace=True)])
                    cx, cy, cz = x[use].mean(), y[use].mean(), z[use].mean()
                    feats = np.stack([(x[use] - cx) / coord_scale, (y[use] - cy) / coord_scale,
                                      (z[use] - cz) / coord_scale, inten[use]]).astype(np.float32)
                    logits = net(torch.from_numpy(feats).unsqueeze(0).to(device))
                    pred[idx] = logits.argmax(dim=1)[0].cpu().numpy()[: len(idx)]
        finally:
            torch.set_rng_state(rng_state)

        las.classification = np.asarray(codes, dtype=np.uint8)[pred]
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        las.write(output_path)

        result = {
            "success": True, "output_path": output_path, "n_points": int(n), "model": registry_name,
            "used_pretrained_checkpoint": checkpoint_path is not None,
            "meaningful": checkpoint_path is not None,
            "n_chunks": len(chunks),
            "class_counts": {names[i]: int((pred == i).sum()) for i in range(num_classes)},
        }
        if warning:
            result["warning"] = warning
        return result

    # ── building footprints ─────────────────────────────────────────────────

    def extract_buildings(
        self,
        las_path: str,
        output_path: str,
        building_class: int = 6,
        resolution: float = 0.5,
        min_area_m2: float = 10.0,
        gap_fill_m: float = 1.5,
        simplify_m: float = 0.0,
    ) -> dict[str, Any]:
        """Polygons for the points of one class (LAS 6 = building) written as GeoJSON.

        Steps: points of ``building_class`` are put on a grid of ``resolution``-sized cells; a morphological
        closing joins cells separated by gaps up to about ``gap_fill_m`` (LiDAR returns are scattered, so
        without it one roof breaks into many pieces); the mask is vectorised and polygons under
        ``min_area_m2`` are dropped. Raise ``gap_fill_m`` for sparse clouds, lower it if neighbouring buildings
        merge. The outline follows the cell grid: its position is only as good as ``resolution`` and it is
        not a regularised (square-cornered) footprint; ``simplify_m`` > 0 smooths the staircase.

        The CRS must be projected (metres) because areas are measured in it; a geographic CRS raises
        ``ValueError``. A file without CRS is assumed to be in metres (a warning is logged).

        Returns:
            ``{"success": True, "output_path", "n_buildings", "n_building_points", "total_area_m2"}``; when
            the file has no point of that class ``{"success": False, "error": ...}`` and nothing is written.
        """
        import json

        import numpy as np
        from scipy.ndimage import binary_closing
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape as shp_shape

        if resolution <= 0 or gap_fill_m < 0 or min_area_m2 < 0 or simplify_m < 0:
            raise ValueError("resolution must be > 0 and gap_fill_m, min_area_m2, simplify_m >= 0")
        pc = self.read(las_path)
        out_crs = _to_crs(pc["crs"])
        if out_crs is not None and out_crs.is_geographic:
            raise ValueError(f"{las_path} is in a geographic CRS ({pc['crs']}); building areas need a projected "
                             "CRS in metres. Reproject the point cloud first.")
        if out_crs is None:
            logger.warning("%s: no CRS in the header; assuming coordinates are in metres.", las_path)

        x, y, cls = pc["x"], pc["y"], pc["classification"]
        mask = cls == building_class
        if not mask.any():
            return {"success": False,
                    "error": (f"No points with classification={building_class} found. Run classify_points() first "
                              "if this file lacks building classification, or pass the correct class id.")}

        xmin, ymin, xmax, ymax = pc["bounds"]
        W, H, transform = _grid(xmin, ymin, xmax, ymax, resolution)
        cols = np.clip(((x[mask] - xmin) / resolution).astype(int), 0, W - 1)
        rows = np.clip(((ymax - y[mask]) / resolution).astype(int), 0, H - 1)
        raster = np.zeros((H, W), dtype=bool)
        raster[rows, cols] = True

        # closing with n iterations bridges gaps up to 2n cells; pad so buildings at the tile edge are not eroded
        n_iter = max(1, math.ceil(gap_fill_m / (2 * resolution)))
        padded = np.pad(raster, n_iter + 1)
        closed = binary_closing(padded, iterations=n_iter)[n_iter + 1:-(n_iter + 1), n_iter + 1:-(n_iter + 1)]

        features = []
        for geom, val in rio_shapes(closed.astype(np.uint8), mask=closed, transform=transform):
            if val != 1:
                continue
            poly = shp_shape(geom)
            if simplify_m > 0:
                poly = poly.simplify(simplify_m, preserve_topology=True)
            if poly.is_empty or poly.area < min_area_m2:
                continue
            features.append({"type": "Feature", "geometry": poly.__geo_interface__,
                             "properties": {"area_m2": round(poly.area, 1), "source": "LiDAR"}})

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
        epsg = out_crs.to_epsg() if out_crs is not None else None
        if epsg:
            fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
        with open(output_path, "w") as f:
            json.dump(fc, f)
        return {
            "success": True, "output_path": output_path, "n_buildings": len(features),
            "n_building_points": int(mask.sum()),
            "total_area_m2": round(sum(ft["properties"]["area_m2"] for ft in features), 1),
        }


__all__ = ["PointCloudProcessor"]
