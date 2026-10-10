"""YOLOv8/v9 for geospatial object detection — fully native.

``GeoYOLO`` wraps ultralytics with the geospatial pieces it lacks:

* ``weights=`` loads your own fine-tuned ``.pt`` (class names come from the
  checkpoint) or builds from a ``.yaml`` (no download needed).
* ``detect_tiled`` runs a large GeoTIFF/orthomosaic through the model in
  overlapping windows, de-duplicates across tile seams, and returns
  boxes in pixel, CRS and lon/lat coordinates (optionally as GeoJSON).
* ``fit`` trains, resumes an interrupted run, and returns the best weights.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Real, standard COCO 80-class ordering (Ultralytics' pretrained YOLOv8/v9
# checkpoints use this exact index order). Needed because a confirmed,
# severe bug relabeled EVERY detection with whatever class_names the
# caller passed, regardless of what the model actually detected -- e.g.
# client.detection.ships() labeled a detected "person" and "car" both as
# "ship", since class_names=["ship"] has only one entry and every
# detection's class index got clamped to it.
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
# Real, relevant COCO class indices for the specific detection targets
# this module exposes -- not every COCO class, just the ones that
# genuinely correspond to "ship"/"car" from an overhead view.
COCO_SHIP_CLASSES = {8}   # "boat" -- the closest real COCO class to "ship"
COCO_CAR_CLASSES = {2, 7}  # "car", "truck"


def build_yolo(variant: str = "yolov8-m", num_classes: int = 5, **kwargs) -> Any:
    """Build YOLOv8/v9 for satellite object detection.

    Args:
        variant: "yolov8-n|s|m|l|x" or "yolov9-c|e"
        num_classes: Object classes

    Example::

        model = build_yolo("yolov8-m", num_classes=3)  # ships, planes, vehicles
    """
    try:
        from ultralytics import YOLO
        size_map = {"yolov8-n": "yolov8n", "yolov8-s": "yolov8s", "yolov8-m": "yolov8m",
                    "yolov8-l": "yolov8l", "yolov8-x": "yolov8x",
                    "yolov9-c": "yolov9c", "yolov9-e": "yolov9e"}
        model_str = size_map.get(variant, "yolov8m") + ".pt"
        model = YOLO(model_str)
        return model
    except ImportError as err:
        raise ImportError("pip install ultralytics") from err




# --------------------------------------------------------------------------
# Pure helpers (no ultralytics needed -- unit-testable on their own)
# --------------------------------------------------------------------------

def tile_starts(dim: int, tile: int, stride: int) -> list[int]:
    """Window start offsets along one axis.

    Always covers the full extent: the last window is shifted back so it
    ends exactly at ``dim`` (instead of being a thin sliver), and an axis
    shorter than ``tile`` yields a single window of the whole axis.
    """
    if dim <= tile:
        return [0]
    starts = list(range(0, dim - tile + 1, stride))
    if starts[-1] + tile < dim:
        starts.append(dim - tile)
    return starts


def _iou_matrix(a, b):
    import numpy as np
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter, aa, ab


def merge_tile_detections(boxes, scores, classes, touches_seam=None,
                          iou_thr: float = 0.5, contain_thr: float | None = 0.8):
    """Cross-tile de-duplication. Returns indices to keep.

    1. Seam-fragment removal (first!): an object cut by an *interior* tile
       border is reported there as a fragment. A box that touches a seam is
       dropped when >= ``contain_thr`` of its area lies inside a clearly
       larger box of the same class. This must run before NMS: a fragment
       overlaps the whole object by IoU ~ area ratio, which can exceed
       ``iou_thr``, and NMS would then keep whichever scored higher --
       sometimes the fragment.
    2. Per-class NMS at ``iou_thr`` -- removes the same object detected in
       two overlapping tiles.

    boxes: (N,4) xyxy in full-image pixels; scores/classes: (N,).
    touches_seam: optional (N,) bool. ``contain_thr=None`` disables step 1.
    """
    import numpy as np
    import torch
    from torchvision.ops import batched_nms

    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=int)
    alive = np.arange(n)
    if touches_seam is not None and contain_thr is not None:
        inter, area, _ = _iou_matrix(boxes, boxes)
        drop = np.zeros(n, bool)
        for i in np.flatnonzero(np.asarray(touches_seam, bool)):
            if area[i] <= 0:
                continue
            same = (classes == classes[i])
            same[i] = False
            big = same & (area > area[i] * 1.05) & (inter[i] / area[i] >= contain_thr)
            drop[i] = big.any()
        alive = alive[~drop]
    keep = batched_nms(torch.as_tensor(boxes[alive], dtype=torch.float32),
                       torch.as_tensor(scores[alive], dtype=torch.float32),
                       torch.as_tensor(classes[alive], dtype=torch.int64), iou_thr).numpy()
    return alive[keep]


def _px_box_to_geo(transform, box):
    """Pixel xyxy -> [minx, miny, maxx, maxy] in CRS units (upper-left corner
    convention, y axis flipped by the affine -- so min/max are re-sorted)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    xs, ys = [], []
    for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        gx = transform.a * cx + transform.b * cy + transform.c
        gy = transform.d * cx + transform.e * cy + transform.f
        xs.append(gx)
        ys.append(gy)
    return [min(xs), min(ys), max(xs), max(ys)]


def _geo_box_to_lonlat(crs, gbox):
    """[minx,miny,maxx,maxy] in ``crs`` -> [minlon,minlat,maxlon,maxlat]."""
    if crs is None:
        return None
    if crs.is_geographic and (crs.to_epsg() == 4326):
        return list(gbox)
    from rasterio.warp import transform as warp_transform
    x1, y1, x2, y2 = gbox
    xs, ys = warp_transform(crs, "EPSG:4326", [x1, x2, x2, x1], [y1, y1, y2, y2])
    return [min(xs), min(ys), max(xs), max(ys)]


def detections_to_geojson(detections: list[dict], path: str | None = None) -> dict:
    """Detections (with ``bbox_lonlat``) -> RFC 7946 FeatureCollection."""
    feats = []
    for d in detections:
        bb = d.get("bbox_lonlat")
        if bb is None:
            raise ValueError("detections have no lon/lat: the raster has no CRS, "
                             "so GeoJSON cannot be written")
        x1, y1, x2, y2 = bb
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]},
            "properties": {k: d[k] for k in ("class", "class_id", "confidence",
                                             "area_m2", "bbox_px") if k in d},
        })
    fc = {"type": "FeatureCollection", "features": feats}
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(fc))
    return fc


def _count_train_labels(data_yaml: str, split: str = "train") -> int | None:
    """Number of non-empty YOLO label files for ``split``; None if unknown."""
    try:
        import yaml
        cfg = yaml.safe_load(Path(data_yaml).read_text())
        root = Path(cfg.get("path") or Path(data_yaml).parent)
        ent = cfg.get(split)
        ents = ent if isinstance(ent, (list, tuple)) else [ent]
        n = 0
        for e in ents:
            p = Path(e) if Path(e).is_absolute() else root / e
            if p.is_dir():
                imgs = [q for q in p.rglob("*") if q.suffix.lower() in
                        {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}]
            elif p.suffix == ".txt":
                imgs = [Path(ln.strip() if Path(ln.strip()).is_absolute() else root / ln.strip())
                        for ln in p.read_text().splitlines() if ln.strip()]
            else:
                return None
            for im in imgs:
                lab = Path(str(im).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
                if lab.exists() and lab.stat().st_size > 0 and lab.read_text().strip():
                    n += 1
        return n
    except Exception as exc:  # unknown layout -- do not block training on a guess
        logger.debug("label count skipped: %s", exc)
        return None


class _ArrayRaster:
    """Minimal rasterio-like view over an in-memory ``(bands, H, W)`` array,
    so tiled detection can run on a raster that never touched disk."""

    def __init__(self, arr, transform=None, crs=None, nodata=None):
        import numpy as np
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError(f"expected (bands, H, W); got {arr.shape}")
        self.arr = arr
        self.count, self.height, self.width = arr.shape
        self.transform = transform if transform is not None else Affine.identity()
        self.crs = CRS.from_user_input(crs) if crs is not None else None
        self.nodata = nodata

    def read(self, bands, window=None, out_shape=None):
        import numpy as np
        a = self.arr[[b - 1 for b in bands]]
        if window is not None:
            x0, y0 = int(window.col_off), int(window.row_off)
            a = a[:, y0:y0 + int(window.height), x0:x0 + int(window.width)]
        if out_shape is not None:
            _, oh, ow = out_shape
            ys = np.linspace(0, a.shape[1] - 1, oh).astype(int)
            xs = np.linspace(0, a.shape[2] - 1, ow).astype(int)
            a = a[:, ys][:, :, xs]
        return a


class GeoYOLO:
    """YOLOv8/v9 wrapper with geospatial pre/post-processing.

    Two ways to use it:

    * **Generic COCO detector** (``weights=None``): a COCO-pretrained
      checkpoint trained on ground-level photos. Accuracy on overhead
      imagery is limited by that domain shift.
    * **Your own detector** (``weights="best.pt"``): a fine-tuned
      checkpoint. Class names are read from the checkpoint, so detections
      are labelled by what the model was trained on. ``weights`` may also
      be a ``.yaml`` model definition (trains from scratch, no download).

    Detection results carry pixel, CRS and lon/lat boxes. Use
    ``detect_tiled`` for orthomosaics larger than the model input size.
    """

    def __init__(self, variant: str = "yolov8-m", num_classes: int = 5,
                 class_names: list[str] | None = None,
                 coco_class_filter: set[int] | None = None,
                 weights: str | os.PathLike | None = None) -> None:
        self.variant = variant
        self.num_classes = num_classes
        self._explicit_names = class_names is not None
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        # Only detections in coco_class_filter are kept and they are labelled
        # with the caller's name; without a filter the model's real class
        # names are used (never a blind relabel).
        self.coco_class_filter = coco_class_filter
        self.weights = str(weights) if weights is not None else None
        if self.weights and self.weights.endswith(".pt") and not os.path.exists(self.weights) \
                and (os.sep in self.weights or "/" in self.weights):
            raise FileNotFoundError(f"weights file not found: {self.weights}")
        self._model = None
        self.best_weights: str | None = None
        self.last_save_dir: str | None = None

    # ------------------------------------------------------------------ model
    def _load(self) -> None:
        if self._model is not None:
            return
        if self.weights:
            try:
                from ultralytics import YOLO
            except ImportError as err:
                raise ImportError("pip install ultralytics") from err
            self._model = YOLO(self.weights)
        else:
            self._model = build_yolo(self.variant, self.num_classes)

    @property
    def names(self) -> dict[int, str]:
        """Class-id -> name as stored in the loaded model."""
        self._load()
        n = getattr(self._model, "names", None) or {}
        return dict(n) if isinstance(n, dict) else dict(enumerate(n))

    def _label(self, cls: int) -> str:
        if self.coco_class_filter is not None:
            return self.class_names[0] if self.class_names else COCO_CLASS_NAMES[cls]
        if self._explicit_names and cls < len(self.class_names):
            return self.class_names[cls]
        nm = self.names.get(cls)
        if nm is not None:
            return str(nm)
        return COCO_CLASS_NAMES[cls] if cls < len(COCO_CLASS_NAMES) else f"class_{cls}"

    # --------------------------------------------------------------- raster IO
    @staticmethod
    def _stretch_limits(src, bands: list[int], stretch):
        """Per-band (lo, hi). ``stretch``: 'percentile' (2-98 over the whole
        raster, from a decimated read so every tile shares one stretch),
        'none' (data already 0-255), or a (lo, hi) pair."""
        import numpy as np
        if isinstance(stretch, (tuple, list)) and len(stretch) == 2:
            return [(float(stretch[0]), float(stretch[1]))] * len(bands)
        if stretch == "none":
            return [(0.0, 255.0)] * len(bands)
        if stretch != "percentile":
            raise ValueError("stretch must be 'percentile', 'none' or (lo, hi)")
        scale = max(1, int(max(src.width, src.height) // 2048))
        out_h, out_w = max(1, src.height // scale), max(1, src.width // scale)
        arr = src.read(bands, out_shape=(len(bands), out_h, out_w)).astype("float64")
        nd = src.nodata
        lims = []
        for b in range(arr.shape[0]):
            v = arr[b]
            m = np.isfinite(v)
            if nd is not None:
                m &= v != nd
            if nd is None:
                m &= v != 0  # no nodata tag: treat exact zeros (mosaic borders) as empty
            v = v[m] if m.any() else v.ravel()
            lo, hi = np.percentile(v, (2, 98))
            lims.append((float(lo), float(hi)))
        return lims

    @staticmethod
    def _to_bgr(data, lims):
        """(bands,H,W) -> uint8 (H,W,3) **BGR** (what ultralytics expects for
        numpy input; RGB would make the model see red and blue swapped)."""
        import numpy as np
        data = np.nan_to_num(data.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        data = data[:3]
        for b in range(3):
            lo, hi = lims[min(b, len(lims) - 1)]
            data[b] = np.clip((data[b] - lo) / (hi - lo + 1e-8) * 255.0, 0, 255)
        rgb = data.transpose(1, 2, 0).astype(np.uint8)
        return np.ascontiguousarray(rgb[..., ::-1])

    def _predict(self, images, conf, iou, imgsz, device, max_det=1000):
        kw = {"conf": conf, "iou": iou, "verbose": False, "max_det": max_det}
        if imgsz:
            kw["imgsz"] = imgsz
        if device:
            kw["device"] = device
        return self._model.predict(images, **kw)

    def _record(self, box_px, cls, score, transform, crs) -> dict:
        gbox = _px_box_to_geo(transform, box_px)
        ll = _geo_box_to_lonlat(crs, gbox) if crs is not None else None
        area = None
        if crs is not None and crs.is_projected:
            try:
                f = crs.linear_units_factor[1]
            except Exception:
                f = 1.0
            area = round((gbox[2] - gbox[0]) * (gbox[3] - gbox[1]) * f * f, 3)
        x1, y1, x2, y2 = box_px
        return {
            "class": self._label(int(cls)),
            "class_id": int(cls),
            "confidence": round(float(score), 4),
            "bbox_px": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "bbox_geo": [float(v) for v in gbox],
            "bbox_lonlat": [float(v) for v in ll] if ll else None,
            "center_lonlat": ([(ll[0] + ll[2]) / 2, (ll[1] + ll[3]) / 2] if ll else None),
            "area_m2": area,
        }

    # --------------------------------------------------------------- detection
    def detect(self, image_path: str, conf: float = 0.25, iou: float = 0.45,
               output_path: str | None = None, imgsz: int | None = None,
               stretch="percentile", bands: list[int] | None = None,
               device: str | None = None) -> dict[str, Any]:
        """Detect objects in a whole GeoTIFF (single pass, no tiling).

        For rasters much larger than the model input use ``detect_tiled``.
        ``output_path`` (optional) writes the detections as GeoJSON.

        Returns:
            Dict with detections (bbox in pixel, CRS and lon/lat coords),
            class labels, confidence scores.
        """
        self._load()
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                bands = bands or list(range(1, min(src.count, 3) + 1))
                lims = self._stretch_limits(src, bands, stretch)
                bgr = self._to_bgr(src.read(bands), lims)
                transform, crs = src.transform, src.crs
        except Exception as exc:
            return {"error": str(exc)}

        detections = []
        for r in self._predict([bgr], conf, iou, imgsz, device):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box, cls, sc in zip(r.boxes.xyxy.cpu().numpy(),
                                    r.boxes.cls.cpu().numpy().astype(int),
                                    r.boxes.conf.cpu().numpy()):
                if self.coco_class_filter is not None and int(cls) not in self.coco_class_filter:
                    continue  # not a real match for the requested target -- skip, don't relabel
                detections.append(self._record(box, cls, sc, transform, crs))

        out = {"n_detections": len(detections), "detections": detections,
               "model": self.weights or self.variant, "image_path": image_path}
        if output_path:
            detections_to_geojson(detections, output_path)
            out["geojson_path"] = str(output_path)
        return out

    def detect_tiled(self, image, tile_size: int = 640, overlap: float = 0.2,
                     conf: float = 0.25, iou: float = 0.45, merge_iou: float = 0.5,
                     contain_thr: float | None = 0.8, batch_size: int = 8,
                     stretch="percentile", bands: list[int] | None = None,
                     skip_empty: bool = True, output_path: str | None = None,
                     device: str | None = None, transform=None, crs=None,
                     nodata=None) -> dict[str, Any]:
        """Detect objects in a large GeoTIFF / orthomosaic (or an in-memory array).

        ``image`` is a raster path, or a ``(bands, H, W)`` array together with
        its ``transform`` (affine) and ``crs`` -- use ``stretch=(lo, hi)`` to
        say how its values map to 0-255 (e.g. ``(0, 1)`` for reflectance
        scaled to 0-1). Without a CRS, boxes are pixel-only.

        A raster path is read window-by-window (never fully in memory), each
        window is run at ``imgsz=tile_size``, boxes are mapped back to
        full-image pixels, and duplicates / seam fragments are removed with
        a per-class merge (see ``merge_tile_detections``). One contrast
        stretch is computed for the whole raster so neighbouring tiles are
        consistent.

        ``overlap`` should exceed the largest object's size as a fraction
        of the tile, otherwise a big object can be split with no tile
        containing it whole. ``output_path`` (optional) writes GeoJSON.
        """
        import numpy as np
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        self._load()
        import contextlib

        import rasterio
        from rasterio.windows import Window

        if isinstance(image, (str, os.PathLike)):
            image_path = str(image)
            opener = rasterio.open(image_path)
        else:
            image_path = None
            opener = contextlib.nullcontext(_ArrayRaster(image, transform, crs, nodata))
        with opener as src:
            bands = bands or list(range(1, min(src.count, 3) + 1))
            lims = self._stretch_limits(src, bands, stretch)
            transform, crs, nd = src.transform, src.crs, src.nodata
            W, H = src.width, src.height
            stride = max(1, int(tile_size * (1 - overlap)))
            xs, ys = tile_starts(W, tile_size, stride), tile_starts(H, tile_size, stride)

            all_boxes, all_scores, all_cls, all_seam = [], [], [], []
            n_tiles = n_skipped = 0
            pending: list[tuple[int, int, int, int, Any]] = []

            def flush():
                if not pending:
                    return
                results = self._predict([p[4] for p in pending], conf, iou, tile_size, device)
                for (x0, y0, w, h, _), r in zip(pending, results):
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    b = r.boxes.xyxy.cpu().numpy()
                    c = r.boxes.cls.cpu().numpy().astype(int)
                    s = r.boxes.conf.cpu().numpy()
                    for bb, cc, ss in zip(b, c, s):
                        if self.coco_class_filter is not None and int(cc) not in self.coco_class_filter:
                            continue
                        # which tile borders are interior seams (not the raster edge)?
                        seam = ((x0 > 0 and bb[0] <= 2) or (y0 > 0 and bb[1] <= 2) or
                                (x0 + w < W and bb[2] >= w - 2) or (y0 + h < H and bb[3] >= h - 2))
                        all_boxes.append([bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0])
                        all_scores.append(ss)
                        all_cls.append(cc)
                        all_seam.append(seam)
                pending.clear()

            for y0 in ys:
                for x0 in xs:
                    w, h = min(tile_size, W - x0), min(tile_size, H - y0)
                    raw = src.read(bands, window=Window(x0, y0, w, h))
                    if skip_empty:
                        empty = (not np.isfinite(raw).any()) or (
                            (raw == nd).all() if nd is not None else (raw == 0).all())
                        if empty:
                            n_skipped += 1
                            continue
                    n_tiles += 1
                    pending.append((x0, y0, w, h, self._to_bgr(raw, lims)))
                    if len(pending) >= batch_size:
                        flush()
            flush()

        if all_boxes:
            boxes = np.asarray(all_boxes, dtype=np.float32)
            scores = np.asarray(all_scores, dtype=np.float32)
            classes = np.asarray(all_cls, dtype=np.int64)
            keep = merge_tile_detections(boxes, scores, classes, np.asarray(all_seam),
                                         iou_thr=merge_iou, contain_thr=contain_thr)
            order = keep[np.argsort(-scores[keep])]
        else:
            boxes, scores, classes = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
            order = np.zeros(0, dtype=int)

        detections = [self._record(boxes[i], classes[i], scores[i], transform, crs) for i in order]
        out = {"n_detections": len(detections), "detections": detections,
               "model": self.weights or self.variant, "image_path": image_path,
               "n_tiles": n_tiles, "n_tiles_skipped_empty": n_skipped,
               "crs": crs.to_string() if crs is not None else None}
        if output_path:
            detections_to_geojson(detections, output_path)
            out["geojson_path"] = str(output_path)
        return out

    # ---------------------------------------------------------------- training
    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640,
              batch: int = 16, device: str | None = None, **kwargs) -> Any:
        """Fine-tune on a YOLO-format dataset. Returns ultralytics' metrics.

        ``device`` defaults to the GPU when one exists, else CPU. See
        ``fit`` for resume support and a ready-to-use result dict.
        """
        self._load()
        res = self._model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                                device=device or self._auto_device(), **kwargs)
        tr = getattr(self._model, "trainer", None)
        if tr is not None:
            self.last_save_dir = str(tr.save_dir)
            w = Path(tr.save_dir) / "weights"
            self.best_weights = str(w / "best.pt") if (w / "best.pt").exists() else str(w / "last.pt")
        return res

    def evaluate(self, data_yaml: str, split: str = "val", imgsz: int = 640,
                 batch: int = 16, device: str | None = None, conf: float = 0.001,
                 **kwargs) -> dict[str, Any]:
        """Score the loaded model on a dataset split.

        Returns ``{"map50","map50_95","precision","recall"}`` as floats. An
        empty split scores 0 -- check the labels before trusting the number.
        """
        self._load()
        m = self._model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch,
                            device=device or self._auto_device(), conf=conf,
                            verbose=False, plots=False, **kwargs)
        b = m.box
        return {"map50": float(b.map50), "map50_95": float(b.map),
                "precision": float(b.mp), "recall": float(b.mr)}

    def fit(self, data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16,
            device: str | None = None, project: str = "runs/geoyolo", name: str = "train",
            resume: bool = False, allow_empty: bool = False, **kwargs) -> dict[str, Any]:
        """Train (or resume) and return where the weights went.

        * Runs land in a fixed ``project/name`` directory (``exist_ok``), so
          the path is predictable and ``resume=True`` can find ``last.pt``.
        * ``resume=True``: continues an interrupted run from ``last.pt``
          (optimizer, epoch and schedule restored by ultralytics). If the
          run already finished, nothing is re-trained.
        * Refuses a dataset whose training labels are all empty (a model
          trained on nothing reports mAP 0 and looks like a bug later).

        Returns ``{"best_weights","last_weights","save_dir","metrics",
        "resumed","already_finished"}``; ``self.best_weights`` is also set.
        """
        import torch
        project = os.path.abspath(project)
        save_dir = Path(project) / name
        last = save_dir / "weights" / "last.pt"
        device = device or self._auto_device()

        resumed = finished = False
        results = None
        if resume and last.exists():
            from ultralytics import YOLO
            ck = torch.load(str(last), map_location="cpu", weights_only=False)
            if ck.get("epoch", 0) == -1 or ck.get("optimizer") is None:
                finished = True
                logger.info("run %s already finished; nothing to resume", save_dir)
            else:
                logger.info("resuming %s from epoch %s", last, int(ck["epoch"]) + 1)
                self._model = YOLO(str(last))
                results = self._model.train(resume=True, device=device)
                resumed = True
        if not resumed and not finished:
            if not allow_empty:
                n = _count_train_labels(data_yaml)
                if n == 0:
                    raise ValueError(
                        f"No non-empty label files found for the training split of {data_yaml}. "
                        "Training on empty labels produces a model that detects nothing. "
                        "Label the images first (or pass allow_empty=True to override).")
            self._load()
            results = self._model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                                        device=device, project=project, name=name,
                                        exist_ok=True, **kwargs)

        w = save_dir / "weights"
        best = w / "best.pt" if (w / "best.pt").exists() else w / "last.pt"
        self.best_weights, self.last_save_dir = str(best), str(save_dir)
        metrics = {}
        if results is not None and hasattr(results, "results_dict"):
            metrics = {k: float(v) for k, v in results.results_dict.items()
                       if isinstance(v, (int, float))}
        return {"best_weights": str(best), "last_weights": str(w / "last.pt"),
                "save_dir": str(save_dir), "metrics": metrics,
                "resumed": resumed, "already_finished": finished}
