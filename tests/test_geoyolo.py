"""GeoYOLO: custom weights, tiled geo-referenced detection, resumable training.

Two kinds of test, deliberately separate:

* **Logic tests** use a deterministic fake model (finds bright squares), so
  tile windows, seam de-duplication, BGR order and geo-referencing are
  checked against exact known answers.
* **Plumbing tests** use the real ultralytics package, built from the
  offline ``yolov8n.yaml`` (no downloads). A model trained for a couple of
  epochs on 24 synthetic images does NOT learn to detect -- these tests
  prove training/resume/loading work, not detection accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pygeovision.models.detection.yolo import (
    GeoYOLO,
    _count_train_labels,
    _geo_box_to_lonlat,
    _px_box_to_geo,
    detections_to_geojson,
    merge_tile_detections,
    tile_starts,
)

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

# ------------------------------------------------------------------ pure helpers


class TestTileStarts:
    def test_exact_cover_and_last_window_shifted_back(self):
        s = tile_starts(1000, 640, 512)
        assert s[0] == 0 and s[-1] + 640 == 1000

    def test_short_axis_single_window(self):
        assert tile_starts(300, 640, 512) == [0]
        assert tile_starts(640, 640, 512) == [0]

    def test_every_pixel_covered(self):
        for dim in (641, 1000, 1951, 4141):
            cov = np.zeros(dim, bool)
            for s in tile_starts(dim, 640, 512):
                cov[s:s + 640] = True
            assert cov.all(), dim


class TestMerge:
    def test_duplicate_across_tiles_removed(self):
        b = np.array([[100, 100, 140, 140], [102, 101, 141, 142]], np.float32)
        keep = merge_tile_detections(b, np.array([.9, .8]), np.array([0, 0]))
        assert list(keep) == [0]

    def test_different_classes_both_kept(self):
        b = np.array([[100, 100, 140, 140], [100, 100, 140, 140]], np.float32)
        keep = merge_tile_detections(b, np.array([.9, .8]), np.array([0, 1]))
        assert sorted(keep) == [0, 1]

    def test_seam_fragment_removed_but_only_when_touching_seam(self):
        whole = [100, 100, 200, 160]
        frag = [100, 100, 130, 160]   # IoU with whole = 0.3: NMS alone keeps it
        b = np.array([whole, frag], np.float32)
        sc, cl = np.array([.7, .95]), np.array([0, 0])
        assert sorted(merge_tile_detections(b, sc, cl, None)) == [0, 1]
        assert list(merge_tile_detections(b, sc, cl, np.array([False, True]))) == [0]
        # same geometry but the small box is NOT on a seam -> a real nested object
        assert sorted(merge_tile_detections(b, sc, cl, np.array([False, False]))) == [0, 1]

    def test_empty(self):
        assert len(merge_tile_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0))) == 0


class TestGeo:
    def test_px_to_geo_north_up_sorted(self):
        t = from_origin(1000.0, 5000.0, 10.0, 10.0)
        assert _px_box_to_geo(t, [2, 3, 6, 8]) == [1020.0, 4920.0, 1060.0, 4970.0]

    def test_lonlat_from_utm(self):
        crs = CRS.from_epsg(32630)   # Atewa region
        t = from_origin(754940.0, 702100.0 + 1951 * 10, 10.0, 10.0)
        g = _px_box_to_geo(t, [0, 0, 100, 100])
        ll = _geo_box_to_lonlat(crs, g)
        assert -0.8 < ll[0] < ll[2] < -0.4 and 6.0 < ll[1] < ll[3] < 6.7

    def test_geographic_4326_passthrough(self):
        assert _geo_box_to_lonlat(CRS.from_epsg(4326), [1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_geojson_roundtrip_and_no_crs_error(self, tmp_path):
        d = [{"class": "a", "class_id": 0, "confidence": .9, "bbox_px": [0, 0, 1, 1],
              "area_m2": 4.0, "bbox_lonlat": [1, 2, 3, 4]}]
        p = tmp_path / "o" / "d.geojson"
        fc = detections_to_geojson(d, str(p))
        assert json.loads(p.read_text()) == fc
        ring = fc["features"][0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1] and len(ring) == 5
        with pytest.raises(ValueError, match="no CRS"):
            detections_to_geojson([{**d[0], "bbox_lonlat": None}])


# ------------------------------------------------------------- fake-model logic


class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = _T(xyxy), _T(cls), _T(conf)
    def __len__(self): return len(self.xyxy.a)


class _R:
    def __init__(self, boxes): self.boxes = boxes


class FakeModel:
    """Detects axis-aligned blocks of strong RED (class 0) / GREEN (class 1).

    Receives BGR arrays, as ultralytics does; records what it was given.
    """
    names = {0: "excavator", 1: "pit"}

    def __init__(self): self.calls, self.seen = 0, []

    def predict(self, images, **kw):
        from scipy import ndimage
        self.calls += 1
        self.seen.append((len(images), kw.get("imgsz")))
        out = []
        for im in images:
            xyxy, cls, conf = [], [], []
            for c, ch in ((0, 2), (1, 1)):          # BGR: red=2, green=1
                m = (im[..., ch] > 200) & (im[..., [i for i in range(3) if i != ch]].max(-1) < 100)
                lab, n = ndimage.label(m)
                for sl in ndimage.find_objects(lab):
                    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                    if h * w < 25:
                        continue
                    xyxy.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
                    cls.append(c)
                    conf.append(0.9)
            out.append(_R(_Boxes(np.array(xyxy, np.float32).reshape(-1, 4),
                                 np.array(cls, np.float32), np.array(conf, np.float32))))
        return out


def _mosaic(path, objs, size=(900, 1300), crs="EPSG:32630", nodata=None, dtype="uint8"):
    """RGB GeoTIFF, dark noise background + coloured blocks at known pixels."""
    H, W = size
    rs = np.random.RandomState(1)
    arr = rs.randint(10, 60, (3, H, W)).astype(dtype)
    for (x0, y0, x1, y1, cls) in objs:
        arr[:, y0:y1, x0:x1] = 0
        arr[0 if cls == 0 else 1, y0:y1, x0:x1] = 255     # red or green in RGB
    prof = {"driver": "GTiff", "height": H, "width": W, "count": 3, "dtype": dtype,
            "crs": crs, "transform": from_origin(754940.0, 704000.0, 10.0, 10.0)}
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)
    return path


def _geo(model=None, **kw):
    """GeoYOLO around the fake model. The synthetic mosaics are already 0-255
    so stretch defaults to 'none' here (an explicit stretch= still wins)."""
    from functools import partial
    g = GeoYOLO(**kw)
    g._model = model or FakeModel()
    g.detect_tiled = partial(g.detect_tiled, stretch="none")
    g.detect = partial(g.detect, stretch="none")
    return g


class TestDetectTiledLogic:
    # one block fully inside a tile, one straddling the x seam (tile stride
    # is 512 at 640/0.2 -> seam region x in [512, 640]), one near the bottom
    # edge, one of the other class.
    OBJS = [(100, 100, 150, 140, 0), (490, 300, 560, 360, 0),
            (600, 700, 660, 760, 1), (1200, 850, 1250, 890, 0)]

    def test_finds_each_object_exactly_once_with_exact_boxes(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", self.OBJS)
        g = _geo()
        r = g.detect_tiled(str(p), tile_size=640, overlap=0.2, conf=0.1)
        assert r["n_detections"] == len(self.OBJS), r["detections"]
        got = sorted((d["bbox_px"], d["class"]) for d in r["detections"])
        want = sorted(([x0, y0, x1, y1], "excavator" if c == 0 else "pit")
                      for x0, y0, x1, y1, c in self.OBJS)
        assert got == want

    def test_geo_coords_match_affine(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", self.OBJS)
        r = _geo().detect_tiled(str(p))
        d = next(x for x in r["detections"] if x["bbox_px"] == [100, 100, 150, 140])
        # origin (754940, 704000), 10 m px, y down
        assert d["bbox_geo"] == [754940 + 1000, 704000 - 1400, 754940 + 1500, 704000 - 1000]
        assert d["area_m2"] == pytest.approx(500 * 400)
        lon, lat = d["center_lonlat"]
        assert -0.8 < lon < -0.2 and 6.0 < lat < 6.9

    def test_seam_object_is_not_split_or_duplicated(self, tmp_path):
        # the 70-px block (490..560) crosses the first x seam
        p = _mosaic(tmp_path / "m.tif", [(490, 300, 560, 360, 0)])
        r = _geo().detect_tiled(str(p), tile_size=640, overlap=0.2)
        assert r["n_detections"] == 1
        assert r["detections"][0]["bbox_px"] == [490, 300, 560, 360]

    def test_model_receives_bgr_and_requested_imgsz(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", [(100, 100, 150, 140, 0)])
        m = FakeModel()
        r = _geo(m).detect_tiled(str(p), tile_size=640)
        # a RED block in the file is class 0 only if the model got BGR
        assert r["detections"][0]["class_id"] == 0
        assert all(imgsz == 640 for _, imgsz in m.seen)

    def test_single_pass_detect_also_bgr_and_geo(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", [(100, 100, 150, 140, 0)], size=(300, 300))
        r = _geo().detect(str(p))
        assert r["n_detections"] == 1 and r["detections"][0]["class"] == "excavator"
        assert r["detections"][0]["bbox_geo"][0] == 754940 + 1000

    def test_batching(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", self.OBJS)
        m = FakeModel()
        _geo(m).detect_tiled(str(p), batch_size=4)
        assert max(n for n, _ in m.seen) <= 4 and sum(n for n, _ in m.seen) == 6

    def test_empty_tiles_skipped(self, tmp_path):
        p = tmp_path / "m.tif"
        H, W = 900, 1300
        arr = np.zeros((3, H, W), np.uint8)               # mosaic border = zeros
        arr[:, 0:300, 0:400] = np.random.RandomState(0).randint(10, 60, (3, 300, 400))
        arr[0, 100:140, 100:150] = 255
        with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=3, dtype="uint8",
                           crs="EPSG:32630", transform=from_origin(754940.0, 704000.0, 10, 10)) as d:
            d.write(arr)
        m = FakeModel()
        r = _geo(m).detect_tiled(str(p))
        assert r["n_tiles_skipped_empty"] > 0 and r["n_detections"] >= 1

    def test_geojson_output(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", self.OBJS)
        out = tmp_path / "d.geojson"
        r = _geo().detect_tiled(str(p), output_path=str(out))
        fc = json.loads(out.read_text())
        assert len(fc["features"]) == r["n_detections"] == len(self.OBJS)
        assert fc["features"][0]["properties"]["class"] in ("excavator", "pit")

    def test_no_crs_geojson_raises_but_pixels_work(self, tmp_path):
        p = tmp_path / "n.tif"
        with rasterio.open(p, "w", driver="GTiff", height=200, width=200, count=3, dtype="uint8") as d:
            a = np.full((3, 200, 200), 20, np.uint8)
            a[0, 50:90, 50:90] = 255
            d.write(a)
        r = _geo().detect_tiled(str(p))
        assert r["n_detections"] == 1 and r["detections"][0]["bbox_lonlat"] is None
        with pytest.raises(ValueError, match="no CRS"):
            _geo().detect_tiled(str(p), output_path=str(tmp_path / "x.geojson"))

    def test_uint16_data_gets_one_global_stretch(self, tmp_path):
        # a bright block in a uint16 image must still be found: needs a stretch
        p = tmp_path / "u.tif"
        a = np.random.RandomState(0).randint(500, 1500, (3, 300, 300)).astype(np.uint16)
        a[:, 100:140, 100:150] = 0
        a[0, 100:140, 100:150] = 9000
        with rasterio.open(p, "w", driver="GTiff", height=300, width=300, count=3, dtype="uint16",
                           crs="EPSG:32630", transform=from_origin(0, 3000, 10, 10)) as d:
            d.write(a)
        r = _geo().detect_tiled(str(p), stretch=(500, 9000))
        assert r["n_detections"] == 1

    def test_in_memory_array_matches_file(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", self.OBJS)
        with rasterio.open(p) as src:
            arr, tr, crs = src.read(), src.transform, src.crs
        from_file = _geo().detect_tiled(str(p))
        from_arr = _geo().detect_tiled(arr, transform=tr, crs=crs)
        assert from_arr["image_path"] is None and from_arr["crs"] == "EPSG:32630"
        key = lambda r: sorted((d["bbox_px"], d["class"], d["bbox_geo"]) for d in r["detections"])  # noqa: E731
        assert key(from_arr) == key(from_file)

    def test_array_float_reflectance_with_nan_and_fixed_stretch(self, tmp_path):
        # the mining project hands over float 0-1 reflectance with NaN gaps
        rs = np.random.RandomState(0)
        arr = (rs.rand(3, 700, 900) * 0.2).astype(np.float32)
        arr[:, 100:140, 100:150] = 0
        arr[0, 100:140, 100:150] = 1.0
        arr[:, 0:50, 0:50] = np.nan
        r = _geo().detect_tiled(arr, stretch=(0.0, 1.0), transform=from_origin(0, 7000, 10, 10),
                                crs="EPSG:32630", skip_empty=True)
        assert r["n_detections"] == 1 and r["detections"][0]["bbox_px"] == [100, 100, 150, 140]

    def test_array_without_crs_is_pixel_only(self):
        arr = np.full((3, 200, 200), 20, np.uint8)
        arr[0, 50:90, 50:90] = 255
        r = _geo().detect_tiled(arr)
        assert r["n_detections"] == 1 and r["crs"] is None
        assert r["detections"][0]["bbox_lonlat"] is None

    def test_validation(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", [], size=(100, 100))
        with pytest.raises(ValueError):
            _geo().detect_tiled(str(p), overlap=1.0)
        with pytest.raises(ValueError):
            _geo().detect_tiled(str(p), stretch="bogus")

    def test_coco_filter_keeps_legacy_semantics(self, tmp_path):
        p = _mosaic(tmp_path / "m.tif", self.OBJS)
        g = _geo(num_classes=1, class_names=["thing"], coco_class_filter={1})
        r = g.detect_tiled(str(p))
        assert r["n_detections"] == 1 and r["detections"][0]["class"] == "thing"


class TestLabels:
    def test_names_from_model_by_default_and_override(self):
        assert _geo()._label(1) == "pit"
        assert _geo(class_names=["x", "y"])._label(1) == "y"

    def test_unknown_class_falls_back(self):
        assert _geo()._label(7) == "truck"      # COCO fallback, legacy behaviour


# ------------------------------------------- real ultralytics (offline, plumbing)

ultralytics = pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")


def _make_ds(root: Path, empty_labels=False):
    import yaml
    rs = np.random.RandomState(0)
    for split, n in (("train", 12), ("val", 4)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            im = (rs.rand(128, 128, 3) * 60).astype("uint8")
            lines = []
            w, h = rs.randint(30, 50, 2)
            x, y = rs.randint(0, 128 - w), rs.randint(0, 128 - h)
            im[y:y + h, x:x + w] = (200, 40, 40)
            lines.append(f"0 {(x + w / 2) / 128:.5f} {(y + h / 2) / 128:.5f} {w / 128:.5f} {h / 128:.5f}")
            cv2.imwrite(str(root / split / "images" / f"{i}.jpg"), im)
            (root / split / "labels" / f"{i}.txt").write_text("" if empty_labels else "\n".join(lines))
    yaml.dump({"path": str(root), "train": "train/images", "val": "val/images", "names": {0: "a"}},
              (root / "d.yaml").open("w"))
    return str(root / "d.yaml")


_FIT = {"imgsz": 128, "batch": 4, "workers": 0, "plots": False, "amp": False,
        "pretrained": False, "verbose": False}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("yolo")
    data = _make_ds(root / "ds")
    g = GeoYOLO(weights="yolov8n.yaml")
    res = g.fit(data, epochs=2, project=str(root / "runs"), name="t", **_FIT)
    return {"root": root, "data": data, "res": res, "g": g}


class TestRealPlumbing:
    def test_fit_returns_existing_best_and_sets_attrs(self, trained):
        r = trained["res"]
        assert Path(r["best_weights"]).exists() and Path(r["last_weights"]).exists()
        assert trained["g"].best_weights == r["best_weights"]
        assert "metrics/mAP50(B)" in r["metrics"]
        assert not r["resumed"] and not r["already_finished"]

    def test_weights_load_names_from_checkpoint(self, trained):
        g = GeoYOLO(weights=trained["res"]["best_weights"])
        assert g.names == {0: "a"}
        assert g._label(0) == "a"

    def test_missing_weights_path_raises(self):
        with pytest.raises(FileNotFoundError):
            GeoYOLO(weights="/no/such/dir/best.pt")

    def test_detect_tiled_real_model_runs_end_to_end(self, trained, tmp_path):
        # untrained net: we assert the pipeline runs and returns well-formed
        # geo-referenced output, NOT that it finds anything.
        p = _mosaic(tmp_path / "m.tif", [(100, 100, 150, 140, 0)], size=(300, 400))
        g = GeoYOLO(weights=trained["res"]["best_weights"])
        r = g.detect_tiled(str(p), tile_size=128, overlap=0.25, conf=0.001, device="cpu",
                           output_path=str(tmp_path / "o.geojson"))
        assert r["n_tiles"] >= 6 and r["crs"] == "EPSG:32630"
        for d in r["detections"]:
            x1, y1, x2, y2 = d["bbox_px"]
            assert 0 <= x1 <= x2 <= 400 and 0 <= y1 <= y2 <= 300
            assert d["class"] == "a"
        assert json.loads((tmp_path / "o.geojson").read_text())["type"] == "FeatureCollection"

    def test_evaluate_returns_metrics(self, trained):
        g = GeoYOLO(weights=trained["res"]["best_weights"])
        m = g.evaluate(trained["data"], split="val", imgsz=128, batch=4, device="cpu", workers=0)
        assert set(m) == {"map50", "map50_95", "precision", "recall"}
        assert all(0.0 <= v <= 1.0 for v in m.values())

    def test_resume_finished_run_is_noop(self, trained):
        g = GeoYOLO(weights="yolov8n.yaml")
        r = g.fit(trained["data"], epochs=2, project=str(trained["root"] / "runs"), name="t",
                  resume=True, **_FIT)
        assert r["already_finished"] and not r["resumed"]
        assert Path(r["best_weights"]).exists()

    def test_resume_interrupted_run_continues_and_completes(self, tmp_path):
        import torch
        data = _make_ds(tmp_path / "ds")
        proj = str(tmp_path / "runs")

        class BoomError(Exception):
            pass

        def crash(trainer):
            if trainer.epoch == 0:
                raise BoomError

        g1 = GeoYOLO(weights="yolov8n.yaml")
        g1._load()
        g1._model.add_callback("on_model_save", crash)
        with pytest.raises(BoomError):
            g1.fit(data, epochs=3, project=proj, name="r", **_FIT)
        last = Path(proj) / "r" / "weights" / "last.pt"
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
        assert ck["epoch"] == 0 and ck["optimizer"] is not None

        g2 = GeoYOLO(weights="yolov8n.yaml")
        r = g2.fit(data, epochs=3, project=proj, name="r", resume=True, **_FIT)
        assert r["resumed"] and not r["already_finished"]
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
        assert ck["epoch"] == -1                      # finished + stripped
        rows = (Path(proj) / "r" / "results.csv").read_text().strip().splitlines()
        assert len(rows) - 1 == 3                      # exactly 3 epochs, none repeated

    def test_fit_refuses_empty_labels_before_training(self, tmp_path):
        data = _make_ds(tmp_path / "ds", empty_labels=True)
        assert _count_train_labels(data) == 0
        g = GeoYOLO(weights="yolov8n.yaml")
        with pytest.raises(ValueError, match="non-empty label"):
            g.fit(data, epochs=1, project=str(tmp_path / "runs"), **_FIT)
        assert g._model is None                        # nothing was even built

    def test_label_counter_counts_nonempty(self, trained):
        assert _count_train_labels(trained["data"]) == 12
