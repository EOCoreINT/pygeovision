"""Regression tests for the Label Studio integration (offline: the HTTP API is faked).

Real defects fixed:
  1. Brush ("segmentation") annotations -- the default task type -- were decoded as zlib/base64 bytes, which is
     not Label Studio's RLE. The decode error was swallowed, so every painted tile became an all-background
     label reported as a success.
  2. Any parse failure, an unsupported result type, or a label missing from ``class_names`` (silently mapped to
     class 1) produced a half-empty mask instead of an error.
  3. Tasks were matched with ``tile_path.stem in image_url``: tile ``t_1`` received the annotations of ``t_10``.
  4. Only the first 1000 tasks of a project were ever fetched, and fetched again for every tile.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
pytest.importorskip("PIL")

from pygeovision.ai.labeling.label_studio import LabelStudioLabeler, decode_ls_rle, encode_ls_rle  # noqa: E402


def _tile(path, size=16):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3, dtype="uint8",
                       crs="EPSG:32632", transform=from_origin(500000, 5500000, 1, 1)) as d:
        d.write(np.random.randint(1, 255, (3, size, size)).astype("uint8"))
    return path


def _brush_result(mask, label, size=None):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), "uint8")
    rgba[:, :, 3] = np.where(mask > 0, 255, 0)
    return {"type": "brushlabels", "original_width": w, "original_height": h,
            "value": {"format": "rle", "rle": encode_ls_rle(rgba.ravel()), "brushlabels": [label]}}


def _lab(tmp_path, **kw):
    return LabelStudioLabeler(api_key="k", project_id=1, class_names=["background", "building", "road"],
                              tile_export_dir=tmp_path / "exp", **kw)


def test_rle_round_trip_including_long_runs():
    rng = np.random.default_rng(0)
    for arr in (rng.integers(0, 2, 1000, dtype=np.uint8) * 255, np.zeros(200_000, np.uint8),
                np.concatenate([np.zeros(70_000, np.uint8), np.full(5, 255, np.uint8)])):
        assert (decode_ls_rle(encode_ls_rle(arr)) == arr).all()


def test_corrupt_rle_raises():
    with pytest.raises(ValueError):
        decode_ls_rle([0, 0, 0, 5])


def test_brush_annotation_becomes_the_painted_mask(tmp_path):
    painted = np.zeros((16, 16), "uint8")
    painted[2:9, 3:10] = 1
    lab = _lab(tmp_path)
    mask = lab._annotation_to_mask({"result": [_brush_result(painted, "building")]}, 16, 16, "segmentation")
    assert (mask == painted).all() and set(np.unique(mask)) == {0, 1}


def test_brush_is_resized_when_the_annotated_image_had_another_size(tmp_path):
    painted = np.zeros((8, 8), "uint8")
    painted[:4, :4] = 1
    lab = _lab(tmp_path)
    mask = lab._annotation_to_mask({"result": [_brush_result(painted, "road")]}, 16, 16, "segmentation")
    assert (mask[:8, :8] == 2).all() and (mask[8:, :] == 0).all() and (mask[:, 8:] == 0).all()


def test_later_results_overwrite_earlier_ones_and_labels_map_to_class_names(tmp_path):
    a = np.zeros((16, 16), "uint8"); a[:8] = 1
    b = np.zeros((16, 16), "uint8"); b[4:12] = 1
    lab = _lab(tmp_path)
    mask = lab._annotation_to_mask({"result": [_brush_result(a, "building"), _brush_result(b, "road")]},
                                   16, 16, "segmentation")
    assert (mask[:4] == 1).all() and (mask[4:12] == 2).all() and (mask[12:] == 0).all()


def test_unknown_label_and_unsupported_type_are_errors(tmp_path):
    from pygeovision.core.exceptions import LabelingError
    lab = _lab(tmp_path)
    painted = np.ones((16, 16), "uint8")
    with pytest.raises(LabelingError, match="not in class_names"):
        lab._annotation_to_mask({"result": [_brush_result(painted, "tree")]}, 16, 16, "segmentation")
    with pytest.raises(LabelingError, match="unsupported"):
        lab._annotation_to_mask({"result": [{"type": "keypointlabels", "value": {}}]}, 16, 16, "segmentation")
    bad = _brush_result(painted, "building")
    bad["original_width"] = 99
    with pytest.raises(LabelingError, match="decodes to"):
        lab._annotation_to_mask({"result": [bad]}, 16, 16, "segmentation")


def test_polygon_and_rectangle_use_percent_coordinates(tmp_path):
    lab = _lab(tmp_path)
    ann = {"result": [
        {"type": "rectanglelabels", "value": {"x": 0, "y": 0, "width": 50, "height": 50, "rectanglelabels": ["building"]}},
        {"type": "polygonlabels", "value": {"points": [[50, 50], [100, 50], [100, 100], [50, 100]], "polygonlabels": ["road"]}},
    ]}
    mask = lab._annotation_to_mask(ann, 16, 16, "polygon")
    assert (mask[:8, :8] == 1).all() and (mask[9:, 9:] == 2).all() and mask[0, 15] == 0


def test_label_tile_writes_the_mask_or_reports_an_error_without_a_file(tmp_path, monkeypatch):
    tile = _tile(tmp_path / "t_1.tif")
    painted = np.zeros((16, 16), "uint8"); painted[:8, :8] = 1
    lab = _lab(tmp_path)
    task = {"id": 5, "meta": {"tile_path": str(tile)}, "annotations": [{"id": 9, "result": [_brush_result(painted, "building")]}]}
    monkeypatch.setattr(lab, "_api_get", lambda ep: [task])
    out = tmp_path / "lab.tif"
    r = lab.label_tile(tile, None, out)
    assert r.success
    with rasterio.open(out) as o:
        assert (o.read(1)[:8, :8] == 1).all() and o.crs == rasterio.open(tile).crs

    corrupt = {"id": 5, "meta": {"tile_path": str(tile)},
               "annotations": [{"id": 9, "result": [{"type": "brushlabels", "original_width": 16, "original_height": 16,
                                                     "value": {"rle": [1, 2, 3], "brushlabels": ["building"]}}]}]}
    lab2 = _lab(tmp_path)
    monkeypatch.setattr(lab2, "_api_get", lambda ep: [corrupt])
    out2 = tmp_path / "lab2.tif"
    r = lab2.label_tile(tile, None, out2)
    assert not r.success and r.error and not out2.exists()


def test_task_matching_is_exact_not_substring(tmp_path, monkeypatch):
    t1, t10 = _tile(tmp_path / "t_1.tif"), _tile(tmp_path / "t_10.tif")
    lab = _lab(tmp_path)
    tasks = [{"id": 10, "data": {"image": "http://ls/data/t_10.png"}},
             {"id": 1, "data": {"image": "http://ls/data/t_1.png?x=1"}}]
    monkeypatch.setattr(lab, "_api_get", lambda ep: tasks)
    assert lab._find_task_for_tile(t1)["id"] == 1
    assert lab._find_task_for_tile(t10)["id"] == 10
    assert lab._find_task_for_tile(tmp_path / "t_100.tif") is None


def test_tasks_are_paginated_and_fetched_once_per_run(tmp_path, monkeypatch):
    lab = _lab(tmp_path)
    lab._PAGE = 3
    calls = []
    everything = [{"id": i, "data": {"image": f"x/t_{i}.png"}} for i in range(1, 8)]

    def fake(ep):
        calls.append(ep)
        page = int(ep.split("page=")[1].split("&")[0])
        return everything[(page - 1) * 3: page * 3]
    monkeypatch.setattr(lab, "_api_get", fake)
    assert lab._find_task_for_tile(tmp_path / "t_7.tif")["id"] == 7
    assert lab._find_task_for_tile(tmp_path / "t_2.tif")["id"] == 2
    assert len(calls) == 3                                           # pages 1-3, then cached


def test_server_that_ignores_page_does_not_loop_forever(tmp_path, monkeypatch):
    lab = _lab(tmp_path)
    lab._PAGE = 2
    monkeypatch.setattr(lab, "_api_get", lambda ep: [{"id": 1, "data": {}}, {"id": 2, "data": {}}])
    assert len(lab._all_tasks()) == 2


def test_no_task_and_no_annotation_are_distinct_errors(tmp_path, monkeypatch):
    tile = _tile(tmp_path / "t_1.tif")
    lab = _lab(tmp_path)
    monkeypatch.setattr(lab, "_api_get", lambda ep: [])
    assert lab.label_tile(tile, None, tmp_path / "a.tif").error == "no_task_found"
    lab = _lab(tmp_path)
    monkeypatch.setattr(lab, "_api_get", lambda ep: [{"id": 1, "meta": {"tile_path": str(tile)}, "annotations": []}])
    assert lab.label_tile(tile, None, tmp_path / "b.tif").error == "no_annotations"
    assert not (tmp_path / "a.tif").exists() and not (tmp_path / "b.tif").exists()
