"""SAMAutoLabeler end-to-end on a tiny random SAM (no weights, no network).

A random SAM produces meaningless masks, so these tests do not judge segmentation quality;
they pin the *plumbing* the old implementation got wrong: arguments reaching the pipeline,
tile coverage, scene coordinates, area units, failure reporting, GeoJSON CRS, and the
GroundingDINO box convention.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
rasterio = pytest.importorskip("rasterio")

import transformers as T  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.labeling import sam_auto  # noqa: E402
from pygeovision.labeling.sam_auto import SAMAutoLabeler, _pixel_area_m2, _tile_origins  # noqa: E402


def _tiny_sam():
    vc = T.SamVisionConfig(hidden_size=32, output_channels=16, num_hidden_layers=2, num_attention_heads=2,
                           mlp_dim=64, image_size=64, patch_size=16, window_size=2,
                           global_attn_indexes=[1], num_pos_feats=8)
    pc = T.SamPromptEncoderConfig(hidden_size=16, image_size=64, patch_size=16)
    mc = T.SamMaskDecoderConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                mlp_dim=32, iou_head_hidden_dim=16)
    torch.manual_seed(0)
    return T.SamModel(T.SamConfig(vision_config=vc, prompt_encoder_config=pc, mask_decoder_config=mc)).eval()


def _labeler():
    lab = SAMAutoLabeler(model="sam-vit-base", device="cpu")
    lab._model = _tiny_sam()
    lab._processor = T.SamProcessor(
        image_processor=T.SamImageProcessor(size={"longest_edge": 64}, pad_size={"height": 64, "width": 64}))
    return lab


def _write_tif(path, h=96, w=96, crs="EPSG:32633", res=10.0, count=3):
    arr = (np.random.default_rng(0).random((count, h, w)) * 1000).astype("float32")
    tr = from_origin(500000, 4000000, res, res) if crs != "EPSG:4326" else from_origin(15.0, 48.0, 1e-4, 1e-4)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=count, dtype="float32",
                       crs=crs, transform=tr) as dst:
        dst.write(arr)
    return path


class TestTiling:
    def test_covers_scene_and_ends_exactly_at_the_edge(self):
        for size, chip, ov in [(100, 64, 16), (1024, 1024, 128), (2500, 1024, 128), (64, 64, 0)]:
            starts = _tile_origins(size, chip, ov)
            assert starts[0] == 0 and starts[-1] + chip >= size
            covered = np.zeros(size, bool)
            for s in starts:
                covered[s:s + chip] = True
            assert covered.all()

    def test_no_wholly_redundant_tile(self):
        # the old range(0, size, stride) added a 128-px sliver at 896 for a 1024 scene
        assert _tile_origins(1024, 1024, 128) == [0]

    @pytest.mark.parametrize("chip,overlap", [(32, 0), (128, 128), (128, 200)])
    def test_bad_geometry_is_rejected(self, tmp_path, chip, overlap):
        with pytest.raises(ValueError):
            _labeler().auto_label(_write_tif(tmp_path / "a.tif"), tmp_path / "o.tif",
                                  chip_size=chip, overlap=overlap)


class TestPixelArea:
    def test_projected_crs_uses_the_transform(self):
        assert _pixel_area_m2(from_origin(0, 0, 10, 10), CRS.from_epsg(32633), 100) == pytest.approx(100.0)

    def test_geographic_crs_is_converted_from_degrees(self):
        # 1e-4 deg at 48 N is ~7.45 m x 11.13 m.  The old |a*e| gave 1e-8 "m2".
        area = _pixel_area_m2(from_origin(15.0, 48.0, 1e-4, 1e-4), CRS.from_epsg(4326), 100)
        assert 70 < area < 90


class TestAutoLabel:
    def test_arguments_reach_the_pipeline(self, tmp_path, monkeypatch):
        lab = _labeler()
        seen = {}

        def fake_generator(img, **kw):
            seen.update(kw)
            return {"masks": [], "scores": []}

        monkeypatch.setattr(lab, "_load_mask_generator", lambda: fake_generator)
        lab.auto_label(_write_tif(tmp_path / "a.tif"), tmp_path / "o.tif", points_per_side=7,
                       pred_iou_thresh=0.5, stability_score_thresh=0.6, chip_size=64, overlap=0)
        assert seen["points_per_crop"] == 7
        assert seen["pred_iou_thresh"] == 0.5 and seen["stability_score_thresh"] == 0.6

    def test_iou_score_is_not_refiltered_against_the_stability_threshold(self, tmp_path, monkeypatch):
        lab = _labeler()
        seg = np.zeros((64, 64), bool)
        seg[10:30, 10:30] = True
        monkeypatch.setattr(lab, "_load_mask_generator",
                            lambda: (lambda img, **kw: {"masks": [torch.from_numpy(seg)], "scores": [0.9]}))
        r = lab.auto_label(_write_tif(tmp_path / "a.tif", 64, 64), tmp_path / "o.tif",
                           pred_iou_thresh=0.8, stability_score_thresh=0.95, chip_size=64, overlap=0,
                           min_area_m2=1)
        assert r["success"] and r["n_masks"] == 1   # old code dropped it: 0.9 < 0.95

    def test_masks_are_placed_in_scene_coordinates_across_tiles(self, tmp_path, monkeypatch):
        lab = _labeler()

        def gen(img, **kw):
            seg = np.zeros(img.size[::-1], bool)
            seg[5:15, 5:15] = True          # always at tile-local (5..15)
            return {"masks": [torch.from_numpy(seg)], "scores": [0.99]}

        monkeypatch.setattr(lab, "_load_mask_generator", lambda: gen)
        r = lab.auto_label(_write_tif(tmp_path / "a.tif", 96, 96), tmp_path / "o.tif",
                           chip_size=64, overlap=32, min_area_m2=1, merge_overlapping=False)
        assert r["n_tiles"] == 4 and r["n_masks"] == 4
        with rasterio.open(tmp_path / "o.tif") as src:
            lbl = src.read(1)
        # tile origins are 0 and 32 on each axis
        for r0 in (0, 32):
            for c0 in (0, 32):
                assert lbl[r0 + 5:r0 + 15, c0 + 5:c0 + 15].max() > 0

    def test_small_object_inside_a_large_one_keeps_its_own_id(self, tmp_path, monkeypatch):
        lab = _labeler()
        big = np.zeros((64, 64), bool); big[:, :] = True
        small = np.zeros((64, 64), bool); small[20:30, 20:30] = True
        # small is reported FIRST, big second: in report order big would overwrite small
        monkeypatch.setattr(lab, "_load_mask_generator", lambda: (
            lambda img, **kw: {"masks": [torch.from_numpy(small), torch.from_numpy(big)], "scores": [0.9, 0.9]}))
        lab.auto_label(_write_tif(tmp_path / "a.tif", 64, 64), tmp_path / "o.tif", chip_size=64, overlap=0,
                       min_area_m2=1, merge_overlapping=False)
        with rasterio.open(tmp_path / "o.tif") as src:
            lbl = src.read(1)
        assert lbl[25, 25] != lbl[2, 2] and lbl[25, 25] > 0

    def test_geographic_scene_uses_real_area_for_the_filter(self, tmp_path, monkeypatch):
        lab = _labeler()
        seg = np.zeros((64, 64), bool); seg[:10, :10] = True   # 100 px ~ 8 000 m2
        monkeypatch.setattr(lab, "_load_mask_generator",
                            lambda: (lambda img, **kw: {"masks": [torch.from_numpy(seg)], "scores": [0.9]}))
        kw = dict(chip_size=64, overlap=0, merge_overlapping=False)
        tif = _write_tif(tmp_path / "g.tif", 64, 64, crs="EPSG:4326")
        assert lab.auto_label(tif, tmp_path / "a.tif", min_area_m2=1000, **kw)["n_masks"] == 1
        # with deg^2 treated as m2 (old) the area would be ~1e-6 and this would also pass the
        # *small* threshold; the large one is what distinguishes real from fake units
        assert lab.auto_label(tif, tmp_path / "b.tif", min_area_m2=20000, **kw)["n_masks"] == 0

    def test_failed_tiles_are_reported_not_swallowed(self, tmp_path, monkeypatch):
        lab = _labeler()

        def boom(img, **kw):
            raise RuntimeError("cuda out of memory")

        monkeypatch.setattr(lab, "_load_mask_generator", lambda: boom)
        r = lab.auto_label(_write_tif(tmp_path / "a.tif"), tmp_path / "o.tif", chip_size=64, overlap=0)
        assert r["success"] is False
        assert r["n_chip_errors"] == r["n_tiles"] == 4
        assert "cuda out of memory" in r["error"]

    def test_runs_with_a_real_sam_graph(self, tmp_path):
        """Real transformers mask-generation pipeline + a tiny random SAM, no monkeypatching."""
        lab = _labeler()
        r = lab.auto_label(_write_tif(tmp_path / "a.tif", 64, 64), tmp_path / "o.tif", chip_size=64,
                           overlap=0, points_per_side=3, pred_iou_thresh=-1.0, stability_score_thresh=-1.0,
                           min_area_m2=0)
        assert r["n_chip_errors"] == 0, r["error"]
        assert r["n_tiles"] == 1 and (tmp_path / "o.tif").exists()

    def test_vector_export_is_wgs84(self, tmp_path, monkeypatch):
        lab = _labeler()
        seg = np.zeros((64, 64), bool); seg[10:30, 10:30] = True
        monkeypatch.setattr(lab, "_load_mask_generator",
                            lambda: (lambda img, **kw: {"masks": [torch.from_numpy(seg)], "scores": [0.9]}))
        r = lab.auto_label(_write_tif(tmp_path / "a.tif", 64, 64), tmp_path / "o.tif",
                           output_vector=str(tmp_path / "v.geojson"), chip_size=64, overlap=0, min_area_m2=1)
        assert r["vector_error"] is None
        feats = json.loads((tmp_path / "v.geojson").read_text())["features"]
        lon, lat = feats[0]["geometry"]["coordinates"][0][0]
        assert 10 < lon < 20 and 30 < lat < 40       # UTM 33N (500000, 4000000) is ~ 15E, 36N

    def test_rgb_band_order_is_respected(self, tmp_path):
        tif = tmp_path / "a.tif"
        arr = np.zeros((3, 8, 8), "float32"); arr[0] = 100; arr[1] = 0; arr[2] = 0
        arr[0, :4] = 0; arr[0, 4:] = 100          # band 1 has structure
        with rasterio.open(tif, "w", driver="GTiff", height=8, width=8, count=3, dtype="float32") as dst:
            dst.write(arr)
        with rasterio.open(tif) as src:
            a = SAMAutoLabeler._read_rgb(src, None)
            b = SAMAutoLabeler._read_rgb(src, (3, 2, 1))
        assert a[..., 0].max() == 255 and b[..., 2].max() == 255 and b[..., 0].max() == 0
        with rasterio.open(tif) as src, pytest.raises(ValueError):
            SAMAutoLabeler._read_rgb(src, (1, 2, 9))


class TestModelNames:
    def test_sam2_ids_are_transformers_conversions(self):
        assert SAMAutoLabeler.HF_MODELS["sam2-hiera-l"].endswith("-hf")
        assert SAMAutoLabeler.HF_MODELS["sam2-hiera-b"].endswith("-hf")

    def test_nonexistent_geosam_is_refused_up_front(self):
        with pytest.raises(ValueError, match="not available"):
            SAMAutoLabeler(model="geosam-vit-h")


class TestGroundingDinoConventions:
    def test_normalised_cxcywh_becomes_pixel_xyxy(self):
        out = SAMAutoLabeler._gdino_boxes_to_pixels(torch.tensor([[0.5, 0.5, 0.2, 0.4]]), width=200, height=100)
        np.testing.assert_allclose(out, [[80, 30, 120, 70]])

    def test_segment_box_returns_one_hw_mask(self):
        lab = _labeler()
        rgb = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype("uint8")
        m = lab._segment_box(rgb, [10, 10, 40, 40])
        assert m.shape == (64, 64) and m.dtype == bool

    def test_grounded_label_wiring_with_a_stand_in_detector(self, tmp_path, monkeypatch):
        """No groundingdino package here, so a stand-in checks only what *we* do around it:
        tensor input, box-format conversion, one class id per prompt."""
        import sys
        import types

        calls = []

        def predict(model, image, caption, box_threshold, text_threshold, device="cpu"):
            calls.append((image, caption))
            return torch.tensor([[0.5, 0.5, 0.5, 0.5]]), torch.tensor([0.9]), [caption]

        pkg = types.ModuleType("groundingdino")
        for name in ("groundingdino.util", "groundingdino.util.inference", "groundingdino.datasets",
                     "groundingdino.datasets.transforms"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "groundingdino", pkg)
        sys.modules["groundingdino.util.inference"].predict = predict
        lab = _labeler()
        monkeypatch.setattr(lab, "_load_gdino", lambda: object())
        monkeypatch.setattr(lab, "_gdino_image_tensor", staticmethod(lambda rgb: torch.zeros(3, 8, 8)))
        seen_boxes = []
        monkeypatch.setattr(lab, "_segment_box", lambda rgb, box: (seen_boxes.append(list(box)),
                                                                   np.pad(np.ones((8, 8), bool), 4))[1])
        r = lab.grounded_label(_write_tif(tmp_path / "a.tif", 16, 16), ["building", "pool"], tmp_path / "o.tif")
        assert r["success"] and r["n_masks_applied"] == 2
        assert all(isinstance(c[0], torch.Tensor) for c in calls)        # not a PIL image
        assert seen_boxes[0] == [4.0, 4.0, 12.0, 12.0]                   # 0.5/0.5 w,h 0.5 on 16 px
        with rasterio.open(tmp_path / "o.tif") as src:
            assert set(np.unique(src.read(1))) == {0, 2}                 # later prompt painted last


def test_module_exports_helpers():
    assert hasattr(sam_auto, "_tile_origins") and hasattr(sam_auto, "_pixel_area_m2")
