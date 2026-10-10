"""DINOv3 integration: real repo ids, honest failures, registered features, real CHMv2 wiring.

No network or gated weights are available here, so layout tests use *tiny randomly initialised instances of
the real transformers classes* (DINOv3ViTModel / DINOv3ConvNextModel), and content tests use deterministic
stand-ins. Nothing here proves the pretrained checkpoints behave - only that the code reads them correctly.
"""
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pygeovision.core.exceptions import ModelLoadError  # noqa: E402
from pygeovision.models.foundation import dinov3 as d3  # noqa: E402
from pygeovision.models.foundation.dinov3 import (  # noqa: E402
    CHMv2Model,
    DINOv3Backbone,
    DINOv3Text,
    finetune_dinov3,
    load_dinov3_hf,
    load_dinov3_local,
)

EXPECTED_REPOS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "dinov3_vitl16_sat": "facebook/dinov3-vitl16-pretrain-sat493m",
    "dinov3_vit7b16_sat": "facebook/dinov3-vit7b16-pretrain-sat493m",
    "dinov3_convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "dinov3_convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "dinov3_convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "dinov3_convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}


def _tiny_vit(registers=4):
    cfg = transformers.DINOv3ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                       intermediate_size=64, patch_size=16, num_register_tokens=registers)
    return transformers.DINOv3ViTModel(cfg).eval()


def _tiny_cnx():
    cfg = transformers.DINOv3ConvNextConfig(hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 1, 1])
    return transformers.DINOv3ConvNextModel(cfg).eval()


def _backbone(model, name="dinov3_vits16", **kw):
    b = DINOv3Backbone(name, device="cpu", **kw)
    b._model = model
    return b


def _hwc(h, w, seed=0):
    return np.random.default_rng(seed).random((h, w, 3)).astype("float32")


# ---------------------------------------------------------------------------------------------- registry
class TestRegistry:
    def test_every_variant_points_at_its_real_dinov3_repo(self):
        assert {n: s["hf_id"] for n, s in d3.DINOV3_MODELS.items()} == EXPECTED_REPOS

    def test_no_entry_loads_dinov2(self):
        assert not [n for n, s in d3.DINOV3_MODELS.items() if "dinov2" in s["hf_id"].lower()]

    def test_sat_flags_and_gating(self):
        assert d3.list_satellite_models() == ["dinov3_vitl16_sat", "dinov3_vit7b16_sat"]
        assert all(s["gated"] for s in d3.DINOV3_MODELS.values())
        assert len(d3.list_dinov3_models()) == 12

    def test_info_names_the_right_transform(self):
        assert "ImageNet web stats" in d3.get_dinov3_info("dinov3_vitl16")["transform"]
        assert "satellite stats" in d3.get_dinov3_info("dinov3_vitl16_sat")["transform"]
        with pytest.raises(ValueError):
            d3.get_dinov3_info("nope")

    def test_web_and_sat_statistics_differ(self):
        assert d3.WEB_MEAN != d3.SAT_MEAN and d3.WEB_STD != d3.SAT_STD

    def test_transform_picks_stats_and_shape(self):
        from PIL import Image
        img = Image.fromarray(np.zeros((64, 64, 3), np.uint8))
        out = d3.get_transform("dinov3_vitl16_sat")(img)
        assert out.shape == (3, 224, 224)
        assert float(out[0, 0, 0]) == pytest.approx(-d3.SAT_MEAN[0] / d3.SAT_STD[0], abs=1e-4)


# ---------------------------------------------------------------------------------------------- loading
class TestLoading:
    def test_requests_the_dinov3_repo_and_reports_gating_on_failure(self, monkeypatch):
        seen = []

        def boom(repo, *a, **k):
            seen.append(repo)
            raise OSError("401 gated")

        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", boom)
        with pytest.raises(ModelLoadError) as ei:
            load_dinov3_hf("dinov3_vitl16_sat")
        assert seen == ["facebook/dinov3-vitl16-pretrain-sat493m"]
        assert "gated" in str(ei.value) and "Nothing was substituted" in str(ei.value)

    def test_success_attaches_preprocessor_statistics(self, monkeypatch):
        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: _tiny_vit())
        proc = SimpleNamespace(image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])
        monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: proc)
        m = load_dinov3_hf("dinov3_vits16")
        assert m._pgv_mean == (0.1, 0.2, 0.3) and m._pgv_std == (0.4, 0.5, 0.6) and not m._pgv_is_sat

    def test_missing_preprocessor_falls_back_to_documented_constants(self, monkeypatch):
        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: _tiny_vit())

        def no_proc(*a, **k):
            raise OSError("no preprocessor")

        monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", no_proc)
        assert load_dinov3_hf("dinov3_vitl16_sat")._pgv_mean == d3.SAT_MEAN
        assert load_dinov3_hf("dinov3_vitl16")._pgv_mean == d3.WEB_MEAN

    def test_hub_loading_is_gone(self):
        with pytest.raises(NotImplementedError):
            d3.load_dinov3_hub("dinov3_vitl16")
        with pytest.raises(NotImplementedError):
            DINOv3Backbone("dinov3_vitl16", method="hub", device="cpu")._load()

    def test_local_snapshot_roundtrip_and_family_check(self, tmp_path):
        _tiny_vit().save_pretrained(tmp_path / "vit")
        m = load_dinov3_local("dinov3_vits16", str(tmp_path / "vit"))
        assert isinstance(m, transformers.DINOv3ViTModel)
        with pytest.raises(ModelLoadError, match="needs 'dinov3_convnext'"):
            load_dinov3_local("dinov3_convnext_tiny", str(tmp_path / "vit"))

    def test_local_rejects_loose_weight_files(self, tmp_path):
        f = tmp_path / "w.pth"
        f.write_bytes(b"x")
        with pytest.raises(ValueError, match="snapshot directory"):
            load_dinov3_local("dinov3_vits16", str(f))
        with pytest.raises(ModelLoadError):
            load_dinov3_local("dinov3_vits16", str(tmp_path))     # directory without config.json


# ---------------------------------------------------------------------------------------------- backbone
class TestBackbone:
    def test_init_is_lazy_and_flags_sat(self):
        b = DINOv3Backbone(device="cpu")
        assert b.model_name == "dinov3_vitl16_sat" and b._is_sat and b._model is None
        assert "300" in repr(b) and "SAT-493M" in repr(b)

    def test_vit_features_skip_cls_and_register_tokens(self):
        model = _tiny_vit(registers=4)
        b = _backbone(model)
        img = _hwc(100, 70)
        feats = b.extract_features(img)
        assert feats.shape == (7, 5, 32)                     # 100x70 padded to 112x80 -> 7x5 patches
        # same thing computed by hand, slicing past CLS + 4 registers
        x, gh, gw, _, _ = b._prepare(img)
        with torch.no_grad():
            tok = model(pixel_values=x).last_hidden_state[0]
        assert tok.shape[0] == 1 + 4 + 35
        np.testing.assert_allclose(feats.reshape(35, -1), tok[5:].numpy(), atol=1e-5)

    def test_embedding_is_the_cls_token(self):
        model = _tiny_vit()
        b = _backbone(model)
        img = _hwc(64, 64)
        emb = b.extract_embeddings(img)
        x = b._prepare(img)[0]
        with torch.no_grad():
            cls = model(pixel_values=x).last_hidden_state[0, 0]
        assert emb.shape == (1, 32)
        np.testing.assert_allclose(emb[0], cls.numpy(), atol=1e-5)

    def test_patch_features_are_row_major_flattening(self):
        b = _backbone(_tiny_vit())
        img = _hwc(48, 80)
        np.testing.assert_allclose(b.extract_patch_features(img), b.extract_features(img).reshape(-1, 32))

    def test_convnext_layout(self):
        b = _backbone(_tiny_cnx(), "dinov3_convnext_tiny")
        feats = b.extract_features(_hwc(100, 70))
        assert feats.shape == (4, 3, 64)                     # 128x96 padded / 32
        assert b.extract_embeddings(_hwc(64, 64)).shape == (1, 64)

    def test_geometry_registers_cells_with_source_pixels(self):
        b = _backbone(_tiny_vit(), max_size=64)
        b.extract_features(_hwc(200, 100))
        g = b.last_geometry
        assert g["scale"] == pytest.approx(0.32) and g["cell_px"] == pytest.approx(16 / 0.32)
        assert g["grid"] == (4, 2) and g["source_size"] == (200, 100)

    def test_dense_features_follow_the_image_not_a_centre_crop(self):
        """Brightness stand-in: a bright block in the left column of a wide image must show up in the
        left grid columns (a centre crop would drop it)."""
        class Bright(torch.nn.Module):
            config = SimpleNamespace(patch_size=16, hidden_size=1, num_register_tokens=0)

            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, pixel_values):
                b, _, h, w = pixel_values.shape
                t = pixel_values.mean(1).reshape(b, h // 16, 16, w // 16, 16).mean((2, 4)).reshape(b, -1, 1)
                return SimpleNamespace(last_hidden_state=torch.cat([torch.zeros(b, 1, 1), t], 1))

        img = np.zeros((32, 160, 3), "float32")
        img[:, :32] = 1.0
        f = _backbone(Bright()).extract_features(img)[..., 0]
        assert f.shape == (2, 10)
        assert (f[:, :2] > f[:, 2:].max()).all()

    def test_ragged_edges_are_padded_by_replication(self):
        """A uniform image must give identical features in every cell, including the partly padded ones;
        zero padding would make the edge cells look like a different surface."""
        class Bright(torch.nn.Module):
            config = SimpleNamespace(patch_size=16, hidden_size=1, num_register_tokens=0)

            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, pixel_values):
                b, _, h, w = pixel_values.shape
                t = pixel_values.mean(1).reshape(b, h // 16, 16, w // 16, 16).mean((2, 4)).reshape(b, -1, 1)
                return SimpleNamespace(last_hidden_state=torch.cat([torch.zeros(b, 1, 1), t], 1))

        f = _backbone(Bright()).extract_features(np.ones((20, 20, 3), "float32"))[..., 0]
        assert f.shape == (2, 2) and np.ptp(f) < 1e-6

    def test_uses_statistics_attached_by_the_loader(self):
        model = _tiny_vit()
        b = _backbone(model, "dinov3_vitl16_sat")
        assert b._stats() == (d3.SAT_MEAN, d3.SAT_STD)
        model._pgv_mean, model._pgv_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        assert b._stats() == ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def test_raster_input_is_nodata_aware(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin
        arr = np.random.default_rng(1).random((3, 64, 64)).astype("float32") * 1000
        arr[:, :32] = -9999
        p = tmp_path / "a.tif"
        with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=3, dtype="float32",
                           crs="EPSG:32633", transform=from_origin(0, 640, 10, 10), nodata=-9999) as d:
            d.write(arr)
        rgb, valid = d3._to_rgb01(str(p))
        assert valid[32:].all() and not valid[:32].any()
        assert rgb[:, :32].max() == 0.0 and rgb[:, 32:].max() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="not available"):
            d3._to_rgb01(str(p), (1, 2, 7))

    def test_attention_maps(self):
        cfg = transformers.DINOv3ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                           intermediate_size=64, patch_size=16, num_register_tokens=4,
                                           attn_implementation="eager")
        b = _backbone(transformers.DINOv3ViTModel(cfg).eval())
        m = b.get_attention_maps(_hwc(64, 48), head_idx=1)
        assert m.shape == (4, 3) and (m >= 0).all()
        assert b.get_attention_maps(_hwc(64, 48)).shape == (4, 3)

    def test_attention_never_returns_a_placeholder(self):
        class NoAttn(torch.nn.Module):
            config = SimpleNamespace(patch_size=16, num_register_tokens=0)

            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, pixel_values, output_attentions=False):
                return SimpleNamespace(last_hidden_state=torch.zeros(1, 17, 4), attentions=None)

        with pytest.raises(RuntimeError, match="attention weights"):
            _backbone(NoAttn()).get_attention_maps(_hwc(16, 16))
        with pytest.raises(NotImplementedError):
            _backbone(_tiny_cnx(), "dinov3_convnext_tiny").get_attention_maps(_hwc(32, 32))

    def test_build_classifier_forward_and_freezing(self):
        b = _backbone(_tiny_vit())
        clf = b.build_classifier(5, freeze_backbone=True)
        out = clf(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 5)
        assert all(not p.requires_grad for p in clf.backbone.parameters())
        assert all(p.requires_grad for p in clf.head.parameters())
        cc = _backbone(_tiny_cnx(), "dinov3_convnext_tiny").build_classifier(3)
        assert cc(torch.randn(1, 3, 64, 64)).shape == (1, 3)

    def test_finetune_config(self):
        cfg = DINOv3Backbone(device="cpu").finetune_config()
        assert cfg["optimizer"] == "AdamW" and cfg["weight_decay"] == 0.05


# ---------------------------------------------------------------------------------------------- CHMv2
class FakeCHM(torch.nn.Module):
    """predicted_depth = 10 * mean-channel of the (de-normalised) input: a known per-pixel function."""

    def __init__(self, mean, std):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.mean, self.std = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
        self.shapes = []

    def forward(self, pixel_values):
        self.shapes.append(tuple(pixel_values.shape))
        raw = pixel_values * self.std + self.mean
        return SimpleNamespace(predicted_depth=raw.mean(1) * 10.0)


def _chm(**kw):
    chm = CHMv2Model(device="cpu", **kw)
    chm._model = FakeCHM(chm._mean, chm._std)
    return chm


def _write(path, arr, dtype, nodata=None, crs="EPSG:32633"):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    n, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=n, dtype=dtype, crs=crs,
                       transform=from_origin(500000, 4000000, 10, 10), nodata=nodata) as d:
        d.write(arr.astype(dtype))
    return str(path)


class TestCHMv2:
    def test_uses_the_released_checkpoint_and_no_random_decoder(self):
        assert d3.CHMV2_HF_ID == "facebook/dinov3-vitl16-chmv2-dpt-head"
        chm = CHMv2Model(device="cpu")
        assert chm._model is None and not hasattr(chm, "_build_decoder") and not hasattr(chm, "_decoder")

    def test_load_failure_is_reported_not_faked(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise OSError("gated")

        monkeypatch.setattr(transformers.AutoModelForDepthEstimation, "from_pretrained", boom)
        res = CHMv2Model(device="cpu").predict_canopy_height(
            _write(tmp_path / "a.tif", np.full((3, 32, 32), 100), "uint8"))
        assert "height_map" not in res and "Nothing was substituted" in res["error"]

    def test_missing_file(self):
        assert "error" in CHMv2Model(device="cpu").predict_canopy_height("nonexistent_xyz.tif")

    def test_tiled_prediction_reproduces_a_pixelwise_function_exactly(self, tmp_path):
        """500x300 with 384 tiles / 32 overlap: every pixel must get the value for *its* input pixel, so a
        misplaced or mis-cropped tile shows up as an error."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (3, 300, 500)).astype("uint8")
        chm = _chm()
        res = chm.predict_canopy_height(_write(tmp_path / "a.tif", img, "uint8"), tmp_path / "h.tif")
        expect = img.astype("float32").mean(0) / 255.0 * 10.0
        np.testing.assert_allclose(res["height_map"], expect, atol=1e-4)
        assert all(s[-2] % 16 == 0 and s[-1] % 16 == 0 for s in chm._model.shapes)
        assert len(chm._model.shapes) >= 2
        import rasterio
        with rasterio.open(tmp_path / "h.tif") as src:
            assert src.dtypes[0] == "float32" and src.crs.to_epsg() == 32633
            np.testing.assert_allclose(src.read(1), expect, atol=1e-4)

    def test_nodata_is_nan_and_excluded_from_statistics(self, tmp_path):
        img = np.full((3, 64, 64), 255.0)
        img[:, :32] = 0
        res = _chm().predict_canopy_height(_write(tmp_path / "a.tif", img, "uint8", nodata=0))
        h = res["height_map"]
        assert np.isnan(h[:32]).all() and np.isfinite(h[32:]).all()
        assert res["statistics"]["mean_m"] == pytest.approx(10.0, abs=1e-3)

    def test_non_uint8_rasters_are_stretched_not_clipped(self, tmp_path):
        arr = np.random.default_rng(2).random((3, 48, 48)).astype("float32") * 5000
        res = _chm().predict_canopy_height(_write(tmp_path / "a.tif", arr, "float32"))
        assert 0.0 <= np.nanmin(res["height_map"]) and np.nanmax(res["height_map"]) <= 10.0 + 1e-3
        assert np.nanmax(res["height_map"]) > 9.0          # the 98th-percentile stretch reaches ~full scale

    def test_biomass_is_not_offered(self):
        with pytest.raises(NotImplementedError):
            _chm().estimate_biomass("x.tif")
        assert not hasattr(CHMv2Model, "_BIOMASS_COEF_A")

    def test_deforestation_needs_the_same_grid(self, tmp_path):
        a = _write(tmp_path / "a.tif", np.full((3, 32, 32), 200), "uint8")
        b = _write(tmp_path / "b.tif", np.full((3, 48, 48), 200), "uint8")
        assert "same grid" in _chm().detect_deforestation(a, b)["error"]

    def test_deforestation_mask_and_area_from_the_transform(self, tmp_path):
        before = np.full((3, 32, 32), 255)                        # height 10
        after = before.copy()
        after[:, :16] = 0 + 25                                    # first half drops to ~1 (loss > 2)
        a, b = _write(tmp_path / "a.tif", before, "uint8"), _write(tmp_path / "b.tif", after, "uint8")
        res = _chm().detect_deforestation(a, b, min_height_loss_m=2.0)
        assert res["deforestation_mask"][:16].all() and not res["deforestation_mask"][16:].any()
        assert res["deforested_pct"] == pytest.approx(50.0)
        assert res["area_ha"] == pytest.approx(16 * 32 * 100 / 10000)     # 10 m pixels from the transform

    def test_geographic_crs_gives_no_invented_area(self, tmp_path):
        a = _write(tmp_path / "a.tif", np.full((3, 16, 16), 255), "uint8", crs="EPSG:4326")
        b = _write(tmp_path / "b.tif", np.full((3, 16, 16), 0), "uint8", crs="EPSG:4326")
        assert _chm().detect_deforestation(a, b)["area_ha"] is None


# ---------------------------------------------------------------------------------------------- removed
class TestRemoved:
    def test_dino_txt_stand_in_is_gone(self):
        with pytest.raises(NotImplementedError, match="CLIP"):
            DINOv3Text()

    def test_finetune_only_does_classification(self):
        for task in ("segmentation", "detection", "regression"):
            with pytest.raises(NotImplementedError):
                finetune_dinov3(task=task)

    def test_finetune_prepares_a_trainable_classifier(self, monkeypatch, tmp_path):
        monkeypatch.setattr(DINOv3Backbone, "_load", lambda self: setattr(self, "_model", _tiny_vit()))
        res = finetune_dinov3("dinov3_vits16", num_classes=4, output_dir=str(tmp_path))
        assert res["status"] == "ready" and res["config"]["optimizer"] == "AdamW"
        assert res["model"](torch.randn(1, 3, 32, 32)).shape == (1, 4)
        assert any(p.requires_grad for p in res["model"].backbone.parameters())   # not frozen
