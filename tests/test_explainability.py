"""Tests for pygeovision.explainability — real bugs found and fixed:

  1. GradCAMPlusPlus claimed "element-wise square of gradients for
     improved localisation" but just called the parent GradCAM's
     unchanged implementation — silently identical output under a
     different name.
  2. AttentionMapExtractor's exception fallback returned a hardcoded
     (64, 64) zero array regardless of actual input image size —
     inconsistent with the sibling GradCAM.explain(), which correctly
     uses the real input shape.
  3. GeospatialSHAP's background_samples (constructor) and n_samples
     (method) parameters were both accepted but never used anywhere —
     the background was always a single all-zero image regardless.
  4. UncertaintyEstimator computed aleatoric uncertainty then discarded
     it — missing entirely from the output despite being explicitly
     promised in the class docstring. save_epistemic/save_aleatoric were
     both dead parameters that never controlled anything.
  5. UncertaintyEstimator crashed with a cryptic Python-internal
     ValueError when chip_size <= overlap, instead of a clear message.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
import torch.nn as nn
from rasterio.transform import from_bounds


def _make_geotiff(path, bands=4, size=64):
    transform = from_bounds(0, 0, size * 10, size * 10, size, size)
    data = np.random.rand(bands, size, size).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands,
                        dtype="float32", crs="EPSG:32630", transform=transform) as dst:
        dst.write(data)


class TinyConvNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestGradCAMPlusPlus:
    """Regression tests for the real 'silent alias' bug."""

    def test_produces_different_output_from_vanilla_gradcam(self):
        from pygeovision.explainability.gradcam import GradCAM, GradCAMPlusPlus
        torch.manual_seed(0)
        model = TinyConvNet()
        image = torch.randn(1, 3, 32, 32)

        cam = GradCAM(model, target_layer="conv2")
        sal = cam.explain(image, class_idx=1)

        cam_pp = GradCAMPlusPlus(model, target_layer="conv2")
        sal_pp = cam_pp.explain(image, class_idx=1)

        assert np.abs(sal - sal_pp).mean() > 1e-6, "GradCAM++ is identical to GradCAM"

    def test_output_shape_matches_input(self):
        from pygeovision.explainability.gradcam import GradCAMPlusPlus
        model = TinyConvNet()
        image = torch.randn(1, 3, 40, 24)
        cam_pp = GradCAMPlusPlus(model, target_layer="conv2")
        sal = cam_pp.explain(image, class_idx=0)
        assert sal.shape == (40, 24)

    def test_normalised_output_in_unit_range(self):
        from pygeovision.explainability.gradcam import GradCAMPlusPlus
        model = TinyConvNet()
        image = torch.randn(1, 3, 32, 32)
        cam_pp = GradCAMPlusPlus(model, target_layer="conv2")
        sal = cam_pp.explain(image, class_idx=0, normalize=True)
        assert sal.min() >= 0.0 and sal.max() <= 1.0 + 1e-6


class TestAttentionMapExtractorFallbackShape:
    """Regression test for the real hardcoded-(64,64) fallback bug."""

    def test_fallback_shape_matches_real_input_not_hardcoded_64(self):
        from pygeovision.explainability.attention import AttentionMapExtractor

        class BrokenModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("simulated failure")

        extractor = AttentionMapExtractor(BrokenModel())
        image = torch.randn(1, 3, 17, 23)  # deliberately NOT 64x64
        result = extractor.extract(image)
        assert result.shape == (17, 23), f"fallback used wrong shape: {result.shape}"


class TestGeospatialSHAPDeadParameters:
    """Regression tests for background_samples/n_samples being silently
    ignored — the background was always a single all-zero image."""

    def test_background_samples_actually_used(self, monkeypatch):
        pytest.importorskip("shap", reason="shap not installed")
        from pygeovision.explainability.shap_geo import GeospatialSHAP

        captured = {}
        import shap as shap_mod

        class _FakeExplainer:
            def __init__(self, model, background):
                captured["background_shape"] = tuple(background.shape)
            def shap_values(self, image, check_additivity=True):
                import numpy as np
                return np.zeros((1, image.shape[1]) + tuple(image.shape[2:]))

        monkeypatch.setattr(shap_mod, "DeepExplainer", _FakeExplainer)

        model = TinyConvNet(in_ch=4)
        shap_exp = GeospatialSHAP(model, background_samples=17)
        image = torch.randn(1, 4, 8, 8)
        shap_exp.band_importance(image)

        assert captured["background_shape"][0] == 17, (
            f"background_samples=17 was not used — background had "
            f"{captured['background_shape'][0]} samples instead"
        )

    def test_n_samples_overrides_constructor_default(self, monkeypatch):
        pytest.importorskip("shap", reason="shap not installed")
        from pygeovision.explainability.shap_geo import GeospatialSHAP

        captured = {}
        import shap as shap_mod

        class _FakeExplainer:
            def __init__(self, model, background):
                captured["background_shape"] = tuple(background.shape)
            def shap_values(self, image, check_additivity=True):
                import numpy as np
                return np.zeros((1, image.shape[1]) + tuple(image.shape[2:]))

        monkeypatch.setattr(shap_mod, "DeepExplainer", _FakeExplainer)

        model = TinyConvNet(in_ch=4)
        shap_exp = GeospatialSHAP(model, background_samples=50)
        image = torch.randn(1, 4, 8, 8)
        shap_exp.band_importance(image, n_samples=5)

        assert captured["background_shape"][0] == 5


class TestUncertaintyEstimatorAleatoric:
    """Regression tests for the real discarded-aleatoric-uncertainty bug
    and the dead save_epistemic/save_aleatoric parameters."""

    class DropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 3, padding=1)
            self.dropout = nn.Dropout2d(0.3)
            self.classifier = nn.Conv2d(8, 3, 1)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.dropout(x)
            return self.classifier(x)

    def test_aleatoric_present_in_result(self, tmp_path):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img_path = tmp_path / "scene.tif"
        _make_geotiff(img_path, size=64)

        torch.manual_seed(0)
        model = self.DropoutNet()
        estimator = UncertaintyEstimator(model, n_passes=3)
        result = estimator.estimate(
            str(img_path), str(tmp_path / "unc.tif"),
            chip_size=64, overlap=8, save_epistemic=True, save_aleatoric=True,
        )
        assert result["success"] is True
        assert "mean_aleatoric" in result
        assert result["mean_aleatoric"] > 0

    def test_save_aleatoric_flag_genuinely_controls_output(self, tmp_path):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img_path = tmp_path / "scene.tif"
        _make_geotiff(img_path, size=64)

        torch.manual_seed(0)
        model = self.DropoutNet()
        estimator = UncertaintyEstimator(model, n_passes=3)

        out_with = tmp_path / "with_aleatoric.tif"
        estimator.estimate(str(img_path), str(out_with), chip_size=64, overlap=8,
                            save_epistemic=True, save_aleatoric=True)
        out_without = tmp_path / "without_aleatoric.tif"
        estimator.estimate(str(img_path), str(out_without), chip_size=64, overlap=8,
                            save_epistemic=True, save_aleatoric=False)

        with rasterio.open(out_with) as src:
            assert "aleatoric" in src.tags().get("bands", "")
            n_with = src.count
        with rasterio.open(out_without) as src:
            assert "aleatoric" not in src.tags().get("bands", "")
            n_without = src.count
        assert n_with == n_without + 1

    def test_save_epistemic_flag_genuinely_controls_output(self, tmp_path):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img_path = tmp_path / "scene.tif"
        _make_geotiff(img_path, size=64)

        torch.manual_seed(0)
        model = self.DropoutNet()
        estimator = UncertaintyEstimator(model, n_passes=3)
        out = tmp_path / "no_epistemic.tif"
        estimator.estimate(str(img_path), str(out), chip_size=64, overlap=8,
                            save_epistemic=False, save_aleatoric=False)
        with rasterio.open(out) as src:
            assert "epistemic" not in src.tags().get("bands", "")
            assert src.count == 2  # label + entropy only

    def test_chip_size_le_overlap_gives_clear_error_not_crash(self, tmp_path):
        """Regression test for a real bug: chip_size <= overlap crashed
        with a cryptic Python-internal ValueError (range() arg 3 must
        not be zero) instead of a clear, actionable message."""
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img_path = tmp_path / "scene.tif"
        _make_geotiff(img_path, size=64)

        model = self.DropoutNet()
        estimator = UncertaintyEstimator(model, n_passes=2)
        result = estimator.estimate(str(img_path), str(tmp_path / "bad.tif"),
                                     chip_size=64, overlap=64)
        assert result["success"] is False
        assert "chip_size" in result["error"]


# --------------------------------------------------------------------------------------
# Regression tests added with the explainability documentation pass
# --------------------------------------------------------------------------------------
class SegNet(nn.Module):
    """Conv + BatchNorm + Dropout2d + 1x1 classifier: a minimal segmentation model."""

    def __init__(self, in_ch=4, n_classes=3):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.do = nn.Dropout2d(0.3)
        self.c2 = nn.Conv2d(8, n_classes, 1)

    def forward(self, x):
        return self.c2(self.do(torch.relu(self.bn(self.c1(x)))))


class TestGradCAMRobustness:
    def test_segmentation_model_without_class_idx(self):
        """The default class used to be a flat index into the whole logit volume (IndexError)."""
        from pygeovision.explainability.gradcam import GradCAM
        sal = GradCAM(SegNet()).explain(torch.randn(1, 4, 32, 32))
        assert sal.shape == (32, 32)

    def test_inplace_relu_resnet(self):
        """register_full_backward_hook fails on torchvision ResNets (in-place ReLU)."""
        tv = pytest.importorskip("torchvision")
        from pygeovision.explainability.gradcam import GradCAM
        sal = GradCAM(tv.models.resnet18(num_classes=3)).explain(torch.randn(1, 3, 64, 64))
        assert sal.shape == (64, 64) and sal.max() > 0

    def test_unknown_target_layer_names_the_choices(self):
        from pygeovision.explainability.gradcam import GradCAM
        with pytest.raises(ValueError, match="c1"):
            GradCAM(SegNet(), target_layer="nope").explain(torch.randn(1, 4, 16, 16))

    def test_stale_gradients_not_reused(self):
        from pygeovision.explainability.gradcam import GradCAM
        cam = GradCAM(SegNet())
        a = cam.explain(torch.randn(1, 4, 16, 16), class_idx=0)
        b = cam.explain(torch.randn(1, 4, 16, 16), class_idx=1)
        assert a.shape == b.shape == (16, 16)

    @pytest.mark.parametrize("bands", [1, 3, 4])
    def test_visualise_any_band_count(self, tmp_path, bands):
        """visualise() used to read the wrong bands and call a removed matplotlib API."""
        pytest.importorskip("matplotlib")
        from pygeovision.explainability.gradcam import GradCAM
        img = tmp_path / "s.tif"
        _make_geotiff(img, bands=bands, size=32)
        out = tmp_path / "o.png"
        GradCAM(SegNet()).visualise(str(img), np.random.rand(16, 16), str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_batch_explain_rejects_bad_tiling_without_raising(self, tmp_path):
        from pygeovision.explainability.gradcam import GradCAM
        img = tmp_path / "s.tif"
        _make_geotiff(img, size=32)
        res = GradCAM(SegNet()).batch_explain(str(img), str(tmp_path / "o.tif"), chip_size=16, overlap=16)
        assert res["success"] is False and "overlap" in res["error"]

    def test_batch_explain_writes_map(self, tmp_path):
        from pygeovision.explainability.gradcam import GradCAM
        img = tmp_path / "s.tif"
        _make_geotiff(img, size=48)
        res = GradCAM(SegNet()).batch_explain(str(img), str(tmp_path / "o.tif"), class_idx=1,
                                              chip_size=32, overlap=8)
        assert res["success"] and res["n_chips"] == 4
        with rasterio.open(res["output_path"]) as src:
            assert src.count == 1 and src.shape == (48, 48)


class TestUncertaintyRobustness:
    def test_batchnorm_statistics_untouched_and_model_left_in_eval(self, tmp_path):
        """A model in train mode used to have BatchNorm statistics updated during estimation."""
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img = tmp_path / "s.tif"
        _make_geotiff(img, size=64)
        model = SegNet().train()
        before = model.bn.running_mean.clone()
        UncertaintyEstimator(model, n_passes=3).estimate(str(img), None, chip_size=64, overlap=8)
        assert torch.equal(before, model.bn.running_mean)
        assert not model.training and not model.do.training

    def test_confident_model_gives_finite_nonnegative_aleatoric(self, tmp_path):
        """float16 accumulation used to turn confident predictions into NaN."""
        from pygeovision.explainability.uncertainty import UncertaintyEstimator

        class Confident(nn.Module):
            def __init__(self):
                super().__init__()
                self.do = nn.Dropout2d(0.0)

            def forward(self, x):
                out = torch.zeros(x.shape[0], 3, *x.shape[2:])
                out[:, 0] = 60.0
                return self.do(out)

        img = tmp_path / "s.tif"
        _make_geotiff(img, size=32)
        r = UncertaintyEstimator(Confident(), n_passes=3).estimate(str(img), None, chip_size=32, overlap=4)
        assert np.isfinite(r["mean_aleatoric"]) and r["mean_aleatoric"] >= 0
        assert r["mean_entropy"] >= 0 and r["high_uncertainty_fraction"] == 0

    def test_input_not_modified_by_first_chip_normalisation(self, tmp_path):
        """Running with more chips than one must not depend on chip order or earlier chips."""
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img = tmp_path / "s.tif"
        _make_geotiff(img, size=96)

        class NoDrop(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(4, 3, 1)

            def forward(self, x):
                return self.c(x)

        torch.manual_seed(0)
        m = NoDrop()
        r1 = UncertaintyEstimator(m, n_passes=1).estimate(str(img), str(tmp_path / "a.tif"), chip_size=64, overlap=16)
        r2 = UncertaintyEstimator(m, n_passes=1).estimate(str(img), str(tmp_path / "b.tif"), chip_size=64, overlap=16)
        assert r1["mean_entropy"] == r2["mean_entropy"]

    def test_no_dropout_warns_and_runs_once(self, tmp_path):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img = tmp_path / "s.tif"
        _make_geotiff(img, size=32)
        r = UncertaintyEstimator(nn.Conv2d(4, 3, 1), n_passes=10).estimate(str(img), None, chip_size=32, overlap=4)
        assert r["n_dropout_layers"] == 0 and r["n_mc_passes"] == 1
        assert "no dropout" in r["warning"] and r["mean_epistemic"] == 0

    def test_classification_output_reports_clear_error(self, tmp_path):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator

        class Clf(nn.Module):
            def __init__(self):
                super().__init__()
                self.do = nn.Dropout(0.5)
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(self.do(x.mean(dim=(-2, -1))))

        img = tmp_path / "s.tif"
        _make_geotiff(img, size=32)
        r = UncertaintyEstimator(Clf(), n_passes=2).estimate(str(img), None, chip_size=32, overlap=4)
        assert r["success"] is False and "(B, classes, H, W)" in r["error"]

    def test_output_bands_are_described(self, tmp_path):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        img = tmp_path / "s.tif"
        _make_geotiff(img, size=32)
        out = tmp_path / "u.tif"
        UncertaintyEstimator(SegNet(), n_passes=2).estimate(str(img), str(out), chip_size=32, overlap=4)
        with rasterio.open(out) as src:
            assert list(src.descriptions) == ["label", "entropy", "epistemic", "aleatoric"]


class TestAttentionRobustness:
    class Tiny(nn.Module):
        def __init__(self, cls_token=False):
            super().__init__()
            self.proj = nn.Conv2d(3, 16, 8, 8)
            self.attn = nn.MultiheadAttention(16, 2, batch_first=True)
            self.cls = nn.Parameter(torch.zeros(1, 1, 16)) if cls_token else None

        def forward(self, x):
            t = self.proj(x).flatten(2).transpose(1, 2)
            if self.cls is not None:
                t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
            return self.attn(t, t, t)[0]

    def test_patch_grid_map(self):
        from pygeovision.explainability.attention import AttentionMapExtractor
        a = AttentionMapExtractor(self.Tiny()).extract(torch.randn(1, 3, 32, 32))
        assert a.shape == (4, 4) and np.isclose(a.sum(), 1.0, atol=1e-4)

    def test_class_token_is_dropped(self):
        from pygeovision.explainability.attention import AttentionMapExtractor
        a = AttentionMapExtractor(self.Tiny(cls_token=True)).extract(torch.randn(1, 3, 32, 32))
        assert a.shape == (4, 4)

    def test_single_head_selection(self):
        from pygeovision.explainability.attention import AttentionMapExtractor
        ex = AttentionMapExtractor(self.Tiny())
        assert ex.extract(torch.randn(1, 3, 32, 32), head_idx=0).shape == (4, 4)
        # nn.MultiheadAttention returns head-averaged weights, i.e. a single head
        with pytest.raises(IndexError, match="1 head"):
            ex.extract(torch.randn(1, 3, 32, 32), head_idx=1, strict=True)

    def test_hooks_are_removed_after_each_call(self):
        from pygeovision.explainability.attention import AttentionMapExtractor
        m = self.Tiny()
        ex = AttentionMapExtractor(m)
        for _ in range(3):
            ex.extract(torch.randn(1, 3, 32, 32))
        assert sum(len(mod._forward_hooks) for mod in m.modules()) == 0

    def test_strict_raises_and_default_returns_zeros(self):
        from pygeovision.explainability.attention import AttentionMapExtractor
        m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))        # no attention at all
        ex = AttentionMapExtractor(m)
        assert ex.extract(torch.randn(1, 3, 20, 12)).shape == (20, 12)
        with pytest.raises(RuntimeError, match="no attention weights"):
            ex.extract(torch.randn(1, 3, 20, 12), strict=True)

    def test_visualise_writes_png(self, tmp_path):
        pytest.importorskip("matplotlib")
        from pygeovision.explainability.attention import AttentionMapExtractor
        img = tmp_path / "s.tif"
        _make_geotiff(img, bands=3, size=32)
        ex = AttentionMapExtractor(self.Tiny())
        out = tmp_path / "a.png"
        ex.visualise(str(img), ex.extract(torch.randn(1, 3, 32, 32)), str(out))
        assert out.exists()


class TestSHAPOutputs:
    def _patch(self, monkeypatch, values):
        import sys
        import types
        fake = types.ModuleType("shap")

        class _E:
            def __init__(self, model, background):
                pass

            def shap_values(self, image, check_additivity=True):
                return values(image)

        fake.DeepExplainer = _E
        monkeypatch.setitem(sys.modules, "shap", fake)

    def test_class_idx_selects_one_output(self, monkeypatch):
        from pygeovision.explainability.shap_geo import GeospatialSHAP

        def values(image):
            shape = tuple(image.shape)
            return [np.zeros(shape), np.ones(shape) * 2.0]      # output 0 explains nothing, output 1 everything

        self._patch(monkeypatch, values)
        g = GeospatialSHAP(TinyConvNet(in_ch=4), background_samples=2)
        img = torch.randn(1, 4, 8, 8)
        assert g.band_importance(img, class_idx=1)["band_importance"]["band_1"] == 2.0
        assert g.band_importance(img, class_idx=0)["band_importance"]["band_1"] == 0.0
        assert g.band_importance(img)["band_importance"]["band_1"] == 1.0      # mean over outputs
        with pytest.raises(ValueError, match="out of range"):
            g.band_importance(img, class_idx=5)

    def test_stacked_array_layout(self, monkeypatch):
        from pygeovision.explainability.shap_geo import GeospatialSHAP

        def values(image):
            arr = np.zeros(tuple(image.shape) + (2,))
            arr[..., 1] = 3.0
            return arr

        self._patch(monkeypatch, values)
        g = GeospatialSHAP(TinyConvNet(in_ch=4), background_samples=2)
        out = g.band_importance(torch.randn(1, 4, 8, 8), class_idx=1)
        assert out["band_importance"]["band_4"] == 3.0

    def test_plot_checks_band_name_count(self):
        pytest.importorskip("matplotlib")
        from pygeovision.explainability.shap_geo import GeospatialSHAP
        g = GeospatialSHAP(TinyConvNet(in_ch=4))
        with pytest.raises(ValueError, match="band names"):
            g.plot_band_importance({"band_importance": {"band_1": 1.0, "band_2": 2.0}}, band_names=["only_one"])


class TestExplainCLI:
    def test_gradcam_failure_is_reported_not_celebrated(self, tmp_path):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        img = tmp_path / "s.tif"
        _make_geotiff(img, size=32)
        res = CliRunner().invoke(cli, ["explain", "gradcam", str(img), "-m", "no_such_model_xyz"])
        assert res.exit_code != 0
