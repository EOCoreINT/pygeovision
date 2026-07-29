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
