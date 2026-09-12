"""Regression tests for bugs found auditing whether pygeovision's models
genuinely work end-to-end with real prepared geospatial data (not just
synthetic tensors in isolation).

Three real, previously-hidden bugs found this pass:
  1. TiledInference._predict_chip() silently swallowed every exception at
     debug level and returned an all-zero chip — a model with an
     incompatible interface would silently corrupt output with no visible
     error.
  2. TiledInference's default half_precision=True converted the *input* to
     float16 regardless of device, but only cast the *model* to half on
     CUDA — meaning every single chip prediction crashed on CPU (the
     common case for users without a GPU), previously hidden entirely by
     bug #1.
  3. ChangeDetection(model_variant=...) claimed to support 'bit'/
     'bitemporal'/'changestar' but always silently built ChangeFormer
     regardless of what was actually requested.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds


def _make_geotiff(path, bands=4, size=64, crs="EPSG:32630", seed=0):
    rng = np.random.default_rng(seed)
    transform = from_bounds(818000, 615000, 818000 + size * 10, 615000 + size * 10, size, size)
    data = rng.random((bands, size, size)).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands,
                        dtype="float32", crs=crs, transform=transform) as dst:
        dst.write(data)
    return transform


class TestTiledInferenceLoudFailure:
    """Regression tests for bug #1 — chip-prediction failures must not be
    silently swallowed into an all-zero output."""

    def test_incompatible_model_caught_by_preflight_check(self, tmp_path):
        """A model whose forward() signature doesn't match TiledInference's
        single-tensor-in/logits-out assumption must not silently succeed.
        Verified directly: this specific incompatibility (wrong argument
        count) is now caught even earlier than the original version of
        this test expected -- a real pre-flight dummy-forward-pass check
        (added in a separate audit round) catches it before the tile loop
        starts at all, returning an honest success=False with a clear
        error rather than wasting time on the full tile loop first."""
        from pygeovision.inference.tiled import TiledInference

        class IncompatibleModel(torch.nn.Module):
            def forward(self, x1, x2):  # wrong signature — needs 2 args
                return x1

        raster_path = tmp_path / "scene.tif"
        _make_geotiff(raster_path)

        inf = TiledInference(IncompatibleModel(), chip_size=32, overlap=0, num_classes=2)
        result = inf.infer(str(raster_path), str(tmp_path / "out.tif"))
        assert result["success"] is False
        assert "does not accept" in result["error"] or "missing" in result["error"]

    def test_tolerate_chip_failures_opts_into_old_behavior(self, tmp_path):
        """With tolerate_chip_failures=True, failures are logged and
        zero-filled instead of raising — an explicit, documented opt-in,
        not the silent default."""
        from pygeovision.inference.tiled import TiledInference

        class IncompatibleModel(torch.nn.Module):
            def forward(self, x1, x2):
                return x1

        raster_path = tmp_path / "scene.tif"
        _make_geotiff(raster_path)

        inf = TiledInference(IncompatibleModel(), chip_size=32, overlap=0, num_classes=2,
                              tolerate_chip_failures=True)
        result = inf.infer(str(raster_path), str(tmp_path / "out.tif"))
        assert result["success"] is True


class TestTiledInferenceHalfPrecisionDeviceBug:
    """Regression test for bug #2 — half_precision=True (the default) must
    not crash on CPU."""

    def test_default_settings_work_on_cpu(self, tmp_path):
        from pygeovision.ai.models.registry import registry as native_registry
        from pygeovision.inference.tiled import TiledInference

        model = native_registry.build("unet_resnet50", num_classes=2, in_channels=4, pretrained=False)
        raster_path = tmp_path / "scene.tif"
        _make_geotiff(raster_path)

        # Default half_precision=True — this must NOT raise on CPU.
        inf = TiledInference(model, chip_size=32, overlap=8, num_classes=2)
        assert inf._device == "cpu"
        result = inf.infer(str(raster_path), str(tmp_path / "out.tif"))
        assert result["success"] is True
        assert result["n_chips"] > 0

    def test_output_has_real_spatial_variation(self, tmp_path):
        """The underlying probabilities (not necessarily the post-argmax
        label map, which can collapse for an untrained model) must show
        genuine per-pixel variation, confirming real per-chip computation
        is happening rather than a fallback/zero-fill path."""
        from pygeovision.ai.models.registry import registry as native_registry
        from pygeovision.inference.tiled import TiledInference

        model = native_registry.build("unet_resnet50", num_classes=2, in_channels=4, pretrained=False)
        raster_path = tmp_path / "scene.tif"
        _make_geotiff(raster_path, size=96)

        inf = TiledInference(model, chip_size=32, overlap=8, num_classes=2)
        out_path = tmp_path / "probs.tif"
        inf.infer(str(raster_path), str(out_path), return_probabilities=True)

        with rasterio.open(out_path) as src:
            probs = src.read()
        assert probs.std() > 0, "output has zero variation — likely silently zero-filled"


class TestChangeDetectionDispatch:
    """Regression tests for bug #3 — model_variant must genuinely route to
    a different architecture, not silently always build ChangeFormer."""

    def test_bit_variant_returns_bit_detector(self):
        from pygeovision.models.change_detection.changeformer import ChangeDetection
        from pygeovision.models.change_detection.bit import BITChangeDetector
        det = ChangeDetection(model_variant="bit", num_classes=2, in_channels=4)
        assert isinstance(det, BITChangeDetector)

    def test_bitemporal_alias_also_returns_bit_detector(self):
        from pygeovision.models.change_detection.changeformer import ChangeDetection
        from pygeovision.models.change_detection.bit import BITChangeDetector
        det = ChangeDetection(model_variant="bitemporal", num_classes=2, in_channels=4)
        assert isinstance(det, BITChangeDetector)

    def test_dsamnet_variant_returns_dsamnet_detector(self):
        from pygeovision.models.change_detection.changeformer import ChangeDetection
        from pygeovision.models.change_detection.dsamnet import DSAMNetChangeDetector
        det = ChangeDetection(model_variant="dsamnet", num_classes=2, in_channels=4)
        assert isinstance(det, DSAMNetChangeDetector)

    def test_changeformer_variant_still_works(self):
        from pygeovision.models.change_detection.changeformer import ChangeDetection, ChangeFormer
        det = ChangeDetection(model_variant="changeformer", num_classes=2, in_channels=4)
        assert isinstance(det, ChangeFormer)

    def test_changestar_raises_instead_of_silently_substituting(self):
        from pygeovision.models.change_detection.changeformer import ChangeDetection
        with pytest.raises(NotImplementedError):
            ChangeDetection(model_variant="changestar")

    def test_unknown_variant_raises_value_error(self):
        from pygeovision.models.change_detection.changeformer import ChangeDetection
        with pytest.raises(ValueError):
            ChangeDetection(model_variant="not_a_real_model")


class TestBITEndToEnd:
    """Real end-to-end test: two real georeferenced GeoTIFFs -> BIT -> a
    real georeferenced change mask. Not synthetic tensors — actual raster
    I/O, matching what a notebook user would actually do."""

    def test_detect_real_georeferenced_files(self, tmp_path):
        from pygeovision.models.change_detection.bit import BITChangeDetector

        before = tmp_path / "before.tif"
        after = tmp_path / "after.tif"
        transform = _make_geotiff(before, seed=1)
        _make_geotiff(after, seed=2)

        detector = BITChangeDetector(num_classes=2, in_channels=4, backbone="resnet18")
        out_path = tmp_path / "change.tif"
        result = detector.detect(str(before), str(after), str(out_path))

        assert result["model"] == "BIT"
        assert 0 <= result["change_pct"] <= 100
        with rasterio.open(out_path) as src:
            assert src.crs == "EPSG:32630"
            assert src.transform == transform
            assert src.count == 1
            assert src.dtypes[0] == "uint8"

    def test_handles_channel_mismatch(self, tmp_path):
        """A 3-band 'before' against a 4-band 'after' must not crash —
        channels get fixed to the configured in_channels."""
        from pygeovision.models.change_detection.bit import BITChangeDetector

        before = tmp_path / "before.tif"
        after = tmp_path / "after.tif"
        _make_geotiff(before, bands=3, seed=1)
        _make_geotiff(after, bands=4, seed=2)

        detector = BITChangeDetector(num_classes=2, in_channels=4, backbone="resnet18")
        result = detector.detect(str(before), str(after), str(tmp_path / "change.tif"))
        assert "error" not in result


class TestDSAMNetEndToEnd:
    """Real end-to-end test for DSAMNet, including the distance-map output."""

    def test_detect_real_georeferenced_files(self, tmp_path):
        from pygeovision.models.change_detection.dsamnet import DSAMNetChangeDetector

        before = tmp_path / "before.tif"
        after = tmp_path / "after.tif"
        transform = _make_geotiff(before, seed=1)
        _make_geotiff(after, seed=2)

        detector = DSAMNetChangeDetector(num_classes=2, in_channels=4, backbone="resnet18")
        out_path = tmp_path / "change.tif"
        result = detector.detect(str(before), str(after), str(out_path))

        assert result["model"] == "DSAMNet"
        with rasterio.open(out_path) as src:
            assert src.crs == "EPSG:32630"
            assert src.transform == transform

    def test_distance_map_has_real_variation(self, tmp_path):
        from pygeovision.models.change_detection.dsamnet import DSAMNetChangeDetector

        before = tmp_path / "before.tif"
        after = tmp_path / "after.tif"
        _make_geotiff(before, seed=1)
        _make_geotiff(after, seed=2)

        detector = DSAMNetChangeDetector(num_classes=2, in_channels=4, backbone="resnet18")
        result = detector.detect(
            str(before), str(after), str(tmp_path / "change.tif"), return_distance_map=True,
        )
        assert "distance_map_path" in result
        with rasterio.open(result["distance_map_path"]) as src:
            dist = src.read(1)
        assert dist.std() > 0
        assert (dist >= 0).all()  # Euclidean distance is never negative
