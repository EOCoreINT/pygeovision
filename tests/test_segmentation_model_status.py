"""Status of every registered segmentation model in the general registry.

History: U-Net/DeepLab/PSPNet/FCN/UPerNet were once registered but unbuildable, and before
that the three U-Net names silently returned one identical generic network.  They are now
real, backbone-specific builds, and these tests verify that they are *different networks*
(parameter counts match the published sizes) and that they really run on multispectral
input -- a shape assertion alone would also be satisfied by a fake fallback.

Entries that remain refusals are listed explicitly so a change in either direction is noticed.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("segmentation_models_pytorch", reason="segmentation-models-pytorch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from pygeovision.models.registry import get_model, model_registry  # noqa: E402


def _params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@pytest.mark.parametrize("name", [
    "unet-r50", "unet-r101", "unet-efficientb4",
    "deeplab-r50", "deeplab-r101", "pspnet-r50", "fcn-r50", "upernet-swin-b",
    "segformer-b0", "segformer-b2", "segformer-b5",
])
class TestSegmentationModelsAreRealAndMultispectral:
    @pytest.mark.parametrize("bands", [3, 4, 13])
    def test_forward_gives_full_resolution_logits(self, name, bands):
        model = get_model(name, num_classes=5, in_channels=bands, pretrained=False).eval()
        with torch.no_grad():
            out = model(torch.randn(1, bands, 96, 96))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 5, 96, 96)

    def test_parameter_count_matches_the_published_architecture(self, name):
        spec = model_registry[name]
        model = get_model(name, num_classes=5, in_channels=3, pretrained=False)
        # The registry's params_m is the published size of the architecture; the head size
        # changes it slightly, so allow 6 %.
        assert _params_m(model) == pytest.approx(spec.params_m, rel=0.06)

    def test_trains_one_step(self, name):
        model = get_model(name, num_classes=3, in_channels=4, pretrained=False).train()
        x, y = torch.randn(2, 4, 64, 64), torch.randint(0, 3, (2, 64, 64))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


class TestUNetVariantsAreDistinctNetworks:
    """The original bug: all three names returned the same fallback network."""

    def test_three_variants_have_different_encoders(self):
        sizes = {n: _params_m(get_model(n, num_classes=2, in_channels=3, pretrained=False))
                 for n in ("unet-r50", "unet-r101", "unet-efficientb4")}
        assert len({round(v, 1) for v in sizes.values()}) == 3, sizes
        assert sizes["unet-r101"] > sizes["unet-r50"] > sizes["unet-efficientb4"]

    def test_native_registry_equivalent_still_builds(self):
        from pygeovision.ai.models.registry import registry as native_registry
        model = native_registry.build("unet_resnet50", num_classes=2, in_channels=3, pretrained=False)
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


class TestHuggingFaceHeadsAreNotDropped:
    """AutoModel returned the headless encoder, so num_classes was silently ignored."""

    def test_segformer_has_a_segmentation_head_with_the_requested_classes(self):
        model = get_model("segformer-b0", num_classes=7, in_channels=4, pretrained=False)
        assert model.hf_model.config.num_labels == 7
        assert model.hf_model.decode_head.classifier.out_channels == 7

    def test_segformer_output_is_input_resolution_not_a_quarter(self):
        model = get_model("segformer-b0", num_classes=2, in_channels=3, pretrained=False).eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 128, 160)).shape[-2:] == (128, 160)


class TestStillRefusedWithAnAccurateReason:
    @pytest.mark.parametrize("name", ["dinov3_segmentor"])
    def test_registered_but_unbuildable(self, name):
        with pytest.raises(NotImplementedError):
            get_model(name, num_classes=2, in_channels=3, pretrained=False)

    @pytest.mark.parametrize("name", ["mask2former-swin-t", "mask2former-swin-b"])
    def test_hub_only_config_fails_clearly_not_as_not_implemented(self, name, monkeypatch):
        """These need their config from the Hugging Face Hub.  Offline that is a load error
        naming the Hub, not a misleading 'no implementation'."""
        import transformers

        from pygeovision.core.exceptions import ModelLoadError

        def _offline(*a, **k):
            raise OSError("simulated: Hub unreachable")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _offline)
        with pytest.raises(ModelLoadError, match="Hub"):
            get_model(name, num_classes=2, in_channels=3, pretrained=False)
