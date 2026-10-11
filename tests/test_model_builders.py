"""Multispectral input, pretrained-weights policy and output contract of the model builders.

Defects pinned here (each was reproduced before being fixed):

* every torchvision detector crashed in ``forward`` on anything but 3 bands, because the
  built-in normaliser has a 3-element mean (the backbone alone had been widened);
* ``pretrained=True`` that failed to download quietly returned random weights;
* the Hugging Face path built the headless ``AutoModel`` (``num_classes`` ignored) and
  segmentation logits came out at 1/4 resolution;
* DETR got ``num_classes + 1`` labels on top of its own "no object" slot;
* a super-resolution model got 3 output bands for a 4-band input.
"""
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
pytest.importorskip("torchvision")
transformers = pytest.importorskip("transformers")

from pygeovision.core.exceptions import ModelLoadError  # noqa: E402
from pygeovision.models import _adapt  # noqa: E402
from pygeovision.models.registry import get_model, model_registry  # noqa: E402


def _n_params_m(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


# --------------------------------------------------------------------------- _adapt
class TestAdaptFirstConv:
    def make(self, cin=3, bias=False):
        torch.manual_seed(0)
        return nn.Conv2d(cin, 8, 7, stride=2, padding=3, bias=bias, dilation=1)

    def test_pretrained_keeps_rgb_filters_and_means_the_rest(self):
        conv = self.make()
        new = _adapt.adapt_first_conv(conv, 6, pretrained_loaded=True)
        assert new.in_channels == 6 and new.weight.shape == (8, 6, 7, 7)
        torch.testing.assert_close(new.weight[:, :3], conv.weight)
        mean = conv.weight.mean(dim=1, keepdim=True)
        for k in range(3, 6):
            torch.testing.assert_close(new.weight[:, k:k + 1], mean)
        assert new.stride == conv.stride and new.padding == conv.padding

    def test_not_pretrained_does_not_copy(self):
        conv = self.make()
        new = _adapt.adapt_first_conv(conv, 4, pretrained_loaded=False)
        assert not torch.equal(new.weight[:, :3], conv.weight)

    def test_fewer_than_three_channels_uses_the_first_filters(self):
        conv = self.make()
        new = _adapt.adapt_first_conv(conv, 1, pretrained_loaded=True)
        torch.testing.assert_close(new.weight, conv.weight[:, :1])

    def test_bias_is_copied(self):
        conv = self.make(bias=True)
        new = _adapt.adapt_first_conv(conv, 5, pretrained_loaded=True)
        torch.testing.assert_close(new.bias, conv.bias)

    def test_same_channel_count_returns_the_same_module(self):
        conv = self.make()
        assert _adapt.adapt_first_conv(conv, 3, True) is conv

    def test_grouped_conv_refused(self):
        with pytest.raises(ValueError, match="grouped"):
            _adapt.adapt_first_conv(nn.Conv2d(4, 4, 3, groups=2), 6, True)

    def test_keeps_dtype(self):
        conv = self.make().double()
        assert _adapt.adapt_first_conv(conv, 4, True).weight.dtype == torch.float64

    def test_extend_stats(self):
        assert _adapt.extend_stats([1.0, 2.0, 3.0], 5) == [1.0, 2.0, 3.0, 2.0, 2.0]
        assert _adapt.extend_stats([1.0, 2.0, 3.0], 2) == [1.0, 2.0]


# --------------------------------------------------------------------- detectors
@pytest.mark.parametrize("name", ["faster-rcnn-r50", "mask-rcnn-r50", "fcos-r50"])
@pytest.mark.parametrize("bands", [3, 4, 13])
def test_torchvision_detectors_run_on_any_band_count(name, bands):
    model = get_model(name, num_classes=2, in_channels=bands, pretrained=False,
                      min_size=96, max_size=96).eval()
    with torch.no_grad():
        out = model([torch.rand(bands, 96, 96)])
    assert set(out[0]) >= {"boxes", "scores", "labels"}
    assert list(model.transform.image_mean.__len__() for _ in [0]) == [bands]


def test_detector_normaliser_uses_imagenet_for_rgb_and_their_mean_for_extras():
    model = get_model("faster-rcnn-r50", num_classes=2, in_channels=5, pretrained=False)
    mean = model.transform.image_mean
    assert mean[:3] == list(_adapt.IMAGENET_MEAN)
    assert mean[3:] == pytest.approx([sum(_adapt.IMAGENET_MEAN) / 3] * 2)


def test_detector_accepts_sensor_statistics():
    mean, std = [0.1, 0.2, 0.3, 0.4], [0.05, 0.06, 0.07, 0.08]
    model = get_model("fcos-r50", num_classes=2, in_channels=4, pretrained=False,
                      image_mean=mean, image_std=std)
    assert model.transform.image_mean == mean and model.transform.image_std == std


def test_detector_statistics_of_the_wrong_length_rejected():
    with pytest.raises(ValueError, match="4 values"):
        get_model("fcos-r50", num_classes=2, in_channels=4, pretrained=False,
                  image_mean=[0.1, 0.2], image_std=[0.1, 0.2])


def test_detector_trains_one_step_on_four_bands():
    model = get_model("faster-rcnn-r50", num_classes=1, in_channels=4, pretrained=False,
                      min_size=64, max_size=64).train()
    img = torch.rand(4, 64, 64)
    tgt = {"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]), "labels": torch.tensor([1])}
    losses = model([img], [tgt])
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)


# ---------------------------------------------------------- pretrained-weights policy
class TestPretrainedFailureIsNeverSilent:
    @pytest.fixture
    def offline_timm(self, monkeypatch):
        import timm
        real = timm.create_model

        def fake(name, pretrained=False, **kw):
            if pretrained:
                raise OSError("simulated download failure")
            return real(name, pretrained=False, **kw)

        monkeypatch.setattr(timm, "create_model", fake)

    def test_timm_failure_raises(self, offline_timm):
        with pytest.raises(ModelLoadError, match="simulated download failure"):
            get_model("resnet50", num_classes=3, in_channels=4, pretrained=True)

    def test_opt_in_fallback_is_flagged(self, offline_timm):
        m = get_model("resnet50", num_classes=3, in_channels=4, pretrained=True,
                      allow_random_init=True)
        assert m._pygeovision_is_random_init is True
        assert "simulated download failure" in m._pygeovision_random_init_reason

    def test_not_pretrained_is_not_flagged(self, offline_timm):
        m = get_model("resnet50", num_classes=3, in_channels=4, pretrained=False)
        assert not getattr(m, "_pygeovision_is_random_init", False)

    def test_hugging_face_failure_raises(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("simulated hub failure")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", boom)
        with pytest.raises(ModelLoadError, match="simulated hub failure"):
            get_model("segformer-b0", num_classes=2, in_channels=3, pretrained=True)

    def test_hugging_face_opt_in_fallback_uses_the_local_architecture(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("simulated hub failure")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", boom)
        m = get_model("segformer-b0", num_classes=2, in_channels=3, pretrained=True,
                      allow_random_init=True)
        assert m._pygeovision_is_random_init is True
        assert m(torch.rand(1, 3, 64, 64)).shape == (1, 2, 64, 64)

    def test_torchvision_resnet_fallback_when_timm_missing_is_not_silent(self, monkeypatch):
        import torchvision

        def boom(*a, **k):
            raise OSError("simulated download failure")

        monkeypatch.setattr(torchvision.models, "resnet50", boom)
        monkeypatch.setitem(__import__("sys").modules, "timm", None)  # makes `import timm` fail
        with pytest.raises(OSError, match="simulated download failure"):
            get_model("resnet50", num_classes=3, in_channels=3, pretrained=True)


# ---------------------------------------------------- Hugging Face construction
HF_LOCAL = ["vit-b16", "vit-l16", "dinov2-s", "dinov2-b", "segformer-b0", "segformer-b2",
            "segformer-b5", "detr-r50", "detr-r101", "rt-detr-l", "upernet-swin-b", "swinir-m"]


@pytest.mark.parametrize("name", HF_LOCAL)
def test_offline_architecture_matches_the_published_parameter_count(name):
    """If the in-code config were wrong, the size would be off by far more than 6 %."""
    spec = model_registry[name]
    model = get_model(name, num_classes=3, in_channels=3, pretrained=False)
    assert _n_params_m(model) == pytest.approx(spec.params_m, rel=0.06)


def test_detr_label_count_is_the_number_of_real_classes():
    model = get_model("detr-r50", num_classes=3, in_channels=3, pretrained=False)
    assert model.config.num_labels == 3
    assert model.class_labels_classifier.out_features == 4      # 3 + DETR's own "no object"


@pytest.mark.parametrize("bands", [4, 13])
def test_detr_runs_on_multispectral_input(bands):
    model = get_model("detr-r50", num_classes=2, in_channels=bands, pretrained=False).eval()
    with torch.no_grad():
        out = model(pixel_values=torch.rand(1, bands, 96, 96))
    assert out.logits.shape[-1] == 3


def test_classifier_has_the_requested_head_and_returns_logits():
    model = get_model("vit-b16", num_classes=6, in_channels=4, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 4, 224, 224))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 6)


def test_hf_stem_adaptation_keeps_pretrained_rgb_filters():
    cfg = transformers.SegformerConfig(depths=[1, 1, 1, 1], hidden_sizes=[16, 32, 64, 128],
                                       decoder_hidden_size=32, num_labels=2)
    model = transformers.SegformerForSemanticSegmentation(cfg)
    def stem():  # transformers 4.x and 5.x place the patch embedding at different paths
        return next(m for m in model.modules() if isinstance(m, nn.Conv2d))

    before = stem().weight.detach().clone()
    assert stem().in_channels == 3
    _adapt.adapt_hf_input_channels(model, 5, pretrained_loaded=True)
    after = stem()
    assert after.in_channels == 5
    torch.testing.assert_close(after.weight[:, :3], before)
    assert model.config.num_channels == 5
    with torch.no_grad():
        assert model(pixel_values=torch.rand(1, 5, 64, 64)).logits.shape[1] == 2


def test_vit_img_size_override_changes_the_expected_chip_size():
    model = get_model("vit-b16", num_classes=2, in_channels=3, pretrained=False, img_size=64).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 3, 64, 64)).shape == (1, 2)


class TestSuperResolution:
    @pytest.mark.parametrize("bands", [3, 4, 13])
    def test_output_has_the_same_bands_as_the_input(self, bands):
        model = get_model("swinir-m", num_classes=2, in_channels=bands, pretrained=False).eval()
        with torch.no_grad():
            out = model(pixel_values=torch.rand(1, bands, 32, 32)).reconstruction
        assert out.shape == (1, bands, 128, 128)

    def test_pretrained_multiband_refused(self):
        with pytest.raises(ValueError, match="RGB-only"):
            get_model("swinir-m", in_channels=4, pretrained=True)


# ----------------------------------------------- the package's own trainer and tiler
def test_geotrainer_trains_a_registry_segformer(tmp_path):
    """The wrapper is what makes this possible: raw HF output is a ModelOutput at 1/4 size."""
    from pygeovision.ai.training.trainer import GeoTrainer, TrainingConfig

    model = get_model("segformer-b0", num_classes=2, in_channels=4, pretrained=False)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 4, 64, 64, generator=g)
    y = (x[:, 0] > 0.6).long()
    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=4)
    cfg = TrainingConfig(output_dir=str(tmp_path), max_epochs=2, batch_size=4, num_workers=0,
                         persistent_workers=False, pin_memory=False, augment=False, precision="fp32",
                         learning_rate=1e-3, early_stopping_patience=99, warmup_epochs=0, seed=0,
                         devices="cpu")
    result = GeoTrainer(model, cfg).fit(train_loader=dl, val_loader=dl, loss_fn=nn.CrossEntropyLoss())
    assert result["epochs_trained"] == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in result["history"]["train_loss"])


def test_tiled_inference_stitches_a_registry_segformer_at_full_resolution():
    import numpy as np

    from pygeovision.inference.tiled import TiledInference

    model = get_model("segformer-b0", num_classes=3, in_channels=4, pretrained=False).eval()
    image = np.random.RandomState(0).rand(4, 150, 170).astype("float32")
    probs = TiledInference(model, chip_size=64, overlap=16, num_classes=3, half_precision=False,
                           device="cpu").predict_array(image)
    assert probs.shape == (3, 150, 170)
    np.testing.assert_allclose(probs.sum(0), 1.0, atol=1e-4)


# -------------------------------------------------------------- registry integrity
REFUSED = {"rf-detr-b", "rf-detr-l", "dinov3_segmentor", "changestar-r18", "changeformer-mit-b4",
           "satlas-pretrain", "chatearthnet", "lisat-7b"}
# Wrappers around libraries/weights fetched at construction time, and the Hub-only configs.
NOT_OFFLINE = {"grounding-dino", "mask2former-swin-t", "mask2former-swin-b", "dofa-base",
               "dofa-large", "prithvi-100m", "prithvi-300m", "prithvi_eo_1_0", "prithvi_eo_2_0",
               "prithvi_burn_scar"}
WRAPPERS = {"clip", "moondream", "sam", "sam2", "tessera", "alphaearth"}


def _registry():
    from pygeovision.models import registry as r
    return r._REGISTRY


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_refused_entries_say_so(name):
    with pytest.raises(NotImplementedError):
        get_model(name, pretrained=False)


def test_no_other_entry_falls_through_to_not_implemented():
    """Every entry is either built or refuses with a reason that is not 'no build path'."""
    skipped = REFUSED | NOT_OFFLINE
    for name, spec in _registry().items():
        if name in skipped or spec.family in WRAPPERS or spec.task == "3d" or spec.params_m > 150:
            continue
        try:
            get_model(name, num_classes=2, in_channels=3, pretrained=False)
        except NotImplementedError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{name}: unexpectedly unbuildable: {exc}")


def test_registry_size_is_unchanged():
    # docs and README quote this number
    assert len(_registry()) == 79


def test_every_hub_id_is_a_repo_that_exists():
    """Ids verified against the Hugging Face API; a stale id fails here, not in a user's run."""
    known_bad = {"caidas/swin2SR-realworld-sr-x4-64", "ibm-nasa-geospatial/Prithvi-100M",
                 "ibm-nasa-geospatial/Prithvi-300M", "ibm-nasa-geospatial/Prithvi-100M-burn-scar",
                 "facebook/sam2-hiera-large", "roboflow/rf-detr-base", "roboflow/rf-detr-large"}
    used = {s.hf_id for s in _registry().values() if s.hf_id}
    assert not used & known_bad, used & known_bad
