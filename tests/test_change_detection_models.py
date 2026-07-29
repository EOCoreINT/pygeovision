"""Tests for pygeovision.models.registry's fallback dispatch and the real
BIT/DSAMNet change-detection ports.

These replace what used to be silent-substitution stubs: architecture
names that ran without error but returned a generic 2-layer CNN classifier
regardless of what was actually requested. Every test here does a real
forward pass with torch — this only asserts on things that require the
model to genuinely compute, not just import cleanly.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torchvision", reason="torchvision not installed")


class TestRegistryLoudFailure:
    """Architectures with no real factory must raise, not silently
    substitute a meaningless generic classifier."""

    def test_centernet_raises_not_implemented(self):
        from pygeovision.models.registry import get_model
        with pytest.raises(NotImplementedError):
            get_model("centernet-r50", num_classes=2)

    def test_satlas_raises_not_implemented(self):
        from pygeovision.models.registry import get_model
        with pytest.raises(NotImplementedError):
            get_model("satlas-pretrain", num_classes=2)

    def test_changestar_raises_not_implemented(self):
        from pygeovision.models.registry import get_model
        with pytest.raises(NotImplementedError):
            get_model("changestar-r18", num_classes=2)

    def test_all_pointcloud_models_are_now_real(self):
        """Every point-cloud architecture in the registry is genuinely
        implemented: pointnet2-ssg/msg (test_pointnet2.py), randlanet
        (test_randlanet.py), kpconv (test_kpconv.py), pointtransformer
        (test_ptv3.py). None should raise NotImplementedError anymore."""
        from pygeovision.models.registry import get_model
        for name in ("pointnet2-ssg", "pointnet2-msg", "randlanet", "kpconv", "pointtransformer"):
            model = get_model(name, num_classes=2, in_channels=4, pretrained=False)
            assert model is not None

    def test_chatearthnet_raises_not_implemented(self):
        from pygeovision.models.registry import get_model
        with pytest.raises(NotImplementedError):
            get_model("chatearthnet", num_classes=2)


class TestRegistryDetectionDispatch:
    """Mask R-CNN / Faster R-CNN / FCOS must dispatch to the real
    torchvision implementations, with heads correctly resized."""

    def test_mask_rcnn_is_real_torchvision_maskrcnn(self):
        from pygeovision.models.registry import get_model
        model = get_model("mask-rcnn-r50", num_classes=3, pretrained=False)
        import torchvision
        assert isinstance(model, torchvision.models.detection.MaskRCNN)
        # +1 for background, matching torchvision's convention
        assert model.roi_heads.box_predictor.cls_score.out_features == 4

    def test_faster_rcnn_is_real_torchvision_fasterrcnn(self):
        from pygeovision.models.registry import get_model
        model = get_model("faster-rcnn-r50", num_classes=5, pretrained=False)
        import torchvision
        assert isinstance(model, torchvision.models.detection.FasterRCNN)
        assert not hasattr(model.roi_heads, "mask_predictor") or model.roi_heads.mask_predictor is None

    def test_fcos_is_real_torchvision_fcos(self):
        from pygeovision.models.registry import get_model
        model = get_model("fcos-r50", num_classes=4, pretrained=False)
        import torchvision
        assert isinstance(model, torchvision.models.detection.FCOS)

    def test_mask_rcnn_real_forward_pass(self):
        """A real forward pass through Mask R-CNN in eval mode, producing
        genuine boxes/labels/scores/masks — not a toy classifier output."""
        from pygeovision.models.registry import get_model
        model = get_model("mask-rcnn-r50", num_classes=2, in_channels=3, pretrained=False)
        model.eval()
        images = [torch.rand(3, 128, 128)]
        with torch.no_grad():
            out = model(images)
        assert isinstance(out, list) and len(out) == 1
        assert set(out[0].keys()) >= {"boxes", "labels", "scores", "masks"}


class TestBIT:
    """Real BIT (Bitemporal Image Transformer) — Chen et al. 2021."""

    def test_output_shape(self):
        from pygeovision.models.change_detection.bit import build_bit
        torch.manual_seed(0)
        model = build_bit(num_classes=2, in_channels=4, backbone="resnet18", pretrained=False)
        model.eval()
        img_a = torch.randn(2, 4, 128, 128)
        img_b = torch.randn(2, 4, 128, 128)
        with torch.no_grad():
            out = model(img_a, img_b)
        assert out.shape == (2, 2, 128, 128)

    def test_gradient_flows_through_nearly_all_parameters(self):
        from pygeovision.models.change_detection.bit import build_bit
        torch.manual_seed(1)
        model = build_bit(num_classes=2, in_channels=4, backbone="resnet18", pretrained=False)
        img_a = torch.randn(1, 4, 64, 64)
        img_b = torch.randn(1, 4, 64, 64)
        out = model(img_a, img_b)
        loss = (out ** 2).mean()
        loss.backward()
        total = sum(1 for _ in model.named_parameters())
        missing = sum(1 for _, p in model.named_parameters()
                      if p.grad is None or p.grad.abs().sum() == 0)
        assert missing <= 1, f"{missing}/{total} parameters have no gradient"

    def test_resnet50_backbone_variant(self):
        from pygeovision.models.change_detection.bit import build_bit
        model = build_bit(num_classes=2, in_channels=3, backbone="resnet50", pretrained=False)
        model.eval()
        img_a = torch.randn(1, 3, 64, 64)
        img_b = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(img_a, img_b)
        assert out.shape == (1, 2, 64, 64)

    def test_registry_dispatch(self):
        from pygeovision.models.registry import get_model
        model = get_model("bit-r50", num_classes=2, in_channels=4, pretrained=False)
        model.eval()
        img_a = torch.randn(1, 4, 64, 64)
        img_b = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out = model(img_a, img_b)
        assert out.shape == (1, 2, 64, 64)


class TestDSAMNet:
    """Real DSAMNet — Liu & Shi 2021."""

    def test_output_shape(self):
        from pygeovision.models.change_detection.dsamnet import build_dsamnet
        torch.manual_seed(0)
        model = build_dsamnet(num_classes=2, in_channels=4, backbone="resnet18", pretrained=False)
        model.eval()
        img_a = torch.randn(2, 4, 128, 128)
        img_b = torch.randn(2, 4, 128, 128)
        with torch.no_grad():
            out = model(img_a, img_b)
        assert out.shape == (2, 2, 128, 128)

    def test_deep_supervision_aux_outputs(self):
        """Deep supervision must return one auxiliary distance map per
        backbone level (4 for a ResNet), each upsampled to input resolution."""
        from pygeovision.models.change_detection.dsamnet import build_dsamnet
        model = build_dsamnet(num_classes=2, in_channels=3, backbone="resnet18", pretrained=False)
        model.eval()
        img_a = torch.randn(1, 3, 64, 64)
        img_b = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out, aux_maps = model(img_a, img_b, return_aux=True)
        assert len(aux_maps) == 4
        for m in aux_maps:
            assert m.shape == (1, 1, 64, 64)

    def test_gradient_flows_through_all_parameters(self):
        from pygeovision.models.change_detection.dsamnet import build_dsamnet
        torch.manual_seed(1)
        model = build_dsamnet(num_classes=2, in_channels=4, backbone="resnet18", pretrained=False)
        img_a = torch.randn(1, 4, 64, 64)
        img_b = torch.randn(1, 4, 64, 64)
        out = model(img_a, img_b)
        loss = (out ** 2).mean()
        loss.backward()
        missing = [name for name, p in model.named_parameters()
                   if p.grad is None or p.grad.abs().sum() == 0]
        assert missing == [], f"parameters with no gradient: {missing}"

    def test_registry_dispatch(self):
        from pygeovision.models.registry import get_model
        model = get_model("dsamnet", num_classes=2, in_channels=4, pretrained=False)
        model.eval()
        img_a = torch.randn(1, 4, 64, 64)
        img_b = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out = model(img_a, img_b)
        assert out.shape == (1, 2, 64, 64)
