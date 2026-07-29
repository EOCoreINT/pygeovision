"""Tests for the fixed CLIPGeo (dual-backend: transformers vs open_clip)
and the registry's clip/moondream dispatch.

Real bugs found and fixed:
  1. RemoteCLIP/GeoRSCLIP were loaded via transformers.CLIPModel, but the
     real official checkpoints (chendelong/RemoteCLIP, Zilun/GeoRSCLIP)
     are raw open_clip-format state dicts, not transformers-compatible —
     this would have failed or produced garbage.
  2. The registry routed clip-family and moondream-family models through
     a generic AutoModel.from_pretrained() call instead of the dedicated,
     correct wrapper classes that already existed elsewhere in the codebase.

huggingface.co isn't reachable from this sandbox, so the open_clip
round-trip test below builds a real open_clip architecture directly,
saves ITS OWN weights, and has the loader "download" that exact file —
this exercises every real line of the loading/inference path except the
actual network call, with a completely genuine open_clip CLIP model.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
open_clip = pytest.importorskip("open_clip", reason="open_clip_torch not installed")

import numpy as np
from rasterio.transform import from_bounds


def _make_geotiff(path, bands=4, size=64):
    transform = from_bounds(818000, 615000, 818000 + size * 10, 615000 + size * 10, size, size)
    data = np.random.randint(0, 5000, (bands, size, size)).astype("uint16")
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands,
                        dtype="uint16", crs="EPSG:32630", transform=transform) as dst:
        dst.write(data)


class TestCLIPGeoBackendSelection:
    def test_remoteclip_uses_open_clip_backend(self):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo, _OPENCLIP_CHECKPOINTS
        assert "remoteclip-b32" in _OPENCLIP_CHECKPOINTS
        assert "remoteclip-l14" in _OPENCLIP_CHECKPOINTS

    def test_georsclip_uses_open_clip_backend(self):
        from pygeovision.advanced.vlm.clip_geo import _OPENCLIP_CHECKPOINTS
        assert "georsclip" in _OPENCLIP_CHECKPOINTS

    def test_standard_clip_uses_transformers_backend(self):
        from pygeovision.advanced.vlm.clip_geo import _TRANSFORMERS_HF_MODELS
        assert "openclip-b32" in _TRANSFORMERS_HF_MODELS
        assert "openclip-l14" in _TRANSFORMERS_HF_MODELS
        assert "clip-vit-b32" in _TRANSFORMERS_HF_MODELS

    def test_remoteclip_points_at_the_real_official_repo(self):
        """Regression test for the actual bug: the wrong hf_id
        ('BAAI/RemoteCLIP-ViT-B-32') pointed at a repo that either doesn't
        exist or isn't transformers-compatible. The real official release
        is chendelong/RemoteCLIP."""
        from pygeovision.advanced.vlm.clip_geo import _OPENCLIP_CHECKPOINTS
        arch, repo_id, filename = _OPENCLIP_CHECKPOINTS["remoteclip-b32"]
        assert repo_id == "chendelong/RemoteCLIP"


class TestCLIPGeoRealOpenCLIPRoundTrip:
    """Real end-to-end verification of the open_clip loading path using a
    genuine architecture (weights round-tripped through the real loader
    code, only the network download itself is mocked)."""

    def _patched_clip(self, monkeypatch, tmp_path, model_name="remoteclip-b32"):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo, _OPENCLIP_CHECKPOINTS
        arch = _OPENCLIP_CHECKPOINTS[model_name][0]
        real_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=False)
        ckpt_path = tmp_path / "fake_checkpoint.pt"
        torch.save(real_model.state_dict(), ckpt_path)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(ckpt_path))

        return CLIPGeo(model=model_name)

    def test_zero_shot_real_forward_pass(self, tmp_path, monkeypatch):
        clip = self._patched_clip(monkeypatch, tmp_path)
        img_path = tmp_path / "scene.tif"
        _make_geotiff(img_path)

        scores = clip.zero_shot(str(img_path), ["urban area", "farmland", "water body"])

        assert clip._backend == "open_clip"
        assert set(scores.keys()) == {"urban area", "farmland", "water body"}
        assert abs(sum(scores.values()) - 1.0) < 1e-3  # softmax sums to 1

    def test_embed_image_is_normalised(self, tmp_path, monkeypatch):
        clip = self._patched_clip(monkeypatch, tmp_path)
        img_path = tmp_path / "scene.tif"
        _make_geotiff(img_path)

        emb = clip.embed_image(str(img_path))
        assert emb.shape == (512,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-3

    def test_embed_text_is_normalised(self, tmp_path, monkeypatch):
        clip = self._patched_clip(monkeypatch, tmp_path)
        emb = clip.embed_text("a photo of a city")
        assert emb.shape == (512,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-3

    def test_georsclip_loading_path_also_works(self, tmp_path, monkeypatch):
        clip = self._patched_clip(monkeypatch, tmp_path, model_name="georsclip")
        img_path = tmp_path / "scene.tif"
        _make_geotiff(img_path)
        scores = clip.zero_shot(str(img_path), ["forest", "urban"])
        assert clip._backend == "open_clip"
        assert abs(sum(scores.values()) - 1.0) < 1e-3


class TestRegistryVLMDispatch:
    """Regression tests for the real bug: clip/moondream families were
    routed through a generic AutoModel.from_pretrained() call instead of
    the dedicated, correct wrapper classes."""

    def test_all_clip_variants_return_clipgeo(self):
        from pygeovision.models.registry import get_model
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        for name in ("clip-vit-b32", "remoteclip-b32", "remoteclip-l14",
                     "georsclip", "openclip-b32", "openclip-l14"):
            model = get_model(name, num_classes=2)
            assert isinstance(model, CLIPGeo), f"{name} -> {type(model)}"
            assert model.model_name == name

    def test_moondream_returns_moondreamgeo(self):
        from pygeovision.models.registry import get_model
        from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
        model = get_model("moondream2", num_classes=2)
        assert isinstance(model, MoondreamGeo)
