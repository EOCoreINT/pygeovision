"""CLIP / RemoteCLIP / Moondream: one implementation, honest loading, no network."""
import types

import pytest

torch = pytest.importorskip("torch")

from pygeovision.core.exceptions import ModelLoadError  # noqa: E402


class TestSingleImplementation:
    def test_old_import_paths_are_the_same_class(self):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo as OldClip
        from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo as OldMoondream
        from pygeovision.models.vlm.clip import CLIPGeo
        from pygeovision.models.vlm.moondream import MoondreamGeo
        assert OldClip is CLIPGeo and OldMoondream is MoondreamGeo

    def test_registry_clip_entries_resolve_to_a_real_backend(self):
        from pygeovision.models.registry import get_model, model_registry
        from pygeovision.models.vlm.clip import CLIPGeo
        names = model_registry.list(family="clip")
        assert len(names) >= 5
        for name in names:
            assert isinstance(get_model(name), CLIPGeo)
            assert name in CLIPGeo.HF_MODELS or name in CLIPGeo.OPENCLIP_CHECKPOINTS, name

    def test_remoteclip_points_at_the_open_clip_repo_that_exists(self):
        from pygeovision.models.vlm.clip import CLIPGeo
        for name in ("remoteclip-b32", "remoteclip-l14"):
            arch, repo, fname = CLIPGeo.OPENCLIP_CHECKPOINTS[name]
            assert repo == "chendelong/RemoteCLIP" and fname.startswith("RemoteCLIP-")


class TestRemoteClipLoadingIsStrict:
    def test_checkpoint_that_leaves_layers_random_is_refused(self, tmp_path, monkeypatch):
        """strict=False used to log a warning and return a half-random 'RemoteCLIP'."""
        open_clip = pytest.importorskip("open_clip")
        from pygeovision.models.vlm.clip import CLIPGeo

        ckpt = tmp_path / "bad.pt"
        torch.save({"visual.class_embedding": torch.zeros(3)}, ckpt)
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(ckpt))
        monkeypatch.setattr(open_clip, "create_model_and_transforms",
                            lambda arch, pretrained=False: (torch.nn.Linear(2, 2), None, None))
        with pytest.raises(ModelLoadError, match="randomly initialised"):
            CLIPGeo(model="remoteclip-b32")._load()


class TestMoondream:
    def _loaded(self, model):
        from pygeovision.models.vlm.moondream import MoondreamGeo
        md = MoondreamGeo()
        md._model = model
        md._load_pil = staticmethod(lambda p: "IMG")      # no raster needed
        return md

    def test_revision_is_pinned(self):
        from pygeovision.models.vlm.moondream import MoondreamGeo
        assert MoondreamGeo.REVISION and MoondreamGeo.REVISION != "main"
        assert MoondreamGeo(revision="2024-01-01").revision == "2024-01-01"

    def test_current_api(self):
        m = types.SimpleNamespace(
            caption=lambda img, length: {"caption": f"cap:{img}:{length}"},
            query=lambda img, q: {"answer": f"ans:{q}"})
        md = self._loaded(m)
        assert md.caption("x.tif") == "cap:IMG:normal"
        assert md.vqa("x.tif", "how many?") == "ans:how many?"

    def test_legacy_api_still_works(self):
        m = types.SimpleNamespace(encode_image=lambda img: "ENC",
                                  answer_question=lambda enc, q, tok: f"{enc}|{q}|{tok}")
        md = self._loaded(m)
        md._tokenizer = "TOK"
        assert md.vqa("x.tif", "q?") == "ENC|q?|TOK"
        assert "Describe" in md.caption("x.tif")

    def test_batch_reports_a_failing_file_instead_of_aborting(self, tmp_path):
        for n in ("a", "b"):
            (tmp_path / f"{n}.tif").write_bytes(b"")
        calls = []

        def cap(img, length):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("bad tile")
            return {"caption": "ok"}

        md = self._loaded(types.SimpleNamespace(caption=cap, query=lambda *a: {"answer": ""}))
        out = md.batch_caption(str(tmp_path))
        assert [r["caption"] for r in out] == [None, "ok"]
        assert "bad tile" in out[0]["error"]

    def test_hub_failure_is_a_model_load_error(self, monkeypatch):
        pytest.importorskip("transformers")
        import transformers

        from pygeovision.models.vlm.moondream import MoondreamGeo

        def _fail(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _fail)
        with pytest.raises(ModelLoadError, match="moondream2"):
            MoondreamGeo()._load()
