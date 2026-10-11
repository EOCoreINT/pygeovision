"""Without --cache-dir the download must land where get_model(pretrained=True) reads (the HF cache)."""
import sys
import types

from pygeovision.models.weights.downloader import WeightDownloader


def _fake_hub(monkeypatch, calls):
    mod = types.SimpleNamespace(
        snapshot_download=lambda **kw: calls.append(kw) or "/x",
        hf_hub_download=lambda **kw: calls.append(kw) or "/x",
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def test_default_uses_hf_cache(tmp_path, monkeypatch):
    calls = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setattr(WeightDownloader, "DEFAULT_CACHE", tmp_path / "private")
    WeightDownloader().download("org/model")
    assert calls[0]["cache_dir"] is None


def test_explicit_cache_is_respected(tmp_path, monkeypatch):
    calls = []
    _fake_hub(monkeypatch, calls)
    WeightDownloader(cache_dir=str(tmp_path / "mine")).download("org/model")
    assert calls[0]["cache_dir"] == str(tmp_path / "mine")
