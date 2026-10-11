"""The configuration file and PYGEOVISION_* variables must reach the objects that use them."""
import pytest

from pygeovision import PyGeoVision


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for k in ("PYGEOVISION_GPU_DEVICE", "PYGEOVISION_LOG_LEVEL", "PYGEOVISION_TRAINING_BATCH_SIZE"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_user_config_sets_providers_and_cache_ttl(home):
    (home / ".pygeovision").mkdir()
    (home / ".pygeovision" / "config.yaml").write_text(
        "pygeofetch:\n  default_providers: [planetary_computer]\n  cache_ttl_seconds: 120\n"
    )
    c = PyGeoVision()
    assert c.data.default_providers == ["planetary_computer"]
    assert c.data._cache_ttl == 120
    assert c.data._resolve_providers(None, None, None) == ["planetary_computer"]


def test_project_file_overrides_user_file_and_env_overrides_both(home, monkeypatch):
    (home / ".pygeovision").mkdir()
    (home / ".pygeovision" / "config.yaml").write_text("training:\n  batch_size: 8\n")
    (home / ".pygeovision.yaml").write_text("training:\n  batch_size: 4\n")
    assert PyGeoVision().config.training.batch_size == 4
    monkeypatch.setenv("PYGEOVISION_TRAINING_BATCH_SIZE", "2")
    assert PyGeoVision().config.training.batch_size == 2


def test_defaults_without_any_file(home):
    c = PyGeoVision()
    assert c.data._resolve_providers(None, None, None) == ["planetary_computer", "aws_earth", "element84"]
    assert c.data._cache_ttl == 3600
