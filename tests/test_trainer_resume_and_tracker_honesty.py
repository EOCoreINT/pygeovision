"""Resuming with a different max_epochs warns; a tracker with no usable backend says so."""
import logging

import pytest

torch = pytest.importorskip("torch")

from pygeovision.training import GeoTrainer, TrainingConfig  # noqa: E402


class _Ds(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        x = torch.rand(3, 16, 16)
        return x, (x[0] > 0.5).long()


def _cfg(tmp_path, **kw):
    base = dict(output_dir=str(tmp_path), num_classes=2, in_channels=3, batch_size=2, num_workers=0,
                pin_memory=False, persistent_workers=False, devices="cpu", augment=False, warmup_epochs=0)
    base.update(kw)
    return TrainingConfig(**base)


def test_resume_with_changed_max_epochs_warns(tmp_path, caplog):
    GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, max_epochs=2)).fit(_Ds(), _Ds())
    with caplog.at_level(logging.WARNING):
        res = GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, max_epochs=3, resume=True)).fit(_Ds(), _Ds())
    assert res["resumed_from_epoch"] == 2
    assert "max_epochs=3" in caplog.text and "used 2" in caplog.text


def test_resume_with_same_max_epochs_is_quiet(tmp_path, caplog):
    GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, max_epochs=2)).fit(_Ds(), _Ds())
    with caplog.at_level(logging.WARNING):
        GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, max_epochs=2, resume=True)).fit(_Ds(), _Ds())
    assert "learning-rate schedule" not in caplog.text


def test_tracker_without_any_working_backend_is_reported(tmp_path, caplog, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mlflow", None)        # makes `import mlflow` fail
    with caplog.at_level(logging.WARNING):
        GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, max_epochs=1, tracker="mlflow")).fit(_Ds(), _Ds())
    assert "no backend could start" in caplog.text


def test_metric_fn_is_reported_as_unused(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, max_epochs=1)).fit(_Ds(), _Ds(), metric_fn=lambda *a: 0)
    assert "metric_fn" in caplog.text and "not used" in caplog.text
