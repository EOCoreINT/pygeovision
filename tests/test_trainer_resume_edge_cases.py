# ruff: noqa: E702, N802, C408, E731
"""Behaviour documented on docs/training/resuming.md: what resume restores, refuses and does at the edges."""
import logging

import pytest

torch = pytest.importorskip("torch")

from pygeovision.training import GeoTrainer, TrainingConfig  # noqa: E402


class _Ds(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.rand(3, 16, 16, generator=g), torch.randint(0, 2, (16, 16), generator=g)


def _fit(path, model=None, **kw):
    base = dict(output_dir=str(path), num_classes=2, in_channels=3, batch_size=2, num_workers=0, pin_memory=False,
                persistent_workers=False, devices="cpu", augment=False, warmup_epochs=0)
    base.update(kw)
    return GeoTrainer(model or torch.nn.Conv2d(3, 2, 1), TrainingConfig(**base)).fit(_Ds(), _Ds())


def test_last_pth_holds_the_documented_keys(tmp_path):
    _fit(tmp_path, max_epochs=2)
    state = torch.load(tmp_path / "checkpoints" / "last.pth", weights_only=True)
    assert {"epoch", "model_state_dict", "optimizer_state_dict", "optimizer_class", "scheduler_state_dict",
            "scaler_state_dict", "history", "global_step", "early_stopping", "checkpoints", "metrics",
            "max_epochs"} <= set(state)
    assert state["epoch"] == 2 and state["optimizer_class"] == "AdamW"
    assert not (tmp_path / "checkpoints" / "last.pth.tmp").exists()


def test_resuming_a_finished_run_trains_nothing(tmp_path):
    _fit(tmp_path, max_epochs=3)
    res = _fit(tmp_path, max_epochs=3, resume=True)
    assert res["epochs_trained"] == 3 and res["resumed_from_epoch"] == 3 and len(res["history"]["val_loss"]) == 3


def test_resuming_an_early_stopped_run_trains_one_more_epoch(tmp_path):
    first = _fit(tmp_path, max_epochs=10, early_stopping_patience=1)
    res = _fit(tmp_path, max_epochs=10, early_stopping_patience=1, resume=True)
    assert res["resumed_from_epoch"] == first["epochs_trained"]
    assert res["epochs_trained"] == first["epochs_trained"] + 1


def test_resume_without_last_pth_starts_fresh(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        res = _fit(tmp_path, max_epochs=1, resume=True)
    assert res["resumed_from_epoch"] is None and "starting a fresh run" in caplog.text


def test_resume_with_save_last_off_has_nothing_to_resume(tmp_path):
    _fit(tmp_path, max_epochs=1, save_last=False)
    assert not (tmp_path / "checkpoints" / "last.pth").exists()
    assert _fit(tmp_path, max_epochs=1, save_last=False, resume=True)["resumed_from_epoch"] is None


def test_resume_into_a_different_model_is_refused(tmp_path):
    _fit(tmp_path, max_epochs=1)
    with pytest.raises(RuntimeError, match="could not be restored into this model"):
        _fit(tmp_path, model=torch.nn.Conv2d(3, 5, 1), num_classes=5, max_epochs=2, resume=True)


def test_resume_with_a_different_optimizer_is_refused_clearly(tmp_path):
    _fit(tmp_path, max_epochs=1, optimizer="adamw")
    with pytest.raises(RuntimeError, match="written with the AdamW optimizer.*builds SGD"):
        _fit(tmp_path, max_epochs=2, optimizer="sgd", resume=True)
