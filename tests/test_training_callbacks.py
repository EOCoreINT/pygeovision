"""pygeovision.training.callbacks: the checkpoint callback must survive an epoch worse than the k best."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from pygeovision.training.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402


def _trainer():
    model = torch.nn.Linear(2, 2)
    return types.SimpleNamespace(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))


def test_checkpoint_keeps_top_k_and_last_even_when_an_epoch_is_worse(tmp_path):
    cb = ModelCheckpoint(str(tmp_path), monitor="score", mode="max", save_top_k=2, save_last=True)
    tr = _trainer()
    for epoch, score in enumerate([0.5, 0.9, 0.8, 0.1, 0.2], start=1):    # epochs 4 and 5 are worse than the two best
        cb.on_epoch_end(tr, epoch, {"score": score})
    names = sorted(p.name for p in tmp_path.glob("*.pth"))
    assert names == ["epoch002.pth", "epoch003.pth", "last.pth"]
    assert cb.best_model_path.endswith("epoch002.pth")
    last = torch.load(tmp_path / "last.pth", weights_only=False)
    assert last["epoch"] == 5                       # last.pth is the final epoch, not the best one


def test_checkpoint_mode_min_keeps_lowest(tmp_path):
    cb = ModelCheckpoint(str(tmp_path), monitor="loss", mode="min", save_top_k=1, save_last=False)
    tr = _trainer()
    for epoch, loss in enumerate([0.9, 0.3, 0.6], start=1):
        cb.on_epoch_end(tr, epoch, {"loss": loss})
    assert [p.name for p in tmp_path.glob("*.pth")] == ["epoch002.pth"]


def test_early_stopping_restores_best_weights():
    tr = _trainer()
    cb = EarlyStopping(monitor="score", patience=2, mode="max")
    cb.on_epoch_end(tr, 1, {"score": 0.5})
    best = {k: v.clone() for k, v in tr.model.state_dict().items()}
    with torch.no_grad():
        for p in tr.model.parameters():
            p.add_(1.0)                              # the model drifts after the best epoch
    cb.on_epoch_end(tr, 2, {"score": 0.4})
    cb.on_epoch_end(tr, 3, {"score": 0.3})
    assert cb.stop_training and cb.stopped_epoch == 3
    for k, v in tr.model.state_dict().items():
        assert torch.equal(v, best[k])
