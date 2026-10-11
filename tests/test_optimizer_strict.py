"""Unknown optimizer / scheduler names must fail, not turn into AdamW / cosine."""
import pytest

torch = pytest.importorskip("torch")

from pygeovision.ai.training.trainer import TrainingConfig
from pygeovision.training.optimizer import build_optimizer, build_scheduler


def _model():
    return torch.nn.Linear(4, 2)


def test_unknown_optimizer_is_an_error():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(_model(), TrainingConfig(optimizer="adamx"))


def test_lars_is_not_pretended():
    with pytest.raises(ValueError, match="LARS is not implemented"):
        build_optimizer(_model(), TrainingConfig(optimizer="lars"))


def test_lion_without_its_package_is_an_error_not_adamw(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "lion_pytorch":
            raise ImportError("no lion")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="lion-pytorch"):
        build_optimizer(_model(), TrainingConfig(optimizer="lion"))


def test_unknown_scheduler_is_an_error():
    opt = build_optimizer(_model(), TrainingConfig())
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(opt, TrainingConfig(scheduler="cosin"))


def test_none_scheduler_means_constant_rate():
    cfg = TrainingConfig(scheduler="none")
    assert build_scheduler(build_optimizer(_model(), cfg), cfg) is None


@pytest.mark.parametrize("name", ["cosine", "cosine_warmup", "warmup_cosine", "linear", "step", "plateau", "onecycle"])
def test_every_documented_scheduler_builds(name):
    cfg = TrainingConfig(scheduler=name, max_epochs=10)
    assert build_scheduler(build_optimizer(_model(), cfg), cfg, steps_per_epoch=20) is not None
