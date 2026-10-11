"""cfg.tracker must reach every backend it names, not only MLflow."""
import pytest

torch = pytest.importorskip("torch")

from pygeovision.ai.training.trainer import GeoTrainer, TrainingConfig
from pygeovision.training import experiment


class _DS(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.rand(3, 8, 8), torch.randint(0, 2, (8, 8))


class _Fake:
    instances = []

    def __init__(self, name, run_name="", backends=None, **kw):
        self.backends, self.logged, self.started, self.ended = backends, [], False, False
        _Fake.instances.append(self)

    def start_run(self, config=None):
        self.started = True
        return self

    def log(self, metrics, step=None):
        self.logged.append((step, dict(metrics)))

    def end_run(self):
        self.ended = True
        return {}


def _fit(tmp_path, tracker):
    cfg = TrainingConfig(output_dir=str(tmp_path), max_epochs=2, batch_size=2, num_workers=0, pin_memory=False,
                         persistent_workers=False, num_classes=2, in_channels=3, augment=False, devices="cpu",
                         warmup_epochs=0, tracker=tracker)
    return GeoTrainer(torch.nn.Conv2d(3, 2, 1), cfg).fit(_DS(), _DS())


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    _Fake.instances.clear()
    monkeypatch.setattr(experiment, "ExperimentTracker", _Fake)


def test_wandb_and_tensorboard_receive_metrics(tmp_path):
    _fit(tmp_path, "wandb,tensorboard")
    t = _Fake.instances[0]
    assert t.backends == ["wandb", "tensorboard"] and t.started and t.ended
    assert [s for s, _ in t.logged] == [1, 2]
    assert "val_loss" in t.logged[0][1] and all(isinstance(v, float) for v in t.logged[0][1].values())


def test_all_means_every_backend(tmp_path):
    _fit(tmp_path, "all")
    assert _Fake.instances[0].backends == ["mlflow", "wandb", "tensorboard"]


def test_none_starts_nothing_and_unknown_is_ignored(tmp_path):
    _fit(tmp_path, "none")
    _fit(tmp_path, "comet")
    assert _Fake.instances == []
