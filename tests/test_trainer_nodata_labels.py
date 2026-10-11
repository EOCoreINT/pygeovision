"""Labels carrying the package nodata value 255 must not crash the default loss."""
import pytest

torch = pytest.importorskip("torch")

from pygeovision.ai.training.trainer import GeoTrainer, TrainingConfig


class _DS(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        x = torch.rand(3, 16, 16)
        y = torch.randint(0, 2, (16, 16))
        y[:4] = 255
        return x, y


def test_default_loss_ignores_255(tmp_path):
    model = torch.nn.Conv2d(3, 2, 1)
    cfg = TrainingConfig(output_dir=str(tmp_path), max_epochs=1, batch_size=4, num_workers=0,
                         pin_memory=False, persistent_workers=False, num_classes=2, in_channels=3,
                         augment=False, devices="cpu", warmup_epochs=0)
    res = GeoTrainer(model, cfg).fit(_DS(), _DS())
    assert res["epochs_trained"] == 1
    assert res["history"]["train_loss"][0] == res["history"]["train_loss"][0]  # finite, not NaN
