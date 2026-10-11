"""A misspelt early-stopping / checkpoint metric must fail loudly, not stop the run silently."""
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
    return TrainingConfig(output_dir=str(tmp_path), num_classes=2, in_channels=3, max_epochs=2, batch_size=2,
                          num_workers=0, pin_memory=False, persistent_workers=False, devices="cpu",
                          augment=False, warmup_epochs=0, **kw)


def test_unknown_early_stopping_metric_raises(tmp_path):
    with pytest.raises(ValueError, match="early_stopping_metric='val_iuo'"):
        GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, early_stopping_metric="val_iuo")).fit(_Ds(), _Ds())


def test_unknown_checkpoint_metric_raises(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_metric='nope'"):
        GeoTrainer(torch.nn.Conv2d(3, 2, 1), _cfg(tmp_path, checkpoint_metric="nope")).fit(_Ds(), _Ds())


def test_per_class_metric_is_accepted(tmp_path):
    res = GeoTrainer(torch.nn.Conv2d(3, 2, 1),
                     _cfg(tmp_path, early_stopping_metric="val_iou_class_1",
                          checkpoint_metric="val_iou_class_1")).fit(_Ds(), _Ds())
    assert res["best_metric_name"] == "val_iou_class_1"
