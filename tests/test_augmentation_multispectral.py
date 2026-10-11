"""Default training augmentation must work for multispectral (non-RGB) stacks."""
import logging

import numpy as np
import pytest

pytest.importorskip("albumentations")

from pygeovision.ai.data.augmentations import GeoAugmentationPipeline  # noqa: E402
from pygeovision.ai.training.trainer import (  # noqa: E402
    TrainingConfig,
    _AugmentedSegmentationDataset,
)


@pytest.mark.parametrize("bands", [1, 3, 4, 6, 9])
@pytest.mark.parametrize("mode", ["light", "medium", "heavy"])
def test_pipeline_never_fails_for_any_band_count(bands, mode):
    pipe = GeoAugmentationPipeline.for_task("segmentation", mode=mode, n_channels=bands)
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, bands), dtype=np.float32)
    mask = rng.integers(0, 3, (64, 64)).astype("int64")
    mask[:4] = 255
    for _ in range(150):
        out = pipe(image=img, mask=mask)
        assert out["image"].shape == img.shape
        assert out["mask"].shape == mask.shape
        assert set(np.unique(out["mask"])) <= {0, 1, 2, 255}


def test_trainer_wrapper_uses_band_count_of_the_data():
    import torch

    class Ds:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.rand(6, 32, 32), torch.randint(0, 2, (32, 32))

    ds = _AugmentedSegmentationDataset(Ds())
    for _ in range(150):
        img, msk = ds[0]
        assert tuple(img.shape) == (6, 32, 32) and tuple(msk.shape) == (32, 32)


def test_from_yaml_warns_about_unknown_keys(tmp_path, caplog):
    f = tmp_path / "c.yaml"
    f.write_text("max_epochs: 3\nlerning_rate: 0.1\n")
    with caplog.at_level(logging.WARNING):
        cfg = TrainingConfig.from_yaml(f)
    assert cfg.max_epochs == 3
    assert "lerning_rate" in caplog.text
