"""Linear blending must not zero out the outermost pixels of the scene."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pygeovision.inference.tiled import TiledInference  # noqa: E402


class _AlwaysClassOne(torch.nn.Module):
    def forward(self, x):
        out = torch.zeros(x.shape[0], 2, *x.shape[2:])
        out[:, 1] = 5.0
        return out


@pytest.mark.parametrize("mode", ["gaussian", "linear", "constant"])
def test_every_pixel_gets_a_prediction(mode):
    img = np.random.rand(4, 200, 230).astype("float32")
    ti = TiledInference(_AlwaysClassOne(), chip_size=64, overlap=16, blend_mode=mode, num_classes=2, device="cpu")
    labels = ti.predict_array(img).argmax(0)
    assert (labels == 1).all(), f"{mode}: {(labels != 1).sum()} pixels lost their prediction"
