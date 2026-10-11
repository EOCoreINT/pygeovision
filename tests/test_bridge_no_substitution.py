"""Operations with no PyGeoVision implementation must fail, not return the input or another operation's output."""
import pytest

from pygeovision.core.exceptions import PreprocessingError
from pygeovision.data.pgf_bridge import PyGeoFetchBridge


@pytest.fixture()
def bridge():
    b = PyGeoFetchBridge()
    b._pgf_v2 = False          # as on an install without the pygeofetch processing engine
    return b


@pytest.mark.parametrize("call", [
    lambda b: b.preprocess_atmos("scene.tif"),
    lambda b: b.preprocess_pansharpen("pan.tif", "ms.tif"),
    lambda b: b.preprocess_tile("scene.tif"),
    lambda b: b.preprocess_topo_correct("scene.tif", "dem.tif"),
    lambda b: b.preprocess_cloud_fill("cloudy.tif", ["a.tif"]),
])
def test_missing_engine_raises(bridge, call):
    with pytest.raises(PreprocessingError, match="pygeofetch"):
        call(bridge)
