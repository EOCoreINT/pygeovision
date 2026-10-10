"""TiledInference.predict_array: in-memory tiled inference that leaves features untouched."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from pygeovision.inference.tiled import TiledInference  # noqa: E402


def pointwise_model(in_ch=5, classes=2, seed=0):
    """1x1 convolution: each pixel's output depends only on that pixel, so ANY tiling must
    reproduce the whole-image forward pass exactly. That makes seams / offsets / edge padding
    bugs show up as numerical differences."""
    torch.manual_seed(seed)
    return torch.nn.Conv2d(in_ch, classes, 1)


def whole_image(model, image, activation="softmax"):
    with torch.no_grad():
        logits = model(torch.from_numpy(image)[None])
    return (torch.softmax(logits, 1) if activation == "softmax" else torch.sigmoid(logits))[0].numpy()


@pytest.mark.parametrize("shape,chip,overlap", [
    ((5, 70, 90), 32, 8),      # ragged edges
    ((5, 64, 64), 64, 0),      # exactly one chip
    ((5, 20, 25), 32, 8),      # smaller than a chip
    ((5, 100, 37), 32, 16),
])
def test_matches_a_single_forward_pass_for_any_tiling(shape, chip, overlap):
    rs = np.random.RandomState(1)
    image = rs.randn(*shape).astype("float32")
    model = pointwise_model(shape[0])
    got = TiledInference(model, chip_size=chip, overlap=overlap, num_classes=2, half_precision=False,
                         device="cpu").predict_array(image)
    assert got.shape == (2, shape[1], shape[2]) and got.dtype == np.float32
    np.testing.assert_allclose(got, whole_image(model, image), atol=1e-5)


def test_features_are_not_percentile_stretched():
    """infer() stretches every band to its 2-98 percentile range; predict_array must not."""
    rs = np.random.RandomState(2)
    image = (rs.rand(5, 40, 40) * 1000 + 5000).astype("float32")     # large offset: a stretch changes everything
    model = pointwise_model()
    got = TiledInference(model, chip_size=32, overlap=8, half_precision=False, device="cpu").predict_array(image)
    np.testing.assert_allclose(got, whole_image(model, image), atol=1e-5)


def test_nan_pixels_become_nan_value_like_in_training():
    rs = np.random.RandomState(3)
    image = rs.randn(5, 30, 30).astype("float32")
    image[:, 5:9, 5:9] = np.nan
    image[0, 20, 20] = np.inf
    model = pointwise_model()
    ti = TiledInference(model, chip_size=16, overlap=4, half_precision=False, device="cpu")
    clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    got = ti.predict_array(image)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, whole_image(model, clean), atol=1e-5)
    got1 = ti.predict_array(image, nan_value=1.0)
    clean1 = np.nan_to_num(image, nan=1.0, posinf=1.0, neginf=1.0)
    np.testing.assert_allclose(got1, whole_image(model, clean1), atol=1e-5)


def test_sigmoid_single_channel_model():
    rs = np.random.RandomState(4)
    image = rs.randn(5, 50, 50).astype("float32")
    model = pointwise_model(classes=1)
    got = TiledInference(model, chip_size=32, overlap=8, num_classes=1, activation="sigmoid",
                         half_precision=False, device="cpu").predict_array(image)
    assert got.shape == (1, 50, 50)
    np.testing.assert_allclose(got, whole_image(model, image, "sigmoid"), atol=1e-5)


def test_agrees_with_the_geotiff_path(tmp_path):
    """predict_array and infer(normalise=False) run the same blending."""
    from rasterio.transform import from_bounds
    rs = np.random.RandomState(5)
    image = rs.rand(5, 70, 90).astype("float32")
    path = tmp_path / "in.tif"
    with rasterio.open(path, "w", driver="GTiff", height=70, width=90, count=5, dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(0, 0, 1, 1, 90, 70)) as dst:
        dst.write(image)
    model = pointwise_model()
    ti = TiledInference(model, chip_size=32, overlap=8, half_precision=False, device="cpu")
    r = ti.infer(path, tmp_path / "out.tif", return_probabilities=True, normalise=False)
    assert r["success"]
    from_file = rasterio.open(tmp_path / "out.tif").read().astype("float32") / 255.0
    np.testing.assert_allclose(ti.predict_array(image), from_file, atol=1.5 / 255)


def test_wrong_band_count_is_a_clear_error():
    model = pointwise_model(in_ch=5)
    ti = TiledInference(model, chip_size=16, overlap=4, half_precision=False, device="cpu")
    with pytest.raises(ValueError, match="does not accept 3 input band"):
        ti.predict_array(np.zeros((3, 20, 20), dtype="float32"))


def test_rejects_non_3d_input_and_bad_overlap():
    ti = TiledInference(pointwise_model(), chip_size=16, overlap=4, half_precision=False, device="cpu")
    with pytest.raises(ValueError, match=r"\(bands, H, W\)"):
        ti.predict_array(np.zeros((20, 20), dtype="float32"))
    with pytest.raises(ValueError, match="overlap"):
        TiledInference(pointwise_model(), chip_size=16, overlap=16)
    with pytest.raises(ValueError, match="overlap"):
        TiledInference(pointwise_model(), chip_size=16, overlap=-1)


def test_does_not_modify_the_callers_array():
    image = np.full((5, 20, 20), np.nan, dtype="float32")
    TiledInference(pointwise_model(), chip_size=16, overlap=4, half_precision=False, device="cpu").predict_array(image)
    assert np.isnan(image).all()
