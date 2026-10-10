"""labeling.foundation.FoundationModelLabeler (DINOv2 + clustering): registration, nodata, honest failures.

A deterministic stand-in backbone replaces DINOv2: each patch's single feature is its mean brightness, so
the tests can state exactly which pixels must share a cluster.
"""
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("sklearn")

from rasterio.transform import from_origin  # noqa: E402

from pygeovision.labeling.foundation import NODATA_LABEL, FoundationModelLabeler  # noqa: E402

PATCH = 14


class FakeDino(torch.nn.Module):
    def __init__(self, n_prefix=1):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(patch_size=PATCH)
        self.n_prefix = n_prefix

    def forward(self, pixel_values):
        b, _, h, w = pixel_values.shape
        tok = pixel_values.mean(1).reshape(b, h // PATCH, PATCH, w // PATCH, PATCH).mean((2, 4))
        tok = tok.reshape(b, -1, 1)
        prefix = torch.zeros(b, self.n_prefix, 1)
        return SimpleNamespace(last_hidden_state=torch.cat([prefix, tok], 1))


def _tif(path, arr, nodata=None):
    n, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=n, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(500000, 4000000, 10, 10), nodata=nodata) as d:
        d.write(arr.astype("float32"))
    return pathlib.Path(path)


def _lab(n_prefix=1, **kw):
    lab = FoundationModelLabeler(device="cpu", **kw)
    lab._loaded = (FakeDino(n_prefix), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), PATCH)
    return lab


def _scene():
    img = np.full((3, 100, 70), 0.1, "float32")        # not a multiple of the patch or of tile_size
    img[:, 28:, 42:] = 0.9              # bright block, grid-aligned, running into the ragged bottom edge
    img += np.random.default_rng(0).normal(0, 0.002, img.shape).astype("float32")
    return img


@pytest.mark.parametrize("n_prefix", [1, 5])             # CLS only, or CLS + 4 register tokens
def test_clusters_land_on_the_pixels_they_describe(tmp_path, n_prefix):
    tif = _tif(tmp_path / "a.tif", _scene())
    res = _lab(n_prefix).pseudo_label(tif, tmp_path / "o.tif", n_classes=2, tile_size=56)
    assert res["success"], res
    with rasterio.open(tmp_path / "o.tif") as src, rasterio.open(tif) as ref:
        lbl = src.read(1)
        assert lbl.shape == (100, 70) and src.crs == ref.crs and src.transform == ref.transform
    bright, dark = lbl[40, 55], lbl[5, 5]
    assert bright != dark
    assert (lbl[28:, 42:] == bright).all()      # the block exactly, incl. the 2-px sliver at the bottom
    assert (lbl[:28] == dark).all() and (lbl[:, :42] == dark).all()   # everything else
    assert res["cell_px"] == PATCH and res["n_classes"] == 2


def test_nodata_is_left_unlabelled_and_not_clustered(tmp_path):
    img = _scene()
    img[:, :40, :] = -9999.0                                          # not a multiple of the patch
    tif = _tif(tmp_path / "a.tif", img, nodata=-9999.0)
    res = _lab().pseudo_label(tif, tmp_path / "o.tif", n_classes=2, tile_size=56)
    assert res["success"], res
    with rasterio.open(tmp_path / "o.tif") as src:
        lbl = src.read(1)
        assert src.nodata == NODATA_LABEL
    assert (lbl[:40] == NODATA_LABEL).all() and (lbl[40:] != NODATA_LABEL).all()
    assert res["valid_fraction"] == pytest.approx(60 / 100)


def test_clip_annotate_is_refused_not_ignored(tmp_path):
    with pytest.raises(NotImplementedError):
        _lab().pseudo_label(tmp_path / "x.tif", tmp_path / "o.tif", clip_annotate=True)


def test_patch_size_mismatch_is_an_error(tmp_path):
    tif = _tif(tmp_path / "a.tif", _scene())
    with pytest.raises(ValueError, match="uses 14"):
        _lab().pseudo_label(tif, tmp_path / "o.tif", patch_size=16)


def test_load_failure_substitutes_nothing(tmp_path, monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained",
                        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(OSError("offline"))))
    tif = _tif(tmp_path / "a.tif", _scene())
    res = FoundationModelLabeler(device="cpu").pseudo_label(tif, tmp_path / "o.tif")
    assert res["success"] is False and "Nothing was substituted" in res["error"]
    assert not (tmp_path / "o.tif").exists()


def test_rgb_bands_selects_and_validates(tmp_path):
    img = np.zeros((4, 56, 56), "float32")
    img[3, :, 28:] = 1.0                                              # only band 4 carries structure
    tif = _tif(tmp_path / "a.tif", img)
    ok = _lab(rgb_bands=(4, 4, 4)).pseudo_label(tif, tmp_path / "o.tif", n_classes=2, tile_size=56)
    assert ok["success"]
    with rasterio.open(tmp_path / "o.tif") as s:
        lbl = s.read(1)
    assert lbl[0, 0] != lbl[0, 55]
    bad = _lab(rgb_bands=(1, 2, 9)).pseudo_label(tif, tmp_path / "o2.tif")
    assert bad["success"] is False and "not available" in bad["error"]


def test_hdbscan_noise_is_unlabelled_not_a_class(tmp_path):
    hdbscan = pytest.importorskip("hdbscan")  # noqa: F841
    tif = _tif(tmp_path / "a.tif", _scene())
    res = _lab().pseudo_label(tif, tmp_path / "o.tif", clustering="hdbscan", tile_size=56)
    assert res["success"], res
    with rasterio.open(tmp_path / "o.tif") as s:
        lbl = s.read(1)
    classes = set(np.unique(lbl)) - {NODATA_LABEL}
    assert len(classes) == res["n_classes"]
