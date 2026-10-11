"""VLM image loading, GeoImageRetrieval index handling and CLI error messages (no model downloads)."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.models.vlm._image import IMAGE_SUFFIXES, list_images, load_rgb_pil  # noqa: E402


def _write(path, arr, nodata=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:32630", transform=from_origin(0, 64, 1, 1), nodata=nodata) as dst:
        dst.write(arr)
    return str(path)


class TestLoadRgb:
    def test_three_bands_keep_colour_balance(self, tmp_path):
        a = np.zeros((3, 16, 16), "float32")
        a[0], a[1], a[2] = 1000, 500, 100
        a += np.random.default_rng(0).uniform(0, 50, a.shape).astype("float32")
        img = np.asarray(load_rgb_pil(_write(tmp_path / "a.tif", a)))
        assert img.shape == (16, 16, 3)
        # joint stretch: red stays brighter than green, green brighter than blue
        assert img[..., 0].mean() > img[..., 1].mean() > img[..., 2].mean()

    def test_per_band_stretch_removes_balance(self, tmp_path):
        a = np.zeros((3, 16, 16), "float32")
        a[0], a[1], a[2] = 1000, 500, 100
        a += np.random.default_rng(1).uniform(0, 50, a.shape).astype("float32")
        p = _write(tmp_path / "a.tif", a)
        img = np.asarray(load_rgb_pil(p, stretch="per_band")).astype(float)
        assert abs(img[..., 0].mean() - img[..., 2].mean()) < 20

    def test_one_and_two_bands(self, tmp_path):
        g = np.random.default_rng(2).uniform(0, 100, (1, 8, 8)).astype("float32")
        one = np.asarray(load_rgb_pil(_write(tmp_path / "one.tif", g)))
        assert (one[..., 0] == one[..., 1]).all() and (one[..., 1] == one[..., 2]).all()
        two = np.asarray(load_rgb_pil(_write(tmp_path / "two.tif", np.concatenate([g, g * 2]))))
        assert (two[..., 2] == two[..., 0]).all()

    def test_nodata_and_nan_are_black_and_do_not_distort_stretch(self, tmp_path):
        a = np.random.default_rng(3).uniform(100, 200, (3, 16, 16)).astype("float32")
        a[:, :4, :] = -9999
        a[:, 8, 8] = np.nan
        img = np.asarray(load_rgb_pil(_write(tmp_path / "n.tif", a, nodata=-9999)))
        assert (img[:4] == 0).all() and (img[8, 8] == 0).all()
        assert img[4:].max() > 200      # valid area uses the full range

    def test_all_nodata_image_is_black_not_an_error(self, tmp_path):
        a = np.full((3, 4, 4), -9999, "float32")
        img = np.asarray(load_rgb_pil(_write(tmp_path / "z.tif", a, nodata=-9999)))
        assert img.sum() == 0

    def test_list_images_recursive_case_insensitive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        for n in ("a.TIF", "sub/b.png", "sub/c.JPG", "d.txt", "e.tiff"):
            (tmp_path / n).write_bytes(b"x")
        names = [p.name for p in list_images(tmp_path)]
        assert set(names) == {"a.TIF", "b.png", "c.JPG", "e.tiff"}
        assert ".txt" not in IMAGE_SUFFIXES


class _StubClip:
    """Deterministic embeddings: a file's vector comes from the digit in its name."""
    def __init__(self, dim=4):
        self.dim = dim

    def embed_image(self, path):
        name = str(path)
        if "bad" in name:
            raise OSError("cannot read")
        v = np.zeros(self.dim)
        v[int(name.rsplit("img", 1)[1][0]) % self.dim] = 1.0
        return v

    def embed_text(self, text):
        v = np.zeros(self.dim)
        v[2] = 1.0
        return v


@pytest.fixture
def retr(monkeypatch):
    from pygeovision.advanced.vlm import retrieval
    from pygeovision.models.vlm import clip

    monkeypatch.setattr(clip, "CLIPGeo", lambda model="openclip-b32", device=None, bands=None: _StubClip())
    return retrieval.GeoImageRetrieval(model="openclip-b32")


def _folder(tmp_path):
    for n in ("img0.tif", "img1.png", "img2.JPG", "bad_img3.tif"):
        (tmp_path / n).write_bytes(b"x")
    return tmp_path


class TestRetrieval:
    def test_build_search_and_skipped(self, retr, tmp_path):
        d = _folder(tmp_path)
        assert retr.build_index(str(d)) == 3
        assert any("bad_img3" in k for k in retr.skipped) and "OSError" in next(iter(retr.skipped.values()))
        top = retr.search_by_text("anything", top_k=2)
        assert top[0]["path"].endswith("img2.JPG") and top[0]["score"] == 1.0
        assert len(top) == 2

    def test_search_by_image_excludes_itself(self, retr, tmp_path):
        d = _folder(tmp_path)
        retr.build_index(str(d))
        q = str(d / "img0.tif")
        res = retr.search_by_image(q, top_k=5)
        assert all(r["path"] != q for r in res) and len(res) == 2

    def test_save_load_roundtrip_and_model_mismatch(self, retr, tmp_path, monkeypatch):
        d = _folder(tmp_path)
        idx = tmp_path / "i.json"
        retr.build_index(str(d), save_path=str(idx))
        assert json.loads(idx.read_text())["model"] == "openclip-b32"
        retr._index = {}
        retr.load_index(str(idx))
        assert len(retr._index) == 3
        retr.model = "remoteclip-b32"
        with pytest.raises(ValueError, match="built with model 'openclip-b32'"):
            retr.load_index(str(idx))

    def test_legacy_plain_dict_index_still_loads(self, retr, tmp_path):
        p = tmp_path / "old.json"
        p.write_text(json.dumps({"a.tif": [1, 0, 0, 0]}))
        retr.load_index(str(p))
        assert retr.search_by_text("x")[0]["path"] == "a.tif"

    def test_empty_index_raises(self, retr):
        with pytest.raises(RuntimeError, match="empty"):
            retr.search_by_text("flood")

    def test_dimension_mismatch_is_explained(self, retr, tmp_path):
        p = tmp_path / "old.json"
        p.write_text(json.dumps({"a.tif": [1, 0]}))
        retr.load_index(str(p))
        with pytest.raises(ValueError, match="different model"):
            retr.search_by_text("x")


class TestCliVlmErrors:
    def test_missing_directory_is_a_click_error(self):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        r = CliRunner().invoke(cli, ["vlm", "search", "flood", "/nonexistent_dir_xyz"])
        assert r.exit_code != 0 and "Traceback" not in r.output
        assert "does not exist" in r.output.lower()

    def test_model_load_failure_is_one_line(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        from pygeovision.core.exceptions import ModelLoadError
        from pygeovision.models.vlm import moondream

        def boom(self, *a, **k):
            raise ModelLoadError("Could not load moondream: offline")
        monkeypatch.setattr(moondream.MoondreamGeo, "caption", boom)
        img = tmp_path / "a.tif"
        _write(img, np.ones((3, 4, 4), "uint8"))
        r = CliRunner().invoke(cli, ["vlm", "caption", str(img)])
        assert r.exit_code != 0 and "Traceback" not in r.output and "Could not load moondream" in r.output


class TestBands:
    def test_bands_select_and_reorder(self, tmp_path):
        a = np.zeros((4, 16, 16), "float32")
        a[0] = 100 + np.random.default_rng(0).uniform(0, 5, (16, 16))     # B2 blue (dim)
        a[1] = 400 + np.random.default_rng(1).uniform(0, 5, (16, 16))     # B3 green
        a[2] = 900 + np.random.default_rng(2).uniform(0, 5, (16, 16))     # B4 red (bright)
        a[3] = 3000
        p = _write(tmp_path / "s2.tif", a)
        default = np.asarray(load_rgb_pil(p))
        natural = np.asarray(load_rgb_pil(p, bands=(3, 2, 1)))
        assert default[..., 0].mean() < default[..., 2].mean()            # first three bands: red channel is B2 (blue)
        assert natural[..., 0].mean() > natural[..., 2].mean()

    def test_bad_band_numbers(self, tmp_path):
        p = _write(tmp_path / "x.tif", np.ones((3, 8, 8), "float32"))
        with pytest.raises(ValueError, match="3 band"):
            load_rgb_pil(p, bands=(1, 2, 4))
        with pytest.raises(ValueError):
            load_rgb_pil(p, bands=(0, 1, 2))

    def test_bands_reach_the_models_and_the_cli(self, tmp_path):
        from click.testing import CliRunner

        from pygeovision.cli.main import cli
        from pygeovision.models.vlm.clip import CLIPGeo
        from pygeovision.models.vlm.moondream import MoondreamGeo
        assert CLIPGeo(bands=(3, 2, 1)).bands == (3, 2, 1) and MoondreamGeo(bands=[3, 2, 1]).bands == (3, 2, 1)
        img = tmp_path / "a.tif"
        _write(img, np.ones((3, 4, 4), "uint8"))
        r = CliRunner().invoke(cli, ["vlm", "caption", str(img), "--bands", "1,x"])
        assert r.exit_code != 0 and "band numbers" in r.output
        r = CliRunner().invoke(cli, ["vlm", "caption", str(img), "--bands", "1,2,9"])
        assert r.exit_code != 0 and "does not fit" in r.output and "Traceback" not in r.output


def test_search_by_image_excludes_itself_even_with_a_different_path_spelling(retr, tmp_path, monkeypatch):
    d = _folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    retr.build_index(".")                                   # indexed as ./img0.tif style relative paths
    res = retr.search_by_image(str(d / "img0.tif"), top_k=5)    # queried by absolute path
    assert all(not r["path"].endswith("img0.tif") for r in res)
