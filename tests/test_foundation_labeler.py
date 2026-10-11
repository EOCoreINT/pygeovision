"""FoundationModelLabeler: registration of labels with pixels, no silent substitution, no fake embeddings.

The embedding model is replaced by a deterministic stand-in whose embedding of a patch is its mean
brightness, so the tests can say exactly which pixels must receive which label.
"""
import ast
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("sklearn")

from rasterio.transform import from_origin  # noqa: E402

from pygeovision.ai.labeling.foundation_labeler import FoundationModelLabeler  # noqa: E402


def _tif(path, arr, crs="EPSG:32633"):
    n, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=n, dtype="float32",
                       crs=crs, transform=from_origin(500000, 4000000, 10, 10)) as dst:
        dst.write(arr.astype("float32"))
    return pathlib.Path(path)


class BrightnessModel:
    """Stand-in for Prithvi: token = mean reflectance of its patch (D=1)."""

    def __init__(self, patch):
        self.patch = patch
        self._model = self        # the labeler calls self._model._model(pixel_values=...)

    def __call__(self, pixel_values):
        x = pixel_values.float()
        b, c, h, w = x.shape
        p = self.patch
        tok = x.mean(1).reshape(b, h // p, p, w // p, p).mean((2, 4))          # (B, gh, gw)
        return SimpleNamespace(grid=(h // p, w // p),
                               frame_tokens=tok.reshape(b, 1, -1, 1))


def _labeler(size=64, patch=16, **kw):
    lab = FoundationModelLabeler(model="prithvi", device="cpu", num_classes=2, **kw)
    lab._model = BrightnessModel(patch)
    lab._model_info = {**lab._model_info, "image_size": size, "patch_size": patch}
    return lab


class TestRegistration:
    def test_labels_land_on_the_pixels_they_describe(self, tmp_path):
        img = np.full((6, 100, 70), 0.05, "float32")           # not a multiple of the patch or tile
        img[:, 20:48, 32:64] = 0.60                             # a bright block
        tif = _tif(tmp_path / "a.tif", img)
        out = tmp_path / "o.tif"
        res = _labeler().label_tile(tif, None, out)
        assert res.success, res.error
        with rasterio.open(out) as src:
            lbl = src.read(1)
            assert lbl.shape == (100, 70) and src.crs == rasterio.open(tif).crs
        bright = lbl[34, 48]
        assert lbl[2, 2] != bright and lbl[95, 5] != bright     # background cells differ
        # the labelled block must hug the real block to within one 16-px patch
        ys, xs = np.nonzero(lbl == bright)
        assert abs(ys.min() - 20) <= 16 and abs(ys.max() - 47) <= 16
        assert abs(xs.min() - 32) <= 16 and abs(xs.max() - 63) <= 16
        # cells are 16x16-aligned: labels are constant within each patch
        for r in range(0, 96, 16):
            for c in range(0, 64, 16):
                assert len(np.unique(lbl[r:r + 16, c:c + 16])) == 1

    def test_upsample_covers_exactly_the_cell_footprint_and_crops_the_edge(self):
        grid = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)          # 2 rows x 3 cols
        out = FoundationModelLabeler._upsample_patch_labels(grid, (2, 3), cell_px=16, height=20, width=40)
        assert out.shape == (20, 40)
        assert out[0, 0] == 1 and out[0, 16] == 2 and out[0, 32] == 3
        assert out[16, 0] == 4 and out[19, 39] == 6 and out[15, 31] == 2

    def test_edge_tiles_are_padded_by_replication_not_zeros(self, tmp_path):
        seen = []

        class Spy(BrightnessModel):
            def __call__(self, pixel_values):
                seen.append(pixel_values.clone())
                return super().__call__(pixel_values)

        lab = _labeler()
        lab._model = Spy(16)
        img = np.full((6, 40, 40), 0.5, "float32")
        lab.label_tile(_tif(tmp_path / "a.tif", img), None, tmp_path / "o.tif")
        assert all(float(t.min()) == pytest.approx(0.5) for t in seen)   # no zero-filled border


class TestNoSilentSubstitution:
    def test_inference_error_fails_the_tile_instead_of_clustering_zeros(self, tmp_path):
        class Broken(BrightnessModel):
            def __call__(self, pixel_values):
                raise RuntimeError("CUDA out of memory")

        lab = _labeler()
        lab._model = Broken(16)
        out = tmp_path / "o.tif"
        res = lab.label_tile(_tif(tmp_path / "a.tif", np.full((6, 64, 64), 0.3, "float32")), None, out)
        assert res.success is False and "CUDA out of memory" in res.error
        assert not out.exists()

    def test_load_failure_names_the_model_and_substitutes_nothing(self, tmp_path, monkeypatch):
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
        lab = FoundationModelLabeler(model="prithvi", device="cpu")
        res = lab.label_tile(_tif(tmp_path / "a.tif", np.full((6, 64, 64), 0.3, "float32")), None,
                             tmp_path / "o.tif")
        assert res.success is False and "prithvi" in res.error and "Nothing was substituted" in res.error

    @pytest.mark.parametrize("name", ["clay", "satmae"])
    def test_models_without_a_verified_loader_are_refused(self, name):
        with pytest.raises(NotImplementedError):
            FoundationModelLabeler(model=name)

    def test_unknown_model_task_and_misplaced_prompts(self):
        with pytest.raises(ValueError):
            FoundationModelLabeler(model="nope")
        with pytest.raises(NotImplementedError):
            FoundationModelLabeler(task="embedding")
        with pytest.raises(ValueError, match="remote_clip"):
            FoundationModelLabeler(model="prithvi", text_prompts=["water"])

    def test_wrong_band_count_is_an_error_not_padding(self, tmp_path):
        res = _labeler().label_tile(_tif(tmp_path / "a.tif", np.full((3, 64, 64), 0.3, "float32")), None,
                                    tmp_path / "o.tif")
        assert res.success is False and "needs 6 bands" in res.error


class TestRemoteClipPath:
    def test_text_prompts_label_cells_and_names_are_reported(self, tmp_path):
        class FakeClip:
            def embed_pil(self, img):
                a = np.asarray(img).astype(float)
                v = np.array([a[..., 0].mean(), a[..., 2].mean()]) + 1e-3     # red-ness, blue-ness
                return v / np.linalg.norm(v)

            def embed_text(self, t):
                v = np.array([1.0, 0.0]) if t == "red thing" else np.array([0.0, 1.0])
                return v

        lab = FoundationModelLabeler(model="remote_clip", device="cpu", text_prompts=["red thing", "blue thing"])
        lab._model = FakeClip()
        lab._model_info = {**lab._model_info, "image_size": 32, "patch_size": 32}
        img = np.zeros((3, 64, 64), "float32")
        img[0, :, :32] = 1.0                                   # left half red
        img[2, :, 32:] = 1.0                                   # right half blue
        res = lab.label_tile(_tif(tmp_path / "a.tif", img), None, tmp_path / "o.tif")
        assert res.success, res.error
        with rasterio.open(tmp_path / "o.tif") as src:
            lbl = src.read(1)
        assert set(np.unique(lbl[:, :32])) == {0} and set(np.unique(lbl[:, 32:])) == {1}
        assert res.metadata["class_names"] == ["red thing", "blue thing"] and res.metadata["cell_px"] == 32


def test_no_labeler_passes_a_path_to_the_tile_oriented_writer():
    """`_write_label_geotiff(mask, tile, output_dir)` takes TileMetadata + a directory; five labelers
    called it with (mask, output_path, profile) and so never produced a file."""
    root = pathlib.Path(__file__).resolve().parents[1] / "pygeovision" / "ai" / "labeling"
    offenders = []
    for py in root.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_write_label_geotiff" and len(node.args) >= 2
                    and ast.unparse(node.args[1]) in ("output_path", "label_path")):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, offenders


def test_label_studio_leaves_no_all_zero_label_when_there_is_no_task(tmp_path, monkeypatch):
    from pygeovision.ai.labeling.label_studio import LabelStudioLabeler
    lab = LabelStudioLabeler.__new__(LabelStudioLabeler)
    lab.config = SimpleNamespace(project_id=7)
    monkeypatch.setattr(lab, "_find_task_for_tile", lambda p: None, raising=False)
    tif = _tif(tmp_path / "a.tif", np.zeros((1, 16, 16), "float32"))
    out = tmp_path / "o.tif"
    res = lab.label_tile(tif, None, out)
    assert res.success is False and res.error == "no_task_found" and not out.exists()


# ---------------------------------------------------------------------------------------------
# Labeler contract: LabelingResult fields, label_tiles dispatch, Google Open Buildings WKT
# ---------------------------------------------------------------------------------------------
def _meta(tile_id, tif):
    from pygeovision.ai.data.dataset import TileMetadata
    return TileMetadata(tile_id=tile_id, source_file=tif, bounds=(500000, 3999360, 500640, 4000000),
                        crs="EPSG:32633", transform=[10, 0, 500000, 0, -10, 4000000],
                        row_off=0, col_off=0, height=64, width=64, bands=[1, 2, 3, 4, 5, 6],
                        extra={"tile_path": str(tif)})


class TestLabelTilesContract:
    def test_label_tiles_drives_path_oriented_labelers(self, tmp_path):
        tif = _tif(tmp_path / "chip.tif", np.full((6, 64, 64), 0.3, "float32"))
        out_dir = tmp_path / "labels"
        res = _labeler().label_tiles([_meta("t0", tif)], out_dir, task="segmentation", show_progress=False)
        assert len(res) == 1 and res[0].success, res[0].error
        assert res[0].tile_id == "t0" or res[0].tile_id.endswith("chip.tif")
        assert (out_dir / "t0.tif").exists()          # where TileDataset looks for it

    def test_missing_chip_is_an_error_result_not_a_crash(self, tmp_path):
        from pygeovision.ai.data.dataset import TileMetadata
        m = TileMetadata(tile_id="x", source_file=tmp_path / "s.tif", bounds=(0, 0, 1, 1), crs="EPSG:4326",
                         transform=[1, 0, 0, 0, -1, 1], row_off=0, col_off=0, height=1, width=1, bands=[1])
        res = _labeler().label_tiles([m], tmp_path / "l", show_progress=False)
        assert not res[0].success and "not on disk" in res[0].error

    def test_failed_results_never_point_at_a_file(self, tmp_path):
        res = _labeler().label_tile(_tif(tmp_path / "a.tif", np.zeros((3, 64, 64), "float32")), None,
                                    tmp_path / "o.tif")
        assert not res.success and res.label_path is None

    def test_tiling_records_the_chip_path(self, tmp_path):
        from pygeovision.ai.data.tiling import TilingConfig, TilingEngine
        src = _tif(tmp_path / "scene.tif", np.random.rand(3, 128, 128).astype("float32"))
        cfg = TilingConfig(tile_size=64, overlap=0, save_tiles=True, output_dir=tmp_path / "chips")
        (tmp_path / "chips").mkdir()
        tiles = TilingEngine(cfg).tile(src)
        assert tiles and all(pathlib.Path(t.extra["tile_path"]).exists() for t in tiles)


def test_google_labeler_reads_footprints_through_the_shared_reader(tmp_path, monkeypatch):
    """The labeler's own tile-URL/CSV code was wrong and is gone; WKT parsing and the confidence filter are tested
    in tests/test_reference_data.py against the shared reader, which the labeler now calls."""
    import geopandas as gpd
    from shapely.geometry import box

    from pygeovision.ai.labeling.google_buildings import GoogleBuildingsLabeler
    from pygeovision.data import footprints as fp
    seen = {}

    def fake(bbox, **kw):
        seen.update(kw, bbox=bbox)
        return gpd.GeoDataFrame({"confidence": [0.9]}, geometry=[box(0.4, 0.4, 0.6, 0.6)], crs="EPSG:4326")
    monkeypatch.setattr(fp, "google_footprints", fake)
    lab = GoogleBuildingsLabeler.__new__(GoogleBuildingsLabeler)
    lab.config = SimpleNamespace(cache_dir=tmp_path, request_timeout=5, confidence_threshold=0.7)
    got = lab._fetch_buildings((0.0, 0.0, 1.0, 1.0))
    assert len(got) == 1 and got[0]["confidence"] == 0.9
    assert seen["min_confidence"] == 0.7 and seen["bbox"] == (0.0, 0.0, 1.0, 1.0)
