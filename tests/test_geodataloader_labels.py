"""Regression tests for GeoDataLoader.prepare label generation.

Real defects fixed:
  1. The per-tile results of ``label_tiles`` were thrown away: a labeler that failed (or skipped) every tile
     still produced a "ready" dataset with no labels, and the only trace was a log line per exception.
  2. ``preprocessing_steps`` built a ``GeoPreprocessor`` and discarded it, silently doing nothing.
  3. ``augment=True`` built the pipeline without the band count, so 4+-band tiles crashed in colour jitter.
  4. With fewer than three source scenes the split put every tile into "test" and none into "train".
"""
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from pygeovision.ai.data.dataloader import GeoDataLoader  # noqa: E402
from pygeovision.ai.labeling.base_labeler import LabelingResult  # noqa: E402
from pygeovision.core.exceptions import DatasetError  # noqa: E402


def _scene(path, bands=4, size=128):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands, dtype="uint16",
                       crs="EPSG:32632", transform=from_origin(500000, 5500000, 10, 10)) as d:
        d.write(np.random.randint(100, 3000, (bands, size, size)).astype("uint16"))
    return path


class _FakeLabeler:
    def __init__(self, mode):
        self.mode = mode

    def label_tiles(self, tiles, output_dir, task, **kw):
        out = []
        for t in tiles:
            if self.mode == "fail":
                out.append(LabelingResult(tile_id=t.tile_id, error="Overpass query failed: HTTP 429"))
            elif self.mode == "skip":
                out.append(LabelingResult(tile_id=t.tile_id, skipped=True, error="No OSM features found"))
            else:
                p = output_dir / f"{t.tile_id}.tif"
                p.write_bytes(b"x")
                out.append(LabelingResult(tile_id=t.tile_id, label_path=p))
        return out


def _loader(monkeypatch, mode):
    loader = GeoDataLoader(data_pipeline=None)
    monkeypatch.setattr(loader, "_get_labeler", lambda s: _FakeLabeler(mode))
    return loader


def test_all_tiles_failing_raises_with_the_reason(monkeypatch, tmp_path):
    src = _scene(tmp_path / "s.tif")
    with pytest.raises(DatasetError, match="HTTP 429"):
        _loader(monkeypatch, "fail").prepare([str(src)], labels="openstreetmap", tile_size=64, overlap=0,
                                            output_dir=tmp_path / "ds", augment=False)


def test_all_tiles_skipped_raises_with_the_reason(monkeypatch, tmp_path):
    src = _scene(tmp_path / "s.tif")
    with pytest.raises(DatasetError, match="No OSM features"):
        _loader(monkeypatch, "skip").prepare([str(src)], labels="openstreetmap", tile_size=64, overlap=0,
                                            output_dir=tmp_path / "ds", augment=False)


def test_successful_labelling_still_builds_the_dataset(monkeypatch, tmp_path, caplog):
    src = _scene(tmp_path / "s.tif")
    with caplog.at_level("INFO"):
        ds = _loader(monkeypatch, "ok").prepare([str(src)], labels="openstreetmap", tile_size=64, overlap=0,
                                               output_dir=tmp_path / "ds", augment=False)
    assert ds is not None and "labelled" in caplog.text


def test_preprocessing_steps_warn_that_they_are_not_applied(monkeypatch, tmp_path, caplog):
    src = _scene(tmp_path / "s.tif")
    with caplog.at_level("WARNING"):
        _loader(monkeypatch, "ok").prepare([str(src)], labels="openstreetmap", tile_size=64, overlap=0,
                                          output_dir=tmp_path / "ds", augment=False,
                                          preprocessing_steps=["normalize"])
    assert "not applied" in caplog.text


def test_augmentation_pipeline_knows_the_band_count(monkeypatch, tmp_path):
    pytest.importorskip("albumentations")
    src = _scene(tmp_path / "s.tif", bands=4)
    seen = {}
    from pygeovision.ai.data import dataloader as dl
    real = dl.GeoAugmentationPipeline.medium

    def spy(task="segmentation", n_channels=None):
        seen["n"] = n_channels
        return real(task=task, n_channels=n_channels)
    monkeypatch.setattr(dl.GeoAugmentationPipeline, "medium", staticmethod(spy))
    _loader(monkeypatch, "ok").prepare([str(src)], labels="openstreetmap", tile_size=64, overlap=0,
                                      output_dir=tmp_path / "ds", augment=True)
    assert seen["n"] == 4


# ── spatial split with few scenes ────────────────────────────────────────────
def _tile(i, src, row, col):
    from pygeovision.ai.data.dataset import TileMetadata

    return TileMetadata(tile_id=f"{src}_{i}", source_file=Path(f"{src}.tif"), bounds=(0, 0, 1, 1), crs="EPSG:4326",
                        transform=[1, 0, 0, 0, -1, 0], row_off=row, col_off=col, height=64, width=64, bands=[1])


def test_single_scene_split_leaves_training_data():
    from pygeovision.ai.data.dataloader import GeoDataLoader

    tiles = [_tile(r * 10 + c, "a", r * 64, c * 64) for r in range(10) for c in range(10)]
    split = GeoDataLoader._spatial_split(tiles, val_split=0.15, test_split=0.10, seed=0)
    counts = {k: sum(v == k for v in split.values()) for k in ("train", "val", "test")}
    assert counts["train"] > counts["val"] > 0 and counts["test"] > 0, counts
    assert sum(counts.values()) == 100
    rows = {name: {t.row_off for t in tiles if split[t.tile_id] == name} for name in counts}
    assert max(rows["train"]) < min(rows["val"]) < min(rows["test"])        # contiguous bands, no row shared


def test_two_scenes_each_split_into_bands():
    from pygeovision.ai.data.dataloader import GeoDataLoader

    tiles = [_tile(r, s, r * 64, 0) for s in "ab" for r in range(8)]
    split = GeoDataLoader._spatial_split(tiles, val_split=0.15, test_split=0.10, seed=0)
    assert {"train", "val", "test"} <= set(split.values())
    assert sum(v == "train" for v in split.values()) >= 8


def test_many_scenes_still_split_by_file():
    from pygeovision.ai.data.dataloader import GeoDataLoader

    tiles = [_tile(i, f"s{s}", 0, i * 64) for s in range(10) for i in range(3)]
    split = GeoDataLoader._spatial_split(tiles, val_split=0.2, test_split=0.1, seed=1)
    by_scene = {}
    for t in tiles:
        by_scene.setdefault(str(t.source_file), set()).add(split[t.tile_id])
    assert all(len(v) == 1 for v in by_scene.values())          # a scene is never in two splits


def test_tiny_scene_is_all_train():
    from pygeovision.ai.data.dataloader import GeoDataLoader

    tiles = [_tile(0, "a", 0, 0), _tile(1, "a", 0, 64)]
    assert set(GeoDataLoader._spatial_split(tiles, 0.15, 0.1, 0).values()) == {"train"}
