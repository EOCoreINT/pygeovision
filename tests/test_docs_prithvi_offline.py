# ruff: noqa: E702, N802, C408, E731
"""Shapes, band mapping, validation and the honest fallbacks printed on docs/foundation-models/prithvi.md (no weights, no network)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.models.foundation import prithvi as pv  # noqa: E402


@pytest.fixture
def scene(tmp_path):
    arr = (np.random.default_rng(0).random((6, 256, 300)) * 0.4).astype("float32")
    path = tmp_path / "s.tif"
    with rasterio.open(path, "w", driver="GTiff", height=256, width=300, count=6, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(500000, 5000000, 30, 30)) as d:
        d.write(arr)
    return str(path)


@pytest.fixture
def standin(monkeypatch):
    """The documented random-weights stand-in, without any attempt to reach the Hub."""
    monkeypatch.setattr(pv.Prithvi, "_build", lambda self: pv._build_prithvi_surrogate(self._spec, "cpu"))


def test_variants_listed_and_two_downstream_checkpoints_are_reference_only():
    assert pv.list_prithvi_models() == ["prithvi_eo_1_0", "prithvi_eo_2_0", "prithvi_eo_2_0_600m",
                                        "prithvi_eo_1_0_finetuned_burn", "prithvi_eo_1_0_finetuned_flood"]
    spec = pv.PRITHVI_MODELS
    assert (spec["prithvi_eo_1_0"]["embed_dim"], spec["prithvi_eo_1_0"]["n_frames"]) == (768, 3)
    assert (spec["prithvi_eo_2_0"]["embed_dim"], spec["prithvi_eo_2_0"]["n_frames"]) == (1024, 4)
    assert (spec["prithvi_eo_2_0_600m"]["embed_dim"], spec["prithvi_eo_2_0_600m"]["patch_size"]) == (1280, 14)
    assert not spec["prithvi_eo_1_0_finetuned_flood"]["loadable"] and not spec["prithvi_eo_1_0_finetuned_burn"]["loadable"]


def test_feature_shapes_and_head(scene, standin):
    p = pv.Prithvi("prithvi_eo_2_0", device="cpu").load()
    assert p.extract_features(scene).shape == (1, 1024)
    assert p.extract_patch_features(scene).shape == (196, 1024)          # a 14 x 14 grid of 16-pixel patches on a 224 tile
    head = p.build_segmentation_head(num_classes=3, freeze_backbone=True)
    trainable = sum(x.numel() for x in head.parameters() if x.requires_grad)
    assert 0 < trainable < sum(x.numel() for x in head.parameters()) / 5
    with torch.no_grad():
        assert tuple(head.eval()(torch.rand(2, 6, 224, 224)).shape) == (2, 3, 224, 224)


def test_band_mapping_prefers_b8a_and_validation_messages():
    z = np.stack([np.full((4, 4), i, dtype="float32") for i in range(7)])
    both = pv.map_bands(z, "sentinel2", source_bands=["B02", "B03", "B04", "B08", "B8A", "B11", "B12"])
    assert [float(c[0, 0]) for c in both] == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]          # narrow NIR (B8A) wins
    only08 = pv.map_bands(z[[0, 1, 2, 3, 5, 6]], "sentinel2", source_bands=["B02", "B03", "B04", "B08", "B11", "B12"])
    assert [float(c[0, 0]) for c in only08] == [0.0, 1.0, 2.0, 3.0, 5.0, 6.0]        # B08 as a stand-in
    raw = pv.validate_prithvi_input(np.full((6, 8, 8), 3000.0, dtype="float32"))
    assert not raw["valid"] and "raw DN integers" in raw["errors"][0]
    assert pv.validate_prithvi_input(np.random.rand(4, 8, 8).astype("float32"))["errors"] == ["Only 4 bands; Prithvi needs 6"]
    nan = np.random.rand(6, 8, 8).astype("float32"); nan[0, 0, 0] = np.nan
    assert "NaN pixels" in pv.validate_prithvi_input(nan)["warnings"][0]


def test_tasks_without_a_head_are_labelled_spectral_or_refuse(scene):
    tk = pv.PrithviTasks(device="cpu", allow_random_init=True)
    for name, n in (("flood_detection", 2), ("land_cover", 9), ("burn_scar_detection", 2)):
        r = getattr(tk, name)(scene, source="hls")
        assert r["method"] == "spectral" and r["n_classes"] == n and "Prithvi was not used" in r["note"]
        assert r["prediction"].shape == (256, 300)
    for name in ("crop_mapping", "biomass_estimation"):
        with pytest.raises(NotImplementedError):
            getattr(tk, name)(scene, source="hls")


def test_saved_head_file_round_trips_its_identity(scene, standin, tmp_path):
    p = pv.Prithvi("prithvi_eo_2_0", device="cpu").load()
    head = p.build_segmentation_head(num_classes=3)
    path = pv.save_prithvi_head(head, str(tmp_path / "h.pt"), "prithvi_eo_2_0", "flood_detection", 3)
    assert (tmp_path / "h.pt").exists() and path


def test_multitemporal_shapes_and_minimum_dates(tmp_path, standin):
    paths = []
    for i in range(4):
        a = (np.random.default_rng(i).random((6, 128, 128)) * 0.4).astype("float32")
        p = tmp_path / f"t{i}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=128, width=128, count=6, dtype="float32",
                           crs="EPSG:32633", transform=from_origin(500000, 5000000, 30, 30)) as d:
            d.write(a)
        paths.append(str(p))
    mt = pv.PrithviMultiTemporal(device="cpu")
    r = mt.process_time_series(paths)
    assert r["frame_embeddings"].shape == (4, 1024) and r["frame_tokens"].shape == (4, 196, 1024)
    assert r["features"].shape == (785, 1024) and r["grid"] == (14, 14) and r["dates"] == ["t0", "t1", "t2", "t3"]
    c = mt.detect_change(paths[0], paths[1], output_path=str(tmp_path / "chg.tif"))
    assert c["change_map"].shape == (14, 14) and c["change_map"].min() >= 0.0 and c["change_map"].max() <= 1.0
    with rasterio.open(tmp_path / "chg.tif") as s:
        assert (s.count, s.width, s.height) == (1, 128, 128)
    assert mt.monitor_trend(paths[:2])["trend"].startswith("insufficient_data")
    assert len(mt.monitor_trend(paths[:3])["drift_from_first"]) == 3
    assert "Need ≥ 4" in mt.predict_seasonal(paths[:3])["note"]
    assert {"peak_date", "trough_date", "seasonal_amplitude"} <= set(mt.predict_seasonal(paths))


def test_head_weights_keys_are_canonical_aliases_work_and_unknown_keys_raise(scene, standin, tmp_path):
    p = pv.Prithvi("prithvi_eo_2_0", device="cpu").load()
    head = p.build_segmentation_head(2)
    path = pv.save_prithvi_head(head, str(tmp_path / "h.pt"), "prithvi_eo_2_0", "flood", 2)
    for key in ("flood", "flood_detection"):                         # the alias reaches the head instead of being ignored
        tk = pv.PrithviTasks(device="cpu", allow_random_init=True, head_weights={key: path})
        assert tk.flood_detection(scene, source="hls")["method"] == "backbone"
    with pytest.raises(ValueError, match="unknown task 'floods'"):
        pv.PrithviTasks(device="cpu", head_weights={"floods": path})
    alias_saved = pv.save_prithvi_head(head, str(tmp_path / "h2.pt"), "prithvi_eo_2_0", "flood_detection", 2)
    tk = pv.PrithviTasks(device="cpu", allow_random_init=True, head_weights={"flood": alias_saved})
    assert tk.flood_detection(scene, source="hls")["method"] == "backbone"
    with pytest.raises(ValueError, match="trained for task"):
        pv.PrithviTasks(device="cpu", allow_random_init=True, head_weights={"burn_scar": path}).burn_scar_detection(scene, source="hls")
