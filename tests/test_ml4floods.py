"""The ml4floods WF2_unetv2_bgriswirs integration: loader, post-processing and the flood pipeline.

No pretrained weights are used (they are CC BY-NC and not in the repository). The network is built with a fixed seed,
its state dict is saved the way the released checkpoint is (a Lightning module: keys prefixed ``network.``), and the
pipeline logic is tested with a stand-in classifier whose answer is known by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

torch = pytest.importorskip("torch")

from test_sar_flood import BBOX, RASTER, RES, H, W, lonlat_mask  # noqa: E402

from pygeovision.ai.models import ml4floods as m4f  # noqa: E402
from pygeovision.ai.pipelines import ModelWeightsRequired  # noqa: E402
from pygeovision.ai.pipelines import ml4floods_flood as pipe_mod  # noqa: E402
from pygeovision.ai.pipelines.ml4floods_flood import Ml4FloodsFloodPipeline  # noqa: E402

CONFIG = {
    "model_params": {
        "model_version": "v2",
        "hyperparameters": {"model_type": "unet", "num_classes": 2, "num_channels": 6},
    },
    "data_params": {"channel_configuration": "bgriswirs"},
}


def save_checkpoint(
    folder: Path, *, prefix: str = "network.", config: dict | None = CONFIG, channels: int = 6
):
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    net = m4f.build_network(in_channels=channels)
    torch.save({prefix + k: v for k, v in net.state_dict().items()}, folder / "model.pt")
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    return folder


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return save_checkpoint(tmp_path_factory.mktemp("wf2") / "WF2_unetv2_bgriswirs")


# --------------------------------------------------------------------------------------------------- the network
def test_network_has_the_size_of_the_released_checkpoint():
    """model.pt of WF2_unetv2_bgriswirs is 31.1 MB on Hugging Face; a 6-in, 2-out U-Net of these widths is that big."""
    net = m4f.build_network()
    n = sum(p.numel() for p in net.parameters())
    assert 7_700_000 < n < 7_900_000
    assert 31.0 < n * 4 / 1e6 < 31.3


def test_layer_names_match_the_ml4floods_checkpoint_layout():
    keys = set(m4f.build_network().state_dict())
    for k in (
        "dconv_down1.0.weight",
        "dconv_down1.2.bias",
        "dconv_down4.0.weight",
        "dconv_up3.0.weight",
        "dconv_up1.2.weight",
        "conv_last.weight",
    ):
        assert k in keys, k
    assert m4f.build_network().dconv_down1[0].weight.shape == (64, 6, 3, 3)


def test_forward_shape_and_two_outputs():
    out = m4f.build_network().eval()(torch.zeros(1, 6, 16, 24))
    assert out.shape == (1, 2, 16, 24)


# --------------------------------------------------------------------------------------------------- loading
def test_load_model_strips_the_lightning_prefix_and_records_the_checksum(ckpt):
    model = m4f.load_model(ckpt, "cpu")
    assert model.device == "cpu" and len(model.weights_sha256) == 64
    assert (
        model.weights_path.name == "model.pt"
        and model.config["data_params"]["channel_configuration"] == "bgriswirs"
    )
    # also accepts the model.pt file itself, and a plain (unprefixed) state dict
    assert m4f.load_model(ckpt / "model.pt", "cpu").weights_sha256 == model.weights_sha256
    plain = save_checkpoint(ckpt.parent / "plain", prefix="", config=None)
    assert m4f.load_model(plain, "cpu").config is None


def test_load_model_finds_the_huggingface_folder_layout(tmp_path):
    save_checkpoint(tmp_path / m4f.HF_SUBFOLDER)
    assert m4f.load_model(tmp_path, "cpu").weights_path.parent.name == m4f.MODEL_ID


def test_load_model_refuses_what_is_not_this_network(tmp_path, ckpt):
    with pytest.raises(m4f.ML4FloodsError, match="no model.pt"):
        m4f.load_model(tmp_path / "nothing")
    three = save_checkpoint(tmp_path / "rgb", channels=3, config=None)
    with pytest.raises(m4f.ML4FloodsError, match="not WF2_unetv2_bgriswirs"):
        m4f.load_model(three)
    for change, text in (
        ({"data_params": {"channel_configuration": "rgbi"}}, "channel_configuration"),
        (
            {"data_params": {"channel_configuration": "bgriswirs", "add_mndwi_input": True}},
            "add_mndwi_input",
        ),
        (
            {"model_params": {"model_version": "v1", "hyperparameters": {"model_type": "unet"}}},
            "model_version",
        ),
        (
            {
                "model_params": {
                    "model_version": "v2",
                    "hyperparameters": {"model_type": "hrnet_small"},
                }
            },
            "model_type",
        ),
    ):
        folder = save_checkpoint(tmp_path / text, config={**CONFIG, **change})
        with pytest.raises(m4f.ML4FloodsError, match=text):
            m4f.load_model(folder)


def test_load_model_is_strict(tmp_path):
    folder = save_checkpoint(tmp_path / "bad", config=None)
    state = torch.load(folder / "model.pt", weights_only=True)
    state["network.surplus.weight"] = torch.zeros(1)
    torch.save(state, folder / "model.pt")
    with pytest.raises(m4f.ML4FloodsError, match="does not fit"):
        m4f.load_model(folder)


# --------------------------------------------------------------------------------------------------- licence
def test_weights_are_not_downloaded_without_the_licence_being_accepted(monkeypatch, tmp_path):
    calls = []
    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    with pytest.raises(m4f.LicenseNotAccepted, match="CC BY-NC"):
        m4f.download_weights(tmp_path)
    assert calls == []


def test_download_with_acceptance_uses_the_pinned_revision(monkeypatch, tmp_path):
    calls = []

    def fake_download(**kw):
        calls.append(kw)
        target = Path(kw["local_dir"]) / kw["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    folder = m4f.download_weights(tmp_path, accept_noncommercial=True)
    assert folder == tmp_path / m4f.HF_SUBFOLDER
    assert [c["filename"] for c in calls] == [
        f"{m4f.HF_SUBFOLDER}/config.json",
        f"{m4f.HF_SUBFOLDER}/model.pt",
    ]
    assert {c["repo_id"] for c in calls} == {"isp-uv-es/ml4floods"}
    assert {c["revision"] for c in calls} == {m4f.HF_REVISION}


def test_the_weights_are_not_in_the_package():
    root = Path(m4f.__file__).resolve().parents[2]
    assert not list(root.rglob("model.pt")) and not list(root.rglob("WF2_unetv2*"))


# --------------------------------------------------------------------------------------------------- prediction
def image(h=24, w=40, value=1500.0):
    return np.full((6, h, w), value, dtype="float32")


def test_invalid_pixels_are_zero_or_not_finite():
    img = image()
    img[:, 0, 0] = 0.0
    img[2, 1, 1] = np.nan
    bad = m4f.invalid_mask(img)
    assert bad[0, 0] and bad[1, 1] and bad.sum() == 2


def test_select_bands_from_a_13_band_stack():
    stack = np.arange(13, dtype="float32")[:, None, None] * np.ones((13, 2, 2), dtype="float32")
    assert m4f.select_bands(stack)[:, 0, 0].tolist() == [1, 2, 3, 7, 11, 12]
    with pytest.raises(ValueError, match="6 bands"):
        m4f.select_bands(np.zeros((4, 2, 2)))


def test_probabilities_cover_an_awkward_size_and_leave_invalid_pixels_nan(ckpt):
    model = m4f.load_model(ckpt, "cpu")
    img = image(37, 45)  # not a multiple of 8
    img[:, :3, :3] = 0.0
    p = m4f.predict_probabilities(model, img, tile_size=16, context=16)
    assert p.shape == (2, 37, 45) and np.isnan(p[:, :3, :3]).all()
    ok = p[:, 5:, 5:]
    assert np.isfinite(ok).all() and (ok >= 0).all() and (ok <= 1).all()


def test_tiles_agree_with_one_pass_away_from_the_borders(ckpt):
    model = m4f.load_model(ckpt, "cpu")
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(2500, 800, (6, 96, 96)), 50, 9000).astype("float32")
    whole = m4f.predict_probabilities(model, img, tile_size=512)
    tiled = m4f.predict_probabilities(model, img, tile_size=48, context=64)
    assert np.abs(whole - tiled).max() < 1e-3


def test_prediction_is_deterministic_and_does_not_change_the_input(ckpt):
    model = m4f.load_model(ckpt, "cpu")
    img = image()
    before = img.copy()
    a = m4f.predict_probabilities(model, img)
    b = m4f.predict_probabilities(model, img)
    assert np.array_equal(a, b) and np.array_equal(img, before)


def test_bad_arguments():
    with pytest.raises(ValueError, match="must be"):
        m4f.predict_probabilities(types.SimpleNamespace(), np.zeros((3, 8, 8)))
    with pytest.raises(ValueError, match="tile_size"):
        m4f.predict_probabilities(types.SimpleNamespace(), image(), tile_size=4)
    with pytest.raises(ValueError, match="th_water"):
        m4f.classify(image(), np.zeros((2, 24, 40)), th_water=1.5)


# --------------------------------------------------------------------------------------------------- classes
def test_classify_rules():
    h, w = 4, 6
    img = np.full(
        (6, h, w), 1500.0, dtype="float32"
    )  # dark: brightness sqrt(3 * 1500^2) = 2598 < 3500
    img[:3, 0, 1] = 3000.0  # bright (5196)
    img[:, 3, 5] = 0.0  # invalid
    p = np.zeros((2, h, w), dtype="float32")
    p[1, 0, 0] = 0.9  # water
    p[1, 0, 2] = 0.4  # below the threshold
    p[0, 0, 1] = 0.9  # cloud and bright
    p[0, 1, 0] = 0.9  # cloud but dark: thin cloud / haze, not masked
    p[1, 1, 0] = 0.9  # ... and water under it
    p[0, 1, 1] = p[1, 1, 1] = 0.9  # cloud and water, dark
    p[1, 3, 5] = 0.9  # on an invalid pixel
    c = m4f.classify(img, p, th_water=0.5)
    assert c[0, 0] == 2 and c[0, 2] == 1 and c[0, 1] == 3
    assert c[1, 0] == 2 and c[1, 1] == 2  # dark cloud does not hide the water
    assert c[3, 5] == 0
    c2 = m4f.classify(img, p, mask_clouds=False)
    assert (
        c2[1, 0] == 3
    )  # without the brightness rule every cloud prediction counts, and clouds win over water


def test_flood_trace_is_water_with_negative_mndwi():
    img = np.full((6, 1, 2), 1500.0, dtype="float32")
    img[1, 0, 0], img[4, 0, 0] = 2000.0, 500.0  # MNDWI > 0
    img[1, 0, 1], img[4, 0, 1] = 500.0, 2000.0  # MNDWI < 0
    p = np.zeros((2, 1, 2), dtype="float32")
    p[1] = 0.9
    c = m4f.classify(img, p, distinguish_flood_traces=True)
    assert c[0].tolist() == [2, 4]


def test_geotiff_round_trip(ckpt, tmp_path):
    img = np.clip(np.random.default_rng(2).normal(2000, 600, (13, 20, 24)), 1, 9000).astype(
        "float32"
    )
    path = tmp_path / "s2.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=24,
        count=13,
        dtype="float32",
        crs="EPSG:32630",
        transform=from_origin(500000, 650000, 10, 10),
    ) as d:
        d.write(img)
    out = m4f.predict_geotiff(path, tmp_path / "o", ckpt, device="cpu")
    with rasterio.open(out["classes"]) as c, rasterio.open(out["probabilities"]) as p:
        assert (
            c.count == 1
            and p.count == 2
            and c.crs.to_string() == "EPSG:32630"
            and c.read(1).shape == (20, 24)
        )
        assert set(np.unique(c.read(1))) <= {0, 1, 2, 3}
        assert c.tags()["model"] == m4f.MODEL_ID


# --------------------------------------------------------------------------------------------------- pipeline
LAKE = (0.4510, 5.9150, 0.4540, 5.9200)  # water in both windows
FLOOD = (0.4560, 5.9050, 0.4620, 5.9110)  # water appears in the flood window
NOREF = (0.4650, 5.9100, 0.4700, 5.9150)  # flooded, but the reference view is masked (cloud shadow)
CLOUD = (0.4650, 5.9200, 0.4700, 5.9220)  # bright cloud the network must call cloud
SHADOW = (0.4560, 5.9150, 0.4600, 5.9200)  # dark "water" that the SCL masked as cloud shadow (NaN)
SHRINK = 0.0004


def prepared_stack(tmp_path, tag, water, masked=(), cloudy=()):
    """What ``_search_and_download`` hands over: 6 bands, reflectance 0-1, SCL cloud/shadow = NaN.

    Water: green 0.12 / SWIR1 0.03; land: 0.15 / 0.25. ``cloudy`` = very bright (the network must call it cloud).
    """
    shape = (H, W)

    def mask(boxes):
        m = np.zeros(shape, bool)
        for b in boxes:
            m |= lonlat_mask(b)
        return m

    w, cl, nan = mask(water), mask(cloudy), mask(masked)
    bands = np.stack(
        [
            np.where(w, 0.09, 0.12),  # blue
            np.where(w, 0.12, 0.15),  # green
            np.where(w, 0.09, 0.14),  # red
            np.where(w, 0.04, 0.30),  # nir
            np.where(w, 0.03, 0.25),  # swir1
            np.where(w, 0.02, 0.18),  # swir2
        ]
    ).astype("float32")
    bands[:, cl] = 0.60
    bands[:, nan] = np.nan
    path = tmp_path / f"{tag}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=6,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(RASTER[0], RASTER[3], RES, RES),
        nodata=np.nan,
    ) as d:
        d.write(bands)
    return path


def stand_in_classifier(model, image, **kw):
    """What a good network would say about these scenes: water where the SWIR is dark, cloud where it is very bright."""
    swir = image[4]
    classes = np.ones(swir.shape, dtype="uint8")
    classes[swir < 600] = 2
    classes[image[1] > 5000] = 3
    classes[~np.isfinite(swir)] = 0
    probs = np.stack([(classes == 3).astype("float32"), (classes == 2).astype("float32")])
    return classes, probs


class Imagery:
    """Stands in for the standard imagery path: records how it was called, returns prepared images."""

    def __init__(self, before, after):
        self.before, self.after, self.calls = before, after, []

    def __call__(self, bbox, date_before, date_after, output_dir, **kw):
        self.calls.append({"bbox": bbox, "dates": (date_before, date_after), **kw})
        return self.before, self.after


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe_mod.m4f, "predict_scene", stand_in_classifier)
    ref = prepared_stack(tmp_path, "ref", [LAKE], masked=[NOREF])
    flood = prepared_stack(
        tmp_path, "flood", [LAKE, FLOOD, NOREF, SHADOW], masked=[SHADOW], cloudy=[CLOUD]
    )
    return tmp_path, Imagery(ref, flood)


def run(tmp_path, imagery, ckpt, monkeypatch, **kw):
    p = Ml4FloodsFloodPipeline(None)
    monkeypatch.setattr(p, "_search_and_download_pair", imagery)
    args = {
        "date_before": "2026-06-01/2026-06-20",
        "date_after": "2026-06-27/2026-07-06",
        "checkpoint": ckpt,
        "device": "cpu",
        "output_dir": tmp_path / "out",
    }
    args.update(kw)
    return p, p.run(BBOX, **args)


def block(cls, src, box):
    t = src.transform
    rows, cols = np.mgrid[0 : cls.shape[0], 0 : cls.shape[1]]
    from pyproj import Transformer

    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(
        t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e
    )
    m = (
        (lon > box[0] + SHRINK)
        & (lon < box[2] - SHRINK)
        & (lat > box[1] + SHRINK)
        & (lat < box[3] - SHRINK)
    )
    assert m.sum() > 20, "the block is too small to test"
    return set(np.unique(cls[m]).tolist())


def test_catalogue_entry():
    from pygeovision.ai.pipelines import get_pipeline, list_pipelines, pipeline_kinds

    assert "ml4floods_flood_mapping" in list_pipelines()
    assert pipeline_kinds()["ml4floods_flood_mapping"] == "ready"
    assert get_pipeline("ml4floods_flood_mapping", None).domain == "disaster"


def test_it_uses_the_standard_imagery_path_with_the_six_network_bands(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, img, ckpt, monkeypatch)
    assert res.success, res.error
    (call,) = img.calls
    assert call["bbox"] == tuple(BBOX)
    assert call["dates"] == ("2026-06-01/2026-06-20", "2026-06-27/2026-07-06")
    assert call["bands"] == ("blue", "green", "red", "nir", "swir1", "swir2")
    assert call["apply_scale"] is True and call["apply_cloud_mask"] is True


def test_the_scene_classification_layer_can_be_switched_off(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, img, ckpt, monkeypatch, use_scl=False)
    assert res.success and img.calls[0]["apply_cloud_mask"] is False


def test_without_checkpoint_the_weights_are_fetched_before_any_imagery(tmp_path, monkeypatch):
    """No checkpoint: the weights come from Hugging Face (here failing), and the run stops before fetching imagery."""
    img = Imagery(None, None)

    def no_network(dest=None):
        raise OSError("offline")

    monkeypatch.setattr(m4f, "ensure_weights", no_network)
    _, r = run(tmp_path, img, None, monkeypatch)
    assert not r.success and "Hugging Face" in r.error and "checkpoint=" in r.error
    assert img.calls == []


def test_ensure_weights_downloads_once_and_then_reuses_the_cache(monkeypatch, tmp_path):
    calls = []

    def fake_download(**kw):
        calls.append(kw["filename"])
        target = Path(kw["local_dir"]) / kw["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 2_000_000 if kw["filename"].endswith(".pt") else b"{}")

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    first = m4f.ensure_weights(tmp_path)
    second = m4f.ensure_weights(tmp_path)
    assert first == second == tmp_path / m4f.HF_SUBFOLDER
    assert len(calls) == 2  # config.json and model.pt, once


def test_checkpoint_auto_means_the_download(world, ckpt, monkeypatch):
    tmp_path, img = world
    asked = []
    monkeypatch.setattr(m4f, "ensure_weights", lambda dest=None: asked.append(1) or ckpt)
    _, res = run(tmp_path, img, "auto", monkeypatch)
    assert res.success, res.error
    assert asked == [1]


def test_wrong_checkpoint_stops_before_any_imagery(tmp_path, ckpt, monkeypatch):
    bad = save_checkpoint(tmp_path / "bad", config=None, channels=3)
    img = Imagery(None, None)
    _, res = run(tmp_path, img, bad, monkeypatch)
    assert not res.success and "not WF2_unetv2_bgriswirs" in res.error and img.calls == []


@pytest.mark.parametrize(
    "kw, text",
    [
        ({"date_before": None}, "needs date_before"),
        ({"date_after": "2026-06-10"}, "must end before"),
        ({"th_water": 2.0}, "between 0 and 1"),
        ({"min_patch_ha": -1}, "cannot be negative"),
    ],
)
def test_bad_arguments_are_reported(tmp_path, ckpt, monkeypatch, kw, text):
    img = Imagery(None, None)
    _, res = run(tmp_path, img, ckpt, monkeypatch, **kw)
    assert not res.success and text in res.error and img.calls == []


def test_no_flood_window_image(tmp_path, ckpt, monkeypatch):
    _, res = run(tmp_path, Imagery(None, None), ckpt, monkeypatch)
    assert not res.success and "flood window" in res.error


def test_flood_classes_are_right_by_construction(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, img, ckpt, monkeypatch)
    assert res.success, res.error
    with rasterio.open(res.output_path) as src:
        cls = src.read(1)
        assert block(cls, src, FLOOD) == {1}  # new water, clear reference
        assert block(cls, src, LAKE) == {2}  # water in both
        assert block(cls, src, NOREF) == {3}  # water, reference masked
        assert block(cls, src, CLOUD) == {255}  # cloud: unknown, never dry
        assert block(cls, src, SHADOW) == {255}  # cloud shadow masked by the SCL: unknown
    s = res.stats
    assert (
        s["flooded_area_ha"] > 0
        and s["flooded_unresolved_ha"] > 0
        and s["permanent_water_area_ha"] > 0
    )
    assert s["flooded_upper_ha"] == pytest.approx(s["flooded_area_ha"] + s["flooded_unresolved_ha"])
    assert 0 < s["valid_fraction"] < 1
    assert s["model"] == m4f.MODEL_ID and len(s["model_weights_sha256"]) == 64
    assert "Level-1C" in s["input_product"] and "CC BY-NC" in s["model_license"]
    assert "Portales-Julia" in s["product_note"]
    assert (res.output_path.parent / "prob_water_after.tif").is_file()


def test_area_uses_the_real_pixel_size_of_a_geographic_image(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, img, ckpt, monkeypatch)
    with rasterio.open(res.output_path) as src:
        n = int((src.read(1) == 1).sum())
        lat = src.transform.f + src.transform.e * src.height / 2
        px = (
            abs(src.transform.a)
            * 111_320.0
            * np.cos(np.radians(lat))
            * abs(src.transform.e)
            * 110_574.0
        )
    assert res.stats["flooded_area_ha"] == pytest.approx(n * px / 10_000, rel=1e-6)


def test_missing_reference_image_leaves_everything_unresolved(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, Imagery(None, img.after), ckpt, monkeypatch)
    assert (
        res.success and res.stats["flooded_area_ha"] == 0 and res.stats["flooded_unresolved_ha"] > 0
    )
    assert "reference window" in res.stats["data_notes"]


def test_small_patches_are_removed(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, img, ckpt, monkeypatch, min_patch_ha=10_000.0)
    assert res.stats["flooded_area_ha"] == 0 and res.stats["removed_small_patch_ha"] > 0


def test_grid_larger_than_the_limit_is_refused_before_any_imagery(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, img, ckpt, monkeypatch, max_pixels=1000)
    assert not res.success and "million pixels" in res.error and img.calls == []


def test_progress_is_logged(world, ckpt, monkeypatch, caplog):
    tmp_path, img = world
    with caplog.at_level("INFO", logger=pipe_mod.logger.name):
        _, res = run(tmp_path, img, ckpt, monkeypatch)
    assert res.success, res.error
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "flood window: image" in text and "reference window: image" in text and "area " in text


def test_report_spec_exists_and_is_honest():
    from pygeovision.reporting.specs import spec_for

    spec = spec_for("ml4floods_flood_mapping")
    text = " ".join(spec.limitations + spec.method)
    assert "Level-1C" in text and "CC BY-NC" in text and "accuracy is not assessed" in text
    assert any("Scientific Reports" in r for r in spec.references)


# ------------------------------------------------------------------------- helpful errors for a missing model.pt
def test_missing_weights_error_says_where_it_looked_and_what_is_there(tmp_path):
    folder = tmp_path / "models" / "WF2_unetv2_bgriswirs"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    with pytest.raises(m4f.ML4FloodsError) as exc:
        m4f.load_model(folder, "cpu")
    msg = str(exc.value)
    assert str(folder.resolve()) in msg and "config.json" in msg and "download_weights" in msg


def test_missing_relative_path_error_names_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(m4f.ML4FloodsError, match="does not exist") as exc:
        m4f.load_model("ml4floods/models/WF2_unetv2_bgriswirs", "cpu")
    assert str(tmp_path.resolve()) in str(exc.value)


def test_git_lfs_pointer_is_recognised(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    (folder / "model.pt").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 31100000\n"
    )
    with pytest.raises(m4f.ML4FloodsError, match="Git LFS pointer"):
        m4f.load_model(folder, "cpu")


def test_a_single_differently_named_checkpoint_file_is_used(tmp_path):
    folder = save_checkpoint(tmp_path / "w")
    (folder / "model.pt").rename(folder / "weights.pth")
    assert m4f._resolve(folder)[0].name == "weights.pth"
    assert m4f.load_model(folder, "cpu").weights_sha256


# ---------------------------------------------------------------- scenes named in the result, errors of the weights fetch
def test_the_result_names_the_scenes_and_dates_the_images_were_made_from(world, ckpt, monkeypatch):
    tmp_path, img = world
    p = Ml4FloodsFloodPipeline(None)

    def imagery(bbox, date_before, date_after, output_dir, **kw):
        for label, ids in (
            ("before", ["S2B_MSIL2A_20260610T101559_N0510_R065_T32TQM"]),
            ("after", ["S2A_32TQM_20260630_0_L2A", "S2A_32TQM_20260703_0_L2A"]),
        ):
            p._coverage_log.append(
                {"image": label, "aoi_coverage_fraction": 1.0, "scenes_used": ids, "scenes_failed": []}
            )
        return img.before, img.after

    monkeypatch.setattr(p, "_search_and_download_pair", imagery)
    res = p.run(
        BBOX,
        date_before="2026-06-01/2026-06-20",
        date_after="2026-06-27/2026-07-06",
        checkpoint=ckpt,
        device="cpu",
        output_dir=tmp_path / "out",
    )
    assert res.success, res.error
    assert res.metadata["scene_ids"]["reference"] == ["S2B_MSIL2A_20260610T101559_N0510_R065_T32TQM"]
    assert res.metadata["scene_ids"]["flood"] == ["S2A_32TQM_20260630_0_L2A", "S2A_32TQM_20260703_0_L2A"]
    assert res.stats["reference_dates"] == ["2026-06-10"]
    assert res.stats["flood_dates"] == ["2026-06-30", "2026-07-03"]
    assert res.metadata["date_after"] == "2026-06-30 – 2026-07-03"


def test_without_scene_ids_the_image_file_name_stands_in(world, ckpt, monkeypatch):
    tmp_path, img = world
    _, res = run(tmp_path, img, ckpt, monkeypatch)
    assert res.success, res.error
    assert res.metadata["scene_ids"]["flood"] == [Path(img.after).stem]
    assert res.metadata["date_after"] == "2026-06-27 – 2026-07-06"  # falls back to the requested window


@pytest.mark.parametrize(
    "sid,day",
    [
        ("S2B_MSIL2A_20230613T101559_N0509_R022_T31UFS", "2023-06-13"),
        ("S2A_32TQM_20230613_0_L2A", "2023-06-13"),
        ("no-date-here", ""),
    ],
)
def test_scene_dates_are_read_from_sentinel2_ids(sid, day):
    from pygeovision.ai.pipelines.ml4floods_flood import _scene_date

    assert _scene_date(sid) == day


def test_any_failure_fetching_the_weights_is_reported_not_raised(tmp_path, monkeypatch):
    """Not only OSError: a hub or torch error (here a RuntimeError) must also end as a clear failed result."""
    img = Imagery(None, None)

    def broken(dest=None):
        raise RuntimeError("hub client exploded")

    monkeypatch.setattr(m4f, "ensure_weights", broken)
    _, r = run(tmp_path, img, None, monkeypatch)
    assert not r.success and "hub client exploded" in r.error and "checkpoint=" in r.error
    assert img.calls == []


def test_the_weights_helpers_are_public():
    assert {"ensure_weights", "download_weights", "DEFAULT_CACHE"} <= set(m4f.__all__)


def test_a_tilde_destination_is_expanded(monkeypatch, tmp_path):
    """``~`` in a destination folder must mean the home directory, not a folder literally called ``~``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = {}

    def fake_hub(repo_id, filename, revision, local_dir):
        seen["dir"] = local_dir
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")

    mod = types.ModuleType("huggingface_hub")
    mod.hf_hub_download = fake_hub
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    m4f.download_weights("~/w", accept_noncommercial=True)
    assert seen["dir"] == str(tmp_path / "w")
