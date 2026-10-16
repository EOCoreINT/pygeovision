# ruff: noqa: E702, N802, C408, E731
"""Behaviour printed on docs/foundation-models/radar.md (all offline)."""
import logging

import numpy as np
import pytest

from pygeovision.models.adapters.sar_channel_manager import (
    SARSpeckleAugmentation,
    sar_to_hls_6ch,
    validate_sar_ai_input,
)
from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter


def _scene(h=96):
    rng = np.random.default_rng(0)
    vv = np.clip(rng.normal(0.6, 0.1, (h, h)), 0, 1).astype("float32")
    vh = np.clip(rng.normal(0.5, 0.1, (h, h)), 0, 1).astype("float32")
    vh[:, : h // 3] = np.clip(rng.normal(0.15, 0.05, (h, h // 3)), 0, 1)       # a dark "water" strip
    vv[:, : h // 3] = np.clip(rng.normal(0.30, 0.05, (h, h // 3)), 0, 1)
    return vv, vh


def test_physics_guided_channel_layout():
    vv, vh = _scene()
    s = sar_to_hls_6ch(vv, vh)
    assert s.shape == (6, 96, 96) and s.dtype == np.float32
    np.testing.assert_allclose(s[0], vh); np.testing.assert_allclose(s[2], vv)
    np.testing.assert_allclose(s[1], (vh + vv) / 2, atol=1e-6)
    np.testing.assert_allclose(s[3], 1 - vh, atol=1e-6); np.testing.assert_allclose(s[4], 1 - vv, atol=1e-6)
    assert s[5].min() >= 0 and s[5].max() <= 1


def test_other_mappings_and_errors():
    vv, vh = _scene()
    assert np.allclose(sar_to_hls_6ch(vv, vh, mapping="mean_repeat")[0], sar_to_hls_6ch(vv, vh, mapping="mean_repeat")[5])
    r = sar_to_hls_6ch(vv, vh, mapping="replicate"); np.testing.assert_allclose(r[0], vv); np.testing.assert_allclose(r[1], vh)
    with pytest.raises(ValueError, match="Unknown mapping"):
        sar_to_hls_6ch(vv, vh, mapping="x")
    with pytest.raises(ValueError, match="!= VH shape"):
        sar_to_hls_6ch(vv, vh[:50])
    z = sar_to_hls_6ch(vv, vh, normalise=True)
    assert np.allclose(z.mean(axis=(1, 2)), 0, atol=1e-4) and np.allclose(z.std(axis=(1, 2)), 1, atol=1e-3)


def test_missing_vh_is_guessed_from_vv_with_a_warning(caplog):
    vv, _ = _scene()
    with caplog.at_level(logging.WARNING):
        s = sar_to_hls_6ch(vv)
    assert "approximating from VV" in caplog.text and np.allclose(s[0], np.clip(vv * 0.6, 0, 1), atol=1e-6)


def test_validation_messages():
    vv, vh = _scene()
    ok = validate_sar_ai_input(sar_to_hls_6ch(vv, vh))
    assert ok["valid"] and not ok["warnings"] and not ok["errors"]
    assert "outside expected" in validate_sar_ai_input(sar_to_hls_6ch(vv * 30 - 35, vh * 30 - 35))["warnings"][0]
    small = validate_sar_ai_input(sar_to_hls_6ch(vv[:40, :40], vh[:40, :40]))
    assert not small["valid"] and "below minimum 64" in small["errors"][0]
    assert "square" in validate_sar_ai_input(sar_to_hls_6ch(vv[:, :70], vh[:, :70]))["warnings"][0]
    nan = sar_to_hls_6ch(vv, vh); nan[2, 0, 0] = np.nan
    assert not validate_sar_ai_input(nan)["valid"]


def test_zero_shot_is_a_vh_threshold_not_prithvi():
    vv, vh = _scene()
    r = SARPrithviAdapter().run(vv, vh)
    assert r["method"] == "vh_threshold" and r["prediction"].dtype == np.uint8 and r["confidence"] is None
    p = r["prediction"]
    assert p[:, :32].mean() == 1.0 and p[:, 32:].mean() < 0.15
    np.testing.assert_array_equal(p, (vh < 0.35).astype(np.uint8))


def test_too_small_chip_returns_no_prediction_and_says_why():
    vv, vh = _scene()
    r = SARPrithviAdapter().run(vv[:40, :40], vh[:40, :40])
    assert r["prediction"] is None and "below minimum 64" in r["warnings"][0]


def test_adapter_argument_errors():
    with pytest.raises(ValueError, match="Unknown mode"):
        SARPrithviAdapter(mode="x")
    with pytest.raises(ValueError, match="Unknown task"):
        SARPrithviAdapter(task="x")


@pytest.mark.parametrize("n", [1, 4, 8])
def test_speckle_augmentation_has_unit_mean_and_looks_dependent_spread(n):
    np.random.seed(0)
    out = SARSpeckleAugmentation((n, n))(np.ones((2, 200, 200), dtype="float32"))
    assert abs(out.mean() - 1) < 0.02 and abs(out.std() - 1 / np.sqrt(n)) < 0.02


def test_normalise_sar_for_ai_equals_the_fixed_bounds_formula(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from pygeovision.data.processors.sar import normalise_sar_for_ai

    db = np.random.default_rng(1).uniform(-40, 10, (1, 40, 40)).astype("float32")
    with rasterio.open(tmp_path / "db.tif", "w", driver="GTiff", height=40, width=40, count=1, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(0, 400, 10, 10)) as dst:
        dst.write(db)
    out = normalise_sar_for_ai(str(tmp_path / "db.tif"), str(tmp_path / "n.tif"))
    np.testing.assert_allclose(out, np.clip((db + 35) / 40, 0, 1), atol=1e-6)


def test_channel_manager_adapter_zero_shot_is_labelled_and_finetune_modes_are_not_faked():
    from pygeovision.models.adapters.sar_channel_manager import SARPrithviAdapter as Legacy

    vv, vh = _scene()
    r = Legacy(mode="zero_shot").run(vv, vh)
    assert r["success"] and r["method"] == "vh_threshold" and r["prediction"].dtype == np.uint8
    np.testing.assert_allclose(r["probability"], 1 - np.clip(vh, 0, 1), atol=1e-6)
    for mode in ("fine_tune", "spt"):
        with pytest.raises(NotImplementedError, match="sar_prithvi.SARPrithviAdapter"):
            Legacy(mode=mode).run(vv, vh)
    bad = Legacy().run(vv[:40, :40], vh[:40, :40])
    assert bad["success"] is False and bad["prediction"] is None
