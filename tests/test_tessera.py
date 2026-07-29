"""Tests for pygeovision.advanced.foundation.tessera_geo.TesseraGeo.

Real gap found and fixed: TESSERA (satellite embeddings foundation model)
was a confirmed honest stub — models/foundation/tessera.py correctly
delegated to registry.get_model(), but the registry never actually had a
'tessera' spec at all, so get_model('tessera') failed with "model not
found" before reaching any dispatch logic whatsoever.

TESSERA's own design philosophy is precomputed embeddings, not running an
encoder locally — this wraps the real `geotessera` library (confirmed via
direct introspection of its actual API: fetch_mosaic_for_region returns
(H, W, 128), not (128, H, W) as one might assume), not a fake stand-in.
"""
import pytest

import numpy as np

geotessera = pytest.importorskip("geotessera", reason="geotessera not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")


def _mock_tessera(monkeypatch=None):
    from pygeovision.advanced.foundation.tessera_geo import TesseraGeo
    from unittest.mock import MagicMock

    tessera = TesseraGeo.__new__(TesseraGeo)
    tessera.dataset_version = "v1"
    tessera.cache_dir = None
    tessera._client = MagicMock()
    return tessera


class TestRegistrySpecExists:
    """Regression test for the real gap: the registry never had a
    'tessera' spec at all, so get_model('tessera') failed with 'model not
    found' before any dispatch logic ever ran."""

    def test_tessera_spec_exists_in_registry(self):
        from pygeovision.models.registry import _REGISTRY
        assert "tessera" in _REGISTRY

    def test_get_model_returns_real_tesserageo(self):
        from pygeovision.models.registry import get_model
        from pygeovision.advanced.foundation.tessera_geo import TesseraGeo
        model = get_model("tessera")
        assert isinstance(model, TesseraGeo)


class TestTesseraGeoEmbeddingsForBbox:
    """Real API shape verification: geotessera's fetch_mosaic_for_region
    returns (height, width, 128) — channel-LAST, confirmed by reading its
    actual source. This wrapper must correctly transpose to pygeovision's
    channel-first convention, not assume the wrong axis order."""

    def test_correct_shape_and_channel_ordering(self, tmp_path):
        from rasterio.transform import from_bounds
        tessera = _mock_tessera()

        fake_mosaic = np.random.randn(20, 24, 128).astype(np.float32)  # (H, W, C)
        fake_transform = from_bounds(-0.25, 5.52, -0.20, 5.60, 24, 20)
        tessera._client.fetch_mosaic_for_region.return_value = (fake_mosaic, fake_transform, "EPSG:4326")

        out_path = tmp_path / "tessera.tif"
        result = tessera.embeddings_for_bbox(
            (-0.25, 5.52, -0.20, 5.60), year=2024, output_path=str(out_path),
        )
        assert result["success"] is True
        assert result["shape"] == (128, 20, 24)  # (C, H, W) — correctly transposed
        assert result["n_channels"] == 128

        with rasterio.open(out_path) as src:
            assert src.count == 128
            assert src.height == 20
            assert src.width == 24

    def test_geotiff_tags_record_provenance(self, tmp_path):
        from rasterio.transform import from_bounds
        tessera = _mock_tessera()
        fake_mosaic = np.random.randn(10, 10, 128).astype(np.float32)
        fake_transform = from_bounds(0, 0, 100, 100, 10, 10)
        tessera._client.fetch_mosaic_for_region.return_value = (fake_mosaic, fake_transform, "EPSG:4326")

        out_path = tmp_path / "tessera.tif"
        tessera.embeddings_for_bbox((0, 0, 1, 1), year=2023, output_path=str(out_path))

        with rasterio.open(out_path) as src:
            tags = src.tags()
            assert tags.get("source") == "TESSERA"
            assert tags.get("year") == "2023"

    def test_handles_fetch_failure_gracefully(self):
        tessera = _mock_tessera()
        tessera._client.fetch_mosaic_for_region.side_effect = ValueError("no tiles found")
        result = tessera.embeddings_for_bbox((0, 0, 1, 1), year=2024)
        assert result["success"] is False
        assert "no tiles found" in result["error"]


class TestTesseraGeoEmbeddingsAtPoints:
    def test_returns_real_point_embeddings(self):
        tessera = _mock_tessera()
        tessera._client.sample_embeddings_at_points.return_value = np.random.randn(3, 128).astype(np.float32)

        points = [(-0.22, 5.55), (-0.21, 5.56), (-0.20, 5.57)]
        emb = tessera.embeddings_at_points(points, year=2024)
        assert emb.shape == (3, 128)
        tessera._client.sample_embeddings_at_points.assert_called_once_with(points, year=2024)


class TestTesseraGeoPCAVisualisation:
    """Verifies the real, direct-sklearn-PCA approach (not a guess at
    geotessera's internal per-tile tuple format for its own PCA helpers,
    which this wrapper deliberately avoids relying on)."""

    def test_produces_real_uint8_rgb_geotiff(self, tmp_path):
        pytest.importorskip("sklearn", reason="scikit-learn not installed")
        from rasterio.transform import from_bounds
        tessera = _mock_tessera()

        fake_mosaic = np.random.randn(15, 15, 128).astype(np.float32)
        fake_transform = from_bounds(0, 0, 150, 150, 15, 15)
        tessera._client.fetch_mosaic_for_region.return_value = (fake_mosaic, fake_transform, "EPSG:4326")

        out_path = tmp_path / "pca.tif"
        result = tessera.pca_visualisation((0, 0, 1, 1), str(out_path), year=2024, n_components=3)

        assert result["success"] is True
        assert len(result["explained_variance_ratio"]) == 3

        with rasterio.open(out_path) as src:
            assert src.count == 3
            assert src.dtypes[0] == "uint8"
            data = src.read()
            assert data.min() >= 0 and data.max() <= 255
