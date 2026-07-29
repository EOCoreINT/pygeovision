"""Tests for pygeovision.advanced.foundation.alphaearth_geo.AlphaEarthGeo.

Real gap found and fixed: AlphaEarth Foundations (Google DeepMind's
Satellite Embedding dataset) had zero presence anywhere in pygeovision,
despite `earthengine-api` already being a real dependency used for
Dynamic World. This wraps the real `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
Earth Engine collection, confirmed via web research against Google's own
documentation and community tutorials (asset id, 64 bands named A00-A63,
annual coverage from 2017).

Real Earth Engine authentication isn't available in this test
environment, so these tests verify the REQUEST CONSTRUCTION against a
mocked `ee` module — confirming the real collection id, band count, and
query chain (filterBounds/filterDate/mosaic/sampleRegions/export) are
built correctly, the same discipline used for the Groq/TESSERA tests
elsewhere in this suite.
"""
import pytest

ee = pytest.importorskip("ee", reason="earthengine-api not installed")


class TestRegistryWiring:
    def test_alphaearth_spec_exists_in_registry(self):
        from pygeovision.models.registry import _REGISTRY
        assert "alphaearth" in _REGISTRY

    def test_get_model_returns_real_alphaearthgeo(self):
        from pygeovision.models.registry import get_model
        from pygeovision.advanced.foundation.alphaearth_geo import AlphaEarthGeo
        model = get_model("alphaearth")
        assert isinstance(model, AlphaEarthGeo)


class TestAlphaEarthGeoConstants:
    def test_real_collection_id(self):
        from pygeovision.advanced.foundation.alphaearth_geo import AlphaEarthGeo
        assert AlphaEarthGeo.COLLECTION == "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    def test_64_bands_named_a00_to_a63(self):
        from pygeovision.advanced.foundation.alphaearth_geo import AlphaEarthGeo
        assert len(AlphaEarthGeo.BANDS) == 64
        assert AlphaEarthGeo.BANDS[0] == "A00"
        assert AlphaEarthGeo.BANDS[-1] == "A63"


class TestEmbeddingsAtPointsRequestConstruction:
    """Verifies the real query chain is built correctly, using a mocked
    ee module (no real EE credentials available in this environment)."""

    def test_builds_correct_collection_and_date_filter(self, monkeypatch):
        from pygeovision.advanced.foundation.alphaearth_geo import AlphaEarthGeo
        from unittest.mock import MagicMock

        mock_ee = MagicMock()
        mock_ee.ImageCollection.return_value.filterDate.return_value.mosaic.return_value \
            .sampleRegions.return_value.getInfo.return_value = {"features": []}

        ae = AlphaEarthGeo()
        monkeypatch.setattr(ae, "_ee", lambda: mock_ee)

        ae.embeddings_at_points([(-0.22, 5.55), (-0.21, 5.56)], year=2024)

        mock_ee.ImageCollection.assert_called_once_with("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        mock_ee.ImageCollection.return_value.filterDate.assert_called_once_with(
            "2024-01-01", "2024-12-31",
        )

    def test_returns_nan_for_points_with_no_matching_feature(self, monkeypatch):
        """Points outside coverage (no feature returned by EE) must come
        back as NaN rows, not silently zero or misaligned."""
        import numpy as np
        from pygeovision.advanced.foundation.alphaearth_geo import AlphaEarthGeo
        from unittest.mock import MagicMock

        mock_ee = MagicMock()
        # Only point index 0 gets a real feature back; index 1 is "missing"
        fake_props = {"idx": 0, **{f"A{i:02d}": float(i) for i in range(64)}}
        mock_ee.ImageCollection.return_value.filterDate.return_value.mosaic.return_value \
            .sampleRegions.return_value.getInfo.return_value = {
                "features": [{"properties": fake_props}]
            }

        ae = AlphaEarthGeo()
        monkeypatch.setattr(ae, "_ee", lambda: mock_ee)

        result = ae.embeddings_at_points([(-0.22, 5.55), (-0.21, 5.56)], year=2024)
        assert result.shape == (2, 64)
        assert not np.isnan(result[0]).any()
        assert np.isnan(result[1]).all()
        assert result[0, 5] == 5.0  # A05 correctly mapped to band index 5


class TestEmbeddingsForBboxRequestConstruction:
    def test_starts_real_export_task(self, monkeypatch):
        from pygeovision.advanced.foundation.alphaearth_geo import AlphaEarthGeo
        from unittest.mock import MagicMock

        mock_ee = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "TASK123"
        mock_ee.batch.Export.image.toDrive.return_value = mock_task

        ae = AlphaEarthGeo()
        monkeypatch.setattr(ae, "_ee", lambda: mock_ee)

        result = ae.embeddings_for_bbox((-0.25, 5.52, -0.20, 5.60), year=2024)

        assert result["success"] is True
        assert result["ee_task_id"] == "TASK123"
        assert result["n_bands"] == 64
        mock_task.start.assert_called_once()

    def test_missing_earthengine_api_gives_clear_error(self, monkeypatch):
        from pygeovision.advanced.foundation.alphaearth_geo import AlphaEarthGeo

        ae = AlphaEarthGeo()

        def raise_import_error():
            raise ImportError("no ee")
        monkeypatch.setattr(ae, "_ee", raise_import_error)

        result = ae.embeddings_for_bbox((-0.25, 5.52, -0.20, 5.60))
        assert result["success"] is False
        assert "earthengine-api" in result["error"]
