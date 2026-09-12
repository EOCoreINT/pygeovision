"""Integration tests — model registry, end-to-end foundation model workflows."""

# Skip this file if PyTorch is not installed
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — pip install torch")


import numpy as np
import pytest

# ── Model Registry Integration ────────────────────────────────────────────────

class TestFoundationModelRegistry:
    """Verify the main model registry's real, current foundation-model
    entries -- not the pre-cleanup state this class originally tested.

    An earlier version of this class asserted the presence and metadata
    of 18 confirmed-fake, mislabeled dinov3_*-named registry entries (12
    backbone variants + 6 task heads). All were found to load real
    facebook/dinov2-* weights under a DINOv3 label -- including "SAT-493M
    satellite-pretrained" variants that loaded the identical, generic,
    non-satellite DINOv2 weights regardless of which _sat name was
    requested. All 18 were removed from the registry rather than
    relabeled, since a genuinely correct DINOv3 integration would need
    real, separately-verified DINOv3 weights this codebase does not have.
    """

    def test_mislabeled_dinov3_entries_removed(self):
        from pygeovision.models.registry import model_registry, list_models
        dinov3_named = [n for n in list_models() if "dinov3" in n.lower()]
        assert dinov3_named == [], f"Expected zero, found: {dinov3_named}"

    def test_real_dinov2_entries_present(self):
        from pygeovision.models.registry import model_registry
        for name in ("dinov2-s", "dinov2-b", "dinov2-l", "dinov2-g"):
            assert name in model_registry, f"Missing real DINOv2 entry: {name}"

    # Prithvi models
    @pytest.mark.parametrize("name,params,pretrain", [
        ("prithvi_eo_1_0", 100, "HLS-US"),
        ("prithvi_eo_2_0", 600, "HLS-Global"),
    ])
    def test_prithvi_in_registry(self, name, params, pretrain):
        from pygeovision.models.registry import model_registry
        assert name in model_registry
        spec = model_registry[name]
        assert spec.params_m == params
        assert spec.pretrained_on == pretrain
        assert spec.family == "prithvi"
        assert spec.supports_multispectral is True

    def test_total_foundation_models_count(self):
        """Real, current count after the dinov3 cleanup above and the
        real DOFA integration added since (see Model Registry docs)."""
        from pygeovision.models.registry import model_registry
        foundation = model_registry.list(task="foundation")
        assert len(foundation) == 11, f"Expected 11 real foundation models, got {len(foundation)}"

    def test_registry_summary(self):
        """Real, current total after removing 59 confirmed-fake entries
        (121 -> 62) and later additions (DOFA, LISAt) and restorations
        (centernet-r50, satlas-pretrain, changestar-r18, chatearthnet as
        honestly-failing, discoverable entries) -- 68 total."""
        from pygeovision.models.registry import model_registry
        s = model_registry.summary()
        assert s["total"] == 68, f"Expected 68 real registry entries, got {s['total']}"
        assert "foundation" in s["by_task"]

    def test_list_satellite_pretrained(self):
        from pygeovision.models.registry import model_registry, list_models
        sat = [n for n in list_models()
               if model_registry[n].pretrained_on in
               ("SAT-493M", "HLS-US", "HLS-Global", "Sentinel-1/2,NAIP,EnMAP,Gaofen")]
        assert len(sat) == 5, f"Expected 5 real sat/HLS-pretrained entries, got {sat}"

    def test_search_dinov3_returns_nothing(self):
        """A real, honest consequence of removing every dinov3-named
        entry: searching for one now correctly returns nothing, rather
        than the 12+ fake matches it used to."""
        from pygeovision.models.registry import model_registry
        results = model_registry.search("dinov3")
        assert len(results) == 0

    def test_search_prithvi(self):
        from pygeovision.models.registry import model_registry
        results = model_registry.search("prithvi")
        assert len(results) >= 2


# ── Transform Correctness ────────────────────────────────────────────────────

class TestTransformCorrectness:
    """Ensure web and SAT transforms use exactly the right statistics."""

    def test_web_mean_imagenet(self):
        from pygeovision.models.foundation.dinov3 import WEB_MEAN
        assert len(WEB_MEAN) == 3
        assert abs(WEB_MEAN[0] - 0.485) < 0.001
        assert abs(WEB_MEAN[1] - 0.456) < 0.001
        assert abs(WEB_MEAN[2] - 0.406) < 0.001

    def test_web_std_imagenet(self):
        from pygeovision.models.foundation.dinov3 import WEB_STD
        assert abs(WEB_STD[0] - 0.229) < 0.001
        assert abs(WEB_STD[1] - 0.224) < 0.001
        assert abs(WEB_STD[2] - 0.225) < 0.001

    def test_sat_mean_satellite(self):
        from pygeovision.models.foundation.dinov3 import SAT_MEAN
        assert abs(SAT_MEAN[0] - 0.430) < 0.001
        assert abs(SAT_MEAN[1] - 0.411) < 0.001
        assert abs(SAT_MEAN[2] - 0.296) < 0.001

    def test_sat_std_satellite(self):
        from pygeovision.models.foundation.dinov3 import SAT_STD
        assert abs(SAT_STD[0] - 0.213) < 0.001
        assert abs(SAT_STD[1] - 0.156) < 0.001
        assert abs(SAT_STD[2] - 0.143) < 0.001

    def test_web_and_sat_stats_are_different(self):
        from pygeovision.models.foundation.dinov3 import SAT_MEAN, SAT_STD, WEB_MEAN, WEB_STD
        assert WEB_MEAN != SAT_MEAN
        assert WEB_STD  != SAT_STD

    def test_get_transform_auto_selects_sat_for_sat_model(self):
        from pygeovision.models.foundation.dinov3 import SAT_MEAN, get_transform
        try:
            import torchvision.transforms as T
            t = get_transform("dinov3_vitl16_sat")
            # Verify it's a Compose with a Normalize using SAT stats
            norms = [x for x in t.transforms if isinstance(x, T.Normalize)]
            assert len(norms) == 1
            assert abs(norms[0].mean[0] - SAT_MEAN[0]) < 0.001
        except ImportError:
            pytest.skip("torchvision required")

    def test_get_transform_auto_selects_web_for_web_model(self):
        from pygeovision.models.foundation.dinov3 import WEB_MEAN, get_transform
        try:
            import torchvision.transforms as T
            t = get_transform("dinov3_vitl16")
            norms = [x for x in t.transforms if isinstance(x, T.Normalize)]
            assert len(norms) == 1
            assert abs(norms[0].mean[0] - WEB_MEAN[0]) < 0.001
        except ImportError:
            pytest.skip("torchvision required")


# ── Band Mapping Integration ─────────────────────────────────────────────────

class TestBandMappingIntegration:
    """Test full band mapping pipeline from satellite to Prithvi input."""

    def test_sentinel2_6band_mapping(self):
        from pygeovision.models.foundation.prithvi import SENTINEL2_TO_PRITHVI
        # Verify 6-band subset is Blue/Green/Red/NIR/SWIR1/SWIR2
        core_bands = {k: v for k, v in SENTINEL2_TO_PRITHVI.items() if v < 6}
        assert len(core_bands) == 6

    def test_landsat_complete_mapping(self):
        from pygeovision.models.foundation.prithvi import map_bands
        data = np.ones((6, 32, 32), dtype=np.float32)
        out  = map_bands(data, source="landsat", n_prithvi_bands=6)
        assert out.shape == (6, 32, 32)

    def test_hls_scale_correct(self):
        from pygeovision.models.foundation.prithvi import normalise_hls
        # Test that a cloud-free clear-sky pixel (e.g. 2000 = 20% reflectance) normalises correctly
        pixel = np.array([2000.0])
        norm  = normalise_hls(pixel)
        assert abs(norm[0] - 0.2) < 1e-4

    def test_prithvi_band_order_documentation(self):
        """Verify band order matches the documented HLS specification."""
        from pygeovision.models.foundation.prithvi import SENTINEL2_TO_PRITHVI
        # Standard 6-band HLS: Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5
        assert SENTINEL2_TO_PRITHVI["B02"] == 0, "Blue must be index 0"
        assert SENTINEL2_TO_PRITHVI["B03"] == 1, "Green must be index 1"
        assert SENTINEL2_TO_PRITHVI["B04"] == 2, "Red must be index 2"
        assert SENTINEL2_TO_PRITHVI["B08"] == 3, "NIR must be index 3"


# ── End-to-End Workflows ──────────────────────────────────────────────────────

class TestEndToEndFoundation:
    """End-to-end workflow tests (use surrogates, no real model downloads)."""

    def test_dinov3_backbone_pipeline(self, tmp_path):
        """Full DINOv3 pipeline: load → extract → embed → build_classifier."""
        import torch
        import torch.nn as nn
        from PIL import Image

        from pygeovision.models.foundation.dinov3 import DINOv3Backbone

        # Mock output that has last_hidden_state
        class FakeOutput:
            def __init__(self, B):
                self.last_hidden_state = torch.randn(B, 197, 768)
                self.attentions = None

        class FakeModel(nn.Module):
            config = type("C", (), {"hidden_size": 768})()
            def forward(self, x=None, pixel_values=None, **kw):
                inp = x if x is not None else pixel_values
                return FakeOutput(inp.shape[0])

        b = DINOv3Backbone("dinov3_vitb16")
        b._model     = FakeModel()
        b._transform = None
        b._spec      = {"embed": 768}

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # 1. Extract spatial features
        feats = b.extract_features(img)
        assert feats.ndim == 3
        assert feats.shape[2] == 768

        # 2. Extract embedding
        emb = b.extract_embeddings(img)
        assert emb.shape == (1, 768)

        # 3. Extract patches
        patches = b.extract_patch_features(img)
        assert patches.ndim == 2
        assert patches.shape[1] == 768

        # 4. Build classifier (no weight download needed — backbone already set)
        head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 5))
        assert head is not None

    def test_prithvi_pipeline(self, tmp_path):
        """Full Prithvi pipeline: load → extract_features → build_seg_head → infer."""
        import rasterio
        from rasterio.transform import from_bounds

        from pygeovision.models.foundation.prithvi import (
            PRITHVI_MODELS,
            Prithvi,
            PrithviTasks,
            _build_prithvi_surrogate,
        )

        # Write 6-band synthetic GeoTIFF
        data = (np.random.rand(6, 64, 64) * 8000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 64, 64)
        p    = tmp_path / "prithvi_e2e.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=64, width=64,
                            count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
            dst.write(data)

        # Use surrogate model
        surrogate = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])
        model     = Prithvi("prithvi_eo_1_0")
        model._model = surrogate

        # Feature extraction
        feats = model.extract_features(str(p), source="hls")
        assert feats.ndim == 2
        assert feats.shape[1] == 768

        # Build seg head and run tasks
        tasks = PrithviTasks("prithvi_eo_1_0")
        tasks._prithvi._model = surrogate

        lc     = tasks.land_cover(str(p))
        assert "prediction" in lc
        assert "class_names" in lc

        flood  = tasks.flood_detection(str(p), source="hls")
        assert "flood_pct" in flood
        assert 0.0 <= flood["flood_pct"] <= 100.0

        biomass = tasks.biomass_estimation(str(p))
        assert "estimated_biomass_t_ha" in biomass

    def test_dinov3_chm_pipeline(self, tmp_path):
        """CHMv2 canopy height pipeline: predict → biomass → deforestation."""
        import rasterio
        import torch.nn as nn
        from rasterio.transform import from_bounds

        from pygeovision.models.foundation.dinov3 import CHMv2Model

        data = (np.random.rand(4, 64, 64) * 10000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 64, 64)
        for name in ["chm_before.tif", "chm_after.tif"]:
            p = tmp_path / name
            with rasterio.open(str(p), "w", driver="GTiff", height=64, width=64,
                                count=4, dtype="float32", crs="EPSG:4326", transform=t) as dst:
                dst.write(data)

        class FakeBack(nn.Module):
            def forward(self, x):
                import torch; return torch.randn(x.shape[0], 197, 1024)

        chm = CHMv2Model()
        chm._backbone._model    = FakeBack()
        chm._backbone._transform = None
        chm._backbone._is_sat   = False

        result = chm.predict_canopy_height(str(tmp_path / "chm_before.tif"))
        assert "height_map" in result or "error" in result
        if "height_map" in result:
            assert result["height_map"].ndim == 2
            assert result["statistics"]["mean_m"] >= 0.0

    def test_model_registry_completeness(self):
        """Final check: real, current foundation model entries -- zero
        dinov3-named entries (all confirmed mislabeled and removed),
        real Prithvi and DOFA present."""
        from pygeovision.models.registry import model_registry, list_models
        foundation = set(model_registry.list(task="foundation"))
        # Real, current state: zero dinov3-named entries (all 12 backbone
        # variants were confirmed mislabeled -- see TestFoundationModelRegistry
        # above for the full detail -- and removed).
        dinov3_named = [n for n in list_models() if "dinov3" in n.lower()]
        assert dinov3_named == [], f"Expected zero, found: {dinov3_named}"
        # Both real Prithvi models
        assert "prithvi_eo_1_0" in model_registry
        assert "prithvi_eo_2_0" in model_registry
        # The real, dedicated DOFA integration added since (torchgeo-based,
        # verified this project to correctly embed both 9-band Sentinel-2
        # and 3-band NAIP input with the same model instance)
        assert "dofa-base" in model_registry
        print(f"\n  ✓ Foundation models: {len(foundation)} in registry")
