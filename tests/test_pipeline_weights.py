"""The pipelines that run a neural network never run an untrained one.

Before launch, building_footprints, solar_detection, crop_monitoring,
disaster_assessment, deforestation, urban_growth (and the model branches of
change_detection / land_cover / water_bodies) built their architecture with
random weights and returned ``success=True``. Two identical runs disagreed on
~38% of pixels. These tests pin the replacement behaviour: no checkpoint, no
run -- and a real, matching checkpoint runs deterministically.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from pipeline_synth import BBOX, no_network, patch_downloads, read, write_raster  # noqa: E402

from pygeovision.ai.pipelines import (  # noqa: E402
    BuildingFootprintsPipeline, ChangeDetectionPipeline, CropMonitoringPipeline, LandslideDetectionPipeline, RoadExtractionPipeline,
    DeforestationPipeline, DisasterAssessmentPipeline, LandCoverPipeline,
    ModelWeightsRequired, SolarDetectionPipeline, UrbanGrowthPipeline, WaterBodiesPipeline,
    pipeline_kinds,
)

TILE = dict(tile_size=64, overlap=16)


def make_checkpoint(path, name="unet_resnet50", num_classes=2, in_channels=3, seed=0, wrap=True):
    from pygeovision.ai.models.registry import registry
    torch.manual_seed(seed)
    model = registry.build(name, num_classes=num_classes, in_channels=in_channels, pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1} if wrap else model.state_dict(), path)
    return path


@pytest.fixture
def scene(tmp_path):
    rs = np.random.RandomState(0)
    return write_raster(tmp_path / "scene.tif", rs.rand(3, 96, 96))


@pytest.fixture
def pair(tmp_path):
    rs = np.random.RandomState(1)
    return (write_raster(tmp_path / "b.tif", rs.rand(3, 96, 96)),
            write_raster(tmp_path / "a.tif", rs.rand(3, 96, 96)))


# (class, kwargs that make the call otherwise valid)
GATED = [
    (SolarDetectionPipeline, {}),
    (RoadExtractionPipeline, {}),
    (LandslideDetectionPipeline, {"date_before": "2024-01", "date_after": "2024-02"}),
    (CropMonitoringPipeline, {"crop_classes": ["wheat", "maize"]}),
    (DisasterAssessmentPipeline, {"date_before": "2024-01", "date_after": "2024-02"}),
    (DeforestationPipeline, {"date_before": "2020", "date_after": "2024"}),
    (UrbanGrowthPipeline, {"date_before": "2018", "date_after": "2024"}),
    (ChangeDetectionPipeline, {"method": "model"}),
    (WaterBodiesPipeline, {"method": "model"}),
    (LandCoverPipeline, {"source": "unet_resnet50"}),
]


class TestNoWeightsNoRun:
    @pytest.mark.parametrize("cls,kw", GATED, ids=[c.__name__ for c, _ in GATED])
    def test_refuses_and_does_not_download(self, tmp_path, monkeypatch, cls, kw):
        hit = no_network(monkeypatch)
        r = cls(object()).run(BBOX, tmp_path / "o", **kw)
        assert not r.success
        assert "trained weights" in r.error and "checkpoint" in r.error
        assert hit["n"] == 0, "must fail before spending bandwidth on imagery"
        assert r.output_path is None

    def test_exception_type_is_public(self):
        assert issubclass(ModelWeightsRequired, RuntimeError)

    def test_missing_file(self, tmp_path, monkeypatch):
        no_network(monkeypatch)
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path, checkpoint=str(tmp_path / "nope.pth"))
        assert not r.success and "not found" in r.error

    def test_kind_marks_exactly_the_weight_pipelines(self):
        kinds = pipeline_kinds()
        needs = {n for n, k in kinds.items() if k == "needs_weights"}
        assert needs == {"solar_detection", "crop_monitoring", "disaster_assessment", "deforestation", "urban_growth",
                         "road_extraction", "landslide_detection"}

    def test_crop_monitoring_needs_class_count(self, tmp_path, monkeypatch):
        no_network(monkeypatch)
        r = CropMonitoringPipeline(object()).run(BBOX, tmp_path, checkpoint="x.pth")
        assert not r.success and "crop_classes" in r.error


class TestCheckpointValidation:
    def test_wrong_class_count_is_rejected_not_silently_loaded(self, tmp_path, monkeypatch, scene):
        patch_downloads(monkeypatch, single=scene)
        ck = make_checkpoint(tmp_path / "c3.pth", num_classes=3)
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path / "o", checkpoint=str(ck), num_classes=2, **TILE)
        assert not r.success and "does not match" in r.error

    def test_wrong_band_count_is_rejected(self, tmp_path, monkeypatch, scene):
        patch_downloads(monkeypatch, single=scene)
        ck = make_checkpoint(tmp_path / "c.pth", in_channels=3)
        r = BuildingFootprintsPipeline(object()).run(
            BBOX, tmp_path / "o", checkpoint=str(ck), bands=("red", "green", "blue", "nir"), **TILE)
        assert not r.success

    def test_hub_fallback_to_another_architecture_is_refused(self, tmp_path, monkeypatch, scene):
        from pygeovision.ai.models.hub import ModelHub
        ck = make_checkpoint(tmp_path / "c.pth")
        real_load = ModelHub.load

        def fallback(self, *a, **k):
            m = real_load(self, *a, **k)
            m._pygeovision_fallback_used, m._pygeovision_actual_name = True, "other_net"
            return m

        monkeypatch.setattr(ModelHub, "load", fallback)
        patch_downloads(monkeypatch, single=scene)
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path / "o", checkpoint=str(ck), **TILE)
        assert not r.success and "refusing" in r.error


class TestRunsWithRealCheckpoint:
    def test_single_image_pipeline_is_deterministic(self, tmp_path, monkeypatch, scene):
        patch_downloads(monkeypatch, single=scene)
        ck = make_checkpoint(tmp_path / "c.pth")
        a = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path / "a", checkpoint=str(ck), **TILE)
        b = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path / "b", checkpoint=str(ck), **TILE)
        assert a.success, a.error
        assert read(a.output_path).shape == (1, 96, 96)
        assert np.array_equal(read(a.output_path), read(b.output_path))
        assert 0.0 <= a.stats["building_coverage"] <= 1.0
        assert a.stats["building_coverage"] == pytest.approx(float((read(a.output_path)[0] == 1).mean()))

    def test_different_weights_give_different_maps(self, tmp_path, monkeypatch, scene):
        """The checkpoint is what determines the output (it is actually loaded)."""
        patch_downloads(monkeypatch, single=scene)
        outs = []
        for seed in (0, 1):
            ck = make_checkpoint(tmp_path / f"c{seed}.pth", seed=seed)
            r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path / f"o{seed}", checkpoint=str(ck), **TILE)
            outs.append(read(r.output_path))
        assert not np.array_equal(outs[0], outs[1])

    def test_bare_state_dict_checkpoints_load_too(self, tmp_path, monkeypatch, scene):
        patch_downloads(monkeypatch, single=scene)
        ck = make_checkpoint(tmp_path / "bare.pth", wrap=False)
        assert BuildingFootprintsPipeline(object()).run(BBOX, tmp_path / "o", checkpoint=str(ck), **TILE).success

    def test_checkpoint_written_by_the_trainers_checkpoint_manager(self, tmp_path, monkeypatch, scene):
        from pygeovision.ai.models.registry import registry
        from pygeovision.ai.training.trainer import CheckpointManager
        patch_downloads(monkeypatch, single=scene)
        model = registry.build("unet_resnet50", num_classes=2, in_channels=3, pretrained=False)
        cm = CheckpointManager(tmp_path / "run")
        cm.save(model, 1, {"val_iou": 0.7})
        r = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path / "o", checkpoint=str(cm.best_path()), **TILE)
        assert r.success, r.error

    def test_solar_detection_uses_small_tiles_by_default(self):
        import inspect
        sig = inspect.signature(SolarDetectionPipeline.run)
        assert sig.parameters["tile_size"].default == 256 and sig.parameters["overlap"].default == 32

    def test_crop_monitoring_derives_class_count(self, tmp_path, monkeypatch, scene):
        patch_downloads(monkeypatch, single=scene)
        ck = make_checkpoint(tmp_path / "c.pth", num_classes=3)
        r = CropMonitoringPipeline(object()).run(BBOX, tmp_path / "o", crop_classes=["wheat", "maize"],
                                                 model="unet_resnet50", checkpoint=str(ck), **TILE)
        assert r.success, r.error
        assert set(np.unique(read(r.output_path))) <= {0, 1, 2}

    def test_water_bodies_model_branch(self, tmp_path, monkeypatch, scene):
        patch_downloads(monkeypatch, single=scene)
        ck = make_checkpoint(tmp_path / "c.pth")
        r = WaterBodiesPipeline(object()).run(BBOX, tmp_path / "o", method="model", checkpoint=str(ck), **TILE)
        assert r.success, r.error
        assert 0.0 <= r.stats["water_coverage"] <= 1.0


class TestSiamesePipelines:
    def _ck(self, tmp_path, num_classes):
        return make_checkpoint(tmp_path / f"s{num_classes}.pth", name="siamese_unet", num_classes=num_classes)

    def test_disaster_assessment(self, tmp_path, monkeypatch, pair):
        patch_downloads(monkeypatch, pair=pair)
        r = DisasterAssessmentPipeline(object()).run(
            BBOX, tmp_path / "o", date_before="2024-01", date_after="2024-02",
            checkpoint=str(self._ck(tmp_path, 4)), num_classes=4, **TILE)
        assert r.success, r.error
        assert set(np.unique(read(r.output_path))) <= {0, 1, 2, 3}

    def test_change_detection_model_branch(self, tmp_path, monkeypatch, pair):
        patch_downloads(monkeypatch, pair=pair)
        r = ChangeDetectionPipeline(object()).run(
            BBOX, tmp_path / "o", method="model", model="siamese_unet",
            checkpoint=str(self._ck(tmp_path, 2)), **TILE)
        assert r.success, r.error
        assert 0.0 <= r.stats["change_fraction"] <= 1.0

    def test_legacy_and_year_only_date_arguments(self, tmp_path, monkeypatch, pair):
        """pre_date/post_date, baseline_year/analysis_year and start_year/end_year still work,
        and a bare year means a month (July, or June for Landsat)."""
        rec: dict = {}
        patch_downloads(monkeypatch, pair=pair, record=rec)
        ck2, ck3 = self._ck(tmp_path, 2), self._ck(tmp_path, 3)
        results = [
            DisasterAssessmentPipeline(object()).run(
                BBOX, tmp_path / "a", pre_date="2023-05", post_date="2023-06",
                checkpoint=str(self._ck(tmp_path, 4)), num_classes=4, **TILE),
            DeforestationPipeline(object()).run(
                BBOX, tmp_path / "b", baseline_year="2019", analysis_year="2023",
                model="siamese_unet", checkpoint=str(ck3), num_classes=3, **TILE),
            UrbanGrowthPipeline(object()).run(
                BBOX, tmp_path / "c", start_year="2017", end_year="2022",
                checkpoint=str(ck2), **TILE),
        ]
        assert all(r.success for r in results), [r.error for r in results]
        got = [(c["date_before"], c["date_after"]) for c in rec["calls"]]
        assert got == [("2023-05", "2023-06"), ("2019-07", "2023-07"), ("2017-06", "2022-06")]
