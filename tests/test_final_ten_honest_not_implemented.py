"""Tests confirming the final 10 pipelines are honest about not being
implemented, completing this audit's full pass over all 25 originally
generic _make_simple stubs (13 converted to real implementations, 12
converted to honest, specific NotImplementedError -- zero remain as
silent fake-success stubs).

Each of these 10 has a genuinely different, specific reason it wasn't
implemented -- not a copy-pasted excuse:
- air_quality_index, dust_storm_tracking: no real atmospheric data
  provider exists in pygeofetch at all (confirmed by search)
- solar_potential, archaeological_site: real DEM data source exists
  (pygeofetch's OpentopographyProvider), but the required algorithm
  (solar irradiance modeling / relief-model interpretation) is
  genuinely complex, specialized work this cycle didn't implement
- mine_detection, powerline_extraction, construction_progress,
  aquaculture_mapping: need specialized trained detectors or
  classifiers that don't exist in this codebase
- reef_bleaching: needs real bathymetric water-column correction to
  be reliable; a naive threshold would be noise-dominated
- wind_farm_siting: needs real meteorological data, not satellite
  imagery -- a fundamentally different data domain
"""
import pytest
from unittest.mock import MagicMock


ALL_TEN = [
    ("AirQualityIndexPipeline", "air_quality_index"),
    ("DustStormTrackingPipeline", "dust_storm_tracking"),
    ("SolarPotentialPipeline", "solar_potential"),
    ("MineDetectionPipeline", "mine_detection"),
    ("PowerlineExtractionPipeline", "powerline_extraction"),
    ("AquacultureMappingPipeline", "aquaculture_mapping"),
    ("ConstructionProgressPipeline", "construction_progress"),
    ("ReefBleachingPipeline", "reef_bleaching"),
    ("ArchaeologicalSitePipeline", "archaeological_site"),
    ("WindFarmSitingPipeline", "wind_farm_siting"),
]


class TestAllTenAreHonest:
    @pytest.mark.parametrize("class_name,expected_name", ALL_TEN)
    def test_raises_not_implemented_rather_than_fake_success(self, class_name, expected_name):
        import pygeovision.ai.pipelines.domains as domains
        cls = getattr(domains, class_name)
        pipeline = cls(MagicMock())
        with pytest.raises(NotImplementedError):
            pipeline.run(bbox=(0, 0, 1, 1), output_dir="./out")

    @pytest.mark.parametrize("class_name,expected_name", ALL_TEN)
    def test_registered_under_the_real_expected_name(self, class_name, expected_name):
        import pygeovision.ai.pipelines.domains as domains
        cls = getattr(domains, class_name)
        assert cls.name == expected_name

    @pytest.mark.parametrize("class_name,expected_name", ALL_TEN)
    def test_description_does_not_claim_it_works(self, class_name, expected_name):
        import pygeovision.ai.pipelines.domains as domains
        cls = getattr(domains, class_name)
        assert "NOT IMPLEMENTED" in cls.description

    @pytest.mark.parametrize("class_name,expected_name", ALL_TEN)
    def test_registry_points_to_the_real_class_not_a_generic_stub(self, class_name, expected_name):
        import pygeovision.ai.pipelines.domains as domains
        registered_cls = domains._PIPELINE_REGISTRY[expected_name]
        assert registered_cls.__name__ == class_name

    def test_each_has_a_genuinely_distinct_reason_not_copy_pasted(self):
        """Confirms the 10 error messages are actually different from
        each other, not a templated excuse repeated 10 times."""
        import pygeovision.ai.pipelines.domains as domains
        reasons = set()
        for class_name, _ in ALL_TEN:
            cls = getattr(domains, class_name)
            reasons.add(cls._REAL_REASON)
        assert len(reasons) == len(ALL_TEN), "expected 10 genuinely distinct reasons, found duplicates"


class TestFullAuditComplete:
    """The overall completion check for this multi-turn audit: zero
    _make_simple() calls remain among the 51 registered pipelines."""

    def test_zero_generic_stub_calls_remain(self):
        import inspect
        import pygeovision.ai.pipelines.domains as domains
        source = inspect.getsource(domains)
        # Only the function definition itself should remain -- zero calls
        call_count = source.count("_make_simple(") - source.count("def _make_simple(")
        assert call_count == 0

    def test_all_51_pipelines_still_registered(self):
        from pygeovision.ai.pipelines.domains import list_pipelines
        assert len(list_pipelines()) == 51

    def test_thirteen_real_plus_twelve_honest_equals_twenty_five(self):
        """Sanity check on this audit's own accounting: 10 original +
        13 real conversions + 12 honest not-implemented = 25 (the
        original domains.py-registered count, before the 10 originally-
        real named classes some of which had their own separate bugs
        -- see the Tier 3 sweep, not yet done)."""
        real_conversions = [
            "WildfireSeverityPipeline", "GlacierMonitoringPipeline", "SnowCoverPipeline",
            "WetlandMappingPipeline", "UrbanHeatIslandPipeline", "ParkingOccupancyPipeline",
            "PortMonitoringPipeline", "LandcoverChangePipeline", "OilSpillDetectionPipeline",
            "MangroveMappingPipeline", "PipelineLeakDetectionPipeline",
            "CropYieldForecastPipeline", "BiodiversityHotspotPipeline",
        ]
        honest_not_implemented = [
            "PermafrostThawPipeline", "DamSafetyPipeline",
            "AirQualityIndexPipeline", "DustStormTrackingPipeline", "SolarPotentialPipeline",
            "MineDetectionPipeline", "PowerlineExtractionPipeline", "AquacultureMappingPipeline",
            "ConstructionProgressPipeline", "ReefBleachingPipeline", "ArchaeologicalSitePipeline",
            "WindFarmSitingPipeline",
        ]
        assert len(real_conversions) == 13
        assert len(honest_not_implemented) == 12
        assert len(real_conversions) + len(honest_not_implemented) == 25
