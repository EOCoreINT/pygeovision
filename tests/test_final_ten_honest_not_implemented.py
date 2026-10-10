"""Tests confirming the real, current, honest state of the pipelines this
audit's earlier round left as NotImplementedError stubs.

This supersedes an earlier version of this test file that documented a
prior state: at that point, 10 pipelines were honest NotImplementedError
stubs. In a later audit round, 9 of those 10 were given real, verified
implementations (Horn's method slope/aspect for solar_potential, a real
Local Relief Model for archaeological_site, real shape-fill-ratio
classification for aquaculture_mapping, real size-filtered land-cover
transitions for mine_detection/construction_progress, real NDVI-gap +
shape-elongation detection for powerline_extraction, a real bi-temporal
brightness-anomaly proxy for reef_bleaching, a real terrain/land-use
suitability screen for wind_farm_siting, and a real visible-band haze
proxy for dust_storm_tracking) -- each verified against synthetic ground
truth or independent physical calculation at the time, not just written
and assumed correct.

Only air_quality_index remains a deliberate stub, for a different,
specific reason than "no data provider exists" (which was true, but is
not the deciding reason): the same haze-proxy technique used for
dust_storm_tracking would risk being read as calibrated health guidance
under this specific name, and no real, calibrated atmospheric-composition
data source exists in this codebase's real dependencies to justify that
framing even if the technique itself were reused.
"""
import pytest
from unittest.mock import MagicMock


NOW_REAL = [
    "DustStormTrackingPipeline", "SolarPotentialPipeline", "MineDetectionPipeline",
    "PowerlineExtractionPipeline", "AquacultureMappingPipeline",
    "ConstructionProgressPipeline", "ReefBleachingPipeline",
    "ArchaeologicalSitePipeline", "WindFarmSitingPipeline",
]


class TestNineNowHaveRealImplementations:
    """These 9 were honest NotImplementedError stubs at one point in this
    audit; verify they no longer are, and their descriptions honestly
    reflect that (say what they really do, not "NOT IMPLEMENTED")."""

    @pytest.mark.parametrize("class_name", NOW_REAL)
    def test_description_no_longer_claims_not_implemented(self, class_name):
        import pygeovision.ai.pipelines.domains as domains
        cls = getattr(domains, class_name)
        assert "NOT IMPLEMENTED" not in cls.description, (
            f"{class_name} has a real implementation now -- its description "
            f"should say what it actually does, not claim it's unimplemented."
        )
        assert cls.description.startswith("Real "), (
            f"{class_name}'s real description should be explicit that this "
            f"is a genuine implementation, consistent with every other real "
            f"pipeline's description convention in this codebase."
        )

    @pytest.mark.parametrize("class_name", NOW_REAL)
    def test_run_does_not_raise_not_implemented(self, class_name, tmp_path):
        """A real implementation may still fail for other real reasons
        (no imagery found, invalid input) -- the one thing it must not do
        is raise NotImplementedError, which would mean it's still a stub
        despite its description claiming otherwise."""
        import pygeovision.ai.pipelines.domains as domains
        cls = getattr(domains, class_name)
        pipeline = cls(MagicMock())
        try:
            pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path))
        except NotImplementedError:
            pytest.fail(
                f"{class_name}.run() raised NotImplementedError -- its "
                f"description claims a real implementation, but the code "
                f"doesn't match that claim."
            )
        except Exception:
            pass  # any other real failure (mocked client, no real data) is fine


class TestAirQualityIndexStillHonestlyUnimplemented:
    """The one pipeline that's still a deliberate stub, and why."""

    def test_raises_not_implemented(self):
        import pygeovision.ai.pipelines.domains as domains
        pipeline = domains.AirQualityIndexPipeline(MagicMock())
        with pytest.raises(NotImplementedError):
            pipeline.run(bbox=(0, 0, 1, 1), output_dir="./out")

    def test_description_says_not_implemented(self):
        import pygeovision.ai.pipelines.domains as domains
        assert "NOT IMPLEMENTED" in domains.AirQualityIndexPipeline.description


class TestFullAuditComplete:
    """The overall pipeline count, corrected for InSAR removal (a later,
    separate audit decision than the stub-implementation work above):
    two InSAR-dependent pipelines (permafrost_thaw, dam_safety) were
    removed from scope entirely, not converted to stubs or
    implementations -- 51 - 2 = 49."""

    def test_all_49_pipelines_registered(self):
        from pygeovision.ai.pipelines.domains import list_pipelines
        names = list_pipelines()
        assert len(names) == 49
        assert "permafrost_thaw" not in names, "Removed with InSAR support"
        assert "dam_safety" not in names, "Removed with InSAR support"

    def test_zero_generic_stub_calls_remain(self):
        import inspect
        import pygeovision.ai.pipelines.domains as domains
        source = inspect.getsource(domains)
        call_count = source.count("_make_simple(") - source.count("def _make_simple(")
        assert call_count == 0
