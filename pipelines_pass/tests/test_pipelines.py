"""The pipeline catalogue: what exists, what was removed, and how it is reached.

PyGeoVision shipped 49 named pipelines; 24 had no working implementation
(placeholder indices, untrained networks, invented constants). They were
removed before launch and 25 remain: 19 run out of the box ("ready") and
6 run a network you trained yourself ("needs_weights").
"""
from __future__ import annotations

import ast
import inspect

import pytest

from pygeovision.ai import pipelines as P
from pygeovision.ai.pipelines import domains
from pygeovision.ai.pipelines.removed import REMOVED_PIPELINES

NEEDS_WEIGHTS = {"building_footprints", "solar_detection", "crop_monitoring",
                 "disaster_assessment", "deforestation", "urban_growth"}


class TestCatalogue:
    def test_counts(self):
        kinds = P.pipeline_kinds()
        assert len(kinds) == 25
        assert sum(k == "ready" for k in kinds.values()) == 19
        assert {n for n, k in kinds.items() if k == "needs_weights"} == NEEDS_WEIGHTS

    def test_sorted_and_unique(self):
        names = P.list_pipelines()
        assert names == sorted(set(names))

    def test_original_49_are_exactly_kept_plus_removed(self):
        kept, removed = set(P.list_pipelines()), set(REMOVED_PIPELINES)
        assert not kept & removed, "a pipeline cannot be both available and removed"
        assert len(kept | removed) == 49
        assert len(removed) == 24

    def test_every_pipeline_is_described(self):
        for name, cls in P.pipeline_catalogue().items():
            assert cls.name == name
            assert cls.description and cls.domain and cls.kind in ("ready", "needs_weights")

    def test_every_removal_says_why(self):
        for name, reason in REMOVED_PIPELINES.items():
            assert len(reason) > 20, name

    @pytest.mark.parametrize("name", sorted(REMOVED_PIPELINES))
    def test_removed_pipeline_raises_with_reason(self, name):
        with pytest.raises(ValueError, match="was removed") as exc:
            P.get_pipeline(name, object())
        assert REMOVED_PIPELINES[name] in str(exc.value)

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            P.get_pipeline("not_a_pipeline", object())

    def test_no_stub_machinery_left(self):
        src = inspect.getsource(domains)
        assert "_make_simple(" not in src
        assert "NotImplementedError" not in src
        # no silent catch-alls in the pipeline modules
        for mod in (P, domains):
            tree = ast.parse(inspect.getsource(mod))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is not None:
                    assert not (isinstance(node.type, ast.Name) and node.type.id == "Exception"
                                and all(isinstance(s, ast.Pass) for s in node.body)), \
                        f"silent 'except Exception: pass' at line {node.lineno}"


class TestEntryPointsAgree:
    def test_one_registry_everywhere(self):
        import pygeovision
        import pygeovision.pipelines as top
        expected = P.list_pipelines()
        assert list(pygeovision.list_all_pipelines()) == expected
        assert top.list_pipelines() == expected
        assert domains.list_pipelines() == expected

    def test_one_result_type_everywhere(self):
        import pygeovision.pipelines as top
        assert top.PipelineResult is P.PipelineResult is domains.PipelineResult

    def test_domain_pipeline_reachable_from_client(self, tmp_path, monkeypatch):
        """Regression: client.pipeline() used to say 'Unknown pipeline' for every domain pipeline."""
        from pipeline_synth import BBOX, no_network
        from pygeovision import PyGeoVision
        no_network(monkeypatch)
        client = PyGeoVision()
        # reaches the pipeline (and its own argument check) instead of "Unknown pipeline"
        r = client.pipeline("wildfire_severity", bbox=BBOX, output_dir=str(tmp_path))
        assert not r.success and "date_before" in r.error

    def test_client_rejects_removed_pipeline_with_reason(self, tmp_path):
        from pygeovision import PyGeoVision
        with pytest.raises(ValueError, match="was removed"):
            PyGeoVision().pipeline("carbon_estimation", bbox=(0, 0, 1, 1), output_dir=str(tmp_path))


class TestPipelineResult:
    def test_str(self, tmp_path):
        ok = P.PipelineResult(pipeline="p", output_path=tmp_path / "o.tif")
        bad = P.PipelineResult(pipeline="p", success=False, error="boom")
        assert "p" in str(ok) and "o.tif" in str(ok)
        assert "FAILED" in str(bad) and "boom" in str(bad)

    def test_name_alias(self):
        assert P.PipelineResult(pipeline="x").name == "x"


class TestFailureModes:
    def test_empty_search_is_a_failure_with_a_message(self, tmp_path):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.search.return_value = []
        r = P.get_pipeline("water_bodies", client).run((-0.15, 51.47, -0.10, 51.52), tmp_path, date="2024-06")
        assert not r.success and r.error

    def test_flood_mapping_requires_both_dates(self, tmp_path):
        r = P.get_pipeline("flood_mapping", object()).run((-0.15, 51.47, -0.10, 51.52), tmp_path)
        assert not r.success and "date_before" in r.error and "date_after" in r.error


click = pytest.importorskip("click")


class TestChannelCli:
    @pytest.fixture
    def run(self):
        from click.testing import CliRunner
        from pygeovision.cli.main import cli
        runner = CliRunner()
        return lambda *args: runner.invoke(cli, ["channel", *args])

    def test_list(self, run):
        out = run("list")
        assert out.exit_code == 0
        assert "25 PyGeoVision pipelines" in out.output
        assert "19 ready" in out.output and "6 need your trained weights" in out.output
        for name in ("flood_mapping", "building_footprints"):
            assert name in out.output

    def test_removed_name_exits_2_with_reason(self, run):
        out = run("carbon_estimation", "--bbox", "0", "0", "1", "1")
        assert out.exit_code == 2
        assert "was removed" in out.output

    def test_unknown_name_exits_2(self, run):
        assert run("nope", "--bbox", "0", "0", "1", "1").exit_code == 2

    def test_missing_bbox_exits_1(self, run):
        assert run("flood_mapping").exit_code == 1

    def test_needs_weights_without_checkpoint_fails_clearly(self, run, tmp_path, monkeypatch):
        from pipeline_synth import no_network
        no_network(monkeypatch)
        out = run("building_footprints", "--bbox", "-0.15", "51.47", "-0.10", "51.52",
                  "--date", "2024-06", "-o", str(tmp_path))
        assert out.exit_code == 1
        assert "checkpoint" in out.output

    def test_param_is_parsed_and_forwarded(self, run, tmp_path, monkeypatch):
        seen = {}

        def fake_run(self, bbox, output_dir, **kw):
            seen.update(kw)
            return P.PipelineResult(pipeline="water_bodies")

        monkeypatch.setattr(P.WaterBodiesPipeline, "run", fake_run)
        out = run("water_bodies", "--bbox", "0", "0", "1", "1", "--date", "2024-06",
                  "--param", "threshold=0.25", "--param", "method=ndwi", "--dates", "2024-01, 2024-02")
        assert out.exit_code == 0, out.output
        assert seen == {"date": "2024-06", "threshold": 0.25, "method": "ndwi",
                        "dates": ["2024-01", "2024-02"]}

    def test_bad_param_is_rejected(self, run):
        assert run("water_bodies", "--bbox", "0", "0", "1", "1", "--param", "oops").exit_code != 0
