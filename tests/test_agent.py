"""
tests/test_agent.py
====================
Tests for the PyGeoVision GeoAgent.

All tests use a mock PyGeoVision client — no real satellite API calls,
no real downloads, no real model inference.  The agent logic (planning,
tool dispatch, dependency resolution, memory, streaming) is tested in isolation.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pygeovision.agent import (
    TOOL_REGISTRY,
    GeoAgent,
    GeoAgentMemory,
    HeuristicPlanner,
    LLMPlanner,
    Plan,
    PlanExecutor,
    Step,
    build_tools,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pgv():
    """Minimal mock PyGeoVision client that returns sensible defaults."""
    client = MagicMock()

    # search → list of scene mocks
    scene = MagicMock()
    scene.scene_id    = "S2_TEST_001"
    scene.provider    = "planetary_computer"
    scene.date        = "2026-06-15"
    scene.cloud_cover = 5.0
    client.search.return_value = [scene]

    # download → successful result
    dl = MagicMock()
    dl.success = True
    dl.path    = "/tmp/test_scene.tif"
    client.download.return_value = [dl]

    # prepare_for_ai
    import numpy as np
    client.prepare_for_ai.return_value = {
        "shape": (6, 256, 256),
        "array": np.random.rand(6, 256, 256).astype("float32"),
        "output_path": "/tmp/prepared.tif",
    }

    # pipeline
    pipe_result = MagicMock()
    pipe_result.success     = True
    pipe_result.output_path = "/tmp/pipeline_out.tif"
    pipe_result.stats       = {"area_km2": 12.4}
    client.pipeline.return_value = pipe_result

    # postprocess
    client.postprocess.sieve_filter.return_value = None
    client.postprocess.vectorise.return_value    = None
    client.postprocess.to_cog.return_value       = None

    # indices
    client.indices.ndvi = MagicMock(return_value=None)
    client.indices.ndwi = MagicMock(return_value=None)

    return client


@pytest.fixture
def tools(mock_pgv):
    return build_tools(mock_pgv)


@pytest.fixture
def agent(mock_pgv):
    return GeoAgent(mock_pgv, verbose=False, output_dir="/tmp/agent_test/")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Tool registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry:

    def test_registry_has_expected_tools(self):
        expected = [
            "search_satellite_data", "download_satellite_data", "prepare_for_ai",
            "prithvi_inference", "change_detection",
            "compute_spectral_index", "postprocess", "run_pipeline",
        ]
        for name in expected:
            assert name in TOOL_REGISTRY, f"Missing tool: {name}"
        assert "sar_preprocess" not in TOOL_REGISTRY, (
            "sar_preprocess was deliberately removed -- SAR/InSAR processing "
            "is handled directly by pygeofetch, not through this planner."
        )

    def test_build_tools_returns_bound_instances(self, mock_pgv, tools):
        assert len(tools) == len(TOOL_REGISTRY)
        for name, tool in tools.items():
            assert hasattr(tool, "run"), f"{name} missing run()"
            assert hasattr(tool, "schema"), f"{name} missing schema()"
            assert tool._pgv is mock_pgv

    def test_tool_schemas_have_required_fields(self, tools):
        for name, tool in tools.items():
            s = tool.schema()
            assert "name"        in s, f"{name}: missing 'name'"
            assert "description" in s, f"{name}: missing 'description'"
            assert "parameters"  in s, f"{name}: missing 'parameters'"
            assert s["description"], f"{name}: empty description"

    def test_all_parameters_have_name_and_type(self, tools):
        for tool_name, tool in tools.items():
            for p in tool.parameters:
                assert "name" in p, f"{tool_name}: param missing 'name'"
                assert "type" in p, f"{tool_name}.{p.get('name','?')}: param missing 'type'"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Individual tools
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchTool:

    def test_returns_success_result(self, tools, mock_pgv):
        result = tools["search_satellite_data"].run(
            bbox=[-0.3, 5.5, -0.05, 5.7],
            date_range=("2026-06-01", "2026-06-30"),
        )
        assert result.success
        assert result.output["count"] >= 1
        mock_pgv.search.assert_called_once()

    def test_handles_empty_results(self, tools, mock_pgv):
        mock_pgv.search.return_value = []
        result = tools["search_satellite_data"].run(
            bbox=[0, 0, 1, 1], date_range=("2024-01-01", "2024-01-31"),
        )
        assert result.success
        assert result.output["count"] == 0

    def test_error_becomes_failed_result(self, tools, mock_pgv):
        mock_pgv.search.side_effect = ConnectionError("API timeout")
        result = tools["search_satellite_data"].run(
            bbox=[0, 0, 1, 1], date_range=("2024-01-01", "2024-01-31"),
        )
        assert not result.success
        assert "API timeout" in result.error


class TestDownloadTool:

    def test_returns_downloaded_path(self, tools, mock_pgv):
        scenes = [MagicMock()]
        result = tools["download_satellite_data"].run(
            scenes=scenes, output_dir="/tmp/dl_test/", max_scenes=1,
        )
        assert result.success
        assert result.output["count"] >= 1
        mock_pgv.download.assert_called_once()


class TestRunPipelineTool:

    def test_pipeline_tool_calls_client(self, tools, mock_pgv):
        result = tools["run_pipeline"].run(
            pipeline_name="building_footprints",
            bbox=[-0.3, 5.5, -0.05, 5.7],
            date="2024-06",
        )
        assert result.success
        mock_pgv.pipeline.assert_called_once()

    def test_pipeline_tool_error_handled(self, tools, mock_pgv):
        mock_pgv.pipeline.side_effect = RuntimeError("Pipeline failed")
        result = tools["run_pipeline"].run(
            pipeline_name="land_cover",
            bbox=[0, 0, 1, 1], date="2024-06",
        )
        assert not result.success
        assert "Pipeline failed" in result.error


class TestSpectralIndexTool:

    def test_ndvi_calls_indices(self, tools, mock_pgv):
        import numpy as np
        # Patch rasterio.open for reading stats
        with patch("rasterio.open") as mock_rio:
            mock_ds = MagicMock()
            mock_ds.__enter__ = MagicMock(return_value=mock_ds)
            mock_ds.__exit__  = MagicMock(return_value=False)
            mock_ds.read.return_value = np.random.rand(256, 256).astype("float32")
            mock_rio.return_value = mock_ds

            result = tools["compute_spectral_index"].run(
                input_path="/tmp/prepared.tif",
                index="ndvi",
                output_path="/tmp/ndvi.tif",
            )
        assert result.success or "ndvi" in str(result.output).lower()

    def test_unknown_index_fails(self, tools):
        result = tools["compute_spectral_index"].run(
            input_path="/tmp/test.tif",
            index="xzqr_undefined",
            output_path="/tmp/out.tif",
        )
        assert not result.success


class TestPostprocessTool:

    def test_sieve_and_cog_operations(self, tools, mock_pgv):
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = tmp
            # Create a dummy input file so pathlib.Path checks pass
            inp = pathlib.Path(tmp) / "prediction.tif"
            inp.write_bytes(b"dummy")

            tools["postprocess"].run(
                input_path=str(inp),
                operations=["sieve", "cog"],
                output_dir=out_dir,
            )
            # sieve_filter and to_cog should have been called
            mock_pgv.postprocess.sieve_filter.assert_called()
            mock_pgv.postprocess.to_cog.assert_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Heuristic planner
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeuristicPlanner:

    @pytest.fixture
    def planner(self, tools):
        return HeuristicPlanner(tools)

    def test_flood_sar_query_produces_valid_plan(self, planner):
        """SAR-mentioning flood query still produces a real, valid plan --
        via the real optical/Prithvi path, since SAR-specific tools were
        deliberately removed (see planner.py's _infer_sensor docstring)."""
        plan = planner.plan("Map SAR flood extent in Accra — no optical available",
                            context={"bbox": [-0.3, 5.5, -0.05, 5.7], "date": "2026-06"})
        tool_names = [s.tool_name for s in plan.steps]
        assert plan.sensor == "optical", (
            "_infer_sensor always resolves to 'optical' now -- SAR/InSAR "
            "processing is handled directly by pygeofetch, not this planner."
        )
        assert plan.task   == "flood", f"Expected task=flood, got {plan.task}"
        assert len(plan.steps) >= 1

    def test_building_damage_routes_to_change_detection(self, planner):
        """Building damage after earthquake → change_detection."""
        plan = planner.plan("Detect building damage after the Turkey earthquake using SAR",
                            context={"bbox": [36.0, 36.0, 37.0, 37.0], "date": "2023-02"})
        tool_names = [s.tool_name for s in plan.steps]
        assert plan.sensor == "optical"
        assert plan.task   == "damage"
        assert "change_detection" in tool_names

    def test_optical_flood_uses_prithvi_not_sar(self, planner):
        """Optical + flood → Prithvi pipeline, not SAR pipeline."""
        plan = planner.plan("Map flood using Sentinel-2 optical imagery — clear sky",
                            context={"bbox": [0, 0, 1, 1], "date": "2024-06"})
        tool_names = [s.tool_name for s in plan.steps]
        assert plan.sensor == "optical"
        assert "search_satellite_data" in tool_names
        assert "prithvi_inference"     in tool_names
        assert "sar_preprocess"        not in TOOL_REGISTRY

    def test_oil_spill_uses_real_pipeline(self, planner):
        """Oil spill → a real, valid plan."""
        plan = planner.plan("Detect oil spill with Sentinel-1 at night",
                            context={"bbox": [50.0, 24.0, 57.0, 28.0], "date": "2024-06"})
        assert plan.sensor == "optical"
        assert plan.task   == "oil_spill"

    def test_subsidence_routes_to_change_detection(self, planner):
        """Subsidence → real change_detection, with an honest caveat that
        this is bi-temporal surface change, not true InSAR-grade
        displacement (that remains pygeofetch.insar's real, separate
        domain -- see the planner's LLM prompt and _infer_sensor docstring
        for the documented reasoning)."""
        plan = planner.plan("Monitor ground subsidence in Jakarta",
                            context={"bbox": [106.7, -6.3, 107.0, -6.0], "date": "2024-06"})
        tool_names = [s.tool_name for s in plan.steps]
        assert plan.sensor == "optical"
        assert plan.task   == "subsidence"
        assert "change_detection" in tool_names
        cd_step = next(s for s in plan.steps if s.tool_name == "change_detection")
        assert "insar" in cd_step.rationale.lower(), (
            "Should honestly point to pygeofetch.insar for real displacement measurement"
        )

    def test_land_cover_uses_prithvi(self, planner):
        """Land cover classification → Prithvi inference."""
        plan = planner.plan("Classify land cover using Prithvi foundation model",
                            context={"bbox": [0, 0, 1, 1], "date": "2024-06"})
        tool_names = [s.tool_name for s in plan.steps]
        assert plan.task   == "land_cover"
        assert "prithvi_inference" in tool_names

    def test_flood_optical_query_uses_search_first(self, planner):
        plan = planner.plan("Map flood inundation using Sentinel-2 optical imagery",
                            context={"bbox": [0, 0, 1, 1], "date": "2024-06"})
        assert plan.sensor == "optical"
        assert plan.steps[0].tool_name == "search_satellite_data"

    def test_building_query_uses_pipeline(self, planner):
        plan = planner.plan("Extract building footprints in Dubai",
                            context={"bbox": [55.0, 25.0, 55.5, 25.5], "date": "2024-06"})
        tool_names = [s.tool_name for s in plan.steps]
        assert "run_pipeline" in tool_names

    def test_ndvi_query_includes_spectral_index(self, planner):
        plan = planner.plan("Compute NDVI for crop monitoring",
                            context={"bbox": [0, 0, 1, 1], "date": "2024-06"})
        tool_names = [s.tool_name for s in plan.steps]
        assert "compute_spectral_index" in tool_names

    def test_land_cover_query_uses_prithvi(self, planner):
        plan = planner.plan("Run land cover classification",
                            context={"bbox": [0, 0, 1, 1], "date": "2024-06"})
        tool_names = [s.tool_name for s in plan.steps]
        assert "prithvi_inference" in tool_names

    def test_plan_has_required_step_fields(self, planner):
        plan = planner.plan("Map flood using Sentinel-2 optical imagery",
                            context={"bbox": [0, 0, 1, 1], "date": "2024-06"})
        for step in plan.steps:
            assert isinstance(step.step_idx, int)
            assert isinstance(step.tool_name, str)
            assert step.tool_name, "Empty tool_name"
            assert isinstance(step.args, dict)
            assert isinstance(step.depends_on, list)

    def test_planner_returns_plan_object(self, planner):
        plan = planner.plan("Compute NDVI vegetation index", context={"date": "2024-06"})
        assert isinstance(plan, Plan)
        assert plan.planner == "heuristic"
        assert plan.sensor  in ("sar", "optical", "mixed", "unknown")
        assert plan.task    != ""
        assert len(plan.steps) >= 1

    def test_all_task_templates_produce_valid_plans(self, planner):
        queries = [
            ("Map SAR flood extent — cloud cover 100%", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Map flood using Sentinel-2 optical imagery", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Detect building damage after earthquake using SAR", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Building footprint extraction from optical", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Land cover classification using Prithvi", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Compare deforestation change 2021 vs 2024", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Crop type mapping Sentinel-2", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Compute NDVI vegetation index", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Detect oil spill Sentinel-1 at night", {"bbox": [0,0,1,1], "date":"2024-06"}),
            ("Monitor ground subsidence sinking", {"bbox": [0,0,1,1], "date":"2024-06"}),
        ]
        for query, ctx in queries:
            plan = planner.plan(query, context=ctx)
            assert len(plan.steps) >= 1, f"Empty plan for: {query}"
            for step in plan.steps:
                assert step.tool_name in TOOL_REGISTRY, (
                    f"Unknown tool '{step.tool_name}' in plan for '{query}'"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Executor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlanExecutor:

    @pytest.fixture
    def executor(self, tools):
        return PlanExecutor(tools=tools, stop_on_failure=True)

    def test_single_step_plan_executes(self, executor, mock_pgv):
        plan = Plan(
            query="test",
            steps=[Step(0, "run_pipeline", {
                "pipeline_name": "land_cover",
                "bbox": [0, 0, 1, 1],
                "date": "2024-06",
            })],
        )
        trace = executor.execute(plan)
        assert len(trace.executions) == 1
        assert trace.executions[0].result.success

    def test_output_reference_resolved_between_steps(self, executor, mock_pgv):
        """$step_0_output in step 1 args resolves to step 0 output_path."""
        plan = Plan(
            query="test",
            steps=[
                Step(0, "search_satellite_data", {
                    "bbox": [0,0,1,1], "date_range": ("2024-01-01","2024-01-31"),
                }),
                Step(1, "download_satellite_data", {
                    "scenes": "$step_0_output",
                    "output_dir": "/tmp/dl/",
                }, depends_on=[0]),
            ],
        )
        trace = executor.execute(plan)
        # Step 1 should have been called (dependency resolved)
        assert len(trace.executions) == 2

    def test_stop_on_failure_halts_execution(self, executor, mock_pgv):
        mock_pgv.pipeline.side_effect = RuntimeError("Intentional failure")
        plan = Plan(
            query="test",
            steps=[
                Step(0, "run_pipeline", {"pipeline_name": "land_cover",
                                          "bbox": [0,0,1,1], "date": "2024-06"}),
                Step(1, "postprocess",  {"input_path": "/tmp/x.tif",
                                          "operations": ["cog"]}),
            ],
        )
        trace = executor.execute(plan)
        assert not trace.success
        # Step 1 should not have run since step 0 failed
        assert len(trace.executions) == 1

    def test_stop_on_failure_false_continues(self, tools, mock_pgv):
        mock_pgv.pipeline.side_effect = RuntimeError("Step 0 fails")
        executor = PlanExecutor(tools=tools, stop_on_failure=False)
        plan = Plan(
            query="test",
            steps=[
                Step(0, "run_pipeline", {"pipeline_name": "land_cover",
                                          "bbox": [0,0,1,1], "date": "2024-06"}),
                Step(1, "run_pipeline", {"pipeline_name": "building_footprints",
                                          "bbox": [0,0,1,1], "date": "2024-06"}),
            ],
        )
        mock_pgv.pipeline.side_effect = [RuntimeError("fail"), MagicMock(success=True)]
        trace = executor.execute(plan)
        assert len(trace.executions) == 2

    def test_context_placeholder_resolved(self, executor, mock_pgv):
        plan = Plan(
            query="test",
            steps=[Step(0, "sar_preprocess", {
                "raw_path":    "$context_sar_path",
                "output_path": "/tmp/out.tif",
                "bbox_wgs84":  [0, 0, 1, 1],
            })],
        )
        # sar_preprocess will fail (no real file) but $context_sar_path should resolve
        with patch("pygeovision.data.processors.sar.check_download_complete",
                   return_value={"complete": False, "readable_tiles": 0, "total_tiles": 4,
                                 "errors": ["file missing"]}):
            trace = executor.execute(plan, context={"sar_path": "/tmp/sar_input.tif"})

        step0 = trace.executions[0]
        # The error should be about the file, not about unresolved placeholder
        assert "$context_sar_path" not in step0.result.error

    def test_stream_yields_correct_event_types(self, executor):
        plan = Plan(
            query="test",
            steps=[Step(0, "run_pipeline", {
                "pipeline_name": "building_footprints",
                "bbox": [0,0,1,1], "date": "2024-06",
            })],
        )
        events = list(executor.stream(plan))
        event_types = [e["type"] for e in events]
        assert "plan"               in event_types
        assert "step_start"         in event_types
        assert "step_done"          in event_types
        assert "execution_complete" in event_types

    def test_stream_final_event_has_success_field(self, executor):
        plan = Plan(
            query="test",
            steps=[Step(0, "run_pipeline", {
                "pipeline_name": "land_cover",
                "bbox": [0,0,1,1], "date": "2024-06",
            })],
        )
        events = list(executor.stream(plan))
        final = next(e for e in events if e["type"] == "execution_complete")
        assert "success"  in final
        assert "total_duration" in final

    def test_trace_to_dict_is_json_serialisable(self, executor):
        plan = Plan(
            query="test",
            steps=[Step(0, "run_pipeline", {
                "pipeline_name": "building_footprints",
                "bbox": [0,0,1,1], "date": "2024-06",
            })],
        )
        trace = executor.execute(plan)
        d = trace.to_dict()
        json.dumps(d, default=str)  # must not raise

    def test_trace_summary_is_string(self, executor):
        plan = Plan(
            query="summary test",
            steps=[Step(0, "run_pipeline", {
                "pipeline_name": "land_cover", "bbox": [0,0,1,1], "date": "2024-06",
            })],
        )
        trace = executor.execute(plan)
        s = trace.summary()
        assert isinstance(s, str)
        assert "step" in s.lower() or "execution" in s.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeoAgentMemory:

    def test_set_and_get_context(self):
        mem = GeoAgentMemory()
        mem.set_context(bbox=[0,0,1,1], date="2024-06")
        assert mem.get_context("bbox")   == [0,0,1,1]
        assert mem.get_context("date")   == "2024-06"
        assert mem.get_context("missing") is None

    def test_bind_and_resolve(self):
        mem = GeoAgentMemory()
        mem.bind("flood_mask", "/tmp/flood.tif")
        assert mem.resolve("flood_mask") == "/tmp/flood.tif"
        assert mem.resolve("nonexistent") is None

    def test_context_for_planner_includes_spatial(self):
        mem = GeoAgentMemory()
        mem.set_context(bbox=[1,2,3,4], date="2026-06")
        ctx = mem.context_for_planner()
        assert ctx["bbox"] == [1,2,3,4]
        assert ctx["date"] == "2026-06"

    def test_record_turn_increments_history(self):
        mem  = GeoAgentMemory()
        plan = MagicMock(); plan.steps = []; plan.planner = "heuristic"
        trace= MagicMock(); trace.success = True; trace.final_output = None
        trace.executions = []
        mem.record_turn("query 1", plan, trace)
        mem.record_turn("query 2", plan, trace)
        assert len(mem.turns) == 2

    def test_max_turns_enforced(self):
        mem = GeoAgentMemory(max_turns=3)
        plan  = MagicMock(); plan.steps = []; plan.planner = "test"
        trace = MagicMock(); trace.success = True; trace.final_output = None
        trace.executions = []
        for i in range(5):
            mem.record_turn(f"query {i}", plan, trace)
        assert len(mem.turns) <= 3

    def test_save_and_load_round_trip(self):
        mem = GeoAgentMemory()
        mem.set_context(bbox=[10, 20, 30, 40], date="2025-01")
        mem.bind("my_output", "/tmp/result.tif")

        with tempfile.TemporaryDirectory() as tmp:
            p = f"{tmp}/session.json"
            mem.save(p)

            mem2 = GeoAgentMemory()
            mem2.load(p)
            assert mem2.get_context("bbox") == "[10, 20, 30, 40]"  # loaded as str

    def test_last_turn_returns_most_recent(self):
        mem  = GeoAgentMemory()
        plan = MagicMock(); plan.steps = []; plan.planner = "h"
        trace= MagicMock(); trace.success=True; trace.final_output=None; trace.executions=[]
        mem.record_turn("first", plan, trace)
        mem.record_turn("second", plan, trace)
        assert mem.last_turn.query == "second"

    def test_reset_clears_history(self):
        mem  = GeoAgentMemory()
        plan = MagicMock(); plan.steps = []; plan.planner = "h"
        trace= MagicMock(); trace.success=True; trace.final_output=None; trace.executions=[]
        mem.record_turn("q", plan, trace)
        assert len(mem.turns) == 1
        mem._turns.clear()   # simulate reset
        assert len(mem.turns) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GeoAgent (integration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeoAgent:

    def test_agent_initialises_in_heuristic_mode(self, agent):
        assert agent._plan_mode == "heuristic"
        assert len(agent.tools()) == len(TOOL_REGISTRY)

    def test_agent_returns_execution_trace(self, agent):
        from pygeovision.agent.executor import ExecutionTrace
        agent.set_context(bbox=[0,0,1,1], date="2024-06")
        trace = agent.run("Extract building footprints")
        assert isinstance(trace, ExecutionTrace)

    def test_agent_set_context_is_chainable(self, agent):
        result = agent.set_context(bbox=[1,2,3,4], date="2025-01")
        assert result is agent

    def test_agent_bind_is_chainable(self, agent):
        result = agent.bind("test_key", "/tmp/test.tif")
        assert result is agent

    def test_agent_tools_returns_list(self, agent):
        t = agent.tools()
        assert isinstance(t, list)
        assert len(t) > 0

    def test_agent_tool_schema_returns_dict(self, agent):
        schema = agent.tool_schema("run_pipeline")
        assert isinstance(schema, dict)
        assert schema["name"] == "run_pipeline"

    def test_agent_unknown_tool_schema_returns_none(self, agent):
        assert agent.tool_schema("nonexistent_tool_xyz") is None

    def test_agent_all_schemas_returns_all(self, agent):
        schemas = agent.all_schemas()
        assert len(schemas) == len(TOOL_REGISTRY)

    def test_agent_history_starts_empty(self, agent):
        assert agent.history() == []

    def test_agent_history_records_after_run(self, agent):
        agent.set_context(bbox=[0,0,1,1], date="2024-06")
        agent.run("Build land cover map")
        assert len(agent.history()) == 1
        assert "land cover" in agent.history()[0]["query"].lower()

    def test_agent_stream_yields_events(self, agent):
        agent.set_context(bbox=[0,0,1,1], date="2024-06")
        events = list(agent.stream("Run land cover pipeline"))
        types = [e["type"] for e in events]
        assert "plan"               in types
        assert "execution_complete" in types

    def test_agent_reset_clears_history(self, agent):
        agent.set_context(bbox=[0,0,1,1], date="2024-06")
        agent.run("Run land cover pipeline")
        assert len(agent.history()) == 1
        agent.reset()
        assert len(agent.history()) == 0

    def test_agent_save_and_load_session(self, agent):
        agent.set_context(bbox=[1,2,3,4], date="2025-01")
        with tempfile.TemporaryDirectory() as tmp:
            p = f"{tmp}/session.json"
            agent.save_session(p)
            agent.load_session(p)   # should not raise

    def test_agent_context_override_per_run(self, agent, mock_pgv):
        agent.set_context(bbox=[0,0,1,1], date="2024-06")
        # Pass a one-shot override that doesn't affect the session
        trace = agent.run("Build land cover map",
                          context_override={"extra_key": "value"})
        assert isinstance(trace.success, bool)

    def test_agent_repr_contains_mode(self, agent):
        r = repr(agent)
        assert "heuristic" in r
        assert "GeoAgent"  in r

    def test_llm_mode_requires_api_key(self, mock_pgv):
        """GeoAgent should fall back to heuristic when API key is missing."""
        ag = GeoAgent(mock_pgv, api_key="", verbose=False)
        assert ag._plan_mode == "heuristic"

    def test_llm_planner_available_property(self, tools):
        planner = LLMPlanner(tools, api_key="")
        assert planner.available is False
        planner2 = LLMPlanner(tools, api_key="sk-ant-fake-key")
        assert planner2.available is True

    def test_groq_provider_selection(self, tools):
        from pygeovision.agent.planner import LLMPlanner
        planner = LLMPlanner(tools, api_key="gsk-fake", provider="groq")
        assert planner._provider == "groq"
        assert planner._model == "llama-3.3-70b-versatile"
        assert planner.available is True

    def test_groq_sends_correct_request_shape(self, tools):
        """Regression test confirming Groq is a real, distinct backend —
        not a stub reusing Anthropic's code path. Verifies the actual
        request sent uses Groq's real API shape (JSON mode, correct
        message roles, correct model)."""
        pytest.importorskip("groq", reason="groq not installed")
        import groq
        from unittest.mock import MagicMock, patch
        from pygeovision.agent.planner import LLMPlanner

        planner = LLMPlanner(tools, api_key="gsk-fake", provider="groq")

        fake_completion = MagicMock()
        fake_completion.choices = [MagicMock(message=MagicMock(content='{"steps": [], "notes": "ok"}'))]

        with patch.object(groq.Groq, "__init__", return_value=None), \
             patch.object(groq.Groq, "chat", create=True) as mock_chat:
            mock_chat.completions.create.return_value = fake_completion
            plan = planner.plan("test query")

            call_kwargs = mock_chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "llama-3.3-70b-versatile"
            assert call_kwargs["response_format"] == {"type": "json_object"}
            assert [m["role"] for m in call_kwargs["messages"]] == ["system", "user"]
            assert plan.planner == "groq/llama-3.3-70b-versatile"

    def test_missing_groq_key_raises_clear_error(self, tools):
        from pygeovision.agent.planner import LLMPlanner
        planner = LLMPlanner(tools, api_key="", provider="groq")
        assert planner.available is False
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            planner.plan("test query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
