"""Validation of YAML workflows (``pygeovision pipeline run``): cycles, duplicates, unknown steps and overrides."""
import pytest

from pygeovision.pipelines.yaml_parser import PipelineYAMLParser

P = PipelineYAMLParser()


def cfg(*steps):
    return {"name": "t", "steps": list(steps)}


def step(name, deps=(), action="search"):
    return {"name": name, "action": action, "depends_on": list(deps)}


def test_a_valid_chain_passes():
    P.validate(cfg(step("a"), step("b", ["a"], "download"), step("c", ["b"], "infer")))


def test_circular_dependency_is_rejected_with_the_cycle():
    with pytest.raises(ValueError, match=r"a -> b -> a"):
        P.validate(cfg(step("a", ["b"]), step("b", ["a"], "download")))


def test_self_dependency_is_rejected():
    with pytest.raises(ValueError, match="circle"):
        P.validate(cfg(step("a", ["a"])))


def test_duplicate_step_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        P.validate(cfg(step("a"), step("a", action="download")))


def test_unknown_dependency_and_action_are_rejected():
    with pytest.raises(ValueError, match="unknown step"):
        P.validate(cfg(step("a", ["zzz"])))
    with pytest.raises(ValueError, match="action"):
        P.validate(cfg(step("a", action="magic")))


def test_overrides_are_json_or_text_and_need_a_known_step():
    c = cfg(step("search"))
    P.apply_overrides(c, ["search.bbox=[1,2,3,4]", "search.name=abc"])
    assert c["steps"][0]["params"] == {"bbox": [1, 2, 3, 4], "name": "abc"}
    with pytest.raises(ValueError, match="no step named"):
        P.apply_overrides(c, ["nosuch.x=1"])
    with pytest.raises(ValueError, match="step.param=value"):
        P.apply_overrides(c, ["bbox=1"])
