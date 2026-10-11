"""The documentation must stay true to the code.

These tests do not build the Sphinx site (``make -C docs html`` does, with warnings as errors); they check the things a build
cannot: that generated pages are current, that hand-written reference tables name everything that exists, that example
commands name real pipelines, and that Markdown does not contain constructs that silently swallow the rest of a page.
"""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

pytestmark = pytest.mark.skipif(not DOCS.exists(), reason="docs not in this checkout")


def _md_files():
    return [p for p in DOCS.rglob("*.md") if "_build" not in p.parts and p.name != "README.md"]


def _run_check(script: str):
    return subprocess.run([sys.executable, str(ROOT / "tools" / script), "--check"], cwd=ROOT,
                          capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})


# --------------------------------------------------------------------------- generated pages
def test_cli_reference_is_current():
    r = _run_check("generate_reference_docs.py")
    assert r.returncode == 0, f"run: PYTHONPATH=. python tools/generate_reference_docs.py\n{r.stdout}{r.stderr}"


def test_pipeline_pages_are_current():
    r = _run_check("generate_pipeline_pages.py")
    assert r.returncode == 0, f"run: PYTHONPATH=. python tools/generate_pipeline_pages.py\n{r.stdout}{r.stderr}"


def test_every_cli_group_has_notes_and_a_page():
    from pygeovision.cli.main import cli

    notes = yaml.safe_load((DOCS / "_data" / "cli_notes.yaml").read_text())
    groups = set(cli.commands)
    assert set(notes) <= groups, f"notes for unknown groups: {sorted(set(notes) - groups)}"
    for g in groups:
        assert (DOCS / "reference" / "cli" / f"{g}.md").exists(), f"no CLI page for {g}"
    for g, n in notes.items():
        assert {"intro", "when", "expect"} <= set(n), f"cli_notes.yaml: '{g}' needs intro, when and expect"


# --------------------------------------------------------------------------- hand-written reference tables
def test_exceptions_page_lists_every_exception_class():
    import pygeovision.core.exceptions as ex

    text = (DOCS / "reference" / "exceptions.md").read_text()
    for name, cls in inspect.getmembers(ex, inspect.isclass):
        if issubclass(cls, ex.PyGeoVisionError):
            assert f"`{name}`" in text, f"docs/reference/exceptions.md does not mention {name}"


def test_exceptions_page_marks_raised_classes_correctly():
    """The 'raised today' column must match the source: a class marked reserved must not be raised, and vice versa."""
    import pygeovision.core.exceptions as ex

    sources = {p: p.read_text(errors="ignore") for p in (ROOT / "pygeovision").rglob("*.py")
               if p.name != "exceptions.py"}
    page = (DOCS / "reference" / "exceptions.md").read_text()
    rows: dict[str, str] = {}
    for m in re.finditer(r"^\| `(\w+)`.*$", page, re.M):
        rows.setdefault(m.group(1), m.group(0))      # the class table comes first; later tables repeat names
    for name, cls in inspect.getmembers(ex, inspect.isclass):
        if not issubclass(cls, ex.PyGeoVisionError) or name not in rows:
            continue
        raised = any(re.search(rf"raise\s+(\w+\.)?{name}\b", t) for t in sources.values())
        cells = [c.strip() for c in rows[name].split("|")]
        status = cells[3] if len(cells) > 3 else ""
        if status == "base":
            continue
        if raised:
            assert "**yes**" in status, f"{name} is raised in the source but not marked '**yes**' on the page"
        else:
            assert "**yes**" not in status, f"{name} is marked raised but nothing raises it"


def test_config_page_lists_every_setting():
    from pygeovision.core.config import PyGeoVisionConfig

    text = (DOCS / "reference" / "config.md").read_text()
    for section, field in PyGeoVisionConfig.model_fields.items():
        ann = field.annotation
        if inspect.isclass(ann) and hasattr(ann, "model_fields"):
            for key in ann.model_fields:
                assert f"`{section}.{key}`" in text, f"docs/reference/config.md misses {section}.{key}"
        else:
            assert f"`{section}`" in text, f"docs/reference/config.md misses {section}"


def test_config_page_names_every_environment_variable():
    src = (ROOT / "pygeovision" / "core" / "config.py").read_text()
    text = (DOCS / "reference" / "config.md").read_text()
    for var in re.findall(r'"(PYGEOVISION_[A-Z_]+)"', src):
        assert var in text, f"{var} is read by config.py but not documented in docs/reference/config.md"


# --------------------------------------------------------------------------- examples name real things
def test_example_pages_name_only_real_pipelines():
    from pygeovision.ai.pipelines import list_pipelines
    from pygeovision.ai.pipelines.removed import REMOVED_PIPELINES

    live = set(list_pipelines())
    for page in (DOCS / "examples").glob("*.md"):
        for name in re.findall(r"pygeovision channel (\w+)", page.read_text()):
            if name == "list":
                continue
            assert name in live, f"{page.name}: 'pygeovision channel {name}' is not a pipeline"
            assert name not in REMOVED_PIPELINES


def test_example_commands_give_a_checkpoint_to_pipelines_that_need_one():
    from pygeovision.ai.pipelines import pipeline_kinds

    needs = {n for n, k in pipeline_kinds().items() if k != "ready"}
    for page in list(DOCS.rglob("*.md")):
        if "_build" in page.parts:
            continue
        text = page.read_text()
        for m in re.finditer(r"pygeovision channel (\w+)((?:[^\n`]|\\\n)*)", text):
            if m.group(1) in needs:
                block = text[m.start(): m.start() + 600].split("```")[0]
                assert "--checkpoint" in block, f"{page.relative_to(DOCS)}: '{m.group(1)}' needs --checkpoint"


def test_example_commands_use_real_pipeline_parameters():
    """`--param KEY=VALUE` in the example pages must be an argument the pipeline's run() accepts."""
    from pygeovision.ai.pipelines import pipeline_catalogue

    cat = pipeline_catalogue()
    for page in (DOCS / "examples").glob("*.md"):
        text = page.read_text()
        for m in re.finditer(r"pygeovision channel (\w+)", text):
            name = m.group(1)
            # the command ends at the first line that does not continue with a backslash
            rest = ""
            for line in text[m.end():].split("\n"):
                rest += " " + line
                if not line.rstrip().endswith("\\"):
                    break
            cls = cat.get(name)
            if cls is None:
                continue
            params = inspect.signature(cls.run).parameters
            for key in re.findall(r"--param\s+([A-Za-z_]+)=", rest):
                documented = (DOCS / "pipelines" / "catalog" / f"{name}.md").read_text()
                assert key in params or f"| `{key}` |" in documented, \
                    f"{page.name}: --param {key} is not a parameter of {name} (see its page for the argument table)"


# --------------------------------------------------------------------------- markdown hygiene
def test_no_fence_opener_with_backticks_in_its_info_string():
    """A line like ```{admonition} Title with `code` is not a fence; the page renders wrong and no warning is given."""
    bad = []
    for p in _md_files():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.match(r"^```\{[a-z-]+\}.*`", line):
                bad.append(f"{p.relative_to(DOCS)}:{i}")
    assert not bad, "use ':::' fences when a directive title contains backticks: " + ", ".join(bad)


def test_code_fences_are_balanced():
    bad = []
    for p in _md_files():
        depth = None
        for line in p.read_text().splitlines():
            m = re.match(r"^(\s*)(`{3,}|~{3,})", line)
            if not m:
                continue
            fence = m.group(2)
            if depth is None:
                depth = fence
            elif fence[0] == depth[0] and len(fence) >= len(depth) and line.strip() == fence:
                depth = None
        if depth is not None:
            bad.append(str(p.relative_to(DOCS)))
    assert not bad, f"unclosed code fence in: {bad}"


def test_relative_links_point_at_existing_files():
    bad = []
    for p in _md_files():
        text = re.sub(r"```.*?```", "", p.read_text(), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+?\.md)(?:#[^)]*)?\)", text):
            if target.startswith(("http://", "https://")):
                continue
            if not (p.parent / target).resolve().exists():
                bad.append(f"{p.relative_to(DOCS)} -> {target}")
    assert not bad, "dead links:\n" + "\n".join(bad)


def test_counts_in_the_home_page_match_the_code():
    from pygeovision.ai.pipelines import pipeline_kinds
    from pygeovision.cli.main import cli

    home = (DOCS / "index.md").read_text()
    kinds = pipeline_kinds()
    ready = sum(1 for k in kinds.values() if k == "ready")
    assert f"**{len(kinds)} pipelines**" in home
    assert f"{ready} run out of the box" in home
    assert f"{len(kinds) - ready} run *your*" in home
    n_groups = len(cli.commands)
    def leaves(cmd):
        return sum(leaves(c) for c in cmd.commands.values()) if hasattr(cmd, "commands") else 1

    n_cmds = sum(leaves(g) for g in cli.commands.values())
    assert f"**{n_groups} CLI groups, {n_cmds} commands**" in home
