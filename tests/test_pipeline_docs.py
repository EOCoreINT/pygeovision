"""The documentation must describe the pipelines that actually exist."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from pygeovision.ai.pipelines import list_pipelines
from pygeovision.ai.pipelines.removed import REMOVED_PIPELINES

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ["docs/pipelines/ready.md", "docs/pipelines/needs-weights.md",
             "docs/pipelines/removed.md", "docs/reference/pipeline-catalog.md"]

pytestmark = pytest.mark.skipif(not (ROOT / "docs").exists(), reason="docs not in this checkout")


def test_generated_pages_are_current(tmp_path):
    before = {p: (ROOT / p).read_text() for p in GENERATED}
    subprocess.run([sys.executable, str(ROOT / "tools" / "generate_pipeline_docs.py")], check=True, cwd=ROOT)
    for p in GENERATED:
        assert (ROOT / p).read_text() == before[p], f"{p} is stale: run python tools/generate_pipeline_docs.py"


def test_every_pipeline_has_a_section():
    text = (ROOT / "docs/pipelines/ready.md").read_text() + (ROOT / "docs/pipelines/needs-weights.md").read_text()
    for name in list_pipelines():
        assert f"## `{name}`" in text


def test_readme_names_only_real_pipelines():
    readme = (ROOT / "README.md").read_text()
    section = readme[readme.index("## 🛰️ End-to-end pipelines"):readme.index("## 🧠 Models and training")]
    table_names = set(re.findall(r"`([a-z_]+)`", "\n".join(l for l in section.splitlines() if l.startswith("|"))))
    live = set(list_pipelines())
    assert not table_names & set(REMOVED_PIPELINES)
    assert live <= table_names, sorted(live - table_names)


def test_counts_in_docs_match_the_code():
    kinds = [c for c in __import__("pygeovision.ai.pipelines", fromlist=["x"]).pipeline_kinds().values()]
    total, ready = len(kinds), kinds.count("ready")
    readme = (ROOT / "README.md").read_text()
    assert f"Pipelines-{total}-" in readme
    assert f"**{total} pipelines**" in readme and f"**{ready} run out of the box" in readme
    assert f"{total} pipelines" in (ROOT / "docs/pipelines/index.md").read_text()
