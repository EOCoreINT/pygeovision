"""Sphinx configuration for the PyGeoVision documentation.

Build locally with ``make -C docs html`` (English plus every translation, see ``tools/build_all.sh``).
Set ``DOCS_SKIP_OTHER_LANGS=1`` for a fast English-only build while writing.
"""
from __future__ import annotations

import importlib.metadata
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

project = "PyGeoVision"
author = "PyGeoVision Contributors"
copyright = "PyGeoVision Contributors, MIT licence"  # noqa: A001 - Sphinx's own variable name
try:
    release = importlib.metadata.version("pygeovision")
except importlib.metadata.PackageNotFoundError:
    release = re.search(r'"([^"]+)"', (ROOT / "pygeovision" / "_version.py").read_text(encoding="utf-8")).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_reredirects",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_generated_src", "README.md", "locale", "tools"]
templates_path = ["_templates"]

# --- Markdown (MyST) -----------------------------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "attrs_inline", "tasklist", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"version": release}
suppress_warnings = ["myst.header"]

# --- HTML ----------------------------------------------------------------------------------------------------------
html_theme = "furo"
html_title = "PyGeoVision"
html_theme_options = {
    "source_repository": "https://github.com/pygeovision/pygeovision/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    # The top navigation bar (templates/page.html) is rendered through Furo's announcement slot, so it sits above
    # the whole page. Furo only emits that slot when an announcement is set; the text itself is never shown.
    "announcement": "x",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["pgv.js"]
html_favicon = "_static/img/logo.svg"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# --- Pages that moved when the documentation was reorganised -----------------------------------------------------------
# Old URLs keep working: each old document name redirects to its new location.
_MOVED = {
    # old document name -> where its content lives now (document names, not URLs)
    "contributing": "project/contributing",
    "reference/changelog": "project/changelog",
    "reference/roadmap": "project/roadmap",
    "advanced/index": "guides/python",
    "advanced/automl": "experiments/automl",
    "advanced/multitask": "training/multitask",
    "advanced/few-shot": "classification/few-shot",
    "advanced/timeseries": "timeseries/index",
    "advanced/pointcloud": "lidar/index",
    "advanced/vlm": "vision-language/index",
    "benchmark/index": "experiments/benchmark",
    "edge-cloud/index": "deployment/index",
    "edge-cloud/edge": "deployment/edge",
    "edge-cloud/cloud": "deployment/cloud",
    "cli/direct-inference": "inference/cli",
    "models/foundation": "foundation-models/index",
    "training/advanced": "training/fine-tuning",
    "inference/detection": "detection/predict",
    # the documentation was rewritten for users of the package: pages about the code were removed
    "architecture": "getting-started/concepts",
    "reference/python-api": "guides/python",
    "reference/testing": "project/contributing",
    "inference/engine": "inference/index",
}
# The generated Python API reference (one page per package) was removed; its old addresses lead to the Python guide.
for _name in ("advanced", "benchmark", "builders-3d", "builders-change-detection", "builders-classification", "builders-detection",
              "builders-foundation", "builders-segmentation", "builders-vlm", "client", "core", "data-access", "data-prepare",
              "datasets", "deployment", "engine", "explainability", "inference", "labeling", "losses", "models", "monitoring",
              "obia", "pipelines", "reporting", "training", "workflows"):
    _MOVED[f"reference/api/{_name}"] = "guides/python"


def _relative_url(old: str, new: str) -> str:
    """A redirect target relative to the old page, as sphinx_reredirects expects (``../experiments/automl.html``)."""
    return os.path.relpath(new, os.path.dirname(old) or ".").replace(os.sep, "/") + ".html"


redirects = {old: _relative_url(old, new) for old, new in _MOVED.items()}

# --- Internationalisation ------------------------------------------------------------------------------------------
# Translations live in locale/<lang>/LC_MESSAGES/*.po (one file per page, plus chrome.po for the navigation bar).
# Text that is not translated falls back to English, so every language build is complete; pages without a
# translation show a "partly translated" note. Build everything with tools/build_all.sh.
language = os.environ.get("DOCS_LANG", "en")
locale_dirs = ["locale/"]
gettext_compact = False
gettext_uuid = False
LANGUAGES = [
    ("en", "English"), ("fr", "Français"), ("es", "Español"), ("pt", "Português"),
    ("de", "Deutsch"), ("zh_CN", "简体中文"), ("ar", "العربية"),
]
html_context = {"languages": LANGUAGES, "current_language": language,
                "pgv_rtl": language in ("ar",), "pgv_partial": False,
                "repo_url": "https://github.com/pygeovision/pygeovision",
                "pypi_url": "https://pypi.org/project/pygeovision/"}


def _load_chrome(lang: str) -> dict:
    """Navigation-bar strings from locale/<lang>/LC_MESSAGES/chrome.po (empty for English or when polib is missing)."""
    try:
        import polib
        po = polib.pofile(str(HERE / "locale" / lang / "LC_MESSAGES" / "chrome.po"))
        return {e.msgid: e.msgstr for e in po if e.msgstr}
    except Exception:  # noqa: BLE001 - a missing translation must never break the English build
        return {}


_chrome = _load_chrome(language)


def setup(app):  # noqa: D103
    def _ctx(app, pagename, templatename, context, doctree):
        context["pgv_t"] = lambda s: _chrome.get(s, s)
        if language != "en":
            mo = HERE / "locale" / language / "LC_MESSAGES" / (pagename + ".mo")
            po = HERE / "locale" / language / "LC_MESSAGES" / (pagename + ".po")
            context["pgv_partial"] = not (mo.exists() or po.exists())

    app.connect("html-page-context", _ctx)

    # A plain ``sphinx-build`` (as Read the Docs runs it) produces English at the site root and then builds every
    # other language into <outdir>/<lang>/, so the language switcher works without custom CI.
    # DOCS_SUBBUILD is set for those inner builds so they do not recurse.
    def _all_langs(app, exc):
        import shutil
        import subprocess
        import tempfile
        import time

        from sphinx.util import logging as sphinx_logging

        if exc or app.builder.name != "html" or language != "en":
            return
        if os.environ.get("DOCS_SUBBUILD") or os.environ.get("DOCS_SKIP_OTHER_LANGS"):
            return
        log = sphinx_logging.getLogger(__name__)
        out = Path(str(app.outdir))
        codes = [c for c, _label in LANGUAGES if c != "en" and (HERE / "locale" / c).is_dir()]
        jobs = max(1, int(os.environ.get("DOCS_JOBS", min(2, os.cpu_count() or 1))))
        pending, running, results = list(codes), {}, {}
        with tempfile.TemporaryDirectory() as scratch:
            while pending or running:
                while pending and len(running) < jobs:
                    code = pending.pop(0)
                    env = dict(os.environ, DOCS_LANG=code, DOCS_SUBBUILD="1")
                    logfile = open(Path(scratch) / (code + ".log"), "w+")  # noqa: SIM115 - closed when the build ends
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "sphinx", "-b", "html", "-q", "-d", str(Path(scratch) / ("doctrees-" + code)),
                         "-D", "language=" + code, str(app.srcdir), str(out / code)],
                        env=env, stdout=logfile, stderr=subprocess.STDOUT)
                    running[code] = (proc, logfile)
                for code, (proc, logfile) in list(running.items()):
                    if proc.poll() is None:
                        continue
                    logfile.seek(0)
                    results[code] = (proc.returncode, logfile.read())
                    logfile.close()
                    del running[code]
                if running:
                    time.sleep(1)

        # A language build that stopped early leaves pages without the static files they load (pgv.js for the sidebar
        # accordion, the stylesheet, the logo), which breaks the whole page. Report it, and fill in whatever is missing
        # from the English build's copy so a language page never points at a file that is not there.
        sys.path.insert(0, str(HERE / "tools"))
        from langstatic import fill_missing_static

        for code in codes:
            rc, text = results[code]
            if rc:
                tail = "\n".join(text.strip().splitlines()[-15:])
                log.warning("the %s language build failed (exit code %s):\n%s", code, rc, tail)
            missing = fill_missing_static(out, code)
            if missing:
                log.warning("the %s build has no %s in _static", code, ", ".join(missing))

    app.connect("build-finished", _all_langs)


# --- i18n fix ------------------------------------------------------------------------------------------------------
# MyST overwrites a document's heading-slug table every time Sphinx re-parses a translated fragment, which breaks
# cross-page anchors in translated builds. Merge the tables instead.
try:
    from myst_parser.mdit_to_docutils.base import DocutilsRenderer as _Renderer

    _orig_finalise = _Renderer._render_finalise

    def _render_finalise(self):
        env = getattr(self, "sphinx_env", None)
        old = dict(env.metadata[env.docname].get("myst_slugs", {})) if env else {}
        _orig_finalise(self)
        if env and old:
            new = env.metadata[env.docname].get("myst_slugs", {})
            env.metadata[env.docname]["myst_slugs"] = {**new, **old}

    _Renderer._render_finalise = _render_finalise
except Exception:  # noqa: BLE001 - only matters for translated builds
    pass
