"""Sphinx configuration for the PyGeoVision documentation (built by Read the Docs; build locally with ``make -C docs html``)."""
from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

project = "PyGeoVision"
author = "PyGeoVision Contributors"
copyright = "PyGeoVision Contributors, MIT licence"  # noqa: A001 - Sphinx's own variable name
try:
    release = importlib.metadata.version("pygeovision")
except importlib.metadata.PackageNotFoundError:
    import re
    release = re.search(r'"([^"]+)"', (ROOT / "pygeovision" / "_version.py").read_text(encoding="utf-8")).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_generated_src", "README.md"]
templates_path = ["_templates"]

# --- Markdown (MyST) -----------------------------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "attrs_inline", "tasklist", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"version": release}
suppress_warnings = ["myst.header"]

# --- API reference -------------------------------------------------------------------------------------------------
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True, "member-order": "bysource"}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
# Heavy optional dependencies are not needed to render documentation; importing a module must not require a GPU stack.
autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio", "timm", "segmentation_models_pytorch", "transformers", "huggingface_hub",
    "einops", "albumentations", "optuna", "mlflow", "wandb", "onnx", "onnxruntime", "onnxsim", "open_clip", "faiss",
    "hdbscan", "umap", "shap", "captum", "laspy", "s2sphere", "boto3", "sagemaker", "azure", "google", "statsmodels",
    "fastapi", "uvicorn", "websockets", "cv2", "ultralytics", "segment_anything", "sam2", "pygeofetch", "folium",
    "plotly", "seaborn", "rioxarray", "xarray", "passlib", "jose", "cryptography", "sqlalchemy",
]
nitpicky = False

# --- HTML ----------------------------------------------------------------------------------------------------------
html_theme = "furo"
html_title = f"PyGeoVision {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/pygeovision/pygeovision/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "light_css_variables": {"color-brand-primary": "#1b4965", "color-brand-content": "#1b4965"},
    "dark_css_variables": {"color-brand-primary": "#62b6cb", "color-brand-content": "#62b6cb"},
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# On Read the Docs the build must fail on a broken page, not publish it.
if os.environ.get("READTHEDOCS"):
    nitpicky = False
