# Installation

## Requirements

- Python ≥ 3.10 (confirmed from `pyproject.toml`'s `requires-python`)
- pip ≥ 23.0 recommended

---

## Quick Install

```bash
pip install pygeovision
```

This installs the core package and its always-required dependencies:
`pygeofetch>=2.4.0` (the real, independent data layer package),
`pystac`/`pystac-client`/`planetary-computer` (STAC fallback),
`pydantic`, `click`, `pyyaml`, `httpx`, `tenacity`, `numpy`, `rich`,
`shapely`, `pyproj`.

It does **not** install PyTorch, rasterio, or any of the heavier,
task-specific dependencies below — pick the extras you actually need.

---

## Install Extras

```bash
# Recommended minimal install for most AI workflows
pip install "pygeovision[geo,train]"

# Everything except cloud/edge (install those explicitly if needed)
pip install "pygeovision[all]"

# Individual extras
pip install "pygeovision[geo]"          # raster/vector I/O
pip install "pygeovision[viz]"          # visualization
pip install "pygeovision[train]"        # PyTorch training stack
pip install "pygeovision[serve]"        # FastAPI serving + ONNX runtime
pip install "pygeovision[labeling]"     # auto-labeling sources
pip install "pygeovision[vlm]"          # vision-language models
pip install "pygeovision[xai]"          # explainability
pip install "pygeovision[foundation]"   # DINOv3, Prithvi-EO-2.0
pip install "pygeovision[advanced]"     # few-shot, AutoML, timeseries
pip install "pygeovision[cloud]"        # AWS/Azure/GCP deployment
pip install "pygeovision[edge]"         # ONNX Runtime for edge inference
pip install "pygeovision[geo3d]"        # point cloud / LiDAR
pip install "pygeovision[monitoring]"   # drift detection
pip install "pygeovision[enterprise]"   # auth, audit, compliance
pip install "pygeovision[dev]"          # pytest, ruff, mypy, black
```

---

## Extras Reference

The table below lists the **real, exact packages** from each extra in
`pyproject.toml` — not a paraphrase. If you need a specific package,
check here first rather than guessing which extra provides it.

| Extra | Real packages (from `pyproject.toml`) |
|---|---|
| `geo` | `rasterio`, `geopandas`, `rioxarray`, `pyogrio`, `shapely`, `pyproj` |
| `viz` | `matplotlib`, `folium`, `plotly`, `seaborn`, `scikit-learn` |
| `train` | `torch`, `torchvision`, `timm`, `segmentation-models-pytorch`, `albumentations`, `optuna`, `mlflow`, `wandb` |
| `serve` | `fastapi`, `uvicorn[standard]`, `onnxruntime`, `onnxsim`, `websockets` |
| `labeling` | `requests`, `laspy`, `s2sphere`, `scipy`, `Pillow`, `faiss-cpu` |
| `vlm` | `transformers`, `torch`, `Pillow`, `open-clip-torch`, `faiss-cpu` |
| `xai` | `shap`, `captum`, `torch`, `matplotlib` |
| `foundation` | `transformers`, `torch`, `timm`, `hdbscan`, `scikit-learn`, `umap-learn`, `faiss-cpu` |
| `advanced` | `transformers`, `torch`, `optuna`, `scikit-learn`, `statsmodels` |
| `cloud` | `boto3`, `sagemaker`, `azure-ai-ml`, `azure-identity`, `google-cloud-aiplatform`, `google-cloud-storage` |
| `edge` | `onnxruntime`, `onnxsim` |
| `geo3d` | `laspy`, `scipy`, `pandas` |
| `monitoring` | `scipy`, `matplotlib`, `requests` |
| `enterprise` | `cryptography`, `passlib[bcrypt]`, `python-jose[cryptography]`, `pydantic`, `sqlalchemy` |
| `dev` | `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `mypy`, `black`, `pre-commit`, `types-pyyaml`, `types-requests` |
| `all` | `geo,viz,train,serve,labeling,vlm,xai,foundation,advanced,monitoring,geo3d,enterprise` combined — deliberately **excludes** `cloud` and `edge`; install those explicitly |
| `minimal` | `geo,train` combined |

There is no `timeseries` extra — time-series functionality (`pygeovision.advanced.timeseries`) is covered by the `advanced` extra, not a dedicated one, despite what an earlier version of this page claimed.

---

## System Dependencies

`rasterio`/`geopandas` (pulled in by the `geo` extra) depend on GDAL.
Most platforms get a working GDAL via prebuilt wheels automatically,
but if you hit a build error:

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
    gdal-bin libgdal-dev libproj-dev libgeos-dev \
    python3-dev build-essential
```

**macOS (Homebrew):**
```bash
brew install gdal proj geos
```

---

## GPU Support

PyTorch (via the `train`/`foundation`/`vlm`/`xai` extras) will use
CUDA or Apple Silicon (MPS) automatically if available, falling back
to CPU otherwise. For a specific CUDA version, install PyTorch first
with the matching index, then install PyGeoVision:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "pygeovision[train,foundation]"
```

---

## Verify Installation

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
print(client)
# PyGeoVision(v2.1.8 | pygeofetch=✓ | pgf_v2=✓v2 | ai=✓torch | datasets=503 |
#             models=98 | pipelines=51 | ...)
```

Note: the `models=98` in this repr reports `pygeovision.ai.models.zoo.model_zoo`'s count specifically, a different, separate registry from `pygeovision.models.registry.model_registry` (121 names) — see [Architecture](architecture.md#a-real-confirmed-pattern-duplicate-parallel-implementations) for why there are three overlapping model catalogs in this codebase.

If this prints without error, the core package and `pygeofetch` are
correctly installed. `ai=✓torch` requires the `train` extra (or any
extra that pulls in `torch`) — without it, this shows a different,
honest status rather than pretending PyTorch is available.

```bash
# Confirm the CLI is on PATH
pygeovision --help
pygeovision doctor    # real environment/dependency diagnostic command
```
