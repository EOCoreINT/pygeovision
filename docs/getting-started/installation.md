# Installation

PyGeoVision is a pure-Python package. The core install is small; the heavier capabilities (deep learning,
reports, foundation models, cloud deployment) are optional **extras** so you only install what you use.

## Requirements

| Requirement | Detail |
|---|---|
| Python | 3.10 or newer (the package is tested on 3.13) |
| Operating system | Linux, macOS or Windows. Everything except GPU training runs on a laptop |
| Network | Needed to *search and download* imagery and to read the published reference products. Not needed to run on files you already have, to run the unit tests, or to run the offline smoke test |
| GPU | Optional. Used automatically by training and inference when PyTorch sees one (`gpu.device: auto`). Pipelines that compute indices, terrain products and reference maps never need one |
| Disk | Scenes are cached under the output directory you choose; a Sentinel-2 band of one tile is about 65 MB |

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pygeovision            # the core: data layer, indices, CLI
```

The core install brings in [`pygeofetch`](https://pygeofetch.readthedocs.io/) (search, download, authentication),
`pystac-client`, `planetary-computer`, `pydantic`, `click`, `numpy`, `shapely`, `pyproj` and `rich`.

Most users want one of these bundles:

```bash
pip install "pygeovision[minimal]"     # raster/vector stack + PyTorch training + deliverables (map, PDF, Word, Excel)
pip install "pygeovision[ai]"          # deep learning: training, foundation models, labelling, vision-language models
pip install "pygeovision[all]"         # everything except cloud and edge deployment
```

### The extras

| Extra | What it adds | Install it when you want to… |
|---|---|---|
| `geo` | `rasterio`, `geopandas`, `rioxarray`, `pyogrio` | read and write rasters and vectors (**every pipeline needs this**) |
| `report` | `matplotlib`, `reportlab`, `python-docx`, `openpyxl`, `pillow`, `scipy` and the geo stack | get the map, the charts and the PDF / Word / Excel reports with every pipeline run |
| `obia` | `scikit-image`, `scikit-learn`, `pandas`, `scipy`, `matplotlib` | use [object-based image analysis](../obia/index.md) |
| `viz` | `matplotlib`, `folium`, `plotly`, `seaborn` | draw maps and plots yourself |
| `train` | `torch`, `torchvision`, `timm`, `segmentation-models-pytorch`, `albumentations`, `optuna`, `mlflow`, `wandb` | [train or fine-tune](../training/index.md) a network |
| `foundation` | `transformers`, `timm`, `einops`, `huggingface-hub`, `hdbscan`, `umap-learn`, `faiss-cpu` | use Prithvi-EO, DINOv3, embeddings and clustering |
| `labeling` | `transformers`, `torch`, `requests`, `laspy`, `s2sphere`, `scipy`, `Pillow`, `faiss-cpu` | generate [training labels](../labeling/index.md) (SAM, OSM, building footprints, WorldCover) |
| `vlm` | `transformers`, `open-clip-torch`, `Pillow`, `faiss-cpu` | caption, query and search imagery with CLIP / RemoteCLIP / Moondream |
| `xai` | `shap`, `captum`, `matplotlib` | [explain predictions](../explainability/index.md) |
| `advanced` | `transformers`, `optuna`, `scikit-learn`, `statsmodels` | few-shot learning, AutoML, multi-task, time-series modelling |
| `monitoring` | `scipy`, `matplotlib`, `requests` | [drift detection](../monitoring/index.md) |
| `geo3d` | `laspy`, `scipy`, `pandas` | LiDAR / point-cloud products |
| `serve` | `fastapi`, `uvicorn`, `onnxruntime`, `onnxsim`, `websockets` | serve models |
| `edge` | `onnx`, `onnxruntime`, `onnxsim` | [export to ONNX](../deployment/edge.md) |
| `cloud` | `boto3`, `sagemaker`, Azure ML, Vertex AI | [deploy to a cloud endpoint](../deployment/cloud.md) |
| `enterprise` | `cryptography`, `passlib`, `python-jose`, `sqlalchemy` | authentication and audit components |
| `docs` | Sphinx, MyST, Furo | build this documentation |
| `dev` | `pytest`, `ruff`, `mypy`, `black`, `pre-commit` | develop PyGeoVision itself |

```{tip}
Install the geo stack and the report extra first (`pip install "pygeovision[geo,report]"`). That is all that is
needed to run all 34 *ready* pipelines and to receive their maps and reports. Add `train` only when you have a
network of your own to train or run.
```

### GPU builds of PyTorch

`pip install "pygeovision[train]"` installs the default PyTorch wheel for your platform. For a specific CUDA
build, install PyTorch first from <https://pytorch.org/get-started/locally/> and then install the extra; pip keeps
the version you already have.

## Install from source (development)

```bash
git clone https://github.com/pygeovision/pygeovision
cd pygeovision
pip install -e ".[all,dev]"
pytest                          # the unit tests need no network, GPU or pretrained weights
```

An *editable* install means `import pygeovision` and the `pygeovision` command always use the files in your
checkout; there is nothing to reinstall after you edit or copy a file.

## Verify the installation

```bash
pygeovision --version
```

```text
pygeovision, version 2.1.8
```

```bash
pygeovision doctor
```

```text
══════════════════════════════════════════
         PyGeoVision Diagnostics
══════════════════════════════════════════
  ✓  pygeovision
     python            : 3.13.16
     platform          : Linux
  ✓  pygeofetch          v2.6.3.0.4
  ✓  torch               v2.14.1+cu130
  ✓  rasterio            v1.5.2
  ✓  geopandas           v1.2.0
  ✓  credentials         0 provider(s) with stored credentials
  ✓  cache               1 entries

  → 7/7 components healthy
```

Every line should carry a tick. A cross names the component to install or repair (for example a missing `torch`
means the `train` extra is not installed; that is fine if you only run index pipelines).

```bash
pygeovision status
```

```text
PyGeoVision v2.1.8 | Python 3.13.16

pygeofetch  ✓  v2.6.3.0.4 | 22 providers | 10 open-access
torch       ✓  v2.14.1+cu130 | device=cpu
rasterio    ✓  1.5.2
geopandas   ✓  1.2.0

Registered AI models: 14
```

`status` shows the compute device PyTorch will use (`cuda`, `mps` or `cpu`) and how many providers and models are
registered. The same information is available from Python:

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
print(client.status())          # a dict: versions, device, provider count, registered models
print(client.doctor())          # diagnostics of the data layer
```

## Try it without a network

The repository ships an offline test that runs **all 41 pipelines** through the real command-line interface on
small synthetic rasters, and checks that each one writes its rasters, a map, vector files and the PDF, Word and
Excel reports:

```bash
python tools/offline_pipeline_smoke.py
```

```text
OK  active_fire                3 reports, 2 map(s), 3 vector file(s)
…
41/41 pipelines succeeded
```

It is the quickest way to confirm that your environment can produce every deliverable before you spend time
downloading real imagery.

## Next steps

- [Quick start](quickstart.md): your first real run.
- [Concepts](concepts.md): bounding boxes, dates, grids, `nodata`, pipeline results.
- [Configuration](configuration.md): config files, environment variables, credentials.
- [Troubleshooting](troubleshooting.md): what the common error messages mean.
