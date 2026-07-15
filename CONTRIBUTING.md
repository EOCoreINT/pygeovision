# Contributing to PyGeoVision

Thank you for contributing to PyGeoVision — the world's most complete geospatial AI platform. This guide covers every pathway for contributing code, models, datasets, pipelines, bug fixes, and documentation.

---

## Quick Start

```bash
git clone https://github.com/appiahkubis14/pygeovision
cd pygeovision
pip install -e ".[dev,geo,train]"
pre-commit install
pytest tests/ -q                    # 580 tests, <3 min
```

---

## Ways to Contribute

### 🐛 Bug Reports

Open a GitHub Issue with:
- `pygeovision status` output
- Python + OS version
- Minimal reproducible example
- Full traceback

For data validation / model inference bugs, include:
- Input raster metadata (`client.preprocess.info("scene.tif")`)
- Validation report (`client.validator.generate_report("scene.tif", "report.html")`)

### 🔧 Bug Fixes

```bash
git checkout -b fix/describe-the-bug
# Write a failing test first
pytest tests/test_<module>.py::test_your_regression -v
# Fix the bug, then:
pytest tests/ -q
```

### ✨ New Features

Open a GitHub Discussion before starting large features. Small improvements (a new spectral index, a CLI alias) can go straight to a PR.

---

## Architecture Overview

```
PyGeoVision (standalone core)
├── data/
│   ├── fetch.py          SatelliteFetcher — PyGeoFetch Python API wrapper
│   ├── pgf_bridge.py     PyGeoFetchBridge — definitive integration layer
│   ├── validator.py      DataValidator — mandatory before every model
│   ├── indices.py        SpectralIndices — 22 validated indices
│   └── postprocess.py    PostProcessor — 15 prediction operations
├── preprocess/
│   └── core.py           Preprocessor — stack/clip/mask/normalise/resample
├── models/               119 architectures (own registry)
│   ├── foundation/
│   │   ├── dinov3.py     12 DINOv3 variants (SAT-493M + Web)
│   │   └── prithvi.py    Prithvi-EO-2.0 (600M)
│   └── change_detection/changeformer.py
├── ai/
│   ├── models/           Model zoo + registry
│   ├── pipelines/        51 E2E pipelines + orchestrator
│   └── geoai/            GeoAI optional plugin wrapper
├── training/             GeoTrainer + losses + callbacks
├── inference/            TiledInference + EnsembleInference
├── serving/              FastAPI + WebSocket
├── edge/                 ONNX + Jetson TensorRT
├── cloud/                AWS + Azure + GCP deploy
├── monitoring/           DriftDetector + alerts
├── advanced/             Few-shot + AutoML + VLM + timeseries + 3D
└── cli/main.py           15 CLI command groups
```

**Key invariant:** Every dataset MUST pass through `DataValidator` before any model touches it. The `prepare_for_ai()` method on the client enforces this.

---

## Adding a Spectral Index

Edit `pygeovision/data/indices.py`:

```python
def my_index(
    self,
    source: Union[str, np.ndarray],
    band_a: int = 3,
    band_b: int = 4,
    output_path: Optional[str] = None,
) -> Union[str, np.ndarray]:
    """My Index (Author Year).

    MY_INDEX = (Band_A - Band_B) / (Band_A + Band_B)

    Range: [-1, 1]. Values > 0.X indicate Y.
    """
    a, b, ref = self._get_two_bands(source, band_a, band_b)
    result    = (a - b) / (a + b + _EPS)
    result    = np.clip(_validate_arr(result, "MY_INDEX"), -1.0, 1.0)
    return self._out(result, ref, output_path, "my_index")
```

Then add to `compute_all()` `_MAP` dict and register in the CLI `indices_compute` command.

Tests go in `tests/test_data_layer.py`.

---

## Adding a Model Architecture

Add a spec to `pygeovision/ai/models/zoo.py`:

```python
ModelSpec(
    name             = "my_arch_b2",
    task             = "segmentation",
    architecture     = "MyArch-B2",
    backbone         = "my_backbone_medium",
    pretrained_available = True,
    hf_model_id      = "username/my-arch-b2-geo",
    params_m         = 27.5,
    description      = "My architecture for satellite segmentation",
    tags             = ["transformer", "remote_sensing"],
)
```

If the model needs a custom `nn.Module`, add it to `pygeovision/models/` following the existing patterns in `changeformer.py` or `prithvi.py`.

---

## Adding an E2E Pipeline

Add to `pygeovision/ai/pipelines/domains.py`:

```python
class MyDomainPipeline(BasePipeline):
    name        = "my_domain_pipeline"
    description = "Description of what it does end-to-end"
    sensors     = ["sentinel2", "landsat"]
    tags        = ["domain", "change_detection"]

    def run(self, bbox, date, output_dir="./results", **kw):
        import pathlib
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Search via PyGeoFetch (optional — works with local data too)
        results = self.client.search(bbox=bbox, date_range=..., limit=3)

        # 2. Download + prepare for AI
        dl    = self.client.download(results[:1], output_dir)
        ready = self.client.prepare_for_ai(
            dl[0].path, bbox=bbox, normalise="scale_factor",
            model_type="segmentation",
        )

        # 3. Run inference with own model
        from pygeovision.models import get_model
        from pygeovision.inference.tiled import TiledInference
        model  = get_model("segformer-b2", num_classes=2, pretrained=True)
        inf    = TiledInference(model)
        result = inf.infer(ready["output_path"],
                            str(pathlib.Path(output_dir) / "prediction.tif"))

        # 4. Postprocess
        vec = self.client.postprocess.vectorise(
            result["output_path"], str(pathlib.Path(output_dir) / "output.geojson"))

        return {"output_path": vec, "pipeline": self.name}


# Register it
_PIPELINE_REGISTRY["my_domain_pipeline"] = MyDomainPipeline
```

For simple pipelines that use the standard `validate → preprocess → model → postprocess` chain, use `_make_simple()`:

```python
"my_simple_pipeline": _make_simple(
    "my_simple_pipeline",
    "Short description of what it does",
    ["sentinel2"],
    ["domain", "tag"],
),
```

---

## Adding a Dataset

Add to `pygeovision/datasets/registry.py`:

```python
DatasetInfo(
    name         = "MyDataset",
    domain       = "urban",
    year         = 2024,
    n_samples    = 10000,
    sample_size  = "512×512",
    n_classes    = 5,
    modality     = "multispectral",
    resolution_m = 10.0,
    volume_gb    = 15.0,
    tasks        = ["segmentation"],
    description  = "Short description of the dataset and its use case",
    download_url = "https://example.com/dataset",
    paper_url    = "https://arxiv.org/abs/xxxx.xxxxx",
)
```

---

## Preprocessing Operations

To add a new preprocessing operation to `pygeovision/preprocess/core.py`:

```python
def my_operation(
    self,
    input_path: str,
    output_path: Optional[str] = None,
    **kwargs,
) -> str:
    """My preprocessing operation.

    Args:
        input_path: Source GeoTIFF.
        output_path: Destination path. Defaults to ``"_myop"``.

    Returns:
        Output path string.
    """
    rasterio = _require_rasterio()
    output_path = _safe_output(input_path, output_path, "_myop")

    with rasterio.open(input_path) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile.copy()

    # ... operation ...

    profile.update(compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

    logger.info("my_operation → %s", output_path)
    return output_path
```

Then expose it via the bridge in `pgf_bridge.py` and add a CLI subcommand in `cli/main.py`.

---

## Code Standards

| Tool | Command | Standard |
|---|---|---|
| Format | `black pygeovision/ tests/` | Line length 100 |
| Lint | `ruff check pygeovision/ tests/ --fix` | PEP 8 + extras |
| Type | `mypy pygeovision/ --ignore-missing-imports` | Typed public APIs |
| Test | `pytest tests/ -q --cov=pygeovision` | ≥95% coverage target |
| Docs | Google-style docstrings | All public functions |

---

## PR Checklist

- [ ] `pytest tests/ -q` — all 580 tests pass
- [ ] `ruff check . && black --check .` — clean
- [ ] New public functions have Google-style docstrings
- [ ] New features have tests in `tests/`
- [ ] `CHANGELOG.md` — entry in `[Unreleased]` section
- [ ] No secrets, credentials, or large binary files committed
- [ ] Breaking changes documented and discussed in issue first

---

## Release Process

1. Bump version in `pygeovision/_version.py`
2. Update `CHANGELOG.md` — move `[Unreleased]` to `[X.Y.Z] — YYYY-MM-DD`
3. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. CI runs tests → publishes to PyPI automatically
5. Update `CITATION.cff` version and date

---

## Community

- **GitHub Discussions** — Feature requests, questions, showcase your work
- **Issues** — Bug reports and specific problems
- **PRs** — All code contributions welcome

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.
