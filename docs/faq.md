# Frequently Asked Questions

Every answer below was checked against the real, current source before
being included here — several claims in an earlier version of this
page turned out to be false or stale, and are corrected inline below
rather than silently dropped.

---

## General

**Which Python versions are supported?**

Python 3.10, 3.11, and 3.12 (`pyproject.toml`'s real `requires-python`).

**Does it work on Windows?**

Yes, with one real caveat: GDAL/rasterio on Windows requires installing
from [Christoph Gohlke's wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
or using conda. Everything else runs natively.

**Is a GPU required?**

No. Every model supports real CPU inference (slow but functional). A
GPU is recommended for training. For large GeoTIFFs, two genuinely
separate `TiledInference` engines exist — see
[Tiled Inference](inference/tiled-inference.md) for which one your
code path actually uses; both have real, verified band-count
validation and chip-failure handling as of this project's audit.

---

## Data

**Which satellite providers are supported?**

This is real, but owned entirely by `pygeofetch`, not pygeovision —
see [pygeofetch's own provider list](https://pygeofetch.readthedocs.io/en/latest/core-features/providers.html)
for the real, current, authoritative answer rather than a copy here
that could go stale.

**How do I handle large GeoTIFFs (>1 GB)?**

```python
from pygeovision.inference.tiled import TiledInference

inf = TiledInference(model, chip_size=512, overlap=64, num_classes=5)
result = inf.infer("large_100km.tif", "output.tif")
```

See [Tiled Inference](inference/tiled-inference.md) for the real
band-validation and chip-failure-handling behavior, including a real
bug found and fixed this project where chip failures could be silently
zero-filled with no visible error.

---

## Models

**How do I add my own model?**

```python
from pygeovision.models.registry import ModelSpec, register_model

register_model(ModelSpec(
    name="my-custom-unet",
    task="segmentation",
    family="unet",
    params_m=12.5,
    description="Custom U-Net for wetland mapping",
))
```

**Can I use a model fine-tuned on my own data?**

```python
import torch
from pygeovision.models import get_model

model = get_model("segformer-b2", num_classes=5, pretrained=False)
model.load_state_dict(torch.load("my_checkpoint.pth"))
```

**How do I export a model for edge deployment?**

```bash
pygeovision edge export-onnx segformer-b2 --output model.onnx --classes 7
pygeovision edge benchmark-onnx model.onnx
```

```{note}
These are the two real, existing `edge` subcommands. An earlier
version of this FAQ referenced a `deploy-jetson` command and used
`pgv` as the CLI name — neither is real. The actual installed command
is `pygeovision` (confirmed directly against `pyproject.toml`'s
`[project.scripts]` entry).
```

---

## Foundation Models

**Is DINOv3 real in this codebase?**

Partially, and this is worth being direct about. This codebase's
`dinov3_*`-named entries were audited and found to load
`facebook/dinov2-*` weights — genuine, real DINOv2, not DINOv3 — under
a DINOv3 label. The "SAT-493M satellite-pretrained" variants had the
identical issue: the same generic, non-satellite DINOv2 weights
regardless of which `_sat` name was requested. All 12 mislabeled
variants were removed from the registry. The real module
(`pygeovision.models.foundation.dinov3`) still exists with real code
and real callers, and carries an explicit warning in its own docstring
about this. If you need this class of model, DINOv2 (correctly
labeled) is real and available; a genuine DINOv3 integration is not,
today.

```{warning}
An earlier version of this FAQ presented the SAT-493M claim as true,
including a specific-looking normalization-statistics table for it.
That table was describing weights this codebase never actually loads.
```

**What spectral bands does Prithvi expect?**

HLS (Harmonized Landsat Sentinel-2) band order: Blue, Green, Red, NIR,
SWIR1, SWIR2.

```python
from pygeovision.models.foundation.prithvi import map_bands

data_hls = map_bands(sentinel2_data, source="sentinel2", n_prithvi_bands=6)
```

**Can I fine-tune Prithvi on my own dataset?**

```python
from pygeovision.models.foundation.prithvi import finetune_prithvi

result = finetune_prithvi("prithvi_eo_2_0", task="land_cover", num_classes=10)
```

See [Finetuning](training/finetuning.md) — `Prithvi`'s real loading
path raises clearly on a genuine weight-load failure by default rather
than silently substituting an untrained model; `allow_random_init=True`
is a real, explicit opt-in for development/testing only.

---

## Training

**Can I resume training from a checkpoint?**

```python
from pygeovision.training.trainer import GeoTrainer, TrainingConfig

cfg = TrainingConfig(output_dir="./checkpoints", max_epochs=50)
```

For finetuning from an existing checkpoint specifically (a different,
real capability — loading pretrained weights into a new training run,
not resuming an interrupted one with optimizer state intact), see
[Finetuning](training/finetuning.md) — that capability lives on the
*other* real `GeoTrainer` (`pygeovision.ai.training.trainer`), not
this one. See [Architecture](architecture.md) for why there are two.

```{note}
An earlier version of this FAQ described a `pygeovision.training.
distributed.launch_ddp` distributed-training helper. Checked directly:
this function exists but is never called anywhere else in the
codebase — confirmed dead code, not a real, working path today.
```

---

## Deployment

**How do I deploy to AWS?**

```python
from pygeovision.cloud.deploy import AWSDeployer

result = AWSDeployer(region="us-east-1").deploy(
    "./model.onnx", endpoint_name="pygeovision-prod",
    instance_type="ml.g4dn.xlarge",
)
```

This is a real, genuine `boto3`/`sagemaker` integration, not a stub —
confirmed by reading its real implementation.

```{note}
An earlier version of this FAQ described a `pygeovision.serving.
InferenceServer` for self-hosted inference over HTTP. That module was
confirmed to have zero real imports anywhere in the codebase and was
removed entirely during this project's audit. If you need a real HTTP
inference server, ONNX export (above) plus your own FastAPI/Flask
wrapper around `ONNXRuntimeInference` is the real, current path.
```

---

## Getting Help

- GitHub Issues: [github.com/pygeovision/pygeovision/issues](https://github.com/pygeovision/pygeovision/issues)
- Discussions: [github.com/pygeovision/pygeovision/discussions](https://github.com/pygeovision/pygeovision/discussions)
