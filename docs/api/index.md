# API Reference

Reference for PyGeoVision's modules. Numbers below are confirmed real
(checked directly against `repr(PyGeoVision())` and the code this cycle) —
several corrections from earlier versions of this page are noted inline.

---

## Modules

| Module | Description | Verification |
|---|---|---|
| [PyGeoVision Client](pygeovision.md) | Main client — search, download, labeling, inference | Core methods verified |
| [Pipelines](pipelines.md) | 51 registered pipeline names, 10 real | See breakdown |
| [Model Layer](models.md) | 98 architectures (not 119) in the registry | Import verified |
| [Auto-Labeling](labeling.md) | 7+ sources | Import verified; separate from `ai.labeling` |
| [Loss Functions](losses.md) | 10 geospatial losses | Import verified |
| [Inference Engine](inference.md) | Tiled/batch/streaming/ensemble | Import verified; separate from `ai.inference` |
| [Explainability](explainability.md) | GradCAM, SHAP, uncertainty | Import verified |
| [Monitoring](monitoring.md) | Drift detection, performance tracking | Import verified |
| [Training Framework](training.md) | GeoTrainer, distributed, mixed precision | Import verified |
| [Serving API](serving.md) | FastAPI inference server | Import verified |
| [Dataset Registry](datasets.md) | 503-entry benchmark database | Count confirmed real |
| [Foundation Models](foundation.md) | DINOv3, Prithvi-EO-2.0 | Import verified |
| [Edge Deployment](edge.md) | ONNX Runtime, Jetson | Import verified |
| [Cloud Deployment](cloud.md) | AWS / Azure / GCP | Import verified |
| [Advanced AI](advanced.md) | Few-shot, multi-task, AutoML | Import verified |
| [Vision-Language Models](vlm.md) | CLIP, Moondream | Import verified |
| [Time Series](timeseries.md) | NDVI/NDWI trends, anomaly detection | Import verified |
| [Point Cloud (3D)](pointcloud.md) | LiDAR, CHM — real class is `PointCloudProcessor`, not `LiDARProcessor` | Corrected this cycle |
| [Agent](agent.md) | Natural-language task planning, 8 real tools | Tested this cycle |
| [Visualization](viz.md) | Interactive maps, raster/vector viewers | Import verified |
| [CLI Reference](cli.md) | ~26 command groups, not 15 — only `channel` tested | Partially verified |

"Import verified" means the primary class/function shown for that module
was confirmed to actually exist and import correctly this cycle — it does
not mean the full behavior of every method was tested. See each page's
own verification note, and [Architecture](../architecture.md) for the
overall picture.

---

## Import Paths

```python
# Main client
import pygeovision as pgv
client = pgv.PyGeoVision()

# Model layer
from pygeovision.models import get_model, model_registry

# Foundation models
from pygeovision.models.foundation.dinov3 import DINOv3Backbone, CHMv2Model, DINOv3Text

# Inference (audited, memory-efficient version)
from pygeovision.ai.inference.tiled_inference import TiledInference

# Inference (separate, unaudited version -- used by client.inference/client.detection.custom)
from pygeovision.inference.tiled import TiledInference as UnauditedTiledInference

# Training
from pygeovision.training.trainer import GeoTrainer
from pygeovision.training.checkpoint import CheckpointManager

# Losses
from pygeovision.losses.segmentation import DiceLoss, FocalLoss, TverskyLoss

# Serving
from pygeovision.serving import InferenceServer

# Pipelines -- 10 real, audited ones
from pygeovision.ai.pipelines import LandCoverPipeline, BuildingFootprintsPipeline

# Dataset registry
from pygeovision.datasets.registry import dataset_registry

# Edge / Cloud
from pygeovision.edge.onnx_rt import ONNXRuntimeInference
from pygeovision.cloud.deploy import AWSDeployer

# Point cloud (real class name, corrected this cycle)
from pygeovision.advanced.pointcloud import PointCloudProcessor

# Agent
from pygeovision.agent import GeoAgent
```
