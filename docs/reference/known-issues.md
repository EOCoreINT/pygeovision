# Known issues

Limitations and behaviours that differ from what you might expect. Each item says what happens, why, and what to do. The dated
record of everything that was found and fixed is the [roadmap](../project/roadmap.md); this page is only what is still true today.

## Scope: what PyGeoVision does not ship

| Issue | Why | What to do |
|---|---|---|
| **No trained task models.** Seven pipelines (`crop_monitoring`, `deforestation`, `disaster_assessment`, `landslide_detection`, `road_extraction`, `solar_detection`, `urban_growth`) refuse to run without `--checkpoint`. | A network that has not been trained produces confident-looking random output, so none is shipped. The refusal happens before any imagery is downloaded. | Train your own ([Training](../training/index.md)), or use the ready sibling named on each pipeline page (`forest_change`, `urban_expansion`, `road_network`, the flood pipelines). |
| **The dataset catalogue is metadata.** The 460 entries describe datasets; nothing is downloaded. | Datasets have licences and terms the user must accept. | `DatasetLoader.download` explains where to get a dataset ([Datasets](../datasets/index.md)). |
| **`lisat-7b` is listed but cannot be built.** | Its model card's usage example does not match its documented architecture, so it stays unimplemented rather than guessed. | Use another vision-language model ([VLM](../vision-language/index.md)). |
| **No air-quality or dust pipeline.** | It would need aerosol data, and an uncalibrated proxy could be read as health guidance. | None. |

## Accuracy: what has not been measured

| Issue | Detail |
|---|---|
| **Accuracy on real imagery is not published.** | Methods are documented and checked on scenes with known answers. No pipeline has an accuracy figure on real ground, and none is claimed. Measure it on your own reference data ([Validation](../data/validation.md), [OBIA accuracy](../obia/accuracy.md)). |
| **Example outputs may come from synthetic scenes.** | Where a page shows numbers from the offline test harness it says so in a note. Yours will differ. |
| **Reference land-cover maps are model outputs.** | ESA WorldCover is itself a classification. Scoring a network against it measures agreement with WorldCover, not with the ground. |
| **Pretrained weights are not exercised in the documentation environment.** | Real DINOv2, CLIP/RemoteCLIP, Moondream, DOFA, Prithvi and SAM weights need a network and, for some, a GPU. The surrounding logic is tested with small local models. |
| **LiDAR classification accuracy is unmeasured.** | The plumbing is tested; there is no pretrained point-cloud network. |

## Behaviour that can surprise you

| Issue | Detail | What to do |
|---|---|---|
| **A search can return nothing in a rainy season.** | The default cloud limit is strict (10% in many pipelines). June over West Africa has few clear scenes. | The log names the limit to raise. Pass a larger `cloud_cover_max`, or choose another month ([Scene selection](../data/scene-selection.md)). |
| **255 and NaN mean "unknown", not "none".** | Cloud, shadow and out-of-image pixels are excluded from percentages and areas. | Check `valid_fraction` before reading any percentage ([Core concepts](../getting-started/concepts.md)). |
| **`gpu.*` and `training.*` configuration is stored, not read.** | Training and inference take their device and batch size from their own arguments. | Pass `devices=` and `TrainingConfig(batch_size=…)` ([Configuration reference](config.md)). |
| **Most exception classes are never raised.** | About nine of the 36 classes in `pygeovision.core.exceptions` are used; the rest are reserved names. | Catch the built-in exception, or test `result.success` ([Exceptions](exceptions.md)). |
| **Pipelines return failures instead of raising.** | `client.pipeline(...)` gives `success=False` and `error`. `ModelWeightsRequired` is the one exception. | Always test `result.success`. |
| **`area_m2` is in map units².** | In a geographic CRS (degrees) it is square degrees. | Use a projected CRS for areas ([OBIA conventions](../obia/index.md#conventions)). |

## OBIA

| Issue | Detail |
|---|---|
| **`multires` is not eCognition.** | It implements the published Baatz & Schäpe criterion, not the proprietary algorithm. Its boundaries will differ from eCognition's. |
| **Everything runs in memory on one machine.** | Scenes of a few thousand pixels on a side are comfortable ([Segmentation](../obia/segmentation.md#speed-and-size)). |
| **There is no built-in sample scene.** | OBIA works on real imagery only and needs a network for `obia fetch` (Planetary Computer and AWS Earth Search need no credentials). |
| **Rule strings are evaluated with `pandas.DataFrame.eval`.** | Treat a rules file as code ([Security](security.md)). |
| **Unnamed bands break NDVI rules.** | A GeoTIFF without band descriptions gets `b1, b2, …`, so no NDVI or NDWI is derived. Add the descriptions once ([OBIA](../obia/index.md#conventions)). |

## Code structure

| Issue | Detail |
|---|---|
| **The `PyGeoVision` client is one large module.** | `pygeovision/__init__.py` should move to its own module. |
| **Seven pairs of classes exist twice.** | The trainer pair was merged by re-export; the others remain ([Architecture](../architecture.md#systems-that-exist-twice)). |
| **`pygeovision/training/callbacks.py` is imported by nothing.** | The trainer uses `pygeovision/ai/training/callbacks.py`. |

## Not verified in the documentation environment

Ray Tune, MLflow, Weights & Biases and TensorBoard logging, mixed precision, DDP, TensorRT, and the live AWS, Azure and GCP paths.
The documentation environment had no GPU and no cloud account. The pages for these tools carry a "Verified here?" line saying exactly
what was and was not run.

Found something not listed? Open an issue with the output of [`pygeovision doctor`](../getting-started/diagnostics.md).
