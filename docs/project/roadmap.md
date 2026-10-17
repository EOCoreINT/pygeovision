# Roadmap

What exists today, what does not exist yet, and what has not been checked on real data. Nothing here is softened for
presentation. For what changed in each release, see the [changelog](changelog.md); for the dated record of past audits, see the
[audit history](audit-history.md).

## Where things stand (this release, 2.1.8)

| | |
|---|---|
| Pipelines | 41: 34 run out of the box, 7 need a checkpoint you trained. 21 stay [removed](../pipelines/removed.md) |
| Providers | 22 (through pygeofetch) |
| CLI | 25 groups, 92 commands, each documented with a captured `--help` |
| Models | 79 buildable architectures, 14 names for your checkpoints, 96 catalogue entries ([why three](../models/registry.md)) |
| Tests | Synthetic data throughout. No test ran against live imagery, a GPU, or a downloaded pretrained checkpoint |

## Open items

### Things that do not exist yet

- **No trained task models.** By design; see [what PyGeoVision will and will not do](../index.md). Seven pipelines need your checkpoint.
- **No automatic dataset download.** The 460-entry [catalogue](../datasets/index.md) is metadata. `DatasetLoader.download` explains where to get a dataset; it never downloads one.
- **`lisat-7b`** is listed but cannot be built: its model card's usage example does not match its documented architecture, so it stays unimplemented rather than guessed.
- **SkySense** (weights hosted outside the Hugging Face standard) and **RS-YOLO** (a category of papers, several with unavailable weights) are not in the registry.
- **No air-quality or dust pipeline**: both would need aerosol data (Sentinel-5P / MODIS AOD), and an uncalibrated proxy could be read as health guidance.

### Things that exist but are not verified on real data

- Outputs of real DINOv2, CLIP/RemoteCLIP, Moondream, DOFA, Prithvi and SAM weights. The documentation environment had no network and no GPU. The surrounding logic is tested with tiny random local models.
- Ray Tune, MLflow / W&B / TensorBoard logging, mixed precision, DDP, TensorRT and the live AWS / Azure / GCP paths.
- LiDAR **classification accuracy**. Only the plumbing is tested; there is no pretrained point-cloud network.
- Accuracy of any pipeline on real imagery. Methods are documented and checked on synthetic scenes with known answers. Validate against your own reference data ([validation](../data/validation.md)).

### Settings that are accepted but do nothing

- The `gpu` and `training` sections of the [configuration file](../reference/config.md) are stored but not read by training or inference.
- Of the 36 exception classes ([exceptions](../reference/exceptions.md)), about nine are raised; the others are reserved names.

See also [known issues](../reference/known-issues.md), the problems that are still true today.

```{toctree}
:hidden:
:maxdepth: 1

audit-history
```
