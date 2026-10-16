# The training configuration

Everything `GeoTrainer` can be told is held in one object, `TrainingConfig`. This page explains every setting: what it
does, its default, when to change it, and a few that exist but do **not** do what their name suggests. Treat the
tables as the reference and the *Why* paragraphs as the advice.

```python
from pygeovision.training import TrainingConfig

cfg = TrainingConfig(output_dir="runs/crops_v1", num_classes=5, in_channels=6, max_epochs=60, batch_size=8)
```

Anything you do not set keeps its default. A configuration can be written to a file and read back, so a run can be
reproduced:

```python
cfg.save("runs/crops_v1/config.yaml")
cfg = TrainingConfig.from_yaml("runs/crops_v1/config.yaml")
```

`from_yaml` ignores unknown keys, but logs a warning that names them, so a misspelt `lerning_rate` does not pass
silently.

```{admonition} Must match your data
:class: important
Three settings must agree with your data and with the model you built: **`num_classes`**, **`in_channels`** and
**`task`**. They are not read from the model. A mismatch usually shows up as a shape error on the first batch, but a
wrong `num_classes` for the *metrics* can run without complaint and report nonsense.
```

## Run identity and output

| Setting | Default | What it does |
|---|---|---|
| `output_dir` | `"./training_output"` | Folder for everything the run writes (`checkpoints/`, `tb_logs/`, `model.onnx`). Created if missing. **Not cleared** between runs: older `epoch_*` files stay |
| `experiment_name` | `"pgv_experiment"` | Name of the experiment in MLflow, Weights & Biases and TensorBoard. Has no effect on file names |
| `run_name` | `""` | Name of this run inside the experiment tracker |
| `seed` | 42 | Seed for Python, NumPy and PyTorch random numbers. Same seed and same data give similar, but not bit-identical, runs on a GPU |

*Why:* give every experiment its own `output_dir` (`runs/crops_v1`, `runs/crops_v2`) so checkpoints never mix.

## Data loading

| Setting | Default | What it does |
|---|---|---|
| `batch_size` | 16 | Chips per step. The first thing to lower when memory runs out |
| `num_workers` | 4 | Background processes that read chips. `0` loads in the main process: simplest and the right choice on a laptop, Windows, or when debugging |
| `pin_memory` | `True` | Faster copy to the GPU. Set `False` on CPU |
| `prefetch_factor` | 2 | Batches each worker prepares ahead. Ignored when `num_workers=0` |
| `persistent_workers` | `True` | Keep workers alive between epochs. Ignored when `num_workers=0`. Set `False` with `num_workers=0` to avoid a PyTorch warning |
| `train_val_split` | 0.8 | **Not used.** `GeoTrainer` trains on the dataset you pass first and validates on the second. Split the data yourself ([Preparing training data](data.md#splitting-train-and-validation)). A warning is logged if you change it |

If you pass ready-made data loaders (`fit(train_loader=..., val_loader=...)`), these settings are not applied to them.

## The model

| Setting | Default | What it does |
|---|---|---|
| `task` | `"segmentation"` | `"segmentation"` (a mask per pixel) or `"detection"` (boxes). Changes the loss, the validation metric and the collation of batches |
| `num_classes` | 2 | Number of classes the metrics should count, background included. For detection, the classes **including** the background of the detector |
| `in_channels` | 3 | Bands in. Used by the ONNX export; must equal the model's input |
| `pretrained` | `True` | Recorded in the configuration only. The weights are whatever the model you pass in already has: `get_model(..., pretrained=True)` decides |
| `freeze_backbone` | `False` | Keep the encoder (the part that was pretrained) fixed for the whole run and train only the head. See [Advanced training](fine-tuning.md#staged-fine-tuning) |
| `unfreeze_backbone_epoch` | `None` | Train the head alone for the first *N* epochs, then train everything. Overrides `freeze_backbone` |
| `backbone_lr_scale` | 1.0 | Learning rate of the encoder = `learning_rate × backbone_lr_scale`. `0.1` fine-tunes a pretrained encoder gently |
| `checkpoint_path` | `None` | Start from the weights in this file (fine-tuning). Loaded **non-strictly**: layers that fit are filled in, others are reported in a warning. Use it to move to a different `num_classes` |

```{admonition} How the encoder is found
:class: note
Freezing and `backbone_lr_scale` treat as "encoder" every parameter whose name contains `encoder`, `backbone` or
`patch_embed`. That covers the built-in U-Nets, SegFormers and detectors. For a model of your own, name the
encoder submodule `self.encoder` or `self.backbone`. If nothing matches, a warning says nothing was frozen.
```

`checkpoint_path` versus `get_model(pretrained=True)`: the first continues from **your** earlier training; the second
starts from public weights. Neither is needed to resume an interrupted run; that is `resume=True`
([below](#checkpoints-and-resuming)).

## Optimiser

| Setting | Default | What it does |
|---|---|---|
| `optimizer` | `"adamw"` | `adamw`, `adam`, `sgd` or `lion` (needs `pip install lion-pytorch`). Any other name raises an error: nothing is silently replaced |
| `learning_rate` | 1e-4 | The step size. The most influential setting. 1e-4 suits fine-tuning a pretrained model; from scratch 1e-3 can work. If `loss` is `nan` or jumps, lower it |
| `weight_decay` | 1e-4 | A small pull of all weights towards zero, which discourages over-fitting |
| `momentum` | 0.9 | `sgd` only |
| `nesterov` | `True` | `sgd` only |
| `grad_clip` | 1.0 | Limits the overall size of a gradient step, which tames sudden spikes. `0` turns it off |
| `grad_accumulation_steps` | 1 | Add up gradients over this many batches before updating: an effective batch of `batch_size × steps` with the memory of one. See [Advanced training](performance.md#gradient-accumulation) |

*Which optimiser:* `adamw` is the dependable default. Choose `sgd` (learning rate around 1e-2, with momentum) when you
want to follow a recipe that was published with SGD.

## Learning-rate schedule

The schedule changes the learning rate over the run: large at first to make progress, small at the end to settle.

| `scheduler` | Behaviour | Stepped |
|---|---|---|
| `"cosine"` (default) | Smooth decay from `learning_rate` down to `min_lr` over `max_epochs`. **No warm-up** | each epoch |
| `"cosine_warmup"` / `"warmup_cosine"` | Starts at `learning_rate × warmup_lr_scale`, rises to `learning_rate` over `warmup_epochs`, then decays by cosine to `min_lr` | each step |
| `"linear"` | Straight line from `learning_rate` to 0 over the whole run | each step |
| `"step"` | Multiplies the rate by `lr_gamma` every `lr_step_size` epochs | each epoch |
| `"plateau"` | Multiplies the rate by `lr_gamma` when `val_loss` has not improved for 5 epochs; never below `min_lr` | each epoch, on `val_loss` |
| `"onecycle"` | Rises quickly (first 10%) then decays | each step |
| `"none"` | Constant rate | |

| Setting | Default | Used by | Meaning |
|---|---|---|---|
| `warmup_epochs` | 5 | `cosine_warmup`, `warmup_cosine` only | Length of the warm-up. With the default `cosine` it does nothing |
| `warmup_lr_scale` | 0.1 | the same two | Starting rate as a fraction of `learning_rate` |
| `min_lr` | 1e-6 | `cosine`, both warm-up cosines, `plateau` | The floor |
| `lr_step_size` | 30 | `step` | Epochs between drops |
| `lr_gamma` | 0.1 | `step`, `plateau` | The factor each drop multiplies by |

An unknown name raises an error. A cosine schedule is laid out for `max_epochs`: if early stopping ends the run sooner,
the rate has not reached `min_lr`, which is fine.

## Length and early stopping

| Setting | Default | What it does |
|---|---|---|
| `max_epochs` | 100 | Upper limit on epochs (one epoch is one pass over the training chips) |
| `early_stopping_patience` | 15 | Stop after this many epochs without improvement of the monitored metric. Needs a validation set |
| `early_stopping_metric` | `"val_iou"` | What to watch. See [Metric names](#metric-names). For `task="detection"` the default `val_iou` is switched to `val_map50` automatically |
| `early_stopping_mode` | `"max"` | `"max"` if bigger is better (IoU), `"min"` for losses (`val_loss`) |

The log prints `Early stopping at epoch N (patience=P)` when it triggers. If your run always runs to `max_epochs`,
the metric is still improving: raise `max_epochs`.

## Precision and speed

| Setting | Default | What it does |
|---|---|---|
| `precision` | `"auto"` | `"fp32"`: full precision. `"fp16"` / `"bf16"`: mixed precision on a GPU, faster and lighter on memory. `"auto"`: `bf16` if the GPU supports it, else `fp16`; `fp32` on a CPU |
| `compile_model` | `False` | `torch.compile` the model (PyTorch 2). Faster steps after a slow first one; if compilation fails a warning is logged and training goes on without it |

Mixed precision only takes effect on a GPU; on a CPU the setting is ignored.

## Several devices

| Setting | Default | What it does |
|---|---|---|
| `devices` | `"auto"` | `"auto"` picks CUDA, then Apple MPS, then CPU. `"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"` or a GPU number choose directly. A list uses its first entry |
| `strategy` | `"auto"` | With more than one GPU visible, `auto` uses DistributedDataParallel; `"ddp"`, `"fsdp"`, `"dp"` ask for one explicitly |
| `sync_batchnorm` | `True` | Share batch-norm statistics across GPUs (DDP) |
| `find_unused_parameters` | `False` | Passed to DDP; set `True` if DDP complains about parameters that received no gradient |

With one GPU or a CPU these do nothing beyond choosing the device. Read [Advanced training](performance.md#several-gpus)
before relying on multi-GPU: how the built-in loop behaves there is limited.

## Checkpoints and resuming

| Setting | Default | What it does |
|---|---|---|
| `save_top_k` | 3 | Keep this many best epochs as `epoch_XXXX_<metric>_<score>.pth`; worse ones are deleted. `best.pth` is always a copy of the best |
| `checkpoint_metric` | `"val_iou"` | Which metric decides "best" for these files. Switched to `val_map50` for detection |
| `checkpoint_mode` | `"max"` | `"max"` or `"min"` |
| `checkpoint_every_n_epochs` | 1 | Consider an epoch for saving only every *N* epochs |
| `save_last` | `True` | Write `last.pth` (model, optimiser, scheduler, history, early-stopping state) after every epoch. Needed for `resume` |
| `resume` | `False` | If `True` and `output_dir/checkpoints/last.pth` exists, continue from it. The model must be built with the same architecture; if the file does not fit, training stops with an error rather than starting over. If there is no `last.pth`, a fresh run starts |

`checkpoint_metric` and `early_stopping_metric` are separate. The returned `best_metric_name` reports the early-stopping
one, so set both to the same metric unless you have a reason not to.

Checkpoints are written only when a validation set is given. Without one you get `last.pth` and nothing else, and
`best_checkpoint` is `None`.

## Augmentation

Augmentation shows the network slightly altered copies of each training chip (flipped, shifted, brightened) so it
learns the object and not the exact pixels.

| Setting | Default | What it does |
|---|---|---|
| `augment` | `True` | Augment the **training** set (segmentation only). Needs the `albumentations` package (installed with `pygeovision[train]`); without it `fit` raises an `ImportError` that says so. For detection it is ignored and a warning says so |
| `augment_mode` | `"medium"` | Strength: `"light"`, `"medium"`, `"heavy"`, `"geometric_only"`, `"radiometric_only"` |
| `flip_prob`, `rotate_limit`, `scale_limit`, `brightness_limit`, `contrast_limit` | 0.5, 30, 0.1, 0.2, 0.2 | **Not used.** Kept so old configuration files still load. Strength comes from `augment_mode`; a warning is logged if you change one of these |

What the augmentation does, per mode:

| Mode | Geometry (flip, turn by 90°, small shift/scale, transpose) | Brightness, contrast, gamma | Haze, noise, blur of resolution | Dropped bands |
|---|---|---|---|---|
| `light` | flips and 90° turns | mild | sensor noise only | no |
| `medium` | + shift, scale, transpose | yes | haze, noise, softening | no |
| `heavy` | + elastic and grid distortion | stronger | stronger | occasionally |
| `geometric_only` | as `light` | no | no | no |
| `radiometric_only` | no | yes | yes | no |

Rotations are in multiples of 90° only, so that roads and field edges stay geographically plausible.

Points to know:

- **Hue/saturation and CLAHE are used only for 1- or 3-band chips.** They are defined for colour images; for 4, 6 or more
  bands they are left out, so multispectral training works.
- **Values are expected between 0 and 1.** Several transforms clip to that range. With z-scored or decibel data, set
  `augment=False` and augment in your own dataset.
- It is applied after the dataset, so do not also set `GeoSegDataset(augment=True)`.

## Experiment tracking

| Setting | Default | What it does |
|---|---|---|
| `tracker` | `"none"` | `"mlflow"`, `"wandb"`, `"tensorboard"`, `"all"`, or several joined with commas (`"mlflow,tensorboard"`). A missing package or backend logs a warning and is skipped; training never stops because of tracking |
| `mlflow_uri` | `"mlruns"` | MLflow tracking location (a folder or a server URL) |
| `wandb_project` | `"pygeovision"` | Weights & Biases project |
| `log_every_n_steps` | 10 | Interval for per-step debug messages (visible only with `logging.DEBUG`) |

Per-epoch numbers are logged to the trackers. See [Advanced training](../experiments/tracking.md#experiment-tracking).

## Export

| Setting | Default | What it does |
|---|---|---|
| `export_onnx` | `False` | After training, write `output_dir/model.onnx` (input 1 × `in_channels` × 512 × 512, dynamic batch size) |
| `export_torchscript` | `False` | After training, write a TorchScript file |
| `onnx_opset` | 17 | ONNX operator set version |

The export is of the model **as it is after the last epoch**, not of `best.pth`, and a failure only logs a warning.
For a deployable file from the best weights, load `best.pth` and export it yourself
([Advanced training](export.md#exporting-and-shrinking-a-model)). Detection models do not export through this path.

## Metric names

The names you can give `early_stopping_metric` and `checkpoint_metric` (segmentation):

| Name | Meaning | Mode |
|---|---|---|
| `val_loss` | Validation loss | `min` |
| `val_iou` | Mean IoU over classes | `max` |
| `val_f1` | Mean F1 over classes | `max` |
| `val_accuracy` | Fraction of pixels right (dominated by the big class) | `max` |
| `val_iou_class_K`, `val_f1_class_K`, `val_precision_class_K`, `val_recall_class_K` | The same for class number `K` | `max` |

For detection: `val_map50` (mean average precision at IoU 0.5), `val_map50_95`, and `val_loss`.

*Why per-class names matter:* when one class is rare and is the reason you are training, `val_iou_class_1` as the
metric stops training from being steered by the easy majority class.

```python
cfg = TrainingConfig(early_stopping_metric="val_iou_class_1", checkpoint_metric="val_iou_class_1", ...)
```

A metric name that is not produced never improves, so early stopping would trigger after `patience` epochs: if the run
stops at epoch `patience + 1`, check the spelling.

## A full example file

```yaml
# runs/crops_v1/config.yaml
output_dir: runs/crops_v1
experiment_name: crops
task: segmentation
num_classes: 5
in_channels: 6
batch_size: 8
num_workers: 4
max_epochs: 60
optimizer: adamw
learning_rate: 0.0001
scheduler: cosine_warmup
warmup_epochs: 3
early_stopping_patience: 12
early_stopping_metric: val_iou
precision: auto
augment: true
augment_mode: medium
tracker: tensorboard
```

## What to do when…

| Symptom | Likely cause | What to do |
|---|---|---|
| Warning "TrainingConfig fields … have no effect" | You set `flip_prob`, `rotate_limit`, … | Use `augment_mode` |
| Warning "Ignoring unknown TrainingConfig keys" | A misspelt or removed key in the YAML | Fix the key |
| Run stops after `early_stopping_patience + 1` epochs | The monitored metric name is wrong, or it truly never improved | Check the spelling against [Metric names](#metric-names) |
| `best_checkpoint` is `None` | No validation set was given | Pass a validation dataset |
| `ValueError: Unknown optimizer / scheduler` | A name that does not exist | Pick from the tables above |
| `ImportError: optimizer='lion' needs lion-pytorch` | Optional package missing | `pip install lion-pytorch` or use `adamw` |
| `RuntimeError … could not be restored` on `resume=True` | The model or optimiser differs from the run in `last.pth` | Rebuild exactly the same model and settings, or set `resume=False` |
| Warm-up has no effect | `scheduler="cosine"` has none | Use `cosine_warmup` |
| ONNX file missing | The export failed (see the warning), or you used a detector | Export yourself after training |
