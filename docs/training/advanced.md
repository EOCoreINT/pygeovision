# Advanced training

These are the techniques you reach for once the basic loop in [Training a model](index.md) works: saving memory,
continuing a stopped run, fine-tuning gently, using several GPUs, searching for good settings, tracking experiments, and
getting the finished model out of PyTorch. Each section says **why** you would do it, **how**, **what you should see**,
and **what to do when** it does not work.

```{admonition} What was and was not run for this page
:class: note
Everything below that shows a log was run for real on a CPU with synthetic data. **Mixed precision, multi-GPU
training, Optuna, MLflow, Weights & Biases and TensorBoard need a GPU or packages that were not available**, so those
sections describe the behaviour from the code and tell you what to check. Where that is the case it says so.
```

## Staged fine-tuning

**Why.** A pretrained encoder already knows edges, textures and shapes. A freshly created segmentation head knows
nothing, and its first updates are large and wrong. If everything trains at once, those wrong updates flow back into the
encoder and damage what it knew. Training the head first, then the encoder at a lower learning rate, protects the
pretrained knowledge. It matters most with few labels.

**How.**

```python
cfg = TrainingConfig(
    output_dir="runs/staged", num_classes=2, in_channels=4,
    learning_rate=3e-4,
    unfreeze_backbone_epoch=1,     # the encoder stays fixed for the first epoch, then joins
    backbone_lr_scale=0.1,         # and learns at one tenth of the head's rate
    max_epochs=30,
)
model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=True)
GeoTrainer(model, cfg).fit(train, val)
```

The simpler forms: `freeze_backbone=True` keeps the encoder fixed for the **whole** run (a good first experiment with
fewer than ~100 chips); `backbone_lr_scale=0.1` alone trains everything from the start with a gentler encoder.

**Expected log** (a real run, 3 epochs, `unfreeze_backbone_epoch=1`):

```text
Froze 159 backbone parameter tensor(s) for the first 1 epoch(s).
Starting training: 3 epochs | device=cpu | precision=auto | strategy=auto
Epoch 1/3 | loss=0.7557 | val_loss=0.6604 | val_iou=0.7804 | lr=3.00e-04
Epoch 2: backbone unfrozen (learning rate x0.1).
Epoch 2/3 | loss=0.5748 | val_loss=0.6163 | val_iou=0.4401 | lr=2.25e-04
```

The `lr=` shown is the head's. The score may dip when the encoder joins; watch whether it recovers over the next epochs
before deciding the unfreeze was too early (try a later epoch or a smaller `backbone_lr_scale`).

**What to do when.**

| Symptom | Fix |
|---|---|
| Warning "no parameters matched the backbone-name patterns" | Your model's encoder is not called `encoder`, `backbone` or `patch_embed`. Rename the submodule, or freeze by hand: `for p in model.my_encoder.parameters(): p.requires_grad = False` |
| Score collapses when the encoder is unfrozen | Later `unfreeze_backbone_epoch`, smaller `backbone_lr_scale` (0.01) |

## Fine-tuning from your own checkpoint

**Why.** You trained on region A and now want region B, or you trained a 2-class model and now need 5 classes. Starting
from your earlier weights beats starting from scratch.

**How.** Build the model with the **new** class count and point `checkpoint_path` at the old weights:

```python
model = get_model("unet-r50", num_classes=3, in_channels=4, pretrained=False)
cfg = TrainingConfig(output_dir="runs/ft", num_classes=3, in_channels=4,
                     checkpoint_path="runs/res/checkpoints/best.pth", learning_rate=1e-4)
GeoTrainer(model, cfg).fit(train, val)
```

The weights load **non-strictly**: layers that fit are filled in, the rest keep their fresh values. Real log when a
2-class checkpoint goes into a 3-class model:

```text
Loaded checkpoint runs/res/checkpoints/best.pth with a partial match (strict=False) -- 2 missing key(s), 0 unexpected
key(s). This is expected if finetuning into a different num_classes/head; unexpected if you intended an exact match.
missing=['segmentation_head.0.weight', 'segmentation_head.0.bias']
```

Into a model of the same shape the log says `Loaded checkpoint … (exact match).`

This is the one place where PyGeoVision tolerates a partial match, and it is on purpose: changing the head is the
point. Read the `missing=` list. If it names layers in the **middle** of the network, you loaded the wrong
architecture. If the checkpoint cannot be read at all, training stops with an error rather than starting from random
weights.

## Resuming an interrupted run

**Why.** A job ran out of time, the machine rebooted, you pressed Ctrl-C. Resuming continues from the last finished
epoch with the same optimiser, schedule, history and early-stopping counter, instead of starting over.

**How.** Run the **same** script again with `resume=True` and the same `output_dir`:

```python
cfg = TrainingConfig(output_dir="runs/res", max_epochs=50, resume=True, ...)
```

Real log of a run with `max_epochs=4` that had been stopped after two epochs:

```text
Resuming from runs/res/checkpoints/last.pth after epoch 2.
Starting training: 4 epochs | device=cpu | precision=auto | strategy=auto
Epoch 3/4 | ...
```

The result reports `resumed_from_epoch: 2` and `history` covers all four epochs. If there is no `last.pth`, a fresh run
starts and the log says so. Resume needs `save_last=True` (the default).

```{admonition} Do not change max_epochs when resuming
:class: warning
The learning-rate schedule is stored in `last.pth` with the length it was created for. Resuming a 2-epoch run with
`max_epochs=4` gave `lr=1.00e-06` at epoch 3 (the old schedule had already finished) and then `lr=1.50e-04` at epoch 4
as the cosine curve started over. A warning is logged when the numbers differ. To train **longer** than planned, start
a new run from the best weights instead: `checkpoint_path=".../best.pth"`.
```

| Symptom | Fix |
|---|---|
| `RuntimeError … could not be restored into this model` | The model or settings differ from the run that wrote `last.pth` (different architecture, classes, optimiser, `unfreeze_backbone_epoch`…). Rebuild the same, or set `resume=False` |
| Resume starts from epoch 0 | `last.pth` is missing: wrong `output_dir`, or `save_last=False` earlier |

## Gradient accumulation

**Why.** Bigger batches give steadier gradients, but the largest batch that fits in memory may be 2. Accumulation
runs several small batches, adds up their gradients, and updates once, as if the batch were larger.

**How.** `batch_size=4, grad_accumulation_steps=4` behaves like a batch of 16 in terms of optimiser steps while holding
only 4 chips in memory. The learning-rate schedule counts optimiser steps, and a final incomplete group at the end of an
epoch is still applied.

**Watch out.** Batch-norm layers still see only the small batch of 4 when computing their statistics, so very small
batches can hurt networks with batch norm regardless of accumulation. For batches of 1 or 2, prefer a model with
group- or layer-norm (SegFormer, Swin), or freeze the batch-norm layers.

## Mixed precision

**Why.** Half-precision numbers use half the memory and run faster on modern GPUs, allowing larger batches or models
with almost no loss of accuracy.

**How.** `precision="auto"` (the default) picks `bf16` where the GPU supports it, else `fp16`; on a CPU it stays in
full precision. `"fp16"` and `"bf16"` force the choice, `"fp32"` turns it off. With `fp16` a gradient scaler guards
against numbers too small for half precision.

*Not run here (no GPU).* The first log line shows what was asked (`precision=auto`), not what was chosen. Confirm the
effect with `nvidia-smi`: memory use should drop noticeably.

| Symptom | Fix |
|---|---|
| `loss` becomes `nan` only with `fp16` | Use `bf16` if the GPU supports it, or `fp32`; lower the learning rate |
| Setting `fp16` on a CPU changes nothing | Correct: mixed precision applies on GPUs only |

## Several GPUs

**What exists.** With more than one visible GPU and `strategy="auto"` or `"ddp"`, the trainer wraps the model in
PyTorch's DistributedDataParallel (`"fsdp"` for FullyShardedDataParallel); `sync_batchnorm` shares batch-norm statistics.
Any other `strategy` value logs a warning and trains on one device.

**Treat this as experimental.** It was not run for this documentation, and from the code:

- The process group is created from environment variables, so the script must be started with `torchrun
  --nproc_per_node=N train.py`, not plain `python`.
- The loaders `GeoTrainer` builds from datasets use ordinary shuffling and **no** `DistributedSampler`, so every process
  would train on the whole dataset instead of its share. Build your own loaders with a `DistributedSampler` and pass
  `fit(train_loader=..., val_loader=...)`; call `sampler.set_epoch(epoch)` yourself if you need reshuffling.
- Every process writes checkpoints into the same folder.

If a single GPU is enough, use it: it is the path that is exercised. If you need several, write a short loop with
`pygeovision.training.distributed` helpers (`setup_distributed`, `wrap_ddp`, `launch_ddp`) or a framework such as
PyTorch Lightning, and keep `GeoTrainer`'s model, losses and metrics.

## Hyper-parameter search

**Why.** Learning rate, batch size and weight decay interact, and good values differ between datasets. A search tries
many combinations and reports the best, instead of you guessing.

**How.** Needs `pip install optuna` (in `pygeovision[train]`). Each trial must train a **fresh model with a fresh
trainer**, and return the score to maximise.

```python
from pygeovision.models import get_model
from pygeovision.training import GeoTrainer, TrainingConfig, OptunaHPO

def objective(params):
    cfg = TrainingConfig(output_dir=f"runs/hpo/{params['learning_rate']:.1e}", num_classes=2, in_channels=4,
                         max_epochs=10, num_workers=0, **params)
    model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=True)
    return GeoTrainer(model, cfg).fit(train, val)["best_val_iou"]

best = OptunaHPO(n_trials=20, direction="maximize").run(
    objective_fn=objective,
    search_space={
        "learning_rate": ("float_log", 1e-5, 1e-2),
        "weight_decay":  ("float_log", 1e-6, 1e-2),
        "batch_size":    ("categorical", [4, 8, 16]),
    },
)
print(best)   # {'best_params': {...}, 'best_value': 0.83, 'n_trials': 20, 'study_name': 'pgv_hpo'}
```

Search-space entries are `("float", lo, hi)`, `("float_log", lo, hi)`, `("int", lo, hi)`, `("int_log", lo, hi)` and
`("categorical", [choices])`. Anything else raises `ValueError`. Without Optuna installed you get an `ImportError`
telling you to install it (not an empty result).

Notes:

- Use a **short** `max_epochs` per trial; you are ranking settings, not producing the final model. Retrain the winner
  for the full length afterwards.
- `storage="sqlite:///hpo.db"` saves the study so it can be continued; `n_jobs>1` runs trials in parallel threads
  (mind GPU memory).
- The `pruner` setting is accepted, but trials report only their final score, so nothing is pruned early.
- `hpo.importance()` returns which parameters mattered most (needs enough trials).
- *Verified here with a real Optuna install on a toy objective (15 trials; `importance()` returned shares that sum to 1);
  a search whose trials train real networks was not run.* The search-space handling and error paths are also covered by tests.
- A second, more general search tool exists: [`GeoAutoML`](../advanced/automl.md) takes any function (not only
  `GeoTrainer`), supports pruning, and records failed trials instead of aborting.

## Experiment tracking

**Why.** After twenty runs you will not remember which learning rate gave which curve. A tracker records settings and
per-epoch metrics for every run so you can compare them.

**How.** One setting:

```python
cfg = TrainingConfig(tracker="tensorboard", experiment_name="crops", run_name="unet-lr3e-4", ...)
# several at once:  tracker="mlflow,tensorboard"     everything: tracker="all"
```

| Tracker | Needs | Where to look |
|---|---|---|
| `tensorboard` | `pip install tensorboard` | `tensorboard --logdir runs/<output_dir>/tb_logs` |
| `mlflow` | `pip install mlflow` | `mlflow ui` in the folder given by `mlflow_uri` (default `./mlruns`) |
| `wandb` | `pip install wandb` and `wandb login` | the Weights & Biases web page for `wandb_project` |

Logged per epoch: `val_loss`, `val_iou`, `val_f1`, `val_accuracy`, per-class scores, `train_loss`, `lr`, `epoch`. The
settings (`TrainingConfig`) are logged once at the start.

**If a tracker cannot start, training goes on.** Real log with `tracker="mlflow"` and MLflow not installed:

```text
mlflow not installed
Experiment tracking requested (mlflow) but no backend could start; continuing without it.
```

A line `Experiment tracking: <names>` means those backends really started. *Not run here with a live backend.* The
logging calls are verified against a stand-in tracker in the tests.

## Callbacks and custom metrics

`GeoTrainer` has **no callback hooks**, and `fit(metric_fn=...)` is accepted but not used (a warning says so). The
classes in `pygeovision.training.callbacks` (`EarlyStopping`, `ModelCheckpoint`, `LearningRateMonitor`) are for loops
you write yourself. To get your own score:

1. train as usual;
2. load `best.pth`;
3. evaluate on validation data with `SegmentationMetrics` or your own code
   ([Evaluating](../benchmark/index.md)).

```python
from pygeovision.training import SegmentationMetrics

metrics = SegmentationMetrics(num_classes=2)         # ignores label 255
metrics.update(predicted_labels, true_labels)         # (B, H, W) integer tensors
print(metrics.compute())   # mean_iou, mean_f1, accuracy, iou_class_0, f1_class_0, precision_class_0, recall_class_0, ...
```

## Exporting and shrinking a model

**Why.** The `.pth` file needs PyTorch and your code to load. Exporting makes a file other tools can run (ONNX Runtime
on a server or an edge device, TensorRT, a browser), and compression makes it smaller or faster at some cost in
accuracy.

### ONNX

```python
import torch
from pygeovision.models import get_model
from pygeovision.training.hpo import ModelOptimizer

model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)
model.load_state_dict(torch.load("runs/demo/checkpoints/best.pth", weights_only=True)["model_state_dict"])

opt = ModelOptimizer(model, num_classes=2, in_channels=4)
path = opt.export_onnx("model.onnx", input_size=512)       # any batch size, exactly 512 x 512 pixels; needs `onnx`
out = opt.onnx_inference(some_array, path)                 # (4, H, W) or (1, 4, H, W) float32 array; needs onnxruntime
```

Expected: `model.onnx` is written (a few MB for a small net, over 100 MB for U-Net/ResNet-50), `onnx_inference` returns an
array shaped `(1, classes, H, W)` that matches the PyTorch output to about 1e-4. Pass `simplify=False` to skip graph
simplification (it runs only if `onnxsim` is installed). **The file accepts exactly the `input_size` you export with**
(any batch size, but one height and width): another size fails with `Got invalid dimensions for input`. Pass
`dynamic_spatial=True` to leave height and width free; it worked for U-Net/ResNet-50, FCN-ResNet-50 and SegFormer-B0
(outputs matched PyTorch at 96, 128 and 160 pixels), but check your own architecture on a second size, because models
with fixed reshapes or resizes can export wrongly. Deployment details are in [Edge deployment](../edge-cloud/edge.md).

The same can be done from the shell for a registry model with weights you trained:

```bash
pygeovision edge export-onnx unet-r50 --checkpoint runs/demo/checkpoints/best.pth \
        --classes 2 --in-channels 4 --input-size 512 -o model.onnx      # add --dynamic-size for any height/width
```

`TrainingConfig(export_onnx=True)` also writes `model.onnx` after training, but of the **last epoch**, not the best.
Prefer exporting `best.pth` as above. Older PyTorch exporters needed an extra package (`onnxscript`); PyGeoVision
uses the classic exporter, so only `onnx` is required.

### TorchScript

`opt.export_torchscript("model.pt", method="trace", input_size=512)` writes a self-contained file that loads with
`torch.jit.load` and no source code. Tracing fixes the control flow for the example input; if your model branches on
input values use `method="script"`.

### INT8 quantisation

```python
q = ModelOptimizer(model, in_channels=4).quantize_int8(calibration_loader, "model_int8.pth")
```

The calibration loader should yield batches of **real, training-like** chips (`(images, labels)` pairs are fine). The
result runs on a **CPU** with `float32` input and output. Layers are not fused, so the speed-up and accuracy are
modest. Many segmentation networks use operations PyTorch cannot quantise (upsampling, concatenation, some
activations). In that case you get a `RuntimeError` saying so rather than a model that fails later: export to ONNX and
quantise there. **Always compare IoU on validation data before and after**; a quantised model that runs is not
necessarily one that is still accurate.

### Pruning

`opt.prune(amount=0.3)` sets the 30% smallest weights of every convolution and linear layer to zero (`method="global"`
prunes across the whole network instead). Batch-norm layers are left alone. The model keeps its ordinary weight names,
so it loads into the original architecture. Zeros alone do not shrink the file or speed up ordinary dense inference; use
pruning as a starting point for fine-tuning or a sparse runtime. Fine-tune afterwards to recover accuracy.

### Speed

`opt.benchmark_speed(input_size=512, n_runs=100, device="cuda")` returns `{"mean_ms", "std_ms", "fps", "p95_ms", ...}`
for one image (falls back to the CPU if CUDA is unavailable). Measure on the hardware you will deploy on.

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError … onnx` | The `onnx` package is not installed | `pip install onnx onnxruntime` (the `edge` extra) |
| ONNX output differs from PyTorch by more than 1e-3 | Model not in `eval()` mode before export, or dropout/batch-norm in training mode | `ModelOptimizer` calls `eval()`; if you export yourself, call `model.eval()` first |
| Export fails on a detector | Detectors take lists of images, not a tensor | Export the backbone or use the framework's own detection export |
| Quantised model much worse | Calibration data not representative, or accuracy-sensitive layers | Calibrate on more, varied chips; keep the float model |

## What to do when…

| Symptom | Likely cause | What to do |
|---|---|---|
| Out of memory | Batch, chip or model too large | `batch_size` first; then `grad_accumulation_steps` to keep the effective batch; then `fp16`/`bf16`; then smaller chips or model |
| Training is slower than expected on GPU | Data loading is the bottleneck | `num_workers` 4–8, `pin_memory=True`, `persistent_workers=True`; keep chips on a local SSD |
| Results change between identical runs | GPU kernels are not fully deterministic | Same `seed` gives similar, not identical, results; compare several seeds before concluding one setting is better |
| Validation score fluctuates wildly | Validation set too small, or small batches with batch-norm | More validation chips; larger effective batch; lower learning rate |
| `val_iou` fine but one class is poor | Averaged over classes | Early-stop and checkpoint on that class: `early_stopping_metric="val_iou_class_K"` |
