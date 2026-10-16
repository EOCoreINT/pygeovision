# Speed and memory

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
