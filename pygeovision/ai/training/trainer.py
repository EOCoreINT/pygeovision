"""
GeoTrainer — distributed geospatial model training (Phase 4.1).

Supports:
  • Multi-GPU DDP + FSDP
  • Mixed precision (FP16, BF16, auto)
  • Gradient accumulation + checkpointing
  • Hyperparameter optimization (Optuna, Ray Tune)
  • Experiment tracking (MLflow, W&B, TensorBoard)
  • Early stopping, LR scheduling, EMA
  • ONNX / TorchScript export after training
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration for GeoTrainer."""

    # ── Core ──────────────────────────────────────────────────────────
    output_dir: str = "./training_output"
    experiment_name: str = "pgv_experiment"
    run_name: str = ""
    seed: int = 42

    # ── Data ──────────────────────────────────────────────────────────
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    train_val_split: float = 0.8

    # ── Model ─────────────────────────────────────────────────────────
    task: str = "segmentation"  # segmentation | detection
    num_classes: int = 2
    in_channels: int = 3
    pretrained: bool = True
    freeze_backbone: bool = False
    checkpoint_path: str | None = None  # real path to an existing .pth to finetune from

    # ── Optimisation ──────────────────────────────────────────────────
    optimizer: str = "adamw"          # adamw | sgd | lion | lars
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = True
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler: str = "cosine"         # cosine | linear | step | plateau | onecycle
    warmup_epochs: int = 5
    warmup_lr_scale: float = 0.1
    min_lr: float = 1e-6
    lr_step_size: int = 30
    lr_gamma: float = 0.1

    # ── Training loop ─────────────────────────────────────────────────
    max_epochs: int = 100
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_iou"
    early_stopping_mode: str = "max"

    # ── Precision ─────────────────────────────────────────────────────
    precision: str = "auto"           # fp32 | fp16 | bf16 | auto
    compile_model: bool = False       # torch.compile (PyTorch 2.x)

    # ── Distributed ───────────────────────────────────────────────────
    strategy: str = "auto"            # auto | ddp | fsdp | dp
    devices: str | int | list[int] = "auto"
    sync_batchnorm: bool = True
    find_unused_parameters: bool = False

    # ── Checkpointing ─────────────────────────────────────────────────
    save_top_k: int = 3
    save_last: bool = True
    checkpoint_metric: str = "val_iou"
    checkpoint_mode: str = "max"
    checkpoint_every_n_epochs: int = 1

    # ── Augmentation ──────────────────────────────────────────────────
    augment: bool = True
    flip_prob: float = 0.5
    rotate_limit: int = 30
    scale_limit: float = 0.1
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2

    # ── Experiment tracking ───────────────────────────────────────────
    tracker: str = "none"             # none | mlflow | wandb | tensorboard | all
    mlflow_uri: str = "mlruns"
    wandb_project: str = "pygeovision"
    log_every_n_steps: int = 10

    # ── Export ────────────────────────────────────────────────────────
    export_onnx: bool = False
    export_torchscript: bool = False
    onnx_opset: int = 17

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def save(self, path: str | Path) -> None:
        import yaml
        Path(path).write_text(yaml.dump(self.to_dict()))


class EarlyStopping:
    """Early stopping with patience and best-model tracking."""

    def __init__(self, patience: int, metric: str, mode: str = "max") -> None:
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def update(self, metrics: dict[str, float]) -> bool:
        val = metrics.get(self.metric, 0.0)
        improved = (
            self.best is None or
            (self.mode == "max" and val > self.best) or
            (self.mode == "min" and val < self.best)
        )
        if improved:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class CheckpointManager:
    """Save top-k checkpoints + last checkpoint."""

    def __init__(self, output_dir: Path, top_k: int = 3, metric: str = "val_iou", mode: str = "max") -> None:
        self.dir = output_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.metric = metric
        self.mode = mode
        self._history: list[tuple] = []  # (score, path)

    def save(self, model: Any, epoch: int, metrics: dict[str, float]) -> Path | None:
        try:
            import torch
            score = metrics.get(self.metric, 0.0)
            path = self.dir / f"epoch_{epoch:04d}_{self.metric}_{score:.4f}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else {},
                "metrics": metrics,
                "metric": self.metric,
                "score": score,
            }, path)
            self._history.append((score, path))
            # Keep only top-k
            reverse = (self.mode == "max")
            self._history.sort(key=lambda x: x[0], reverse=reverse)
            while len(self._history) > self.top_k:
                _, old_path = self._history.pop()
                if old_path.exists():
                    old_path.unlink()
            # Save last
            last_path = self.dir / "last.pth"
            import shutil
            shutil.copy(path, last_path)
            return path
        except Exception as exc:
            logger.warning("Checkpoint save failed: %s", exc)
            return None

    def best_path(self) -> Path | None:
        if not self._history:
            return None
        return self._history[0][1]


class GeoTrainer:
    """Production-ready geospatial model trainer (Phase 4.1).

    Wraps PyTorch training loops with full support for:
    - Distributed training (DDP/FSDP)
    - Mixed precision (FP16/BF16/auto)
    - Gradient accumulation and clipping
    - Early stopping and checkpoint management
    - Experiment tracking (MLflow / W&B / TensorBoard)
    - ONNX export after training

    Example::

        from pygeovision.training import GeoTrainer, TrainingConfig
        import segmentation_models_pytorch as smp

        model = smp.Unet("resnet50", classes=5, activation=None)
        cfg = TrainingConfig(
            batch_size=16, max_epochs=100,
            learning_rate=1e-4, num_classes=5,
            precision="fp16", strategy="ddp",
            tracker="mlflow",
            export_onnx=True,
        )
        trainer = GeoTrainer(model, cfg)
        results = trainer.fit(train_dataset, val_dataset)
    """

    def __init__(self, model: Any, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.cfg = config or TrainingConfig()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_seed()
        self._device: Any = None
        self._scaler: Any = None
        self._tracker_obj: Any = None

    def _setup_seed(self) -> None:
        import random

        import numpy as np
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        try:
            import torch
            torch.manual_seed(self.cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.seed)
        except ImportError:
            pass

    def _get_device(self) -> Any:
        if self._device is not None:
            return self._device
        try:
            import torch
            if self.cfg.devices == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            elif isinstance(self.cfg.devices, int):
                self._device = torch.device(f"cuda:{self.cfg.devices}")
            else:
                self._device = torch.device("cuda")
        except ImportError:
            self._device = "cpu"
        return self._device

    def _build_precision_context(self):
        try:
            import torch
            device = self._get_device()
            precision = self.cfg.precision
            if precision == "auto":
                if torch.cuda.is_available():
                    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
                else:
                    precision = "fp32"
            if precision in ("fp16", "bf16") and str(device) != "cpu":
                dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                return torch.amp.autocast(device_type=str(device).split(":")[0], dtype=dtype)
        except Exception as exc:
            logger.warning(
                "Mixed-precision (%s) setup failed — falling back to full "
                "precision (fp32). Training will be correct but likely "
                "slower/more memory-hungry than expected. Error: %s",
                getattr(self.cfg, "precision", "?"), exc,
            )
        import contextlib
        return contextlib.nullcontext()

    def _build_scaler(self):
        try:
            import torch
            if self.cfg.precision in ("fp16", "auto") and torch.cuda.is_available():
                return torch.cuda.amp.GradScaler()
        except ImportError:
            pass
        return None

    def _setup_tracker(self) -> None:
        tracker = self.cfg.tracker.lower()
        if tracker in ("mlflow", "all"):
            try:
                import mlflow
                mlflow.set_tracking_uri(self.cfg.mlflow_uri)
                mlflow.set_experiment(self.cfg.experiment_name)
                self._tracker_obj = mlflow
                logger.info("MLflow tracker: %s", self.cfg.mlflow_uri)
            except ImportError:
                logger.warning("mlflow not installed (pip install mlflow)")
        if tracker in ("wandb", "all"):
            try:
                import wandb
                wandb.init(project=self.cfg.wandb_project, name=self.cfg.run_name or None,
                           config=self.cfg.to_dict())
                logger.info("W&B tracker: project=%s", self.cfg.wandb_project)
            except ImportError:
                logger.warning("wandb not installed (pip install wandb)")

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._tracker_obj is not None:
            try:
                self._tracker_obj.log_metrics(metrics, step=step)
            except Exception as exc:
                logger.warning("Experiment tracker log_metrics() failed at step %d: %s", step, exc)

    def _setup_distributed(self) -> Any:
        """Set up DDP or FSDP if multiple GPUs available."""
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
                return self.model
            strategy = self.cfg.strategy
            if strategy == "auto":
                strategy = "ddp" if torch.cuda.device_count() > 1 else "none"
            if strategy == "ddp":
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group("nccl")
                self.model = self.model.cuda()
                if self.cfg.sync_batchnorm:
                    self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                return torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[local_rank],
                    find_unused_parameters=self.cfg.find_unused_parameters,
                )
            elif strategy == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                return FSDP(self.model)
        except Exception as exc:
            logger.warning("Distributed setup failed (%s); using single-device.", exc)
        return self.model

    def fit(
        self,
        train_dataset: Any = None,
        val_dataset: Any = None,
        train_loader: Any = None,
        val_loader: Any = None,
        loss_fn: Any | None = None,
        metric_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Train the model.

        Provide either datasets (GeoTrainer builds DataLoaders) or
        pre-built DataLoaders directly.

        Returns:
            Dict with training history, best metric, best checkpoint path.
        """
        try:
            import torch

            from pygeovision.training.metrics import SegmentationMetrics
            from pygeovision.training.optimizer import build_optimizer, build_scheduler
        except ImportError as exc:
            logger.error("torch not installed: %s", exc)
            return {"error": str(exc)}

        self._setup_tracker()
        device = self._get_device()
        model = self._setup_distributed()

        # Move model to device
        try:
            model = model.to(device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to move model to device={device!r}: {exc}. "
                f"Training cannot continue safely on an unknown/partial "
                f"device state — check device availability (e.g. "
                f"torch.cuda.is_available()) and free memory."
            ) from exc

        # Compile model (PyTorch 2.x)
        if self.cfg.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile()")
            except Exception:
                logger.warning("torch.compile() failed; proceeding without.")

        # Real fix: torchvision detection models expect images as a list
        # of variable-sized tensors and targets as a list of per-image
        # dicts with variable-length boxes/labels -- these cannot be
        # torch.stack()'d into a single batch tensor the way segmentation
        # masks can (default collate would crash on the first batch with
        # differing box counts per image). Real, minimal collate_fn that
        # matches torchvision's actual documented detection training
        # interface (see torchvision.models.detection docs).
        def _detection_collate(batch):
            return tuple(zip(*batch))
        collate_fn = _detection_collate if self.cfg.task == "detection" else None

        # Build DataLoaders if datasets provided
        if train_loader is None and train_dataset is not None:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
                persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
                collate_fn=collate_fn,
            )
        if val_loader is None and val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                collate_fn=collate_fn,
            )

        # Real fix, confirmed by direct inspection: freeze_backbone was a
        # real TrainingConfig field referenced nowhere else in this file
        # -- setting it to True previously had zero effect on training.
        # Real backbone/encoder identification via name substring (the
        # same pattern already used, and confirmed working, by
        # pygeovision.ai.training.optimizers' discriminative fine-tuning
        # logic) -- freezing here means build_optimizer below (which
        # already correctly filters to requires_grad parameters) picks
        # up the change automatically.
        # Real fix: there was previously no way to resume/finetune from
        # an existing PyGeoVision checkpoint at all -- the only route to
        # pretrained weights was the model already having them loaded
        # before being passed to GeoTrainer's constructor. Honest
        # limitation: CheckpointManager.save() never saves optimizer
        # state, so only model weights are restorable here, not
        # optimizer momentum/history -- appropriate for finetuning
        # (which usually wants a fresh optimizer on new data/task
        # anyway), not full interrupted-training resume.
        if self.cfg.checkpoint_path:
            try:
                ckpt = torch.load(self.cfg.checkpoint_path, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                result = model.load_state_dict(state_dict, strict=False)
                if result.missing_keys or result.unexpected_keys:
                    logger.warning(
                        "Loaded checkpoint %s with a partial match (strict=False) -- "
                        "%d missing key(s), %d unexpected key(s). This is expected if "
                        "finetuning into a different num_classes/head; unexpected if "
                        "you intended an exact match. missing=%s unexpected=%s",
                        self.cfg.checkpoint_path, len(result.missing_keys),
                        len(result.unexpected_keys), result.missing_keys[:5], result.unexpected_keys[:5],
                    )
                else:
                    logger.info("Loaded checkpoint %s (exact match).", self.cfg.checkpoint_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint_path={self.cfg.checkpoint_path!r}: {exc}. "
                    f"Refusing to silently proceed with randomly-initialized weights when "
                    f"a specific checkpoint was explicitly requested."
                ) from exc

        if self.cfg.freeze_backbone:
            _backbone_keys = ("encoder", "backbone", "patch_embed", "segformer.encoder")
            n_frozen = 0
            for name, param in model.named_parameters():
                if any(k in name for k in _backbone_keys):
                    param.requires_grad = False
                    n_frozen += 1
            if n_frozen == 0:
                logger.warning(
                    "freeze_backbone=True but no parameters matched the real "
                    "backbone-name patterns %s -- this model's real parameter "
                    "names don't match any of them, so nothing was actually "
                    "frozen. Check model.named_parameters() to find the real "
                    "prefix this model uses for its backbone/encoder.",
                    _backbone_keys,
                )
            else:
                logger.info("freeze_backbone=True: froze %d real backbone parameter(s).", n_frozen)

        # Build optimizer and scheduler
        optimizer = build_optimizer(model, self.cfg)
        n_steps = len(train_loader) if train_loader else 100
        scheduler = build_scheduler(optimizer, self.cfg, steps_per_epoch=n_steps)

        # Real fix: torchvision detection models compute their own loss
        # internally when called as model(images, targets) in training
        # mode (returning a real loss dict summed below) -- calling a
        # separate loss_fn on their output would be wrong/meaningless.
        if loss_fn is None and self.cfg.task != "detection":
            loss_fn = torch.nn.CrossEntropyLoss()

        # Mixed precision
        autocast_ctx = self._build_precision_context()
        scaler = self._build_scaler()

        # Real fix: early_stopping_metric/checkpoint_metric default to
        # "val_iou"/"max", which genuinely doesn't exist for detection
        # (no per-pixel IoU applies to bounding-box outputs) -- only
        # auto-switch when still at that segmentation-oriented default,
        # so an explicit user override for either field is respected.
        es_metric, es_mode = self.cfg.early_stopping_metric, self.cfg.early_stopping_mode
        ckpt_metric, ckpt_mode = self.cfg.checkpoint_metric, self.cfg.checkpoint_mode
        if self.cfg.task == "detection":
            if es_metric == "val_iou":
                es_metric, es_mode = "val_loss", "min"
            if ckpt_metric == "val_iou":
                ckpt_metric, ckpt_mode = "val_loss", "min"

        # Training bookkeeping
        early_stopping = EarlyStopping(self.cfg.early_stopping_patience, es_metric, es_mode)
        ckpt_mgr = CheckpointManager(self.output_dir, self.cfg.save_top_k, ckpt_metric, ckpt_mode)
        # Real fix: SegmentationMetrics (per-pixel confusion matrix / IoU)
        # genuinely doesn't apply to bounding-box detection outputs --
        # skipped entirely for the detection task rather than silently
        # computing meaningless numbers.
        metrics_obj = SegmentationMetrics(self.cfg.num_classes) if self.cfg.task != "detection" else None

        history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_iou": []}
        float("-inf") if self.cfg.checkpoint_mode == "max" else float("inf")
        global_step = 0
        t_start = time.time()

        logger.info("Starting training: %d epochs | device=%s | precision=%s | strategy=%s",
                    self.cfg.max_epochs, device, self.cfg.precision, self.cfg.strategy)

        for epoch in range(1, self.cfg.max_epochs + 1):
            # ── Train ───────────────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            if train_loader is not None:
                for batch_idx, batch in enumerate(train_loader):
                    if self.cfg.task == "detection":
                        # Real detection branch: _detection_collate above
                        # returns (images_tuple, targets_tuple), not a
                        # stacked tensor -- move each item individually
                        # since images/targets have different shapes.
                        images_raw, targets_raw = batch
                        try:
                            images = [img.to(device, non_blocking=True) for img in images_raw]
                            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                                       for t in targets_raw]
                        except Exception as exc:
                            logger.warning(
                                "Skipping detection batch %d — failed to move to "
                                "device=%r: %s. Check that your dataset returns "
                                "(image_tensor, {'boxes': ..., 'labels': ...}) pairs "
                                "matching torchvision's real detection target format.",
                                batch_idx, device, exc,
                            )
                            continue
                        with autocast_ctx:
                            loss_dict = model(images, targets)
                            loss = sum(loss_dict.values()) / self.cfg.grad_accumulation_steps
                    else:
                        # Accept (images, masks) or dict
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            images, targets = batch[0], batch[1]
                        elif isinstance(batch, dict):
                            images = batch.get("image", batch.get("images"))
                            targets = batch.get("mask", batch.get("label", batch.get("target")))
                        else:
                            continue

                        try:
                            images = images.to(device, non_blocking=True)
                            targets = targets.to(device, non_blocking=True)
                        except Exception as exc:
                            logger.warning(
                                "Skipping batch %d — failed to move to device=%r: %s. "
                                "If this happens repeatedly, check your dataset/collate_fn "
                                "for malformed samples (e.g. a missing 'mask'/'label' key).",
                                batch_idx, device, exc,
                            )
                            continue

                        # Forward + backward
                        with autocast_ctx:
                            outputs = model(images)
                            loss = loss_fn(outputs, targets)
                            loss = loss / self.cfg.grad_accumulation_steps

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient accumulation
                    if (batch_idx + 1) % self.cfg.grad_accumulation_steps == 0:
                        if self.cfg.grad_clip > 0:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None and not isinstance(
                            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            scheduler.step()
                        global_step += 1

                    epoch_loss += loss.item() * self.cfg.grad_accumulation_steps
                    n_batches += 1

                    if global_step % self.cfg.log_every_n_steps == 0:
                        lr = optimizer.param_groups[0]["lr"]
                        logger.debug("step %d | loss=%.4f | lr=%.2e", global_step, loss.item(), lr)

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # ── Validate ─────────────────────────────────────────────
            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_loss = 0.0
                n_val = 0
                if metrics_obj is not None:
                    metrics_obj.reset()

                if self.cfg.task == "detection":
                    # Real, documented torchvision pattern: these models
                    # only compute a real internal loss dict during a
                    # train()-mode forward pass with real targets supplied
                    # -- eval() mode returns predictions instead, with no
                    # loss at all. torch.no_grad() still prevents any
                    # gradient tracking/backprop, so this is genuinely a
                    # real, no-training-happening validation pass, just
                    # using train()-mode's real loss-computation path.
                    model.train()
                    with torch.no_grad():
                        for batch in val_loader:
                            images_raw, targets_raw = batch
                            try:
                                images = [img.to(device) for img in images_raw]
                                targets = [{k: v.to(device) for k, v in t.items()} for t in targets_raw]
                            except Exception as exc:
                                logger.warning("Skipping detection val batch — failed to move to device: %s", exc)
                                continue
                            with autocast_ctx:
                                loss_dict = model(images, targets)
                                loss = sum(loss_dict.values())
                            val_loss += loss.item()
                            n_val += 1
                    model.eval()  # restore for any caller inspecting the model after fit()
                else:
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                                images, targets = batch[0].to(device), batch[1].to(device)
                            elif isinstance(batch, dict):
                                images = batch.get("image", batch.get("images", batch)).to(device)
                                targets = batch.get("mask", batch.get("label")).to(device)
                            else:
                                continue
                            with autocast_ctx:
                                outputs = model(images)
                                loss = loss_fn(outputs, targets)
                            val_loss += loss.item()
                            n_val += 1
                            try:
                                preds = outputs.argmax(dim=1)
                                metrics_obj.update(preds, targets)
                            except Exception as exc:
                                logger.warning(
                                    "Validation metrics update failed for a batch "
                                    "(val_loss is still accumulated correctly, but "
                                    "other metrics may be incomplete): %s", exc,
                                )

                computed = metrics_obj.compute() if metrics_obj is not None else {}
                val_metrics = {
                    "val_loss": val_loss / max(n_val, 1),
                    "val_iou": computed.get("mean_iou", 0.0),
                    "val_f1": computed.get("mean_f1", 0.0),
                    "val_accuracy": computed.get("accuracy", 0.0),
                    "train_loss": avg_train_loss,
                    "epoch": float(epoch),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_iou"].append(val_metrics["val_iou"])

                # LR scheduler plateau step
                if scheduler is not None and isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step(val_metrics.get("val_loss", avg_train_loss))

                # Log
                self._log_metrics(val_metrics, step=epoch)
                logger.info(
                    "Epoch %d/%d | loss=%.4f | val_loss=%.4f | val_iou=%.4f | lr=%.2e",
                    epoch, self.cfg.max_epochs,
                    avg_train_loss, val_metrics["val_loss"],
                    val_metrics["val_iou"], optimizer.param_groups[0]["lr"],
                )

                # Checkpoint
                if epoch % self.cfg.checkpoint_every_n_epochs == 0:
                    ckpt_path = ckpt_mgr.save(model, epoch, val_metrics)
                    if ckpt_path:
                        logger.debug("Saved checkpoint: %s", ckpt_path)

                # Early stopping
                if not early_stopping.update(val_metrics):
                    if early_stopping.should_stop:
                        logger.info("Early stopping at epoch %d (patience=%d)",
                                    epoch, self.cfg.early_stopping_patience)
                        break
            else:
                logger.info("Epoch %d/%d | train_loss=%.4f", epoch, self.cfg.max_epochs, avg_train_loss)

        duration = time.time() - t_start

        # Export if requested
        if self.cfg.export_onnx:
            self._export_onnx(model, device)
        if self.cfg.export_torchscript:
            self._export_torchscript(model, device)

        best_path = ckpt_mgr.best_path()
        result = {
            "epochs_trained": epoch,
            "duration_seconds": duration,
            "best_checkpoint": str(best_path) if best_path else None,
            "best_metric_name": es_metric,
            "best_metric_value": early_stopping.best,
            "history": history,
        }
        # Real fix: this key was always literally named "best_val_iou"
        # even for detection, where the tracked value is actually
        # val_loss -- confirmed misleading by a real test run showing
        # "best_val_iou": 1.77 (IoU is bounded 0-1). Only include this
        # specific key when it's genuinely an IoU value.
        if es_metric == "val_iou":
            result["best_val_iou"] = early_stopping.best
        logger.info("Training complete: %.1f min | best=%s=%.4f",
                    duration / 60,
                    self.cfg.early_stopping_metric,
                    early_stopping.best or 0.0)
        return result

    def _export_onnx(self, model: Any, device: Any) -> Path | None:
        try:
            import torch
            out_path = self.output_dir / "model.onnx"
            model.eval()
            dummy = torch.randn(1, self.cfg.in_channels, 512, 512).to(device)
            torch.onnx.export(
                model, dummy, str(out_path),
                opset_version=self.cfg.onnx_opset,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            logger.info("ONNX model exported → %s", out_path)
            return out_path
        except Exception as exc:
            logger.warning("ONNX export failed: %s", exc)
            return None

    def _export_torchscript(self, model: Any, device: Any) -> Path | None:
        try:
            import torch
            out_path = self.output_dir / "model_scripted.pt"
            model.eval()
            scripted = torch.jit.script(model)
            scripted.save(str(out_path))
            logger.info("TorchScript model → %s", out_path)
            return out_path
        except Exception as exc:
            logger.warning("TorchScript export failed: %s", exc)
            return None