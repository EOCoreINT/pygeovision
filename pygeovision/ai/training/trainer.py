"""
GeoTrainer — distributed geospatial model training (Phase 4.1).

Supports:
  • Multi-GPU DDP + FSDP
  • Mixed precision (FP16, BF16, auto)
  • Gradient accumulation + checkpointing
  • Hyperparameter optimization (Optuna, Ray Tune)
  • Experiment tracking (MLflow, W&B, TensorBoard)
  • Early stopping and LR scheduling
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


class _AugmentedSegmentationDataset:
    """Real fix, confirmed necessary by direct inspection: `augment`/
    `flip_prob`/`rotate_limit`/`scale_limit`/`brightness_limit`/
    `contrast_limit` were real TrainingConfig fields with zero
    references anywhere in this file before this fix -- setting
    `augment=True` had no effect on training at all, despite a real,
    substantial, already-built augmentation module
    (`pygeovision.ai.data.augmentations.GeoAugmentationPipeline`)
    existing elsewhere in this codebase, just never wired in here.

    Wraps a real segmentation dataset, applying that real pipeline to
    each sample. Assumes the standard PyTorch segmentation-dataset
    convention: `dataset[i]` returns a real `(image, mask)` tuple,
    `image` as a real `(C, H, W)` tensor/array. Scoped to segmentation
    only -- detection samples have a genuinely different real format
    (boxes/labels, not a mask) that the real augmentation pipeline
    isn't built for; `augment=True` has no effect for
    `task="detection"`, logged once at construction.
    """

    def __init__(self, dataset: Any, mode: str = "medium") -> None:
        from pygeovision.ai.data.augmentations import GeoAugmentationPipeline
        self._dataset = dataset
        # The band count decides which radiometric transforms are valid
        # (hue/saturation and CLAHE exist only for 1- and 3-channel images).
        image, _ = dataset[0]
        n_channels = image.shape[0] if getattr(image, "ndim", 0) == 3 else None
        self._pipeline = GeoAugmentationPipeline.for_task(
            "segmentation", mode=mode, n_channels=n_channels)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        import numpy as np
        import torch

        image, mask = self._dataset[idx]
        image_np = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
        mask_np = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        # CHW -> HWC for albumentations' real, documented convention
        image_hwc = np.transpose(image_np, (1, 2, 0)) if image_np.ndim == 3 else image_np

        result = self._pipeline(image=image_hwc, mask=mask_np)
        aug_image, aug_mask = result["image"], result["mask"]
        aug_image_chw = np.transpose(aug_image, (2, 0, 1)) if aug_image.ndim == 3 else aug_image

        return torch.as_tensor(aug_image_chw.copy()).float(), torch.as_tensor(aug_mask.copy()).long()


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
    #: NOT used by GeoTrainer, which takes explicit train and validation sets.
    #: Split the data yourself (see ``make_geo_loaders(split=...)``); a warning
    #: is logged if you change this value.
    train_val_split: float = 0.8

    # ── Model ─────────────────────────────────────────────────────────
    task: str = "segmentation"  # segmentation | detection
    num_classes: int = 2
    in_channels: int = 3
    pretrained: bool = True
    freeze_backbone: bool = False
    checkpoint_path: str | None = None  # real path to an existing .pth to finetune from
    #: Train the backbone only after the first N epochs (the head trains alone
    #: before that). ``None`` keeps ``freeze_backbone`` semantics (all or nothing).
    unfreeze_backbone_epoch: int | None = None
    #: Backbone learning rate = ``learning_rate * backbone_lr_scale`` (e.g. 0.1
    #: to fine-tune a pretrained encoder gently while the head learns faster).
    backbone_lr_scale: float = 1.0

    # ── Optimisation ──────────────────────────────────────────────────
    optimizer: str = "adamw"          # adamw | adam | sgd | lion (needs lion-pytorch)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = True
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler: str = "cosine"         # cosine | cosine_warmup | warmup_cosine | linear | step | plateau | onecycle | none
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
    #: Continue an interrupted run from ``<output_dir>/checkpoints/last.pth``
    #: (model, optimizer, scheduler, history and early-stopping state).
    resume: bool = False

    # ── Augmentation ──────────────────────────────────────────────────
    augment: bool = True              # segmentation only; needs albumentations
    #: Strength of the training augmentation: light | medium | heavy |
    #: geometric_only | radiometric_only. Values are assumed to lie in [0, 1].
    augment_mode: str = "medium"
    #: Accepted for backward compatibility, NOT used: the augmentation
    #: probabilities are fixed per ``augment_mode``. A warning is logged if
    #: you change any of them.
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
        import dataclasses
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            logger.warning("Ignoring unknown TrainingConfig keys in %s: %s", path, unknown)
        return cls(**{k: v for k, v in d.items() if k in known})

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


_BACKBONE_KEYS = ("encoder", "backbone", "patch_embed", "segformer.encoder")


def _is_backbone(param_name: str) -> bool:
    return any(k in param_name for k in _BACKBONE_KEYS)


def _unwrap(model: Any) -> Any:
    """The plain module under DDP / DataParallel / torch.compile wrappers.

    State dicts saved from a wrapped model carry ``module.`` / ``_orig_mod.``
    key prefixes that a plain model cannot load, so checkpoints are always
    written from the unwrapped module.

    Only real ``nn.Module`` wrappers are unwrapped, by looking at their
    registered submodules (never ``hasattr``, which a mock or any object with a
    dynamic ``__getattr__`` answers "yes" to forever), and the walk is bounded.
    """
    try:
        import torch
    except ImportError:
        return model
    for _ in range(8):
        if not isinstance(model, torch.nn.Module):
            return model
        inner = model._modules.get("_orig_mod")                       # torch.compile
        if inner is None and type(model).__name__ in ("DistributedDataParallel", "DataParallel"):
            inner = model._modules.get("module")
        if inner is None:
            return model
        model = inner
    return model


class CheckpointManager:
    """Best-k model checkpoints plus a full-state ``last.pth`` for resuming.

    * ``epoch_XXXX_<metric>_<score>.pth``: weights of the best ``top_k`` epochs.
    * ``best.pth``: copy of the single best one (a stable name for inference).
    * ``last.pth``: the complete training state after the most recent epoch,
      written every epoch (atomically) whether or not it was a top-k epoch.
    """

    def __init__(self, output_dir: Path, top_k: int = 3, metric: str = "val_iou", mode: str = "max") -> None:
        self.dir = Path(output_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.metric = metric
        self.mode = mode
        self._history: list[tuple] = []  # (score, path), best first

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pth"

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def save(self, model: Any, epoch: int, metrics: dict[str, float]) -> Path | None:
        """Write this epoch's weights if it is in the best ``top_k``; else return None."""
        if self.top_k <= 0:
            return None
        try:
            import torch
            score = float(metrics.get(self.metric, 0.0))
            if len(self._history) >= self.top_k and not self._better(score, self._history[-1][0]):
                return None
            model = _unwrap(model)
            path = self.dir / f"epoch_{epoch:04d}_{self.metric}_{score:.4f}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else {},
                "metrics": metrics,
                "metric": self.metric,
                "score": score,
            }, path)
            self._history.append((score, path))
            self._history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
            while len(self._history) > self.top_k:
                _, old_path = self._history.pop()
                if old_path.exists():
                    old_path.unlink()
            if self._history[0][1] == path:
                import shutil
                shutil.copy(path, self.dir / "best.pth")
            return path
        except Exception as exc:
            logger.warning("Checkpoint save failed: %s", exc)
            return None

    def save_state(self, state: dict[str, Any]) -> Path | None:
        """Atomically write the full training state to ``last.pth``."""
        try:
            import torch
            tmp = self.dir / "last.pth.tmp"
            torch.save(state, tmp)
            os.replace(tmp, self.last_path)
            return self.last_path
        except Exception as exc:
            logger.warning("Saving last.pth failed: %s", exc)
            return None

    def best_path(self) -> Path | None:
        if not self._history:
            return None
        return self._history[0][1]

    def state(self) -> list[tuple[float, str]]:
        return [(score, str(path)) for score, path in self._history]

    def load_state(self, entries: list[Any]) -> None:
        """Restore the best-k list after a resume (files that no longer exist are dropped)."""
        self._history = [(float(sc), Path(pth)) for sc, pth in entries if Path(pth).exists()]
        self._history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))


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
            elif isinstance(self.cfg.devices, str):
                # "cpu", "cuda", "cuda:1", "mps" -- previously every string
                # (including "cpu") silently selected CUDA.
                self._device = torch.device(self.cfg.devices)
            elif isinstance(self.cfg.devices, (list, tuple)) and self.cfg.devices:
                self._device = torch.device(f"cuda:{self.cfg.devices[0]}")
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
        """Start the experiment trackers named by ``cfg.tracker`` (mlflow, wandb, tensorboard, all, none).

        Several can be combined with commas, e.g. ``"mlflow,tensorboard"``. A missing library logs a
        warning and that backend is skipped; training itself is never stopped by tracking.
        """
        tracker = (self.cfg.tracker or "none").lower().replace(" ", "")
        if tracker in ("", "none"):
            return
        known = ("mlflow", "wandb", "tensorboard")
        backends = list(known) if tracker == "all" else [t for t in tracker.split(",") if t]
        unknown = [t for t in backends if t not in known]
        if unknown:
            logger.warning("Unknown tracker(s) %s ignored; choose from %s, 'all' or 'none'.", unknown, known)
        backends = [t for t in backends if t in known]
        if not backends:
            return
        try:
            from pygeovision.training.experiment import ExperimentTracker
            self._tracker_obj = ExperimentTracker(
                self.cfg.experiment_name, run_name=self.cfg.run_name, backends=backends,
                mlflow_uri=self.cfg.mlflow_uri, wandb_project=self.cfg.wandb_project,
                tb_log_dir=str(self.output_dir / "tb_logs"),
            ).start_run(self.cfg.to_dict())
            active = getattr(self._tracker_obj, "active_backends", backends)
            if active:
                logger.info("Experiment tracking: %s", ", ".join(active))
            else:
                logger.warning("Experiment tracking requested (%s) but no backend could start; "
                               "continuing without it.", ", ".join(backends))
                self._tracker_obj = None
        except Exception as exc:  # tracking must never stop training
            logger.warning("Experiment tracking could not start (%s); continuing without it.", exc)
            self._tracker_obj = None

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._tracker_obj is not None:
            numeric = {k: float(v) for k, v in metrics.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            try:
                self._tracker_obj.log(numeric, step=step)
            except Exception as exc:
                logger.warning("Experiment tracker log() failed at step %d: %s", step, exc)

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
            else:
                logger.warning("strategy=%r is not implemented (use 'auto', 'ddp' or 'fsdp'); "
                               "training on a single device.", strategy)
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
            from pygeovision.training.optimizer import (
                build_optimizer,
                build_scheduler,
                scheduler_steps_per_batch,
            )
        except ImportError as exc:
            logger.error("torch not installed: %s", exc)
            return {"error": str(exc)}

        if metric_fn is not None:
            logger.warning(
                "fit(metric_fn=...) is not used: validation always reports val_loss, val_iou, "
                "val_f1, val_accuracy and the per-class values (val_map50 for detection). "
                "To track your own score, evaluate best.pth after training.")
        if self.cfg.train_val_split != TrainingConfig().train_val_split:
            logger.warning(
                "TrainingConfig.train_val_split=%s has no effect: GeoTrainer trains on the "
                "train set and validates on the val set you pass to fit().",
                self.cfg.train_val_split)
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
        if train_dataset is not None and self.cfg.augment:
            if self.cfg.task == "segmentation":
                _defaults = TrainingConfig()
                _ignored = [n for n in ("flip_prob", "rotate_limit", "scale_limit",
                                        "brightness_limit", "contrast_limit")
                            if getattr(self.cfg, n) != getattr(_defaults, n)]
                if _ignored:
                    logger.warning(
                        "TrainingConfig fields %s have no effect: augmentation "
                        "strength is chosen with augment_mode (%r).",
                        _ignored, self.cfg.augment_mode)
                train_dataset = _AugmentedSegmentationDataset(
                    train_dataset, mode=self.cfg.augment_mode)
            else:
                logger.warning(
                    "cfg.augment=True has no effect for task=%r -- the real "
                    "augmentation pipeline (GeoAugmentationPipeline) is only "
                    "wired in for segmentation, since detection samples have "
                    "a genuinely different real format (boxes/labels, not a "
                    "mask) it isn't built for.", self.cfg.task,
                )
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

        # ── Metrics, checkpointing and early-stopping objects ─────────────
        # (built before the optimizer so a resume can restore their state)
        #
        # Detection's default early-stopping/checkpoint metric is "val_map50"/"max"
        # (per-pixel IoU does not exist for boxes). Only auto-switch while the
        # segmentation default "val_iou" is still in place, so an explicit user
        # choice (including "val_loss") is respected.
        es_metric, es_mode = self.cfg.early_stopping_metric, self.cfg.early_stopping_mode
        ckpt_metric, ckpt_mode = self.cfg.checkpoint_metric, self.cfg.checkpoint_mode
        if self.cfg.task == "detection":
            if es_metric == "val_iou":
                es_metric, es_mode = "val_map50", "max"
            if ckpt_metric == "val_iou":
                ckpt_metric, ckpt_mode = "val_map50", "max"

        early_stopping = EarlyStopping(self.cfg.early_stopping_patience, es_metric, es_mode)
        ckpt_mgr = CheckpointManager(self.output_dir, self.cfg.save_top_k, ckpt_metric, ckpt_mode)
        # SegmentationMetrics (per-pixel confusion matrix / IoU) does not apply to
        # bounding-box detection outputs, so it is skipped for the detection task.
        metrics_obj = SegmentationMetrics(self.cfg.num_classes) if self.cfg.task != "detection" else None
        history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_iou": [], "val_map50": []}
        global_step = 0

        # ── Resume an interrupted run ──────────────────────────────────────
        start_epoch = 0
        resume_state: dict[str, Any] | None = None
        if self.cfg.resume:
            if ckpt_mgr.last_path.exists():
                try:
                    resume_state = torch.load(ckpt_mgr.last_path, map_location="cpu", weights_only=True)
                    _unwrap(model).load_state_dict(resume_state["model_state_dict"])
                except Exception as exc:
                    raise RuntimeError(
                        f"resume=True but {ckpt_mgr.last_path} could not be restored into this "
                        f"model: {exc}. Refusing to continue from different weights. Delete the "
                        f"file (or set resume=False) to start a fresh run."
                    ) from exc
                if "optimizer_state_dict" not in resume_state:
                    logger.warning("%s holds weights only (no optimizer state); starting epoch 1 "
                                   "from those weights.", ckpt_mgr.last_path)
                    resume_state = None
                else:
                    start_epoch = int(resume_state["epoch"])
                    history = {k: list(v) for k, v in resume_state.get("history", history).items()}
                    global_step = int(resume_state.get("global_step", 0))
                    logger.info("Resuming from %s after epoch %d.", ckpt_mgr.last_path, start_epoch)
            else:
                logger.info("resume=True but %s does not exist; starting a fresh run.", ckpt_mgr.last_path)

        # ── Freezing (all-or-nothing, or staged) ───────────────────────────
        # freeze_backbone=True freezes the encoder for the whole run.
        # unfreeze_backbone_epoch=N freezes it for the first N epochs only.
        staged = self.cfg.unfreeze_backbone_epoch is not None
        n_unfreeze = int(self.cfg.unfreeze_backbone_epoch or 0)
        freeze_now = (self.cfg.freeze_backbone or staged) and not (staged and start_epoch >= n_unfreeze)
        if self.cfg.freeze_backbone or staged:
            n_backbone = sum(1 for n, _ in model.named_parameters() if _is_backbone(n))
            if n_backbone == 0:
                logger.warning(
                    "Backbone freezing requested but no parameters matched the backbone-name "
                    "patterns %s, so nothing was frozen. Check model.named_parameters() for "
                    "this architecture's encoder prefix.", _BACKBONE_KEYS,
                )
            elif freeze_now:
                for n, prm in model.named_parameters():
                    if _is_backbone(n):
                        prm.requires_grad = False
                logger.info("Froze %d backbone parameter tensor(s)%s.", n_backbone,
                            f" for the first {n_unfreeze} epoch(s)" if staged else "")

        # ── Optimizer and scheduler ────────────────────────────────────────
        # A separate backbone learning rate (or a later unfreeze) needs the
        # backbone in the optimizer from the start: PyTorch skips parameters
        # without gradients, and a parameter missing from the optimizer would
        # stay untrained even after it is unfrozen.
        has_backbone = any(_is_backbone(n) for n, _ in model.named_parameters())
        if has_backbone and (staged or self.cfg.backbone_lr_scale != 1.0):
            head = [prm for n, prm in model.named_parameters() if not _is_backbone(n) and prm.requires_grad]
            groups = [{"params": head, "lr": self.cfg.learning_rate}]
            if not (self.cfg.freeze_backbone and not staged):
                backbone = [prm for n, prm in model.named_parameters() if _is_backbone(n)]
                groups.append({"params": backbone, "lr": self.cfg.learning_rate * self.cfg.backbone_lr_scale})
            optimizer = build_optimizer(model, self.cfg, param_groups=groups)
        else:
            optimizer = build_optimizer(model, self.cfg)
        try:
            n_loader = len(train_loader)
        except TypeError:
            n_loader = None
        accum = max(1, int(self.cfg.grad_accumulation_steps))
        steps_per_epoch = -(-n_loader // accum) if n_loader else 100
        scheduler = build_scheduler(optimizer, self.cfg, steps_per_epoch=steps_per_epoch)
        per_batch_scheduler = scheduler_steps_per_batch(self.cfg.scheduler)

        # Real fix: torchvision detection models compute their own loss
        # internally when called as model(images, targets) in training
        # mode (returning a real loss dict summed below) -- calling a
        # separate loss_fn on their output would be wrong/meaningless.
        if loss_fn is None and self.cfg.task != "detection":
            # 255 is the package-wide nodata value in label rasters; it is never a class.
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

        # Mixed precision
        autocast_ctx = self._build_precision_context()
        scaler = self._build_scaler()

        if resume_state is not None:
            saved_opt = resume_state.get("optimizer_class")
            if saved_opt is not None and saved_opt != type(optimizer).__name__:
                raise RuntimeError(
                    f"Cannot resume from {ckpt_mgr.last_path}: it was written with the {saved_opt} optimizer "
                    f"but this run builds {type(optimizer).__name__}. Use the original optimizer to resume, "
                    f"or resume=False (with checkpoint_path=<best.pth> to keep the weights).")
            try:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                if scheduler is not None and resume_state.get("scheduler_state_dict"):
                    scheduler.load_state_dict(resume_state["scheduler_state_dict"])
                if scaler is not None and resume_state.get("scaler_state_dict"):
                    scaler.load_state_dict(resume_state["scaler_state_dict"])
            except Exception as exc:
                raise RuntimeError(
                    f"Could not restore optimizer/scheduler state from {ckpt_mgr.last_path}: {exc}. "
                    f"The optimizer layout differs from the interrupted run (changed "
                    f"unfreeze_backbone_epoch / backbone_lr_scale / optimizer / scheduler?). "
                    f"Use the original settings to resume, or resume=False to start fresh."
                ) from exc
            saved_max = resume_state.get("max_epochs")
            if saved_max is not None and saved_max != self.cfg.max_epochs:
                logger.warning(
                    "Resuming with max_epochs=%d but the saved run used %d: the learning-rate schedule "
                    "restored from last.pth was laid out for %d epochs, so the rate will not follow a "
                    "fresh %d-epoch curve. To train longer, start a new run with "
                    "checkpoint_path=<best.pth> instead.",
                    self.cfg.max_epochs, saved_max, saved_max, self.cfg.max_epochs)
            es_state = resume_state.get("early_stopping", {})
            early_stopping.best = es_state.get("best")
            early_stopping.counter = int(es_state.get("counter", 0))
            early_stopping.should_stop = bool(es_state.get("should_stop", False))
            ckpt_mgr.load_state(resume_state.get("checkpoints", []))

        t_start = time.time()

        logger.info("Starting training: %d epochs | device=%s | precision=%s | strategy=%s",
                    self.cfg.max_epochs, device, self.cfg.precision, self.cfg.strategy)

        epoch = start_epoch
        for epoch in range(start_epoch + 1, self.cfg.max_epochs + 1):
            # Staged fine-tuning: the backbone joins in after the first N epochs.
            if staged and freeze_now and epoch > n_unfreeze:
                for n, prm in model.named_parameters():
                    if _is_backbone(n):
                        prm.requires_grad = True
                freeze_now = False
                logger.info("Epoch %d: backbone unfrozen (learning rate x%g).",
                            epoch, self.cfg.backbone_lr_scale)

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

                    # Gradient accumulation (the last, partial window is flushed too)
                    if (batch_idx + 1) % accum == 0 or (n_loader is not None and batch_idx + 1 == n_loader):
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
                        # Only step-based schedules advance per optimizer step; epoch-based
                        # ones (cosine, step) advance once per epoch, further below.
                        if scheduler is not None and per_batch_scheduler:
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
                    #
                    # Real fix, addressing a confirmed gap: DetectionMetrics
                    # (real torchmetrics-backed mAP) existed elsewhere in
                    # this codebase, independently verified correct, but
                    # was never actually called here -- validation only
                    # ever computed val_loss, with no real accuracy signal
                    # available for early stopping or checkpoint selection.
                    # A second, eval()-mode forward pass on the same batch
                    # gets real predictions to feed it (the model can't
                    # return both a loss dict and predictions from one
                    # forward call), at the real cost of double compute
                    # during detection validation specifically.
                    from pygeovision.training.metrics import DetectionMetrics
                    det_metrics = DetectionMetrics(num_classes=self.cfg.num_classes)
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

                            model.eval()
                            try:
                                preds = model(images)
                                det_metrics.update(
                                    [{k: v.detach().cpu() for k, v in p.items()} for p in preds],
                                    [{k: v.detach().cpu() for k, v in t.items()} for t in targets],
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Skipping real mAP update for a detection val batch "
                                    "(val_loss is still accumulated correctly): %s", exc,
                                )
                            model.train()
                    model.eval()  # restore for any caller inspecting the model after fit()
                    det_computed = det_metrics.compute()
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
                # Per-class numbers (val_iou_class_1, val_f1_class_1, val_recall_class_1, ...)
                # so a rare class can drive early stopping / checkpoint selection.
                for key, value in computed.items():
                    if key not in ("mean_iou", "mean_f1", "accuracy"):
                        val_metrics[f"val_{key}"] = value
                # Real fix: merge in the real, newly-computed mAP metrics
                # for detection tasks -- det_computed only exists when
                # self.cfg.task == "detection" (set in the branch above).
                if self.cfg.task == "detection":
                    val_metrics["val_map50"] = det_computed.get("mAP50", 0.0)
                    val_metrics["val_map50_95"] = det_computed.get("mAP50_95", 0.0)
                    if "note" in det_computed:
                        val_metrics["_map_note"] = det_computed["note"]
                # A metric name that validation does not produce would read as 0.0 forever and
                # end the run after `patience` epochs for no visible reason: fail loudly instead.
                for _what, _name in (("early_stopping_metric", es_metric), ("checkpoint_metric", ckpt_metric)):
                    if _name not in val_metrics:
                        raise ValueError(
                            f"{_what}={_name!r} is not produced by validation. Available: "
                            f"{sorted(k for k, v in val_metrics.items() if isinstance(v, (int, float)))}")
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_iou"].append(val_metrics["val_iou"])
                history.setdefault("lr", []).append(float(val_metrics["lr"]))
                if self.cfg.task == "detection":
                    history["val_map50"].append(val_metrics["val_map50"])
                for key, value in val_metrics.items():
                    if key.startswith("val_") and "_class_" in key:
                        history.setdefault(key, []).append(value)

                # LR scheduler plateau step
                if scheduler is not None and isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step(val_metrics.get("val_loss", avg_train_loss))

                # Log
                self._log_metrics(val_metrics, step=epoch)
                _shown = "val_map50" if self.cfg.task == "detection" else "val_iou"
                logger.info(
                    "Epoch %d/%d | loss=%.4f | val_loss=%.4f | %s=%.4f | lr=%.2e",
                    epoch, self.cfg.max_epochs,
                    avg_train_loss, val_metrics["val_loss"],
                    _shown, val_metrics.get(_shown, 0.0), optimizer.param_groups[0]["lr"],
                )

                # Early stopping bookkeeping (the break happens after this epoch's state is saved)
                early_stopping.update(val_metrics)
            else:
                val_metrics = {}
                logger.info("Epoch %d/%d | train_loss=%.4f", epoch, self.cfg.max_epochs, avg_train_loss)

            # Epoch-based schedules (cosine, step) advance once per epoch.
            if (scheduler is not None and not per_batch_scheduler
                    and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step()

            # Best-k weights (needs validation metrics)
            if val_loader is not None and epoch % self.cfg.checkpoint_every_n_epochs == 0:
                ckpt_path = ckpt_mgr.save(model, epoch, val_metrics)
                if ckpt_path:
                    logger.debug("Saved checkpoint: %s", ckpt_path)

            # Full state of the finished epoch, for resume=True (written atomically)
            if self.cfg.save_last:
                ckpt_mgr.save_state({
                    "epoch": epoch,
                    "model_state_dict": _unwrap(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_class": type(optimizer).__name__,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "history": history,
                    "global_step": global_step,
                    "early_stopping": {"best": early_stopping.best, "counter": early_stopping.counter,
                                       "should_stop": early_stopping.should_stop},
                    "checkpoints": ckpt_mgr.state(),
                    "metrics": val_metrics,
                    "max_epochs": self.cfg.max_epochs,
                })

            if val_loader is not None and early_stopping.should_stop:
                logger.info("Early stopping at epoch %d (patience=%d)",
                            epoch, self.cfg.early_stopping_patience)
                break

        duration = time.time() - t_start

        if self._tracker_obj is not None:
            try:
                self._tracker_obj.end_run()
            except Exception as exc:
                logger.warning("Experiment tracker end_run() failed: %s", exc)

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
            "last_checkpoint": str(ckpt_mgr.last_path) if ckpt_mgr.last_path.exists() else None,
            "resumed_from_epoch": start_epoch or None,
        }
        # Real fix: this key was always literally named "best_val_iou"
        # even for detection, where the tracked value is actually
        # val_loss -- confirmed misleading by a real test run showing
        # "best_val_iou": 1.77 (IoU is bounded 0-1). Only include this
        # specific key when it's genuinely an IoU value.
        if es_metric == "val_iou":
            result["best_val_iou"] = early_stopping.best
        # Matching, honest convenience alias for the newly-available
        # real mAP metric.
        if es_metric == "val_map50":
            result["best_val_map50"] = early_stopping.best
        logger.info("Training complete: %.1f min | best=%s=%.4f",
                    duration / 60,
                    es_metric,
                    early_stopping.best or 0.0)
        return result

    def _export_onnx(self, model: Any, device: Any) -> Path | None:
        try:
            import torch
            out_path = self.output_dir / "model.onnx"
            from pygeovision._onnx import export_onnx
            model = _unwrap(model)
            model.eval()
            dummy = torch.randn(1, self.cfg.in_channels, 512, 512).to(device)
            export_onnx(
                model, dummy, out_path,
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
