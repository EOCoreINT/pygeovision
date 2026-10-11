"""
Multi-task learning for geospatial models (G5).
Jointly optimise segmentation + classification (+ a simple box-regression head) from one backbone.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MultiTaskLearner:
    """Joint multi-task training with a shared encoder and one small head per task (G5).

    Supported tasks and what each head really is:

    ``segmentation``
        Two transposed convolutions and a 1x1 convolution on the encoder's feature map, then
        bilinear upsampling to the input size: logits ``(B, classes, H, W)``.
    ``classification``
        Global average pooling and a two-layer perceptron: logits ``(B, classes)``.
    ``change_detection``
        The same head as classification, applied to **one** tensor. To use it, stack the two dates
        along the channel axis (``in_channels = 2 x bands``); there is no siamese branch.
    ``detection``
        A pooled-feature regressor that outputs ``classes x 5`` numbers per image (one box and
        confidence per class). It is a toy for single-object-per-class problems, **not** a
        multi-object detector; use ``pygeovision.models`` detectors for real detection.

    Example::

        learner = MultiTaskLearner(
            backbone="resnet50", in_channels=4, pretrained=True,
            tasks=["segmentation", "classification"],
            n_classes={"segmentation": 8, "classification": 5},
        )
        model = learner.build()
        results = learner.train(train_loader, val_loader, epochs=50, save_path="multitask.pt")

    Batches are dictionaries: ``{"inputs": (B, C, H, W), "segmentation": (B, H, W) long,
    "classification": (B,) long}`` with one entry per task being trained.
    """

    TASK_HEADS = {
        "segmentation":     "2 x ConvTranspose2d + 1x1 Conv, upsampled to the input size",
        "detection":        "pooled features -> MLP -> classes x (x, y, w, h, conf)",
        "classification":   "global average pool -> MLP",
        "change_detection": "global average pool -> MLP (one tensor in; stack the two dates as channels)",
    }
    BACKBONES = ("resnet50", "resnet101", "vgg16")

    def __init__(
        self,
        backbone: str = "resnet50",
        tasks: list[str] | None = None,
        n_classes: dict[str, int] | None = None,
        task_weights: dict[str, float] | None = None,
        pretrained: bool = True,
        device: str | None = None,
        in_channels: int = 3,
    ) -> None:
        if backbone not in self.BACKBONES:
            raise ValueError(f"backbone must be one of {list(self.BACKBONES)}; got {backbone!r}")
        self.tasks = list(tasks) if tasks else ["segmentation", "classification"]
        unknown = [t for t in self.tasks if t not in self.TASK_HEADS]
        if unknown:
            raise ValueError(f"unknown task(s) {unknown}; supported: {list(self.TASK_HEADS)}")
        given = n_classes or {}
        self.n_classes = {t: int(given.get(t, 4 if t == "classification" else 2)) for t in self.tasks}
        if any(v < 1 for v in self.n_classes.values()):
            raise ValueError(f"n_classes must be positive; got {self.n_classes}")
        self.task_weights = task_weights or {t: 1.0 / len(self.tasks) for t in self.tasks}
        self.backbone = backbone
        self.pretrained = pretrained
        self.in_channels = int(in_channels)
        self.device = device or self._auto_device()
        self._model = None

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def build(self) -> Any:
        """Build (or rebuild, resetting all weights) the model with a shared encoder and the task heads.

        The model's ``forward(x, tasks=None)`` returns ``{task: output}`` for the requested tasks.
        With ``pretrained=True`` torchvision downloads ImageNet weights (needs a network); an
        ``in_channels`` other than 3 keeps the pretrained filters for the first three channels and
        initialises the extra ones with their mean.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torchvision.models as models
        except ImportError:
            raise ImportError("torch + torchvision required")

        backbone_map = {
            "resnet50":  (models.resnet50,  2048),
            "resnet101": (models.resnet101, 2048),
            "vgg16":     (models.vgg16,     512),
        }

        def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
            if conv.in_channels == in_channels:
                return conv
            new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                            conv.padding, bias=conv.bias is not None)
            with torch.no_grad():
                k = min(3, in_channels)
                new.weight[:, :k] = conv.weight[:, :k]
                if in_channels > k:
                    new.weight[:, k:] = conv.weight.mean(dim=1, keepdim=True)
                if conv.bias is not None:
                    new.bias.copy_(conv.bias)
            return new

        class MultiTaskModel(nn.Module):
            def __init__(self, backbone_name, tasks, n_classes, pretrained, in_channels):
                super().__init__()
                backbone_fn, feat_dim = backbone_map[backbone_name]
                base = backbone_fn(weights="DEFAULT" if pretrained else None)
                if backbone_name.startswith("resnet"):
                    base.conv1 = _adapt_first_conv(base.conv1, in_channels)
                else:
                    base.features[0] = _adapt_first_conv(base.features[0], in_channels)
                self.encoder = nn.Sequential(*list(base.children())[:-2])
                self.feat_dim = feat_dim
                self.tasks = tasks
                self.heads = nn.ModuleDict()

                for task in tasks:
                    nc = n_classes[task]
                    if task == "segmentation":
                        self.heads[task] = nn.Sequential(
                            nn.ConvTranspose2d(feat_dim, 256, 4, stride=4),
                            nn.ReLU(), nn.BatchNorm2d(256),
                            nn.ConvTranspose2d(256, 64, 4, stride=4),
                            nn.ReLU(), nn.BatchNorm2d(64),
                            nn.Conv2d(64, nc, 1),
                        )
                    elif task in ("classification", "change_detection"):
                        self.heads[task] = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(256, nc),
                        )
                    elif task == "detection":
                        self.heads[task] = nn.Sequential(
                            nn.AdaptiveAvgPool2d(7),
                            nn.Flatten(),
                            nn.Linear(feat_dim * 49, 1024), nn.ReLU(),
                            nn.Linear(1024, nc * 5),  # nc × (x, y, w, h, conf)
                        )

            def forward(self, x, tasks=None):
                tasks = list(tasks) if tasks else list(self.tasks)
                unknown = [t for t in tasks if t not in self.heads]
                if unknown:
                    raise ValueError(f"model has no head for {unknown}; it was built with {list(self.heads)}")
                features = self.encoder(x)
                out = {}
                for task in tasks:
                    y = self.heads[task](features)
                    if task == "segmentation" and y.shape[-2:] != x.shape[-2:]:
                        # the encoder is 32x down, the head 16x up: restore the input size
                        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    out[task] = y
                return out

        self._model = MultiTaskModel(self.backbone, self.tasks, self.n_classes,
                                     self.pretrained, self.in_channels).to(self.device)
        logger.info("MultiTaskModel built: backbone=%s tasks=%s in_channels=%d",
                    self.backbone, self.tasks, self.in_channels)
        return self._model

    def compute_loss(self, outputs: dict[str, Any], targets: dict[str, Any]) -> Any:
        """Weighted sum of the per-task losses; returns ``(total, {task: float})``.

        Segmentation uses Dice + cross-entropy (``ComboLoss``; label 255 ignored), classification and
        change detection use label-smoothed cross-entropy / cross-entropy, detection uses smooth-L1
        against a ``(B, classes * 5)`` target. Tasks missing from ``outputs`` or ``targets`` are
        skipped; if none remain a ``ValueError`` is raised (otherwise the loss would be a constant).
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required")

        import torch.nn.functional as F

        from pygeovision.losses.class_balance import LabelSmoothingCrossEntropy
        from pygeovision.losses.segmentation import ComboLoss

        task_losses = {}
        total_loss = None

        for task in self.tasks:
            if task not in outputs or task not in targets:
                continue
            pred, target = outputs[task], targets[task]
            weight = self.task_weights.get(task, 1.0)

            if task == "segmentation":
                loss = ComboLoss()(pred, target)
            elif task == "classification":
                loss = LabelSmoothingCrossEntropy()(pred, target)
            elif task == "detection":
                loss = F.smooth_l1_loss(pred, target.float())
            else:
                loss = F.cross_entropy(pred, target.long())

            task_losses[task] = loss.item()
            total_loss = weight * loss if total_loss is None else total_loss + weight * loss

        if total_loss is None:
            raise ValueError(
                f"no task has both an output and a target (tasks={self.tasks}, outputs={sorted(outputs)}, "
                f"targets={sorted(targets)}); batch dictionaries need a key per task")
        return total_loss, task_losses

    def _split_batch(self, batch: Any) -> tuple[Any, dict[str, Any]]:
        if not isinstance(batch, dict) or "inputs" not in batch:
            keys = sorted(batch) if isinstance(batch, dict) else type(batch).__name__
            raise ValueError(f"each batch must be a dict with an 'inputs' tensor and one entry per task "
                             f"{self.tasks}; got {keys}")
        inputs = batch["inputs"].to(self.device)
        return inputs, {k: v.to(self.device) for k, v in batch.items() if k in self.tasks}

    def train(
        self,
        train_loader: Any,
        val_loader: Any | None,
        epochs: int = 100,
        lr: float = 1e-4,
        save_path: str | None = None,
        log_every: int = 10,
    ) -> dict[str, Any]:
        """Train with AdamW, a cosine schedule and gradient clipping (norm 1).

        The weights with the lowest validation loss (training loss when ``val_loader`` is ``None``) are
        written to ``save_path`` as a state dict; the model object itself keeps the **last** epoch's weights.
        Returns ``success``, ``best_val_loss``, ``best_epoch``, ``epochs_trained`` and ``history`` with
        ``train_loss``, ``val_loss`` and per-epoch averages ``train_task_losses`` / ``val_task_losses``.
        """
        if self._model is None:
            self.build()

        try:
            import torch
        except ImportError:
            return {"success": False, "error": "torch required"}

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
        history: dict[str, list] = {"train_loss": [], "val_loss": [],
                                    "train_task_losses": [], "val_task_losses": []}
        best_val_loss, best_epoch = float("inf"), 0

        def _mean_losses(rows: list[dict[str, float]]) -> dict[str, float]:
            keys = {k for r in rows for k in r}
            return {k: sum(r[k] for r in rows if k in r) / sum(1 for r in rows if k in r) for k in keys}

        for epoch in range(1, epochs + 1):
            self._model.train()
            train_losses, train_rows = [], []
            for batch in train_loader:
                inputs, t_targets = self._split_batch(batch)
                optimizer.zero_grad()
                outputs = self._model(inputs)
                loss, task_losses = self.compute_loss(outputs, t_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
                train_rows.append(task_losses)
            if not train_losses:
                raise ValueError("train_loader produced no batches")

            scheduler.step()
            avg_train = sum(train_losses) / len(train_losses)

            avg_val, val_rows = None, []
            if val_loader is not None:
                self._model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, t_targets = self._split_batch(batch)
                        loss, task_losses = self.compute_loss(self._model(inputs), t_targets)
                        val_losses.append(loss.item())
                        val_rows.append(task_losses)
                if not val_losses:
                    raise ValueError("val_loader produced no batches")
                avg_val = sum(val_losses) / len(val_losses)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            history["train_task_losses"].append(_mean_losses(train_rows))
            history["val_task_losses"].append(_mean_losses(val_rows) if val_rows else {})

            monitored = avg_val if avg_val is not None else avg_train
            if monitored < best_val_loss:
                best_val_loss, best_epoch = monitored, epoch
                if save_path:
                    torch.save(self._model.state_dict(), save_path)

            if log_every and epoch % log_every == 0:
                logger.info("Epoch %d/%d | train=%.4f | val=%s", epoch, epochs, avg_train,
                            "n/a" if avg_val is None else f"{avg_val:.4f}")

        return {
            "success": True,
            "best_val_loss": round(best_val_loss, 6),
            "best_epoch": best_epoch,
            "history": history,
            "epochs_trained": epochs,
        }

    def load_weights(self, path: str) -> MultiTaskLearner:
        """Load a state dict written by :meth:`train` into the model (building it first if needed)."""
        import torch
        if self._model is None:
            self.build()
        self._model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return self

    def predict(self, inputs: Any, tasks: list[str] | None = None) -> dict[str, Any]:
        """Run the model without gradients and return NumPy results per task.

        ``segmentation`` -> class map ``(B, H, W)``; ``classification`` / ``change_detection`` -> class
        ids ``(B,)``; ``detection`` -> raw ``(B, classes * 5)`` values. ``inputs`` is a tensor or array
        ``(B, C, H, W)`` (a ``(C, H, W)`` array gets a batch axis).
        """
        import numpy as np
        import torch
        if self._model is None:
            self.build()
        x = (inputs if torch.is_tensor(inputs) else torch.as_tensor(np.asarray(inputs))).float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            out = self._model(x.to(self.device), tasks=tasks)
        result = {}
        for task, y in out.items():
            result[task] = (y.cpu().numpy() if task == "detection" else y.argmax(dim=1).cpu().numpy())
        return result
