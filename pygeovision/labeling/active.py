"""
Active learning -- choose which unlabeled images are worth a human's time.

``ActiveLearner`` ranks an unlabeled pool with the current model and returns the samples to annotate next.
It does **not** train a model and does not annotate anything: you label the selection, call ``update``,
retrain with your own code (``GeoTrainer``), and ask for the next round. See ``docs/labeling/active.md``.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def default_loader(path: str | Path):
    """Read a raster as a ``(1, C, H, W)`` float tensor, min-max scaled **per image**.

    This is a convenient default, not what your model was trained on. If you trained on raw 0-1
    reflectance, on percentile-stretched data, or on per-band-normalised data, pass a ``loader=`` that
    reproduces that, or the uncertainty scores will not reflect the model's real behaviour.
    """
    import numpy as np
    import rasterio
    import torch

    with rasterio.open(str(path)) as src:
        data = np.nan_to_num(src.read().astype(np.float32))
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return torch.from_numpy(data).unsqueeze(0)


class ActiveLearner:
    """Human-in-the-loop sample selection for geospatial model training.

    Strategies:
        - ``entropy`` / ``least_confidence`` / ``margin`` -- uncertainty of the model's per-pixel class
          probabilities, averaged over the image. Works for classification (``(1, C)`` logits) and
          segmentation (``(1, C, H, W)`` logits; a single channel is treated as a sigmoid).
        - ``coreset`` -- greedy k-centre on a coarse (4x4) pooled probability map: picks images that look
          *different from each other*, not necessarily uncertain ones.
        - ``committee`` -- Monte-Carlo dropout: the mutual information between ``mc_passes`` stochastic
          predictions (disagreement between "committee members"). Needs a model with dropout layers.
        - ``random`` -- the baseline you should compare against.

    ``budget`` is the total number of samples you are willing to annotate, ``batch_size`` how many to
    select per round and ``n_iterations`` the number of rounds you plan; ``update`` counts annotated
    samples against the budget and ``finished`` becomes True when the budget or rounds are used up.

    Example::

        learner = ActiveLearner(strategy="entropy", budget=200, batch_size=20)
        while not learner.finished:
            batch = learner.select(model, pool)                  # most informative first
            # ... annotate batch ...
            learner.update(batch, metrics={"val_iou": retrain_and_evaluate()})
    """

    STRATEGIES = ["entropy", "least_confidence", "margin", "coreset", "committee", "random"]

    def __init__(
        self,
        strategy: str = "entropy",
        budget: int = 100,
        n_iterations: int = 10,
        batch_size: int = 16,
        seed: int = 42,
        loader: Callable[[Any], Any] | None = None,
        mc_passes: int = 10,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        if budget < 1 or batch_size < 1 or n_iterations < 1 or mc_passes < 2:
            raise ValueError("budget, batch_size and n_iterations must be >= 1, and mc_passes >= 2")
        self.strategy = strategy
        self.budget = budget
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.seed = seed
        self.loader = loader or default_loader
        self.mc_passes = mc_passes
        self._labeled: list[dict] = []
        self._iteration = 0
        self._history: list[dict] = []
        self._curve: list[dict] = []

    # ── bookkeeping ────────────────────────────────────────────────────────────
    @property
    def remaining_budget(self) -> int:
        return max(0, self.budget - len(self._labeled))

    @property
    def finished(self) -> bool:
        """True when the annotation budget is spent or ``n_iterations`` rounds have been recorded."""
        return self.remaining_budget == 0 or self._iteration >= self.n_iterations

    # ── input handling ─────────────────────────────────────────────────────────
    def _to_tensor(self, item: Any, device: Any):
        """One pool item -> ``(1, C, H, W)`` (or ``(1, C)``) float tensor on ``device``."""
        import numpy as np
        import torch

        if isinstance(item, dict):
            if "path" not in item:
                raise KeyError("pool dicts need a 'path' key")
            x = self.loader(item["path"])
        elif isinstance(item, (str, Path)):
            x = self.loader(item)
        else:
            x = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"cannot turn a {type(item).__name__} into a tensor")
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x.to(device)

    @staticmethod
    def _probs(logits: Any):
        """Class probabilities ``(C, ...)`` from model output."""
        import torch

        if not isinstance(logits, torch.Tensor):
            if isinstance(logits, (tuple, list)) and logits and isinstance(logits[0], torch.Tensor):
                logits = logits[0]
            else:
                raise TypeError(
                    f"the model returned {type(logits).__name__}; ActiveLearner needs a logits tensor "
                    "(classification (1, C) or segmentation (1, C, H, W)). Detection models are not supported.")
        if logits.shape[0] != 1:
            raise ValueError("the model must return a batch of one")
        if logits.shape[1] == 1:                       # single-logit sigmoid head
            s = torch.sigmoid(logits[0])
            return torch.cat([1 - s, s], dim=0)
        return torch.softmax(logits[0], dim=0)

    @staticmethod
    def _reduce(per_pixel: Any) -> float:
        return float(per_pixel.mean())

    def _score_probs(self, p: Any) -> float:
        import torch

        if self.strategy == "entropy":
            return self._reduce(-(p * torch.log(p + 1e-10)).sum(0))
        if self.strategy == "least_confidence":
            return self._reduce(1.0 - p.max(0).values)
        # margin: small gap between the top two classes = uncertain
        top2 = torch.topk(p, k=2, dim=0).values
        return self._reduce(1.0 - (top2[0] - top2[1]))

    # ── selection ──────────────────────────────────────────────────────────────
    def select(
        self,
        model: Any,
        unlabeled_pool: Any,
        n_select: int | None = None,
        device: str | None = None,
    ) -> list[dict]:
        """Select the most informative samples from the unlabeled pool.

        Args:
            model: A trained ``torch.nn.Module``. It is moved to ``device`` and put in eval mode.
                (Not needed for ``strategy="random"``; pass ``None``.)
            unlabeled_pool: Iterable of ``{"path": ...}`` dicts, raster paths, tensors, or dataset samples
                (the first element is used as the image).
            n_select: How many to return. Default: ``batch_size``, capped by the remaining budget.
            device: ``"cpu"``/``"cuda"``; default CUDA when available.

        Returns:
            Selected items, most informative first. Dict items come back as copies with a ``"score"`` key;
            other items as ``{"idx": position_in_pool, "score": ...}``.

        Raises:
            RuntimeError: Every pool item failed (the first error is quoted), so an empty selection is
                never returned silently. Items that fail individually are skipped with a warning.
            ValueError: ``strategy="committee"`` with a model that has no dropout layers.
        """
        pool = list(unlabeled_pool)
        n_select = n_select or min(self.batch_size, self.remaining_budget)
        if n_select <= 0:
            logger.warning("ActiveLearner: annotation budget (%d) is spent; nothing to select.", self.budget)
            return []
        n_select = min(n_select, len(pool))

        if self.strategy == "random":
            rng = random.Random(self.seed + self._iteration)
            idx = list(range(len(pool)))
            rng.shuffle(idx)
            return [pool[i] for i in idx[:n_select]]

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for model-based active learning") from exc
        if model is None:
            raise ValueError(f"strategy={self.strategy!r} needs a model")
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(dev).eval()

        if self.strategy == "coreset":
            return self._coreset_select(model, pool, n_select, dev)
        if self.strategy == "committee":
            return self._committee_select(model, pool, n_select, dev)
        return self._uncertainty_select(model, pool, n_select, dev)

    def _wrap(self, item: Any, i: int, score: float) -> dict:
        return {**item, "score": score} if isinstance(item, dict) else {"idx": i, "score": score}

    def _score_pool(self, pool: list, fn: Callable[[Any, Any], Any], device: Any) -> list[tuple[int, Any]]:
        """Apply ``fn(tensor, item)`` to every item; failures are skipped, counted and reported."""
        import torch

        out, errors = [], []
        with torch.no_grad():
            for i, item in enumerate(pool):
                try:
                    out.append((i, fn(self._to_tensor(item, device), item)))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"item {i}: {type(exc).__name__}: {exc}")
        if errors:
            logger.warning("ActiveLearner: %d of %d pool items could not be scored; first: %s",
                           len(errors), len(pool), errors[0])
        if not out:
            raise RuntimeError(f"no pool item could be scored. First error: {errors[0] if errors else 'empty pool'}")
        return out

    def _uncertainty_select(self, model, pool, n_select, device) -> list[dict]:
        scored = self._score_pool(pool, lambda x, _: self._score_probs(self._probs(model(x))), device)
        scored.sort(key=lambda t: t[1], reverse=True)
        logger.info("ActiveLearner (%s): selected %d/%d samples", self.strategy, min(n_select, len(scored)), len(scored))
        return [self._wrap(pool[i], i, s) for i, s in scored[:n_select]]

    def _embed(self, model, x) -> Any:
        """Coarse 4x4-pooled class-probability map, flattened (``C*16`` numbers)."""
        import torch

        p = self._probs(model(x))
        if p.ndim >= 3:
            p = torch.nn.functional.adaptive_avg_pool2d(p.unsqueeze(0), 4)[0]
        return p.flatten().cpu().numpy()

    def _coreset_select(self, model, pool, n_select, device) -> list[dict]:
        """Greedy k-centre: repeatedly pick the sample farthest from those already picked."""
        import numpy as np

        embedded = self._score_pool(pool, lambda x, _: self._embed(model, x), device)
        idx = [i for i, _ in embedded]
        feats = np.stack([f for _, f in embedded])
        rng = np.random.default_rng(self.seed + self._iteration)
        chosen = [int(rng.integers(len(feats)))]
        picked_at = [0.0]                         # distance to the already-picked set when each was chosen
        d = np.linalg.norm(feats - feats[chosen[0]], axis=1)
        while len(chosen) < min(n_select, len(feats)):
            d[chosen] = -1.0
            nxt = int(np.argmax(d))
            chosen.append(nxt)
            picked_at.append(float(d[nxt]))
            d = np.minimum(d, np.linalg.norm(feats - feats[nxt], axis=1))
        return [self._wrap(pool[idx[c]], idx[c], s) for c, s in zip(chosen, picked_at)]

    def _committee_select(self, model, pool, n_select, device) -> list[dict]:
        """Monte-Carlo dropout: score = mutual information between stochastic predictions (BALD)."""
        import torch

        drops = [m for m in model.modules() if isinstance(m, torch.nn.modules.dropout._DropoutNd)]
        if not drops:
            raise ValueError(
                "strategy='committee' uses Monte-Carlo dropout and the model has no dropout layers, so every "
                "pass would be identical. Add nn.Dropout/Dropout2d, or use strategy='entropy'.")

        def score(x, _):
            ps = torch.stack([self._probs(model(x)) for _ in range(self.mc_passes)])     # (T, C, ...)
            mean = ps.mean(0)
            h_mean = -(mean * torch.log(mean + 1e-10)).sum(0)
            mean_h = (-(ps * torch.log(ps + 1e-10)).sum(1)).mean(0)
            return self._reduce(h_mean - mean_h)

        for m in drops:
            m.train()
        try:
            scored = self._score_pool(pool, score, device)
        finally:
            model.eval()
        scored.sort(key=lambda t: t[1], reverse=True)
        return [self._wrap(pool[i], i, s) for i, s in scored[:n_select]]

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def update(self, newly_labeled: list[dict], metrics: dict[str, float] | None = None) -> ActiveLearner:
        """Record annotated samples (counts against the budget) and, optionally, the model's metrics after retraining.

        ``metrics`` (e.g. ``{"val_iou": 0.71}``) are stored with the running total of labelled samples
        for ``plot_learning_curve``.
        """
        self._labeled.extend(newly_labeled)
        self._iteration += 1
        if metrics:
            self._curve.append({"n_labeled": len(self._labeled), "iteration": self._iteration, **metrics})
        logger.info("ActiveLearner: %d total labeled samples (round %d, budget left %d)",
                    len(self._labeled), self._iteration, self.remaining_budget)
        return self

    def train_iteration(self, model: Any, config: Any = None, unlabeled_pool: Any = None) -> dict[str, Any]:
        """Select the next batch to annotate. **Despite the name this does not train.**

        ``config`` is ignored (a warning is logged if given). Kept for backward compatibility; the same
        as calling :meth:`select` and recording the round.
        """
        if config is not None:
            logger.warning("ActiveLearner.train_iteration does not train: config is ignored. Retrain with "
                           "GeoTrainer yourself, then call update().")
        selected = self.select(model, unlabeled_pool if unlabeled_pool is not None else [])
        result = {
            "iteration": self._iteration + 1,
            "n_selected": len(selected),
            "n_labeled_total": len(self._labeled),
            "remaining_budget": self.remaining_budget,
            "selected_samples": selected,
            "strategy": self.strategy,
            "note": "Annotate selected_samples, retrain, then call learner.update(annotated, metrics=...)",
        }
        self._history.append(result)
        return result

    @property
    def history(self) -> list[dict]:
        """One record per ``train_iteration`` call (what was selected)."""
        return self._history

    @property
    def learning_curve(self) -> list[dict]:
        """``[{"n_labeled", "iteration", <metric>...}]`` from ``update(..., metrics=...)``."""
        return self._curve

    def plot_learning_curve(self, save_path: str | None = None, metric: str = "val_iou") -> None:
        """Plot ``metric`` against the number of labelled samples (needs ``update(..., metrics=...)``)."""
        points = [c for c in self._curve if metric in c]
        if not points:
            logger.warning("No '%s' recorded: call update(batch, metrics={'%s': value}) after each retrain.",
                           metric, metric)
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib required for plot")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([p["n_labeled"] for p in points], [p[metric] for p in points], "o-", color="steelblue")
        ax.set_xlabel("Labelled samples")
        ax.set_ylabel(metric)
        ax.set_title(f"Active learning ({self.strategy})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        else:
            plt.show()
        plt.close(fig)
