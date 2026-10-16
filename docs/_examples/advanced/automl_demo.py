"""GeoAutoML on a toy objective whose best answer is known: lr near 1e-3, loss 'combo'.

Run from the repository root:  PYTHONPATH=. python docs/_examples/advanced/automl_demo.py
The "training" is a formula, so the whole search takes about a second; with a real training function
each trial costs a full training run.
"""
import math

from pygeovision.advanced.automl import GeoAutoML

LOSS_BONUS = {"ce": 0.00, "combo": 0.10, "focal": 0.05}


def train_fn(config):
    """Pretend validation IoU: peaks at lr = 1e-3 and with the 'combo' loss."""
    if config["backbone"] == "broken":                  # a configuration that crashes (e.g. out of memory)
        raise RuntimeError("CUDA out of memory (simulated)")
    quality = 0.80 - 0.10 * abs(math.log10(config["lr"]) + 3) + LOSS_BONUS[config["loss"]]
    return quality - 0.002 * config["epochs"] / 10


space = {
    "lr": ("float", 1e-5, 1e-1, "log"),
    "loss": ("categorical", ["ce", "combo", "focal"]),
    "epochs": ("int", 5, 30),
    "backbone": ("categorical", ["resnet50", "broken"]),
    "seed": ("constant", 7),
}

automl = GeoAutoML(metric="val_iou", direction="maximize", n_trials=40, seed=1, show_progress_bar=False)
result = automl.search(train_fn, space)

print("success          :", result["success"])
print("best_value       :", result["best_value"])
print("best_params      :", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in result["best_params"].items()})
print("completed/pruned/failed:", result["n_trials_completed"], result["n_trials_pruned"], result["n_trials_failed"])
print("first error      :", result["errors"][:1])
print("importance       :", automl.importance())

# A trial that reports progress can be stopped early
def train_with_pruning(config, trial):
    value = 0.0
    for epoch in range(10):
        value = train_fn({**config, "backbone": "resnet50"}) * (epoch + 1) / 10
        trial.report(value, epoch)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()
    return value

pruned = GeoAutoML(n_trials=40, seed=1, show_progress_bar=False).search(
    train_with_pruning, {k: v for k, v in space.items() if k != "backbone"}, pruner="median")
print("with pruning     : completed", pruned["n_trials_completed"], "pruned", pruned["n_trials_pruned"])

# Every trial fails -> success False, with the reason
bad = GeoAutoML(n_trials=3, show_progress_bar=False).search(lambda c: float("nan"), {"x": ("float", 0, 1)})
print("all NaN          :", bad["success"], "-", bad["error"][:70])
