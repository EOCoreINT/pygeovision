"""Compare four segmentation models on a held-out scene with ModelEvaluator.

Needs torch. Uses the OBIA demo scene, so it runs offline in a few seconds on a CPU:

    python benchmark_demo.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pygeovision import obia
from pygeovision.benchmark import Leaderboard, ModelEvaluator

torch.manual_seed(0)
train_scene = obia.make_demo_scene(seed=0)       # the models learn from this scene
test_scene = obia.make_demo_scene(seed=1)        # ...and are scored on a different one


def chips(scene, n, size=32, seed=0):
    """n random (4, size, size) chips, min-max scaled per band, with 0-based class labels."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n):
        r, c = rng.integers(0, scene.raster.image.shape[1] - size, 2)
        x = scene.raster.image[:, r:r + size, c:c + size].copy()
        for b in range(x.shape[0]):
            x[b] = (x[b] - x[b].min()) / (x[b].max() - x[b].min() + 1e-8)
        X.append(x)
        Y.append(scene.class_raster[r:r + size, c:c + size] - 1)      # class codes start at 1
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)).long()


class Net(nn.Module):
    def __init__(self, width, n_classes=6):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(width, n_classes, 1)

    def forward(self, x):
        return self.head(self.features(x))


class Quarter(nn.Module):
    """Wraps a model so its output is 1/4 of the input size, like many transformer decoders."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return nn.functional.avg_pool2d(self.model(x), 4)


def train(model, X, Y, steps=150):
    opt = torch.optim.Adam(model.parameters(), 5e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (16,))
        loss = nn.functional.cross_entropy(model(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


Xtr, Ytr = chips(train_scene, 200)
Xte, Yte = chips(test_scene, 64, seed=5)
big = train(Net(16), Xtr, Ytr)
small = train(Net(4), Xtr, Ytr)
untrained = Net(16).eval()


class OutOfMemory(nn.Module):
    def forward(self, x):
        raise RuntimeError("CUDA out of memory (simulated)")


loader = DataLoader(TensorDataset(Xte, Yte), batch_size=16)
evaluator = ModelEvaluator(task="segmentation", num_classes=6)
results = evaluator.compare(
    {"net-16": big, "net-4": small, "untrained": untrained,
     "quarter-res": Quarter(big), "broken": OutOfMemory()},
    loader, dataset_name="demo-B", device="cpu")
evaluator.print_leaderboard(results)

best = results[0]
print({k: round(v, 3) for k, v in best.per_class_iou.items()})

board = Leaderboard("./leaderboard.json")
for r in results:
    if not r.failed:                 # the leaderboard refuses failed runs
        board.add(r)
board.print(task="segmentation")
evaluator.save_results(results, "./results.json")
