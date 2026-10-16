"""Compare segmentation models on a held-out part of a real scene with ModelEvaluator.

Needs torch and a network. The scene comes from PyGeoVision's own download tools (Sentinel-2 through
``obia.fetch_scene``, ESA WorldCover as the labels), the models learn from the west of the scene and are scored on
the east, which they never saw:

    python benchmark_demo.py

WorldCover is itself a model's output, so the scores below measure how well a network reproduces WorldCover on new
ground, not how well it maps the truth. For a real study, score against labels you trust.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pygeovision as pgv
from pygeovision import obia
from pygeovision.benchmark import Leaderboard, ModelEvaluator

torch.manual_seed(0)
BBOX = (-1.70, 6.18, -1.655, 6.225)                    # lon/lat, about 5 km x 5 km
BANDS = ("blue", "green", "red", "nir")
# WorldCover code -> class id 0..4. Every other code becomes 255, which the evaluator ignores.
CLASS_IDS = {10: 0, 30: 1, 40: 2, 50: 3, 80: 4}
CLASS_NAMES = ["forest", "grass", "cropland", "built_up", "water"]

client = pgv.PyGeoVision()
rd = obia.fetch_scene(client, BBOX, "2024-01", "./download", bands=BANDS)
wc, _ = obia.fetch_worldcover(client, BBOX, "./download/worldcover.tif", like=rd)

labels = np.full(wc.shape, 255, np.uint8)
for code, cid in CLASS_IDS.items():
    labels[wc == code] = cid
valid = obia.valid_mask(rd)                            # False under cloud and shadow
labels[~valid] = 255
image = np.nan_to_num(rd.image, nan=0.0)


def chips(col_lo, col_hi, n, size=32, seed=0):
    """n random (4, size, size) chips from columns [col_lo, col_hi), min-max scaled per band, mostly on valid ground."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n:
        r = int(rng.integers(0, image.shape[1] - size))
        c = int(rng.integers(col_lo, col_hi - size))
        y = labels[r:r + size, c:c + size]
        if (y != 255).mean() < 0.5:                    # mostly cloud or unmapped: skip
            continue
        x = image[:, r:r + size, c:c + size].copy()
        for b in range(x.shape[0]):
            x[b] = (x[b] - x[b].min()) / (x[b].max() - x[b].min() + 1e-8)
        X.append(x)
        Y.append(y)
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)).long()


class Net(nn.Module):
    def __init__(self, width, n_classes=len(CLASS_NAMES)):
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
        loss = nn.functional.cross_entropy(model(X[idx]), Y[idx], ignore_index=255)
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


W = image.shape[2]
split = int(W * 0.6)
Xtr, Ytr = chips(0, split, 200)                        # learn from the west...
Xte, Yte = chips(split, W, 64, seed=5)                 # ...and be scored on the east
big = train(Net(16), Xtr, Ytr)
small = train(Net(4), Xtr, Ytr)
untrained = Net(16).eval()


class OutOfMemory(nn.Module):
    def forward(self, x):
        raise RuntimeError("CUDA out of memory (simulated)")


loader = DataLoader(TensorDataset(Xte, Yte), batch_size=16)
evaluator = ModelEvaluator(task="segmentation", num_classes=len(CLASS_NAMES))
results = evaluator.compare(
    {"net-16": big, "net-4": small, "untrained": untrained,
     "quarter-res": Quarter(big), "broken": OutOfMemory()},
    loader, dataset_name="east-of-scene", device="cpu")
evaluator.print_leaderboard(results)

best = results[0]
print({k: round(v, 3) for k, v in best.per_class_iou.items()})

board = Leaderboard("./leaderboard.json")
for r in results:
    if not r.failed:                 # the leaderboard refuses failed runs
        board.add(r)
board.print(task="segmentation")
evaluator.save_results(results, "./results.json")
