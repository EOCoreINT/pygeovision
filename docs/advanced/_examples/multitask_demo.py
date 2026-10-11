"""MultiTaskLearner on synthetic 4-band chips: one network, a segmentation head and a classification head.

Chips contain a bright square on a noisy background. Segmentation label: 1 inside the square, 0 outside.
Classification label: which of 3 sizes the square has. Both tasks therefore depend on the same visual evidence,
which is the situation where sharing an encoder helps. Weights are NOT pretrained (no network), so absolute
numbers are those of a small CPU run, not of a real project.

Run from the repository root:  PYTHONPATH=. python docs/advanced/_examples/multitask_demo.py
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pygeovision.advanced.multitask import MultiTaskLearner

SIZES = (8, 16, 24)


class Chips(Dataset):
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.x, self.seg, self.cls = [], [], []
        for _ in range(n):
            k = int(rng.integers(0, 3))
            s = SIZES[k]
            img = rng.normal(0.0, 0.3, (4, 64, 64)).astype("float32")
            r, c = rng.integers(0, 64 - s, 2)
            img[:, r:r + s, c:c + s] += 1.5
            mask = np.zeros((64, 64), "int64")
            mask[r:r + s, c:c + s] = 1
            self.x.append(img); self.seg.append(mask); self.cls.append(k)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"inputs": torch.from_numpy(self.x[i]), "segmentation": torch.from_numpy(self.seg[i]),
                "classification": torch.tensor(self.cls[i])}


torch.manual_seed(0)
learner = MultiTaskLearner(backbone="resnet50", in_channels=4, pretrained=False, device="cpu",
                           tasks=["segmentation", "classification"],
                           n_classes={"segmentation": 2, "classification": 3},
                           task_weights={"segmentation": 0.5, "classification": 0.5})
model = learner.build()
print("model parameters: %.1f M" % (sum(p.numel() for p in model.parameters()) / 1e6))

train = DataLoader(Chips(48, 0), batch_size=8, shuffle=True)
val = DataLoader(Chips(16, 1), batch_size=8)
t0 = time.time()
res = learner.train(train, val, epochs=8, lr=1e-3, save_path="/tmp/multitask_demo.pt", log_every=4)
print("trained in %.0f s" % (time.time() - t0))
print("keys:", sorted(res))
print("best_epoch:", res["best_epoch"], "best_val_loss:", res["best_val_loss"])
h = res["history"]
for e in (0, len(h["train_loss"]) - 1):
    print(f"epoch {e + 1}: train {h['train_loss'][e]:.3f}  val {h['val_loss'][e]:.3f}  per task (val) "
          f"{ {k: round(v, 3) for k, v in h['val_task_losses'][e].items()} }")

# predict with the best weights
learner.load_weights("/tmp/multitask_demo.pt")
ds = Chips(16, 1)
batch = torch.stack([torch.from_numpy(a) for a in ds.x])
out = learner.predict(batch)
print({k: v.shape for k, v in out.items()})
seg_true = np.stack(ds.seg)
inter = ((out["segmentation"] == 1) & (seg_true == 1)).sum()
union = ((out["segmentation"] == 1) | (seg_true == 1)).sum()
print("held-out foreground IoU: %.2f   classification accuracy: %.2f" % (inter / union, (out["classification"] == np.array(ds.cls)).mean()))
print("only one task:", {k: v.shape for k, v in learner.predict(batch[:2], tasks=["classification"]).items()})
