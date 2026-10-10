#!/usr/bin/env python
"""Write a *wiring-test* checkpoint: a randomly initialised network of the right shape.

    python tools/make_wiring_checkpoint.py unet_resnet50 --classes 5 --bands 6 -o wiring.pth

It lets you check that a command's plumbing works (the strict load, band/class checks, tiling, file
writing) without having trained anything. Its predictions are random: NEVER read them as a result.
Train a real checkpoint with ``pygeovision.ai.training.trainer.GeoTrainer`` for actual use.
"""
import argparse

import torch

from pygeovision.ai.models.hub import ModelHub

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model", help="registry name, e.g. unet_resnet50 (see `pygeovision ai-models list`)")
ap.add_argument("--classes", type=int, default=2)
ap.add_argument("--bands", type=int, default=3)
ap.add_argument("-o", "--output", default="wiring.pth")
a = ap.parse_args()
net = ModelHub().load(a.model, num_classes=a.classes, in_channels=a.bands, pretrained=False)
torch.save(net.state_dict(), a.output)
print(f"wrote {a.output}: {a.model}, {a.classes} classes, {a.bands} bands (random weights -- wiring tests only)")
