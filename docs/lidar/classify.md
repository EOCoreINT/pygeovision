# Classifying points with a network you trained

```python
r = proc.classify_points(
    "tile.las", "classified.las", num_classes=3, class_names=["ground", "vegetation", "building"],
    model="randlanet", checkpoint_path="my_randlanet.pt",
    class_codes=[2, 5, 6],                      # write standard LAS codes
)
```

## How a cloud is fed to the network

Point-cloud networks cannot take a whole survey at once (their neighbour search grows with the square of the number of
points). `classify_points` therefore:

1. cuts the cloud into `block_size` × `block_size` squares (default 50 m);
2. splits a square with more than `max_points` points (8 192) at random into several chunks, and pads a chunk with fewer than
   `min_points` (1 024) by repeating points (the padding is discarded afterwards);
3. gives the network, per chunk, **four channels**: x, y, z minus the chunk's centroid, divided by `coord_scale`
   (default 1.0), and intensity divided by `intensity_scale` (default 65 535; zeros if the file has no intensity);
4. writes each point's predicted class into the LAS classification field of a copy of the file (original coordinates intact).
   Every point is predicted exactly once.

**Train with the same input convention**, or the network sees inputs it never learned. If you trained on unit-scaled blocks,
set `coord_scale` to match.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `num_classes` | 3 | Must equal the checkpoint's output classes |
| `class_names` | `class_0…` | Names for the count report |
| `model` | `"randlanet"` | `"randlanet"` or `"pointnet2"` (PointNet++ SSG). Any other value raises `ValueError` |
| `checkpoint_path` | `None` | A `state_dict` saved with `torch.save` (a dict with a `state_dict` / `model_state_dict` key and `module.` prefixes are accepted). Loaded with `weights_only=True` |
| `class_codes` | class index | LAS code written for each class: map to 2 ground, 3–5 vegetation, 6 building … so other software understands the file. Must be 0–31 for LAS point formats 0–5, 0–255 above |
| `block_size`, `max_points`, `min_points` | 50.0, 8192, 1024 | Chunking (above) |
| `coord_scale`, `intensity_scale` | 1.0, 65535 | Input scaling (above) |
| `device` | auto | |
| `seed` | 0 | Fixes the networks' random sampling and the chunk split, so a run is repeatable |

## Expected output without a trained model

```text
{'success': True, 'n_points': 13902, 'model': 'randlanet', 'used_pretrained_checkpoint': False, 'meaningful': False,
 'n_chunks': 4, 'class_counts': {'ground': 0, 'vegetation': 6562, 'building': 7340}}
warning: no checkpoint_path: the network has random weights, so the classes are meaningless (plumbing test only). ...
```

Look at `meaningful: False` and the `warning` key: a random network splits the points between classes arbitrarily (here 47 %
vegetation, 53 % building, for a scene that is mostly ground). This mode exists to test your file handling and output only.

With a checkpoint, `meaningful` is `True`, there is no `warning`, and `used_pretrained_checkpoint` is `True` (the name is
historical: it means "a checkpoint was loaded").

## Cost

Measured on a 2-core CPU: 300 000 points took about **29 s** with PointNet++ and **35 s** with RandLA-Net (64 chunks), peaking
below 1 GB of memory. A GPU is much faster; a real survey of tens of millions of points should be split into files first.
