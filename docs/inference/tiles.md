# Tiles, overlap and seams

A network sees less context at the edge of a tile and is less reliable there. Where neighbouring tiles meet with no
overlap, the two sides can disagree and leave a visible line. The remedy is to make tiles overlap and to **weight**
each prediction by how far the pixel is from the tile's edge, so the confident centre of every tile counts most.

A measurement (a trained U-Net on a 256 × 256 scene; "differs" is the share of pixels whose class differs from the
prediction made on the whole scene in one pass):

| Chip | Overlap | Blend | Differs from whole-scene prediction | Time |
|---:|---:|---|---:|---:|
| 128 | 0 | constant (no blending) | 0.31% | 0.4 s |
| 128 | 0 | gaussian | 0.31% | 0.3 s |
| 128 | 16 | gaussian | **0.03%** | 0.9 s |
| 128 | 64 | gaussian | 0.03% | 1.9 s |
| 128 | 16 | linear | 0.03% | 0.8 s |
| 128 | 16 | constant | 0.04% | 0.7 s |
| 64 | 0 | constant | 0.92% | 0.8 s |
| 64 | 16 | gaussian | 0.14% | 1.9 s |
| 64 | 32 | gaussian | 0.12% | 3.2 s |

What this shows, and does not:

- A **small overlap does most of the work** (0.31% → 0.03%); a large one only costs time.
- **Smaller tiles disagree more** (0.92% at 64 pixels with no overlap) because each sees less context.
- This is one small, synthetic scene. On real imagery with fine structure (roads, small buildings), seams matter more.

Guidance:

| Choose | Rule of thumb |
|---|---|
| `chip_size` | At least the size you trained on, as large as memory allows (256–1024). Larger tiles see more context |
| `overlap` | About 10–25% of `chip_size` (32 for 256, 64 for 512). More gives diminishing returns |
| `blend` | `gaussian` (default) unless you have a reason |

Notes: tiles on the right and bottom edges are padded with zeros so the model always sees a full-size chip, and their
weight is set so they affect only the pixels they cover. Every pixel gets a prediction, including the outermost ones.

## Test-time augmentation

`tta=True` predicts each tile four times (original, flipped horizontally, flipped vertically, both) and averages. It
smooths noisy predictions and typically adds a little accuracy, at about four times the cost. In the same test it
changed 0.28% of pixels relative to a single pass. Use it for final maps, not for exploring.
