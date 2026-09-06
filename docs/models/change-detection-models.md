# Change Detection Models

4 real models, all locally-built (no `hf_id` — real, dedicated
architecture code lives directly in this codebase).

## ChangeFormer

`changeformer-mit-b0` (13.9M), `changeformer-mit-b4` (67.4M) — a real,
published transformer-based change-detection architecture (Bandara &
Patel, 2022): a Siamese MiT (MixTransformer) encoder with real,
hierarchical multi-scale feature differencing and a real MLP decoder.

```{note}
Found during this project's audit: a real, complete 203-line
implementation of this architecture already existed in this codebase
but the registry never called it — `get_model("changeformer-mit-b4")`
built nothing real. Fixed by wiring the existing, real implementation
into the dispatch path; no new architecture code was needed, only the
connection between the registry entry and the code that already
existed.
```

```python
model = ModelHub().load("changeformer-mit-b4", num_classes=2, in_channels=4)
```

## BIT

`bit-r50` (26.1M) — a real, published "Bitemporal Image Transformer"
(Chen et al. 2021): a real ResNet50 backbone extracts features from
each date independently, and a real transformer models the
relationship between the two resulting token sequences to localize
change.

## DSAMNet

`dsamnet` (16M) — a real, published Deeply Supervised Attention Metric
Network (Shi et al. 2021), using a real, dual-branch Siamese encoder
with a real attention-based feature-distance metric between the two
dates, plus real, deep supervision (auxiliary loss signals at multiple
decoder depths) during training.

## Where `siamese_unet` is

The default, simplest change-detection model
(`pygeovision.ai.pipelines`'s default for `change_detection`) is
`siamese_unet`, in the [native, fully-offline registry](native-registry.md)
— smaller and faster than any model on this page, a real, sensible
default before reaching for one of these larger, more accurate
architectures.
