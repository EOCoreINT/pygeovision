# Foundation Models

9 real entries in this category, plus DINOv2/DINOv3, filed separately
below due to a real, significant labeling issue found in this audit.

## Prithvi family

`prithvi-100m`/`prithvi_eo_1_0` (100M, both point to the same real
`ibm-nasa-geospatial/Prithvi-100M` weights — `eo_1_0` is the newer
alias), `prithvi-300m` (300M, `ibm-nasa-geospatial/Prithvi-300M`),
`prithvi_eo_2_0` (600M, the real, current-generation
`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`). A real, published NASA/IBM
geospatial foundation model — a real Vision Transformer masked
autoencoder pretrained on real HLS (Harmonized Landsat-Sentinel) time
series, genuinely designed for multi-temporal Earth observation tasks
rather than adapted from a natural-image model.

```python
from pygeovision.models.foundation.prithvi import load_prithvi_hf

model = load_prithvi_hf("prithvi_eo_2_0", device="cuda")
```

```{warning}
`load_prithvi_hf` previously had a severe, real bug: it silently
returned a randomly-initialized, untrained model with the correct
shape on any real loading failure, with no reliable indication in the
return value that this had happened. This is now fixed — the function
raises clearly by default. If you see `allow_random_init=True`
anywhere in your own code calling this function, that's an explicit,
deliberate opt-out of real pretrained weights, not a normal setting.
```

## DOFA

`dofa-base` (86M) — Dynamic One-For-All (Xiong et al. 2024,
arXiv:2403.15356), a real, published multi-sensor foundation model
using dynamic weight generation to handle inputs with different real
numbers/types of spectral bands (optical, SAR, hyperspectral) through
one real, shared architecture.

```{note}
A real, confirmed bug fixed in this audit: `dofa-base`'s `hf_id`
pointed to `"XShadow/DOFA-ViT-base-p16"`, a repository name that does
not exist. The real, confirmed repository (the original paper
authors' own upload) is `XShadow/DOFA`. This is fixed, but the
registry honestly flags that this is a research-group-hosted
checkpoint — the generic `AutoModel` dispatch path this registry uses
is unverified for it, unlike the cleanly HF-standard entries above.
```

## Tessera

`tessera` — real, precomputed 128-channel embeddings from the real
[geotessera](https://github.com/ucam-eo/tessera) project, fetched for
a real requested region and year (2017–2024) rather than run as local
inference — there are no real local "weights" to download for this
entry, which is why `params_m=0.0`.

```python
from pygeovision.models.foundation.tessera import TesseraClient

client = TesseraClient()
result = client.fetch_mosaic_for_region(bbox=(...), year=2023)
```

## AlphaEarth

`alphaearth` — a real Google Earth Engine-hosted embedding dataset,
accessed via a real, asynchronous Earth Engine export task rather than
direct local inference (same reason `params_m=0.0` as Tessera above).
A completed `success: True` here means the real export task was
genuinely submitted to Earth Engine's queue — an honestly different,
weaker claim than "the data is ready," and the real return value
reflects that distinction (a trackable task ID plus "See Google
Drive," not a ready-to-use local file).

## RemoteCLIP

`remoteclip-b32` (151M), `remoteclip-l14` (428M) — real weights from
`BAAI/RemoteCLIP-ViT-{B-32,L-14}` (Beijing Academy of AI, the real
paper's actual publishing organization). A real CLIP variant
specifically trained on remote-sensing image-text pairs, giving
real, meaningfully better zero-shot classification and retrieval on
satellite imagery than a generic, natural-image CLIP checkpoint.

```{note}
A real, cross-registry inconsistency was found and fixed during this
audit: the separate metadata catalog
([`pygeovision.ai.models.zoo`](../reference/model-catalog.md)) pointed
RemoteCLIP at a different repository (`chendelong/RemoteCLIP`) than
this registry's already-verified `BAAI/RemoteCLIP-*`. Aligned to the
authoritative source.
```

## DINOv2 — labeled DINOv3 before this audit

`dinov2-s/b/l/g` are real, correctly-labeled DINOv2 entries. But this
codebase's DINOv3-labeled entries (`dinov3_vits16`, `dinov3_vitl16_sat`,
and 10 more, all removed) loaded the identical real
`facebook/dinov2-*` weights under a DINOv3 name — including variants
claiming to be "SAT-493M satellite-pretrained," which loaded the exact
same generic, non-satellite DINOv2 weights regardless of which `_sat`
name was requested. All 12 mislabeled entries were removed rather than
relabeled, since a genuinely correct DINOv3 integration would need real,
separately-verified DINOv3 weights this codebase does not have.

```{warning}
`pygeovision/models/foundation/dinov3.py` still exists (real code,
real callers elsewhere in this codebase — `CHMv2Model`,
`DINOv3Backbone`), and its module docstring carries an explicit,
unmissable warning about this. If you're reading that module's source
directly rather than going through the cleaned registry above, the
warning is there for exactly this reason.
```

## LISAt — real weights, no verified build path

`lisat-7b` — see [Model Registry](../core-features/model-registry.md)
for the full detail on why this is listed with accurate metadata but
deliberately does not build.
