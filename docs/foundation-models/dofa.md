# DOFA

DOFA (Dynamic One-For-All) is a foundation model that accepts **any set of spectral bands**. Most networks are built for a fixed
band list (three for photographs, six for HLS); give them a different list and they fail or give nonsense. DOFA instead takes the
**wavelength of each band as an input** and builds its first layer on the fly, so one set of weights reads Sentinel-2 (nine
bands), a three-band aerial photograph, or Sentinel-1 radar.

```{figure} ../_static/img/flow-dofa.svg
:alt: DOFA workflow: choose the sensor or list wavelengths, check the band count, build the first layer from the wavelengths, get an embedding
:width: 100%
:class: pgv-figure

The wavelengths are an input of the model, not a detail: a wrong list gives a wrong embedding with no error.
```

## When to use it

| Situation | DOFA? |
|---|---|
| You work with several sensors and want one model for all | Yes. That is what it is for |
| A sensor that no other foundation model supports, with its wavelengths known | Yes |
| Six HLS-style bands, multi-date, flood or crop mapping | [Prithvi-EO](prithvi.md) is trained for that |
| Photographs or three-band aerial imagery only | [DINOv3](dinov3.md) is usually stronger |
| You want a label per pixel | DOFA gives an **embedding vector** for an image. Put a head on it, or use a [segmentation model](../segmentation/models.md) |

## Install and load

DOFA is built on the implementation in **torchgeo**, not on a Hugging Face `AutoModel`, so it needs that package:

```bash
pip install torchgeo
```

```python
from pygeovision.models.foundation.dofa import load_dofa_hf

model = load_dofa_hf("dofa_base_patch16_224", device="cuda")
```

| Name | Parameters | Embedding size | Published weights |
|---|---:|---:|---|
| `dofa_small_patch16_224` | 22 M | 384 | **None** |
| `dofa_base_patch16_224` | 86 M | 768 | Yes (the default) |
| `dofa_large_patch16_224` | 304 M | 1024 | Yes |
| `dofa_huge_patch14_224` | 630 M | 1280 | **None** |

Only the **base** and **large** models have published weights. Asking for `small` or `huge` raises an error instead of handing you
a random network, and so does a failed download:

```text
RuntimeError: No pretrained weights are published for 'dofa_small_patch16_224' (only 'dofa_base_patch16_224' and
'dofa_large_patch16_224' have them). Refusing to return a randomly-initialized model that would look like a successful load.

RuntimeError: Failed to load real, pretrained DOFA weights for 'dofa_base_patch16_224': ... Refusing to silently substitute
a randomly-initialized model ... Pass allow_random_init=True only if you genuinely want an architecture-only model.
```

`allow_random_init=True` returns an architecture-only model whose attribute `_pygeovision_is_random_init` is `True`. Use it for
development and tests (no network), never for results. Without torchgeo you get `ImportError: DOFA requires torchgeo`.

## Wavelengths

```python
from pygeovision.models.foundation.dofa import get_dofa_wavelengths

get_dofa_wavelengths("sentinel2")       # [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]
```

| `sensor=` | Bands, in the order DOFA expects | Wavelengths (micrometres) |
|---|---|---|
| `"sentinel2"` | B04 red, B03 green, B02 blue, B05, B06, B07, B08 NIR, B11 SWIR1, B12 SWIR2 | 0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19 |
| `"sentinel2_rgb"` | red, green, blue | 0.665, 0.56, 0.49 |
| `"naip"` | red, green, blue | 0.665, 0.56, 0.49 |
| `"sentinel1"` | VV, VH | 3.75, 3.75 (modality placeholders, **not** physical wavelengths; this is how the published weights encode radar) |

The nine Sentinel-2 bands leave out the three 60 m atmospheric bands (B01, B09, B10). **The order is part of the contract**: your
array's first band must be red for `"sentinel2"`, and so on. An unknown sensor name raises `ValueError` naming the known ones,
rather than guessing a wavelength.

For any other sensor, give the wavelengths yourself, **in the order of your array's bands**, in micrometres:

```python
landsat_ols = [0.48, 0.56, 0.655, 0.865, 1.61, 2.2]        # blue, green, red, NIR, SWIR1, SWIR2: centre wavelengths
```

Use the values from the sensor's documentation. This is the one place in the package where a wrong number is not caught: the model
cannot tell that 0.865 should have been 0.84.

## Embeddings

```python
from pygeovision.models.foundation.dofa import extract_dofa_embedding

embedding = extract_dofa_embedding(model, image_array, sensor="sentinel2")        # (768,) for the base model
embedding = extract_dofa_embedding(model, image_array, wavelengths=landsat_ols)   # any other sensor
```

| Argument | Meaning |
|---|---|
| `model` | The object from `load_dofa_hf` |
| `image` | A NumPy array, `(C, H, W)` or `(H, W, C)`. The channel count must equal the number of wavelengths |
| `sensor` | A name from the table. Give **exactly one** of `sensor` and `wavelengths` |
| `wavelengths` | A list of micrometres, one per band, in array order |

The result is a NumPy vector of the model's embedding size: 768 numbers for the base model. A `(H, W, C)` array is turned into
`(C, H, W)` for you, but only when the last axis matches the wavelength count; if both the first and last axes could be the
band axis, pass `(C, H, W)` to be safe.

The array is used **as given**. The model was trained on reflectance-like values, so scale digital numbers to reflectance
(for Sentinel-2 level-2A: divide by 10,000) before extracting, as you would for any model.

### Checks that raise errors

| Call | Error |
|---|---|
| Neither or both of `sensor` and `wavelengths` | `ValueError: Pass exactly one of sensor= or wavelengths=.` |
| `sensor="landsat9"` | `ValueError: No real, verified wavelength values for sensor 'landsat9'. Available: [...]` |
| 6-band array with `sensor="sentinel2"` | `ValueError: Image has 6 real band(s) but 9 real wavelength(s) were given ...` |
| `load_dofa_hf("dofa_tiny")` | `ValueError: Unknown DOFA model: 'dofa_tiny'. Available: [...]` |

These are checked before any heavy work, so they are quick to hit and quick to fix.

## Using an embedding

An embedding describes a whole image (a chip of 224 × 224 pixels is the size DOFA was trained on). Typical uses:

| Use | How |
|---|---|
| Classify chips with few labels | Extract an embedding per chip, train a logistic regression or a small network on them |
| Find similar chips | Cosine similarity between embeddings; the same idea as [DINOv3's "more like this"](dinov3.md#more-like-this-from-one-click) |
| Compare sensors | Embed the same place from Sentinel-2 and from NAIP; the vectors live in one space |
| Cluster an archive | K-means on the embeddings of many chips |

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.stack([extract_dofa_embedding(model, chip, sensor="sentinel2") for chip in chips])   # (n, 768)
clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
print(clf.score(X[test_idx], y[test_idx]))
```

Cut scenes into chips of about 224 pixels first ([Preparing training data](../training/data.md)).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: DOFA requires torchgeo` | Package missing | `pip install torchgeo` |
| `RuntimeError: Failed to load real, pretrained DOFA weights` or `No pretrained weights are published` | No network to the weight host, or `small`/`huge` requested | Check the network; pick `base` or `large`; `allow_random_init=True` only for tests |
| `ValueError: Image has N real band(s) but M ...` | The band count differs from the wavelength list | Make the wavelength list the same length and order as the array's bands |
| Embeddings from two sensors do not line up | A wrong wavelength list, or unscaled values | Check the wavelengths against the sensor documentation and scale to reflectance |
| Results change when you reorder bands | The wavelengths no longer match the bands | The order of the array and of the wavelengths must agree |

## Related

- [Foundation models overview](index.md), [Prithvi-EO](prithvi.md), [DINOv3](dinov3.md), [Embeddings](embeddings.md).
- [The foundation-model builders in the Python API](../reference/api/builders-foundation.md).
