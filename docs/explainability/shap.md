# Band importance (SHAP)

**Class:** `GeospatialSHAP` (`pygeovision.explainability`) · **Python:** `client.xai.shap(model)` · **Needs:** `pip install shap`

## Why

You trained a model on 12 Sentinel-2 bands. Which ones does it use? If the answer is "essentially NIR and red", the other
ten bands cost download time and storage for nothing, and a sensor that lacks them is still usable. If the answer is
"band 9 (water vapour)", something is off. SHAP (SHapley Additive exPlanations) assigns each input value a share of the
prediction; averaging the size of those shares per band gives a **band importance**.

## How to use

```python
from pygeovision.explainability import GeospatialSHAP

shap_exp = GeospatialSHAP(model, background_samples=20)
result = shap_exp.band_importance(chip)                     # chip: (C, H, W) or (1, C, H, W)
print(result)
shap_exp.plot_band_importance(result, band_names=["blue", "green", "red", "nir"], save_path="bands.png")
```

```text
{'band_importance': {'band_1': <value>, 'band_2': <value>, 'band_3': <value>, 'band_4': <value>}, 'method': 'shap'}
```

### `GeospatialSHAP(model, device=None, background_samples=50)`

`device` defaults to the model's own device. `background_samples` is the size of the reference set.

### `band_importance(image, n_samples=None, class_idx=None)`

| Argument | Meaning |
|---|---|
| `image` | One chip, `(C, H, W)` or `(1, C, H, W)` |
| `n_samples` | Overrides `background_samples` for this call |
| `class_idx` | The model output to explain. `None` (default) averages the importance over **all outputs**; an index explains one class |

Returns `{"band_importance": {"band_1": ..., ...}, "method": "shap"}`. The value is the **mean absolute SHAP value** over
all pixels of the chip, for that band. Only ratios between bands matter, not the numbers. If `shap` is not installed you
get `{"error": "pip install shap"}` rather than an exception.

`plot_band_importance(values, band_names=None, save_path=None)` draws a horizontal bar chart; with `save_path` it writes a
PNG, otherwise it shows the figure. `band_names` must have one name per band (`ValueError` otherwise).

### How it is computed

SHAP's `DeepExplainer` compares the model's output for your chip with its output for a **background**: here
`background_samples` all-zero images. Importance is therefore *relative to an all-zero input*. That is a sensible baseline
if your data are standardised (zero = the average pixel) and a poor one for raw reflectance (zero = no light at all,
something the model never saw). **Standardise your chips the way the model was trained** before calling this.

## Reading the result: SHAP against an ablation

Do not trust one method. Cross-check the SHAP ranking with an **ablation**: set a band to zero (or its training mean) and
measure how much accuracy drops on chips the model has not trained on.

```python
import numpy as np
imp = np.mean([list(shap_exp.band_importance(c)["band_importance"].values()) for c in chips], axis=0)

def accuracy(X, Y):
    return float((model(X).argmax(1) == Y).float().mean())

base = accuracy(X_val, Y_val)
for b, name in enumerate(band_names):
    Xb = X_val.clone(); Xb[:, b] = 0
    print(name, "SHAP", round(float(imp[b]), 4), "accuracy drop", round(base - accuracy(Xb, Y_val), 3))
```

Where the two agree, you can trust the ranking. Where they disagree, neither is wrong: they answer different questions. SHAP
measures the contribution to the output *around the zero baseline*; ablation measures what is lost when the band is removed.
When they disagree, believe the ablation if your question is "can I drop this band?", and treat SHAP as a pointer for where to look.

## Cost and limits

- **SHAP is slow.** DeepExplainer scales with the input size and the background size. Use small chips (32–64 pixels) and
  20–50 background samples; expect seconds per chip on a CPU, more for larger chips.
- **One chip is noisy.** Average over many chips (the ranking of the weaker bands can change between one chip and ten).
- **Output layout.** SHAP returns either a list (one array per output) or an array with a trailing output dimension; both
  are handled. For a segmentation model the output is large (`classes × H × W`) and slow; for a quick answer explain a
  chip classifier or a model wrapper that returns one score per chip.
- **Additivity is not checked**: `check_additivity=False` is passed because the check gives false alarms with recent
  PyTorch versions. The values are still SHAP values; the internal consistency test is skipped.
- **Importance is not causation** and depends on the correlation between bands: highly correlated bands share their credit.
- Only band-level importance is returned. The raw per-pixel SHAP values are not exposed.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `{"error": "pip install shap"}` | `shap` or `torch` missing | `pip install shap` |
| Very slow | Large chip or many background samples | Smaller chip, fewer samples |
| Importances differ a lot between chips | One chip is a small sample | Average over 20+ chips |
| All values near zero | Chip nearly identical to the all-zero background, or a model with a saturated output | Standardise the input; check the chip is not blank |
| Raw-reflectance model looks odd | Zero background is not meaningful for raw reflectance | Standardise (subtract mean, divide by SD) before explaining |
| `ValueError: class_idx 5 out of range for 2 model output(s)` | Index outside the model outputs | Use a valid class |
| Error inside `DeepExplainer` | A layer type SHAP cannot handle | Wrap the model, or use ablation (set a band to 0 and compare) |
| `ValueError: N band names for M bands` | Wrong length in `plot_band_importance` | One name per band |
| SHAP says a band is unimportant but removing it hurts | Different question, see above | Trust the ablation for "can I drop it?" |

```{seealso}
[GradCAM](gradcam.md) · [Spectral indices](../data/indices.md) · [Training](../training/index.md)
```
