(sar-adapters)=
# Radar and foundation models

Prithvi and DINOv3 were trained on **optical** imagery. Sentinel-1 radar measures backscatter, a different physical quantity, in
decibels, with speckle. Feeding raw radar to an optical model fails, usually **without an error**: the numbers go in and a confident
looking map comes out. This page explains the three routes PyGeoVision offers, what each really does, and how to prepare the data so
the result means something.

```{figure} ../_static/img/flow-radar.svg
:alt: Using radar with an optical foundation model: prepare on linear power and convert to decibels, normalise to 0 to 1, map VV and VH to six channels, validate, then threshold or fine-tune, and check against a reference
:width: 100%
:class: pgv-figure

The first two steps matter as much as the model. The zero-shot route is a threshold, not a Prithvi run.
```

```{admonition} What was tested
:class: note
Run here, offline, on synthetic scenes with a known dark "water" strip: every channel mapping and its layout, the validation messages, the
zero-shot threshold, the argument errors, the speckle statistics (mean and spread against the theory for 1, 4 and 8 looks), and that
`normalise_sar_for_ai` equals the fixed-bounds formula (`tests/test_docs_radar.py`, `tests/test_sar_dinov3_adapter.py`).
**Not run here:** TerraTorch fine-tuning, the DINO backbone with real weights, and any radar dataset such as Sen1Floods11 (no network, no GPU).
Nothing on this page measures flood-mapping accuracy.
```

## The three routes

| Route | What it does | Honest status |
|---|---|---|
| **Channel mapping** (`sar_to_hls_6ch`) | Rearranges VV and VH into six pseudo-HLS channels (VH → blue, VV → red, the inverses and the VV/VH ratio in the infrared positions) so an HLS model gets inputs in a sensible form | A preprocessing step. It does not make Prithvi understand radar by itself |
| **Zero-shot adapter** (`SARPrithviAdapter(mode="zero_shot")`) | **Thresholds the mapped VH channel** (below 0.35 on the [0, 1] scale = water) | Not a Prithvi inference. `result["method"] == "vh_threshold"`. Its quality is that of a simple threshold |
| **Fine-tuning** (`mode="fine_tune"` or `"spt"`, then `fine_tune()`) | Trains on a radar dataset such as Sen1Floods11 through TerraTorch | The route that actually adapts a model to radar. Needs TerraTorch and labelled data |

For DINOv3 the equivalent of the mapping is a **pseudo-RGB** (three channels built from VV and VH), described [below](#dinov3-with-radar).

## Step 1: prepare the radar first

The mapping expects values **normalised to [0, 1]**. That is the output of `normalise_sar_for_ai` after calibration, speckle filtering on
**linear power** and conversion to decibels ([Radar data](../data/sar.md)). `normalise_sar_for_ai(input_path, output_path, method="minmax_db")`
works on files; its default maps a fixed −35 … +5 dB range onto 0 … 1 (`method="percentile"` uses each scene's 2nd and 98th percentile
instead, which is not comparable between scenes):

```python
from pygeovision.data.processors.sar import normalise_sar_for_ai
normed = normalise_sar_for_ai("s1_vv_db.tif", "s1_vv_norm.tif")      # float32, shape of the input, written to the second path
```

For an array you already hold in memory, the same fixed-bounds rule is one line (checked equal to the function's output):

```python
import numpy as np
vv = np.clip((vv_db + 35.0) / 40.0, 0.0, 1.0)
```

**Fixed bounds matter for before-and-after pairs:** both dates are scaled by the same rule, so a change in the number is a change in the
backscatter, not in the stretch. The mapping does **not** convert decibels for you: feeding it raw dB numbers makes bands such as `1 − VH`
meaningless, and `validate_sar_ai_input` warns when values leave [0, 1]. (Only the `sentinel_6ch_db` mapping is meant for dB input.)

## Step 2: map to six channels

```python
import numpy as np
from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch, validate_sar_ai_input

# vv, vh: float32 arrays (H, W) or (1, H, W), normalised to [0, 1]
six = sar_to_hls_6ch(vv, vh, mapping="physics_guided")        # (6, H, W) float32
check = validate_sar_ai_input(six, model="prithvi")           # {'valid': True, 'warnings': [], 'errors': [], 'shape': (6, 96, 96)}
```

| Argument | Default | Meaning |
|---|---|---|
| `vv`, `vh` | | Arrays `(H, W)` or `(1, H, W)` of the same shape (a mismatch raises `ValueError: VV shape (1, 96, 96) != VH shape (1, 50, 96)`) |
| `mapping` | `"physics_guided"` | The arrangement (next table). An unknown name raises `ValueError` and lists the five valid ones |
| `normalise` | `False` | `True` z-scores each of the six channels (mean 0, spread 1), as the fine-tuning route expects |

If `vh` is left out the function **guesses** it as `vv × 0.6` and logs `sar_to_hls_6ch: VH not provided — approximating from VV (×0.6)`.
That keeps the shapes right but invents data (and the ratio channel becomes constant, which `validate_sar_ai_input` then flags). Supply the real VH.

The mappings, with the mean of each of the six output channels measured on a test scene (VV about 0.6, VH about 0.5, with a dark water strip):

| `mapping` | Position 0 … 5 (Blue, Green, Red, NIR, SWIR1, SWIR2) | Measured means | Use |
|---|---|---|---|
| `physics_guided` | VH · (VH+VV)/2 · VV · 1−VH · 1−VV · VV/VH (ratio capped at 10, rescaled to 0–1) | 0.38 0.44 0.50 0.62 0.50 0.16 | **The default.** Keeps the dark-water, bright-vegetation contrast that HLS models learned |
| `mean_repeat` | (VH+VV)/2 in all six | 0.44 ×6 | Ablation baseline |
| `vv_vh_ratio` | VV · VH · ratio · VV · VH · ratio | 0.50 0.38 0.16 0.50 0.38 0.16 | Looking at it; moderate for models |
| `replicate` | VV · VH alternating | 0.50 0.38 0.50 0.38 0.50 0.38 | Ablation baseline |
| `sentinel_6ch_db` | Physics mapping for **dB** input, rescaled per scene | | Only when you pass dB values |

## Step 3: validate

`validate_sar_ai_input(six, model="prithvi", expected_range=(0.0, 1.0))` returns `{"valid", "warnings", "errors", "shape"}`. What it says,
as measured:

| Situation | Result |
|---|---|
| Good input | `valid: True`, no warnings |
| Values outside the range (dB or z-scored numbers got in) | A warning: `Values [-35.000, 36.000] outside expected [0.0, 1.0]. Ensure normalise_sar_for_ai() was applied before sar_to_hls_6ch().` |
| Smaller than 64 × 64 | **Error**, `valid: False`: `Spatial size 40×40 is below minimum 64×64 for ViT patch tokenisation.` |
| Not square (for Prithvi) | A warning: `Prithvi performs best on square chips. Current: 96×70. Consider padding to 96×96.` |
| A channel that is entirely NaN, or has some NaN | **Error**, `valid: False`, naming the channel (`Channel 2 (Red) contains 1 NaN value(s).`) |
| A channel that is all zeros or constant | A warning naming the channel |
| Aspect ratio beyond 10 : 1 | A warning that a clip may be wrong |
| Wrong shape (not `(6, H, W)`) | **Error** telling you to run `sar_to_hls_6ch` |

## Step 4: choose a route

### Zero-shot: a threshold

```python
from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter

result = SARPrithviAdapter(mode="zero_shot", task="flood_detection").run(vv, vh, channel_mapping="physics_guided")
result["prediction"]      # (H, W) uint8: 1 = VH below 0.35 (treated as water), 0 = not
result["method"]          # 'vh_threshold'
result["confidence"]      # None: a threshold has no confidence
result["warnings"]        # the validation warnings, if any
result["channels"]        # names of the six mapped channels
```

On the test scene the dark strip came out as water in **100 %** of its pixels and the rest of the scene in about 6 %, which shows the
threshold does what it says on easy synthetic data. It says nothing about real floods. The result keys are `model`, `mode`, `task`, `prediction`,
`confidence`, `channels`, `warnings` and `method`. A scene that fails validation (for example 40 × 40) returns `prediction: None` with the reasons
in `warnings`; it does not raise.

The constructor checks its arguments: an unknown `mode` raises `ValueError: Unknown mode 'x'. Use 'zero_shot', 'fine_tune', or 'spt'.`, an
unknown `task` lists the three valid ones (`flood_detection`, `land_cover`, `change_detection`), and any mode other than zero-shot raises
`ImportError: Mode 'fine_tune' requires TerraTorch. Install it with: pip install terratorch`.

A threshold on backscatter cannot tell water from other things that are dark to radar: radar shadow behind hills and buildings, smooth
tarmac, runways, wet sand. That is the usual source of false floods.

### Fine-tuning on labelled radar

`SARPrithviAdapter(mode="fine_tune", ...)` and `mode="spt"` (scattering prompt tuning, for few labels) need TerraTorch (`pip install terratorch`).
`fine_tune(data_root=..., output_dir=..., max_epochs=50, lr=1e-4, freeze_backbone=True)` launches the TerraTorch run on Sen1Floods11 and returns the path
of the best checkpoint; `weights_path=` then points a later adapter at it. `Sen1Floods11Config(...).to_terratorch_yaml("config.yaml")` writes the
data configuration for TerraTorch's own `terratorch fit --config ...` command. These are the routes that adapt the model to radar; **they were not
run here** (no GPU, no dataset). Read the configuration before you start, and check the result on flood events the model has not seen, against a
reference map you trust.

Which class? Two classes are called `SARPrithviAdapter`:

| | `adapters.sar_prithvi.SARPrithviAdapter` (**use this one**) | `adapters.sar_channel_manager.SARPrithviAdapter` (older copy) |
|---|---|---|
| `zero_shot` | VH threshold, `method == "vh_threshold"` | The same threshold, now also labelled `method == "vh_threshold"`, and it adds a `probability` map (`1 − VH`) |
| `fine_tune`, `spt` | TerraTorch route (needs TerraTorch) | **`NotImplementedError`**: it used to fall back to the threshold under the fine-tune name, which hid the fact that nothing was fine-tuned |
| `prepare_finetuning()` | not present (use `fine_tune`) | Returns a configuration dictionary: batch size, learning rates, loss and the Sen1Floods11 statistics |

The Sen1Floods11 normalisation constants live in `sar_channel_manager`: `SEN1FLOODS11_VV_MEAN = -11.79` dB, `SEN1FLOODS11_VV_STD = 5.75`,
`SEN1FLOODS11_VH_MEAN = -17.07`, `SEN1FLOODS11_VH_STD = 6.29`. They are for dB values; they do not apply to data already normalised to [0, 1].

## DINOv3 with radar

DINOv3 expects a three-channel picture. `sar_to_pseudo_rgb(vv, vh, arrangement=...)` builds one:

| `arrangement` | R, G, B |
|---|---|
| `vv_vh_ratio` (default) | VV, VH, VV/VH ratio (rescaled) |
| `physics_rgb` | VV, (VV+VH)/2, VH |
| `vv_vh_diff` | VV, VH, |VV−VH| |
| `vv_vh_geomean` | VV, VH, √(VV·VH) |

Two classes are called `SARDINOv3Adapter`. They differ, and it matters which you import:

| | `pygeovision.models.adapters.sar_channel_manager.SARDINOv3Adapter` | `pygeovision.models.adapters.sar_dinov3.SARDINOv3Adapter` |
|---|---|---|
| Backbone | The package's real DINOv3 loader (`facebook/dinov3-*`, **gated**: accept the licence and sign in) | `facebookresearch/dinov2` from `torch.hub` (a DINOv2 ViT, despite the class name) |
| If the backbone cannot load | `RuntimeError: Real DINOv3 feature extraction failed (...)`. With `allow_pca_fallback=True` you get a 3-number PCA proxy under the same key names (3 values, not 1024), plus a `warning` | `RuntimeError: The DINO backbone could not be loaded ...`. With `allow_mock=True` you get all-zero arrays of the right shape with `is_mock: True` |
| Use it for | DINOv3 features of radar | Plumbing tests, or when you want the DINOv2 weights |

Both refuse to hand back placeholder numbers by default, because a vector of zeros that looks like a feature vector would pass every shape
check and poison everything downstream. The second class used to return zeros without raising; that was changed, with a test.

```python
from pygeovision.models.adapters.sar_channel_manager import SARDINOv3Adapter

feats = SARDINOv3Adapter(pseudo_rgb_arrangement="vv_vh_ratio").extract_features(vv, vh)
feats["cls_token"], feats["patch_features"], feats["pseudo_rgb"]      # needs the gated weights and a network
```

## Training with radar: speckle augmentation

Ordinary optical augmentation (brightness, contrast, colour jitter) is wrong for radar: speckle is **multiplicative**. `SARSpeckleAugmentation`
multiplies a **linear-power** chip by Gamma noise with the number of looks drawn at random. Measured on a chip of ones (mean 1):

| Looks | Spread of the result | Theory (1/√looks) |
|---:|---:|---:|
| 1 | 1.001 | 1.000 |
| 4 | 0.500 | 0.500 |
| 8 | 0.355 | 0.354 |

```python
from pygeovision.models.adapters.sar_channel_manager import SARSpeckleAugmentation
aug = SARSpeckleAugmentation(n_looks_range=(2, 8))
noisy = aug(chip_linear)            # chip (C, H, W) in LINEAR power; apply before normalising to dB and [0, 1]
```

Apply it **before** converting to decibels and normalising, in the data pipeline, together with random flips and 90° rotations (radar has no
"up"). The class in `sar_dinov3` has the same idea with `simulate_gamma_speckle(image, n_looks)` and `simulate_multilook(image, n_looks)` (a
small-window mean) and an extra `noise_strength` setting that is not used by the Gamma draw.

## Aligning a before-and-after pair

`coregister_sar_pair(reference_path, secondary_path, output_dir, method="phase_correlation", refine_subpixel=True)` resamples the second
raster onto the first one's exact grid and optionally refines the residual shift by FFT phase correlation. It writes
`reference_aligned.tif` and `secondary_aligned.tif` in `output_dir` and returns both paths. Use it before differencing two dates or feeding a pair to a
change model.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `Values [-3.9, 5.5] outside expected [0.0, 1.0]` warning | dB or z-scored values reached the mapping | Normalise to [0, 1] first (step 1) |
| `Spatial size … below minimum 64×64` | Chip too small for ViT patches | Use a larger chip |
| `prediction` is `None` | Validation failed; see `warnings` | Fix what it names |
| Flood map includes hills and runways | A threshold on backscatter cannot tell radar shadow from water | Use [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md), which uses terrain to remove them |
| You want real Prithvi radar results | Zero-shot is a threshold | Fine-tune on labelled radar (Sen1Floods11 or your own) |
| `ImportError: Mode 'fine_tune' requires TerraTorch` | TerraTorch is not installed | `pip install terratorch` |
| `RuntimeError: ... backbone could not be loaded` | No network, no licence for the gated weights, or no `weights_path` | Fix access; do not switch on the mock for real work |
| `VH not provided — approximating from VV` in the log | `vh` was left out | Pass the real VH |
| Edges and speckle look wrong after resizing | Speckle was filtered on dB values | Filter linear power, then convert to dB |

## Related

- [Radar data preparation](../data/sar.md), [Foundation models overview](index.md), [Prithvi-EO](prithvi.md), [DINOv3](dinov3.md)
- [`sar_flood_mapping`](../pipelines/catalog/sar_flood_mapping.md), [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md)
