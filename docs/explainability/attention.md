# Attention maps

**Class:** `AttentionMapExtractor` (`pygeovision.explainability`) · **Python:** `client.xai.attention(model)`

## Why

A Vision Transformer (ViT, SegFormer, DINOv2, Prithvi, the SAM encoder) cuts the image into patches and lets every patch
"attend" to every other. The attention weights show **which patches the model combined to form a representation**. For a
quick check that a transformer looks at the object rather than at the background, they are the transformer's counterpart
to GradCAM.

```{warning}
Attention is **not** importance. High attention means "this patch contributed a lot to the mix", not "this patch decided
the class". Treat attention maps as a hint about where to look, and confirm with another method (for example by masking the
patches and re-running the model).
```

## How to use

```python
from pygeovision.explainability import AttentionMapExtractor

ex = AttentionMapExtractor(model)
attn = ex.extract(chip, layer_idx=-1)            # chip: tensor/array (C, H, W) or (1, C, H, W)
print(attn.shape)                                # (4, 4): the patch grid, not pixels
ex.visualise("scene.tif", attn, "attention.png") # resized to the image, overlaid
```

### `AttentionMapExtractor(model, device=None)`

`device` defaults to the device of the model's parameters (so a model on the GPU works without arguments).

### `extract(image, layer_idx=-1, head_idx=None, *, forward_kwargs=None, strict=False)`

| Argument | Meaning |
|---|---|
| `image` | Tensor or array, `(C, H, W)` or `(1, C, H, W)`, preprocessed as in training |
| `layer_idx` | Which captured attention layer, in forward order. `-1` is the last (closest to the output) |
| `head_idx` | One attention head, or `None` to average over heads |
| `forward_kwargs` | Extra arguments for the model call, for example `{"output_attentions": True}` |
| `strict` | `True` raises on failure; `False` (default) logs a warning and returns zeros |

How the attention is found: forward hooks are placed on every module whose name contains `attn` or `attention`. A module
counts if it returns `(output, weights)` with weights shaped `(B, heads, N, N)` or head-averaged `(B, N, N)` (this is what
`torch.nn.MultiheadAttention` does). If the hooks catch nothing, the extractor looks for an `.attentions` tuple on the model
output (the Hugging Face convention). The hooks are removed after every call, and nested modules that return the same
tensor are counted once.

What you get back:

| Token layout | Result |
|---|---|
| `n × n` patch tokens, no class token | `(n, n)`: how much attention each patch **receives** on average from all patches |
| One class token + `n × n` patches | `(n, n)`: how much attention the **class token pays** to each patch (the usual map for ViTs) |
| Any other count of tokens | A 1-D vector, with a log warning |

The map has the resolution of the patch grid: a 64-pixel input with 16-pixel patches gives `(4, 4)`. With 224-pixel
inputs and 16-pixel patches it is 14 × 14. `visualise` enlarges it.

### `visualise(image_path, attention, output_path, alpha=0.6, colormap="jet")`

Writes the same three-panel PNG as [GradCAM](gradcam.md#picture-visualise) (input, map, overlay) after scaling the map
to 0–1. The attention must be 2-D, otherwise `ValueError`.

## Real output

A small `nn.MultiheadAttention` network on a 3-band 32-pixel chip (8-pixel patches, so 4 × 4 tokens), and a randomly
initialised Hugging Face `ViTModel` (64-pixel input, 16-pixel patches, a class token):

```text
MultiheadAttention model          -> (4, 4)   values sum to 1.0     (each patch's received attention)
ViTModel with a class token       -> (4, 4)   class token -> patches
head_idx=0 on the MHA model       -> (4, 4)
head_idx=1 on the MHA model       -> IndexError: head_idx 1 out of range: this layer's attention has 1 head(s)
                                     (modules that return head-averaged weights report a single head)
model with no attention at all    -> zeros of the input's height and width, plus a warning
  ... with strict=True            -> RuntimeError: no attention weights were produced. ...
```

These models are untrained, so the *values* mean nothing; the point is the shapes and the failure behaviour. This
extractor has not been run against the pretrained weights of Prithvi, SAM or DINOv2 in this documentation's tests.

## Which models work

| Model | Works? | Notes |
|---|---|---|
| Modules returning `(out, weights)` (`nn.MultiheadAttention`) | Yes | `weights` are head-averaged by default, so `head_idx` is limited to 0 |
| Hugging Face ViT/SegFormer with `output_attentions` | Yes | Pass `forward_kwargs={"output_attentions": True}` if you get zeros; load with `attn_implementation="eager"` (fused attention kernels do not expose the weights) |
| Models that compute attention inside a fused function (`F.scaled_dot_product_attention`) | No weights available | Re-load the model with eager attention |
| Pure CNNs | No | Use [GradCAM](gradcam.md) |

Only square `N × N` attention matrices are accepted. Attention that uses a reduced number of keys (as hierarchical
models with spatial reduction do) produces non-square matrices and is skipped, so the extractor may find fewer layers
than the model has.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| All zeros and a warning "no attention weights were produced" | Weights not exposed | `forward_kwargs={"output_attentions": True}`, eager attention; run with `strict=True` for the exact error |
| Zeros and "Attention extraction failed: …" | The model call raised | `strict=True` shows the traceback; check input channels and size |
| A 1-D vector | Token count is not a square (+1) | Use the raw weights from the model yourself, or crop the image to a size the patch grid divides |
| `IndexError: head_idx … out of range` | The layer has fewer heads than you asked for | Use `head_idx=None` or a valid index |
| `layer_idx` error | Fewer attention layers captured than expected | The extractor lists only layers that return weights; check how many you got with the default `-1`/`0` |
| Map is blocky | Patch grid resolution | Expected; use a smaller patch size or GradCAM-like methods for finer maps |
| Map highlights the image corner | Position bias or untrained model | Compare with a trained model; check padding |
| Hooks seem to accumulate | They no longer do: they are removed after every call | Update to the current version |
| Model and input on different devices | Old default was CPU | The default is now the model's own device |

```{seealso}
[GradCAM](gradcam.md) · [Foundation models](../foundation-models/index.md)
```
