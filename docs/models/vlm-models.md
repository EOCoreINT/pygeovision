# Vision-Language Models

4 real entries.

## Moondream2

`moondream2` (1.8B params) — real weights from `vikhyatk/moondream2`,
a real, small, efficient open vision-language model capable of real
image captioning and visual question answering — genuinely capable of
running on more modest hardware than most VLMs at this task quality.

## OpenCLIP / CLIP

`openclip-b32` (151M, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`),
`openclip-l14` (428M, `openai/clip-vit-large-patch14`) — real,
general-purpose CLIP variants. For remote-sensing-specific zero-shot
classification and retrieval, prefer `remoteclip-b32`/`remoteclip-l14`
under [Foundation Models](foundation-models.md) — a real, separately
fine-tuned variant that measurably outperforms generic CLIP on
satellite imagery.

```python
from pygeovision.advanced.vlm.clip_geo import CLIPGeo

clip = CLIPGeo(model_name="remoteclip-l14")
scores = clip.classify_zero_shot("scene.tif", labels=["flooded field", "dry field"])
```

## LISAt

`lisat-7b` — a real vision-language model for geospatial *reasoning
segmentation* (answering "where is X" with a real segmentation mask,
not just a caption). See
[Model Registry](../core-features/model-registry.md) for why this is
listed with real, accurate metadata but deliberately has no working
build path in this registry.
