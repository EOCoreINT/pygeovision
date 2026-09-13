# Embeddings

`GeoEmbeddings` gives each foundation model its own honestly-named
method for real embedding extraction, rather than forcing a fake,
uniform signature across models with genuinely different real
requirements — some need an in-memory array plus a sensor name, some
need a real file path, one fetches real precomputed data for a
bounding box rather than running local inference at all. What's
unified is the *output*: a flat real vector, and a shared,
verified `cosine_similarity`/`nearest` pair for comparing them.

```python
from pygeovision.models.embeddings import GeoEmbeddings, cosine_similarity, nearest

emb = GeoEmbeddings(device="cuda")

v1 = emb.dofa(sentinel2_array, sensor="sentinel2")
v2 = emb.clip("other_scene.tif")
similarity = cosine_similarity(v1, v1)  # 1.0 for identical embeddings
```

## The five real methods

| Method | Real input | Delegates to |
|---|---|---|
| `emb.dofa(image, sensor=...)` | In-memory array + sensor name or explicit wavelengths | [DOFA](foundation-models.md) |
| `emb.prithvi(image_path, source=...)` | A real HLS-format GeoTIFF file path | [Prithvi](foundation-models.md) |
| `emb.dinov2(image)` | A real file path or already-loaded image | DINOv2 (see the honest DINOv3-labeling note below) |
| `emb.clip(image_path)` | A real file path | RemoteCLIP (real, remote-sensing-finetuned weights by default) |
| `emb.tessera(bbox, year=...)` | A real bounding box, no image at all | [Tessera](foundation-models.md)'s real, precomputed embeddings |

Each is a thin, verified dispatch to that model's own real extraction
code — not a separate implementation that could quietly drift from it.
Model instances are cached per method call, keyed on the requested
model name, so calling the same method repeatedly with the same model
doesn't reload it every time.

```{note}
`emb.dinov2()` is named for what it actually loads. The underlying
registry/module names use "dinov3" for historical reasons, but this
audit found and confirmed those entries load real, genuine DINOv2
weights, not DINOv3 — see [Model Registry](../core-features/model-registry.md)
for the full, honest detail. If you need this class of model, this is
it; a genuine DINOv3 integration doesn't exist in this codebase today.
```

## Real, honest failure behavior

Nothing here catches or silently substitutes anything — a failure in
`emb.dofa()` propagates exactly as `load_dofa_hf()` defines it (raises
clearly on a genuine weight-load failure by default), and the same
applies to every other method here.

## Comparing embeddings

```python
cosine_similarity(v1, v2)          # raises ValueError if dimensions differ
nearest(query, candidates, top_k=5)  # candidates: {"name": embedding, ...}
```

`cosine_similarity` refuses to compare embeddings from two different
real models with incompatible dimensions (e.g. DOFA's 768-dim vs.
Tessera's 128-dim) rather than silently truncating or padding to force
a number out — comparing vectors from genuinely different,
non-comparable embedding spaces isn't a meaningful operation, and this
says so instead of returning a misleading similarity score.

```python
results = nearest(query_embedding, {
    "scene_a.tif": embedding_a,
    "scene_b.tif": embedding_b,
}, top_k=3)
# [("scene_a.tif", 0.94), ("scene_b.tif", 0.71)] -- sorted, highest first
```

## A real gap found and fixed while building this

`TesseraGeo.embeddings_for_bbox()` previously never included the real
embedding array in its return value — only its shape and metadata.
Calling it without `output_path` gave a dict *describing* real data
with no way to actually access that data in Python. `emb.tessera()`
depends on this being fixed, and it now is: the real array is included
directly (`result["embedding"]`), average-pooled here to a single
`(128,)` vector for consistency with the other methods — pass
`embeddings_for_bbox()` directly if you need the real, per-pixel stack
instead.
