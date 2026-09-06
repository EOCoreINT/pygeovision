# Model Catalog (Quick Reference)

```python
from pygeovision.models.registry import list_models
from pygeovision.ai.models.registry import registry

list_models()                    # 63 real entries, general-purpose registry
[m.name for m in registry.list_models()]  # 14 real entries, fully offline
```

See [Model Registry](../core-features/model-registry.md) for what
"real" means, the honest entries with no verified build path, and the
real fallback mechanism.

## Browsing without building

`pygeovision.ai.models.zoo.model_zoo` is a real, separate, 97-entry
catalog — pure metadata for discovery (`filter()`, `search()`,
`print_table()`), with no `build()`/`load()` method at all. An audit
found 77 of 98 original entries claimed `pretrained_available=True`
with no real `hf_model_id` to back the claim; this is now corrected to
match reality.

```bash
pygeovision models zoo list --task segmentation
pygeovision models zoo search dinov2
```
