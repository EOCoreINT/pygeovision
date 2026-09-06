# Installation

## Basic install

```bash
pip install pygeovision
```

This installs the core package: the AI pipeline system, the model
registry/hub, and the CLI. It does **not** install PyTorch,
`segmentation-models-pytorch`, or any of the real model backends —
those are real, separate, often large dependencies that pull in
whatever compute backend (CPU/CUDA/MPS) is right for your machine.

## Extras

```bash
# Real PyTorch-based training and inference
pip install pygeovision[torch]

# Real model backends: segmentation-models-pytorch, torchvision detection,
# transformers (for hf_id-backed foundation models)
pip install pygeovision[models]

# Real SAM-geospatial integration (segment-geospatial / samgeo)
pip install pygeovision[samgeo]

# Everything
pip install pygeovision[all]
```

```{note}
`segment-geospatial` is a real, separate, peer-reviewed package (Wu &
Osco, 2023, *JOSS*) that `pygeovision` wraps rather than reimplements
for its `SamGeoLabeler`. If you only need `pygeovision`'s own
from-scratch SAM wrapper (`SAMAutoLabeler`), you don't need this extra.
```

## Development install

```bash
git clone https://github.com/<org>/pygeovision.git
cd pygeovision
pip install -e ".[all,dev]"
```

## Verifying the install

```bash
pygeovision --version
pygeovision models list
```

`pygeovision models list` should show 77 real entries across the two
model registries (63 in the general-purpose registry, 14 in the
smaller, fully offline-buildable native registry) — see
[Model Registry](../core-features/model-registry.md) for what "real"
means here and what was removed to get to that number.

```{warning}
Some registry entries have a real, correct `hf_id` but genuinely need
network access to HuggingFace to actually build (they're not fake --
see the Model Registry page for the real distinction). If
`pygeovision models list` shows them but you can't build them offline,
that's expected, not a bug.
```
