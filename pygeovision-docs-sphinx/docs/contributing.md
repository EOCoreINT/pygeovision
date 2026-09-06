# Contributing

## Development setup

```bash
git clone https://github.com/<org>/pygeovision.git
cd pygeovision
pip install -e ".[all,dev]"
pytest
```

## The standard this codebase is held to

Every real fix documented on this site followed the same process:
read the actual code (not the docstring), test the actual behavior
(not the described behavior), and verify any numerical claim against
an independent hand-calculation or known ground truth before trusting
it. A model registry entry, a pipeline, or a "success" flag is only
called real once it's been checked this way.

If you're adding a new model to the registry, that means:

- A real `hf_id`/`timm_id` that you have personally confirmed points
  to the model you're claiming it is (not a same-family model, not a
  different generation of the same architecture family).
- If no real, verified build path exists yet, add the entry with
  accurate metadata and a clear `NotImplementedError` explaining
  exactly what's missing — see `lisat-7b` in
  `pygeovision.models.registry` for the pattern.

## Good first issues

- The ~700 lines of confirmed-unused code in the training package
  (see [Roadmap](reference/roadmap.md)) — either wire it in or remove
  it.
- `tree_species` has no real, verified classification model behind it.

## License

See `LICENSE` in the repository root.
