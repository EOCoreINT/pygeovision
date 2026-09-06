# Data & Authentication

`pygeovision data`/`auth` are real, direct delegations to `pygeofetch`
— confirmed by reading the source (`client.add_credentials()`'s own
docstring: "Delegates to `pygeofetch auth add PROVIDER ...` for secure
storage"). This isn't pygeovision-specific functionality re-implemented
here; it's `pygeofetch`'s real, already-documented credential and
search system, exposed through the same top-level CLI for convenience.

```bash
pygeovision data auth add usgs --username USER --password PASS
pygeovision data auth add planet --api-key PL_KEY
pygeovision data auth list
pygeovision data search --bbox ... --date-range ... --providers planetary_computer,copernicus
pygeovision data download --search-results ./results.json --output ./scenes/
```

```{seealso}
For the real, complete detail on providers, auth modes, and search
flags, see pygeofetch's own documentation directly:
[Authentication](https://pygeofetch.readthedocs.io/en/latest/core-features/authentication.html),
[Searching Satellite Data](https://pygeofetch.readthedocs.io/en/latest/core-features/search.html),
[Providers](https://pygeofetch.readthedocs.io/en/latest/core-features/providers.html).
Re-documenting it here would only risk it going stale relative to the
real source of truth.
```

`pygeovision cache stats`/`cache clear` and `pygeovision status`/
`doctor` are also thin, real diagnostic wrappers. `doctor` specifically
runs real diagnostics on the `pygeofetch` data layer (provider
connectivity, credential status) — it delegates directly to
`client.data.doctor()` per its own docstring. It does not check
whether optional AI/ML dependencies like `torch` are installed; a
model-build failure due to a missing package will show up as a real,
specific `ImportError` from the failing call itself, not from `doctor`.

```bash
pygeovision doctor
```
