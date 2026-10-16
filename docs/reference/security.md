# Security

What PyGeoVision stores, what it executes, and what to treat as untrusted. PyGeoVision is a library and a command-line tool that runs on
your machine with your permissions.

## Credentials

Provider credentials are stored in your **operating-system keyring** through pygeofetch, never in the project folder and never in the
PyGeoVision configuration file. Three providers need no account at all ([authentication](../data/authentication.md)).

- `pygeovision data auth list` shows which providers have credentials, not the secrets.
- `pygeovision data auth remove <provider> --yes` deletes them.
- Do not put keys in YAML workflows, notebooks or command lines you share. Shell history keeps `--password` and `--api-key` values;
  add the credential once with `data auth add` and refer to the provider by name afterwards.
- `pygeovision doctor` reports only a **count** of providers with stored credentials.

## Files you load are code-adjacent

| Input | Risk | Rule |
|---|---|---|
| **Model checkpoints** (`.pt`, `.pth`) | PyTorch checkpoints are pickles; an untrusted file can run code when loaded | Load only checkpoints you trained or downloaded from a source you trust. Pipelines, the model hub and training resume load with `weights_only=True`. Two other loaders (the trainer's `checkpoint_path` and `training/checkpoint.py`) rely on PyTorch's default, which is `weights_only=True` from PyTorch 2.6 on and a full pickle before that |
| **OBIA rules files** | Rule strings are evaluated with `pandas.DataFrame.eval` | Treat a rules file like code from the person who runs it, not like untrusted input |
| **YAML workflows** | Parsed with `yaml.safe_load`, so they cannot construct arbitrary objects | They still choose which pipelines run and where output is written; read a workflow before you run it |
| **Vector and raster inputs** | Malformed files can crash GDAL-based readers | Do not process files from unknown sources in a privileged account |

## Network access

PyGeoVision contacts only the services you ask for: the data providers you search, Hugging Face or the weight URL when you download
pretrained weights, OpenStreetMap's Overpass service for the OSM labelers and `obia.fetch_osm_polygons`, and the cloud endpoint when you deploy.
Nothing is uploaded unless you run a cloud-deployment command. Searches are cached on disk for an hour by default
([configuration](../getting-started/configuration.md)).

## Reports and outputs

Reports (PDF, Word, Excel), maps and vectors are written to the output folder you name. They contain the bounding box, the dates and
the scenes used. Treat them with the sensitivity of the location they describe before you share them.

## Reporting a vulnerability

Do not open a public issue for a security problem. Contact the maintainers privately (the repository's private vulnerability reporting, or a direct message to a maintainer), describe the impact and a way to reproduce it, and allow time for a fix before disclosure.
