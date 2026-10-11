# Authentication

Three providers need **no account** and are searched by default: Microsoft Planetary Computer, AWS Earth Search
(`aws_earth`) and Element 84 (`element84`, the same AWS Earth Search service). With them you can run every *ready*
pipeline. Add an account only when you need something they do not carry (full Copernicus archive, commercial
imagery, NASA collections, Earth Engine).

## Storing credentials

Credentials are stored in your **operating-system keyring** through pygeofetch, not in the project folder, and
not in the PyGeoVision config file.

::::{tab-set}
:::{tab-item} CLI
```bash
pygeovision data auth add usgs        --username YOU --password '…'
pygeovision data auth add planet      --api-key PL_KEY
pygeovision data auth add copernicus  --client-id ID --client-secret SECRET
pygeovision data auth list                 # which providers have credentials
pygeovision data auth test copernicus      # does the login work?
pygeovision data auth remove usgs --yes
```
:::
:::{tab-item} Python
```python
client = pgv.PyGeoVision()
client.add_credentials("usgs", username="YOU", password="…") \
      .add_credentials("planet", api_key="PL_KEY") \
      .add_credentials("copernicus", client_id="ID", client_secret="SECRET")

client.data.list_credentials()          # ['usgs', 'planet', 'copernicus']
client.data.test_provider("planet")     # True / False (reason in the log)
client.data.remove_credentials("usgs")
```
:::
::::

## Which kind of credential each provider uses

| Auth type | Providers | Pass |
|---|---|---|
| none | `planetary_computer`, `aws_earth`, `element84`, `esa_scihub`, `jaxa_earth`, `noaa_big_data`, `isro_bhuvan`, `inpe_cbers`, `digitalglobe` | nothing |
| username + password | `usgs` | `--username`, `--password` |
| OAuth2 client | `copernicus`, `sentinel_hub`, `nasa_earthdata`, `nasa_earthdata_cloud` | `--client-id`, `--client-secret` (NASA: username and password of your Earthdata login) |
| API key | `planet`, `opentopography`, `airbus_oneatlas`, `terrabotics` | `--api-key` |
| other | `maxar_gbdx` (token), `alaska_satellite_facility` (Earthdata login), `google_earth_engine` (service account), `geoserver_generic` (configurable) | see the provider's own documentation |

The full per-provider table, with collections and capabilities, is in [Providers](providers.md).

```{note}
Open data does not mean unlimited. Providers may rate-limit anonymous clients; PyGeoVision retries with back-off and
falls back to another provider where it can.
```

## Testing

`pygeovision data auth test <provider>` (or `client.data.test_provider`) does the right check for each kind:

- **Open providers** — an HTTP request to the service's endpoint must succeed.
- **Providers with accounts** — the stored credentials must produce a valid session.

It returns `False`, and logs the reason, when no credentials are stored, the login is rejected, or the service is
unreachable. `client.data.test_copernicus_auth()` does the same for the Copernicus identity service.

## Searching a provider that needs credentials

Name it explicitly — default searches skip accounts you have not set up:

```python
client.search(bbox=bbox, date_range=("2024-06-01", "2024-06-30"),
              providers=["copernicus"], collections=["SENTINEL-2"])
```

```bash
pygeovision data search --bbox … --date 2024-06 --providers copernicus,usgs
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No stored credentials` in the log, provider skipped | Default searches skip providers without credentials | `data auth add …`, then name it in `--providers` |
| `auth test` fails with 401 | Wrong secret, or the account is not activated | Re-add; log in once on the provider's website |
| Keyring errors on a server or container | No desktop keyring service | Install a keyring backend (for example `keyrings.alt`) or set up `secretstorage`/`dbus`; or use an open provider |
| Works in the CLI, not in a script | Different user or environment | The keyring belongs to the OS user that ran `auth add` |
