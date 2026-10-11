"""
PyGeoVision CLI — production-ready geospatial AI command line interface.

Integrates:
  pygeofetch — satellite data (search, download, auth, pipeline, cache)
  native AI  — segment, detect, classify, change, train (no external AI platform)

Usage:
    pygeovision --help
    pygeovision status
    pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --date 2024-06
    pygeovision data download --from-search results.geojson --output ./data/
    pygeovision data auth add usgs --username USER --password PASS
    pygeovision data auth add planet --api-key YOUR_KEY
    pygeovision data pipeline run weekly.yaml
    pygeovision ai segment buildings --input scene.tif --output buildings.tif
    pygeovision ai detect cars --input aerial.tif --output cars.geojson
    pygeovision ai train segmentation --data ./chips/ --output model.pth
    pygeovision pipeline building_footprints --bbox ... --output ./results
    pygeovision models list
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    from pygeovision._logging import configure_logging

    configure_logging(logging.DEBUG if verbose else logging.INFO, datefmt="%H:%M:%S")


@click.group()
@click.version_option(package_name="pygeovision")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """PyGeoVision — Geospatial AI Platform (pygeofetch + native AI)."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# =========================================================================
# DATA commands — delegates to pygeofetch CLI
# =========================================================================

@cli.group()
def data() -> None:
    """Satellite data commands — powered by pygeofetch.

    \b
    All 22 providers: planetary_computer, copernicus, usgs, nasa_earthdata,
    planet, sentinel_hub, maxar_gbdx, airbus_oneatlas, aws_earth, element84,
    noaa_big_data, alaska_satellite_facility, opentopography, google_earth_engine,
    terrabotics, esa_scihub, jaxa_earth, isro_bhuvan, inpe_cbers, digitalglobe,
    nasa_earthdata_cloud, geoserver_generic
    """


@data.group()
def auth() -> None:
    """Manage satellite provider credentials (stored in system keyring)."""


@auth.command("add")
@click.argument("provider")
@click.option("--username", help="Username (usgs, nasa_earthdata).")
@click.option("--password", help="Password (usgs, nasa_earthdata).")
@click.option("--api-key", help="API key (planet, opentopography, airbus_oneatlas).")
@click.option("--client-id", help="OAuth2 client ID (copernicus, sentinel_hub, maxar_gbdx).")
@click.option("--client-secret", help="OAuth2 client secret.")
def auth_add(provider, username, password, api_key, client_id, client_secret):
    """Add credentials for a satellite data provider.

    \b
    Examples:
        pygeovision data auth add usgs --username USER --password PASS
        pygeovision data auth add planet --api-key PL_KEY
        pygeovision data auth add copernicus --client-id ID --client-secret SECRET
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    client.add_credentials(
        provider,
        username=username, password=password,
        api_key=api_key, client_id=client_id, client_secret=client_secret,
    )
    click.echo(f"✓ Credentials added for '{provider}'")


@auth.command("list")
def auth_list():
    """List providers with stored credentials."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    providers = client.data.list_credentials()
    if not providers:
        click.echo("No credentials stored.")
    else:
        click.echo("Stored credentials for:")
        for p in providers:
            click.echo(f"  • {p}")


@auth.command("test")
@click.argument("provider")
def auth_test(provider):
    """Test connectivity to a provider."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ok = client.test_provider(provider)
    if ok:
        click.echo(f"✓ {provider}: reachable")
    else:
        click.echo(f"✗ {provider}: unreachable or not authenticated", err=True)
        sys.exit(1)


@auth.command("remove")
@click.argument("provider")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def auth_remove(provider, yes):
    """Remove stored credentials for a provider."""
    if not yes:
        click.confirm(f"Remove credentials for '{provider}'?", abort=True)
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    client.data.remove_credentials(provider)
    click.echo(f"✓ Credentials removed for '{provider}'")


@data.command("search")
@click.option("--bbox", nargs=4, type=float, required=True,
              metavar="MIN_LON MIN_LAT MAX_LON MAX_LAT",
              help="Bounding box in WGS84.")
@click.option("--date", "date_str", default=None,
              help="Date (YYYY-MM) or range (YYYY-MM-DD/YYYY-MM-DD).")
@click.option("--start-date", default=None, help="Start date YYYY-MM-DD.")
@click.option("--end-date", default=None, help="End date YYYY-MM-DD.")
@click.option("--providers", default=None,
              help="Comma-separated provider IDs (e.g. planetary_computer,copernicus).")
@click.option("--satellite", default=None,
              help="Satellite shortcut: sentinel-2, landsat, planet, dem, etc.")
@click.option("--collections", default=None,
              help="STAC collection IDs (e.g. sentinel-2-l2a,landsat-c2-l2).")
@click.option("--cloud-max", default=30.0, show_default=True, help="Max cloud cover (percent).")
@click.option("--max-results", default=50, show_default=True)
@click.option("--sort-by", default="datetime",
              type=click.Choice(["datetime", "cloud_cover", "score", "satellite"]))
@click.option("--output", "-o", default=None, help="Save results to GeoJSON file.")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json", "ids"]))
def data_search(bbox, date_str, start_date, end_date, providers, satellite,
                collections, cloud_max, max_results, sort_by, output, fmt):
    """Search satellite imagery across 22+ pygeofetch providers.

    \b
    Examples:
        # Open access — no credentials needed
        pygeovision data search --bbox -0.15 51.47 -0.10 51.52 \\
            --date 2024-06 --collections sentinel-2-l2a --cloud-max 10

    \b
        # Multi-provider with credentials
        pygeovision data search --bbox -74.1 40.6 -73.7 40.9 \\
            --start-date 2024-01-01 --end-date 2024-06-01 \\
            --providers planetary_computer,copernicus,usgs --cloud-max 20

    \b
        # Satellite shortcut
        pygeovision data search --bbox -74.1 40.6 -73.7 40.9 \\
            --date 2024-06 --satellite sentinel-2
    """
    from pygeovision import PyGeoVision

    # Resolve date range
    if date_str:
        if "/" in date_str:
            start, end = date_str.split("/", 1)
        else:                     # YYYY-MM (whole month) or YYYY-MM-DD
            from pygeovision.data._dates import date_window
            start, end = date_window(date_str)
    else:
        start = start_date or "2024-01-01"
        end = end_date or "2024-12-31"

    prov_list = [p.strip() for p in providers.split(",")] if providers else None
    coll_list = [c.strip() for c in collections.split(",")] if collections else None

    client = PyGeoVision()
    click.echo(f"Searching pygeofetch providers... (bbox={tuple(bbox)}, dates={start}→{end})")

    results = client.search(
        bbox=tuple(bbox),
        date_range=(start, end),
        providers=prov_list,
        satellite=satellite,
        collections=coll_list,
        cloud_cover_max=cloud_max,
        max_results=max_results,
        sort_by=sort_by,
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(results)} scenes:\n")

    if fmt == "json":
        data_out = json.dumps([r.to_dict() for r in results], indent=2, default=str)
        click.echo(data_out)
        if output:
            Path(output).write_text(data_out)
    elif fmt == "ids":
        for r in results:
            click.echo(r.id)
        if output:
            Path(output).write_text("\n".join(r.id for r in results))
    else:
        # Table format
        click.echo(f"{'ID':<40} {'Provider':<22} {'Date':<12} {'Cloud':>6} {'Score':>6} {'Satellite'}")
        click.echo("-" * 110)
        for r in results:
            cc = f"{r.cloud_cover:.0f}%" if r.cloud_cover is not None else "N/A"
            score = f"{r.score:.2f}" if r.score else "N/A"
            click.echo(f"{r.id[:38]:<40} {r.provider:<22} {r.date:<12} {cc:>6} {score:>6} {r.satellite}")

        if output:
            # Save as GeoJSON
            features = []
            for r in results:
                features.append({
                    "type": "Feature",
                    "id": r.id,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [r.bbox[0], r.bbox[1]], [r.bbox[2], r.bbox[1]],
                            [r.bbox[2], r.bbox[3]], [r.bbox[0], r.bbox[3]],
                            [r.bbox[0], r.bbox[1]],
                        ]] if r.bbox else None,
                    },
                    "properties": r.to_dict(),
                })
            geojson = {"type": "FeatureCollection", "features": features}
            Path(output).write_text(json.dumps(geojson, indent=2, default=str))
            click.echo(f"\nSaved {len(results)} results → {output}")


@data.command("download")
@click.option("--from-search", "search_file", type=click.Path(exists=True),
              help="GeoJSON results file from 'data search'.")
@click.option("--bbox", nargs=4, type=float, default=None,
              metavar="MIN_LON MIN_LAT MAX_LON MAX_LAT",
              help="Download all results in this bbox (requires --date).")
@click.option("--date", "date_str", default=None)
@click.option("--providers", default=None)
@click.option("--output", "-o", default="./downloads", show_default=True)
@click.option("--parallel", default=4, show_default=True)
@click.option("--verify-checksum", is_flag=True, default=False)
@click.option("--resume", is_flag=True, default=True)
@click.option("--retry", default=5, show_default=True)
@click.option("--post-process", default=None,
              help="Post-processing chain: unzip,reproject:EPSG:4326,compress:lzw,cog")
@click.option("--bandwidth-limit", default=None, help="Download throttle in MB/s.")
@click.option("--on-failure", default="skip",
              type=click.Choice(["skip", "abort", "retry"]))
@click.option("--max-items", default=None, type=int)
def data_download(search_file, bbox, date_str, providers, output, parallel,
                  verify_checksum, resume, retry, post_process, bandwidth_limit,
                  on_failure, max_items):
    """Download satellite scenes via pygeofetch.

    \b
    Post-process actions (chained):
        unzip                    Extract ZIP/TAR archives
        reproject:EPSG:4326      Reproject to CRS
        compress:lzw             Apply compression (lzw/deflate/zstd)
        ndvi                     Compute NDVI band
        ndwi                     Compute NDWI band
        cog                      Cloud Optimized GeoTIFF
        resample:10              Resample to N metres
        clip:area.geojson        Clip to geometry
        merge                    Merge overlapping scenes
        pan-sharpen              Pan-sharpen multispectral
        atmospheric:sen2cor      Atmospheric correction

    \b
    Examples:
        pygeovision data download --from-search results.geojson \\
            --output ./data/ --parallel 4 \\
            --post-process unzip,reproject:EPSG:4326,compress:lzw,cog

    \b
        pygeovision data download --from-search results.geojson \\
            --output ./data/ --verify-checksum"""
    from pygeovision import PyGeoVision
    from pygeovision.data.fetch import SatelliteFetcher

    client = PyGeoVision()
    items = []

    if search_file:
        # Parse GeoJSON from search command
        fetcher = SatelliteFetcher()
        items = fetcher._parse_stac_geojson_file(Path(search_file))
    elif bbox and date_str:
        from pygeovision.data._dates import date_window
        start, end = date_window(date_str)
        prov_list = [p.strip() for p in providers.split(",")] if providers else None
        items = client.search(
            bbox=tuple(bbox), date_range=(start, end), providers=prov_list
        )
    else:
        click.echo("Provide --from-search or (--bbox + --date)", err=True)
        sys.exit(1)

    if not items:
        click.echo("No items to download.")
        return

    if max_items:
        items = items[:max_items]

    pp = [p.strip() for p in post_process.split(",")] if post_process else None

    click.echo(f"Downloading {len(items)} scenes → {output}...")
    results = client.download(
        items,
        output_dir=output,
        parallel=parallel,
        verify_checksum=verify_checksum,
        resume=resume,
        retry_attempts=retry,
        post_process=pp,
        bandwidth_limit_mb=float(bandwidth_limit) if bandwidth_limit else None,
        on_failure=on_failure,
    )

    ok = sum(1 for r in results if r.success)
    fail = len(results) - ok
    total_mb = sum(r.size_mb for r in results if r.success)

    click.echo(f"\n✓ {ok} succeeded ({total_mb:.1f} MB total)")
    if fail:
        click.echo(f"✗ {fail} failed", err=True)
    for r in results:
        click.echo(f"  {r}")
    if fail:
        sys.exit(1)


@data.group()
def pipeline() -> None:
    """Manage and run pygeofetch YAML data pipelines."""


@pipeline.command("run")
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option("--step", default=None, help="Run only this named step.")
def data_pipeline_run(yaml_file, step):
    """Run a pygeofetch YAML pipeline.

    \b
    Example pipeline (weekly-sentinel2.yaml):
        name: weekly-sentinel2-ndvi
        schedule: "0 6 * * 1"
        steps:
          - search:
              providers: [planetary_computer, copernicus]
              date_range: last_7_days
              cloud_cover: 0-10
              bbox: "-74.1,40.6,-73.7,40.9"
          - filter:
              expression: "data.cloud_cover < 5"
          - download:
              parallel: 4
              output: ./raw/
              verify_checksum: true
          - process:
              actions: [reproject:EPSG:4326, cog]
          - export:
              destination: s3://my-bucket/
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    click.echo(f"Running pipeline: {yaml_file}")
    try:
        result = client.run_pipeline_yaml(yaml_file, step=step)
    except Exception as exc:  # noqa: BLE001 -- bad YAML, unknown step, missing credentials ...
        raise click.ClickException(str(exc)) from exc
    if result.get("success"):
        click.echo(f"✓ Pipeline completed successfully ({result.get('duration_seconds', 0):.1f}s).")
        for st in result.get("steps", []):
            click.echo(f"  {st['step']}: {st['status']}")
    else:
        msg = result.get("error") or result.get("stderr") or "see log"
        click.echo(f"✗ Pipeline failed: {str(msg)[:300]}", err=True)
        sys.exit(1)


@pipeline.command("validate")
@click.argument("yaml_file", type=click.Path(exists=True))
def pipeline_validate(yaml_file):
    """Validate a pygeofetch pipeline YAML without running it."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ok = client.data.validate_pipeline(yaml_file)
    if ok:
        click.echo(f"✓ Pipeline valid: {yaml_file}")
    else:
        click.echo("✗ Pipeline invalid", err=True)
        sys.exit(1)


@pipeline.command("schedule")
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Schedule name.")
@click.option("--cron", default=None, help="Cron expression (e.g. '0 6 * * 1').")
def pipeline_schedule(yaml_file, name, cron):
    """Schedule a pygeofetch pipeline for recurring execution."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ok = client.data.schedule_pipeline(yaml_file, name=name, cron=cron)
    if ok:
        click.echo(f"✓ Pipeline scheduled: {name or yaml_file}")
    else:
        click.echo("✗ Scheduling failed: the pipeline YAML is invalid (see the log line above)", err=True)
        sys.exit(1)


@pipeline.command("list")
def pipeline_list():
    """List all scheduled pygeofetch pipelines."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    pipelines = client.data.list_scheduled_pipelines()
    if not pipelines:
        click.echo("No scheduled pipelines.")
    else:
        for p in pipelines:
            click.echo(f"  {p.get('name', 'unnamed')}: {p.get('schedule') or 'no cron (manual)'}  [{p.get('file', '')}]")


@pipeline.command("history")
@click.option("--limit", default=20, show_default=True)
def pipeline_history(limit):
    """Show pygeofetch pipeline run history."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    history = client.data.pipeline_history(limit=limit)
    if not history:
        click.echo("No pipeline history.")
    else:
        for run in history:
            mark = "✓" if run.get("success") else "✗"
            click.echo(f"  {mark} {run.get('started_at', '?')}  {run.get('pipeline')}  "
                       f"{run.get('duration_seconds', 0):.1f}s" + (f"  {run['error']}" if run.get("error") else ""))


@data.group()
def cache() -> None:
    """Manage pygeofetch search result cache."""


@cache.command("stats")
def cache_stats():
    """Show cache statistics."""
    from pygeovision import PyGeoVision
    stats = PyGeoVision().cache_stats()
    click.echo(f"Entries: {stats.get('entries', 0)}")
    click.echo(f"Size:    {stats.get('size_mb', 0):.1f} MB")
    click.echo(f"Location: {stats.get('location', 'N/A')}")


@cache.command("clear")
@click.option("--provider", default=None, help="Clear only this provider.")
@click.option("--older-than", default=None, help="Clear entries older than (e.g. 7d, 1h).")
@click.option("--dry-run", is_flag=True)
def cache_clear(provider, older_than, dry_run):
    """Clear the pygeofetch search result cache."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    client.data.clear_cache(provider=provider, older_than=older_than, dry_run=dry_run)
    if not dry_run:
        click.echo("✓ Cache cleared.")
    else:
        click.echo("(dry-run) Would clear cache.")


@data.command("providers")
@click.option("--open-only", is_flag=True, help="Only open-access providers.")
@click.option("--sar", is_flag=True, help="Only SAR-capable providers.")
@click.option("--sub-meter", is_flag=True, help="Only sub-metre resolution providers.")
def data_providers(open_only, sar, sub_meter):
    """List all 22 pygeofetch satellite data providers."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    caps = []
    if sar: caps.append("sar")
    if sub_meter: caps.append("sub_meter")
    providers = client.list_providers(open_only=open_only, capabilities=caps or None)
    click.echo(f"\n{'ID':<30} {'Name':<30} {'Auth':>8} {'SAR':>5} {'<1m':>5} {'STAC':>5}")
    click.echo("-" * 90)
    for pid, info in sorted(providers.items()):
        auth = "🔐" if info.get("auth") else "🌐"
        sar_mark = "✓" if info.get("sar") else ""
        sub_mark = "✓" if info.get("sub_meter") else ""
        stac_mark = "✓" if info.get("stac") else ""
        click.echo(f"{pid:<30} {info['name']:<30} {auth:>8} {sar_mark:>5} {sub_mark:>5} {stac_mark:>5}")
    click.echo(f"\nTotal: {len(providers)} providers")


# =========================================================================
# =========================================================================
# AI commands — native (no external AI-platform dependency)
# =========================================================================

@cli.group()
def ai() -> None:
    """Geospatial AI commands — fully native (SAM, YOLO, ChangeFormer, CLIP).

    \b
    Capabilities:
        segment  — Building footprints, water, general SAM segmentation
        detect   — Generic / ships / cars object detection (native YOLO)
        classify — Scene classification (CLIP), land cover (ESA WorldCover)
        change   — Bi-temporal change detection (ChangeFormer)
        train    — Train segmentation / detection models
        infer    — Tiled inference on large GeoTIFF scenes
        chips    — Export aligned imagery/label training chips
        cloud-mask — Native brightness/NDSI-style cloud masking
    """


def _load_user_model(model, checkpoint, num_classes, in_channels, raster=None):
    """Build ``model`` and load the user's trained ``checkpoint`` into it, strictly.

    Inference CLI commands run *your* trained network: the registry ships architectures, not trained
    weights, so without ``--checkpoint`` an architecture would answer with random output that looks
    like a result. No model is substituted and a mismatching checkpoint fails loudly.
    """
    if not checkpoint:
        raise click.UsageError(
            "--checkpoint is required: PyGeoVision ships architectures, not trained weights. Train one "
            "(pygeovision.ai.training.trainer.GeoTrainer) on labelled chips of your area and pass "
            "its .pth here; it must match --model, --classes/--num-classes and the band count.")
    if in_channels is None:
        if raster is None:
            raise click.UsageError("--in-channels is required (could not infer the band count)")
        import rasterio
        with rasterio.open(raster) as src:
            in_channels = src.count
    from pygeovision.ai.models.hub import ModelHub
    try:
        return ModelHub().load(model, checkpoint=str(checkpoint), num_classes=num_classes,
                               in_channels=in_channels, device="cpu")
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc


@ai.command("segment")
@click.argument("target", type=click.Choice(["buildings", "water", "sam", "custom"]))
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output mask GeoTIFF.")
@click.option("--model", default=None, help="Registry model name (target=custom), e.g. unet_resnet50.")
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Your trained weights for --model (target=custom; required there).")
@click.option("--num-classes", default=2, show_default=True, help="Classes the checkpoint was trained for.")
@click.option("--green-band", type=int, default=None,
              help="target=water: 1-based green band (default 2 of a blue-green-red-NIR stack).")
@click.option("--nir-band", type=int, default=None,
              help="target=water: 1-based near-infrared band (default 4).")
def ai_segment(target, input_path, output, model, checkpoint, num_classes, green_band, nir_band):
    """Segment geospatial features natively.

    \b
    Examples:
        pygeovision ai segment buildings --input sentinel2.tif --output buildings.tif
        pygeovision ai segment water --input s2.tif --output water.tif
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()

    click.echo(f"Running native {target} segmentation on {input_path}...")

    if target == "buildings":
        client.segmentation.buildings(input_path, output_path=output)
    elif target == "water":
        try:
            client.segmentation.water(input_path, output_path=output,
                                      green_band=green_band, nir_band=nir_band)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    elif target == "sam":
        client.segmentation.sam(input_path, output_path=output)
    elif target == "custom":
        if not model:
            click.echo("--model required for custom segmentation", err=True)
            sys.exit(1)
        m = _load_user_model(model, checkpoint, num_classes, None, raster=input_path)
        res = client.segmentation.custom(input_path, m, output_path=output, num_classes=num_classes)
        if isinstance(res, dict) and res.get("success") is False:
            raise click.ClickException(str(res.get("error")))

    click.echo(f"✓ Segmentation complete → {output}")


@ai.command("detect")
@click.argument("target", type=click.Choice(["generic", "ships", "cars", "custom"]))
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True,
              help="generic/ships/cars: GeoJSON file of boxes (use .geojson). custom: a GeoTIFF of per-pixel "
                   "class scores (the model must output one score map per class, not boxes).")
@click.option("--model", default=None, help="Registry model name (target=custom).")
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Your trained weights for --model (target=custom; required there).")
@click.option("--num-classes", default=2, show_default=True, help="Classes the checkpoint was trained for.")
def ai_detect(target, input_path, output, model, checkpoint, num_classes):
    """Detect objects in satellite/aerial imagery using native YOLO.

    \b
    Examples:
        pygeovision ai detect generic --input aerial.tif --output detections.geojson
        pygeovision ai detect ships --input port.tif --output ships.geojson
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()

    click.echo(f"Running native {target} detection on {input_path}...")

    if target in ("generic", "ships", "cars"):
        if Path(output).suffix.lower() in (".tif", ".tiff"):
            raise click.ClickException(
                f"'{target}' writes the detected boxes as GeoJSON, not a raster; "
                f"use a path ending in .geojson (got {output}).")
        res = getattr(client.detection, target)(input_path, output_path=output)
        if isinstance(res, dict) and res.get("error"):
            raise click.ClickException(str(res["error"]))
        click.echo(f"  {res.get('n_detections', 0)} object(s) found")
    elif target == "custom":
        if not model:
            click.echo("--model required for custom detection", err=True)
            sys.exit(1)
        m = _load_user_model(model, checkpoint, num_classes, None, raster=input_path)
        res = client.detection.custom(input_path, m, output_path=output, num_classes=num_classes)
        if isinstance(res, dict) and res.get("success") is False:
            raise click.ClickException(str(res.get("error")))

    click.echo(f"✓ Detection complete → {output}")


@ai.command("classify")
@click.argument("mode", type=click.Choice(["scene", "land-cover"]))
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--classes", default=None, help="Comma-separated class names (scene mode).")
@click.option("--output", "-o", default="./output/land_cover.tif",
              help="Output GeoTIFF (land-cover mode).")
def ai_classify(mode, input_path, classes, output):
    """Classify satellite imagery natively (CLIP zero-shot / ESA WorldCover).

    \b
    Examples:
        pygeovision ai classify scene --input tile.tif \\
            --classes "forest,water,urban,agriculture"
        pygeovision ai classify land-cover --input s2.tif --output lc.tif
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()

    if mode == "scene":
        class_list = [c.strip() for c in classes.split(",")] if classes else None
        result = client.classification.scene(input_path, categories=class_list)
        click.echo(f"Classification: {result}")
    elif mode == "land-cover":
        client.classification.land_cover(input_path, output_path=output)
        click.echo(f"✓ Land cover classification → {output}")


@ai.command("change")
@click.option("--before", required=True, type=click.Path(exists=True), help="Pre-event raster.")
@click.option("--after", required=True, type=click.Path(exists=True), help="Post-event raster.")
@click.option("--output", "-o", required=True, help="Output change mask GeoTIFF.")
@click.option("--method", default="spectral_diff", show_default=True,
              type=click.Choice(["spectral_diff", "changeformer", "bit", "dsamnet"]),
              help="spectral_diff = change vector analysis + Otsu (no weights). The others run a "
                   "siamese network and need --checkpoint.")
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Trained weights (required for changeformer / bit / dsamnet).")
@click.option("--num-classes", default=2, show_default=True, help="Network output classes.")
@click.option("--min-magnitude", type=float, default=None,
              help="spectral_diff only: absolute magnitude floor, in the raster's units.")
@click.option("--backbone", type=click.Choice(["resnet18", "resnet50"]), default=None,
              help="bit / dsamnet only: the encoder the checkpoint was trained with (default resnet18; "
                   "the registry's bit-r50 is resnet50).")
def ai_change(before, after, output, method, checkpoint, num_classes, min_magnitude, backbone):
    """Bi-temporal change detection.

    The default is change vector analysis with an Otsu threshold; it needs no
    weights. The network methods refuse to run without --checkpoint rather than
    returning an untrained model's output.

    \b
    Example:
        pygeovision ai change --before 2020.tif --after 2024.tif -o change.tif
        pygeovision ai change --before a.tif --after b.tif -o c.tif \\
            --method changeformer --checkpoint cf.pth
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()

    click.echo(f"Running {method} change detection...")
    kw = {}
    if method == "spectral_diff":
        if backbone is not None:
            raise click.UsageError("--backbone only applies to the bit and dsamnet methods")
        if checkpoint:
            raise click.UsageError("--checkpoint only applies to the network methods")
        if min_magnitude is not None:
            kw["min_magnitude"] = min_magnitude
    else:
        if min_magnitude is not None:
            raise click.UsageError("--min-magnitude only applies to --method spectral_diff")
        if backbone is not None:
            if method == "changeformer":
                raise click.UsageError("--backbone applies to bit and dsamnet; changeformer has one fixed architecture")
            kw["backbone"] = backbone
        kw.update(checkpoint=checkpoint, num_classes=num_classes)
    try:
        result = client.change.detect(before, after, output_path=output, method=method, **kw)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if isinstance(result, dict) and result.get("error"):
        raise click.ClickException(str(result["error"]))
    click.echo(f"✓ Change detection complete → {output}")
    if isinstance(result, dict):
        for k in ("threshold", "change_fraction", "change_pct", "valid_fraction"):
            if k in result:
                click.echo(f"  {k}: {result[k]}")


@ai.command("train")
@click.argument("task", type=click.Choice(["segmentation", "detection"]))
@click.option("--data", "-d", "data_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output model checkpoint path.")
@click.option("--num-classes", default=2, show_default=True)
@click.option("--epochs", default=50, show_default=True)
@click.option("--backbone", default=None,
              help="Native registry model name (default: unet-r50 for segmentation, faster-rcnn-r50 for detection).")
def ai_train(task, data_dir, output, num_classes, epochs, backbone):
    """Configure a native training run (segmentation or detection).

    Builds a GeoTrainer configuration for the requested task. Dataset
    wiring (train/val split, augmentation) is left to the caller, since
    that's dataset-specific — see pygeovision.training.trainer.GeoTrainer.

    \b
    Example:
        pygeovision ai train segmentation --data ./building_chips/ \\
            --output building_model.pth --num-classes 2 --epochs 100
    """
    from pygeovision.ai.training.trainer import TrainingConfig

    backbone = backbone or ("faster-rcnn-r50" if task == "detection" else "unet-r50")
    click.echo(f"Preparing native {task} training config...")
    click.echo(f"  Data: {data_dir} | Classes: {num_classes} | Epochs: {epochs} | Backbone: {backbone}")

    cfg = TrainingConfig(task=task, num_classes=num_classes, max_epochs=epochs)
    click.echo("✓ Config ready. This command does NOT train anything itself -- build "
               "your model with the chosen backbone, then train it yourself:")
    click.echo("    from pygeovision.models import get_model")
    click.echo("    from pygeovision.ai.training.trainer import GeoTrainer")
    click.echo(f"    model = get_model({backbone!r}, num_classes={num_classes}, pretrained=True)")
    click.echo("    # To finetune from an existing checkpoint instead of pretrained weights:")
    click.echo("    # cfg.checkpoint_path = '/path/to/existing_checkpoint.pth'")
    click.echo("    GeoTrainer(model, cfg).fit(train_ds, val_ds)")
    click.echo(f"  Real output path (only written once you actually call .fit() yourself, "
               f"not by this command): {output}")
    click.echo(f"  Config: {cfg.__dict__}")


@ai.command("infer")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--model", "-m", required=True, help="Registry model name (see `pygeovision ai-models list`).")
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Your trained weights for --model (state dict from GeoTrainer).")
@click.option("--output", "-o", required=True, help="Output prediction GeoTIFF.")
@click.option("--num-classes", default=2, show_default=True)
@click.option("--in-channels", type=int, default=None, help="Bands the model was trained on (default: the input's).")
@click.option("--tile-size", default=512, show_default=True)
@click.option("--overlap", default=64, show_default=True)
@click.option("--normalise/--no-normalise", "normalise", default=True, show_default=True,
              help="Stretch every band to its own 2-98 percentile range before the model sees it. "
                   "Use --no-normalise if the model was trained on values that were already scaled "
                   "(for example reflectance 0-1), otherwise the prediction is made on differently "
                   "scaled data and can be wrong.")
def ai_infer(input_path, model, checkpoint, output, num_classes, in_channels, tile_size, overlap, normalise):
    """Run tiled inference on a large GeoTIFF with your trained model.

    \b
    Example:
        pygeovision ai infer --input large_scene.tif --model unet_resnet50 \\
            --checkpoint best.pth --output pred.tif --num-classes 5
    """
    from pygeovision.inference.tiled import TiledInference

    click.echo(f"Running tiled inference: {input_path} → {output}")
    m = _load_user_model(model, checkpoint, num_classes, in_channels, raster=input_path)
    engine = TiledInference(model=m, chip_size=tile_size, overlap=overlap, num_classes=num_classes)
    result = engine.infer(input_path, output, normalise=normalise)
    if not result.get("success"):
        raise click.ClickException(str(result.get("error")))
    click.echo(f"✓ Inference complete → {output}  ({result.get('n_chips', 0)} chips)")


@ai.command("chips")
@click.option("--image", required=True, type=click.Path(exists=True), help="Source imagery GeoTIFF.")
@click.option("--label", required=True, type=click.Path(exists=True), help="Label/mask GeoTIFF.")
@click.option("--output", "-o", required=True, help="Output chip directory.")
@click.option("--chip-size", default=256, show_default=True)
@click.option("--overlap", default=0, show_default=True)
@click.option("--prefix", default="chip", show_default=True,
              help="File-name prefix. Use a different prefix per scene when several scenes go into one folder, "
                   "otherwise chips of the later scene overwrite those of the earlier one.")
def ai_chips(image, label, output, chip_size, overlap, prefix):
    """Export aligned imagery/label training chips (native windowed tiling).

    \b
    Example:
        pygeovision ai chips --image sentinel2.tif --label buildings.tif \\
            --output ./training_chips/ --chip-size 256
    """
    import rasterio
    from rasterio.windows import Window

    out_dir = Path(output)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    stride = chip_size - overlap
    n_chips = 0
    with rasterio.open(image) as img_src, rasterio.open(label) as lbl_src:
        # Chips are cut by pixel window, so image and label must be on the same grid;
        # otherwise every chip pairs an image patch with the wrong piece of the label.
        if (img_src.height, img_src.width) != (lbl_src.height, lbl_src.width) or (
            img_src.crs != lbl_src.crs
            or not img_src.transform.almost_equals(lbl_src.transform)
        ):
            raise click.ClickException(
                f"image and label are not on the same grid: image is {img_src.width}x{img_src.height} "
                f"in {img_src.crs}, label is {lbl_src.width}x{lbl_src.height} in {lbl_src.crs}. "
                "Resample the label onto the image grid (same CRS, size and origin) before cutting chips."
            )
        H, W = img_src.height, img_src.width
        for row in range(0, H - chip_size + 1, stride):
            for col in range(0, W - chip_size + 1, stride):
                win = Window(col, row, chip_size, chip_size)
                img_chip = img_src.read(window=win)
                lbl_chip = lbl_src.read(1, window=win)

                img_profile = img_src.profile.copy()
                img_profile.update(height=chip_size, width=chip_size,
                                   transform=img_src.window_transform(win))
                lbl_profile = lbl_src.profile.copy()
                lbl_profile.update(height=chip_size, width=chip_size, count=1,
                                   transform=lbl_src.window_transform(win))

                img_path = out_dir / "images" / f"{prefix}_{row}_{col}.tif"
                lbl_path = out_dir / "labels" / f"{prefix}_{row}_{col}.tif"
                from pygeovision.data._profile import writable_profile
                img_profile = writable_profile(img_profile)
                with rasterio.open(img_path, "w", **img_profile) as dst:
                    dst.write(img_chip)
                from pygeovision.data._profile import writable_profile
                lbl_profile = writable_profile(lbl_profile)
                with rasterio.open(lbl_path, "w", **lbl_profile) as dst:
                    dst.write(lbl_chip, 1)
                n_chips += 1

    click.echo(f"✓ {n_chips} chips exported → {output}")


@ai.command("cloud-mask")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output cloud mask GeoTIFF.")
@click.option("--brightness-threshold", default=0.7, show_default=True,
              help="Reflectance threshold (0-1, normalised) above which pixels are flagged cloudy.")
def ai_cloud_mask(input_path, output, brightness_threshold):
    """Generate a cloud mask via native brightness thresholding
    (bright + spectrally flat pixels across visible bands = likely cloud).

    \b
    Example:
        pygeovision ai cloud-mask --input sentinel2.tif --output cloud.tif
    """
    import numpy as np
    import rasterio

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        n_bands = min(src.count, 4)
        data = src.read(list(range(1, n_bands + 1))).astype(np.float32)
        for b in range(data.shape[0]):
            p2, p98 = np.percentile(data[b], (2, 98))
            data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8), 0, 1)

    brightness = data.mean(axis=0)
    spectral_flatness = 1.0 - (data.std(axis=0) / (brightness + 1e-6))
    cloud_mask = ((brightness > brightness_threshold) &
                  (spectral_flatness > 0.5)).astype(np.uint8)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", compress="lzw")
    for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
        profile.pop(_k, None)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(cloud_mask[np.newaxis])

    click.echo(f"✓ Cloud mask → {output} "
               f"({round(float(cloud_mask.mean()) * 100, 1)}% flagged cloudy)")

# End-to-end PIPELINE commands (pygeofetch data + native AI)
# =========================================================================

def _parse_param(raw: str) -> tuple[str, Any]:
    """'key=value' -> (key, value).

    The value is parsed as JSON when it is valid JSON (``0.2``, ``true``, ``[5, 10, 15]``). A plain comma-separated list
    of numbers (``5,10,15``) is read as a list of numbers, because that is how people write one on a command line;
    anything else stays a string.
    """
    if "=" not in raw:
        raise click.BadParameter(f"--param expects KEY=VALUE, got {raw!r}")
    key, _, text = raw.partition("=")
    try:
        return key.strip(), json.loads(text)
    except ValueError:
        pass
    numbers = _number_list(text)
    return key.strip(), numbers if numbers is not None else text


def _number_list(text: str) -> list[float | int] | None:
    """``"5,10,15"`` -> ``[5, 10, 15]``; ``None`` when the text is not a comma-separated list of numbers."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 2:
        return None
    values: list[float | int] = []
    for part in parts:
        try:
            value = json.loads(part)
        except ValueError:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values.append(value)
    return values


@cli.command("channel")
@click.argument("pipeline_name")
@click.option("--bbox", nargs=4, type=float, default=None,
              metavar="MIN_LON MIN_LAT MAX_LON MAX_LAT")
@click.option("--output", "-o", default="./pipeline_output", show_default=True)
@click.option("--date", default=None, help="YYYY-MM acquisition date (single-date pipelines).")
@click.option("--date-before", default=None, help="Earlier date (bi-temporal pipelines).")
@click.option("--date-after", default=None, help="Later date (bi-temporal pipelines).")
@click.option("--dates", default=None, help="Comma-separated dates (vegetation_indices).")
@click.option("--model", default=None, help="Model name (pipelines that run a network).")
@click.option("--checkpoint", default=None, type=click.Path(),
              help="Your trained .pth weights; required by every 'needs_weights' pipeline.")
@click.option("--source", default=None, help="Data source override (land_cover: worldcover | dynamic_world).")
@click.option("--provider", multiple=True,
              help="Restrict search to this provider (repeatable, e.g. "
                   "--provider planetary_computer). An unrecognised name is "
                   "rejected by the search layer itself, not validated here.")
@click.option("--param", "params", multiple=True, metavar="KEY=VALUE",
              help="Any other pipeline argument, e.g. --param anomaly_threshold=0.2 "
                   "--param dem_type=cop30 (repeatable). Values are read as JSON when possible, and 5,10,15 is a list "
                   "of numbers.")
@click.option("--no-report", is_flag=True,
              help="Skip the map, vector layers and reports (also: PYGEOVISION_REPORT=0).")
@click.option("--report-formats", default="pdf,xlsx,docx", show_default=True,
              help="Report formats: any of pdf, xlsx (Excel), docx (Word); 'none' for no reports.")
@click.option("--vector-formats", default="gpkg,geojson", show_default=True,
              help="Vector formats: any of gpkg, geojson, shp, kml; 'none' for no vector files.")
@click.option("--class-names", default=None,
              help="Comma-separated names for the classes 0,1,2... of a network you trained "
                   "(labels the map, tables and vector attributes).")
@click.option("--report-title", default=None, help="Replace the report title.")
def pipeline_cmd(pipeline_name, bbox, output, date, date_before, date_after, dates,
                 model, checkpoint, source, provider, params, no_report, report_formats,
                 vector_formats, class_names, report_title):
    """Run an end-to-end geospatial pipeline by name.

    Downloads imagery with PyGeoFetch, then computes an index, a terrain
    product, a reference map, or runs your own trained network.
    ``pygeovision channel list`` shows every pipeline and whether it runs
    out of the box ("ready") or needs your trained weights
    ("needs_weights", pass --checkpoint). Removed pipelines explain why.

    \b
    Examples:
        pygeovision channel list

    \b
        pygeovision channel water_bodies \\
            --bbox -0.15 51.47 -0.10 51.52 --date 2024-06

    \b
        pygeovision channel wildfire_severity \\
            --bbox -122.6 39.7 -121.9 40.2 \\
            --date-before 2018-10 --date-after 2018-12

    \b
        pygeovision channel building_footprints \\
            --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 \\
            --checkpoint runs/buildings/best.pth

    \b
        pygeovision channel crop_health --bbox ... \\
            --date-before 2024-05 --date-after 2024-08 --param anomaly_threshold=0.2"""
    from pygeovision.ai.pipelines import list_pipelines, pipeline_catalogue, pipeline_kinds

    if pipeline_name == "list":
        kinds = pipeline_kinds()
        catalogue = pipeline_catalogue()
        click.echo(f"\n{len(kinds)} PyGeoVision pipelines "
                   f"({sum(k == 'ready' for k in kinds.values())} ready, "
                   f"{sum(k == 'needs_weights' for k in kinds.values())} need your trained weights):\n")
        for name in list_pipelines():
            click.echo(f"  {name:26s} {kinds[name]:13s} {catalogue[name].description}")
        return

    if not bbox:
        click.echo("--bbox required", err=True)
        sys.exit(1)

    from pygeovision import PyGeoVision
    client = PyGeoVision()

    # Only pass what the user actually gave, so a pipeline's own defaults apply
    # and a flag it does not understand is never silently dropped by us.
    kwargs: dict[str, Any] = {}
    for key, value in (("date", date), ("date_before", date_before), ("date_after", date_after),
                       ("model", model), ("checkpoint", checkpoint), ("source", source)):
        if value:
            kwargs[key] = value
    if dates:
        kwargs["dates"] = [d.strip() for d in dates.split(",") if d.strip()]
    if provider:
        kwargs["providers"] = list(provider)
    for raw in params:
        key, value = _parse_param(raw)
        kwargs[key] = value
    if no_report:
        kwargs["report"] = False
    else:
        kwargs["report_formats"] = report_formats
        kwargs["vector_formats"] = vector_formats
        if class_names:
            kwargs["class_names"] = [c.strip() for c in class_names.split(",")]
        if report_title:
            kwargs["report_title"] = report_title

    click.echo(f"Running pipeline '{pipeline_name}'...")
    click.echo(f"  bbox={tuple(bbox)} | output={output}")

    try:
        result = client.pipeline(pipeline_name, bbox=tuple(bbox), output_dir=output, **kwargs)
    except ValueError as exc:  # unknown or removed pipeline
        click.echo(f"\n✗ {exc}", err=True)
        sys.exit(2)

    if result.success:
        click.echo("\n✓ pipeline complete")
        if result.output_path:
            click.echo(f"  Output: {result.output_path}")
        if result.stats:
            click.echo("  Stats:")
            for k, v in result.stats.items():
                if isinstance(v, dict):
                    continue
                fmt_v = f"{v:.4f}" if isinstance(v, float) else v
                click.echo(f"    {k}: {fmt_v}")
        _echo_deliverables(result)
    else:
        click.echo(f"\n✗ pipeline failed: {result.error}", err=True)
        sys.exit(1)


def _echo_deliverables(result) -> None:
    """Print where the map, vectors and reports went (or why they are missing)."""
    d = getattr(result, "deliverables", None)
    if d is None:
        err = result.metadata.get("deliverables_error")
        if err:
            click.echo(f"\n  Deliverables NOT created: {err}", err=True)
        return
    click.echo(f"\n  Deliverables: {d.directory}")
    names = {"pdf": "PDF report", "docx": "Word report", "xlsx": "Excel report"}
    for fmt, p in d.reports.items():
        click.echo(f"    {names.get(fmt, fmt):13s} {p.name}")
    for p in d.maps:
        click.echo(f"    {'Map':13s} maps/{p.name}")
    for fmt, paths in d.vectors.items():
        click.echo(f"    {'Vectors ' + fmt:13s} {', '.join(x.name for x in paths)}")
    for w in d.warnings:
        click.echo(f"    ! {w}", err=True)


@cli.command("report")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--formats", "report_formats", default="pdf,xlsx,docx", show_default=True,
              help="Report formats: pdf, xlsx, docx ('none' for none).")
@click.option("--vector-formats", default="gpkg,geojson", show_default=True,
              help="Vector formats: gpkg, geojson, shp, kml ('none' for none).")
@click.option("--class-names", default=None, help="Comma-separated class names for a trained network's classes 0,1,2...")
@click.option("--title", default=None, help="Replace the report title.")
@click.option("--no-vectors", is_flag=True, help="Do not write vector files.")
@click.option("--no-maps", is_flag=True, help="Do not draw maps and charts.")
def report_cmd(output_dir, report_formats, vector_formats, class_names, title, no_vectors, no_maps):
    """Rebuild the map, vectors and reports of an earlier pipeline run.

    OUTPUT_DIR is the -o directory of a `pygeovision channel` run. Nothing is
    downloaded or re-analysed: the output rasters and run_summary.json are read.
    Use it to add a format (for example --formats pdf,docx --vector-formats shp,kml)
    or to name the classes of a network you trained.
    """
    from pygeovision.reporting import regenerate
    try:
        d = regenerate(output_dir, report_formats=report_formats, vector_formats=vector_formats,
                       make_vectors=not no_vectors, make_maps=not no_maps, title=title,
                       class_names=[c.strip() for c in class_names.split(",")] if class_names else None)
    except FileNotFoundError as exc:
        click.echo(f"\n✗ {exc}", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - report the reason, not a traceback
        click.echo(f"\n✗ could not rebuild the deliverables: {exc}", err=True)
        sys.exit(1)

    class _R:                       # lets _echo_deliverables print a plain Deliverables
        deliverables, metadata = d, {}
    click.echo("\n✓ deliverables rebuilt")
    _echo_deliverables(_R)


# =========================================================================
# MODELS commands
# =========================================================================

@cli.group("ai-models")
def ai_models() -> None:
    """AI model management commands (legacy AI-engine registry).

    Note: this is distinct from `pygeovision models`, which uses the
    newer, broader native model registry (pygeovision.models.registry).
    """


@ai_models.command("list")
@click.option("--task", default=None,
              type=click.Choice(["segmentation", "detection", "classification",
                                 "change_detection", "super_resolution"]))
@click.option("--pretrained-only", is_flag=True)
def ai_models_list(task, pretrained_only):
    """List the 14 models of the legacy AI-engine registry."""
    from pygeovision.ai.models.registry import registry
    results = registry.list_models(task=task, pretrained_only=pretrained_only)
    if not results:
        click.echo("No models found.")
        return
    click.echo(f"\n{'Name':<35} {'Task':<22} {'Architecture':<25} {'Pretrained'}")
    click.echo("-" * 90)
    for m in results:
        pt = "✓" if m.pretrained_available else ""
        click.echo(f"{m.name:<35} {m.task:<22} {m.architecture:<25} {pt}")
    click.echo(f"\nTotal: {len(results)} models")


@ai_models.command("info")
@click.argument("model_name")
def ai_models_info(model_name):
    """Show details for a specific model."""
    from pygeovision.ai.models.registry import registry
    try:
        info = registry.get(model_name)
        click.echo(f"\nModel: {info.name}")
        click.echo(f"  Task:        {info.task}")
        click.echo(f"  Architecture:{info.architecture}")
        click.echo(f"  Input bands: {info.input_bands}")
        click.echo(f"  Pretrained:  {'Yes' if info.pretrained_available else 'No'}")
        click.echo(f"  Description: {info.description}")
        if info.tags:
            click.echo(f"  Tags:        {', '.join(info.tags)}")
        if info.paper_url:
            click.echo(f"  Paper:       {info.paper_url}")
    except KeyError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@ai_models.command("cache")
@click.option("--clear", is_flag=True)
@click.option("--model", default=None)
def ai_models_cache(clear, model):
    """Manage AI model checkpoint cache."""
    from pygeovision.ai.models import ModelHub
    hub = ModelHub()
    if clear:
        hub.clear_cache(model_name=model)
        click.echo(f"✓ Cache cleared{f' for {model}' if model else ''}.")
    else:
        cached = hub.list_cached()
        if not cached:
            click.echo("No models cached.")
        else:
            click.echo(f"{'Name':<35} {'Size':<10} {'Path'}")
            for m in cached:
                size = f"{m.local_path.stat().st_size / 1024 / 1024:.1f}MB" if m.local_path.exists() else "?"
                click.echo(f"{m.name:<35} {size:<10} {m.local_path}")


# =========================================================================
# STATUS command
# =========================================================================

@cli.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status_cmd(as_json):
    """Show PyGeoVision, pygeofetch, and native AI stack status."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    status = client.status()

    if as_json:
        click.echo(json.dumps(status, indent=2, default=str))
        return

    click.echo("\n══════════════════════════════════════════")
    click.echo("         PyGeoVision System Status        ")
    click.echo("══════════════════════════════════════════")
    click.echo(f"PyGeoVision v{status['pygeovision_version']} | Python {status['python']}")
    click.echo()

    pf = status["pygeofetch"]
    pf_icon = "✓" if pf["available"] else "✗"
    click.echo(f"pygeofetch  {pf_icon}  v{pf['version']} | {pf['providers']} providers | {pf['open_providers']} open-access")

    torch = status["torch"]
    if torch.get("available") is False:
        click.echo("torch       ✗  not installed")
    else:
        cuda = f" | CUDA {torch.get('gpu', 'N/A')}" if torch.get("cuda") else ""
        click.echo(f"torch       ✓  v{torch.get('version', '?')} | device={torch.get('device', 'cpu')}{cuda}")

    rio = status.get("rasterio")
    click.echo(f"rasterio    {'✓' if rio else '✗'}  {rio or 'not installed'}")

    gpd = status.get("geopandas")
    click.echo(f"geopandas   {'✓' if gpd else '✗'}  {gpd or 'not installed'}")

    click.echo(f"\nRegistered AI models: {status.get('registered_ai_models', 0)}")
    click.echo()

    if not pf["available"]:
        click.echo("⚠  pygeofetch CLI not found in PATH.")
        click.echo("   Run: pip install pygeofetch  (installs PyGeoFetch CLI)")
    if torch.get("available") is False:
        click.echo("⚠  torch not installed — AI segmentation/detection/training unavailable.")
        click.echo("   Run: pip install 'pygeovision[train]'")


@cli.command("doctor")
def doctor_cmd():
    """Diagnose PyGeoVision installation and all component connectivity."""
    from pygeovision import PyGeoVision
    try:
        client = PyGeoVision()
        result = client.doctor()
    except Exception as exc:
        click.echo(f"\nCould not initialise PyGeoVision: {exc}", err=True)
        return

    click.echo("\n══════════════════════════════════════════")
    click.echo("         PyGeoVision Diagnostics          ")
    click.echo("══════════════════════════════════════════")
    for key, val in result.items():
        if key == "summary":
            continue
        if isinstance(val, dict):
            ok_icon = "✓" if val.get("ok") else "✗"
            extra = ""
            if val.get("version"):
                extra = f"  v{val['version']}"
            if val.get("cuda"):
                extra += f"  CUDA ({val.get('gpu', 'GPU')})"
            if "providers_configured" in val:
                extra += f"  {val['providers_configured']} provider(s) with stored credentials"
            if "entries" in val:
                extra += f"  {val['entries']} entries"
            if val.get("error"):
                extra = f"  ERROR: {val['error']}"
            click.echo(f"  {ok_icon}  {key:<18}{extra}")
        else:
            click.echo(f"     {key:<18}: {val}")
    click.echo()
    summary = result.get("summary", "Diagnosis complete.")
    click.echo(f"  → {summary}")
    click.echo()

    # Actionable advice; a failed check makes the command exit non-zero
    failed = [k for k, v in result.items() if isinstance(v, dict) and "ok" in v and not v["ok"]]
    for key, val in result.items():
        if isinstance(val, dict) and not val.get("ok") and val.get("error"):
            click.echo(f"  ⚠  {key}: {val['error']}")
    if failed:
        sys.exit(1)



# ═══════════════════════════════════════════════════════════════════════════════
# NEW CLI COMMAND GROUPS — Phase 6
# ═══════════════════════════════════════════════════════════════════════════════

# ── models ────────────────────────────────────────────────────────────────────
@cli.group("models")
def models_grp():
    """Model registry commands — list, info, download architectures."""

@models_grp.command("list")
@click.option("--task", "-t", default=None, help="Filter by task (segmentation|detection|...)")
@click.option("--family", "-f", default=None, help="Filter by family (resnet|vit|unet|...)")
@click.option("--max-params", type=float, default=None, help="Maximum parameters (millions)")
def models_list(task, family, max_params):
    """List available model architectures."""
    from pygeovision.models.registry import model_registry
    names = model_registry.list(task=task, family=family, max_params_m=max_params)
    click.echo(f"\n  Found {len(names)} models:\n")
    click.echo(f"  {'Name':<28} {'Task':<18} {'Family':<14} {'Params M':>8}  Description")
    click.echo("  " + "-"*90)
    for name in sorted(names):
        spec = model_registry[name]
        click.echo(f"  {name:<28} {spec.task:<18} {spec.family:<14} {spec.params_m:>8.1f}  {spec.description[:35]}")
    click.echo()

@models_grp.command("info")
@click.argument("name")
def models_info(name):
    """Show detailed info for a model."""
    from pygeovision.models.registry import model_registry
    try:
        spec = model_registry[name]
    except KeyError as e:
        click.echo(f"Error: {e}", err=True); return
    click.echo(f"\n  Model: {spec.name}")
    click.echo(f"  Task:  {spec.task}")
    click.echo(f"  Family:{spec.family}")
    click.echo(f"  Params:{spec.params_m}M")
    if spec.hf_id:  click.echo(f"  HF ID: {spec.hf_id}")
    if spec.timm_id:click.echo(f"  timm:  {spec.timm_id}")
    click.echo(f"  Desc:  {spec.description}")
    if spec.pretrained_on:
        click.echo(f"  Pretrained on: {spec.pretrained_on}")
    click.echo()

@models_grp.command("download")
@click.argument("name")
@click.option("--cache-dir", default=None, help="Local cache directory")
def models_download(name, cache_dir):
    """Download pretrained weights for a model."""
    from pygeovision.models.registry import model_registry
    from pygeovision.models.weights.downloader import WeightDownloader
    try:
        spec = model_registry[name]
    except KeyError as e:
        click.echo(f"Error: {e}", err=True); return
    if not spec.hf_id and not spec.timm_id:
        click.echo(f"No downloadable weights for {name}"); return
    downloader = WeightDownloader(cache_dir=cache_dir)
    if spec.hf_id:
        click.echo(f"Downloading {name} from HuggingFace Hub...")
        path = downloader.download(spec.hf_id)
        click.echo(f"✓ Downloaded: {path}")
    else:
        click.echo(f"  {name} uses timm — loaded automatically on first use.")

@models_grp.command("summary")
def models_summary():
    """Show registry summary by task."""
    from pygeovision.models.registry import model_registry
    s = model_registry.summary()
    click.echo(f"\n  Model Registry: {s['total']} architectures\n")
    for task, n in sorted(s["by_task"].items()):
        click.echo(f"  {task:<25} {n:>3} models")
    click.echo(f"\n  HF Hub weights:  {s['with_hf_weights']}")
    click.echo(f"  timm weights:    {s['with_timm_weights']}\n")


# ── infer ─────────────────────────────────────────────────────────────────────
@cli.group("infer")
def infer_grp():
    """Run AI inference on satellite imagery."""

@infer_grp.command("predict")
@click.argument("image_path")
@click.option("--model", "-m", default="unet_resnet50", show_default=True, help="Registry model name.")
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Your trained weights for --model (state dict from GeoTrainer).")
@click.option("--output", "-o", default=None)
@click.option("--classes", "-c", type=int, default=2, show_default=True)
@click.option("--in-channels", type=int, default=None, help="Bands the model was trained on (default: the input's).")
@click.option("--chip-size", default=512, show_default=True)
@click.option("--overlap", default=64, show_default=True)
@click.option("--blend", default="gaussian", show_default=True,
               type=click.Choice(["gaussian","linear","constant"]))
@click.option("--device", default=None, help="cuda|cpu|mps")
@click.option("--normalise/--no-normalise", "normalise", default=True, show_default=True,
              help="Stretch every band to its own 2-98 percentile range before the model sees it. "
                   "Use --no-normalise if the model was trained on values that were already scaled "
                   "(for example reflectance 0-1), otherwise the prediction is made on differently "
                   "scaled data and can be wrong.")
def infer_predict(image_path, model, checkpoint, output, classes, in_channels, chip_size, overlap, blend, device,
                  normalise):
    """Run tiled inference on a GeoTIFF with your trained model."""
    import pathlib
    output = output or str(pathlib.Path(image_path).with_suffix("")) + f"_{model}_pred.tif"
    click.echo(f"\n  Inference: {image_path}")
    click.echo(f"  Model:     {model} | classes={classes} | chip={chip_size} | blend={blend}")
    click.echo("  Input:     " + ("per-band 2-98 percentile stretch (--no-normalise to turn off)"
                                 if normalise else "used as stored (no stretch)"))
    m = _load_user_model(model, checkpoint, classes, in_channels, raster=image_path)
    from pygeovision.inference.tiled import TiledInference
    try:
        inf = TiledInference(model=m, chip_size=chip_size, overlap=overlap,
                              blend_mode=blend, num_classes=classes, device=device)
        result = inf.infer(image_path, output, normalise=normalise)
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)
        sys.exit(1)
    if not result.get("success"):
        click.echo(f"  ✗ {result.get('error')}", err=True)
        sys.exit(1)
    click.echo(f"  ✓ Output:  {output}")
    click.echo(f"  ✓ Chips:   {result.get('n_chips',0)} | Time: {result.get('duration_seconds',0)}s\n")

@infer_grp.command("batch")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir")
@click.option("--model", "-m", default="unet_resnet50", show_default=True, help="Registry model name.")
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Your trained weights for --model (state dict from GeoTrainer).")
@click.option("--classes", "-c", type=int, default=2, show_default=True)
@click.option("--in-channels", type=int, default=None,
              help="Bands the model was trained on (default: the first input file's).")
@click.option("--workers", "-w", default=2, show_default=True)
@click.option("--pattern", default="*.tif", show_default=True)
@click.option("--overwrite", is_flag=True,
              help="Predict again even if the output file already exists (default: skip those files).")
@click.option("--normalise/--no-normalise", "normalise", default=True, show_default=True,
              help="Stretch every band to its own 2-98 percentile range before the model sees it. "
                   "Use --no-normalise if the model was trained on values that were already scaled "
                   "(for example reflectance 0-1), otherwise the prediction is made on differently "
                   "scaled data and can be wrong.")
def infer_batch(input_dir, output_dir, model, checkpoint, classes, in_channels, workers, pattern, overwrite,
                normalise):
    """Run batch inference on a directory of GeoTIFFs with your trained model."""
    click.echo(f"\n  Batch inference: {input_dir} → {output_dir}")
    first = next(iter(sorted(Path(input_dir).glob(pattern))), None)
    if first is None:
        raise click.ClickException(f"no files matching {pattern!r} in {input_dir}")
    m = _load_user_model(model, checkpoint, classes, in_channels, raster=first)
    try:
        from pygeovision.inference.batch import BatchInferenceEngine
        engine = BatchInferenceEngine(model=m, n_workers=workers, num_classes=classes, normalise=normalise)
        result = engine.run_directory(input_dir, output_dir, pattern=pattern, skip_existing=not overwrite)
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)
        sys.exit(1)
    if result.get("success") is False:
        raise click.ClickException(str(result.get("error")))
    n_ok, n_bad = result.get("n_success", 0), result.get("n_failed", 0)
    n_skipped = result.get("skipped", 0)
    click.echo(f"  {'✓' if not n_bad else '✗'} {n_ok}/{n_ok + n_bad} succeeded"
               + (f", {n_skipped} skipped (output already exists; --overwrite to redo)" if n_skipped else ""))
    click.echo(f"  Time: {result.get('total_time_s',0)}s | {result.get('throughput_fps',0)} fps\n")
    for bad in result.get("failed", []):
        click.echo(f"  ✗ {bad.get('path')}: {bad.get('error')}", err=True)
    if n_bad or not (n_ok or n_skipped):
        sys.exit(1)


# ── label ─────────────────────────────────────────────────────────────────────
@cli.group("label")
def label_grp():
    """Auto-labeling from OSM, satellite data, and foundation models."""

@label_grp.command("osm")
@click.argument("bbox", nargs=4, type=float, metavar="LON_MIN LAT_MIN LON_MAX LAT_MAX")
@click.option("--output", "-o", default="./labels/osm.tif", show_default=True, help="Label raster to write.")
@click.option("--categories", "-c", multiple=True, default=["buildings", "roads", "water"], show_default=True,
              help="Repeat for several: buildings roads water vegetation agricultural commercial "
                   "residential solar_panels parking sports.")
@click.option("--reference", "reference", type=click.Path(exists=True), default=None,
              help="Imagery raster: write the labels on its exact grid and CRS (recommended for training).")
@click.option("--resolution", default=10.0, show_default=True, type=float,
              help="Pixel size in metres when there is no --reference.")
@click.option("--all-touched", is_flag=True, help="Burn every pixel an area touches (keeps tiny buildings).")
@click.option("--vector/--no-vector", default=True, show_default=True, help="Also write the features as GeoJSON.")
def label_osm(bbox, output, categories, reference, resolution, all_touched, vector):
    """Generate labels from OpenStreetMap (class values: buildings 1, roads 2, water 3, vegetation 4,
    agricultural 5, commercial 6, residential 7, solar_panels 8, parking 9, sports 10)."""
    from pygeovision.labeling.osm import OSMLabeler
    click.echo(f"\n  OSM Labeling: bbox={tuple(bbox)}")
    try:
        result = OSMLabeler().label(bbox, categories=list(categories), output_path=output,
                                    reference_raster=reference, resolution_m=resolution,
                                    save_vector=vector, all_touched=all_touched)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    if result.get("success"):
        click.echo(f"  ✓ {result['n_features']} features → {output}")
        for cat, n in sorted(result["categories"].items(), key=lambda kv: -kv[1]):
            click.echo(f"      {cat:<14} {n:>7}   (raster value {result['class_values'][cat]})")
        if result.get("vector_path"):
            click.echo(f"    vectors → {result['vector_path']}")
    else:
        click.echo(f"  ✗ {result.get('error')}", err=True)
        sys.exit(1)

@label_grp.command("quality")
@click.argument("label_path")
@click.option("--num-classes", type=int, default=None,
              help="Classes are 0..N-1. Default: inferred from the raster.")
@click.option("--check", "checks", multiple=True,
              type=click.Choice(["class_balance", "slivers", "coverage", "connectivity", "boundaries"]),
              help="Run only these checks (repeatable). Default: all except boundaries.")
@click.option("--html", is_flag=True, help="Save HTML report")
def label_quality(label_path, num_classes, checks, html):
    """Assess label quality (raster-only heuristics; not a measure of accuracy)."""
    from pathlib import Path as _P

    from pygeovision.labeling.quality import LabelQualityAssessor
    qa = LabelQualityAssessor(num_classes=num_classes)
    report = qa.assess(label_path, checks=list(checks) or None)
    if report.get("error"):
        raise click.ClickException(f"cannot read {label_path}: {report['error']}")
    click.echo(f"\n  Quality Grade: {report.get('quality_grade')} ({report.get('quality_score',0):.0%})")
    for check, data in report.get("checks", {}).items():
        icon = "✓" if data.get("status") == "ok" else "⚠"
        click.echo(f"  {icon} {check}: score={data.get('score','?')}")
    for rec in report.get("recommendations", []):
        click.echo(f"  ⟶ {rec}")
    if html:
        lp = _P(label_path)
        html_path = lp.with_name(lp.stem + "_quality.html")
        html_path.write_text(qa.report_html(report))
        click.echo(f"  ✓ HTML report: {html_path}")
    click.echo()


# ── explain ───────────────────────────────────────────────────────────────────
@cli.group("explain")
def explain_grp():
    """Explainability — GradCAM saliency maps (SHAP, uncertainty and attention maps are Python APIs)."""

@explain_grp.command("gradcam")
@click.argument("image_path")
@click.option("--model", "-m", required=True)
@click.option("--output", "-o", default=None)
@click.option("--class-idx", "-c", default=1, type=int)
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Your trained weights for --model (strict load). Saliency of an untrained model is noise.")
@click.option("--classes", default=2, show_default=True, type=int)
@click.option("--in-channels", default=None, type=int, help="Default: the image's band count.")
def explain_gradcam(image_path, model, output, class_idx, checkpoint, classes, in_channels):
    """Generate GradCAM saliency map."""
    output = output or image_path.replace(".tif", "_gradcam.tif")
    try:
        from pygeovision.explainability.gradcam import GradCAM
        if in_channels is None:
            import rasterio
            with rasterio.open(image_path) as src:
                in_channels = src.count
        m = _registry_model(model, classes, in_channels, checkpoint)
        cam = GradCAM(m)
        res = cam.batch_explain(image_path, output, class_idx=class_idx)
        if not res.get("success"):
            raise click.ClickException(res.get("error", "GradCAM failed"))
        click.echo(f"  ✓ GradCAM ({res.get('n_chips', '?')} chips) → {output}")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)
        sys.exit(1)


# ── monitor ───────────────────────────────────────────────────────────────────
@cli.group("monitor")
def monitor_grp():
    """Model monitoring — drift detection, performance tracking."""

@monitor_grp.command("drift")
@click.argument("reference_dir")
@click.argument("current_dir")
@click.option("--output", "-o", default="./monitoring/drift_report.json")
def monitor_drift(reference_dir, current_dir, output):
    """Detect data distribution drift."""
    import glob
    import pathlib
    ref_images = glob.glob(f"{reference_dir}/*.tif")[:50]
    cur_images  = glob.glob(f"{current_dir}/*.tif")[:50]
    from pygeovision.monitoring.drift import DriftDetector
    d = DriftDetector()
    d.fit(ref_images)
    report = d.check(cur_images)
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    if report.get("incomplete"):
        click.echo(f"  ✗ drift could not be assessed: {report['data_drift'].get('error')} | Report: {output}", err=True)
        sys.exit(1)
    dd = report["data_drift"]
    click.echo(f"  Drift level: {dd['drift_level']} (PSI {dd['overall_psi']}, worst band {dd['worst_band']}"
               f" {dd['max_psi']}) | Report: {output}")


# ── pipeline ──────────────────────────────────────────────────────────────────
@cli.group("pipeline")
def pipeline_grp():
    """Pipeline management — create, run, schedule workflows."""

@pipeline_grp.command("run")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Check the step graph without running anything.")
@click.option("--set", "overrides", multiple=True, metavar="STEP.PARAM=VALUE",
              help="Override a step parameter (value parsed as JSON, else a string); repeatable.")
def pipeline_run(yaml_path, dry_run, overrides):
    """Run a YAML pipeline (steps: search, download, infer, export).

    \b
    Example:
        pygeovision pipeline run urban.yaml --set search.bbox='[15.0,45.1,15.1,45.2]' \\
            --set search.date_range='["2024-06-01","2024-06-30"]' --set infer.checkpoint=best.pth
    """
    from pygeovision.pipelines.orchestrator import Pipeline
    from pygeovision.pipelines.yaml_parser import PipelineYAMLParser
    click.echo(f"\n  Loading pipeline: {yaml_path}")
    try:
        parser = PipelineYAMLParser()
        config = parser.apply_overrides(parser.load(yaml_path), list(overrides))
        p = Pipeline.from_config(config)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    click.echo(f"  Pipeline: {p.name} ({len(p._steps)} steps){'  [dry run]' if dry_run else ''}")
    ctx = {}
    if not dry_run:
        from pygeovision import PyGeoVision
        ctx["client"] = PyGeoVision()
    result = p.run(ctx, dry_run=dry_run)
    click.echo(f"  {'✓' if result.success else '✗'} completed: {result.steps_completed} in {result.duration_s}s")
    if not result.success:
        click.echo(f"  failed: {result.steps_failed}\n  {result.error}", err=True)
        sys.exit(1)
    click.echo()

@pipeline_grp.command("list-templates")
def pipeline_list_templates():
    """List available YAML pipeline templates."""
    import pygeovision.pipelines as _pp
    tdir = Path(_pp.__file__).parent / "templates"
    templates = sorted(tdir.glob("*.yaml"))
    click.echo(f"\n  Available templates ({len(templates)}) in {tdir}:\n")
    for t in templates:
        click.echo(f"  • {t.stem:<25} ({t.name})")
    click.echo()


# ── edge ─────────────────────────────────────────────────────────────────────
@cli.group("edge")
def edge_grp():
    """Edge deployment — ONNX export, Jetson conversion."""


def _registry_model(name, classes, in_channels, checkpoint=None):
    """``pygeovision.models.registry.get_model`` + optional strict load of the user's trained weights.

    With ``checkpoint`` the architecture is built without downloading pretrained weights (they would be
    overwritten anyway) and the state dict must match it exactly.
    """
    from pygeovision.models.registry import get_model
    kw = {"pretrained": False} if checkpoint else {}
    try:
        m = get_model(name, num_classes=classes, in_channels=in_channels, **kw)
        if checkpoint:
            import torch
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                for key in ("model_state_dict", "state_dict", "model", "network"):
                    if key in state:
                        state = state[key]
                        break
            m.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    return m

@edge_grp.command("export-onnx")
@click.argument("model_name")
@click.option("--output", "-o", default="./model.onnx")
@click.option("--classes", default=2, show_default=True, type=int)
@click.option("--in-channels", default=4, show_default=True, type=int)
@click.option("--input-size", default=512, show_default=True, type=int)
@click.option("--checkpoint", type=click.Path(exists=True), default=None,
              help="Your trained weights (strict load). Without it the registry's pretrained "
                   "initialisation is exported -- fine for latency tests, not a trained model.")
@click.option("--dynamic-size", is_flag=True, default=False,
              help="Let the ONNX model accept any height and width (default: exactly --input-size).")
def edge_export_onnx(model_name, output, classes, in_channels, input_size, checkpoint, dynamic_size):
    """Export a registry model to ONNX (fixed input size unless --dynamic-size)."""
    click.echo(f"\n  Exporting {model_name} → ONNX ({output})")
    try:
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        m = _registry_model(model_name, classes, in_channels, checkpoint)
        ONNXRuntimeInference.from_pytorch(
            m, output, input_shape=(1, in_channels, input_size, input_size), simplify=False,
            dynamic_spatial=dynamic_size,
        )
        click.echo(f"  ✓ ONNX exported: {output}\n")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)
        sys.exit(1)

@edge_grp.command("benchmark-onnx")
@click.argument("onnx_path")
@click.option("--device", default="cpu", show_default=True)
@click.option("--runs", default=100, show_default=True, type=int)
def edge_benchmark(onnx_path, device, runs):
    """Benchmark ONNX model inference speed."""
    try:
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        eng = ONNXRuntimeInference(onnx_path, device=device)
        result = eng.benchmark(n_runs=runs)
        click.echo(f"\n  ONNX Benchmark ({onnx_path}):")
        click.echo(f"  Mean: {result['mean_ms']} ms | Median: {result['median_ms']} ms | "
                   f"P95: {result['p95_ms']} ms | {result['fps']} FPS | providers: {', '.join(result['providers'])}\n")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)
        sys.exit(1)


# ── cloud ─────────────────────────────────────────────────────────────────────
@cli.group("cloud")
def cloud_grp():
    """Cloud deployment — AWS SageMaker, Azure ML, GCP Vertex AI."""

@cloud_grp.command("deploy-aws")
@click.argument("model_path")
@click.argument("endpoint_name")
@click.option("--region", default="us-east-1", show_default=True)
@click.option("--instance", default="ml.g4dn.xlarge", show_default=True)
def cloud_deploy_aws(model_path, endpoint_name, region, instance):
    """Deploy model to AWS SageMaker."""
    from pygeovision.cloud.deploy import AWSDeployer
    click.echo(f"\n  Deploying to AWS SageMaker: {endpoint_name}")
    result = AWSDeployer(region=region).deploy(model_path, endpoint_name, instance_type=instance)
    if result.get("success"):
        click.echo(f"  ✓ Endpoint live: {result.get('endpoint_url')}\n")
    else:
        click.echo(f"  ✗ {result.get('error')}\n", err=True)
        sys.exit(1)

@cloud_grp.command("deploy-gcp")
@click.argument("model_path")
@click.argument("endpoint_name")
@click.option("--project", required=True)
@click.option("--region", default="us-central1")
def cloud_deploy_gcp(model_path, endpoint_name, project, region):
    """Deploy model to GCP Vertex AI."""
    from pygeovision.cloud.deploy import GCPDeployer
    click.echo(f"\n  Deploying to GCP Vertex AI: {endpoint_name}")
    result = GCPDeployer(project_id=project, region=region).deploy(model_path, endpoint_name)
    click.echo(f"  {'✓' if result.get('success') else '✗'} {result}\n", err=not result.get("success"))
    if not result.get("success"):
        sys.exit(1)


# ── vlm ───────────────────────────────────────────────────────────────────────
@cli.group("vlm")
def vlm_grp():
    """Vision-Language Models — caption, query, search imagery."""

def _vlm_call(fn):
    """Run a VLM action, turning missing packages, missing files and model-loading failures into clean CLI errors."""
    try:
        return fn()
    except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "ModelLoadError":
            raise click.ClickException(str(exc)) from exc
        raise


def _parse_bands(ctx, param, value):
    """``--bands 3,2,1`` -> (3, 2, 1)."""
    if value is None:
        return None
    try:
        bands = tuple(int(b) for b in value.split(","))
    except ValueError:
        raise click.BadParameter("give band numbers separated by commas, e.g. 3,2,1") from None
    if not 1 <= len(bands) <= 3 or min(bands) < 1:
        raise click.BadParameter("give 1 to 3 band numbers, each 1 or more")
    return bands


_bands_option = click.option("--bands", callback=_parse_bands, default=None,
                             help="File bands shown to the model as R,G,B, e.g. 3,2,1 for a B2,B3,B4,B8 Sentinel-2 file "
                                  "(default: the first three).")


@vlm_grp.command("caption")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@_bands_option
def vlm_caption(image_path, bands):
    """Generate a natural language caption for a satellite image (Moondream2, downloaded on first use)."""
    from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
    click.echo(f"\n  Generating caption for: {image_path}")
    caption = _vlm_call(lambda: MoondreamGeo(bands=bands).caption(image_path))
    click.echo(f"  Caption: {caption}\n")

@vlm_grp.command("query")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("question")
@_bands_option
def vlm_query(image_path, question, bands):
    """Answer a question about a satellite image (Moondream2, downloaded on first use)."""
    from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
    answer = _vlm_call(lambda: MoondreamGeo(bands=bands).vqa(image_path, question))
    click.echo(f"  A: {answer}\n")

@vlm_grp.command("search")
@click.argument("query")
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--top-k", default=5, show_default=True, type=int)
@click.option("--model", default="openclip-b32", show_default=True,
              help="CLIP model: openclip-b32, openclip-l14, clip-vit-b32, remoteclip-b32, remoteclip-l14, georsclip.")
@_bands_option
def vlm_search(query, image_dir, top_k, model, bands):
    """Rank the images in IMAGE_DIR against a text query (every image is embedded on every call)."""
    from pygeovision.advanced.vlm.clip_geo import CLIPGeo
    results = _vlm_call(lambda: CLIPGeo(model=model, bands=bands).search(query, image_dir, top_k=top_k))
    click.echo(f"\n  Top {len(results)} results for '{query}':")
    for r in results:
        click.echo(f"  {r['score']:.3f}  {r['path']}")
    click.echo()


# ── timeseries ────────────────────────────────────────────────────────────────
@cli.group("timeseries")
def timeseries_grp():
    """Time series of a spectral index over a stack of images: per-date statistics, trend, anomalies."""


def _ts_series(images, index, sensor, dates):
    """Run the index series, turning input mistakes into one-line CLI errors."""
    from pygeovision.advanced.timeseries import GeoTimeSeries
    date_list = [d.strip() for d in dates.split(",")] if dates else None
    try:
        ts = GeoTimeSeries(sensor=sensor)
        series = ts.compute_index_series(list(images), index=index, date_strings=date_list)
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    for path, why in series.get("errors", {}).items():
        click.echo(f"  warning: {path}: {why}", err=True)
    return ts, series


@timeseries_grp.command("analyze")
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--index", "-i", default="ndvi", show_default=True,
               type=click.Choice(["ndvi", "ndwi", "evi", "savi", "ndbi"]), help="Index to compute.")
@click.option("--sensor", "-s", default="sentinel2", show_default=True,
               type=click.Choice(["sentinel2", "landsat8", "planet", "naip"]),
               help="Band-order preset (positions of red/nir/... in your files).")
@click.option("--dates", default=None, help="Comma-separated label per image, e.g. 2024-01,2024-02 (default 1..N).")
@click.option("--period", type=int, default=None, help="Steps per seasonal cycle (12 for monthly): trend with the cycle removed.")
@click.option("--output", "-o", default=None, help="Write the full series (mean/std/p10/p90/valid_fraction) as JSON.")
def ts_analyze(images, index, sensor, dates, period, output):
    """Mean index per image, in the order given, plus a linear trend.

    Images must be listed in time order. An image that cannot be read prints N/A and a warning.
    """
    ts, series = _ts_series(images, index, sensor, dates)
    click.echo(f"\n  {index.upper()} series: {len(images)} images")
    for lab, v, vf in zip(series["dates"], series["mean"], series["valid_fraction"]):
        if v is None:
            click.echo(f"  {lab!s:>10}  N/A")
        else:
            click.echo(f"  {lab!s:>10}  {v:.4f}   (valid pixels {vf:.0%})")
    try:
        trend = ts.compute_trend(series, period=period)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if trend["direction"] == "insufficient_data":
        click.echo("  Trend: not enough dates (need 3)\n")
    else:
        sig = {True: "significant", False: "not significant", None: ""}[trend.get("significant")]
        click.echo(f"  Trend: {trend['direction']} (slope={trend['slope']:.6f} per step, "
                   f"R2={trend['r_squared']:.2f}, p={trend.get('p_value')}; {sig})\n")
    if output:
        import json
        with open(output, "w") as f:
            json.dump(series, f, indent=2)
        click.echo(f"  ✓ Saved: {output}\n")


@timeseries_grp.command("anomaly")
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--threshold", "-t", default=None, type=float, help="Score above which a date is flagged (default 2.5 zscore, 3.5 mad).")
@click.option("--method", "-m", default="zscore", show_default=True, type=click.Choice(["zscore", "mad"]),
               help="mad = median-based, robust to the anomaly itself.")
@click.option("--index", "-i", default="ndvi", show_default=True,
               type=click.Choice(["ndvi", "ndwi", "evi", "savi", "ndbi"]))
@click.option("--sensor", "-s", default="sentinel2", show_default=True,
               type=click.Choice(["sentinel2", "landsat8", "planet", "naip"]))
@click.option("--dates", default=None, help="Comma-separated label per image.")
@click.option("--period", type=int, default=None, help="Steps per seasonal cycle (12 for monthly): compare like with like.")
def ts_anomaly(images, threshold, method, index, sensor, dates, period):
    """Flag dates whose mean index is unusual (needs at least 5 images with data)."""
    ts, series = _ts_series(images, index, sensor, dates)
    try:
        anomalies = ts.detect_anomalies(series, method=method, threshold=threshold, period=period)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if anomalies:
        click.echo(f"\n  Found {len(anomalies)} anomalies ({method}):")
        for a in anomalies:
            click.echo(f"  {a['date']}: value={a['value']:.4f} expected={a['expected']:.4f} z={a['zscore']:.2f} ({a['type']})")
    else:
        click.echo("  No anomalies detected.")
    click.echo()

def main() -> None:
    """CLI entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════
# DATASETS commands (Phase 6.2)
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def datasets() -> None:
    """Browse and search a catalogue of about 460 remote sensing datasets (metadata only; nothing is downloaded)."""


@datasets.command("list")
@click.option("--domain", default=None, help="Filter by research domain.")
@click.option("--modality", default=None, help="Filter by data modality.")
@click.option("--task", default=None, help="Filter by ML task (segmentation, detection, …).")
@click.option("--min-year", default=None, type=int)
@click.option("--max-volume-gb", default=None, type=float, help="Max dataset size in GB.")
@click.option("--max-res-m", default=None, type=float, help="Max spatial resolution in metres.")
@click.option("--limit", default=25, show_default=True)
def datasets_list(domain, modality, task, min_year, max_volume_gb, max_res_m, limit):
    """List datasets from the catalogue (about 460), optionally filtered."""
    from pygeovision.datasets.registry import dataset_registry
    items = dataset_registry.filter(
        domain=domain, modality=modality, task=task,
        min_year=min_year, max_volume_gb=max_volume_gb,
        max_resolution_m=max_res_m,
    )
    items = items[:limit]
    dataset_registry.print_table(items)


@datasets.command("search")
@click.argument("query")
@click.option("--limit", default=20, show_default=True)
def datasets_search(query, limit):
    """Search datasets by keyword (name, description, domain, task)."""
    from pygeovision.datasets.registry import dataset_registry
    results = dataset_registry.search(query)[:limit]
    if not results:
        click.echo(f"No datasets found for '{query}'.")
        return
    click.echo(f"\nFound {len(results)} dataset(s) matching '{query}':\n")
    dataset_registry.print_table(results)


@datasets.command("info")
@click.argument("name")
def datasets_info(name):
    """Show full metadata for a dataset."""
    from pygeovision.datasets.loader import DatasetLoader
    try:
        DatasetLoader().info(name)
    except KeyError as exc:
        raise click.ClickException(str(exc.args[0])) from exc


@datasets.command("download")
@click.argument("name")
@click.option("--output", "-o", default=None, help="Output directory.")
def datasets_download(name, output):
    """Show where dataset NAME can be obtained.

    No catalogued dataset has a direct file link (they sit behind project pages, registration or
    terms), so this never downloads: it exits 1 and prints the page to visit.
    """
    from pygeovision.datasets.loader import DatasetLoader
    try:
        DatasetLoader().download(name, output)
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc.args[0])) from exc


@datasets.command("top")
@click.argument("task", default="segmentation")
@click.option("--n", default=5, show_default=True, help="Number of top datasets to show.")
@click.option("--modality", default=None, help="Only datasets of this modality (rgb, multispectral, sar, ...).")
def datasets_top(task, n, modality):
    """Show the top-N datasets for a given ML task (EarthNets ranking).

    The ranking is a metadata heuristic (newer, larger, more classes, finer pixels), not a
    quality measure; it mixes modalities unless you pass --modality.

    \b
    Tasks: segmentation, detection, classification, change_detection,
           multi_label, regression, prediction, self_supervised, vqa
    """
    from pygeovision.datasets.registry import dataset_registry
    top = dataset_registry.top_for_task(task, n=n, modality=modality)
    if not top:
        click.echo(f"No datasets found for task '{task}'" + (f" with modality '{modality}'." if modality else "."))
        return
    click.echo(f"\nTop-{n} datasets for '{task}' (EarthNets ranking):\n")
    dataset_registry.print_table(top)


@datasets.command("similar")
@click.argument("name")
@click.option("--n", default=10, show_default=True)
def datasets_similar(name, n):
    """Find datasets similar to NAME using the EarthNets similarity formula."""
    from pygeovision.datasets.registry import dataset_registry
    try:
        similar = dataset_registry.similar_to(name, n=n)
    except KeyError as exc:
        raise click.ClickException(str(exc.args[0])) from exc
    click.echo(f"\nDatasets most similar to '{name}':\n")
    dataset_registry.print_table(similar)


@datasets.command("domains")
def datasets_domains():
    """List all research domains in the catalog."""
    from pygeovision.datasets.registry import dataset_registry
    s = dataset_registry.summary()
    click.echo(f"\nDatasets by domain ({len(s['domains'])} domains):\n")
    for domain, count in sorted(s["domains"].items(), key=lambda x: -x[1]):
        click.echo(f"  {domain:<25} {count:>4} datasets")


@datasets.command("stats")
def datasets_stats():
    """Show catalog statistics (total datasets, volume, domains, tasks)."""
    from pygeovision.datasets.registry import dataset_registry
    s = dataset_registry.summary()
    click.echo(f"\n{'═'*50}")
    click.echo("  PyGeoVision Dataset Catalog (EarthNets)")
    click.echo(f"{'═'*50}")
    click.echo(f"  Total datasets : {s['total_datasets']}")
    click.echo(f"  Total volume   : {s['total_volume_tb']} TB")
    click.echo(f"  Year range     : {s['year_range'][0]} – {s['year_range'][1]}")
    click.echo(f"  Domains        : {len(s['domains'])}")
    click.echo(f"  Unique tasks   : {len(s['tasks'])}")
    click.echo(f"\n  Tasks: {', '.join(sorted(s['tasks']))}")


# ═══════════════════════════════════════════════════════════════════════
# MODELS zoo commands (Phase 2 / Phase 6.3)
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def zoo() -> None:
    """Browse the AI Model Zoo, a metadata catalogue of about 96 architectures (it describes models; it does not build them)."""


@zoo.command("list")
@click.option("--task", default=None, help="Filter by task (segmentation, detection, …).")
@click.option("--tag", default=None, help="Filter by tag (transformer, real_time, sar, …).")
@click.option("--max-params", default=None, type=float, help="Max parameters in millions.")
@click.option("--hf-only", is_flag=True, help="Only models with HuggingFace weights.")
def zoo_list(task, tag, max_params, hf_only):
    """List models from the AI Model Zoo."""
    from pygeovision.ai.models.zoo import model_zoo
    items = model_zoo.filter(task=task, tag=tag, max_params_m=max_params)
    if hf_only:
        items = [m for m in items if m.hf_model_id]
    model_zoo.print_table(items)


@zoo.command("search")
@click.argument("query")
def zoo_search(query):
    """Search the model zoo by name, architecture, or tag."""
    from pygeovision.ai.models.zoo import model_zoo
    results = model_zoo.search(query)
    if not results:
        click.echo(f"No models found for '{query}'.")
        return
    model_zoo.print_table(results)


@zoo.command("top")
@click.argument("task")
@click.option("--n", default=5, show_default=True)
def zoo_top(task, n):
    """Show the top-N models for a task.

    \b
    Tasks: segmentation, detection, classification, change_detection,
           foundation, vlm, 3d, timeseries, super_resolution
    """
    from pygeovision.ai.models.zoo import model_zoo
    top = model_zoo.top_for_task(task, n=n)
    click.echo(f"\nTop-{n} models for '{task}':\n")
    model_zoo.print_table(top)


@zoo.command("info")
@click.argument("name")
def zoo_info(name):
    """Show full metadata for a model."""
    from pygeovision.ai.models.zoo import model_zoo
    try:
        m = model_zoo[name]
        click.echo(f"\nModel: {m.name}")
        click.echo(f"  Task:          {m.task}")
        click.echo(f"  Architecture:  {m.architecture}")
        click.echo(f"  Backbone:      {m.backbone}")
        click.echo(f"  Parameters:    {m.params_m:.1f}M")
        click.echo(f"  Input bands:   {m.input_bands}")
        click.echo(f"  Pretrained:    {'Yes' if m.pretrained_available else 'No'}")
        if m.hf_model_id:
            click.echo(f"  HF Hub ID:     {m.hf_model_id}")
        if m.description:
            click.echo(f"  Description:   {m.description}")
        if m.tags:
            click.echo(f"  Tags:          {', '.join(m.tags)}")
        if m.paper_url:
            click.echo(f"  Paper:         {m.paper_url}")
    except KeyError as exc:
        click.echo(f"Error: {exc}", err=True)


@zoo.command("stats")
def zoo_stats():
    """Show model zoo statistics."""
    from pygeovision.ai.models.zoo import model_zoo
    s = model_zoo.summary()
    click.echo(f"\n{'═'*50}")
    click.echo("  PyGeoVision AI Model Zoo")
    click.echo(f"{'═'*50}")
    click.echo(f"  Total models      : {s['total_models']}")
    click.echo(f"  With HF weights   : {s['with_hf_weights']}")
    click.echo(f"  Pretrained avail  : {s['pretrained']}")
    click.echo("\n  By task:")
    for task, count in sorted(s["tasks"].items(), key=lambda x: -x[1]):
        click.echo(f"    {task:<22} {count:>3} models")


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK commands (Phase 4.4 / Phase 6.5)
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def benchmark() -> None:
    """Benchmarking tools — evaluate models, compare results, generate leaderboards."""


@benchmark.command("datasets")
@click.argument("task")
@click.option("--n", default=5, show_default=True, help="Number of top datasets to show.")
@click.option("--modality", default=None, help="Only datasets of this modality (rgb, multispectral, sar, ...).")
def benchmark_datasets(task, n, modality):
    """List candidate benchmark datasets for a task, ranked by the EarthNets metadata heuristic.

    \b
    Example:
        pygeovision benchmark datasets segmentation --n 5
        pygeovision benchmark datasets detection
        pygeovision benchmark datasets change_detection
    """
    from pygeovision.datasets.registry import dataset_registry
    top = dataset_registry.top_for_task(task, n=n, modality=modality)
    if not top:
        click.echo(f"No datasets found for task '{task}'" + (f" with modality '{modality}'." if modality else "."))
        return
    click.echo(f"\n{'═'*70}")
    click.echo(f"  Candidate Benchmark Datasets for '{task}' (EarthNets Top-{n})")
    click.echo(f"{'═'*70}")
    dataset_registry.print_table(top)


@benchmark.command("models")
@click.argument("task")
@click.option("--n", default=5, show_default=True)
def benchmark_models(task, n):
    """Show the top-N recommended models to benchmark for a task."""
    from pygeovision.ai.models.zoo import model_zoo
    top = model_zoo.top_for_task(task, n=n)
    if not top:
        n_all = len(model_zoo.filter(task=task))
        if n_all:
            click.echo(f"No models with pretrained weights for task '{task}' "
                       f"({n_all} registered without them; see 'pygeovision zoo list --task {task}').")
        else:
            click.echo(f"No models found for task '{task}'.")
        return
    click.echo(f"\nRecommended models for '{task}' benchmarks:")
    model_zoo.print_table(top)


@benchmark.command("tasks")
def benchmark_tasks():
    """List all available benchmark tasks."""
    from pygeovision.ai.models.zoo import model_zoo
    from pygeovision.datasets.registry import dataset_registry
    d_tasks = set(dataset_registry.tasks())
    m_tasks = set(model_zoo.tasks())
    all_tasks = sorted(d_tasks | m_tasks)
    click.echo(f"\nAvailable benchmark tasks ({len(all_tasks)}):\n")
    for t in all_tasks:
        d_count = len(dataset_registry.filter(task=t))
        m_count = len(model_zoo.filter(task=t))
        click.echo(f"  {t:<25} {d_count:>4} datasets   {m_count:>4} models")


# ═══════════════════════════════════════════════════════════════════════
# VALIDATE command group — mandatory data validation
# ═══════════════════════════════════════════════════════════════════════

@cli.group("validate")
def validate_grp():
    """Validate satellite data before model inference."""


@validate_grp.command("run")
@click.argument("input_path")
@click.option("--required-bands", type=int, default=None, help="Minimum number of bands required.")
@click.option("--value-range", default=None, help="Expected value range, e.g. '0,10000'.")
@click.option("--target-crs", default=None, help="Expected CRS, e.g. EPSG:4326.")
@click.option("--report", default=None, help="Save HTML/JSON report to this path.")
@click.option("--report-fmt", default="html", type=click.Choice(["html","json"]), show_default=True)
@click.option("--mode", default="fix", type=click.Choice(["strict","fix","warn"]), show_default=True)
def validate_run(input_path, required_bands, value_range, target_crs, report, report_fmt, mode):
    """Validate a GeoTIFF — check nulls, dtype, CRS, bounds, outliers.

    \b
    Examples:
        pygeovision validate run scene.tif
        pygeovision validate run scene.tif --required-bands 6 --value-range 0,10000
        pygeovision validate run scene.tif --report validation.html
    """
    from pygeovision.data.validator import DataValidator
    v = DataValidator(mode=mode)
    vr = None
    if value_range:
        lo, hi = [float(x) for x in value_range.split(",")]
        vr = (lo, hi)
    result = v.validate(input_path, required_bands=required_bands,
                        value_range=vr, target_crs=target_crs)
    click.echo(result.summary())
    if report:
        v.generate_report(input_path, report, fmt=report_fmt)
        click.echo(f"\n  Report → {report}")


@validate_grp.command("for-inference")
@click.argument("input_path")
@click.option("--model-type", default="segmentation",
              type=click.Choice(["segmentation","detection","change_detection",
                                 "classification","foundation","regression"]),
              show_default=True)
def validate_for_inference(input_path, model_type):
    """Validate and auto-fix data for a specific model type.

    \b
    Example:
        pygeovision validate for-inference scene.tif --model-type segmentation
    """
    from pygeovision.data.validator import DataValidator
    v = DataValidator(mode="fix")
    arr = v.validate_for_inference(input_path, model_type=model_type)
    click.echo(f"  ✓ Ready for {model_type}  |  shape={arr.shape}  "
               f"dtype={arr.dtype}  range=[{arr.min():.4f}, {arr.max():.4f}]")


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESS command group
# ═══════════════════════════════════════════════════════════════════════

@cli.group("preprocess")
def preprocess_grp():
    """Satellite image preprocessing — stack, clip, mask, normalise, resample."""


@preprocess_grp.command("stack")
@click.argument("scene_dir")
@click.option("--bands", required=True, help="Comma-separated band names, e.g. B02,B03,B04,B08")
@click.option("--output", required=True, help="Output stacked GeoTIFF path.")
@click.option("--validate", is_flag=True, default=False, help="Validate output after stacking.")
def preprocess_stack(scene_dir, bands, output, validate):
    """Stack individual band files from a scene directory.

    \b
    Example:
        pygeovision preprocess stack ./S2C_20240628/ --bands B02,B03,B04,B08,B11,B12 --output stack.tif
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    band_list = [b.strip() for b in bands.split(",")]
    out = pre.stack_from_dir(scene_dir, band_list, output)
    click.echo(f"  ✓ Stacked {len(band_list)} bands → {out}")
    if validate:
        report = pre.validate(out)
        click.echo(report.summary())


@preprocess_grp.command("clip")
@click.argument("input_path")
@click.option("--bbox", required=True, help="minlon,minlat,maxlon,maxlat in WGS84.")
@click.option("--output", default=None, help="Output path (default: _clipped).")
@click.option("--validate", is_flag=True, default=False)
def preprocess_clip(input_path, bbox, output, validate):
    """Clip a raster to a bounding box.

    \b
    Example:
        pygeovision preprocess clip stack.tif --bbox -74.1,40.6,-73.7,40.9
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    try:
        coords = tuple(float(x) for x in bbox.split(","))
    except ValueError:
        raise click.BadParameter("give four numbers: minlon,minlat,maxlon,maxlat", param_hint="--bbox") from None
    if len(coords) != 4:
        raise click.BadParameter("give four numbers: minlon,minlat,maxlon,maxlat", param_hint="--bbox")
    try:
        out = pre.clip_to_bbox(input_path, coords, output_path=output)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"  ✓ Clipped → {out}")
    if validate:
        click.echo(pre.validate(out).summary())


@preprocess_grp.command("normalise")
@click.argument("input_path")
@click.option("--method", default="minmax",
              type=click.Choice(["minmax","zscore","percentile","scale_factor"]),
              show_default=True)
@click.option("--scale-factor", default=10000.0, show_default=True,
              help="Divisor for scale_factor method (Sentinel-2 L2A = 10000).")
@click.option("--percentile", default=2.0, show_default=True)
@click.option("--output", default=None)
@click.option("--validate", is_flag=True, default=False)
def preprocess_normalise(input_path, method, scale_factor, percentile, output, validate):
    """Normalise pixel values.

    \b
    Example:
        pygeovision preprocess normalise stack.tif --method scale_factor --scale-factor 10000
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    out = pre.normalise(input_path, method=method, output_path=output,
                         scale_factor=scale_factor, percentile=percentile)
    click.echo(f"  ✓ Normalised ({method}) → {out}")
    if validate:
        click.echo(pre.validate(out).summary())


@preprocess_grp.command("resample")
@click.argument("input_path")
@click.option("--resolution", required=True, type=float, help="Target pixel size in metres.")
@click.option("--output", default=None)
@click.option("--method", default="bilinear",
              type=click.Choice(["nearest","bilinear","cubic","lanczos","average"]),
              show_default=True)
@click.option("--validate", is_flag=True, default=False)
def preprocess_resample(input_path, resolution, output, method, validate):
    """Resample a raster to a target resolution.

    \b
    Example:
        pygeovision preprocess resample s2_20m.tif --resolution 10
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    out = pre.resample(input_path, resolution, output_path=output, resampling=method)
    click.echo(f"  ✓ Resampled to {resolution}m → {out}")
    if validate:
        click.echo(pre.validate(out).summary())


@preprocess_grp.command("pipeline")
@click.argument("input_path")
@click.option("--output", required=True, help="Final output path.")
@click.option("--bands", default=None, help="Bands to stack, e.g. B02,B03,B04,B08,B11,B12")
@click.option("--bbox", default=None, help="Clip bbox: minlon,minlat,maxlon,maxlat")
@click.option("--scl", default=None, help="SCL band path for cloud masking.")
@click.option("--normalise", default=None,
              type=click.Choice(["minmax","zscore","percentile","scale_factor","none"]))
@click.option("--resample-m", default=None, type=float, help="Resample to N metres.")
@click.option("--validate", is_flag=True, default=False)
def preprocess_pipeline(input_path, output, bands, bbox, scl, normalise, resample_m, validate):
    """Run the full preprocessing pipeline in one command.

    \b
    Example:
        pygeovision preprocess pipeline ./S2C_20240628/ \\
            --output ready.tif \\
            --bands B02,B03,B04,B08,B11,B12 \\
            --bbox -74.1,40.6,-73.7,40.9 \\
            --scl SCL.tif \\
            --normalise scale_factor
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    band_list = [b.strip() for b in bands.split(",")] if bands else None
    bbox_t    = tuple(float(x) for x in bbox.split(",")) if bbox else None
    norm_val  = None if (not normalise or normalise == "none") else normalise

    result = pre.pipeline(
        input_path=input_path,
        output_path=output,
        stack_bands=band_list,
        bbox=bbox_t,
        scl_path=scl,
        normalise=norm_val,
        resample_m=resample_m,
    )
    click.echo("  ✓ Pipeline complete")
    click.echo(f"    Output  : {result['output_path']}")
    click.echo(f"    Shape   : {result['shape']}")
    click.echo(f"    Res     : {result['resolution_m']:.1f}m")
    click.echo(f"    Steps   : {' → '.join(result['steps_applied'])}")
    if validate:
        click.echo(pre.validate(result["output_path"]).summary())


# ═══════════════════════════════════════════════════════════════════════
# INDICES command group — 21 spectral indices
# ═══════════════════════════════════════════════════════════════════════

@cli.group("indices")
def indices_grp():
    """Compute validated spectral indices (NDVI, EVI, NBR, TCT, PCA …)."""


@indices_grp.command("compute")
@click.argument("input_path")
@click.argument("index_name")
@click.option("--output", default=None, help="Output GeoTIFF path.")
@click.option("--red", default=3, show_default=True, type=int, help="1-based red band index.")
@click.option("--nir", default=4, show_default=True, type=int, help="1-based NIR band index.")
@click.option("--green", default=2, show_default=True, type=int)
@click.option("--blue", default=1, show_default=True, type=int)
@click.option("--swir1", default=5, show_default=True, type=int)
@click.option("--swir2", default=6, show_default=True, type=int)
@click.option("--red-edge", "red_edge", default=None, type=int,
              help="1-based red-edge band index (required for ndre; the default six-band stack has none).")
@click.option("--validate", is_flag=True, default=False)
def indices_compute(input_path, index_name, output, red, nir, green, blue, swir1, swir2, red_edge, validate):
    """Compute a spectral index.

    INDEX_NAME: ndvi, evi, savi, msavi, ndwi, mndwi, ndbi, bsi, nbr, bai, ndsi, rvi, arvi, lswi, ndre, wdrvi, vari, exg

    \b
    Examples:
        pygeovision indices compute stack.tif ndvi --output ndvi.tif
        pygeovision indices compute stack.tif nbr  --nir 4 --swir2 6
    """
    from pygeovision.data.indices import SpectralIndices
    ix = SpectralIndices()

    fn_map = {
        "ndvi": lambda: ix.ndvi(input_path, red, nir, output),
        "evi":  lambda: ix.evi( input_path, blue, red, nir, output_path=output),
        "savi": lambda: ix.savi(input_path, red, nir, output),
        "msavi":lambda: ix.msavi(input_path, red, nir, output),
        "ndwi": lambda: ix.ndwi(input_path, green, nir, output),
        "mndwi":lambda: ix.mndwi(input_path, green, swir1, output),
        "ndbi": lambda: ix.ndbi(input_path, swir1, nir, output),
        "bsi":  lambda: ix.bsi( input_path, blue, red, nir, output_path=output),
        "nbr":  lambda: ix.nbr( input_path, nir, swir2, output),
        "bai":  lambda: ix.bai( input_path, red, nir, output),
        "ndsi": lambda: ix.ndsi(input_path, green, swir1, output),
        "rvi":  lambda: ix.rvi( input_path, red, nir, output),
        "arvi": lambda: ix.arvi(input_path, blue, red, nir, output_path=output),
        "lswi": lambda: ix.lswi(input_path, nir, swir1, output),
        "ndre": lambda: ix.ndre(input_path, red_edge, nir, output),
        "wdrvi":lambda: ix.wdrvi(input_path, red, nir, output),
        "vari": lambda: ix.vari(input_path, blue, green, red, output),
        "exg":  lambda: ix.exg( input_path, blue, green, red, output),
    }
    idx = index_name.lower()
    if idx not in fn_map:
        click.echo(f"Error: unknown index '{index_name}'. Available: {sorted(fn_map)}", err=True)
        return
    if idx == "ndre" and red_edge is None:
        click.echo("Error: ndre needs a red-edge band. Pass --red-edge N (the 1-based position of Sentinel-2 B05 in "
                   "your file); the default six-band stack B02,B03,B04,B08,B11,B12 does not contain one.", err=True)
        raise SystemExit(2)

    result = fn_map[idx]()
    if isinstance(result, str):
        click.echo(f"  ✓ {index_name.upper()} → {result}")
        if validate:
            from pygeovision import PyGeoVision
            report = PyGeoVision().validator.validate(result)
            status = "PASSED" if report.passed else "FIXED (issues auto-corrected)"
            click.echo(f"  Validation: {status}")
            click.echo(f"  {report.summary()}")
    else:
        click.echo(f"  ✓ {index_name.upper()}  shape={result.shape}  "
                   f"range=[{float(result.min()):.4f}, {float(result.max()):.4f}]")
        if validate:
            click.echo("  Validation skipped: --validate requires a file output "
                       "(pass --output to write a GeoTIFF instead of an in-memory array).")


@indices_grp.command("all")
@click.argument("input_path")
@click.option("--output-dir", required=True, help="Directory for all index GeoTIFFs.")
@click.option("--indices", default=None,
              help="Comma-separated list. Default: every index the default band layout supports (NDRE is skipped unless a red-edge band is configured).")
def indices_all(input_path, output_dir, indices):
    """Compute all (or selected) spectral indices in one command.

    \b
    Example:
        pygeovision indices all stack.tif --output-dir ./indices/
        pygeovision indices all stack.tif --output-dir ./indices/ --indices ndvi,evi,nbr,ndwi
    """
    from pygeovision.data.indices import SpectralIndices
    ix = SpectralIndices()
    idx_list = [i.strip() for i in indices.split(",")] if indices else None
    results  = ix.compute_all(input_path, indices=idx_list, output_dir=output_dir)
    click.echo(f"\n  Computed {len(results)} indices → {output_dir}")
    for name, path in results.items():
        click.echo(f"    {name:<8} → {path}")


@indices_grp.command("list")
def indices_list():
    """List all available spectral indices."""
    indices = [
        ("ndvi",  "Normalized Difference Vegetation Index",  "Vegetation"),
        ("evi",   "Enhanced Vegetation Index",               "Vegetation"),
        ("savi",  "Soil-Adjusted Vegetation Index",          "Vegetation"),
        ("msavi", "Modified SAVI",                           "Vegetation"),
        ("arvi",  "Atmospherically Resistant Vegetation Index","Vegetation"),
        ("ndre",  "Normalized Difference Red Edge",          "Vegetation"),
        ("rvi",   "Ratio Vegetation Index",                  "Vegetation"),
        ("wdrvi", "Wide Dynamic Range Vegetation Index",     "Vegetation"),
        ("vari",  "Visible Atmospherically Resistant Index", "Vegetation/RGB"),
        ("exg",   "Excess Green Index",                      "Vegetation/RGB"),
        ("ndwi",  "Normalized Difference Water Index",       "Water"),
        ("mndwi", "Modified NDWI",                           "Water"),
        ("lswi",  "Land Surface Water Index",                "Water"),
        ("wri",   "Water Ratio Index",                       "Water"),
        ("ndbi",  "Normalized Difference Built-up Index",    "Urban"),
        ("bsi",   "Bare Soil Index",                         "Soil"),
        ("nbr",   "Normalized Burn Ratio",                   "Fire"),
        ("bai",   "Burn Area Index",                         "Fire"),
        ("ndsi",  "Normalized Difference Snow Index",        "Snow"),
        ("tct",   "Tasseled Cap Transform (B/G/W)",          "Transform"),
        ("pca",   "Principal Component Analysis",            "Transform"),
    ]
    click.echo(f"\n  {'Index':<8} {'Name':<42} Category")
    click.echo(f"  {'─'*8} {'─'*42} {'─'*16}")
    for name, desc, cat in indices:
        click.echo(f"  {name:<8} {desc:<42} {cat}")
    click.echo(f"\n  Total: {len(indices)} indices")


# ═══════════════════════════════════════════════════════════════════════
# POSTPROCESS command group
# ═══════════════════════════════════════════════════════════════════════

@cli.group("postprocess")
def postprocess_grp():
    """Postprocess model predictions — vectorise, smooth, statistics, export."""


@postprocess_grp.command("vectorise")
@click.argument("input_path")
@click.option("--output", required=True, help="Output GeoJSON path.")
@click.option("--target-class", default=None, type=int, help="Only vectorise this class.")
@click.option("--min-area", default=0.0, type=float, help="Minimum polygon area in m².")
@click.option("--simplify", default=0.0, type=float, help="Douglas-Peucker tolerance.")
@click.option("--validate", is_flag=True, default=False)
def post_vectorise(input_path, output, target_class, min_area, simplify, validate):
    """Vectorise a classification raster to GeoJSON polygons.

    \b
    Example:
        pygeovision postprocess vectorise pred.tif --output buildings.geojson --target-class 1 --min-area 50
    """
    from pygeovision.data.postprocess import PostProcessor
    post = PostProcessor()
    out = post.vectorise(input_path, output, target_class=target_class,
                          min_area_m2=min_area, simplify_tolerance=simplify)
    import json
    with open(out) as f:
        features = json.load(f).get("features", [])
    n = len(features)
    click.echo(f"  ✓ Vectorised → {out}  ({n} features)")

    if validate:
        from shapely.geometry import shape as shp_shape
        n_invalid = 0
        for feat in features:
            try:
                geom = shp_shape(feat["geometry"])
                if not geom.is_valid:
                    n_invalid += 1
            except Exception:
                n_invalid += 1
        status = "PASSED" if n_invalid == 0 else f"FAILED ({n_invalid}/{n} invalid geometries)"
        click.echo(f"  Validation: {status}")


@postprocess_grp.command("sieve")
@click.argument("input_path")
@click.option("--min-pixels", default=10, show_default=True, type=int)
@click.option("--output", default=None)
def post_sieve(input_path, min_pixels, output):
    """Remove small patches from a classification raster.

    \b
    Example:
        pygeovision postprocess sieve pred.tif --min-pixels 10
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().sieve_filter(input_path, min_pixels, output_path=output)
    click.echo(f"  ✓ Sieve (min={min_pixels}px) → {out}")


@postprocess_grp.command("smooth")
@click.argument("input_path")
@click.option("--output", required=True)
@click.option("--tolerance", default=0.5, show_default=True, type=float)
def post_smooth(input_path, output, tolerance):
    """Smooth vector polygon boundaries.

    \b
    Example:
        pygeovision postprocess smooth buildings.geojson --output smooth.geojson --tolerance 0.5
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().smooth(input_path, output, tolerance=tolerance)
    click.echo(f"  ✓ Smoothed → {out}")


@postprocess_grp.command("regularise")
@click.argument("input_path")
@click.option("--output", required=True)
@click.option("--min-area", default=10.0, type=float, show_default=True)
def post_regularise(input_path, output, min_area):
    """Regularise building footprints: snap near-axis-aligned edges to the building's dominant orientation.

    \b
    Example:
        pygeovision postprocess regularise buildings.geojson --output regular.geojson
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().regularise_buildings(input_path, output, min_area_m2=min_area)
    click.echo(f"  ✓ Regularised → {out}")


@postprocess_grp.command("zonal-stats")
@click.argument("raster_path")
@click.argument("vector_path")
@click.option("--output", required=True)
@click.option("--stats", default="mean,std,min,max,count", show_default=True)
def post_zonal_stats(raster_path, vector_path, output, stats):
    """Compute per-zone raster statistics.

    \b
    Example:
        pygeovision postprocess zonal-stats ndvi.tif fields.geojson --output stats.geojson
    """
    from pygeovision.data.postprocess import PostProcessor
    stat_list = [s.strip() for s in stats.split(",")]
    out = PostProcessor().zonal_statistics(raster_path, vector_path, output, stats=stat_list)
    click.echo(f"  ✓ Zonal statistics → {out}")


@postprocess_grp.command("accuracy")
@click.argument("prediction_path")
@click.argument("reference_path")
@click.option("--class-names", default=None, help="Comma-separated class names.")
def post_accuracy(prediction_path, reference_path, class_names):
    """Compute accuracy metrics vs a reference raster.

    \b
    Example:
        pygeovision postprocess accuracy pred.tif ref.tif --class-names Background,Building,Road
    """
    from pygeovision.data.postprocess import PostProcessor
    names = [n.strip() for n in class_names.split(",")] if class_names else None
    report = PostProcessor().accuracy_assessment(prediction_path, reference_path, class_names=names)
    click.echo(f"\n  Overall Accuracy : {report['overall_accuracy']:.4f}")
    click.echo(f"  Cohen's κ        : {report['kappa']:.4f}")
    click.echo(f"  Mean IoU         : {report['mean_iou']:.4f}")
    click.echo(f"  Total pixels     : {report['total_pixels']:,}")
    click.echo("\n  Per-class metrics:")
    click.echo(f"  {'Class':<6} {'Name':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'IoU':>8}")
    click.echo(f"  {'─'*6} {'─'*16} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    names_list = report.get("class_names", [])
    for i, (cls, m) in enumerate(report["per_class"].items()):
        nm = names_list[i] if i < len(names_list) else str(cls)
        click.echo(f"  {cls:<6} {nm:<16} {m['precision']:>10.4f} "
                   f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['iou']:>8.4f}")


@postprocess_grp.command("class-stats")
@click.argument("prediction_path")
def post_class_stats(prediction_path):
    """Print per-class pixel count and area statistics.

    \b
    Example:
        pygeovision postprocess class-stats prediction.tif
    """
    from pygeovision.data.postprocess import PostProcessor
    stats = PostProcessor().class_statistics(prediction_path)
    click.echo(f"\n  {'Class':>6} {'Pixels':>12} {'Area (ha)':>12} {'Area km²':>12} {'Pct':>8}")
    click.echo(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")
    for cls, info in sorted(stats.items()):
        click.echo(f"  {cls:>6} {info['pixels']:>12,} {info['area_ha']:>12.3f} "
                   f"{info['area_km2']:>12.5f} {info['pct']:>7.2f}%")


@postprocess_grp.command("cog")
@click.argument("input_path")
@click.option("--output", default=None)
def post_cog(input_path, output):
    """Convert a GeoTIFF to Cloud-Optimized GeoTIFF.

    \b
    Example:
        pygeovision postprocess cog prediction.tif --output prediction_cog.tif
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().to_cog(input_path, output_path=output)
    click.echo(f"  ✓ COG → {out}")


@postprocess_grp.command("export")
@click.argument("input_path")
@click.option("--output", required=True)
@click.option("--format", "fmt", default="geojson",
              type=click.Choice(["geojson","shp","gpkg","kml"]), show_default=True)
def post_export(input_path, output, fmt):
    """Export vector to GeoJSON, Shapefile, GeoPackage, or KML.

    \b
    Example:
        pygeovision postprocess export buildings.geojson --output buildings.gpkg --format gpkg
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().export(input_path, output, fmt=fmt)
    click.echo(f"  ✓ Exported ({fmt.upper()}) → {out}")

# ═══════════════════════════════════════════════════════════════════════════
# obia — object-based image analysis
# ═══════════════════════════════════════════════════════════════════════════
@cli.group("obia")
def obia_grp():
    """Object-based image analysis — segment, describe, classify, assess.

    \b
    Typical sequence:
        pygeovision obia demo ./obia_demo
        pygeovision obia segment ./obia_demo/scene.tif --scale 20 --output-labels labels.tif
        pygeovision obia features ./obia_demo/scene.tif labels.tif --output features.csv
        pygeovision obia classify ./obia_demo/scene.tif labels.tif \\
            --samples ./obia_demo/reference.gpkg --class-column class --output classified.gpkg
        pygeovision obia assess-classification labels.tif classified.gpkg \\
            ./obia_demo/reference.gpkg --class-column class
    """


def _obia_parse_params(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            raise click.BadParameter(f"--param expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def _obia_read_labels(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1).astype("int64"), src.transform, src.crs


@obia_grp.command("demo")
@click.argument("output_dir")
@click.option("--size", default=160, show_default=True, help="Scene width/height in pixels.")
@click.option("--seed", default=0, show_default=True, help="Random seed (a different scene per seed).")
def obia_demo(output_dir, size, seed):
    """Write a synthetic teaching scene with known reference data.

    Creates scene.tif (blue, green, red, nir), reference.gpkg (reference objects
    with a 'class' column) and reference_ids.tif (one id per reference object).
    """
    import os

    from pygeovision.obia import make_demo_scene, save_raster
    os.makedirs(output_dir, exist_ok=True)
    sc = make_demo_scene(size, seed)
    r = sc.raster
    save_raster(os.path.join(output_dir, "scene.tif"), r.image, r.transform, r.crs, r.band_names)
    save_raster(os.path.join(output_dir, "reference_ids.tif"), sc.ref_labels, r.transform, r.crs, dtype="int32")
    sc.reference.to_file(os.path.join(output_dir, "reference.gpkg"), driver="GPKG")
    click.echo(f"  ✓ Demo scene ({size}x{size}, seed {seed}) → {output_dir}")
    click.echo(f"    classes: {', '.join(sc.class_names)}")


@obia_grp.command("segment")
@click.argument("input_path")
@click.option("--method", default="multires", show_default=True,
              type=click.Choice(["multires", "slic", "felzenszwalb", "quickshift", "watershed", "chessboard"]))
@click.option("--scale", default=None, type=float, help="Scale parameter (multires; felzenszwalb).")
@click.option("--color-weight", default=0.9, show_default=True, help="multires: weight of spectral vs shape criterion.")
@click.option("--compactness-weight", default=0.5, show_default=True, help="multires: compactness vs smoothness.")
@click.option("--param", "params", multiple=True, help="Extra method parameter, key=value (repeatable).")
@click.option("--output-labels", required=True, help="Output GeoTIFF of object ids.")
@click.option("--output-vector", default=None, help="Optional vector output (.gpkg/.geojson).")
def obia_segment(input_path, method, scale, color_weight, compactness_weight, params,
                 output_labels, output_vector):
    """Segment a raster into image objects."""
    from pygeovision.obia import segment
    kw = _obia_parse_params(params)
    if method == "multires":
        kw.update(color_weight=color_weight, compactness_weight=compactness_weight)
    if scale is not None:
        kw["scale"] = scale
    seg = segment(input_path, method, **kw)
    seg.save(output_labels)
    click.echo(f"  ✓ {seg.n_objects} objects ({method}) → {output_labels}")
    if output_vector:
        seg.to_vector().to_file(output_vector)
        click.echo(f"  ✓ Object polygons → {output_vector}")


@obia_grp.command("features")
@click.argument("input_path")
@click.argument("labels_path")
@click.option("--output", required=True, help="Output .csv, or a vector file (.gpkg/.geojson) to keep geometry.")
@click.option("--texture", "texture", multiple=True, help="Band/layer name to compute GLCM texture for (repeatable).")
@click.option("--indices/--no-indices", default=True, show_default=True, help="Add NDVI/NDWI when bands allow.")
def obia_features(input_path, labels_path, output, texture, indices):
    """Compute the object feature table (spectral, shape, texture, context)."""
    from pygeovision.obia import extract_features, labels_to_gdf, load_raster, spectral_layers
    rd = load_raster(input_path)
    lab, tr, crs = _obia_read_labels(labels_path)
    extra = spectral_layers(rd) if indices else None
    F = extract_features(lab, rd, extra_layers=extra, texture_layers=list(texture) or None)
    if output.lower().endswith(".csv"):
        F.to_csv(output)
    else:
        labels_to_gdf(lab, tr, crs, F).to_file(output)
    click.echo(f"  ✓ {len(F)} objects × {F.shape[1]} features → {output}")


@obia_grp.command("classify")
@click.argument("input_path")
@click.argument("labels_path")
@click.option("--output", required=True, help="Output vector file (.gpkg/.geojson).")
@click.option("--samples", default=None, help="Vector file of training polygons/points (supervised).")
@click.option("--rules", default=None, help="YAML rule file (rule-based): rules: [{class: water, when: 'mean_ndwi > 0.3'}].")
@click.option("--class-column", default="class", show_default=True)
@click.option("--method", default="rf", show_default=True, type=click.Choice(["rf", "knn", "svm", "tree"]))
@click.option("--output-raster", default=None, help="Also write classes as a GeoTIFF (codes in the class table).")
def obia_classify(input_path, labels_path, output, samples, rules, class_column, method, output_raster):
    """Classify objects from samples (supervised) or a YAML rule set."""
    import geopandas as gpd

    from pygeovision.obia import (
        ObjectClassifier,
        RuleBasedClassifier,
        extract_features,
        load_raster,
        samples_from_geometries,
        save_raster,
        spectral_layers,
    )
    if bool(samples) == bool(rules):
        raise click.UsageError("give exactly one of --samples or --rules")
    rd = load_raster(input_path)
    lab, tr, crs = _obia_read_labels(labels_path)
    F = extract_features(lab, rd, extra_layers=spectral_layers(rd))
    if rules:
        import yaml
        with open(rules) as fh:
            spec = yaml.safe_load(fh)
        try:
            rb = RuleBasedClassifier([(r["class"], r["when"]) for r in spec["rules"]],
                                     default=spec.get("default", "unclassified"))
        except (KeyError, TypeError, ValueError) as exc:
            raise click.ClickException(
                f"{rules}: expected 'rules: [{{class: water, when: \"mean_ndwi > 0.3\"}}, ...]' ({exc!r})")
        try:
            res = rb.classify(F)
        except ValueError as exc:
            raise click.ClickException(str(exc))
    else:
        gdf = gpd.read_file(samples)
        if crs is not None and gdf.crs is not None:
            gdf = gdf.to_crs(crs)
        s = samples_from_geometries(lab, gdf, class_column, tr)
        if s.empty:
            raise click.ClickException("no image object overlaps the sample geometries")
        try:
            res = ObjectClassifier(method).fit(F, s).predict(F)
        except ValueError as exc:
            raise click.ClickException(str(exc))
        click.echo(f"  trained on {len(s)} sample objects ({s.nunique()} classes)")
        click.echo("  note: judge accuracy with reference data that was NOT used for training")
    res.to_vector(lab, tr, crs).to_file(output)
    click.echo(f"  ✓ Classified {len(res.classes)} objects → {output}")
    click.echo("  " + res.summary().to_string().replace("\n", "\n  "))
    if output_raster:
        save_raster(output_raster, res.to_raster(lab), tr, crs, dtype="uint8")
        codes = ", ".join(f"{c}={i}" for c, i in res.codes().items())
        click.echo(f"  ✓ Class raster → {output_raster}  (codes: {codes}; 0 = unclassified)")


@obia_grp.command("scale")
@click.argument("input_path")
@click.option("--scales", default="5,10,15,20,30,40,60,80", show_default=True, help="Comma-separated scales.")
@click.option("--reference", default=None, help="Reference object polygons: also report segmentation quality per scale.")
@click.option("--output", default=None, help="Write the table to CSV.")
def obia_scale(input_path, scales, reference, output):
    """Compare scale parameters (diagnostics; with --reference also over/under-segmentation)."""
    import geopandas as gpd

    from pygeovision.obia import (
        load_raster,
        rasterize_geometries,
        scale_diagnostics,
        scale_vs_reference,
    )
    sc = [float(x) for x in scales.split(",")]
    rd = load_raster(input_path)
    if reference:
        gdf = gpd.read_file(reference)
        if rd.crs is not None and gdf.crs is not None:
            gdf = gdf.to_crs(rd.crs)
        ref, _ = rasterize_geometries(gdf, rd.shape, rd.transform)
        df = scale_vs_reference(rd, sc, ref)
        best = df["mean_D"].idxmin()
        click.echo(df.round(3).to_string())
        click.echo(f"\n  best scale by D (lower = better): {best:g}")
    else:
        df = scale_diagnostics(rd, sc)
        click.echo(df.round(2).to_string())
        cand = df.index[df["candidate"]].tolist()
        click.echo(f"\n  ROC-LV candidate scales (heuristic — verify visually): {cand or 'none'}")
    if output:
        df.to_csv(output)
        click.echo(f"  ✓ → {output}")


@obia_grp.command("assess-segmentation")
@click.argument("labels_path")
@click.argument("reference_path")
@click.option("--output", default=None, help="Per-reference-object table as CSV.")
def obia_assess_seg(labels_path, reference_path, output):
    """Over-/under-segmentation of a label raster against reference objects."""
    import geopandas as gpd

    from pygeovision.obia import rasterize_geometries, segmentation_quality
    lab, tr, crs = _obia_read_labels(labels_path)
    gdf = gpd.read_file(reference_path)
    if crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(crs)
    ref, _ = rasterize_geometries(gdf, lab.shape, tr)
    table, summ = segmentation_quality(lab, ref)
    click.echo(f"  reference objects: {summ['n_reference']}   segments: {summ['n_segments_total']}")
    click.echo(f"  mean over-segmentation : {summ['mean_over_seg']:.3f}   (0 = none)")
    click.echo(f"  mean under-segmentation: {summ['mean_under_seg']:.3f}   (0 = none)")
    click.echo(f"  mean D                 : {summ['mean_D']:.3f}   (0 = perfect)")
    click.echo(f"  mean QR (Jaccard)      : {summ['mean_QR']:.3f}   (1 = perfect)")
    if output:
        table.to_csv(output)
        click.echo(f"  ✓ → {output}")


@obia_grp.command("assess-classification")
@click.argument("labels_path")
@click.argument("classified_path")
@click.argument("reference_path")
@click.option("--class-column", default="class", show_default=True, help="Class column in the reference file.")
@click.option("--weight-by-area/--count-objects", default=True, show_default=True)
@click.option("--output", default=None, help="Write the confusion matrix to CSV.")
def obia_assess_cls(labels_path, classified_path, reference_path, class_column, weight_by_area, output):
    """Object-based accuracy of a classification against reference polygons/points."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from pygeovision.obia import confusion_report, samples_from_geometries
    lab, tr, crs = _obia_read_labels(labels_path)
    cl = gpd.read_file(classified_path)
    pred = cl.set_index("obj_id")["class"]
    ref_gdf = gpd.read_file(reference_path)
    if crs is not None and ref_gdf.crs is not None:
        ref_gdf = ref_gdf.to_crs(crs)
    ref = samples_from_geometries(lab, ref_gdf, class_column, tr)
    if ref.empty:
        raise click.ClickException("no object is covered by the reference geometries")
    w = None
    if weight_by_area:
        w = pd.Series(np.bincount(lab.ravel()), dtype=float)
    rep = confusion_report(ref, pred, w)
    click.echo(f"  assessed {rep['n_objects']} objects ({'area-weighted' if weight_by_area else 'unweighted'})")
    click.echo(f"  overall accuracy: {rep['overall_accuracy']:.3f}   kappa: {rep['kappa']:.3f}")
    click.echo("\n" + rep["per_class"].round(3).to_string())
    click.echo("\n  confusion (rows = reference, columns = predicted):\n" + rep["confusion"].round(0).astype(int).to_string())
    if output:
        rep["confusion"].to_csv(output)
        click.echo(f"\n  ✓ → {output}")
