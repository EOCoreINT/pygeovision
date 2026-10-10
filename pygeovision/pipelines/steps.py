"""Pipeline step definitions.

Every built-in step either does its work or raises :class:`StepError`; none returns an "error" value
that the orchestrator would count as success. The four YAML actions are ``search``, ``download``,
``infer`` and ``export``. Other behaviour is added from Python by giving a :class:`Step` a function.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# keyword arguments of TiledInference that a YAML ``infer`` step may set
_TILED_KEYS = ("chip_size", "overlap", "blend_mode", "batch_tiles", "activation", "tta", "half_precision")


class StepError(RuntimeError):
    """A step cannot do what it was asked to. Deterministic, so the orchestrator does not retry it."""


@dataclass
class Step:
    name: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    retry_on_fail: int = 3
    timeout_s: int = 3600
    _fn: Callable | None = field(default=None, repr=False)

    def run(self, context: dict) -> dict:
        """Execute this step (only steps given a function can run; there is no silent no-op)."""
        if self._fn:
            return self._fn(context, **self.params)
        raise StepError(
            f"step {self.name!r} has action {self.action!r}, which has no built-in implementation. "
            f"YAML supports search, download, infer and export; anything else needs "
            f"Step(..., _fn=your_function) from Python.")


@dataclass
class SearchStep(Step):
    """``client.search(**params)``; an empty result is a failure (every later step needs scenes)."""
    action: str = "search"

    def run(self, context: dict) -> list:
        client = context.get("client")
        if client is None:
            raise StepError("search needs a client in the run context (PyGeoVision())")
        try:
            results = client.search(**self.params)
        except TypeError as exc:     # bbox / date_range are required by client.search
            raise StepError(f"search parameters are incomplete: {exc}. Set them in the YAML or with "
                            f"--set {self.name}.bbox=... / --set {self.name}.date_range=...") from exc
        if not results:
            raise StepError("search returned no scenes")
        return list(results)


@dataclass
class DownloadStep(Step):
    """``client.download(search_results, **params)``; fails unless at least one scene arrived."""
    action: str = "download"

    def run(self, context: dict) -> list:
        client = context.get("client")
        scenes = context.get("search_results")
        if client is None:
            raise StepError("download needs a client in the run context (PyGeoVision())")
        if not scenes:
            raise StepError("download has no scenes: it must depend on a successful search step")
        results = list(client.download(scenes, **self.params))
        ok = [r for r in results if getattr(r, "success", True) and getattr(r, "path", None)]
        if not ok:
            errors = "; ".join(str(getattr(r, "error", "")) for r in results if getattr(r, "error", ""))
            raise StepError(f"no scene was downloaded{': ' + errors if errors else ''}")
        return ok


@dataclass
class InferStep(Step):
    """Tiled inference of *your trained* model over each downloaded scene.

    Params: ``model`` (registry name), ``checkpoint`` (required), ``num_classes``, ``in_channels``
    (default: the scene's band count) and any of ``chip_size / overlap / blend_mode / ...``. A built
    model can instead be passed as ``context["model"]``. Without trained weights the step fails: an
    untrained network's output looks like a result and is not one.
    """
    action: str = "infer"

    def run(self, context: dict) -> list:
        downloads = context.get("downloads")
        if not downloads:
            raise StepError("infer has no scenes: it must depend on a successful download step")
        params = dict(self.params)
        model = context.get("model")
        name, ckpt = params.pop("model", None), params.pop("checkpoint", None)
        num_classes = int(params.pop("num_classes", 2))
        in_channels = params.pop("in_channels", None)
        out_dir = params.pop("output_dir", None)
        tiled = {k: params.pop(k) for k in _TILED_KEYS if k in params}
        if params:
            raise StepError(f"infer got unknown parameter(s) {sorted(params)}")
        if model is None:
            if not name or not ckpt:
                raise StepError(
                    "infer needs your trained model: set 'model' (registry name) and 'checkpoint' "
                    "(a .pth trained with GeoTrainer on your area; PyGeoVision ships architectures, "
                    f"not trained weights) -- e.g. --set {self.name}.checkpoint=best.pth")
            if not Path(ckpt).is_file():
                raise StepError(f"checkpoint not found: {ckpt}")

        from pygeovision.inference.tiled import TiledInference
        results = []
        for dl in downloads:
            path = Path(dl.path if hasattr(dl, "path") else dl)
            net = model
            if net is None:
                import rasterio

                from pygeovision.ai.models.hub import ModelHub
                if in_channels is None:
                    with rasterio.open(path) as src:
                        bands = src.count
                else:
                    bands = int(in_channels)
                net = ModelHub().load(name, checkpoint=str(ckpt), num_classes=num_classes,
                                      in_channels=bands, device="cpu")
            out = (Path(out_dir) if out_dir else path.parent) / f"{path.stem}_pred.tif"
            res = TiledInference(model=net, num_classes=num_classes, **tiled).infer(str(path), str(out))
            if not res.get("success"):
                raise StepError(f"inference on {path.name} failed: {res.get('error')}")
            results.append(res)
        return results


@dataclass
class ExportStep(Step):
    """Vectorise each prediction raster and write it as ``format`` (geojson | gpkg | shp | kml).

    Params: ``format`` (default geojson), ``output_dir`` (default ./output/), ``target_class``
    (default: every class), ``min_area_m2``, ``simplify_tolerance``.
    """
    action: str = "export"

    def run(self, context: dict) -> list:
        predictions = context.get("predictions")
        if not predictions:
            raise StepError("export has no predictions: it must depend on a successful infer step")
        fmt = str(self.params.get("format", "geojson")).lower()
        ext = {"geojson": "geojson", "gpkg": "gpkg", "shp": "shp", "kml": "kml"}.get(fmt)
        if ext is None:
            raise StepError(f"format must be geojson, gpkg, shp or kml, got {fmt!r}")
        out_dir = Path(self.params.get("output_dir", "./output/"))
        out_dir.mkdir(parents=True, exist_ok=True)
        from pygeovision.data.postprocess import PostProcessor
        post = PostProcessor()
        written = []
        for pred in predictions:
            src = Path(pred["output_path"] if isinstance(pred, dict) else pred)
            geojson = out_dir / f"{src.stem}.geojson"
            post.vectorise(str(src), str(geojson),
                           target_class=self.params.get("target_class"),
                           min_area_m2=float(self.params.get("min_area_m2", 0.0)),
                           simplify_tolerance=float(self.params.get("simplify_tolerance", 0.0)))
            if ext != "geojson":
                final = out_dir / f"{src.stem}.{ext}"
                post.export(str(geojson), str(final), fmt=ext)
                geojson.unlink(missing_ok=True)
                geojson = final
            written.append(str(geojson))
        return written
