"""Flood mapping with the pretrained ml4floods U-Net ``WF2_unetv2_bgriswirs`` on Sentinel-2.

``ml4floods_flood_mapping`` runs the released network of Portales-Julia et al. (2023) on the clear Sentinel-2
observations of a dry **reference window** and a **flood window**, and reports where water appeared. It is a
*pretrained-model* pipeline:

* the weights are not part of PyGeoVision (they are CC BY-NC 4.0 and PyGeoVision is MIT). Without ``checkpoint=`` they are
  downloaded once from Hugging Face into ``~/.cache/pygeovision/ml4floods`` (the licence notice is logged), and reused
  afterwards; or pass ``checkpoint=`` (the folder with ``model.pt``). The model is loaded, and the download done,
  *before* any imagery is fetched, so a missing network or a wrong file stops the run early;
* nothing about the network's accuracy is claimed here, and the report says that the input is not the product the
  network was trained on (see ``INPUT_NOTE``).

How it gets and combines the imagery
-------------------------------------
The imagery comes from the **standard path every pipeline uses** (``_search_and_download_pair``): pygeofetch streams only
the pixels of your area from Planetary Computer, the clearest scenes are mosaicked, the six bands are scaled to
reflectance and cloud, shadow and cirrus (Scene Classification Layer) are set to unknown. The network then classifies
each window's image (land, water, cloud). A pixel is **observed** in a window when it is clear: not masked by the Scene
Classification Layer and not predicted cloud, because the network can confuse cloud shadow with water. Pixels without
a clear observation are *unknown*, never dry.

Output ``flood_extent.tif``:

====  ================================================================================================
0     not flooded: clear in the flood window and not water
1     **flood water**: water in the flood window, clear land in the reference window
2     water in both windows (permanent or seasonal water, sea)
3     water in the flood window with **no clear reference view**: new flood or old water, cannot be told
255   no clear observation in the flood window
====  ================================================================================================

``stats['flooded_area_ha']`` is class 1; ``flooded_upper_ha`` adds class 3.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
import time
from typing import Any

import numpy as np

from pygeovision.ai.models import ml4floods as m4f
from pygeovision.ai.pipelines import PipelineResult
from pygeovision.ai.pipelines._raster import write_classes, write_float
from pygeovision.ai.pipelines.reference_products import _Reference
from pygeovision.data._dates import date_window
from pygeovision.data.reference import ReferenceDataError, make_grid

logger = logging.getLogger(__name__)

#: the six network bands as the standard imagery path names them (B2, B3, B4, B8, B11, B12)
NETWORK_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
FLOOD_CLASSES = {
    0: "not flooded",
    1: "flood water",
    2: "water in the reference window",
    3: "water in the flood window, no clear reference view",
}
MODEL_PIPELINE_NOTE = (
    f"Pretrained network {m4f.MODEL_ID} (ml4floods; {m4f.CITATION}). {m4f.LICENSE_NOTE} {m4f.INPUT_NOTE} "
    "Sentinel-2 L2A reflectance from the Microsoft Planetary Computer is used as its input; cloud, cloud shadow and "
    "cirrus are masked with the network's own cloud output and the product's Scene Classification Layer, so they "
    "are unknown, never dry."
)


class Ml4FloodsFloodPipeline(_Reference):
    """Flooded area from the pretrained ml4floods ``WF2_unetv2_bgriswirs`` network on Sentinel-2.

    Arguments: ``date_before`` (a dry reference window) and ``date_after`` (the flood window), each a day, a month or
    ``YYYY-MM-DD/YYYY-MM-DD``; ``checkpoint`` (folder with ``model.pt``; leave it out and the weights are downloaded once from Hugging Face into ``~/.cache/pygeovision/ml4floods``); ``th_water`` (default 0.5, the probability
    above which a pixel is water; the ml4floods tutorials use 0.6 to 0.7 to reduce false water); ``th_cloud`` and
    ``th_brightness`` (the cloud rule of the network's authors: cloud counts only where it is bright);
    ``max_cloud`` (scene cloud-cover limit in %, default 95: clouds are masked per pixel); ``use_scl`` (default True: also drop pixels the Sentinel-2 Scene Classification Layer flags as bad);
    ``min_patch_ha`` (default 0.1); ``device`` (``"auto"``, ``"cpu"``, ``"cuda"``); ``tile_size`` (default 512);
    ``max_pixels`` (default 25 million, about 50 km x 50 km at 10 m).
    """

    name = "ml4floods_flood_mapping"
    description = (
        "Flooded area from the pretrained ml4floods U-Net (WF2_unetv2_bgriswirs) on Sentinel-2; downloads the weights "
        "once (CC BY-NC 4.0, non-commercial)"
    )
    domain = "disaster"
    kind = "ready"
    tags = ["flood", "optical", "sentinel-2", "deep-learning", "pretrained", "ml4floods"]
    collection = "sentinel-2-l2a"
    native_res_m = 10.0

    # -- classifying one prepared image --------------------------------------------------------------------------------
    def _classify_image(
        self,
        path: Path,
        model: m4f.LoadedModel,
        opts: dict[str, Any],
        label: str,
        scene_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the network on one image made by the standard imagery path (``_search_and_download``).

        That path streams only the AOI from the provider, mosaics the clearest scenes, scales to reflectance (0-1) and
        sets cloud, shadow and cirrus (Scene Classification Layer) to NaN. This method turns it into the network's input
        (reflectance x 10000, NaN = unknown) and into per-pixel ``observed`` / ``water`` masks. ``scene_ids`` are the
        Sentinel-2 scenes the image was made from (for the report); the file name stands in when they are unknown.
        """
        import rasterio

        t0 = time.time()
        with rasterio.open(path) as src:
            if src.count != len(m4f.BANDS):
                raise ReferenceDataError(
                    f"{path.name} has {src.count} band(s), the network needs {len(m4f.BANDS)} ({', '.join(m4f.BANDS)})."
                )
            arr = src.read().astype("float32")
            crs, transform = src.crs, src.transform
            width, height = src.width, src.height  # the dataset is closed after this block
            profile = {
                "driver": "GTiff",
                "crs": crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        refl = arr * 10000.0
        # an exact zero means "no data" to the network; keep real (possibly tiny) values above it
        valid = np.all(np.isfinite(refl), axis=0) & np.any(refl > 0, axis=0)
        image = np.where(valid[None], np.maximum(refl, 1.0), np.nan).astype("float32")
        logger.info(
            "%s image %d x %d px, %.0f%% usable: running the network on %s",
            label,
            width,
            height,
            100 * valid.mean(),
            model.device,
        )
        classes, probs = m4f.predict_scene(
            model,
            image,
            tile_size=opts["tile_size"],
            th_water=opts["th_water"],
            th_cloud=opts["th_cloud"],
            th_brightness=opts["th_brightness"],
        )
        clear = valid & np.isin(classes, (1, 2))
        water = clear & (classes == 2)
        logger.info(
            "%s done in %.0f s: %.0f%% clear, %.1f%% of the clear pixels water",
            label,
            time.time() - t0,
            100 * clear.mean(),
            100 * water.sum() / max(int(clear.sum()), 1),
        )
        if (
            crs is not None and crs.is_geographic
        ):  # degrees: size of one pixel at the centre latitude
            lat = transform.f + transform.e * height / 2
            m_x = abs(transform.a) * 111_320.0 * float(np.cos(np.radians(lat)))
            m_y = abs(transform.e) * 110_574.0
        else:
            m_x, m_y = abs(transform.a), abs(transform.e)
        return {
            "observed": clear,
            "water": water,
            "p_water": np.where(clear, np.nan_to_num(probs[1]), np.nan).astype("float32"),
            "scenes": [
                {"id": sid, "date": _scene_date(sid), "clear_fraction": float(clear.mean())}
                for sid in (scene_ids or [path.stem])
            ],
            "skipped": [],
            "profile": profile,
            "px_ha": float(m_x * m_y / 10_000.0),
            "res": float((m_x * m_y) ** 0.5),
            "crs": str(crs),
        }

    # -- run ---------------------------------------------------------------------------------------------------------
    def run(  # type: ignore[override]
        self,
        bbox,
        output_dir="./output",
        date_before=None,
        date_after=None,
        checkpoint=None,
        th_water=0.5,
        th_cloud=0.5,
        th_brightness=m4f.BRIGHTNESS_THRESHOLD,
        max_cloud=95.0,
        use_scl=True,
        min_patch_ha=0.1,
        device="auto",
        tile_size=512,
        max_pixels=25_000_000,
        **kwargs,
    ) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if not date_before or not date_after:
                raise ReferenceDataError(
                    f"{self.name} needs date_before (a dry reference) and date_after (the flood), each a day, a "
                    "month or a range such as 2026-06-27/2026-07-06."
                )
            if not 0.0 <= float(th_water) <= 1.0 or not 0.0 <= float(th_cloud) <= 1.0:
                raise ReferenceDataError("th_water and th_cloud must be between 0 and 1.")
            if float(min_patch_ha) < 0:
                raise ReferenceDataError("min_patch_ha cannot be negative.")
            wb, wa = date_window(str(date_before)), date_window(str(date_after))
            if wb[1] >= wa[0]:
                raise ReferenceDataError(
                    f"the reference window {wb[0]}..{wb[1]} must end before the flood window {wa[0]}..{wa[1]} starts."
                )
            # the model first: a wrong or missing checkpoint must stop the run before any imagery is fetched
            try:
                if not checkpoint or str(checkpoint).lower() in ("auto", "hf", "huggingface"):
                    checkpoint = (
                        m4f.ensure_weights()
                    )  # downloaded once, then reused from the cache folder
                model = m4f.load_model(checkpoint, device)
            except m4f.ML4FloodsError as exc:
                raise ReferenceDataError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - network, disk, hub or torch errors all mean "no usable weights"
                raise ReferenceDataError(
                    f"could not fetch the weights from Hugging Face ({type(exc).__name__}: {str(exc)[:200]}). "
                    f"Download {m4f.HF_REPO}/{m4f.HF_SUBFOLDER} yourself and pass its folder as checkpoint=."
                ) from exc
            # size check only (nothing is fetched): the grid the bbox needs at the network's 10 m
            grid = make_grid(bbox, self.native_res_m, max_pixels=int(max_pixels))
            opts = {
                "th_water": float(th_water),
                "th_cloud": float(th_cloud),
                "th_brightness": float(th_brightness),
                "use_scl": bool(use_scl),
                "tile_size": int(tile_size),
            }
            notes: list[str] = []
            logger.info(
                "area %d x %d pixels at %g m (%.0f km2) on %s; reference window %s..%s, flood window %s..%s",
                grid.width,
                grid.height,
                self.native_res_m,
                grid.width * grid.height * grid.pixel_area_ha / 100.0,
                model.device,
                wb[0],
                wb[1],
                wa[0],
                wa[1],
            )
            if model.device == "cpu" and grid.width * grid.height > 4_000_000:
                logger.warning(
                    "running on the CPU over %.1f million pixels: expect several minutes per image. Try a smaller "
                    "bbox first.",
                    grid.width * grid.height / 1e6,
                )
            # The standard imagery path of every pipeline: streams only the AOI pixels from Planetary Computer /
            # Element 84, picks the clearest scenes, scales to reflectance, masks cloud with the Scene
            # Classification Layer and aligns both dates to one grid.
            before_path, after_path = self._search_and_download_pair(
                tuple(bbox),
                str(date_before),
                str(date_after),
                out / "imagery",
                cloud_cover_max=float(max_cloud),
                bands=NETWORK_BANDS,
                apply_scale=True,
                apply_cloud_mask=bool(use_scl),
            )
            # which scenes made each image, before the next step clears the log
            used = {e.get("image"): list(e.get("scenes_used", [])) for e in self._coverage_log}
            if after_path is None:
                raise ReferenceDataError(
                    f"no Sentinel-2 image of the flood window {wa[0]}..{wa[1]} could be prepared (no scene under "
                    f"{max_cloud:g}% cloud covering the area, or the download failed); see the log above."
                )
            after = self._classify_image(
                Path(after_path), model, opts, "flood window:", used.get("after")
            )
            if not after["observed"].any():
                raise ReferenceDataError(
                    "the flood-window image has no clear pixels over the area (all cloud, shadow or outside the scene)."
                )
            if before_path is not None:
                before = self._classify_image(
                    Path(before_path), model, opts, "reference window:", used.get("before")
                )
            else:
                before = {
                    "observed": np.zeros_like(after["observed"]),
                    "water": np.zeros_like(after["water"]),
                    "p_water": np.full(after["p_water"].shape, np.nan, dtype="float32"),
                    "scenes": [],
                    "skipped": [],
                }
                notes.append("no Sentinel-2 image of the reference window could be prepared")
            if not before["observed"].any():
                notes.append(
                    "no clear Sentinel-2 view in the reference window: new flood water could not be told from old "
                    "water (class 3)"
                )

            obs_a, wat_a = after["observed"], after["water"]
            obs_b, wat_b = before["observed"], before["water"]
            cls = np.zeros(obs_a.shape, dtype=np.uint8)
            cls[wat_a & obs_b & ~wat_b] = 1
            cls[wat_a & wat_b] = 2
            cls[wat_a & ~obs_b] = 3
            removed = 0
            px_ha = after["px_ha"]
            if min_patch_ha:
                from scipy import ndimage

                lab, n = ndimage.label(np.isin(cls, (1, 3)), structure=np.ones((3, 3), int))
                if n:
                    sizes = np.bincount(lab.ravel())
                    small = np.nonzero(sizes * px_ha < float(min_patch_ha))[0]
                    small = small[small != 0]
                    if small.size:
                        sm = np.isin(lab, small)
                        removed = int(sm.sum())
                        cls[sm] = 0

            prof = after["profile"]
            path = write_classes(
                out / "flood_extent.tif",
                cls,
                obs_a,
                prof,
                [f"{k}={v}" for k, v in FLOOD_CLASSES.items()] + ["255=no clear observation"],
            )
            write_float(
                out / "prob_water_after.tif",
                after["p_water"],
                prof,
                [
                    "network: mean probability of water over the clear observations of the flood window (NaN = none)"
                ],
            )

            def ha(mask: np.ndarray) -> float:
                return float(mask.sum() * px_ha)

            n_cells = float(obs_a.size)
            land = obs_a & ~(cls == 2)
            n_land = int(land.sum())
            stats: dict[str, Any] = {
                "flooded_area_ha": ha(cls == 1),
                "flooded_unresolved_ha": ha(cls == 3),
                "flooded_upper_ha": ha((cls == 1) | (cls == 3)),
                "permanent_water_area_ha": ha(cls == 2),
                "land_area_ha": ha(land),
                "flooded_fraction": float((cls == 1).sum() / n_land) if n_land else 0.0,
                "valid_fraction": float(obs_a.sum() / n_cells),
                "reference_observed_fraction": float(obs_b.sum() / n_cells),
                "removed_small_patch_ha": float(removed * px_ha),
                "model": m4f.MODEL_ID,
                "model_weights_sha256": model.weights_sha256,
                "model_device": model.device,
                "th_water": opts["th_water"],
                "th_cloud": opts["th_cloud"],
                "th_brightness": opts["th_brightness"],
                "used_scene_classification": opts["use_scl"],
                "input_product": "Sentinel-2 L2A surface reflectance (the network was trained on Level-1C)",
                "scenes_reference": len(before["scenes"]),
                "scenes_flood": len(after["scenes"]),
                "flood_dates": sorted({s["date"] for s in after["scenes"] if s["date"]}),
                "reference_dates": sorted({s["date"] for s in before["scenes"] if s["date"]}),
                "scenes": {"reference": before["scenes"], "flood": after["scenes"]},
                "min_patch_ha": float(min_patch_ha),
                "data_notes": "; ".join(notes) if notes else "",
                "resolution_m": after["res"],
                "model_citation": m4f.CITATION,
                "model_license": "CC BY-NC 4.0 (weights, non-commercial use only)",
                "product_note": MODEL_PIPELINE_NOTE,
            }

            def span(days: list[str], window: tuple[str, str]) -> str:
                days = days or list(window)
                return days[0] if days[0] == days[-1] else f"{days[0]} – {days[-1]}"

            meta = {
                "date_before": span(stats["reference_dates"], wb),
                "date_after": span(stats["flood_dates"], wa),
                "crs": after["crs"],
                "scene_ids": {
                    "reference": [s["id"] for s in before["scenes"]],
                    "flood": [s["id"] for s in after["scenes"]],
                },
            }
            return self._ok(path, stats, meta, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


def _scene_date(scene_id: str) -> str:
    """``YYYY-MM-DD`` from a Sentinel-2 scene id (``S2B_MSIL2A_20230613T...`` or ``S2A_32TQM_20230613_0_L2A``), else ''."""
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", scene_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


ML4FLOODS_PIPELINES: dict[str, type] = {Ml4FloodsFloodPipeline.name: Ml4FloodsFloodPipeline}