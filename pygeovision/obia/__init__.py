"""Object-based image analysis (OBIA) for PyGeoVision.

Typical workflow::

    import pygeovision as pgv
    from pygeovision import obia

    client = pgv.PyGeoVision()
    rd = obia.fetch_scene(client, (-1.70, 6.18, -1.655, 6.225), "2024-01", "./download")   # real Sentinel-2
    seg = obia.segment(rd, "multires", scale=16)
    feats = obia.extract_features(seg.labels, rd, extra_layers=obia.spectral_layers(rd))
    rules = obia.RuleBasedClassifier([("water", "mean_ndwi > 0.3"), ("vegetation", "mean_ndvi > 0.5")],
                                     default="other")
    result = rules.classify(feats)

Any GeoTIFF works too: ``rd = obia.load_raster("scene.tif")``.  Cloud and shadow pixels (NaN) become label 0
and belong to no object.  The same chain is available from the shell as ``pygeovision obia fetch / segment /
features / classify``, and ``client.obia`` offers the download helpers.

See ``docs/obia`` for the full tutorial sequence.
"""
from .accuracy import (assess_classification, confusion_report, pixel_report,
                       segmentation_quality)
from .classify import (ClassificationResult, FuzzyClassifier, Gaussian, LargerThan,
                       Membership, ObjectClassifier, RuleBasedClassifier, SmallerThan,
                       Triangle, Trapezoid, samples_from_geometries, samples_from_raster)
from .features import (class_context, extract_features, glcm_texture, parent_map,
                       spectral_layers, sub_object_fraction, super_object_features)
from .io import RasterData, load_raster, save_raster, valid_mask
from .merging import RegionMerger
from .scale import scale_diagnostics, scale_vs_reference
from .scene import fetch_osm_polygons, fetch_scene, fetch_worldcover
from .segmentation import (METHODS, SegmentationResult, from_labels, segment,
                           segment_levels)
from .vector import labels_to_gdf, rasterize_geometries

__all__ = [n for n in dir() if not n.startswith("_")]
