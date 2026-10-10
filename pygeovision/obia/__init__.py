"""Object-based image analysis (OBIA) for PyGeoVision.

Typical workflow::

    from pygeovision import obia

    scene = obia.make_demo_scene()                       # or obia.load_raster("scene.tif")
    seg   = obia.segment(scene.raster, "multires", scale=20)
    feats = obia.extract_features(seg.labels, scene.raster,
                                  extra_layers=obia.spectral_layers(scene.raster))
    rules = obia.RuleBasedClassifier([("water", "mean_ndwi > 0.3"), ...])
    result = rules.classify(feats)
    report = obia.assess_classification(result, seg.labels, scene.reference, "class",
                                        scene.raster.transform)

See ``docs/obia`` for the full tutorial sequence.
"""
from .accuracy import (assess_classification, confusion_report, pixel_report,
                       segmentation_quality)
from .classify import (ClassificationResult, FuzzyClassifier, Gaussian, LargerThan,
                       Membership, ObjectClassifier, RuleBasedClassifier, SmallerThan,
                       Triangle, Trapezoid, samples_from_geometries)
from .demo import CLASS_NAMES, DemoScene, make_demo_scene
from .features import (class_context, extract_features, glcm_texture, parent_map,
                       spectral_layers, sub_object_fraction, super_object_features)
from .io import RasterData, load_raster, save_raster
from .merging import RegionMerger
from .scale import scale_diagnostics, scale_vs_reference
from .segmentation import (METHODS, SegmentationResult, from_labels, segment,
                           segment_levels)
from .vector import labels_to_gdf, rasterize_geometries

__all__ = [n for n in dir() if not n.startswith("_")]
