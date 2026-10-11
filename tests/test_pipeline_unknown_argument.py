"""A misspelt pipeline argument must be reported, not silently ignored."""
import logging

from pygeovision import PyGeoVision


class _P:
    name = "demo"

    def run(self, bbox, output_dir=".", date=None, ndwi_threshold=0.3, **kwargs):
        return None


def test_misspelt_argument_is_reported_with_suggestion(caplog):
    with caplog.at_level(logging.WARNING, logger="pygeovision"):
        PyGeoVision._warn_unknown_pipeline_arguments(_P(), {"ndwi_threshhold": 0.2, "providers": ["x"]})
    text = caplog.text
    assert "ndwi_threshhold" in text and "ndwi_threshold" in text
    assert "providers" not in text.split("Arguments:")[0]


def test_known_arguments_are_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="pygeovision"):
        PyGeoVision._warn_unknown_pipeline_arguments(_P(), {"date": "2024-06", "providers": ["a"], "resolution_m": 60})
    assert caplog.text == ""
