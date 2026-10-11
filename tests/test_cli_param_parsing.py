"""``--param KEY=VALUE`` parsing: JSON where it is JSON, numbers for a comma list, a string otherwise."""
import click
import pytest

from pygeovision.cli.main import _parse_param


@pytest.mark.parametrize("raw, expected", [
    ("ndwi_threshold=0.2", ("ndwi_threshold", 0.2)),
    ("use_landcover=false", ("use_landcover", False)),
    ("hand_thresholds_m=[5,10,15]", ("hand_thresholds_m", [5, 10, 15])),
    ("hand_thresholds_m=5,10,15", ("hand_thresholds_m", [5, 10, 15])),
    ("slope_thresholds_deg=7.5, 15", ("slope_thresholds_deg", [7.5, 15])),
    ("min_confidence=nominal", ("min_confidence", "nominal")),
    ("water_db=otsu", ("water_db", "otsu")),
    ("baseline_years=2019,2020,2021", ("baseline_years", [2019, 2020, 2021])),
    ("note=a,b", ("note", "a,b")),
    ("note=1,b", ("note", "1,b")),
    ("flag=true,false", ("flag", "true,false")),
])
def test_parse_param(raw, expected):
    assert _parse_param(raw) == expected


def test_parse_param_requires_equals():
    with pytest.raises(click.BadParameter):
        _parse_param("ndwi_threshold")
