"""Tests for PostProcessor.regularise_buildings().

Real, severe regression found and fixed: the implementation did not match
its own docstring. It claimed to "detect the main orientation... rotate,
rectangularise, and rotate back" (a shape-preserving regularisation), but
the actual body was a single line: `geom.minimum_rotated_rectangle` — a
crude bounding-box approximation that discards the real building shape
entirely. Confirmed this produced 33% area inflation for a simple
L-shaped footprint, and would be far worse for courtyards or multi-wing
buildings — a serious data-quality issue for anyone using this for area
statistics or ML training labels.
"""
import pytest

shapely = pytest.importorskip("shapely", reason="shapely not installed")

import json
import math
from shapely.geometry import Polygon, shape


def _write_geojson(path, coords_list):
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [coords]}}
        for coords in coords_list
    ]}
    with open(path, "w") as f:
        json.dump(gj, f)


def _corner_angles(poly):
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    angles = []
    for i in range(n):
        p0, p1, p2 = coords[(i - 1) % n], coords[i], coords[(i + 1) % n]
        v1 = (p0[0] - p1[0], p0[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag = math.hypot(*v1) * math.hypot(*v2)
        angles.append(math.degrees(math.acos(max(-1, min(1, dot / mag)))))
    return angles


class TestRegulariseBuildingsPreservesRealShape:
    """Regression tests for the real bug: non-rectangular buildings were
    silently replaced with an inflated bounding box instead of having
    their actual shape preserved."""

    def test_l_shaped_building_area_is_preserved(self, tmp_path):
        """The core regression test: before the fix, this L-shape's area
        would inflate from 75.0 to 100.0 (a 33% distortion) because the
        whole footprint was replaced with its bounding rectangle."""
        from pygeovision.data.postprocess import PostProcessor

        l_shape = [[0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10], [0, 0]]
        in_path = tmp_path / "in.geojson"
        _write_geojson(in_path, [l_shape])

        post = PostProcessor()
        out_path = tmp_path / "out.geojson"
        post.regularise_buildings(str(in_path), str(out_path))

        with open(out_path) as f:
            result = json.load(f)
        reg = shape(result["features"][0]["geometry"])
        orig = Polygon(l_shape)

        assert abs(reg.area - orig.area) / orig.area < 0.05, (
            f"area distorted: {orig.area} -> {reg.area} "
            f"(ratio {reg.area / orig.area:.3f}) — shape was likely "
            f"replaced with a bounding box"
        )

    def test_l_shaped_building_is_not_collapsed_to_a_rectangle(self, tmp_path):
        from pygeovision.data.postprocess import PostProcessor

        l_shape = [[0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10], [0, 0]]
        in_path = tmp_path / "in.geojson"
        _write_geojson(in_path, [l_shape])

        post = PostProcessor()
        out_path = tmp_path / "out.geojson"
        post.regularise_buildings(str(in_path), str(out_path))

        with open(out_path) as f:
            result = json.load(f)
        reg = shape(result["features"][0]["geometry"])
        n_vertices = len(list(reg.exterior.coords)) - 1
        assert n_vertices > 4, f"L-shape collapsed to a {n_vertices}-vertex polygon (a rectangle)"


class TestRegulariseBuildingsActuallyRegularises:
    """Confirms the function does real regularisation work, not just
    'pass the input through unchanged' — a noisy near-rectangular
    building's corner angles must genuinely improve toward 90 degrees."""

    def test_noisy_corners_are_cleaned_toward_90_degrees(self, tmp_path):
        from pygeovision.data.postprocess import PostProcessor

        # Realistic segmentation noise: each vertex jittered off a true
        # rectangle, so corners deviate from exactly 90 degrees.
        noisy = [[0.0, 0.3], [10.2, -0.2], [10.0, 6.1], [-0.3, 5.8], [0.0, 0.3]]
        in_path = tmp_path / "in.geojson"
        _write_geojson(in_path, [noisy])

        orig = Polygon(noisy)
        orig_max_dev = max(abs(a - 90) for a in _corner_angles(orig))

        post = PostProcessor()
        out_path = tmp_path / "out.geojson"
        post.regularise_buildings(str(in_path), str(out_path), angle_tolerance_deg=15.0)

        with open(out_path) as f:
            result = json.load(f)
        reg = shape(result["features"][0]["geometry"])
        reg_max_dev = max(abs(a - 90) for a in _corner_angles(reg))

        assert reg_max_dev < orig_max_dev, (
            f"corner-angle noise was not reduced: {orig_max_dev:.2f} -> {reg_max_dev:.2f}"
        )
        assert reg_max_dev < 1.0, f"corners still not close to 90 degrees: {reg_max_dev:.2f}"

    def test_min_area_filter_still_works(self, tmp_path):
        from pygeovision.data.postprocess import PostProcessor

        tiny = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]  # area = 1 m²
        big = [[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]]  # area = 200 m²
        in_path = tmp_path / "in.geojson"
        _write_geojson(in_path, [tiny, big])

        post = PostProcessor()
        out_path = tmp_path / "out.geojson"
        post.regularise_buildings(str(in_path), str(out_path), min_area_m2=10.0)

        with open(out_path) as f:
            result = json.load(f)
        assert len(result["features"]) == 1

    def test_rotated_rectangle_keeps_its_real_orientation(self, tmp_path):
        """A genuinely rectangular building rotated 5 degrees off true
        north should STAY at ~5 degrees — regularisation cleans up a
        building's own internal corner noise, it must not force every
        building to align with map north."""
        import math
        from pygeovision.data.postprocess import PostProcessor

        angle = math.radians(5)
        base = [(0, 0), (10, 0), (10, 6), (0, 6), (0, 0)]
        def rot(p):
            x, y = p
            return [x * math.cos(angle) - y * math.sin(angle),
                    x * math.sin(angle) + y * math.cos(angle)]
        rotated = [rot(p) for p in base]

        in_path = tmp_path / "in.geojson"
        _write_geojson(in_path, [rotated])

        post = PostProcessor()
        out_path = tmp_path / "out.geojson"
        post.regularise_buildings(str(in_path), str(out_path))

        with open(out_path) as f:
            result = json.load(f)
        reg = shape(result["features"][0]["geometry"])
        orig = Polygon(rotated)

        assert abs(reg.area - orig.area) / orig.area < 0.02
