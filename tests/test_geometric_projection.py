"""
Validates range_offset_to_vertical_displacement -- specifically that
it's a real, consistent composition with the existing, already-cited
los_to_vertical_displacement, not a parallel/duplicated formula.
"""
import sys
sys.path.insert(0, "/home/claude/work/pygeofetch_current")
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

geo_mod = _load("pygeofetch.insar.geolocation", "/home/claude/work/pygeofetch_current/pygeofetch/insar/geolocation.py")

import numpy as np


def test_consistent_with_existing_los_function_at_matching_input():
    print("=== 1. range_offset_to_vertical_displacement, fed an equivalent LOS distance, matches los_to_vertical_displacement exactly ===")
    range_offset_px = 3.5
    pixel_spacing_m = 2.3
    incidence_deg = 39.0

    result = geo_mod.range_offset_to_vertical_displacement(range_offset_px, pixel_spacing_m, incidence_deg)

    los_equivalent_m = range_offset_px * pixel_spacing_m
    expected = geo_mod.los_to_vertical_displacement(los_equivalent_m, incidence_deg)

    print(f"  range_offset_to_vertical_displacement: {result:.6f} m")
    print(f"  los_to_vertical_displacement (direct):  {expected:.6f} m")
    assert result == expected, "must be an EXACT, direct composition, not a re-derived approximation"
    print("  PASS -- confirms this is a real reuse of the existing function, not a parallel formula\n")


def test_zero_offset_gives_zero_displacement():
    print("=== 2. Zero range offset gives exactly zero vertical displacement (a real sanity floor) ===")
    result = geo_mod.range_offset_to_vertical_displacement(0.0, pixel_spacing_m=2.3, incidence_angle_deg=39.0)
    assert result == 0.0
    print("  PASS\n")


def test_array_input_matches_elementwise():
    print("=== 3. Real array input matches elementwise scalar computation ===")
    offsets = np.array([0.0, 1.0, -2.5, 5.0])
    pixel_spacing_m = 2.3
    incidence_deg = 39.0
    result = geo_mod.range_offset_to_vertical_displacement(offsets, pixel_spacing_m, incidence_deg)
    for i, off in enumerate(offsets):
        expected = geo_mod.range_offset_to_vertical_displacement(float(off), pixel_spacing_m, incidence_deg)
        assert abs(result[i] - expected) < 1e-12
    print("  PASS\n")


if __name__ == "__main__":
    test_consistent_with_existing_los_function_at_matching_input()
    test_zero_offset_gives_zero_displacement()
    test_array_input_matches_elementwise()
    print("ALL TESTS PASSED")
