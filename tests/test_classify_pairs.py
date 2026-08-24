"""
Validates DataValidator.classify_pairs() against a hand-constructed
network where the correct classification is known by direct
inspection, not just asserted.
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

validate_mod = _load("pygeofetch.insar.validate", "/home/claude/work/pygeofetch_current/pygeofetch/insar/validate.py")
DataValidator = validate_mod.DataValidator

import numpy as np
from dataclasses import dataclass


@dataclass
class FakePair:
    reference_date: str
    secondary_date: str
    coherence: np.ndarray


def make_pair(d1, d2, coh_value, shape=(4, 4)):
    return FakePair(d1, d2, np.full(shape, coh_value, dtype=np.float32))


def test_bridge_correctly_identified():
    print("=== 1. A real bridge (sole connection between two clusters) is classified as bridge_pairs, not excluded ===")
    # Two well-connected clusters {A,B,C} and {D,E,F}, joined by exactly
    # ONE low-coherence pair (C-D). Removing C-D would genuinely split
    # the network -- the textbook definition of a graph bridge.
    dates = ["A", "B", "C", "D", "E", "F"]
    pairs = [
        make_pair("A", "B", 0.8),
        make_pair("B", "C", 0.8),
        make_pair("D", "E", 0.8),
        make_pair("E", "F", 0.8),
        make_pair("C", "D", 0.15),  # the only link between the two clusters -- a real bridge
    ]
    result = DataValidator.classify_pairs(pairs, dates, coherence_threshold=0.3)
    print(f"  {result.summary()}")
    bridge_dates = [(p.reference_date, p.secondary_date) for p in result.bridge_pairs]
    assert ("C", "D") in bridge_dates, "the sole cross-cluster link must be classified as a bridge, not excluded"
    assert len(result.excluded_pairs) == 0, "nothing should be excluded here -- the only weak pair is genuinely necessary"
    assert len(result.good_pairs) == 4
    print("  PASS\n")


def test_redundant_weak_pair_correctly_excluded():
    print("=== 2. A weak pair that ISN'T needed for connectivity is excluded, not kept as a bridge ===")
    # Same two clusters, but now C-D AND a second, redundant low-coherence
    # link B-E also connects them. Neither is individually a bridge, since
    # removing one still leaves the other connecting the clusters.
    dates = ["A", "B", "C", "D", "E", "F"]
    pairs = [
        make_pair("A", "B", 0.8),
        make_pair("B", "C", 0.8),
        make_pair("D", "E", 0.8),
        make_pair("E", "F", 0.8),
        make_pair("C", "D", 0.15),
        make_pair("B", "E", 0.15),  # redundant weak link -- removing it alone doesn't disconnect anything
    ]
    result = DataValidator.classify_pairs(pairs, dates, coherence_threshold=0.3)
    print(f"  {result.summary()}")
    # With two weak links between the same two clusters, NEITHER is
    # individually a bridge -- removing either one still leaves the
    # other connecting them, by the exact graph-theoretic definition.
    assert len(result.bridge_pairs) == 0, "with two redundant weak links, neither is individually a bridge"
    assert len(result.excluded_pairs) == 2
    print("  PASS -- confirms classify_pairs uses the real definition (redundant weak pairs are NOT bridges), not a heuristic\n")


def test_good_pair_never_reclassified_even_if_also_a_bridge():
    print("=== 3. A HIGH-coherence pair that happens to also be structurally critical stays 'good', not 'bridge' ===")
    dates = ["A", "B", "C"]
    pairs = [
        make_pair("A", "B", 0.8),  # this is the only link -- structurally a bridge -- but coherence is high
        make_pair("B", "C", 0.8),
    ]
    result = DataValidator.classify_pairs(pairs, dates, coherence_threshold=0.3)
    print(f"  {result.summary()}")
    assert len(result.good_pairs) == 2
    assert len(result.bridge_pairs) == 0, "high-coherence pairs are always 'good' regardless of bridge status"
    print("  PASS\n")


def test_every_pair_classified_exactly_once():
    print("=== 4. Every input pair appears in exactly one of the three output lists ===")
    dates = ["A", "B", "C", "D"]
    pairs = [
        make_pair("A", "B", 0.9),
        make_pair("B", "C", 0.1),
        make_pair("C", "D", 0.9),
        make_pair("A", "D", 0.1),
    ]
    result = DataValidator.classify_pairs(pairs, dates, coherence_threshold=0.3)
    total_out = len(result.good_pairs) + len(result.bridge_pairs) + len(result.excluded_pairs)
    assert total_out == len(pairs), f"expected {len(pairs)} pairs total across all three lists, got {total_out}"
    print("  PASS\n")


if __name__ == "__main__":
    test_bridge_correctly_identified()
    test_redundant_weak_pair_correctly_excluded()
    test_good_pair_never_reclassified_even_if_also_a_bridge()
    test_every_pair_classified_exactly_once()
    print("ALL TESTS PASSED")
