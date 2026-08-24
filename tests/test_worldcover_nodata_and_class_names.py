"""Tests for two real bugs found in a real production run of the
land_cover CLI channel, after the real ESA WorldCover tile downloaded
successfully for the first time (previously blocked in this sandbox by
a network restriction, so this exact success path had never been
exercised before).

Bug 1 (the crash): _compute_class_distribution's default
nodata_value=255 doesn't match ESA WorldCover's real convention
(0=nodata, real classes are 10-100) -- so genuine nodata=0 pixels near
tile edges after merge/clip were never filtered out of valid_mask, then
looked up in active_classes, a dict that deliberately excludes key 0
(matching WorldCover's real nodata semantics) -- raising
KeyError(np.uint8(0)), reproducing the exact real production error
message.

Bug 2 (found while testing the fix for bug 1): class_id < len(class_names)
is only a valid check for a LIST (a positional index bound by the
list's length) -- but every real caller passes a DICT, where this
length check is meaningless (dict keys aren't bounded by the dict's
size). Real class values (10-100, or even Dynamic World's 0-8) are
essentially never "< len(dict)" for a small dict, so every real class
silently fell through to str(class_id) instead of its actual name --
producing stats keyed '10'/'50' instead of 'tree_cover'/'built_up'.
"""
import pytest

import numpy as np


def _make_labeler():
    from pygeovision.ai.labeling.base_labeler import BaseLabeler

    class FakeLabeler(BaseLabeler):
        name = "fake"
        supported_tasks = ["land_cover"]

        def label_tile(self, *a, **kw):
            pass

    return FakeLabeler.__new__(FakeLabeler)


class TestWorldCoverNodataDoesNotCrash:
    def test_real_worldcover_scenario_no_longer_raises_keyerror(self):
        """Reproduces the exact real scenario: a mask with genuine
        nodata=0 pixels (tile-edge border) mixed with real WorldCover
        classes, using active_classes built exactly as esa_worldcover.py
        builds it (excluding key 0)."""
        labeler = _make_labeler()
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:5, :] = 0     # genuine nodata border
        mask[5:15, :] = 10  # tree_cover
        mask[15:, :] = 50   # built_up
        active_classes = {10: "tree_cover", 50: "built_up"}  # real 0-exclusion

        stats = labeler._compute_class_distribution(mask, active_classes, nodata_value=0)
        assert stats  # did not raise, produced real output

    def test_wrong_nodata_value_incorrectly_counts_nodata_as_a_class(self):
        """With the wrong (old default) nodata_value, real nodata=0
        pixels are no longer filtered out -- they don't crash (the
        dict.get() fix for the second bug prevents that as a side
        effect), but they get semantically miscounted as if they were a
        real class, which is itself a real correctness issue this
        confirms the right nodata_value genuinely fixes."""
        labeler = _make_labeler()
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:5, :] = 0
        mask[5:, :] = 10
        active_classes = {10: "tree_cover"}

        stats_wrong = labeler._compute_class_distribution(mask, active_classes)  # default 255
        assert "0" in stats_wrong, "nodata pixels should show up as a spurious class with the wrong nodata_value"

        stats_right = labeler._compute_class_distribution(mask, active_classes, nodata_value=0)
        assert "0" not in stats_right
        assert set(stats_right.keys()) == {"tree_cover"}


class TestClassDistributionUsesRealNames:
    def test_dict_class_names_produce_real_semantic_keys_not_raw_numbers(self):
        labeler = _make_labeler()
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:5, :] = 0
        mask[5:15, :] = 10
        mask[15:, :] = 50
        active_classes = {10: "tree_cover", 50: "built_up"}

        stats = labeler._compute_class_distribution(mask, active_classes, nodata_value=0)
        assert set(stats.keys()) == {"tree_cover", "built_up"}

    def test_fractions_are_correct_not_just_the_keys(self):
        labeler = _make_labeler()
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:5, :] = 0     # 5 rows nodata, excluded
        mask[5:15, :] = 10  # 10 rows tree_cover
        mask[15:, :] = 50   # 5 rows built_up
        active_classes = {10: "tree_cover", 50: "built_up"}

        stats = labeler._compute_class_distribution(mask, active_classes, nodata_value=0)
        assert abs(stats["tree_cover"] - 10 / 15) < 0.01
        assert abs(stats["built_up"] - 5 / 15) < 0.01

    def test_list_style_class_names_still_works_no_regression(self):
        """Backward compatibility: any caller passing a real list
        (positional class names) must still work correctly."""
        labeler = _make_labeler()
        mask = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
        list_names = ["background", "water", "vegetation"]

        stats = labeler._compute_class_distribution(mask, list_names, nodata_value=255)
        assert set(stats.keys()) == {"background", "water", "vegetation"}

    def test_dynamic_world_style_dict_with_zero_as_a_real_class_works(self):
        """Dynamic World's real class scheme uses 0=water as a genuine
        class (not nodata) -- confirms the fix handles this correctly,
        distinct from WorldCover's different nodata convention."""
        labeler = _make_labeler()
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 0  # water -- a REAL class here, not nodata
        mask[5:, :] = 1  # trees
        dw_classes = {0: "water", 1: "trees"}

        stats = labeler._compute_class_distribution(mask, dw_classes, nodata_value=255)
        assert set(stats.keys()) == {"water", "trees"}
        assert abs(stats["water"] - 0.5) < 0.01
