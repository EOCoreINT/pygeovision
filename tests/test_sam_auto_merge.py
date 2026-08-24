"""Tests for SAMAutoLabeler's real merge_overlapping fix.

Real bug found and fixed: merge_overlapping was accepted, documented,
and defaulted to True, but never referenced anywhere in auto_label()'s
actual logic. Since the function processes an image in overlapping
tiles (chip_size/overlap), any real object spanning the overlap zone
between two adjacent tiles gets detected independently by both tiles'
SAM passes -- without merging, this produces duplicate, overlapping
mask instances that corrupt count/area statistics near tile boundaries.
"""
import pytest

import numpy as np


def _make_labeler():
    from pygeovision.labeling.sam_auto import SAMAutoLabeler
    return SAMAutoLabeler.__new__(SAMAutoLabeler)


class TestMergeOverlappingMasks:
    def test_duplicate_masks_from_tile_overlap_are_merged(self):
        """The core regression test: two masks representing the same
        real object (as independently detected by two adjacent,
        overlapping tiles) must merge into one."""
        labeler = _make_labeler()
        H, W = 200, 200
        mask_a = np.zeros((H, W), dtype=bool)
        mask_a[50:100, 50:100] = True
        mask_b = np.zeros((H, W), dtype=bool)
        mask_b[52:102, 52:102] = True  # same object, slightly shifted detection

        masks = [
            {"full_mask": mask_a, "bbox_full": (50, 50, 100, 100),
             "predicted_iou": 0.9, "stability_score": 0.95},
            {"full_mask": mask_b, "bbox_full": (52, 52, 102, 102),
             "predicted_iou": 0.85, "stability_score": 0.9},
        ]
        merged = labeler._merge_overlapping_masks(masks)
        assert len(merged) == 1

    def test_merged_mask_is_the_real_union_not_just_the_first_one_kept(self):
        labeler = _make_labeler()
        H, W = 200, 200
        mask_a = np.zeros((H, W), dtype=bool)
        mask_a[50:100, 50:100] = True  # 2500 px
        mask_b = np.zeros((H, W), dtype=bool)
        mask_b[52:102, 52:102] = True  # 2500 px, offset by 2

        masks = [
            {"full_mask": mask_a, "bbox_full": (50, 50, 100, 100), "predicted_iou": 0.9, "stability_score": 0.95},
            {"full_mask": mask_b, "bbox_full": (52, 52, 102, 102), "predicted_iou": 0.85, "stability_score": 0.9},
        ]
        merged = labeler._merge_overlapping_masks(masks)
        assert merged[0]["full_mask"].sum() > mask_a.sum(), "merge should be the union, not just the first mask"

    def test_genuinely_distinct_objects_are_not_merged(self):
        """A distant, genuinely separate object must not be incorrectly
        merged just because it's in the same mask list."""
        labeler = _make_labeler()
        H, W = 200, 200
        mask_a = np.zeros((H, W), dtype=bool)
        mask_a[50:100, 50:100] = True
        mask_c = np.zeros((H, W), dtype=bool)
        mask_c[150:170, 150:170] = True  # far away, non-overlapping

        masks = [
            {"full_mask": mask_a, "bbox_full": (50, 50, 100, 100), "predicted_iou": 0.9, "stability_score": 0.95},
            {"full_mask": mask_c, "bbox_full": (150, 150, 170, 170), "predicted_iou": 0.8, "stability_score": 0.88},
        ]
        merged = labeler._merge_overlapping_masks(masks)
        assert len(merged) == 2

    def test_merged_quality_scores_take_the_max_not_the_first(self):
        labeler = _make_labeler()
        H, W = 200, 200
        mask_a = np.zeros((H, W), dtype=bool)
        mask_a[50:100, 50:100] = True
        mask_b = np.zeros((H, W), dtype=bool)
        mask_b[52:102, 52:102] = True

        masks = [
            {"full_mask": mask_a, "bbox_full": (50, 50, 100, 100), "predicted_iou": 0.5, "stability_score": 0.6},
            {"full_mask": mask_b, "bbox_full": (52, 52, 102, 102), "predicted_iou": 0.95, "stability_score": 0.97},
        ]
        merged = labeler._merge_overlapping_masks(masks)
        assert merged[0]["predicted_iou"] == 0.95
        assert merged[0]["stability_score"] == 0.97

    def test_low_iou_masks_with_touching_bboxes_are_not_merged(self):
        """Bounding boxes touching/overlapping slightly isn't enough on
        its own -- the real mask IoU must clear the threshold."""
        labeler = _make_labeler()
        H, W = 200, 200
        mask_a = np.zeros((H, W), dtype=bool)
        mask_a[50:100, 50:100] = True
        mask_b = np.zeros((H, W), dtype=bool)
        mask_b[95:145, 95:145] = True  # bboxes overlap slightly, but real IoU is low

        masks = [
            {"full_mask": mask_a, "bbox_full": (50, 50, 100, 100), "predicted_iou": 0.9, "stability_score": 0.9},
            {"full_mask": mask_b, "bbox_full": (95, 95, 145, 145), "predicted_iou": 0.9, "stability_score": 0.9},
        ]
        merged = labeler._merge_overlapping_masks(masks, iou_threshold=0.3)
        assert len(merged) == 2

    def test_single_mask_returns_unchanged(self):
        labeler = _make_labeler()
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        masks = [{"full_mask": mask, "bbox_full": (10, 10, 20, 20)}]
        result = labeler._merge_overlapping_masks(masks)
        assert result == masks

    def test_empty_list_returns_unchanged(self):
        labeler = _make_labeler()
        assert labeler._merge_overlapping_masks([]) == []
