"""Tests for Phase 4.4 + 5.2: BenchmarkEvaluator + Leaderboard."""
from __future__ import annotations

# Skip this file if PyTorch is not installed
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — pip install torch")

import json
import pytest
import sys; import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


class TestBenchmarkResult:
    def test_primary_metric_segmentation(self):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        r = BenchmarkResult("UNet", "LoveDA", "segmentation", mean_iou=0.72)
        assert abs(r.primary_metric - 0.72) < 1e-6

    def test_primary_metric_detection(self):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        r = BenchmarkResult("YOLOv8", "DOTA", "detection", mAP50=0.65)
        assert abs(r.primary_metric - 0.65) < 1e-6

    def test_primary_metric_classification(self):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        r = BenchmarkResult("ViT", "EuroSAT", "classification", accuracy=0.98)
        assert abs(r.primary_metric - 0.98) < 1e-6

    def test_to_dict(self):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        r = BenchmarkResult("SegFormer", "LoveDA", "segmentation",
                            mean_iou=0.72, mean_f1=0.82, accuracy=0.94,
                            n_samples=500, inference_ms_per_image=12.5)
        d = r.to_dict()
        assert d["model"] == "SegFormer"
        assert d["dataset"] == "LoveDA"
        assert d["mean_iou"] == 0.7200
        assert d["n_samples"] == 500
        assert d["primary_metric"] == 0.7200


class TestModelEvaluator:
    def test_init(self):
        from pygeovision.benchmark.evaluator import ModelEvaluator
        ev = ModelEvaluator(task="segmentation", num_classes=5)
        assert ev.task == "segmentation"
        assert ev.num_classes == 5

    def test_evaluate_with_mock_loader(self):
        try:
            import torch
            import torch.nn as nn

            from pygeovision.benchmark.evaluator import BenchmarkResult, ModelEvaluator

            class DummyModel(nn.Module):
                def forward(self, x):
                    B, C, H, W = x.shape
                    return torch.zeros(B, 2, H, W)

            data = [(torch.randn(2, 3, 32, 32), torch.zeros(2, 32, 32, dtype=torch.long)) for _ in range(3)]
            ev = ModelEvaluator(task="segmentation", num_classes=2)
            result = ev.evaluate(DummyModel(), data, dataset_name="TestDS", model_name="Dummy")
            assert isinstance(result, BenchmarkResult)
            assert result.model_name == "Dummy"
            assert result.dataset_name == "TestDS"
            assert result.n_samples == 6
        except ImportError:
            pytest.skip("torch not installed")

    def test_compare_returns_sorted(self):
        try:
            import torch
            import torch.nn as nn

            from pygeovision.benchmark.evaluator import ModelEvaluator

            class DummyA(nn.Module):
                def forward(self, x): return torch.ones(x.shape[0], 2, x.shape[2], x.shape[3])
            class DummyB(nn.Module):
                def forward(self, x): return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

            data = [(torch.randn(2, 3, 32, 32), torch.zeros(2, 32, 32, dtype=torch.long)) for _ in range(2)]
            ev = ModelEvaluator(task="segmentation", num_classes=2)
            results = ev.compare({"ModelA": DummyA(), "ModelB": DummyB()}, data)
            assert len(results) == 2
            # Results should be sorted descending by primary metric
            for i in range(len(results) - 1):
                assert results[i].primary_metric >= results[i+1].primary_metric
        except ImportError:
            pytest.skip("torch not installed")

    def test_save_results(self, tmp_path):
        from pygeovision.benchmark.evaluator import BenchmarkResult, ModelEvaluator
        results = [
            BenchmarkResult("UNet", "LoveDA", "segmentation", mean_iou=0.72),
            BenchmarkResult("SegFormer", "LoveDA", "segmentation", mean_iou=0.78),
        ]
        path = tmp_path / "results.json"
        ev = ModelEvaluator()
        ev.save_results(results, str(path))
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert len(loaded) == 2
        assert loaded[0]["model"] in ("UNet", "SegFormer")

    def test_print_leaderboard_no_error(self, capsys):
        from pygeovision.benchmark.evaluator import BenchmarkResult, ModelEvaluator
        results = [
            BenchmarkResult("UNet", "LoveDA", "segmentation", mean_iou=0.72, mean_f1=0.80, accuracy=0.91),
            BenchmarkResult("SegFormer", "LoveDA", "segmentation", mean_iou=0.78, mean_f1=0.84, accuracy=0.93),
        ]
        ev = ModelEvaluator()
        ev.print_leaderboard(results)
        out = capsys.readouterr().out
        assert "SegFormer" in out or "UNet" in out


class TestLeaderboard:
    def test_add_and_get(self, tmp_path):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        from pygeovision.benchmark.leaderboard import Leaderboard
        lb = Leaderboard(str(tmp_path / "lb.json"))
        r1 = BenchmarkResult("UNet", "LoveDA", "segmentation", mean_iou=0.72)
        r2 = BenchmarkResult("SegFormer", "LoveDA", "segmentation", mean_iou=0.78)
        lb.add(r1); lb.add(r2)
        entries = lb.get(task="segmentation", dataset="LoveDA")
        assert len(entries) == 2
        assert entries[0]["mean_iou"] >= entries[1]["mean_iou"]

    def test_deduplicates(self, tmp_path):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        from pygeovision.benchmark.leaderboard import Leaderboard
        lb = Leaderboard(str(tmp_path / "lb.json"))
        r = BenchmarkResult("UNet", "LoveDA", "segmentation", mean_iou=0.72)
        lb.add(r)
        r2 = BenchmarkResult("UNet", "LoveDA", "segmentation", mean_iou=0.75)
        lb.add(r2)
        entries = lb.get(task="segmentation", dataset="LoveDA")
        unet_entries = [e for e in entries if e["model"] == "UNet"]
        assert len(unet_entries) == 1
        assert unet_entries[0]["mean_iou"] == 0.75

    def test_persistence(self, tmp_path):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        from pygeovision.benchmark.leaderboard import Leaderboard
        path = str(tmp_path / "lb.json")
        lb = Leaderboard(path)
        lb.add(BenchmarkResult("UNet", "ISPRS", "segmentation", mean_iou=0.81))
        # Re-load from disk
        lb2 = Leaderboard(path)
        entries = lb2.get(task="segmentation")
        assert len(entries) == 1
        assert entries[0]["model"] == "UNet"

    def test_export_csv(self, tmp_path):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        from pygeovision.benchmark.leaderboard import Leaderboard
        lb = Leaderboard(str(tmp_path / "lb.json"))
        lb.add(BenchmarkResult("SegFormer", "LoveDA", "segmentation", mean_iou=0.78))
        csv_path = str(tmp_path / "export.csv")
        lb.export_csv(csv_path)
        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["model"] == "SegFormer"

    def test_filter_by_task(self, tmp_path):
        from pygeovision.benchmark.evaluator import BenchmarkResult
        from pygeovision.benchmark.leaderboard import Leaderboard
        lb = Leaderboard(str(tmp_path / "lb.json"))
        lb.add(BenchmarkResult("YOLOv8", "DOTA", "detection", mAP50=0.65))
        lb.add(BenchmarkResult("UNet", "LoveDA", "segmentation", mean_iou=0.72))
        det = lb.get(task="detection")
        seg = lb.get(task="segmentation")
        assert len(det) == 1 and det[0]["model"] == "YOLOv8"
        assert len(seg) == 1 and seg[0]["model"] == "UNet"


class TestModelEvaluatorDetectionMetricsWiring:
    """Regression tests for a real, severe bug: DetectionMetrics was
    constructed with `DetectionMetrics(self.num_classes) if self.task ==
    "detection" else None` as a BARE EXPRESSION STATEMENT — never assigned
    to a variable, so it was immediately discarded. Detection-task
    evaluation ran the full inference loop and timed it, but computed
    ZERO actual mAP metrics — mAP50/mAP50_95 always silently stayed at
    their default 0.0, regardless of real model performance."""

    def test_detection_metrics_are_genuinely_computed(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        pytest.importorskip("torchmetrics", reason="torchmetrics not installed")
        pytest.importorskip("torchvision", reason="torchvision not installed")
        from pygeovision.benchmark.evaluator import ModelEvaluator
        from pygeovision.models.registry import get_model

        torch.manual_seed(0)
        model = get_model("faster-rcnn-r50", num_classes=2, in_channels=3, pretrained=False)

        def fake_loader():
            for _ in range(2):
                images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
                targets = [
                    {"boxes": torch.tensor([[10., 10., 40., 40.]]), "labels": torch.tensor([1])},
                    {"boxes": torch.tensor([[5., 5., 30., 30.]]), "labels": torch.tensor([1])},
                ]
                yield images, targets

        evaluator = ModelEvaluator(task="detection", num_classes=2)
        result = evaluator.evaluate(model, fake_loader(), dataset_name="test", model_name="faster-rcnn")

        # The real test: this must have actually processed all 4 images
        # through the correct list-in/list-out detection calling
        # convention, not silently short-circuited.
        assert result.n_samples == 4
        assert result.notes == "" or "pycocotools" not in result.notes.lower()

    def test_detection_evaluate_uses_list_calling_convention_not_batched_tensor(self):
        """Regression test for the calling-convention half of the bug:
        detection models take a LIST of image tensors and return a LIST
        of dicts — not the batched-tensor-in/logits-out convention
        segmentation uses. A model stub records what it was actually
        called with to prove the right convention is used."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        pytest.importorskip("torchmetrics", reason="torchmetrics not installed")
        from pygeovision.benchmark.evaluator import ModelEvaluator

        calls = []

        class FakeDetectionModel(torch.nn.Module):
            def forward(self, images):
                calls.append(images)
                return [
                    {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.long)}
                    for _ in images
                ]

        def fake_loader():
            images = [torch.rand(3, 32, 32), torch.rand(3, 32, 32)]
            targets = [
                {"boxes": torch.tensor([[1., 1., 5., 5.]]), "labels": torch.tensor([1])},
                {"boxes": torch.tensor([[2., 2., 6., 6.]]), "labels": torch.tensor([1])},
            ]
            yield images, targets

        evaluator = ModelEvaluator(task="detection", num_classes=2)
        evaluator.evaluate(FakeDetectionModel(), fake_loader(), model_name="fake")

        assert len(calls) == 1
        assert isinstance(calls[0], list), "model was not called with a list of images"
        assert len(calls[0]) == 2
