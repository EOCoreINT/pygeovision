"""Regression tests for the benchmark evaluator (written with the benchmark documentation).

Real problems covered: a failed evaluation looked like a model scoring 0; classes absent from the
test set dragged mIoU down; models with a coarser output (SegFormer-style) or dict/HF-style outputs
were scored as nothing; the leaderboard accepted failed runs.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from pygeovision.benchmark import BenchmarkResult, Leaderboard, ModelEvaluator  # noqa: E402


class Const(nn.Module):
    """Predicts class 1 where the input's first band is above 0.5, else class 0 (an exact rule)."""

    def forward(self, x):
        pos = (x[:, :1] > 0.5).float()
        return torch.cat([1 - pos, pos], dim=1) * 10


def _loader(n_classes_in_target=2, size=16, n=8):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 3, size, size, generator=g)
    y = (x[:, 0] > 0.5).long()
    if n_classes_in_target == 1:
        y = torch.zeros_like(y)
    return DataLoader(TensorDataset(x, y), batch_size=4)


class TestFailuresAreNotScores:
    def test_exception_is_recorded_and_sorted_last(self):
        class Broken(nn.Module):
            def forward(self, x):
                raise RuntimeError("boom")

        ev = ModelEvaluator("segmentation", num_classes=2)
        res = ev.compare({"bad": Broken(), "good": Const()}, _loader(), device="cpu")
        assert [r.model_name for r in res] == ["good", "bad"]
        assert res[1].failed and "boom" in res[1].error and res[1].to_dict()["error"]
        assert not res[0].failed and res[0].mean_iou == pytest.approx(1.0)

    def test_empty_loader_is_an_error(self):
        r = ModelEvaluator("segmentation", num_classes=2).evaluate(Const(), [], device="cpu")
        assert r.failed and "no batches" in r.error

    def test_unscorable_batches_are_an_error(self):
        x = torch.rand(4, 3, 8, 8)
        y = torch.zeros(4, 8, 8, dtype=torch.long)
        y[0, 0, 0] = 9                                           # class index >= num_classes
        r = ModelEvaluator("segmentation", num_classes=2).evaluate(
            Const(), DataLoader(TensorDataset(x, y), batch_size=4), device="cpu")
        assert r.failed and "could not be scored" in r.error

    def test_leaderboard_refuses_failed_results(self, tmp_path):
        lb = Leaderboard(tmp_path / "lb.json")
        with pytest.raises(ValueError, match="failed"):
            lb.add(BenchmarkResult("m", "d", "segmentation", error="boom"))
        assert lb.get() == []

    def test_print_marks_failed(self, capsys):
        ev = ModelEvaluator("segmentation", num_classes=2)
        ev.print_leaderboard([BenchmarkResult("ok", "d", "segmentation", mean_iou=0.5),
                              BenchmarkResult("bad", "d", "segmentation", error="boom")])
        out = capsys.readouterr().out
        assert "FAILED" in out and "boom" in out


class TestOutputsAndMetrics:
    def test_absent_class_does_not_lower_miou(self):
        ev = ModelEvaluator("segmentation", num_classes=3)       # class 2 never occurs
        r = ev.evaluate(Const(), _loader(), device="cpu")
        assert r.mean_iou == pytest.approx(1.0)
        assert r.per_class_iou["iou_class_2"] == 0.0             # still listed per class

    def test_coarse_output_is_resized_to_the_target(self):
        class Coarse(nn.Module):
            def forward(self, x):
                return nn.functional.avg_pool2d(Const()(x), 4)

        r = ModelEvaluator("segmentation", num_classes=2).evaluate(Coarse(), _loader(), device="cpu")
        # pooled-then-upsampled logits blur a random-noise rule, so the score is
        # modest; the point is that the output is resized and scored, not dropped
        assert not r.failed and r.accuracy > 0.5

    @pytest.mark.parametrize("wrap", ["dict", "tuple", "logits"])
    def test_wrapped_outputs(self, wrap):
        class Wrapped(nn.Module):
            def forward(self, x):
                out = Const()(x)
                if wrap == "dict":
                    return {"out": out}
                if wrap == "tuple":
                    return (out, None)

                class O:
                    logits = out
                return O()

        r = ModelEvaluator("segmentation", num_classes=2).evaluate(Wrapped(), _loader(), device="cpu")
        assert r.mean_iou == pytest.approx(1.0)

    def test_dict_batches(self):
        class DS(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, i):
                x = torch.rand(3, 8, 8)
                return {"image": x, "mask": (x[0] > 0.5).long()}

        r = ModelEvaluator("segmentation", num_classes=2).evaluate(Const(), DataLoader(DS(), batch_size=2), device="cpu")
        assert r.mean_iou == pytest.approx(1.0)

    def test_dict_batch_without_mask_names_the_keys(self):
        r = ModelEvaluator("segmentation", num_classes=2).evaluate(
            Const(), [{"foo": torch.rand(1, 3, 4, 4)}], device="cpu")
        assert r.failed and "'mask'" in r.error

    def test_classification(self):
        class Clf(nn.Module):
            def forward(self, x):
                return torch.stack([(x[:, 0].mean((1, 2)) <= 0.5).float(), (x[:, 0].mean((1, 2)) > 0.5).float()], 1)

        x = torch.rand(8, 3, 4, 4)
        y = (x[:, 0].mean((1, 2)) > 0.5).long()
        r = ModelEvaluator("classification", num_classes=2).evaluate(Clf(), DataLoader(TensorDataset(x, y), batch_size=4), device="cpu")
        assert r.accuracy == 1.0 and r.primary_metric == 1.0

    def test_segmentation_metrics_counts_failed_updates_and_noncontiguous(self):
        from pygeovision.training.metrics import SegmentationMetrics
        m = SegmentationMetrics(2)
        p = torch.zeros(2, 4, 4, dtype=torch.long).permute(0, 2, 1)          # non-contiguous view
        m.update(p, p.clone())
        assert m.n_failed_updates == 0 and m.compute()["accuracy"] == pytest.approx(1.0)
        m.update(torch.full((1, 2, 2), 5), torch.zeros(1, 2, 2, dtype=torch.long))
        assert m.n_failed_updates == 1
        m.reset()
        assert m.n_failed_updates == 0

    def test_detection_without_torchmetrics_is_an_error(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "torchmetrics.detection.mean_ap", None)
        boxes = {"boxes": torch.tensor([[0.0, 0, 5, 5]]), "labels": torch.tensor([1])}
        pred = {**boxes, "scores": torch.tensor([0.9])}

        class Det(nn.Module):
            def forward(self, images):
                return [pred for _ in images]

        r = ModelEvaluator("detection", num_classes=2).evaluate(Det(), [([torch.rand(3, 8, 8)], [boxes])], device="cpu")
        assert r.failed and "torchmetrics" in r.error


class TestSavedFiles:
    def test_error_is_saved_in_json(self, tmp_path):
        import json
        ev = ModelEvaluator("segmentation", num_classes=2)
        ev.save_results([BenchmarkResult("m", "d", "segmentation", error="boom")], tmp_path / "r.json")
        assert json.loads((tmp_path / "r.json").read_text())[0]["error"] == "boom"
