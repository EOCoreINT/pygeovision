"""Regression tests for ActiveLearner.

Real defects fixed:
  1. ``margin`` sorted along the *last* image axis and ``least_confidence`` took ``probs.max()`` over the
     whole (C, H, W) array, so both were meaningless for segmentation outputs.
  2. The model was never moved to ``device``: on a CUDA machine every item raised, the exception was
     swallowed and ``select`` returned ``[]``.
  3. Items that failed to load were dropped silently (and in ``coreset`` shifted the index mapping, so the
     wrong items were returned).
  4. ``coreset`` took the first 128 numbers of the last layer's output as a "feature", ignored the seed.
  5. ``committee`` returned the entropy of the mean prediction (not disagreement) and gave identical scores
     for models with no dropout, without warning.
  6. ``budget``, ``batch_size`` and ``n_iterations`` had no effect.
"""
import pytest

torch = pytest.importorskip("torch")

from pygeovision.labeling.active import ActiveLearner  # noqa: E402


class Tiny(torch.nn.Module):
    """logits = a fixed 1x1 conv: class 1 gets +k*x, class 0 gets -k*x -> x=0 is ambiguous, |x| large is confident."""

    def __init__(self, k=8.0, dropout=False):
        super().__init__()
        self.drop = torch.nn.Dropout2d(0.5) if dropout else torch.nn.Identity()
        self.conv = torch.nn.Conv2d(1, 2, 1, bias=False)
        with torch.no_grad():
            self.conv.weight[:] = torch.tensor([[-k], [k]]).view(2, 1, 1, 1)

    def forward(self, x):
        return self.conv(self.drop(x))


def _chip(v, size=8):
    return torch.full((1, size, size), float(v))


POOL = [_chip(1.0), _chip(0.0), _chip(0.5), _chip(-1.0)]      # idx 1 is the most ambiguous, then idx 2


@pytest.mark.parametrize("strategy", ["entropy", "least_confidence", "margin"])
def test_uncertainty_ranks_ambiguous_chips_first_for_segmentation(strategy):
    sel = ActiveLearner(strategy=strategy).select(Tiny(), POOL, n_select=2, device="cpu")
    assert [s["idx"] for s in sel] == [1, 2]
    assert sel[0]["score"] > sel[1]["score"]


def test_mixed_confidence_inside_one_chip_is_averaged():
    half = torch.cat([torch.zeros(1, 8, 4), torch.ones(1, 8, 4)], dim=2)      # half ambiguous
    sel = ActiveLearner(strategy="entropy").select(Tiny(), [_chip(1.0), half, _chip(0.0)], n_select=3, device="cpu")
    assert [s["idx"] for s in sel] == [2, 1, 0]


def test_classification_logits_and_single_logit_heads():
    class Cls(torch.nn.Module):
        def forward(self, x):
            m = x.mean(dim=(1, 2, 3)).unsqueeze(1)
            return torch.cat([-m, m], dim=1)               # (1, 2)

    sel = ActiveLearner().select(Cls(), POOL, n_select=1, device="cpu")
    assert sel[0]["idx"] == 1

    class Sig(torch.nn.Module):
        def forward(self, x):
            return x * 8                                     # (1, 1, H, W) one logit

    sel = ActiveLearner().select(Sig(), POOL, n_select=1, device="cpu")
    assert sel[0]["idx"] == 1


def test_dict_items_keep_their_fields_and_get_a_score(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin
    paths = []
    for i, v in enumerate([0.0, 1.0]):
        p = tmp_path / f"c{i}.tif"
        arr = np.full((1, 8, 8), v, "float32")
        arr[0, 0, 0] = 1.0 - v           # keep the min-max range non-degenerate
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
                           transform=from_origin(0, 8, 1, 1)) as d:
            d.write(arr)
        paths.append({"path": str(p), "tile_id": f"t{i}"})
    sel = ActiveLearner(loader=lambda path: torch.from_numpy(
        __import__("rasterio").open(path).read().astype("float32")).unsqueeze(0)).select(Tiny(), paths, n_select=2, device="cpu")
    assert {s["tile_id"] for s in sel} == {"t0", "t1"} and all("score" in s for s in sel)


def test_failures_are_skipped_loudly_and_total_failure_raises(caplog):
    pool = [_chip(0.0), "does_not_exist.tif", _chip(1.0)]
    with caplog.at_level("WARNING"):
        sel = ActiveLearner().select(Tiny(), pool, n_select=3, device="cpu")
    assert [s["idx"] for s in sel] == [0, 2]
    assert "could not be scored" in caplog.text
    with pytest.raises(RuntimeError, match="no pool item could be scored"):
        ActiveLearner().select(Tiny(), ["a.tif", "b.tif"], device="cpu")


def test_model_is_moved_to_the_requested_device():
    class Spy(Tiny):
        moved_to = None

        def to(self, *a, **k):
            Spy.moved_to = a[0]
            return super().to(*a, **k)
    ActiveLearner().select(Spy(), POOL, n_select=1, device="cpu")
    assert str(Spy.moved_to) == "cpu"


def test_coreset_returns_diverse_samples_with_correct_indices():
    near_a = [_chip(1.0), _chip(1.0), _chip(1.0)]
    far_b = [_chip(-1.0)]
    pool = [near_a[0], "bad.tif", near_a[1], far_b[0], near_a[2]]
    for seed in range(5):
        sel = ActiveLearner(strategy="coreset", seed=seed).select(Tiny(), pool, n_select=2, device="cpu")
        idx = sorted(s["idx"] for s in sel)
        assert 1 not in idx                                          # the failed item is never returned
        assert 3 in idx and any(i in (0, 2, 4) for i in idx)         # one from each cluster
    a = ActiveLearner(strategy="coreset", seed=3).select(Tiny(), pool, n_select=3, device="cpu")
    b = ActiveLearner(strategy="coreset", seed=3).select(Tiny(), pool, n_select=3, device="cpu")
    assert [s["idx"] for s in a] == [s["idx"] for s in b]            # deterministic for a seed


def test_committee_needs_dropout_and_uses_disagreement():
    with pytest.raises(ValueError, match="no dropout"):
        ActiveLearner(strategy="committee").select(Tiny(), POOL, device="cpu")
    model = Tiny(dropout=True)
    torch.manual_seed(0)
    sel = ActiveLearner(strategy="committee", mc_passes=12).select(model, POOL, n_select=4, device="cpu")
    assert len(sel) == 4 and all(s["score"] >= -1e-6 for s in sel)
    assert not model.training and not model.drop.training            # restored to eval
    # a chip of zeros is deterministic under input dropout (still 0) -> no disagreement at all
    zero = next(s for s in sel if s["idx"] == 1)
    assert zero["score"] == pytest.approx(0.0, abs=1e-6)


def test_random_is_seeded_and_does_not_need_a_model():
    pool = [{"path": f"{i}.tif"} for i in range(20)]
    a = ActiveLearner(strategy="random", seed=1).select(None, pool, n_select=5)
    b = ActiveLearner(strategy="random", seed=1).select(None, pool, n_select=5)
    c = ActiveLearner(strategy="random", seed=2).select(None, pool, n_select=5)
    assert a == b and a != c


def test_budget_batch_size_and_rounds_are_enforced():
    pool = [{"path": f"{i}.tif"} for i in range(50)]
    lr = ActiveLearner(strategy="random", budget=10, batch_size=4, n_iterations=5)
    assert len(lr.select(None, pool)) == 4
    lr.update(pool[:4]); lr.update(pool[4:8])
    assert lr.remaining_budget == 2 and not lr.finished
    assert len(lr.select(None, pool)) == 2                           # capped by what is left
    lr.update(pool[8:10])
    assert lr.finished and lr.select(None, pool) == []
    assert ActiveLearner(budget=100, n_iterations=1).update([{"path": "x"}]).finished


def test_train_iteration_warns_about_ignored_config(caplog):
    lr = ActiveLearner(strategy="random", batch_size=3)
    with caplog.at_level("WARNING"):
        out = lr.train_iteration(None, {"epochs": 5}, [{"path": f"{i}.tif"} for i in range(6)])
    assert "does not train" in caplog.text and out["n_selected"] == 3 and out["remaining_budget"] == 100
    assert lr.history == [out]


def test_learning_curve_records_metrics():
    lr = ActiveLearner(strategy="random", budget=50)
    lr.update([{"path": "a"}] * 5, metrics={"val_iou": 0.4})
    lr.update([{"path": "b"}] * 5, metrics={"val_iou": 0.55})
    assert [(c["n_labeled"], c["val_iou"]) for c in lr.learning_curve] == [(5, 0.4), (10, 0.55)]


def test_constructor_validation():
    with pytest.raises(ValueError):
        ActiveLearner(budget=0)
    with pytest.raises(ValueError):
        ActiveLearner(mc_passes=1)
