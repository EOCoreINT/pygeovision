"""GeoTrainer: one implementation, a correct LR schedule, usable checkpoints, resume.

Each test pins a defect found while moving a real project (an illegal-mining
U-Net) onto PyGeoVision:

* two diverging GeoTrainer classes, and ``pygeovision.ai.training.GeoTrainer`` was None;
* epoch-based LR schedules were stepped per *batch* (cosine finished inside epoch 1);
* the checkpoint manager could delete the file it had just written, leaving a stale ``last.pth``;
* only the mean over classes was reported, so a rare class could not drive checkpoint selection;
* ``freeze_backbone`` could not be undone, and a parameter frozen at optimizer-build time never trained;
* an interrupted run could not be resumed.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from pygeovision.ai.training.trainer import CheckpointManager, GeoTrainer, TrainingConfig, _unwrap  # noqa: E402


class TinySeg(nn.Module):
    """encoder -> decoder, so the trainer's backbone-name patterns apply."""

    def __init__(self, classes=2):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 3, padding=1)
        self.decoder = nn.Conv2d(8, classes, 1)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


def make_data(n=20, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, size, size, generator=g)
    y = (x[:, 0] > 0.7).long()           # a learnable rule, and class 1 is the rare class
    return torch.utils.data.TensorDataset(x, y)


def loaders(batch_size=4):
    ds = make_data()
    return (torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False),
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False))


def cfg(tmp_path, **kw):
    base = dict(output_dir=str(tmp_path), max_epochs=4, batch_size=4, num_workers=0,
                persistent_workers=False, pin_memory=False, augment=False, precision="fp32",
                learning_rate=1e-2, early_stopping_patience=99, warmup_epochs=0, seed=0)
    base.update(kw)
    return TrainingConfig(**base)


def record_lrs(monkeypatch):
    lrs: list[list[float]] = []
    orig = torch.optim.AdamW.step

    def step(self, *a, **k):
        lrs.append([g["lr"] for g in self.param_groups])
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    return lrs


class TestOneTrainer:
    def test_same_class_everywhere(self):
        import pygeovision.ai.training as ai_training
        import pygeovision.training as training
        assert ai_training.GeoTrainer is training.GeoTrainer is GeoTrainer
        assert ai_training.TrainingConfig is training.TrainingConfig is TrainingConfig


class TestLearningRateSchedule:
    def test_cosine_follows_epochs_not_batches(self, tmp_path, monkeypatch):
        lrs = record_lrs(monkeypatch)
        tl, vl = loaders()                       # 5 batches per epoch
        GeoTrainer(TinySeg(), cfg(tmp_path, max_epochs=4, scheduler="cosine", min_lr=0.0)).fit(
            train_loader=tl, val_loader=vl)
        assert len(lrs) == 20
        per_epoch = [lrs[e * 5][0] for e in range(4)]
        expected = [1e-2 * 0.5 * (1 + math.cos(math.pi * e / 4)) for e in range(4)]
        assert per_epoch == pytest.approx(expected, rel=1e-6)
        # constant inside an epoch
        assert len({round(l[0], 12) for l in lrs[:5]}) == 1

    def test_onecycle_still_steps_per_batch(self, tmp_path, monkeypatch):
        lrs = record_lrs(monkeypatch)
        tl, vl = loaders()
        GeoTrainer(TinySeg(), cfg(tmp_path, max_epochs=4, scheduler="onecycle")).fit(train_loader=tl, val_loader=vl)
        assert len({round(l[0], 12) for l in lrs[:5]}) > 1

    def test_gradient_accumulation_flushes_the_last_partial_window(self, tmp_path, monkeypatch):
        lrs = record_lrs(monkeypatch)
        tl, vl = loaders()                       # 5 batches, accumulation 2 -> steps at 2, 4, 5
        GeoTrainer(TinySeg(), cfg(tmp_path, max_epochs=1, grad_accumulation_steps=2)).fit(
            train_loader=tl, val_loader=vl)
        assert len(lrs) == 3


class TestCheckpoints:
    def _save(self, cm, model, epoch, score):
        return cm.save(model, epoch, {"val_iou": score})

    def test_worse_checkpoint_is_not_written_and_nothing_is_deleted_by_mistake(self, tmp_path):
        cm = CheckpointManager(tmp_path, top_k=2)
        m = TinySeg()
        assert self._save(cm, m, 1, 0.5)
        assert self._save(cm, m, 2, 0.7)
        assert self._save(cm, m, 3, 0.4) is None       # worse than both kept
        names = sorted(p.name for p in cm.dir.glob("epoch_*.pth"))
        assert names == ["epoch_0001_val_iou_0.5000.pth", "epoch_0002_val_iou_0.7000.pth"]
        assert cm.best_path().name == "epoch_0002_val_iou_0.7000.pth"

    def test_better_checkpoint_evicts_the_worst(self, tmp_path):
        cm = CheckpointManager(tmp_path, top_k=2)
        m = TinySeg()
        for e, sc in [(1, 0.5), (2, 0.7), (3, 0.9)]:
            self._save(cm, m, e, sc)
        assert sorted(p.name for p in cm.dir.glob("epoch_*.pth")) == [
            "epoch_0002_val_iou_0.7000.pth", "epoch_0003_val_iou_0.9000.pth"]
        best = torch.load(cm.dir / "best.pth")
        assert best["epoch"] == 3

    def test_min_mode(self, tmp_path):
        cm = CheckpointManager(tmp_path, top_k=1, metric="val_loss", mode="min")
        m = TinySeg()
        cm.save(m, 1, {"val_loss": 0.5})
        assert cm.save(m, 2, {"val_loss": 0.9}) is None
        assert cm.save(m, 3, {"val_loss": 0.1})
        assert cm.best_path().name.startswith("epoch_0003")

    def test_last_is_written_every_epoch_even_when_the_epoch_is_not_a_best_one(self, tmp_path):
        """Used to be a copy of the newest top-k file, so a worse epoch left last.pth stale."""
        tl, vl = loaders()
        c = cfg(tmp_path, max_epochs=5, save_top_k=1, learning_rate=0.0, min_lr=0.0)   # lr 0: every epoch scores the same
        r = GeoTrainer(TinySeg(), c).fit(train_loader=tl, val_loader=vl)
        last = torch.load(tmp_path / "checkpoints" / "last.pth", weights_only=True)
        assert last["epoch"] == 5
        assert r["last_checkpoint"].endswith("last.pth")
        assert len(list((tmp_path / "checkpoints").glob("epoch_*.pth"))) == 1

    def test_checkpoints_load_into_a_plain_model_even_from_a_wrapped_one(self, tmp_path):
        class DistributedDataParallel(nn.Module):       # same class name torch uses
            def __init__(self, module):
                super().__init__()
                self.module = module

        inner = TinySeg()
        wrapped = DistributedDataParallel(inner)
        assert _unwrap(wrapped) is inner
        cm = CheckpointManager(tmp_path, top_k=1)
        cm.save(wrapped, 1, {"val_iou": 0.5})
        TinySeg().load_state_dict(torch.load(cm.best_path(), weights_only=True)["model_state_dict"])


class TestPerClassMetrics:
    def test_rare_class_numbers_are_reported_and_can_select_the_checkpoint(self, tmp_path):
        tl, vl = loaders()
        c = cfg(tmp_path, max_epochs=6, checkpoint_metric="val_f1_class_1", early_stopping_metric="val_f1_class_1")
        r = GeoTrainer(TinySeg(), c).fit(train_loader=tl, val_loader=vl, loss_fn=nn.CrossEntropyLoss())
        for key in ("val_iou_class_1", "val_f1_class_1", "val_precision_class_1", "val_recall_class_1"):
            assert key in r["history"] and len(r["history"][key]) == r["epochs_trained"]
        assert r["best_metric_name"] == "val_f1_class_1"
        best = torch.load(r["best_checkpoint"], weights_only=True)
        assert best["metric"] == "val_f1_class_1"
        assert best["score"] == pytest.approx(max(r["history"]["val_f1_class_1"]), abs=1e-4)

    def test_mean_iou_hides_a_model_that_never_predicts_the_rare_class(self):
        from pygeovision.training.metrics import SegmentationMetrics
        m = SegmentationMetrics(2)
        target = torch.zeros(1, 10, 10, dtype=torch.long)
        target[0, :2] = 1                                  # 20% positive
        m.update(torch.zeros_like(target), target)          # predicts background everywhere
        out = m.compute()
        assert out["mean_iou"] > 0.35 and out["iou_class_1"] == pytest.approx(0.0, abs=1e-6)
        assert out["recall_class_1"] == pytest.approx(0.0, abs=1e-6)


class TestStagedBackbone:
    def test_frozen_for_n_epochs_then_trains_at_a_lower_rate(self, tmp_path, monkeypatch):
        lrs = record_lrs(monkeypatch)
        model = TinySeg()
        snapshots = []
        orig_train = model.train

        def train_hook(mode=True):                         # called at the start of every epoch
            snapshots.append(model.encoder.weight.detach().clone())
            return orig_train(mode)

        monkeypatch.setattr(model, "train", train_hook)
        tl, vl = loaders()
        GeoTrainer(model, cfg(tmp_path, max_epochs=4, unfreeze_backbone_epoch=2, backbone_lr_scale=0.1,
                              scheduler="step", lr_step_size=100)).fit(train_loader=tl, val_loader=vl)
        epoch_starts = snapshots[0::2][:4] if len(snapshots) > 4 else snapshots   # train() is also called after eval
        # encoder identical at the start of epochs 1, 2 and 3 => untouched during epochs 1 and 2 ...
        torch.testing.assert_close(epoch_starts[0], epoch_starts[1])
        torch.testing.assert_close(epoch_starts[1], epoch_starts[2])
        # ... and different afterwards: it trained in epoch 3 (and was in the optimizer from the start)
        assert not torch.equal(epoch_starts[2], model.encoder.weight.detach())
        assert lrs[0] == pytest.approx([1e-2, 1e-3])

    def test_frozen_parameter_is_in_the_optimizer_so_unfreezing_actually_trains_it(self, tmp_path):
        model = TinySeg()
        before = model.encoder.weight.detach().clone()
        tl, vl = loaders()
        GeoTrainer(model, cfg(tmp_path, max_epochs=3, unfreeze_backbone_epoch=1)).fit(train_loader=tl, val_loader=vl)
        assert not torch.equal(before, model.encoder.weight.detach())

    def test_permanent_freeze_still_works(self, tmp_path):
        model = TinySeg()
        before = model.encoder.weight.detach().clone()
        tl, vl = loaders()
        GeoTrainer(model, cfg(tmp_path, max_epochs=2, freeze_backbone=True)).fit(train_loader=tl, val_loader=vl)
        torch.testing.assert_close(before, model.encoder.weight.detach())


class TestResume:
    def test_interrupted_run_continues_exactly_like_an_uninterrupted_one(self, tmp_path):
        tl, vl = loaders()                                  # 5 batches / epoch, deterministic
        torch.manual_seed(1)
        init = {k: v.clone() for k, v in TinySeg().state_dict().items()}

        def fresh():
            m = TinySeg()
            m.load_state_dict(init)
            return m

        uninterrupted = fresh()
        r_full = GeoTrainer(uninterrupted, cfg(tmp_path / "full", max_epochs=4)).fit(train_loader=tl, val_loader=vl)

        class Crash(Exception):
            pass

        calls = {"n": 0}

        def crashing_loss(logits, target):                  # power cut in the middle of epoch 3
            calls["n"] += 1
            if calls["n"] > 2 * (5 + 5) + 2:                # 2 epochs of train+val batches, then 2 more
                raise Crash()
            return nn.functional.cross_entropy(logits, target)

        with pytest.raises(Crash):
            GeoTrainer(fresh(), cfg(tmp_path / "run", max_epochs=4)).fit(
                train_loader=tl, val_loader=vl, loss_fn=crashing_loss)
        assert torch.load(tmp_path / "run" / "checkpoints" / "last.pth", weights_only=True)["epoch"] == 2

        resumed = fresh()                                   # a new process: fresh model, same output_dir
        r = GeoTrainer(resumed, cfg(tmp_path / "run", max_epochs=4, resume=True)).fit(train_loader=tl, val_loader=vl)

        assert r["resumed_from_epoch"] == 2 and r["epochs_trained"] == 4
        for (k, a), (_, b) in zip(uninterrupted.state_dict().items(), resumed.state_dict().items()):
            torch.testing.assert_close(a, b, msg=k)
        assert r["history"]["train_loss"] == pytest.approx(r_full["history"]["train_loss"], rel=1e-6)

    def test_resume_without_a_previous_run_just_starts(self, tmp_path):
        tl, vl = loaders()
        r = GeoTrainer(TinySeg(), cfg(tmp_path, max_epochs=2, resume=True)).fit(train_loader=tl, val_loader=vl)
        assert r["epochs_trained"] == 2 and r["resumed_from_epoch"] is None

    def test_resume_restores_early_stopping_state(self, tmp_path):
        tl, vl = loaders()
        c = cfg(tmp_path, max_epochs=3, learning_rate=0.0, min_lr=0.0, early_stopping_patience=2)
        GeoTrainer(TinySeg(), c).fit(train_loader=tl, val_loader=vl)       # lr 0 -> no improvement -> stops at epoch 3
        state = torch.load(tmp_path / "checkpoints" / "last.pth", weights_only=True)
        assert state["early_stopping"]["should_stop"] is True
        c2 = cfg(tmp_path, max_epochs=10, learning_rate=0.0, min_lr=0.0, early_stopping_patience=2, resume=True)
        r = GeoTrainer(TinySeg(), c2).fit(train_loader=tl, val_loader=vl)
        assert r["epochs_trained"] <= 4          # stops again right away instead of running to 10

    def test_resume_into_a_different_model_is_refused(self, tmp_path):
        tl, vl = loaders()
        GeoTrainer(TinySeg(2), cfg(tmp_path, max_epochs=1)).fit(train_loader=tl, val_loader=vl)
        with pytest.raises(RuntimeError, match="could not be restored"):
            GeoTrainer(TinySeg(3), cfg(tmp_path, max_epochs=2, resume=True)).fit(train_loader=tl, val_loader=vl)


class TestDeviceAndHistory:
    def test_cpu_string_selects_cpu_even_when_cuda_exists(self, tmp_path):
        # devices="cpu" used to fall through to torch.device("cuda")
        t = GeoTrainer(TinySeg(), cfg(tmp_path, devices="cpu"))
        assert t._get_device().type == "cpu"
        assert GeoTrainer(TinySeg(), cfg(tmp_path, devices="cuda:1"))._get_device().index == 1
        assert GeoTrainer(TinySeg(), cfg(tmp_path, devices=[2, 3]))._get_device().index == 2

    def test_history_records_the_learning_rate_per_epoch(self, tmp_path):
        tr, va = loaders()
        r = GeoTrainer(TinySeg(), cfg(tmp_path, max_epochs=3, devices="cpu")).fit(
            train_loader=tr, val_loader=va, loss_fn=lambda p, t: nn.functional.cross_entropy(p, t))
        lr = r["history"]["lr"]
        assert len(lr) == 3 and all(x > 0 for x in lr)
        assert lr[0] > lr[-1]                      # cosine decays


class TestUnwrap:
    def test_duck_typed_or_mock_models_do_not_hang(self):
        # hasattr(mock, "_orig_mod") is always True and yields a new mock each time;
        # an unbounded unwrap loop never returned (and ate memory until killed)
        from unittest.mock import MagicMock
        m = MagicMock()
        assert _unwrap(m) is m
        o = object()
        assert _unwrap(o) is o

    def test_real_wrappers_are_still_unwrapped(self):
        inner = TinySeg()
        dp = nn.DataParallel(inner)
        assert _unwrap(dp) is inner

        class Compiled(nn.Module):               # shape of torch.compile's OptimizedModule
            def __init__(self, m):
                super().__init__()
                self._orig_mod = m
        assert _unwrap(Compiled(inner)) is inner
        assert _unwrap(Compiled(Compiled(inner))) is inner
