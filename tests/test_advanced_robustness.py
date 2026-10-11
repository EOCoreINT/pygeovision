"""GeoAutoML, MultiTaskLearner and FewShotLearner: the behaviours the documentation promises."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ─────────────────────────────── AutoML ───────────────────────────────

optuna = pytest.importorskip("optuna")

from pygeovision.advanced import GeoAutoML  # noqa: E402


def _automl(**kw):
    kw.setdefault("n_trials", 12)
    kw.setdefault("show_progress_bar", False)
    return GeoAutoML(**kw)


class TestAutoML:
    def test_finds_the_optimum_of_a_simple_function(self):
        r = _automl(n_trials=40).search(lambda c: -(c["x"] - 0.3) ** 2, {"x": ("float", 0.0, 1.0)})
        assert r["success"] and abs(r["best_params"]["x"] - 0.3) < 0.15
        assert r["n_trials_completed"] == 40 and r["n_trials_failed"] == 0

    def test_failed_trials_never_win_when_minimising(self):
        # None used to become -inf, which is the BEST possible value for "minimize"
        def fn(c):
            return None if c["x"] < 0.5 else c["x"]
        r = _automl(direction="minimize", n_trials=30).search(fn, {"x": ("float", 0.0, 1.0)})
        assert r["success"] and r["best_value"] >= 0.5
        assert r["n_trials_failed"] > 0 and r["errors"]

    def test_exceptions_are_failures_not_prunes(self):
        def fn(c):
            if c["x"] > 0.5:
                raise RuntimeError("out of memory")
            return c["x"]
        r = _automl(n_trials=20).search(fn, {"x": ("float", 0.0, 1.0)})
        assert r["n_trials_failed"] > 0 and r["n_trials_pruned"] == 0
        assert r["best_value"] <= 0.5
        assert "RuntimeError: out of memory" in r["errors"][0]

    def test_all_trials_failing_is_reported_not_raised(self):
        def fn(c):
            raise ValueError("bad config")
        r = _automl(n_trials=3).search(fn, {"x": ("float", 0.0, 1.0)})
        assert r["success"] is False and "bad config" in r["error"] and r["n_trials_failed"] == 3

    def test_counts_add_up(self):
        r = _automl(n_trials=15).search(lambda c: c["x"] if c["x"] < 0.8 else None, {"x": ("float", 0, 1)})
        assert r["n_trials_completed"] + r["n_trials_pruned"] + r["n_trials_failed"] == 15

    def test_trial_is_passed_on_request_and_pruning_works(self):
        def fn(config, trial):
            for step in range(10):
                trial.report(config["x"], step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return config["x"]
        r = _automl(n_trials=30).search(fn, {"x": ("float", 0.0, 1.0)}, pruner="median")
        assert r["n_trials_pruned"] > 0 and r["success"]

    def test_pruner_none_means_no_pruning(self):
        def fn(config, trial):
            for step in range(10):
                trial.report(config["x"], step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return config["x"]
        r = _automl(n_trials=30).search(fn, {"x": ("float", 0.0, 1.0)}, pruner=None)
        assert r["n_trials_pruned"] == 0

    def test_same_seed_same_result(self):
        fn = lambda c: -abs(c["x"] - 0.7)          # noqa: E731
        a = _automl(seed=3).search(fn, {"x": ("float", 0, 1)})
        b = _automl(seed=3).search(fn, {"x": ("float", 0, 1)})
        assert a["best_params"] == b["best_params"]

    def test_all_parameter_kinds(self):
        seen = []
        def fn(c):
            seen.append(c)
            return float(c["n"])
        space = {"lr": ("float", 1e-5, 1e-1, "log"), "n": ("int", 1, 5),
                 "loss": ("categorical", ["a", "b"]), "fixed": ("constant", 7)}
        _automl(n_trials=10).search(fn, space)
        assert all(1e-5 <= c["lr"] <= 1e-1 and 1 <= c["n"] <= 5 and c["loss"] in "ab" and c["fixed"] == 7 for c in seen)

    @pytest.mark.parametrize("bad", [{"x": ("flaot", 0, 1)}, {"x": ("float", 1, 0)}, {"x": ("float", 0)},
                                     {"x": ("categorical", [])}, {}, {"x": 5}])
    def test_bad_search_spaces_are_rejected(self, bad):
        with pytest.raises(ValueError):
            _automl().search(lambda c: 0.0, bad)

    def test_bad_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            GeoAutoML(direction="up")
        with pytest.raises(ValueError, match="backend"):
            GeoAutoML(backend="nope")
        with pytest.raises(ValueError, match="sampler"):
            _automl().search(lambda c: 0.0, {"x": ("float", 0, 1)}, sampler="grid")
        with pytest.raises(ValueError, match="pruner"):
            _automl().search(lambda c: 0.0, {"x": ("float", 0, 1)}, pruner="x")

    def test_importance_prefers_the_parameter_that_matters(self):
        a = _automl(n_trials=40)
        a.search(lambda c: c["a"] + 0.01 * c["b"], {"a": ("float", 0, 1), "b": ("float", 0, 1)})
        imp = a.importance()
        assert imp["a"] > imp["b"] and abs(sum(imp.values()) - 1) < 0.05

    def test_importance_before_search_is_empty(self):
        assert _automl().importance() == {}

    def test_persistent_storage_continues_a_study(self, tmp_path):
        url = f"sqlite:///{tmp_path / 's.db'}"
        _automl(n_trials=5, storage=url, study_name="s1").search(lambda c: c["x"], {"x": ("float", 0, 1)})
        r = _automl(n_trials=5, storage=url, study_name="s1").search(lambda c: c["x"], {"x": ("float", 0, 1)})
        assert r["n_trials_completed"] == 10          # earlier trials of the same study are included


# ───────────────────────────── Multi-task ─────────────────────────────

from pygeovision.advanced import MultiTaskLearner  # noqa: E402
from pygeovision.losses.class_balance import LabelSmoothingCrossEntropy  # noqa: E402


@pytest.fixture(scope="module")
def mt():
    torch.manual_seed(0)
    learner = MultiTaskLearner(tasks=["segmentation", "classification"], pretrained=False, device="cpu",
                               n_classes={"segmentation": 3, "classification": 4}, in_channels=4)
    learner.build()
    return learner


def _batches(n=4, size=64, bs=2):
    g = torch.Generator().manual_seed(1)
    out = []
    for _ in range(n):
        out.append({"inputs": torch.rand(bs, 4, size, size, generator=g),
                    "segmentation": torch.randint(0, 3, (bs, size, size), generator=g),
                    "classification": torch.randint(0, 4, (bs,), generator=g)})
    return out


class TestMultiTask:
    @pytest.mark.parametrize("size", [64, 96, 100])
    def test_segmentation_output_matches_the_input_size(self, mt, size):
        out = mt._model.eval()(torch.rand(1, 4, size, size))
        assert out["segmentation"].shape == (1, 3, size, size) and out["classification"].shape == (1, 4)

    def test_loss_works_at_full_resolution(self, mt):
        b = _batches(1)[0]
        loss, per = mt.compute_loss(mt._model.train()(b["inputs"]), {k: b[k] for k in ("segmentation", "classification")})
        assert loss.requires_grad and set(per) == {"segmentation", "classification"}

    def test_extra_input_channels_keep_pretrained_filters_for_rgb(self):
        import torchvision
        a = MultiTaskLearner(tasks=["classification"], pretrained=False, in_channels=5, device="cpu").build()
        assert a.encoder[0].in_channels == 5
        # same code path as pretrained: first three filters copied, extras are their mean
        base = torchvision.models.resnet50(weights=None)
        from pygeovision.advanced import multitask  # noqa: F401
        assert a.encoder[0].weight.shape == (64, 5, 7, 7) and base.conv1.weight.shape == (64, 3, 7, 7)

    def test_training_runs_records_per_task_losses_and_saves(self, mt, tmp_path):
        res = mt.train(_batches(), _batches(2), epochs=2, lr=1e-3, save_path=str(tmp_path / "w.pt"), log_every=0)
        h = res["history"]
        assert res["success"] and len(h["train_loss"]) == 2 and res["best_epoch"] in (1, 2)
        assert set(h["train_task_losses"][0]) == {"segmentation", "classification"}
        assert (tmp_path / "w.pt").exists()

    def test_validation_loader_is_optional(self, mt):
        res = mt.train(_batches(2), None, epochs=1, log_every=0)
        assert res["success"] and res["history"]["val_loss"] == [None]

    def test_saved_weights_reload_to_the_same_predictions(self, mt, tmp_path):
        mt.train(_batches(2), None, epochs=1, save_path=str(tmp_path / "w.pt"), log_every=0)
        x = torch.rand(1, 4, 64, 64)
        mt._model.load_state_dict(torch.load(tmp_path / "w.pt", weights_only=True))   # best == last here
        before = mt.predict(x)
        other = MultiTaskLearner(tasks=["segmentation", "classification"], pretrained=False, device="cpu",
                                 n_classes={"segmentation": 3, "classification": 4}, in_channels=4)
        other.load_weights(str(tmp_path / "w.pt"))
        after = other.predict(x)
        np.testing.assert_array_equal(before["segmentation"], after["segmentation"])
        assert before["segmentation"].shape == (1, 64, 64) and before["classification"].shape == (1,)

    def test_a_task_subset_can_be_requested(self, mt):
        assert set(mt.predict(torch.rand(1, 4, 64, 64), tasks=["classification"])) == {"classification"}
        with pytest.raises(ValueError, match="no head"):
            mt.predict(torch.rand(1, 4, 64, 64), tasks=["detection"])

    def test_bad_batches_get_a_clear_error(self, mt):
        with pytest.raises(ValueError, match="dict with an 'inputs'"):
            mt.train([(torch.rand(1, 4, 64, 64), torch.zeros(1))], None, epochs=1)
        with pytest.raises(ValueError, match="no task has both"):
            mt.compute_loss({"segmentation": torch.rand(1, 3, 8, 8)}, {"other": torch.zeros(1)})
        with pytest.raises(ValueError, match="no batches"):
            mt.train([], None, epochs=1)

    def test_bad_constructor_arguments(self):
        with pytest.raises(ValueError, match="backbone"):
            MultiTaskLearner(backbone="efficientnet")
        with pytest.raises(ValueError, match="unknown task"):
            MultiTaskLearner(tasks=["segmentaton"])

    def test_defaults(self):
        m = MultiTaskLearner()
        assert m.tasks == ["segmentation", "classification"] and m.n_classes == {"segmentation": 2, "classification": 4}
        assert m.task_weights == {"segmentation": 0.5, "classification": 0.5}


class TestLabelSmoothing:
    def test_matches_pytorch(self):
        import torch.nn.functional as F
        g = torch.Generator().manual_seed(0)
        pred, tgt = torch.randn(8, 5, generator=g), torch.randint(0, 5, (8,), generator=g)
        ours = LabelSmoothingCrossEntropy(smoothing=0.1)(pred, tgt)
        assert torch.allclose(ours, F.cross_entropy(pred, tgt, label_smoothing=0.1), atol=1e-6)

    def test_segmentation_with_ignored_pixels(self):
        import torch.nn.functional as F
        g = torch.Generator().manual_seed(0)
        pred = torch.randn(2, 3, 6, 6, generator=g)
        tgt = torch.randint(0, 3, (2, 6, 6), generator=g)
        tgt[:, :2] = 255
        ours = LabelSmoothingCrossEntropy(smoothing=0.2)(pred, tgt)
        ref = F.cross_entropy(pred, tgt, ignore_index=255, label_smoothing=0.2)
        assert torch.allclose(ours, ref, atol=1e-5)


# ───────────────────────────── Few-shot ──────────────────────────────

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.advanced import FewShotLearner  # noqa: E402


def _tif(path, arr, nodata=None):
    arr = np.asarray(arr, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs="EPSG:32633", transform=from_origin(0, 100, 1, 1), nodata=nodata) as d:
        d.write(arr)
    return str(path)


def _colour_learner(**kw):
    """A learner whose 'backbone' is just the mean R, G, B: no download, deterministic."""
    learner = FewShotLearner(device="cpu", **kw)
    learner._embed = lambda imgs, batch_size=16: np.array([a.reshape(-1, 3).mean(0) for a in imgs])
    return learner


def _flat(path, rgb, size=32, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.stack([np.full((size, size), v, dtype="float32") for v in rgb]) + rng.normal(0, noise, (3, size, size))
    arr[:, 0, 0] = 0
    arr[:, 0, 1] = 255                       # keep the percentile stretch from flattening the image
    return _tif(path, arr)


class TestFewShot:
    def test_unsupported_method_and_metric(self):
        with pytest.raises(ValueError, match="not implemented"):
            FewShotLearner(method="maml")
        with pytest.raises(ValueError, match="metric"):
            FewShotLearner(metric="manhattan")
        with pytest.raises(ValueError, match="bands"):
            FewShotLearner(bands=(1, 2))

    def test_short_names_and_passthrough(self):
        assert FewShotLearner(backbone="dinov2-small").model_id == "facebook/dinov2-small"
        assert FewShotLearner(backbone="my-org/my-vit").model_id == "my-org/my-vit"

    def test_classifies_by_nearest_prototype(self, tmp_path):
        learner = _colour_learner()
        learner.fit_support({
            "red": [_flat(tmp_path / f"r{i}.tif", (200, 30, 30), noise=3, seed=i) for i in range(3)],
            "blue": [_flat(tmp_path / f"b{i}.tif", (30, 30, 200), noise=3, seed=i) for i in range(3)],
        })
        preds = learner.predict([_flat(tmp_path / "q1.tif", (190, 40, 35)), _flat(tmp_path / "q2.tif", (25, 35, 190))],
                                return_distances=True)
        assert [p["class"] for p in preds] == ["red", "blue"]
        assert all(abs(sum(p["probabilities"].values()) - 1) < 1e-3 for p in preds)
        assert set(preds[0]["distances"]) == {"red", "blue"}

    def test_feature_dimension_is_not_hard_coded(self, tmp_path):
        learner = _colour_learner()                              # 3 features; the old code assumed 768
        learner.fit_support({"a": [_flat(tmp_path / "a.tif", (200, 30, 30))], "b": [_flat(tmp_path / "b.tif", (30, 30, 200))]})
        bad = tmp_path / "broken.tif"
        bad.write_bytes(b"not a tiff")
        preds = learner.predict([str(bad), _flat(tmp_path / "q.tif", (200, 30, 30))])
        assert preds[0]["class"] is None and "error" in preds[0] and preds[0]["confidence"] == 0.0
        assert preds[1]["class"] == "a"

    def test_unreadable_support_images_are_skipped_not_zero_vectors(self, tmp_path):
        learner = _colour_learner()
        bad = tmp_path / "broken.tif"
        bad.write_bytes(b"x")
        learner.fit_support({"red": [_flat(tmp_path / "r.tif", (200, 30, 30)), str(bad)],
                             "blue": [_flat(tmp_path / "b.tif", (30, 30, 200))]})
        # the prototype is the one good image, not the mean of it and a zero vector
        np.testing.assert_allclose(learner._prototypes["red"], learner._embed([learner._read_rgb(str(tmp_path / "r.tif"))])[0])

    def test_support_validation(self, tmp_path):
        learner = _colour_learner()
        good = _flat(tmp_path / "g.tif", (1, 2, 3))
        bad = tmp_path / "broken.tif"
        bad.write_bytes(b"x")
        with pytest.raises(ValueError, match="at least two"):
            learner.fit_support({"only": [good]})
        with pytest.raises(ValueError, match="none of its"):
            learner.fit_support({"a": [str(bad)], "b": [good]})
        with pytest.raises(ValueError, match="non-empty"):
            learner.fit_support({"a": [], "b": [good]})
        with pytest.raises(RuntimeError, match="fit_support"):
            learner.predict([good])

    def test_cosine_metric_separates_directions(self, tmp_path):
        learner = _colour_learner(metric="cosine")
        learner.fit_support({"red": [_flat(tmp_path / "r.tif", (200, 30, 30))], "blue": [_flat(tmp_path / "b.tif", (30, 30, 200))]})
        p = learner.predict([_flat(tmp_path / "q.tif", (180, 40, 40))])[0]
        assert p["class"] == "red" and p["confidence"] > 0.9

    def test_joint_stretch_keeps_colour_per_band_stretch_removes_it(self):
        data = np.stack([np.full((20, 20), v, dtype="float64") for v in (200, 30, 30)])
        data[:, 0, :] = 0                                      # some dark pixels so the percentiles span a range
        joint = FewShotLearner(stretch="joint")._to_rgb(data)
        per_band = FewShotLearner(stretch="per_band")._to_rgb(data)
        r, g = joint[10, 10, 0], joint[10, 10, 1]
        assert r > 200 and g < 60                              # still a red picture
        assert abs(int(per_band[10, 10, 0]) - int(per_band[10, 10, 1])) < 5   # every band stretched alike: grey

    def test_single_band_and_two_band_files_work(self, tmp_path):
        learner = _colour_learner()
        one = _tif(tmp_path / "one.tif", np.random.default_rng(0).random((1, 16, 16)) * 100)
        two = _tif(tmp_path / "two.tif", np.random.default_rng(1).random((2, 16, 16)) * 100)
        assert learner._read_rgb(one).shape == (16, 16, 3) and learner._read_rgb(two).shape == (16, 16, 3)

    def test_geotiff_map_follows_the_scene(self, tmp_path):
        learner = _colour_learner()
        learner.fit_support({"red": [_flat(tmp_path / "r.tif", (200, 30, 30))], "blue": [_flat(tmp_path / "b.tif", (30, 30, 200))]})
        scene = np.zeros((3, 64, 64), dtype="float32")
        scene[0, :, :32], scene[2, :, 32:] = 200, 200               # left half red, right half blue
        scene[:, 0, 0] = 0
        scene[:, 0, 1] = 255
        r = learner.predict_geotiff(_tif(tmp_path / "scene.tif", scene), str(tmp_path / "out.tif"),
                                    chip_size=16, stride=8)
        assert r["success"], r
        assert r["class_map"] == {"0": "blue", "1": "red"}
        with rasterio.open(tmp_path / "out.tif") as o:
            lab = o.read(1)
            assert o.dtypes == ("uint8",) and (o.height, o.width) == (64, 64) and "class_map" in o.tags()
        assert (lab[8:56, 2:14] == 1).all() and (lab[8:56, 50:62] == 0).all()      # red left, blue right

    def test_geotiff_nodata_chips_are_skipped_and_left_255(self, tmp_path):
        learner = _colour_learner()
        learner.fit_support({"red": [_flat(tmp_path / "r.tif", (200, 30, 30))], "blue": [_flat(tmp_path / "b.tif", (30, 30, 200))]})
        scene = np.full((3, 64, 64), 100.0, dtype="float32")
        scene[:, :, 32:] = -9999
        r = learner.predict_geotiff(_tif(tmp_path / "scene.tif", scene, nodata=-9999), str(tmp_path / "out.tif"),
                                    chip_size=16, stride=16)
        assert r["success"] and r["n_skipped_chips"] > 0
        with rasterio.open(tmp_path / "out.tif") as o:
            lab = o.read(1)
        assert (lab[:, 40:] == 255).all() and (lab[:, :32] != 255).all()

    def test_geotiff_edges_are_covered_when_size_is_not_a_multiple(self, tmp_path):
        learner = _colour_learner()
        learner.fit_support({"a": [_flat(tmp_path / "a.tif", (200, 30, 30))], "b": [_flat(tmp_path / "b.tif", (30, 30, 200))]})
        scene = np.random.default_rng(0).random((3, 50, 70)).astype("float32") * 255
        assert learner.predict_geotiff(_tif(tmp_path / "s.tif", scene), str(tmp_path / "o.tif"), chip_size=32, stride=32)["success"]
        with rasterio.open(tmp_path / "o.tif") as o:
            assert (o.read(1) != 255).all()                 # nothing left uncovered

    def test_geotiff_requires_fit_and_reports_bad_files(self, tmp_path):
        learner = _colour_learner()
        assert learner.predict_geotiff("x.tif", "o.tif")["success"] is False
        learner.fit_support({"a": [_flat(tmp_path / "a.tif", (200, 30, 30))], "b": [_flat(tmp_path / "b.tif", (30, 30, 200))]})
        r = learner.predict_geotiff(str(tmp_path / "nope.tif"), str(tmp_path / "o.tif"))
        assert r["success"] is False and r["error"]


class TestFewShotWithARealTransformerBackbone:
    def test_embedding_pipeline_with_a_tiny_random_dinov2(self):
        transformers = pytest.importorskip("transformers")
        cfg = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                        image_size=56, patch_size=14, intermediate_size=64)
        learner = FewShotLearner(device="cpu")
        learner._feature_extractor = transformers.Dinov2Model(cfg).eval()
        learner._processor = transformers.BitImageProcessor(size={"shortest_edge": 56},
                                                            crop_size={"height": 56, "width": 56})
        feats = learner._embed([np.random.randint(0, 255, (70, 90, 3), dtype=np.uint8) for _ in range(5)], batch_size=2)
        assert feats.shape == (5, 32)                          # d comes from the model, not a constant 768


class TestFewShotTemperature:
    def _fit(self, scale):
        learner = FewShotLearner(device="cpu")
        feats = {"a": np.array([0.0, 0.0]), "b": np.array([scale, 0.0]), "c": np.array([0.0, scale])}
        learner._extract_features = lambda paths: ([feats[p] for p in paths], [None] * len(paths))
        learner.fit_support({k: [k] for k in feats})
        return learner

    def test_scores_are_graded_for_feature_spaces_of_any_scale(self):
        """With a fixed temperature of 1 a feature space whose classes are ~40 apart gave 0.0/1.0 for everything."""
        for scale in (0.5, 40.0):
            learner = self._fit(scale)
            _, probs = learner._scores(np.array([0.3 * scale, 0.0]))          # 30 % of the way from a to b
            assert 0.55 < probs["a"] < 0.95 and probs["b"] > 0.02, (scale, probs)

    def test_user_temperature_is_kept(self):
        learner = FewShotLearner(device="cpu", temperature=2.0)
        learner._extract_features = lambda paths: ([np.array([0.0, 0.0]) if p == "a" else np.array([9.0, 0.0]) for p in paths], [None] * len(paths))
        learner.fit_support({"a": ["a"], "b": ["b"]})
        assert learner.temperature == 2.0


class TestFewShotValueRange:
    def test_percentile_stretch_loses_brightness_value_range_keeps_it(self):
        dim = np.full((3, 8, 8), 100.0)
        dim[:, 0, 0] = 110.0                                  # a hair of variation so percentiles exist
        bright = dim * 5
        pct = FewShotLearner(device="cpu")
        assert (pct._to_rgb(dim) == pct._to_rgb(bright)).all()    # relative scaling: same picture
        fixed = FewShotLearner(device="cpu", value_range=(0, 1000))
        a, b = fixed._to_rgb(dim), fixed._to_rgb(bright)
        assert a.mean() < b.mean() and a.max() <= 28 and b.max() > 125

    def test_nodata_is_black_and_bad_range_rejected(self):
        fixed = FewShotLearner(device="cpu", value_range=(0, 1000))
        d = np.full((3, 4, 4), 500.0)
        d[:, 0, :] = -9999
        out = fixed._to_rgb(d, nodata=-9999)
        assert (out[0] == 0).all() and out[1:].min() > 100
        with pytest.raises(ValueError, match="value_range"):
            FewShotLearner(device="cpu", value_range=(5, 5))
