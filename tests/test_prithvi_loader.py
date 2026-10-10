"""The real Prithvi loading path, with a stand-in ``prithvi_mae.py`` and a fake Hub (no network).

What is verified here is *our* code: which files are fetched at which revision, how the
checkpoint is checked against the architecture, how the input is standardised and laid out, and
that nothing is quietly randomly initialised.  Whether the real ``prithvi_mae.py`` has exactly
the stand-in's constructor/forward signature was checked by reading the published file, not by
running it -- that needs the Hub.
"""
import json
import textwrap

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pygeovision.models.foundation import prithvi as P  # noqa: E402

STAND_IN_MAE = textwrap.dedent('''
    import torch
    import torch.nn as nn

    LAST_INPUT = {}


    class _PatchEmbed(nn.Module):
        def __init__(self, in_chans, embed_dim, patch):
            super().__init__()
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch, stride=patch)


    class PrithviViT(nn.Module):
        def __init__(self, img_size=224, patch_size=(1, 16, 16), num_frames=1, in_chans=3, embed_dim=32,
                     depth=1, num_heads=4, **kw):
            super().__init__()
            self.patch_embed = _PatchEmbed(in_chans, embed_dim, tuple(patch_size))
            n = num_frames * (img_size // patch_size[1]) * (img_size // patch_size[2])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.register_buffer("pos_embed", torch.zeros(1, n + 1, embed_dim))   # fixed, frames-dependent
            self.blocks = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)

        def forward_features(self, x, temporal_coords=None, location_coords=None):
            LAST_INPUT["x"] = x.detach().clone()
            p = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
            tokens = torch.cat([self.cls_token.expand(p.shape[0], -1, -1), p], dim=1) + self.pos_embed
            for b in self.blocks:
                tokens = b(tokens)
            return [self.norm(tokens)]


    class PrithviMAE(nn.Module):
        def __init__(self, img_size=224, patch_size=(1, 16, 16), num_frames=4, in_chans=6, embed_dim=32,
                     depth=1, num_heads=4, decoder_embed_dim=16, **kw):
            super().__init__()
            self.encoder = PrithviViT(img_size, patch_size, num_frames, in_chans, embed_dim, depth, num_heads)
            self.decoder = nn.Linear(embed_dim, decoder_embed_dim)
''')

MEAN = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
STD = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
CFG = {"architecture": "stand_in", "pretrained_cfg": {
    "img_size": 32, "num_frames": 4, "patch_size": [1, 16, 16], "in_chans": 6, "embed_dim": 32,
    "depth": 2, "num_heads": 4, "decoder_embed_dim": 16, "mean": MEAN, "std": STD,
    "bands": ["B02", "B03", "B04", "B05", "B06", "B07"]}}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub repository + a hf_hub_download that serves it and records every call."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "prithvi_mae.py").write_text(STAND_IN_MAE)
    (repo / "config.json").write_text(json.dumps(CFG))
    # build the checkpoint from the stand-in itself (4 frames, like the real pretraining)
    ns: dict = {}
    exec(STAND_IN_MAE, ns)
    torch.manual_seed(0)
    mae = ns["PrithviMAE"](**CFG["pretrained_cfg"])
    torch.save(mae.state_dict(), repo / "Prithvi_EO_V2_300M.pt")
    torch.save(mae.state_dict(), repo / "Prithvi_EO_V1_100M.pt")

    calls: list = []

    def fake_download(repo_id, filename, revision=None, **kw):
        calls.append((repo_id, filename, revision))
        return str(repo / filename)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return type("Hub", (), {"dir": repo, "calls": calls, "state": mae.state_dict()})


class TestLoadFromHub:
    def test_loads_the_checkpoint_weights_and_keeps_the_encoder_only(self, hub):
        m = P.load_prithvi_hf("prithvi_eo_2_0", num_frames=1)
        assert isinstance(m, _bb())
        assert not getattr(m, "_pygeovision_is_random_init", False)
        for k, v in m.encoder.state_dict().items():
            if "pos_embed" not in k:
                torch.testing.assert_close(v, hub.state[f"encoder.{k}"])
        assert not any("decoder" in k for k in m.state_dict())

    def test_three_files_are_fetched_at_the_pinned_revision(self, hub):
        P.load_prithvi_hf("prithvi_eo_2_0")
        spec = P.PRITHVI_MODELS["prithvi_eo_2_0"]
        assert {c[1] for c in hub.calls} == {"config.json", "prithvi_mae.py", "Prithvi_EO_V2_300M.pt"}
        assert {c[2] for c in hub.calls} == {spec["revision"]}
        assert len(spec["revision"]) == 40 and all(c[0] == spec["hf_id"] for c in hub.calls)

    def test_revision_can_be_overridden(self, hub):
        P.load_prithvi_hf("prithvi_eo_2_0", revision="abc123")
        assert {c[2] for c in hub.calls} == {"abc123"}

    def test_positional_embedding_is_rebuilt_for_the_requested_frames(self, hub):
        """The checkpoint holds pos_embed for 4 frames; loading it into a 1-frame model would
        be a size mismatch -- the official script discards it, and so do we."""
        m = P.load_prithvi_hf("prithvi_eo_2_0", num_frames=1)
        assert m.encoder.pos_embed.shape[1] == 1 + 1 * 2 * 2
        assert m.num_frames == 1

    def test_checkpoint_that_does_not_fill_the_architecture_is_refused(self, hub):
        state = dict(torch.load(hub.dir / "Prithvi_EO_V2_300M.pt", weights_only=True))
        del state["encoder.blocks.1.weight"]            # one encoder layer missing
        torch.save(state, hub.dir / "Prithvi_EO_V2_300M.pt")
        with pytest.raises(P.PrithviLoadError, match="randomly initialised"):
            P.load_prithvi_hf("prithvi_eo_2_0")

    def test_extra_checkpoint_keys_only_warn(self, hub, caplog):
        state = dict(torch.load(hub.dir / "Prithvi_EO_V2_300M.pt", weights_only=True))
        state["encoder.something_new"] = torch.zeros(1)
        torch.save(state, hub.dir / "Prithvi_EO_V2_300M.pt")
        with caplog.at_level("WARNING"):
            P.load_prithvi_hf("prithvi_eo_2_0")
        assert "not used" in caplog.text

    def test_hub_failure_raises_and_is_a_runtime_error(self, monkeypatch):
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
        with pytest.raises(RuntimeError, match="Nothing was substituted"):
            P.load_prithvi_hf("prithvi_eo_2_0")

    def test_allow_random_init_returns_a_flagged_stand_in(self, monkeypatch):
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
        m = P.load_prithvi_hf("prithvi_eo_2_0", allow_random_init=True)
        assert m._pygeovision_is_random_init is True

    @pytest.mark.parametrize("fn", [P.load_prithvi_hf, lambda n: P.load_prithvi_local(n, "x.pt")])
    def test_fine_tuned_downstream_checkpoints_are_refused(self, fn):
        with pytest.raises(NotImplementedError, match="TerraTorch"):
            fn("prithvi_eo_1_0_finetuned_burn")

    def test_every_loadable_spec_names_a_checkpoint_and_a_pinned_commit(self):
        for name, spec in P.PRITHVI_MODELS.items():
            if spec.get("loadable", True):
                assert spec["checkpoint"].endswith(".pt") and len(spec["revision"]) == 40, name

    def test_no_known_nonexistent_repo_ids(self):
        ids = {s["hf_id"] for s in P.PRITHVI_MODELS.values()}
        assert not ids & {"ibm-nasa-geospatial/Prithvi-100M", "ibm-nasa-geospatial/Prithvi-100M-burn-scar",
                          "ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification"}
        # the flood entry used to point at the crop-classification repo
        assert P.PRITHVI_MODELS["prithvi_eo_1_0_finetuned_flood"]["hf_id"].endswith("sen1floods11")


class TestLoadLocal:
    def test_loads_next_to_the_weights(self, hub):
        m = P.load_prithvi_local("prithvi_eo_2_0", str(hub.dir / "Prithvi_EO_V2_300M.pt"))
        assert isinstance(m, _bb())

    def test_missing_companion_files_are_named(self, hub, tmp_path):
        lone = tmp_path / "lone"
        lone.mkdir()
        (lone / "w.pt").write_bytes((hub.dir / "Prithvi_EO_V2_300M.pt").read_bytes())
        with pytest.raises(FileNotFoundError, match="config.json"):
            P.load_prithvi_local("prithvi_eo_2_0", str(lone / "w.pt"))

    def test_real_weights_are_not_poured_into_a_different_network(self, hub):
        """The old loader built a different architecture and used strict=False: every key was
        'missing', a warning was logged and a random model came back."""
        state = {f"other.{k}": v for k, v in hub.state.items()}
        torch.save(state, hub.dir / "Prithvi_EO_V2_300M.pt")
        with pytest.raises(P.PrithviLoadError):
            P.load_prithvi_local("prithvi_eo_2_0", str(hub.dir / "Prithvi_EO_V2_300M.pt"))


def _bb():
    return P._backbone_classes()[0]


class TestInputHandling:
    def _model(self, hub, frames=1):
        return P.load_prithvi_hf("prithvi_eo_2_0", num_frames=frames).eval()

    def test_reflectance_is_standardised_with_the_models_own_statistics(self, hub):
        m = self._model(hub)
        refl = torch.zeros(1, 6, 32, 32)
        refl[:] = torch.tensor([0.10, 0.12, 0.15, 0.20, 0.25, 0.30]).view(1, 6, 1, 1)
        m(pixel_values=refl)
        seen = _stand_in_module(hub).LAST_INPUT["x"]              # (B, C, T, H, W)
        dn = np.array([1000, 1200, 1500, 2000, 2500, 3000], dtype="float32")
        expected = (dn - np.array(MEAN)) / np.array(STD)
        np.testing.assert_allclose(seen[0, :, 0, 0, 0].numpy(), expected, rtol=1e-5)

    def test_dividing_by_10000_alone_is_not_what_the_model_sees(self, hub):
        m = self._model(hub)
        m(pixel_values=torch.full((1, 6, 32, 32), 0.1))
        seen = _stand_in_module(hub).LAST_INPUT["x"]
        assert not torch.allclose(seen, torch.full_like(seen, 0.1), atol=0.05)

    def test_time_and_channel_axes_are_not_swapped(self, hub):
        m = self._model(hub, frames=2)
        x = torch.zeros(1, 2, 6, 32, 32)                        # (B, T, C, H, W)
        for t in range(2):
            for c in range(6):
                x[0, t, c] = 0.01 * (1 + t * 6 + c)
        m(pixel_values=x)
        seen = _stand_in_module(hub).LAST_INPUT["x"]               # (B, C, T, H, W)
        assert seen.shape == (1, 6, 2, 32, 32)
        for t in range(2):
            for c in range(6):
                dn = 0.01 * (1 + t * 6 + c) * 10000
                assert seen[0, c, t, 0, 0].item() == pytest.approx((dn - MEAN[c]) / STD[c], rel=1e-4)

    def test_wrong_frames_bands_or_tile_are_rejected_clearly(self, hub):
        m = self._model(hub, frames=1)
        with pytest.raises(ValueError, match="num_frames=2"):
            m(pixel_values=torch.zeros(1, 2, 6, 32, 32))
        with pytest.raises(ValueError, match="6 bands"):
            m(pixel_values=torch.zeros(1, 4, 32, 32))
        with pytest.raises(ValueError, match="32x32"):
            m(pixel_values=torch.zeros(1, 6, 64, 64))

    def test_output_structure_and_token_order(self, hub):
        class Counting(torch.nn.Module):
            """Encoder whose token i has value i, to pin the (t, h, w) ordering assumption."""
            def forward_features(self, x, *a):
                b, c, t, h, w = x.shape
                n = 1 + t * (h // 16) * (w // 16)
                return [torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(b, n, 8)]

        m = _bb()(Counting(), embed_dim=8, patch_size=(1, 16, 16), img_size=32, num_frames=2,
                  n_bands=6, mean=MEAN, std=STD)
        out = m(pixel_values=torch.zeros(1, 2, 6, 32, 32))
        assert out.grid == (2, 2) and out.frame_tokens.shape == (1, 2, 4, 8)
        assert out.cls.shape == (1, 8) and out.cls[0, 0] == 0
        assert out.frame_tokens[0, 1, 0, 0] == 1 + 4        # first patch of the second date
        assert out.frame_tokens[0, 0, 3, 0] == 1 + 3

    def test_standardise_hls_function(self):
        refl = np.full((6, 2, 2), 0.1, dtype="float32")
        z = P.standardise_hls(refl, MEAN, STD)
        np.testing.assert_allclose(z[:, 0, 0], (1000 - np.array(MEAN)) / np.array(STD), rtol=1e-5)


def _stand_in_module(hub):
    """The imported stand-in prithvi_mae module for this test's fake repo."""
    import sys
    return sys.modules[f"_pygeovision_prithvi_mae_{abs(hash(str((hub.dir / 'prithvi_mae.py').resolve())))}"]


def _write_tif(path, n=6, size=64, seed=0):
    import rasterio
    from rasterio.transform import from_bounds
    data = (np.random.default_rng(seed).random((n, size, size)) * 6000 + 200).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=n, dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(0, 0, 1, 1, size, size)) as dst:
        dst.write(data)
    return str(path)


class TestPrithviClassOnTheRealLoadPath:
    def test_features_use_the_models_tile_size_and_pooling(self, hub, tmp_path):
        p = P.Prithvi("prithvi_eo_2_0", device="cpu").load()
        tif = _write_tif(tmp_path / "a.tif")
        cls = p.extract_features(tif, source="hls", pool="cls")
        mean = p.extract_features(tif, source="hls", pool="mean")
        assert cls.shape == mean.shape == (1, 32)
        assert not np.allclose(cls, mean)
        assert p.extract_patch_features(tif).shape == (4, 32)       # 2x2 grid of 16-px patches
        with pytest.raises(ValueError):
            p.extract_features(tif, pool="max")

    def test_multitemporal_reloads_for_the_number_of_dates(self, hub, tmp_path):
        mt = P.PrithviMultiTemporal("prithvi_eo_2_0")
        mt._prithvi.load()
        assert mt._prithvi._model.num_frames == 1
        paths = [_write_tif(tmp_path / f"t{i}.tif", seed=i) for i in range(3)]
        r = mt.process_time_series(paths, dates=["a", "b", "c"])
        assert mt._prithvi._model.num_frames == 3
        assert r["frame_embeddings"].shape == (3, 32) and r["frame_tokens"].shape == (3, 4, 32)
        assert r["features"].shape == (1 + 3 * 4, 32) and r["grid"] == (2, 2)

    def test_detect_change_is_zero_for_identical_dates_and_localised_otherwise(self, hub, tmp_path):
        mt = P.PrithviMultiTemporal("prithvi_eo_2_0")
        a = _write_tif(tmp_path / "a.tif", seed=1)
        same = mt.detect_change(a, a, output_path=str(tmp_path / "same.tif"))
        assert same["raw_distance"].max() < 1e-4 and same["change_pct"] == 0.0
        assert (tmp_path / "same.tif").exists()
        diff = mt.detect_change(a, _write_tif(tmp_path / "b.tif", seed=2))
        assert diff["raw_distance"].max() > 1e-3 and diff["change_map"].max() == pytest.approx(1.0)
        assert diff["change_map"].shape == (2, 2)

    def test_trend_and_seasonal_are_statistics_of_the_embeddings(self, hub, tmp_path):
        mt = P.PrithviMultiTemporal("prithvi_eo_2_0")
        paths = [_write_tif(tmp_path / f"s{i}.tif", seed=i) for i in range(4)]
        tr = mt.monitor_trend(paths)
        assert tr["trend_direction"] in ("diverging", "converging", "stable")
        assert len(tr["drift_from_first"]) == 4 and tr["drift_from_first"][0] == 0.0
        se = mt.predict_seasonal(paths, dates=list("abcd"))
        assert se["seasonal_amplitude"] >= 0 and se["peak_date"] in "abcd" and se["trough_date"] in "abcd"
        assert "not a physical seasonal model" in se["note"]

    def test_segmentation_head_output_is_input_resolution_and_backbone_is_frozen(self, hub):
        p = P.Prithvi("prithvi_eo_2_0").load()
        seg = p.build_segmentation_head(num_classes=5)
        x = torch.rand(2, 6, 32, 32)
        assert seg(x).shape == (2, 5, 32, 32)
        assert all(not q.requires_grad for q in seg.backbone.parameters())
        seg.train()
        assert seg.backbone.training is False and seg.head.training is True
        loss = torch.nn.functional.cross_entropy(seg(x), torch.randint(0, 5, (2, 32, 32)))
        loss.backward()
        assert any(q.grad is not None for q in seg.head.parameters())
        assert all(q.grad is None for q in seg.backbone.parameters())


class TestTasksNeverInventAResult:
    def test_trained_head_gives_a_backbone_prediction_and_wrong_heads_are_refused(self, hub, tmp_path):
        tif = _write_tif(tmp_path / "scene.tif", size=64)
        p = P.Prithvi("prithvi_eo_2_0").load()
        seg = p.build_segmentation_head(num_classes=9)
        good = P.save_prithvi_head(seg, str(tmp_path / "lc.pt"), "prithvi_eo_2_0", "land_cover", 9)
        res = P.PrithviTasks("prithvi_eo_2_0", head_weights={"land_cover": good}).land_cover(tif, source="hls")
        assert res["method"] == "backbone" and res["prediction"].shape == (64, 64)
        assert res["prediction"].max() <= 8

        for variant, task, n in (("prithvi_eo_1_0", "land_cover", 9), ("prithvi_eo_2_0", "flood", 9),
                                 ("prithvi_eo_2_0", "land_cover", 4)):
            bad = P.save_prithvi_head(seg, str(tmp_path / f"bad_{variant}_{task}_{n}.pt"), variant, task, n)
            with pytest.raises(ValueError):
                P.PrithviTasks("prithvi_eo_2_0", head_weights={"land_cover": bad}).land_cover(tif, source="hls")

    def test_spectral_route_does_not_touch_the_model(self, tmp_path, monkeypatch):
        tif = _write_tif(tmp_path / "scene.tif", size=32)
        tasks = P.PrithviTasks("prithvi_eo_2_0")
        monkeypatch.setattr(tasks._prithvi, "_ensure_loaded",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not load")))
        for fn in (tasks.land_cover, tasks.flood_detection, tasks.burn_scar_detection):
            r = fn(tif, source="hls")
            assert r["method"] == "spectral" and "Prithvi was not used" in r["note"]

    def test_biomass_with_a_trained_head(self, hub, tmp_path):
        tif = _write_tif(tmp_path / "scene.tif", size=32)
        head = P.build_regression_head(32)
        f = P.save_prithvi_head(head, str(tmp_path / "bio.pt"), "prithvi_eo_2_0", "biomass", 1)
        r = P.PrithviTasks("prithvi_eo_2_0", head_weights={"biomass": f}).biomass_estimation(tif, source="hls")
        assert isinstance(r["estimated_biomass_t_ha"], float) and r["method"] == "backbone"

    def test_finetune_components_say_they_do_not_train(self, hub, caplog, tmp_path):
        with caplog.at_level("WARNING"):
            r = P.finetune_prithvi("prithvi_eo_2_0", dataset=object(), num_classes=3, epochs=2,
                               output_dir=str(tmp_path))
        assert "not used" in caplog.text and "no training has been run" in r["note"]
        assert all(q.requires_grad for q in r["model"].backbone.parameters())


class TestBandMapping:
    def test_narrow_nir_b8a_is_preferred_over_b08(self):
        data = np.zeros((7, 2, 2), dtype="float32")
        for i in range(7):
            data[i] = i + 1
        names = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
        out = P.map_bands(data, source="sentinel2", n_prithvi_bands=6, source_bands=names)
        assert out[3, 0, 0] == 5            # B8A (5th input), not B08 (4th)

    def test_b08_alone_is_used_with_a_warning(self, caplog):
        data = np.arange(6, dtype="float32").reshape(6, 1, 1) * np.ones((6, 2, 2), "float32")
        with caplog.at_level("WARNING"):
            out = P.map_bands(data, source="sentinel2", n_prithvi_bands=6,
                              source_bands=["B02", "B03", "B04", "B08", "B11", "B12"])
        assert out[3, 0, 0] == 3 and "B08" in caplog.text
