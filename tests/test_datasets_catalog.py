"""Catalogue integrity, filters and helpers of ``pygeovision.datasets``."""
from __future__ import annotations

import json
from collections import Counter

import pytest
from click.testing import CliRunner

from pygeovision.datasets.analysis import DatasetAnalyzer
from pygeovision.datasets.benchmark import BenchmarkBuilder, BenchmarkConfig
from pygeovision.datasets.loader import DatasetLoader
from pygeovision.datasets.registry import _build_catalog, _raw_catalog, dataset_registry as reg


class TestCatalogue:
    def test_one_entry_per_name(self):
        names = [d.name for d in reg.all()]
        assert len(names) == len(set(names)) == len(reg)

    def test_raw_catalogue_had_duplicates_that_are_merged(self):
        raw = Counter(d.name for d in _raw_catalog())
        dup = [n for n, k in raw.items() if k > 1]
        assert dup, "if the source list has no duplicates any more this test can go"
        built = {d.name: d for d in _build_catalog()}
        first = {}
        for d in _raw_catalog():
            first.setdefault(d.name, d)
        for n in dup:
            assert built[n].n_samples == first[n].n_samples          # first entry wins
        # tasks are the union, so a dataset stays findable under every task it was listed for
        assert "time_series" in built["CropHarvest"].tasks and "classification" in built["CropHarvest"].tasks
        # empty links are filled from later entries
        assert built["CropHarvest"].download_url.startswith("https://")

    def test_licence_is_not_invented(self):
        assert {d.license for d in reg.all()} == {"unknown"}

    def test_every_entry_has_a_task_and_positive_size(self):
        for d in reg.all():
            assert d.tasks and d.n_samples > 0 and d.volume_gb > 0 and d.resolution_m > 0

    def test_summary_counts_each_dataset_once(self):
        s = reg.summary()
        assert s["total_datasets"] == len(reg)
        assert sum(s["domains"].values()) == len(reg)
        assert sum(s["modalities"].values()) == len(reg)

    def test_unknown_name_suggests_close_match(self):
        with pytest.raises(KeyError, match="LoveDA"):
            reg["LoveDa"]


class TestFilters:
    def test_zero_is_a_real_bound(self):
        assert reg.filter(max_volume_gb=0.0) == []                      # was "no filter" before
        assert len(reg.filter(max_volume_gb=0.5)) < len(reg)

    def test_conditions_combine(self):
        out = reg.filter(task="segmentation", modality="sar")
        assert out and all("segmentation" in d.tasks and d.modality == "sar" for d in out)

    def test_resolution_bounds_are_ground_sampling_distance(self):
        fine = reg.filter(max_resolution_m=0.5)
        assert fine and all(d.resolution_m <= 0.5 for d in fine)

    def test_top_for_task_modality(self):
        top = reg.top_for_task("segmentation", n=4, modality="rgb")
        assert len(top) == 4 and all(d.modality == "rgb" for d in top)
        assert reg.top_for_task("segmentation", n=3, modality="no-such") == []

    def test_search_is_case_insensitive(self):
        assert reg.search("LOVEDA") and reg.search("loveda")

    def test_print_table_empty_list_prints_no_rows(self, capsys):
        reg.print_table([])
        out = capsys.readouterr().out
        assert "0 shown" in out and "EuroSAT" not in out


class TestLoader:
    def test_download_never_downloads_and_says_why(self):
        with pytest.raises(ValueError, match="no automated download"):
            DatasetLoader().download("EuroSAT")

    def test_download_message_names_the_page_when_known(self):
        with pytest.raises(ValueError, match="github.com/phelber/EuroSAT"):
            DatasetLoader().download("EuroSAT")

    def test_download_without_url_does_not_print_empty_quotes(self):
        name = next(d.name for d in reg.all() if not d.download_url and not d.paper_url)
        with pytest.raises(ValueError) as e:
            DatasetLoader().download(name)
        assert "''" not in str(e.value) and name in str(e.value)

    def test_unknown_name_is_a_key_error(self):
        with pytest.raises(KeyError):
            DatasetLoader().download("nope")


class TestAnalysis:
    def test_bars_are_capped(self, capsys):
        DatasetAnalyzer(reg).modality_distribution(show=True)
        longest = max(len(line) for line in capsys.readouterr().out.splitlines())
        assert longest < 90

    def test_resolution_bins_cover_everything(self):
        assert sum(DatasetAnalyzer(reg).resolution_distribution(show=False).values()) == len(reg)

    def test_correlation_matrix_is_symmetric_with_unit_diagonal(self):
        m = DatasetAnalyzer(reg).correlation_matrix(["EuroSAT", "LoveDA", "DOTA"], show=False)
        assert all(m[i][i] == 1.0 for i in range(3))
        assert all(abs(m[i][j] - m[j][i]) < 1e-12 for i in range(3) for j in range(3))

    def test_cluster_returns_embedding_or_none(self, tmp_path):
        pytest.importorskip("sklearn")
        out = tmp_path / "c.png"
        emb = DatasetAnalyzer(reg).cluster("tsne", save_path=str(out))
        assert emb.shape == (len(reg), 2) and out.exists()


class TestBenchmarkConfig:
    def test_split_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            BenchmarkConfig("segmentation", ["x"], split_ratios={"train": 0.7, "val": 0.2, "test": 0.2})
        with pytest.raises(ValueError):
            BenchmarkConfig("segmentation", ["x"], split_ratios={"train": 0.8, "test": 0.2})

    def test_builder_writes_config_only(self, tmp_path):
        paths = BenchmarkBuilder().save_all(str(tmp_path), n=3)
        assert paths and all(p.endswith(".json") for p in paths)
        cfg = json.loads(open(paths[0]).read())
        assert cfg["split_ratios"] == {"train": 0.6, "val": 0.2, "test": 0.2} and len(cfg["dataset_names"]) == 3

    def test_unknown_task_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="segmentation"):
            BenchmarkBuilder().build("not-a-task")


class TestCLI:
    def _run(self, *args):
        from pygeovision.cli.main import cli
        return CliRunner().invoke(cli, list(args))

    def test_datasets_top_modality(self):
        r = self._run("datasets", "top", "segmentation", "--n", "3", "--modality", "sar")
        assert r.exit_code == 0 and "sar" in r.output

    def test_benchmark_models_without_weights_explains(self):
        r = self._run("benchmark", "models", "change_detection")
        assert r.exit_code == 0 and "without them" in r.output

    def test_datasets_download_exits_1(self):
        r = self._run("datasets", "download", "LoveDA")
        assert r.exit_code == 1 and "Error:" in r.output

    def test_datasets_info_unknown_suggests(self):
        r = self._run("datasets", "info", "LoveDa")
        assert r.exit_code != 0 and "LoveDA" in r.output
