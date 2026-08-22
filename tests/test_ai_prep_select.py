"""Tests for pygeovision.data.ai_prep.EOPipeline.select().

Real bug found and fixed: `strategy` was a documented parameter
("best"/positional selection) but the implementation was
`self._scenes[:n]` — a naive positional slice completely ignoring
whatever strategy was requested. A caller doing
`select(n=1, strategy="best")` expecting the lowest-cloud-cover scene
just got whichever scene happened to be first in the search results.
"""
import pytest

from pygeovision.data.ai_prep import EOPipeline
from pygeovision.data.fetch import SearchResult


def _make_scenes():
    return [
        SearchResult(id="A", provider="p", satellite="sentinel-2",
                     datetime="2024-03-15T10:00:00", cloud_cover=45.0, bbox=None, score=0.5),
        SearchResult(id="B", provider="p", satellite="sentinel-2",
                     datetime="2024-06-01T10:00:00", cloud_cover=2.0, bbox=None, score=0.9),
        SearchResult(id="C", provider="p", satellite="sentinel-2",
                     datetime="2024-01-01T10:00:00", cloud_cover=15.0, bbox=None, score=0.7),
    ]


def _pipeline_with(scenes):
    pipe = EOPipeline.__new__(EOPipeline)
    pipe._scenes = list(scenes)
    return pipe


class TestSelectStrategies:
    def test_best_picks_lowest_cloud_cover(self):
        pipe = _pipeline_with(_make_scenes())
        pipe.select(n=1, strategy="best")
        assert pipe._scenes[0].id == "B"  # cloud_cover=2.0, the lowest

    def test_newest_picks_most_recent_date(self):
        pipe = _pipeline_with(_make_scenes())
        pipe.select(n=1, strategy="newest")
        assert pipe._scenes[0].id == "B"  # 2024-06-01

    def test_oldest_picks_least_recent_date(self):
        pipe = _pipeline_with(_make_scenes())
        pipe.select(n=1, strategy="oldest")
        assert pipe._scenes[0].id == "C"  # 2024-01-01

    def test_score_picks_highest_relevance_score(self):
        pipe = _pipeline_with(_make_scenes())
        pipe.select(n=1, strategy="score")
        assert pipe._scenes[0].id == "B"  # score=0.9, the highest

    def test_best_with_n_2_returns_two_lowest_cloud_cover_in_order(self):
        pipe = _pipeline_with(_make_scenes())
        pipe.select(n=2, strategy="best")
        assert [s.id for s in pipe._scenes] == ["B", "C"]

    def test_none_cloud_cover_sorts_last_for_best_strategy(self):
        scenes = _make_scenes()
        scenes.append(SearchResult(id="D", provider="p", satellite="sentinel-1",
                                    datetime="2024-05-01T10:00:00", cloud_cover=None, bbox=None))
        pipe = _pipeline_with(scenes)
        pipe.select(n=4, strategy="best")
        assert pipe._scenes[-1].id == "D", "scene with unknown cloud cover should sort last, not first/arbitrary"

    def test_unrecognised_strategy_keeps_search_order_with_warning(self, caplog):
        pipe = _pipeline_with(_make_scenes())
        pipe.select(n=1, strategy="bogus_strategy")
        assert pipe._scenes[0].id == "A"  # unchanged search-result order
        assert "unrecognised strategy" in caplog.text.lower()

    def test_as_is_explicitly_keeps_search_order(self):
        pipe = _pipeline_with(_make_scenes())
        pipe.select(n=2, strategy="as_is")
        assert [s.id for s in pipe._scenes] == ["A", "B"]

    def test_returns_self_for_chaining(self):
        pipe = _pipeline_with(_make_scenes())
        result = pipe.select(n=1, strategy="best")
        assert result is pipe
