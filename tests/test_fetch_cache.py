"""SatelliteFetcher local search cache: TTL and size pruning (previously silent no-ops)."""
import json
import os
import time

import pytest

from pygeovision.data.fetch import SatelliteFetcher


@pytest.fixture
def fetcher(tmp_path):
    f = SatelliteFetcher.__new__(SatelliteFetcher)
    f.cache_dir = tmp_path
    f._cache_ttl = 3600
    return f


def _entry(tmp_path, key, age_s=0.0, size=10):
    p = tmp_path / f"{key}.json"
    p.write_text(json.dumps({"version": 2, "results": [], "pad": "x" * size}))
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


class TestCacheTTL:
    def test_entry_older_than_ttl_is_discarded(self, fetcher, tmp_path):
        p = _entry(tmp_path, "a", age_s=120)
        fetcher.set_cache_ttl(60)
        assert fetcher._load_cache("a") is None
        assert not p.exists()

    def test_entry_younger_than_ttl_is_served(self, fetcher, tmp_path):
        _entry(tmp_path, "a", age_s=120)
        fetcher.set_cache_ttl(600)
        assert fetcher._load_cache("a") == []

    def test_default_ttl_applies_without_calling_the_setter(self, fetcher, tmp_path):
        _entry(tmp_path, "a", age_s=7200)
        assert fetcher._load_cache("a") is None  # older than the 3600 s default

    def test_instance_built_without_init_still_has_a_ttl(self, tmp_path):
        f = SatelliteFetcher.__new__(SatelliteFetcher)
        f.cache_dir = tmp_path
        _entry(tmp_path, "a", age_s=10)
        assert f._load_cache("a") == []

    @pytest.mark.parametrize("bad", [-1, "10", None, True])
    def test_invalid_ttl_rejected(self, fetcher, bad):
        with pytest.raises(ValueError):
            fetcher.set_cache_ttl(bad)


class TestPruneCache:
    def test_removes_oldest_first_until_under_limit(self, fetcher, tmp_path):
        old = _entry(tmp_path, "old", age_s=300, size=400_000)
        mid = _entry(tmp_path, "mid", age_s=200, size=400_000)
        new = _entry(tmp_path, "new", age_s=100, size=400_000)
        limit_gb = 0.0009  # ~ 0.97 MB: two of the three ~0.4 MB files fit
        removed = fetcher.prune_cache(max_size_gb=limit_gb)
        assert removed == 1
        assert not old.exists() and mid.exists() and new.exists()

    def test_nothing_removed_when_under_limit(self, fetcher, tmp_path):
        _entry(tmp_path, "a")
        assert fetcher.prune_cache(max_size_gb=1.0) == 0

    def test_zero_limit_empties_cache(self, fetcher, tmp_path):
        _entry(tmp_path, "a")
        _entry(tmp_path, "b")
        assert fetcher.prune_cache(max_size_gb=0) == 2
        assert list(tmp_path.glob("*.json")) == []

    def test_negative_limit_rejected(self, fetcher):
        with pytest.raises(ValueError):
            fetcher.prune_cache(max_size_gb=-1)
