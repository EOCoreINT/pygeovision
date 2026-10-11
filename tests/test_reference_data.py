"""Reference-product readers: window/warp/mosaic on local files, STAC paging and signing, footprint parsers.

Everything is offline: the "remote" tiles are small GeoTIFFs on disk and the HTTP session is a fake that serves
fixtures shaped like the real services (Planetary Computer STAC items, Microsoft ``dataset-links.csv`` +
line-delimited GeoJSON, Google tile manifest + CSV, Overpass JSON).
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pygeovision.data import footprints as fp
from pygeovision.data.reference import (
    PlanetaryComputer,
    ReferenceDataError,
    copernicus_dem_aws_urls,
    item_year,
    make_grid,
    parse_year,
    read_to_grid,
    utm_crs_for,
)


def _tif(path, arr, west, north, res, crs="EPSG:4326", nodata=0, dtype="uint8"):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs=crs, transform=from_origin(west, north, res, res), nodata=nodata) as d:
        d.write(arr.astype(dtype), 1)
    return str(path)


class Resp:
    def __init__(self, content=b"", payload=None, status=200):
        self.content, self._payload, self.status_code = content, payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes, posts=None):
        self.routes, self.posts, self.calls = routes, posts or {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        for key, val in self.routes.items():
            if url.startswith(key):
                v = val(url, params) if callable(val) else val
                return v if isinstance(v, Resp) else Resp(content=v) if isinstance(v, bytes) else Resp(payload=v)
        return Resp(status=404)

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", url, data))
        v = self.posts.get(url, Resp(status=500))
        return v


# ---------------------------------------------------------------------------------------------------------------------
class TestGrid:
    def test_utm_zone(self):
        assert utm_crs_for(0.5, 6.0) == "EPSG:32631"        # lower Volta
        assert utm_crs_for(-1.6, 6.7) == "EPSG:32630"       # Kumasi
        assert utm_crs_for(151.2, -33.9) == "EPSG:32756"    # southern hemisphere

    def test_grid_is_snapped_and_metric(self):
        g = make_grid((0.45, 5.92, 0.55, 5.98), 30)
        assert g.crs == "EPSG:32631" and g.transform.a == 30 and g.transform.e == -30
        assert g.transform.c % 30 == 0 and g.transform.f % 30 == 0
        assert g.pixel_area_ha == pytest.approx(0.09)

    def test_too_large_is_refused_with_advice(self):
        with pytest.raises(ReferenceDataError, match="smaller bounding box"):
            make_grid((0, 0, 5, 5), 10)

    def test_bad_bbox(self):
        with pytest.raises(ReferenceDataError):
            make_grid((1, 1, 0, 2), 30)


class TestReadToGrid:
    def test_mosaic_of_two_tiles_in_different_crs(self, tmp_path):
        a = _tif(tmp_path / "a.tif", np.full((200, 200), 2), 0.40, 6.00, 0.0005)                # EPSG:4326
        # tile B is stored in UTM 31N; covers lon 0.50-0.60 (east of A)
        from pyproj import Transformer
        x0, y1 = Transformer.from_crs(4326, 32631, always_xy=True).transform(0.499, 6.001)
        b = _tif(tmp_path / "b.tif", np.full((400, 400), 5), x0, y1, 30.0, crs="EPSG:32631")
        g = make_grid((0.45, 5.92, 0.55, 5.98), 30)
        r = read_to_grid([a, b], g)
        assert r.valid.all()
        assert set(np.unique(r.data)) <= {2.0, 5.0}
        assert (r.data == 2).sum() > 0.4 * r.data.size and (r.data == 5).sum() > 0.4 * r.data.size
        assert len(r.sources) == 2

    def test_uncovered_pixels_stay_invalid_and_zero_is_not_confused_with_outside(self, tmp_path):
        arr = np.zeros((100, 100), np.uint8)
        arr[:, 50:] = 3
        t = _tif(tmp_path / "t.tif", arr, 0.45, 5.98, 0.0005, nodata=None)      # real zeros, no nodata declared
        g = make_grid((0.45, 5.90, 0.52, 5.98), 30)
        r = read_to_grid([t], g)
        assert 0.3 < r.valid.mean() < 0.9                                        # south of the tile is not covered
        assert (r.data[r.valid] == 0).any(), "a real 0 inside the tile must stay valid"
        assert np.isnan(r.data[~r.valid]).all()

    def test_source_nodata_and_invalid_values(self, tmp_path):
        arr = np.full((100, 100), 1, np.uint8)
        arr[:, :50] = 0                                  # tile nodata
        arr[:, 50:60] = 255                              # product "no data" code
        t = _tif(tmp_path / "t.tif", arr, 0.45, 5.98, 0.0005, nodata=0)
        g = make_grid((0.45, 5.93, 0.50, 5.98), 30)
        r = read_to_grid([t], g, invalid_values=[255])
        vals = np.unique(r.data[r.valid])
        assert set(vals) == {1.0}

    def test_unreadable_source_is_an_error_not_an_empty_map(self, tmp_path):
        g = make_grid((0.45, 5.93, 0.50, 5.98), 30)
        with pytest.raises(ReferenceDataError, match="could not read"):
            read_to_grid([str(tmp_path / "missing.tif")], g)

    def test_skip_missing_skips_only_404(self, tmp_path, monkeypatch):
        import rasterio as rio
        g = make_grid((0.45, 5.93, 0.50, 5.98), 30)
        good = _tif(tmp_path / "t.tif", np.full((400, 400), 7), 0.40, 6.10, 0.0005)
        real_open = rio.open

        def fake_open(path, *a, **k):
            if str(path).startswith("http"):
                raise rio.errors.RasterioIOError(f"HTTP response code: {403 if 'forbidden' in str(path) else 404}")
            return real_open(path, *a, **k)
        monkeypatch.setattr(rio, "open", fake_open)
        r = read_to_grid(["https://x.test/none.tif", good], g, skip_missing=True)
        assert r.valid.all() and r.sources == [good]
        with pytest.raises(ReferenceDataError, match="could not read"):
            read_to_grid(["https://x.test/none.tif", good], g)                         # off by default
        with pytest.raises(ReferenceDataError, match="403"):
            read_to_grid(["https://x.test/forbidden.tif", good], g, skip_missing=True)  # only 404 is skipped

    def test_token_is_never_in_the_error(self, tmp_path):
        g = make_grid((0.45, 5.93, 0.50, 5.98), 30)
        with pytest.raises(ReferenceDataError) as ei:
            read_to_grid([str(tmp_path / "missing.tif") + "?sig=SECRET"], g)
        assert "SECRET" not in str(ei.value) and "token removed" in str(ei.value)

    def test_bilinear_vs_nearest_on_a_ramp(self, tmp_path):
        ramp = np.tile(np.arange(100, dtype="float32"), (100, 1))
        t = _tif(tmp_path / "r.tif", ramp, 0.45, 5.98, 0.0005, nodata=None, dtype="float32")
        g = make_grid((0.46, 5.94, 0.49, 5.97), 20)
        near = read_to_grid([t], g, resampling="nearest")
        lin = read_to_grid([t], g, resampling="bilinear")
        assert len(np.unique(lin.data)) > len(np.unique(near.data))


class TestPlanetaryComputer:
    ITEMS = {"features": [{"id": "31N-2022", "properties": {"start_datetime": "2022-01-01T00:00:00Z", "datetime": None},
                           "assets": {"data": {"href": "https://ai4edataeuwest.blob.core.windows.net/io-lulc/a.tif"}}}],
             "links": [{"rel": "next", "href": "https://pc.test/next"}]}
    PAGE2 = {"features": [{"id": "31N-2023", "properties": {"start_datetime": "2023-01-01T00:00:00Z", "datetime": None},
                           "assets": {"data": {"href": "https://ai4edataeuwest.blob.core.windows.net/io-lulc/b.tif"}}}],
             "links": []}

    def _pc(self):
        sess = FakeSession({"https://pc.test/next": self.PAGE2,
                            "https://pc.test/stac/collections/io-lulc/items": self.ITEMS,
                            "https://pc.test/sas/io-lulc": {"token": "st=1&sig=abc", "msft:expiry": "x"}})
        return PlanetaryComputer(sess, stac_url="https://pc.test/stac", sas_url="https://pc.test/sas", retries=1), sess

    def test_pages_are_followed(self):
        pc, sess = self._pc()
        items = pc.search("io-lulc", (0.4, 5.9, 0.5, 6.0), "2022-03-01/2022-09-30")
        assert [i["id"] for i in items] == ["31N-2022", "31N-2023"]
        first = sess.calls[0]
        assert first[2]["datetime"] == "2022-03-01/2022-09-30" and first[2]["bbox"].startswith("0.400000")

    def test_urls_are_signed_once_and_token_is_cached(self):
        pc, sess = self._pc()
        items = pc.search("io-lulc", (0.4, 5.9, 0.5, 6.0))
        urls = pc.asset_urls("io-lulc", items, "data")
        assert all(u.endswith("?st=1&sig=abc") for u in urls)
        assert sum(1 for c in sess.calls if "/sas/" in c[1]) == 1
        assert pc.sign("https://example.org/x.tif", "io-lulc") == "https://example.org/x.tif"   # not blob storage
        assert pc.sign(urls[0], "io-lulc") == urls[0]                                           # already signed

    def test_sign_endpoint_is_preferred_and_reused_per_container(self):
        calls = []

        def sign(url, params):
            calls.append(params["href"])
            return {"href": params["href"] + "?sv=2025&sr=c&sp=rl&sig=FROM_SIGN"}
        sess = FakeSession({"https://pc.test/sign": sign, "https://pc.test/sas/": {"token": "sig=OLD_ACCOUNT_TOKEN"},
                            "https://pc.test/stac/collections/io-lulc/items": self.ITEMS, "https://pc.test/next": self.PAGE2})
        pc = PlanetaryComputer(sess, stac_url="https://pc.test/stac", sas_url="https://pc.test/sas", retries=1)
        urls = pc.asset_urls("io-lulc", pc.search("io-lulc", (0.4, 5.9, 0.5, 6.0)), "data")
        assert all("FROM_SIGN" in u and "OLD_ACCOUNT" not in u for u in urls)
        assert len(calls) == 1                                    # both tiles live in the same container
        assert not any("/sas/" in c[1] for c in sess.calls)

    def test_blob_level_tokens_are_not_shared_between_files(self):
        n = []

        def sign(url, params):
            n.append(1)
            return {"href": params["href"] + "?sr=b&sig=X"}
        pc = PlanetaryComputer(FakeSession({"https://pc.test/sign": sign}), sas_url="https://pc.test/sas", retries=1)
        pc.sign("https://acct.blob.core.windows.net/c/a.tif", "x")
        pc.sign("https://acct.blob.core.windows.net/c/b.tif", "x")
        assert len(n) == 2

    def test_falls_back_to_collection_token_when_sign_fails(self):
        sess = FakeSession({"https://pc.test/sas/io-lulc": {"token": "sig=COLL"}})
        pc = PlanetaryComputer(sess, sas_url="https://pc.test/sas", retries=1)
        assert pc.sign("https://acct.blob.core.windows.net/c/a.tif", "io-lulc").endswith("?sig=COLL")

    def test_failure_is_reported_not_swallowed(self):
        pc = PlanetaryComputer(FakeSession({}), stac_url="https://pc.test/stac", retries=1)
        with pytest.raises(ReferenceDataError, match="failed after"):
            pc.search("io-lulc", (0, 0, 1, 1))

    def test_year_helpers(self):
        assert item_year(self.ITEMS["features"][0]) == 2022
        assert item_year({"properties": {"datetime": "2020-07-01T00:00:00Z"}}) == 2020
        assert item_year({"properties": {}}) is None
        assert parse_year("2021-06", "x") == 2021 and parse_year(2019, "x") == 2019
        with pytest.raises(ReferenceDataError):
            parse_year("June", "date_before")


# ---------------------------------------------------------------------------------------------------------------------
class TestQuadkeys:
    def test_known_tile(self):
        # Bing tile (x=0, y=0, zoom 1) is "0"; zoom-2 tile x=3,y=3 is "33"
        assert fp.quadkey(0, 0, 1) == "0" and fp.quadkey(3, 3, 2) == "33"

    def test_covering(self):
        keys = fp.quadkeys_for_bbox((0.40, 5.90, 0.50, 6.00))
        assert keys and all(len(k) == 9 for k in keys)
        big = fp.quadkeys_for_bbox((0.0, 5.0, 2.0, 7.0))
        assert len(big) > len(keys)


def _geojsonl(features):
    return gzip.compress("\n".join(json.dumps(f) for f in features).encode())


def _square(lon, lat, d=0.0002):
    return {"type": "Polygon", "coordinates": [[[lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]]}


class TestMicrosoft:
    def _setup(self, tmp_path, qk):
        index = f"Location,QuadKey,Url,Size,UploadDate\nGhana,{qk},https://store.test/gh/{qk}.csv.gz,1MB,2023-01-01\nKenya,999999999,https://store.test/ke.csv.gz,1MB,2023-01-01\n"
        feats = [{"type": "Feature", "geometry": _square(0.45, 5.95), "properties": {"height": 4.0, "confidence": 0.9}},
                 {"type": "Feature", "geometry": _square(0.451, 5.951), "properties": {"height": -1, "confidence": -1}},
                 {"type": "Feature", "geometry": _square(2.0, 2.0), "properties": {}}]           # outside the box
        sess = FakeSession({"https://idx.test/dataset-links.csv": index.encode(),
                            f"https://store.test/gh/{qk}.csv.gz": _geojsonl(feats)})
        return sess

    def test_reads_only_covering_partitions_and_clips(self, tmp_path):
        qk = fp.quadkeys_for_bbox((0.44, 5.94, 0.46, 5.96))[0]
        sess = self._setup(tmp_path, qk)
        gdf = fp.microsoft_footprints((0.44, 5.94, 0.46, 5.96), cache_dir=tmp_path, session=sess,
                                      index_url="https://idx.test/dataset-links.csv")
        assert len(gdf) == 2 and str(gdf.crs) == "EPSG:4326"
        assert gdf.height_m.notna().sum() == 1 and gdf.confidence.notna().sum() == 1     # -1 means unknown -> None
        assert not any("ke.csv" in c[1] for c in sess.calls), "other partitions must not be downloaded"

    def test_second_call_uses_the_cache(self, tmp_path):
        qk = fp.quadkeys_for_bbox((0.44, 5.94, 0.46, 5.96))[0]
        sess = self._setup(tmp_path, qk)
        kw = {"cache_dir": tmp_path, "session": sess, "index_url": "https://idx.test/dataset-links.csv"}
        fp.microsoft_footprints((0.44, 5.94, 0.46, 5.96), **kw)
        n = len(sess.calls)
        fp.microsoft_footprints((0.44, 5.94, 0.46, 5.96), **kw)
        assert len(sess.calls) == n

    def test_uncovered_area_is_an_error_not_an_empty_result(self, tmp_path):
        sess = self._setup(tmp_path, "000000000")
        with pytest.raises(fp.FootprintSourceError, match="do not cover"):
            fp.microsoft_footprints((0.44, 5.94, 0.46, 5.96), cache_dir=tmp_path, session=sess,
                                    index_url="https://idx.test/dataset-links.csv")

    def test_unreachable_index_explains_how_to_fix(self, tmp_path):
        sess = FakeSession({})
        with pytest.raises(fp.FootprintSourceError, match="index_url"):
            fp.microsoft_footprints((0.44, 5.94, 0.46, 5.96), cache_dir=tmp_path, session=sess)

    def test_index_is_discovered_from_the_readme(self, tmp_path):
        qk = fp.quadkeys_for_bbox((0.44, 5.94, 0.46, 5.96))[0]
        new_url = "https://bfppub.blob.core.windows.net/%24web/2031-01-01/dataset-links.csv"
        index = f"Location,QuadKey,Url,Size,UploadDate\nGhana,{qk},https://store.test/gh/{qk}.csv.gz,1MB,x\n"
        sess = FakeSession({fp.MS_README: f"see [links]({new_url}) and {fp.MS_INDEX_URLS[0]}".encode(),
                            new_url: index.encode(),
                            f"https://store.test/gh/{qk}.csv.gz": _geojsonl(
                                [{"type": "Feature", "geometry": _square(0.45, 5.95), "properties": {}}])})
        gdf = fp.microsoft_footprints((0.44, 5.94, 0.46, 5.96), cache_dir=tmp_path, session=sess)
        assert len(gdf) == 1 and any(c[1] == new_url for c in sess.calls)

    def test_truncated_file_is_an_error(self, tmp_path):
        p = tmp_path / "x.gz"
        p.write_bytes(gzip.compress(b'{"type": "Feature", "geometry": {"type": "Poly'))
        with pytest.raises(fp.FootprintSourceError, match="not JSON"):
            list(fp._iter_ms_lines(p))


class TestGoogle:
    def _csv(self):
        rows = ["latitude,longitude,area_in_meters,confidence,geometry,full_plus_code"]
        def poly(lon, lat, d=0.0002):
            return f'"POLYGON(({lon} {lat}, {lon+d} {lat}, {lon+d} {lat+d}, {lon} {lat+d}, {lon} {lat}))"'
        rows.append(f"5.9501,0.4501,50.0,0.9,{poly(0.45, 5.95)},AAA")
        rows.append(f"5.9511,0.4511,50.0,0.5,{poly(0.451, 5.951)},BBB")      # below the confidence threshold
        rows.append(f"2.0,2.0,50.0,0.9,{poly(2.0, 2.0)},CCC")                 # outside the box
        return gzip.compress("\n".join(rows).encode())

    def test_filters_by_box_and_confidence(self, tmp_path):
        manifest = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"tile_url": "https://g.test/a.csv.gz"},
             "geometry": {"type": "Polygon", "coordinates": [[[0, 5], [1, 5], [1, 7], [0, 7], [0, 5]]]}},
            {"type": "Feature", "properties": {"tile_url": "https://g.test/other.csv.gz"},
             "geometry": {"type": "Polygon", "coordinates": [[[30, 5], [31, 5], [31, 7], [30, 7], [30, 5]]]}}]}
        sess = FakeSession({"https://m.test/tiles.geojson": json.dumps(manifest).encode(),
                            "https://g.test/a.csv.gz": self._csv()})
        gdf = fp.google_footprints((0.44, 5.94, 0.46, 5.96), min_confidence=0.65, cache_dir=tmp_path, session=sess,
                                   manifest_url="https://m.test/tiles.geojson")
        assert len(gdf) == 1 and gdf.confidence.iloc[0] == pytest.approx(0.9)
        assert not any("other.csv" in c[1] for c in sess.calls)

    def test_outside_coverage_is_an_error(self, tmp_path):
        manifest = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"tile_url": "u"}, "geometry": {"type": "Polygon", "coordinates": [[[30, 5], [31, 5], [31, 7], [30, 7], [30, 5]]]}}]}
        sess = FakeSession({"https://m.test/t": json.dumps(manifest).encode()})
        with pytest.raises(fp.FootprintSourceError, match="does not cover"):
            fp.google_footprints((0.44, 5.94, 0.46, 5.96), cache_dir=tmp_path, session=sess, manifest_url="https://m.test/t")

    def test_bad_confidence(self, tmp_path):
        with pytest.raises(fp.FootprintSourceError):
            fp.google_footprints((0.44, 5.94, 0.46, 5.96), min_confidence=1.5, cache_dir=tmp_path, session=FakeSession({}))


class TestOverpass:
    EP = "https://op.test/api"

    def _sess(self, body, status=200):
        return FakeSession({}, posts={self.EP: Resp(payload=body, status=status)})

    def test_buildings_become_polygons_and_unclosed_ways_are_dropped(self):
        body = {"elements": [
            {"type": "way", "id": 1, "tags": {"building": "yes", "name": "A"},
             "geometry": [{"lat": 5.95, "lon": 0.45}, {"lat": 5.95, "lon": 0.4502}, {"lat": 5.9502, "lon": 0.4502}, {"lat": 5.95, "lon": 0.45}]},
            {"type": "way", "id": 2, "tags": {"building": "yes"},
             "geometry": [{"lat": 5.95, "lon": 0.46}, {"lat": 5.95, "lon": 0.4602}, {"lat": 5.9502, "lon": 0.4602}]},
            {"type": "node", "id": 3}]}
        gdf = fp.osm_features((0.44, 5.94, 0.47, 5.96), "building", session=self._sess(body), endpoints=[self.EP])
        assert list(gdf.osm_id) == [1] and gdf.geometry.iloc[0].geom_type == "Polygon"

    def test_highways_become_lines(self):
        body = {"elements": [{"type": "way", "id": 9, "tags": {"highway": "primary", "surface": "asphalt"},
                              "geometry": [{"lat": 5.95, "lon": 0.45}, {"lat": 5.96, "lon": 0.46}]}]}
        gdf = fp.osm_features((0.44, 5.94, 0.47, 5.97), "highway", session=self._sess(body), endpoints=[self.EP])
        assert gdf.highway.iloc[0] == "primary" and gdf.geometry.iloc[0].geom_type == "LineString"

    def test_overpass_runtime_error_is_not_returned_as_an_empty_map(self):
        body = {"elements": [], "remark": "runtime error: Query timed out in \"query\" at line 3 after 181 seconds."}
        with pytest.raises(fp.FootprintSourceError, match="smaller bounding box"):
            fp.osm_features((0.44, 5.94, 0.47, 5.97), "building", session=self._sess(body), endpoints=[self.EP])

    def test_genuinely_empty_area_is_an_empty_frame(self):
        gdf = fp.osm_features((0.44, 5.94, 0.47, 5.97), "building", session=self._sess({"elements": []}), endpoints=[self.EP])
        assert len(gdf) == 0

    def test_second_mirror_is_tried(self):
        sess = FakeSession({}, posts={"https://bad": Resp(status=504), self.EP: Resp(payload={"elements": []})})
        gdf = fp.osm_features((0.44, 5.94, 0.47, 5.97), "highway", session=sess, endpoints=["https://bad", self.EP])
        assert len(gdf) == 0 and [c[1] for c in sess.calls] == ["https://bad", self.EP]

    def test_kind_validated(self):
        with pytest.raises(fp.FootprintSourceError):
            fp.osm_features((0, 0, 1, 1), "river", session=self._sess({}), endpoints=[self.EP])


class TestDiagnostics:
    def test_redact(self):
        from pygeovision.data.reference import _redact
        assert _redact("https://a/b.tif?sig=SECRET&se=1") == "https://a/b.tif?<token removed>"
        assert _redact("https://a/b.tif") == "https://a/b.tif"

    def test_403_quotes_the_servers_reason(self, monkeypatch):
        import requests

        from pygeovision.data.reference import _http_reason

        class R:
            status_code = 403
            text = ("<?xml version='1.0'?><Error><Code>AuthorizationFailure</Code>"
                    "<Message>This request is not authorized.\nRequestId:1</Message></Error>")
        monkeypatch.setattr(requests, "get", lambda *a, **k: R())
        out = _http_reason("https://h/x.tif?sig=1", "HTTP response code: 403")
        assert "AuthorizationFailure" in out and "not authorized" in out and "RequestId" not in out

    def test_403_that_a_plain_request_survives_points_at_gdal(self, monkeypatch):
        import requests

        from pygeovision.data.reference import _http_reason

        class R:
            status_code = 206
            text = ""
        monkeypatch.setattr(requests, "get", lambda *a, **k: R())
        assert "specific to the GDAL read" in _http_reason("https://h/x.tif", "HTTP response code: 403")

    def test_probe_failure_is_stated_and_other_errors_are_not_probed(self, monkeypatch):
        import requests

        from pygeovision.data.reference import _http_reason

        def boom(*a, **k):
            raise requests.ConnectionError("down")
        monkeypatch.setattr(requests, "get", boom)
        assert "also failed" in _http_reason("https://h/x.tif", "HTTP response code: 403")
        assert _http_reason("https://h/x.tif", "HTTP response code: 500") == ""
        assert _http_reason("/local/x.tif", "403") == ""


class TestAwsDemTiles:
    def test_tile_named_for_its_south_west_corner(self):
        u = copernicus_dem_aws_urls((11.3, 47.2, 11.4, 47.3))
        assert u == ["https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N47_00_E011_00_DEM/"
                     "Copernicus_DSM_COG_10_N47_00_E011_00_DEM.tif"]

    def test_box_on_whole_degrees_does_not_add_a_neighbour(self):
        assert len(copernicus_dem_aws_urls((11.0, 47.0, 12.0, 48.0))) == 1

    def test_hemispheres_and_several_tiles(self):
        u = copernicus_dem_aws_urls((-0.15, -0.5, 0.1, 0.1))
        names = sorted(x.rsplit("/", 1)[1][len("Copernicus_DSM_COG_10_"):-len("_00_DEM.tif")] for x in u)
        assert names == ["N00_00_E000", "N00_00_W001", "S01_00_E000", "S01_00_W001"]

    def test_bad_bbox(self):
        with pytest.raises(ReferenceDataError):
            copernicus_dem_aws_urls((2, 0, 1, 1))
