"""Tests for pygeovision.obia.

Expected values are derived by hand (see the comments), not copied from output,
so a failure points at a real change in behaviour.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, box  # noqa: E402

from pygeovision import obia  # noqa: E402
from pygeovision.obia._adjacency import adjacency_dict, border_pairs, image_edge_lengths  # noqa: E402


# --------------------------------------------------------------------- adjacency
def test_border_pairs_and_frame_edges():
    lab = np.zeros((4, 4), int); lab[:, :2] = 1; lab[:, 2:] = 2
    a, b, ln = border_pairs(lab)
    assert (a.tolist(), b.tolist(), ln.tolist()) == ([1], [2], [4])
    assert image_edge_lengths(lab, 3).tolist() == [0, 8, 8]      # 2 top + 2 bottom + 4 left (or right) = 8
    assert adjacency_dict(lab) == {1: {2: 4}, 2: {1: 4}}


# ----------------------------------------------------------------------- merging
def test_merge_cost_matches_hand_value():
    # two one-pixel objects with values 0 and 10: n_m*sd_m - 0 = 2*5 = 10
    m = obia.RegionMerger(np.array([[[0.0, 10.0]]]), np.array([[1, 2]]), color_weight=1.0)
    assert m._cost(0, 1, 1) == pytest.approx(10.0)


def test_scale_threshold_is_scale_squared():
    img = np.zeros((1, 10, 10)); img[:, :, 5:] = 100
    init = np.arange(100).reshape(10, 10)
    m = obia.RegionMerger(img, init, color_weight=1.0)
    m.run(60)                       # 60^2 = 3600 < merge cost 100 * 50 = 5000
    assert m.n_objects == 2
    m.run(80)                       # 80^2 = 6400 > 5000
    assert m.n_objects == 1


def test_object_count_never_increases_with_scale():
    img = np.random.default_rng(0).uniform(0, 255, (3, 30, 30))
    m = obia.RegionMerger(img, np.arange(900).reshape(30, 30))
    counts = []
    for s in (5, 15, 30, 60):
        m.run(s)
        counts.append(m.n_objects)
    assert counts == sorted(counts, reverse=True)


def test_merger_rejects_bad_parameters():
    img = np.zeros((1, 4, 4)); init = np.arange(16).reshape(4, 4)
    with pytest.raises(ValueError):
        obia.RegionMerger(img, init, color_weight=1.5)
    with pytest.raises(ValueError):
        obia.RegionMerger(img, init[:3])
    with pytest.raises(ValueError):
        obia.RegionMerger(img, init).run(-1)


# ------------------------------------------------------------------ segmentation
@pytest.fixture(scope="module")
def two_halves():
    rng = np.random.default_rng(1)
    img = np.zeros((4, 60, 60)); img[:, :, :30] = 60; img[:, :, 30:] = 200
    return img + rng.normal(0, 3, img.shape)


@pytest.mark.parametrize("method", obia.METHODS)
def test_every_method_gives_connected_numbered_objects(two_halves, method):
    seg = obia.segment(two_halves, method)
    lab = seg.labels
    assert lab.shape == (60, 60) and lab.min() == 1 and lab.max() == seg.n_objects
    assert set(np.unique(lab)) == set(range(1, seg.n_objects + 1))
    assert all(ndi.label(lab == i)[1] == 1 for i in np.unique(lab))


def test_multires_separates_two_flat_regions(two_halves):
    seg = obia.segment(two_halves, "multires", scale=40)
    assert seg.n_objects == 2 and seg.labels[0, 0] != seg.labels[0, 59]


def test_levels_are_nested(two_halves):
    lv = obia.segment_levels(two_halves, [5, 20, 40])
    fine, coarse = lv[5.0].labels, lv[40.0].labels
    assert all(len(np.unique(coarse[fine == f])) == 1 for f in np.unique(fine))


def test_unknown_method_and_from_labels_validation(two_halves):
    with pytest.raises(ValueError):
        obia.segment(two_halves, "nope")
    with pytest.raises(ValueError):
        obia.from_labels(np.ones((5, 5), int), np.zeros((1, 6, 6)))
    # a label id used by two separate blobs is split into two objects
    lab = np.zeros((3, 7), int); lab[:, 0] = 1; lab[:, 6] = 1
    assert obia.from_labels(lab).n_objects == 3        # two blobs of id 1 + the zero background


def test_vector_roundtrip_preserves_area():
    sc = obia.make_demo_scene(96)
    seg = obia.segment(sc.raster, "multires", scale=20)
    gdf = seg.to_vector()
    assert len(gdf) == seg.n_objects
    assert gdf.geometry.area.sum() == pytest.approx(seg.labels.size * 4.0)   # 2 m pixels
    assert str(gdf.crs).endswith("32633")


# ---------------------------------------------------------------------- features
@pytest.fixture(scope="module")
def eight_by_six():
    lab = np.zeros((6, 8), int); lab[:, :4] = 1; lab[:, 4:] = 2
    b = np.zeros((6, 8)); b[:, :4] = 10; b[:, 4::2] = 20; b[:, 5::2] = 40
    return lab, b


def test_spectral_and_context_features(eight_by_six):
    lab, b = eight_by_six
    f = obia.extract_features(lab, b[None], ["nir"])
    assert f["mean_nir"].tolist() == [10, 30] and np.allclose(f["std_nir"], [0, 10])
    assert f["min_nir"].tolist() == [10, 20] and f["max_nir"].tolist() == [10, 40]
    assert (f["area_px"] == 24).all() and (f["n_neighbors"] == 1).all()
    assert np.allclose(f["diff_to_neighbors_nir"], 20)
    assert f.loc[1, "rel_border_image_edge"] == pytest.approx(14 / 20)   # 14 frame edges, 6 shared


def test_geometry_features_are_exact():
    sq = np.zeros((8, 8), int); sq[2:5, 2:5] = 1
    g = obia.extract_features(sq, np.zeros((1, 8, 8)), ["x"], spectral=False, context=False).loc[1]
    assert g.area_px == 9 and g.perimeter_px == 12
    assert g.shape_index == pytest.approx(1.0) and g.compactness == pytest.approx(np.pi / 4)
    st = np.zeros((12, 12), int); st[5:7, 1:11] = 1
    s = obia.extract_features(st, np.zeros((1, 12, 12)), ["x"], spectral=False, context=False).loc[1]
    assert s.perimeter_px == 24 and s.elongation > 4 and s.compactness < g.compactness


def test_glcm_texture_hand_values():
    one = np.ones((6, 6), int)
    flat = obia.glcm_texture(one, np.full((6, 6), 5.0)).loc[1]
    assert flat.glcm_contrast == 0 and flat.glcm_energy == pytest.approx(1) and flat.glcm_entropy == 0
    stripes = obia.glcm_texture(one, np.tile([0.0, 100.0], (6, 3))).loc[1]
    # step of 15 grey levels; 80 of the 110 pooled pairs straddle a stripe edge
    assert stripes.glcm_contrast == pytest.approx(225 * 80 / 110)


def test_class_context_and_hierarchy(eight_by_six):
    lab, b = eight_by_six
    cc = obia.class_context(lab, pd.Series({1: "a", 2: "b"}))
    assert cc.loc[1, "rel_border_b"] == 1.0 and cc.loc[1, "rel_border_a"] == 0.0
    fine = np.zeros((6, 8), int); fine[:3, :4] = 1; fine[:3, 4:] = 2; fine[3:, :4] = 3; fine[3:, 4:] = 4
    assert obia.parent_map(fine, lab).to_dict() == {1: 1, 2: 2, 3: 1, 4: 2}
    so = obia.super_object_features(fine, lab, b[None], ["nir"])
    assert np.allclose(so["rel_area_of_parent"], 0.5)
    frac = obia.sub_object_fraction(pd.Series({1: "x", 2: "y", 3: "y", 4: "y"}),
                                    pd.Series({1: 12, 2: 12, 3: 12, 4: 12}),
                                    obia.parent_map(fine, lab), "y")
    assert frac.to_dict() == {1: 0.5, 2: 1.0}


def test_layer_names_are_made_expression_safe(eight_by_six):
    lab, b = eight_by_six
    f = obia.extract_features(lab, b[None], ["near-infrared"], extra_layers={"NDVI 2": b / 50})
    assert "mean_near_infrared" in f and "mean_NDVI_2" in f
    with pytest.raises(ValueError):
        obia.extract_features(lab, b[None], ["x"], extra_layers={"bad": np.zeros((2, 2))})


# ---------------------------------------------------------------- classification
def test_membership_functions():
    assert np.allclose(obia.Trapezoid(0, 10, 20, 30)([-1, 5, 10, 15, 20, 25, 35]), [0, .5, 1, 1, 1, .5, 0])
    assert np.allclose(obia.Triangle(0, 10, 20)([0, 5, 10, 15, 20]), [0, .5, 1, .5, 0])
    assert np.allclose(obia.LargerThan(10, 20)([5, 10, 15, 20, 30]), [0, 0, .5, 1, 1])
    assert np.allclose(obia.SmallerThan(10, 20)([5, 10, 15, 20, 30]), [1, 1, .5, 0, 0])
    assert np.allclose(obia.LargerThan(10)([9, 10, 11]), [0, 0, 1])
    assert obia.Gaussian(5, 2)(7) == pytest.approx(np.exp(-0.5))
    with pytest.raises(ValueError):
        obia.Trapezoid(3, 2, 1, 0)


def test_rules_first_match_wins_and_explain():
    F = pd.DataFrame({"mean_nir": [10, 150, 100], "compactness": [.9, .8, .3],
                      "brightness": [50, 130, 130]}, index=[1, 2, 3])
    rb = obia.RuleBasedClassifier([("water", "mean_nir < 40"),
                                   ("building", "brightness > 120 and compactness > 0.6"),
                                   ("field", "mean_nir > 90")])
    assert rb.classify(F).classes.to_dict() == {1: "water", 2: "building", 3: "field"}
    ex = rb.explain(F, 2)
    assert ex["condition_true"].tolist() == [False, True, True]
    assert ex["decides"].tolist() == [False, True, False]      # two true, only the first decides
    with pytest.raises(ValueError, match="available columns"):
        obia.RuleBasedClassifier([("x", "nope > 1")]).classify(F)


def test_fuzzy_stability_is_best_minus_second():
    fz = obia.FuzzyClassifier({"A": {"v": obia.SmallerThan(50, 100)}, "B": {"v": obia.LargerThan(50, 100)}})
    r = fz.classify(pd.DataFrame({"v": [0, 75, 200]}, index=[1, 2, 3]))
    assert np.allclose(r.stability, [1.0, 0.0, 1.0])
    assert r.classes[1] == "A" and r.classes[3] == "B"
    none = obia.FuzzyClassifier({"A": {"v": obia.LargerThan(500)}}, min_membership=.5)
    assert none.classify(pd.DataFrame({"v": [1.0]})).classes.iloc[0] == "unclassified"
    with pytest.raises(KeyError):
        obia.FuzzyClassifier({"A": {"missing": obia.LargerThan(1)}}).classify(pd.DataFrame({"v": [1.0]}))


def test_samples_from_polygons_and_points():
    lab = np.zeros((10, 10), int); lab[:, :5] = 1; lab[:, 5:] = 2
    polys = gpd.GeoDataFrame({"c": ["water", "field"]}, geometry=[box(0, 0, 5, 10), box(5, 0, 10, 10)])
    assert obia.samples_from_geometries(lab, polys, "c").to_dict() == {1: "water", 2: "field"}
    pts = gpd.GeoDataFrame({"c": ["field"]}, geometry=[Point(7.5, 3.5)])
    assert obia.samples_from_geometries(lab, pts, "c").to_dict() == {2: "field"}
    small = gpd.GeoDataFrame({"c": ["field"]}, geometry=[box(5, 0, 6, 2)])      # covers 2 of 50 pixels
    assert obia.samples_from_geometries(lab, small, "c").empty


@pytest.mark.parametrize("method", ["rf", "knn", "svm", "tree"])
def test_supervised_methods_learn_separable_classes(method):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"mean_nir": np.r_[rng.normal(20, 3, 30), rng.normal(150, 5, 30)],
                      "compactness": rng.uniform(0, 1, 60), "centroid_x": rng.uniform(0, 1, 60)},
                     index=range(1, 61))
    y = pd.Series(["water"] * 30 + ["field"] * 30, index=X.index)
    clf = obia.ObjectClassifier(method).fit(X, y)
    assert "centroid_x" not in clf.columns_                     # location is excluded by default
    assert (clf.predict(X).classes == y).mean() > 0.98
    if method == "tree":
        assert "mean_nir" in clf.export_rules()
    with pytest.raises(ValueError):
        obia.ObjectClassifier(method).fit(X, pd.Series("water", index=X.index))


def test_result_raster_vector_summary():
    r = obia.RuleBasedClassifier([("a", "v < 5"), ("b", "v >= 5")]).classify(
        pd.DataFrame({"v": [1, 9]}, index=[1, 2]))
    lab = np.array([[1, 1, 2]])
    assert r.to_raster(lab).tolist() == [[1, 1, 2]] and r.codes() == {"a": 1, "b": 2}
    gdf = r.to_vector(lab)
    assert gdf.set_index("obj_id")["class"].to_dict() == {1: "a", 2: "b"}
    assert r.summary(pd.Series({1: 2.0, 2: 1.0})).loc["a", "area_share"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------- accuracy
def test_confusion_kappa_and_weights():
    ref = pd.Series(list("aaabbc"), index=range(1, 7)); pred = pd.Series(list("aabbbc"), index=range(1, 7))
    r = obia.confusion_report(ref, pred)
    assert r["overall_accuracy"] == pytest.approx(5 / 6) and r["kappa"] == pytest.approx(17 / 23)
    assert np.allclose(r["per_class"]["producers_accuracy"], [2 / 3, 1, 1])
    assert np.allclose(r["per_class"]["users_accuracy"], [1, 2 / 3, 1])
    w = obia.confusion_report(ref, pred, weights=pd.Series([10, 10, 10, 1, 1, 100], index=range(1, 7)))
    assert w["overall_accuracy"] == pytest.approx(122 / 132) and w["weighted"]
    with pytest.raises(ValueError):
        obia.confusion_report(ref, pd.Series(["a"], index=[99]))


def test_segmentation_quality_perfect_over_and_under():
    ref = np.zeros((10, 10), int); ref[:, :5] = 1
    q, _ = obia.segmentation_quality(ref.copy(), ref)
    assert q.loc[1, "D"] == 0 and q.loc[1, "QR"] == 1
    strips = np.tile(np.arange(1, 11), (10, 1))                  # 1-pixel-wide segments
    q, _ = obia.segmentation_quality(strips, ref)
    assert q.loc[1, "over_seg"] == pytest.approx(0.8) and q.loc[1, "under_seg"] == 0
    assert q.loc[1, "QR"] == pytest.approx(0.2) and q.loc[1, "n_parts"] == 5
    q, _ = obia.segmentation_quality(np.ones((10, 10), int), ref)
    assert q.loc[1, "under_seg"] == pytest.approx(0.5) and q.loc[1, "AFI"] == pytest.approx(-1)
    assert q.loc[1, "D"] == pytest.approx(np.sqrt(0.125))


def test_pixel_report_counts_unclassified_as_errors():
    ref = np.array([[1, 1, 2, 2]]); pred = np.array([[1, 0, 2, 2]])
    r = obia.pixel_report(pred, ref, ["a", "b"])
    assert r["overall_accuracy"] == pytest.approx(0.75) and r["unclassified_pixels"] == 1


# ----------------------------------------------------------------------- scale + demo
def test_demo_scene_is_deterministic_and_well_formed():
    a, b = obia.make_demo_scene(96, seed=3), obia.make_demo_scene(96, seed=3)
    assert np.array_equal(a.raster.image, b.raster.image)
    assert not np.array_equal(a.raster.image, obia.make_demo_scene(96, seed=4).raster.image)
    assert a.raster.band_names == ["blue", "green", "red", "nir"]
    assert set(a.reference["class"]) <= set(obia.CLASS_NAMES) and a.ref_labels.min() >= 1
    with pytest.raises(ValueError):
        obia.make_demo_scene(50)


def test_scale_tools_on_demo_scene():
    sc = obia.make_demo_scene(112)
    sweep = obia.scale_vs_reference(sc.raster, [5, 10, 20, 40, 80], sc.ref_labels)
    assert sweep["n_objects"].is_monotonic_decreasing
    assert sweep["mean_over_seg"].iloc[0] > sweep["mean_over_seg"].iloc[-1]     # fewer fragments
    assert sweep["mean_under_seg"].iloc[0] < sweep["mean_under_seg"].iloc[-1]   # more merging errors
    assert sweep["mean_D"].idxmin() in (10.0, 20.0, 40.0)                       # best is in the middle
    diag = obia.scale_diagnostics(sc.raster, [5, 10, 15, 20, 30])
    assert list(diag.columns) == ["n_objects", "lv", "roc_lv", "candidate"]
    with pytest.raises(ValueError):
        obia.scale_diagnostics(sc.raster, [5, 10])


def test_full_workflow_beats_chance_by_a_wide_margin():
    sc = obia.make_demo_scene(128)
    seg = obia.segment(sc.raster, "multires", scale=20)
    F = obia.extract_features(seg.labels, sc.raster, extra_layers=obia.spectral_layers(sc.raster))
    rb = obia.RuleBasedClassifier([
        ("water", "mean_ndwi > 0.3"), ("road", "elongation > 6 and brightness < 110"),
        ("building", "area_px < 400 and compactness > 0.5 and mean_ndvi <= 0.55"),
        ("forest", "mean_ndvi > 0.55 and std_nir > 9 and area_px > 500"),
        ("cropland", "mean_ndvi > 0.55"), ("bare_field", "mean_ndvi <= 0.55")])
    rep = obia.assess_classification(rb.classify(F), seg.labels, sc.reference, "class",
                                     sc.raster.transform, weights=F["area_px"])
    assert rep["overall_accuracy"] > 0.85          # measured 0.91-0.99 across seeds; 1/6 would be chance
