"""Reference checks for every OBIA segmentation algorithm.

Each algorithm is checked against something that is not the code under test:

* a value worked out by hand (the arithmetic is written in the comment next to it),
* a deliberately naive re-implementation that follows the published definition literally (masks and ``np.std`` on
  small images, no incremental bookkeeping), or
* the scikit-image / scipy function the wrapper is supposed to call.

Images are tiny so that a naive implementation is fast and a failing case can be read by eye.
"""
import itertools

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("rasterio")
from scipy import ndimage as ndi  # noqa: E402

from pygeovision import obia  # noqa: E402
from pygeovision.obia.merging import STRATEGIES, RegionMerger, SpectralDifferenceMerger  # noqa: E402
from pygeovision.obia.splitting import contrast_split, multi_threshold, quadtree  # noqa: E402
from tests.obia_testdata import block_scene  # noqa: E402


def canon(lab):
    """Partition of the pixels as ids numbered in order of first appearance (independent of the id values)."""
    lab = np.asarray(lab)
    seen, out = {}, np.zeros(lab.shape, int)
    for idx, v in np.ndenumerate(lab):
        out[idx] = seen.setdefault(int(v), len(seen) + 1)
    return out


def same_partition(a, b):
    return np.array_equal(canon(a), canon(b))


def line(values):
    """(1 band, 1 row, n cols) image from a list of numbers."""
    return np.array(values, float)[None, None, :]


# ============================================================================================== Baatz-Schaepe
def naive_cost(img, lab, a, b, wc, wk, bw):
    """Fusion cost of objects ``a`` and ``b`` from the definition, recomputed from the pixel masks."""

    def stats(mask):
        n = int(mask.sum())
        sd = np.array([np.std(img[k][mask]) for k in range(img.shape[0])])
        padded = np.pad(mask, 1)
        # perimeter in pixel edges: every pixel side whose neighbour is outside the object (or outside the image)
        per = int(
            (padded[1:-1, 1:-1] & ~padded[:-2, 1:-1]).sum() + (padded[1:-1, 1:-1] & ~padded[2:, 1:-1]).sum()
            + (padded[1:-1, 1:-1] & ~padded[1:-1, :-2]).sum() + (padded[1:-1, 1:-1] & ~padded[1:-1, 2:]).sum()
        )
        r, c = np.where(mask)
        bbox = 2 * ((r.max() - r.min() + 1) + (c.max() - c.min() + 1))
        return n, sd, per, bbox

    ma, mb = lab == a, lab == b
    n1, s1, l1, b1 = stats(ma)
    n2, s2, l2, b2 = stats(mb)
    nm, sm, lm, bm = stats(ma | mb)
    h_color = float(np.dot(bw, nm * sm - (n1 * s1 + n2 * s2)))
    h_cmpct = nm * lm / np.sqrt(nm) - (n1 * l1 / np.sqrt(n1) + n2 * l2 / np.sqrt(n2))
    h_smooth = nm * lm / bm - (n1 * l1 / b1 + n2 * l2 / b2)
    h_shape = wk * h_cmpct + (1 - wk) * h_smooth
    return wc * h_color + (1 - wc) * h_shape


def naive_neighbours(lab):
    pairs = set()
    for x, y in zip(lab[:, :-1].ravel(), lab[:, 1:].ravel()):
        if x != y:
            pairs.add((min(x, y), max(x, y)))
    for x, y in zip(lab[:-1].ravel(), lab[1:].ravel()):
        if x != y:
            pairs.add((min(x, y), max(x, y)))
    return sorted(pairs)


def naive_multires(img, scale, wc, wk):
    """Textbook best-first Baatz-Schaepe merging from single pixels, everything recomputed at every step."""
    B, H, W = img.shape
    bw = np.full(B, 1.0 / B)
    lab = np.arange(H * W).reshape(H, W)
    while True:
        pairs = naive_neighbours(lab)
        if not pairs:
            return lab
        costs = [naive_cost(img, lab, a, b, wc, wk, bw) for a, b in pairs]
        k = int(np.argmin(costs))
        if costs[k] >= scale**2:
            return lab
        a, b = pairs[k]
        lab = np.where(lab == b, a, lab)


def test_cost_of_two_pixels_by_hand():
    # pixels 0 and 10, one band.  sd_merged = 5, n_merged = 2, single pixels have sd 0:
    #   h_color = 2*5 - 0 = 10
    img = line([0.0, 10.0])
    m = RegionMerger(img, np.arange(2).reshape(1, 2), color_weight=1.0)
    assert m._cost(0, 1, 1) == pytest.approx(10.0)
    # shape terms of a 1x2 block made of two 1x1 pixels (equal-valued, so h_color = 0):
    #   l_m = 6, l_1 = l_2 = 4, bbox_m = 2*(1+2) = 6, bbox_1 = bbox_2 = 4
    #   h_cmpct  = 2*6/sqrt(2) - 2*(1*4/sqrt(1)) = 8.4853 - 8 = 0.48528
    #   h_smooth = 2*6/6       - 2*(1*4/4)       = 2 - 2      = 0
    flat = line([7.0, 7.0])
    cmp_only = RegionMerger(flat, np.arange(2).reshape(1, 2), color_weight=0.0, compactness_weight=1.0)
    assert cmp_only._cost(0, 1, 1) == pytest.approx(12 / np.sqrt(2) - 8)
    smooth_only = RegionMerger(flat, np.arange(2).reshape(1, 2), color_weight=0.0, compactness_weight=0.0)
    assert smooth_only._cost(0, 1, 1) == pytest.approx(0.0)
    # w_color = 0.5, w_cmpct = 0.5 on the unequal pair: f = 0.5*10 + 0.5*(0.5*0.48528 + 0.5*0)
    mixed = RegionMerger(img, np.arange(2).reshape(1, 2), color_weight=0.5, compactness_weight=0.5)
    assert mixed._cost(0, 1, 1) == pytest.approx(0.5 * 10 + 0.5 * 0.5 * (12 / np.sqrt(2) - 8))


@pytest.mark.parametrize("wc,wk", [(1.0, 0.5), (0.9, 0.5), (0.5, 0.5), (0.5, 0.0), (0.2, 1.0)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multires_equals_naive_baatz_schaepe(seed, wc, wk):
    rng = np.random.default_rng(seed)
    # two noisy plateaus plus a bright dot, so that merges of very different cost occur
    img = rng.normal(0, 4, (2, 6, 7)) + 60
    img[:, :, 4:] += 70
    img[:, 2, 1] += 120
    for scale in (8.0, 25.0, 60.0):
        ref = naive_multires(img, scale, wc, wk)
        got = RegionMerger(img, np.arange(42).reshape(6, 7), wc, wk).run(scale).labels()
        assert same_partition(ref, got), (seed, wc, wk, scale)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_after_run_no_neighbour_pair_is_below_the_threshold(strategy):
    rng = np.random.default_rng(5)
    img = rng.uniform(0, 255, (3, 9, 9))
    for scale in (10.0, 40.0, 90.0):
        m = RegionMerger(img, np.arange(81).reshape(9, 9), 0.8, 0.5).run(scale, strategy)
        assert m.n_objects >= 1
        for a in np.flatnonzero(m.alive):
            for b, shared in m.adj[a].items():
                assert m._cost(min(a, b), max(a, b), shared) >= scale**2


def test_mutual_best_and_best_first_differ_by_hand():
    # spectral difference, max_difference = 3.5, values 4.3  2.8  6.5  9.5   (neighbour differences 1.5, 3.7, 3.0)
    # best first: 1.5 -> {4.3,2.8} mean 3.55;  |3.55-6.5| = 2.95 < 3.0 -> joins 6.5, mean 4.5333;
    #             |4.5333-9.5| = 4.97 > 3.5 stop  =>  {0,1,2} {3}
    # mutual best: 0<->1 (1.5) and 2<->3 (3.0 < 3.7) are mutual pairs, both merged in one pass;
    #             |3.55-8.0| = 4.45 > 3.5 stop    =>  {0,1} {2,3}
    img = line([4.3, 2.8, 6.5, 9.5])
    init = np.arange(4).reshape(1, 4)
    bf = SpectralDifferenceMerger(img, init).run(3.5, "best_first").labels()
    mb = SpectralDifferenceMerger(img, init).run(3.5, "mutual_best").labels()
    assert bf.tolist() == [[1, 1, 1, 2]]
    assert mb.tolist() == [[1, 1, 2, 2]]


def test_mutual_best_is_nested_across_scales_and_deterministic():
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 255, (2, 10, 10))
    init = np.arange(100).reshape(10, 10)
    m = RegionMerger(img, init, 0.9, 0.5)
    counts, snaps = [], []
    for s in (5.0, 15.0, 30.0, 60.0):
        m.run(s, "mutual_best")
        counts.append(m.n_objects)
        snaps.append(m.labels())
    assert counts == sorted(counts, reverse=True)
    for fine, coarse in zip(snaps, snaps[1:]):   # every fine object lies inside exactly one coarser object
        for v in np.unique(fine):
            assert len(np.unique(coarse[fine == v])) == 1
    again = RegionMerger(img, init, 0.9, 0.5).run(60.0, "mutual_best").labels()
    # one run to 60 may differ from the stepwise run, but the same call twice must agree
    assert np.array_equal(again, RegionMerger(img, init, 0.9, 0.5).run(60.0, "mutual_best").labels())


def test_merger_rejects_unknown_strategy():
    m = RegionMerger(line([1.0, 2.0]), np.arange(2).reshape(1, 2))
    with pytest.raises(ValueError, match="strategy"):
        m.run(5.0, "eager")
    with pytest.raises(ValueError):
        m.run(-1.0)


# ============================================================================================ spectral difference
def naive_spectral_difference(img, max_diff, metric, bw):
    """Merge the most similar neighbouring objects while their mean-spectrum difference is <= max_diff."""
    B, H, W = img.shape
    lab = np.arange(H * W).reshape(H, W)

    def diff(a, b):
        d = np.abs(np.array([img[k][lab == a].mean() - img[k][lab == b].mean() for k in range(B)]))
        return d[bw > 0].max() if metric == "max" else float(np.dot(bw / bw.sum(), d))

    while True:
        pairs = naive_neighbours(lab)
        if not pairs:
            return lab
        costs = [diff(a, b) for a, b in pairs]
        k = int(np.argmin(costs))
        if costs[k] > max_diff:
            return lab
        lab = np.where(lab == pairs[k][1], pairs[k][0], lab)


def test_spectral_difference_threshold_is_inclusive_and_by_hand():
    init = np.arange(2).reshape(1, 2)
    img = line([0.0, 10.0])
    assert SpectralDifferenceMerger(img, init).run(9.99).n_objects == 2
    assert SpectralDifferenceMerger(img, init).run(10.0).n_objects == 1
    # means are recomputed from all pixels: 0, 4, 7.5 -> the 3.5 pair merges first (mean 5.75),
    # then |0 - 5.75| = 5.75 > 4, so the 0 stays alone.  Comparing 0 with the *original* 4 would (wrongly) merge.
    m = SpectralDifferenceMerger(line([0.0, 4.0, 7.5]), np.arange(3).reshape(1, 3)).run(4.0)
    assert m.labels().tolist() == [[1, 2, 2]]


def test_spectral_difference_metric_and_band_weights_by_hand():
    # two bands, pixel A = (0, 0), pixel B = (10, 2): per-band differences 10 and 2
    img = np.array([[[0.0, 10.0]], [[0.0, 2.0]]])
    init = np.arange(2).reshape(1, 2)
    assert SpectralDifferenceMerger(img, init, metric="max").run(9.0).n_objects == 2      # max = 10
    assert SpectralDifferenceMerger(img, init, metric="max").run(10.0).n_objects == 1
    assert SpectralDifferenceMerger(img, init, metric="mean").run(5.9).n_objects == 2     # mean = (10+2)/2 = 6
    assert SpectralDifferenceMerger(img, init, metric="mean").run(6.0).n_objects == 1
    w = [1.0, 3.0]                                                                        # weighted mean = (10 + 3*2)/4 = 4
    assert SpectralDifferenceMerger(img, init, w, "mean").run(3.9).n_objects == 2
    assert SpectralDifferenceMerger(img, init, w, "mean").run(4.0).n_objects == 1
    assert SpectralDifferenceMerger(img, init, [0.0, 1.0], "max").run(2.0).n_objects == 1  # band 0 ignored
    with pytest.raises(ValueError, match="metric"):
        SpectralDifferenceMerger(img, init, metric="euclid")


@pytest.mark.parametrize("metric", ["max", "mean"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spectral_difference_equals_naive(seed, metric):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 100, (2, 6, 6))
    bw = np.array([1.0, 2.0])
    for md in (5.0, 15.0, 30.0):
        ref = naive_spectral_difference(img, md, metric, bw)
        got = SpectralDifferenceMerger(img, np.arange(36).reshape(6, 6), bw, metric).run(md).labels()
        assert same_partition(ref, got), (seed, metric, md)


# ===================================================================================================== quadtree
def naive_quadtree(img, scale, min_size, max_size, lab, r0, r1, c0, c1, counter):
    h, w = r1 - r0, c1 - c0
    block = img[:, r0:r1, c0:c1]
    too_big = max_size is not None and (h > max_size or w > max_size)
    heterogeneous = too_big or max(np.std(block[k]) for k in range(img.shape[0])) > scale
    cut_r = h >= 2 and h // 2 >= min_size
    cut_c = w >= 2 and w // 2 >= min_size
    if not heterogeneous or not (cut_r or cut_c):
        counter[0] += 1
        lab[r0:r1, c0:c1] = counter[0]
        return
    rs = [(r0, r0 + h // 2), (r0 + h // 2, r1)] if cut_r else [(r0, r1)]
    cs = [(c0, c0 + w // 2), (c0 + w // 2, c1)] if cut_c else [(c0, c1)]
    for a, b in rs:
        for c, d in cs:
            naive_quadtree(img, scale, min_size, max_size, lab, a, b, c, d, counter)


@pytest.mark.parametrize("shape", [(16, 16), (13, 21), (1, 9), (7, 1)])
@pytest.mark.parametrize("scale,min_size,max_size", [(5.0, 1, None), (30.0, 2, None), (10.0, 1, 4), (0.0, 3, None)])
def test_quadtree_equals_recursive_reference(shape, scale, min_size, max_size):
    rng = np.random.default_rng(7)
    H, W = shape
    img = rng.uniform(0, 255, (2, H, W))
    img[:, : H // 2, : W // 2] = 100 + rng.normal(0, 3, (2, H // 2, W // 2))   # a calm corner
    ref = np.zeros((H, W), int)
    naive_quadtree(img, scale, min_size, max_size, ref, 0, H, 0, W, [0])
    got = quadtree(img, scale, min_size, max_size)
    assert np.array_equal(canon(ref), canon(got))


def test_quadtree_by_hand():
    flat = np.full((2, 8, 8), 5.0)
    assert quadtree(flat, 0.0).max() == 1                                   # nothing to split
    half = np.zeros((1, 8, 8))
    half[:, :, 4:] = 100                                                    # std of the whole = 50 > 5 -> four 4x4
    got = quadtree(half, 5.0)                                               # quarters, each flat -> 4 leaves
    assert got.max() == 4
    assert (got[:4, :4] == got[0, 0]).all() and got[0, 0] != got[0, 4]
    # alternating 0 / 2 columns: population std is exactly 1 -> kept at scale 1, split below it
    alt = np.zeros((1, 4, 4))
    alt[:, :, 1::2] = 2
    assert quadtree(alt, 1.0).max() == 1
    assert quadtree(alt, 0.99).max() > 1
    # min_size: an 8x8 noisy image can only be cut into 4x4 blocks with min_size=4, never smaller
    noise = np.random.default_rng(0).uniform(0, 255, (1, 8, 8))
    assert quadtree(noise, 0.0, min_size=4).max() == 4
    assert quadtree(noise, 0.0).max() == 64
    # max_size forces cuts even in flat areas: 8x8 flat, blocks of at most 2 -> 16 leaves
    assert quadtree(flat, 100.0, max_size=2).max() == 16
    # a band with weight 0 is ignored by the homogeneity test
    two = np.zeros((2, 4, 4))
    two[1] = np.random.default_rng(1).uniform(0, 255, (4, 4))
    assert quadtree(two, 5.0, band_weights=[1, 0]).max() == 1
    assert quadtree(two, 5.0, band_weights=[0, 1]).max() > 1


def test_quadtree_rejects_bad_input():
    img = np.zeros((1, 4, 4))
    for kw in ({"scale": -1}, {"min_size": 0}, {"min_size": 4, "max_size": 2}, {"band_weights": [0.0]}):
        with pytest.raises(ValueError):
            quadtree(img, **{"scale": 1.0, **kw})
    with pytest.raises(ValueError):
        quadtree(np.zeros(5))


def test_quadtree_on_the_block_scene_is_what_the_construction_implies():
    # 64 x 64 = four 32 x 32 quadrants.  Only blue, green and red count (NIR carries the forest checkerboard) and
    # scale = 5 is far above the +-0.5 texture and far below the contrast between blocks, so water, cropland and
    # forest stay single leaves.  In the bare-field quadrant the 8 x 8 building sits at rows 40-47, cols 8-15:
    #   the 16 x 16 block rows 32-47 / cols 0-15 is mixed (blue std = 55 * sqrt(.25 * .75) = 23.8 > 5)
    #   -> four 8 x 8 leaves, one of which is exactly the building; the other three 16 x 16 blocks are flat.
    #   leaves: 3 quadrants + (3 + 4) bare-field blocks + ... = 3 + 7 = 10
    sc = block_scene(64)
    seg = obia.segment(sc.raster, "quadtree", scale=5.0, band_weights=[1, 1, 1, 0], rescale=False)
    assert seg.n_objects == 10
    building = seg.labels[40:48, 8:16]
    assert (building == building[0, 0]).all() and (seg.labels == building[0, 0]).sum() == 64
    assert len(np.unique(seg.labels[:32, :32])) == 1 and len(np.unique(seg.labels[32:, 32:])) == 1
    assert len(np.unique(seg.labels[:32, 32:])) == 1
    assert seg.params["scale"] == 5.0


# ============================================================================================== multi-threshold
def test_multi_threshold_by_hand():
    lay = np.array([[10.0, 10, 50, 50, 90, 90],
                    [10.0, 10, 50, 50, 90, 90]])
    lab, thr = multi_threshold(lay, thresholds=[30, 70])
    assert thr == (30.0, 70.0)
    assert lab.tolist() == [[1, 1, 2, 2, 3, 3]] * 2
    # a value equal to a threshold belongs to the class above it
    lab, _ = multi_threshold(np.array([[0.0, 30.0, 30.0, 100.0]]), thresholds=[30])
    assert lab.tolist() == [[1, 2, 2, 2]]
    # two separate patches of one class are two objects (4-connectivity: touching only at a corner is not joined)
    diag = np.array([[0.0, 1.0], [1.0, 0.0]])
    lab, _ = multi_threshold(diag, thresholds=[0.5])
    assert lab.max() == 4
    far = np.array([[0.0, 1, 0, 1, 0]])
    assert multi_threshold(far, thresholds=[0.5])[0].max() == 5


def test_multi_otsu_matches_exhaustive_search():
    # three clusters of integers; the exhaustive search maximises the between-class variance over all threshold pairs
    rng = np.random.default_rng(2)
    lay = np.concatenate([rng.integers(10, 30, 120), rng.integers(100, 130, 100), rng.integers(200, 230, 80)])
    rng.shuffle(lay)
    lay = lay.reshape(20, 15).astype(float)
    vals = lay.ravel()
    best, best_pair = -1, None
    for t1, t2 in itertools.combinations(range(1, 256), 2):
        cls = np.digitize(vals, [t1, t2])
        if len(np.unique(cls)) < 3:
            continue
        mu = vals.mean()
        bcv = sum((cls == k).sum() * (vals[cls == k].mean() - mu) ** 2 for k in range(3))
        if bcv > best:
            best, best_pair = bcv, (t1, t2)
    expected = np.digitize(lay, best_pair)
    lab, thr = multi_threshold(lay, n_classes=3)
    got_classes = np.digitize(lay, thr)
    assert np.array_equal(got_classes, expected)
    # and the objects are the connected patches of those classes
    ref = np.zeros(lay.shape, int)
    n = 0
    for k in range(3):
        comp, c = ndi.label(expected == k)
        ref[comp > 0] = comp[comp > 0] + n
        n += c
    assert np.array_equal(canon(ref), canon(lab))


def test_multi_threshold_rejects_bad_input():
    lay = np.array([[0.0, 1.0, 2.0]])
    for kw in ({"thresholds": []}, {"thresholds": [2, 1]}, {"thresholds": [1, 1]}, {"n_classes": 1},
               {"n_classes": 5}):
        with pytest.raises(ValueError):
            multi_threshold(lay, **kw)
    with pytest.raises(ValueError):
        multi_threshold(np.zeros(4))


def test_multi_threshold_recovers_the_block_scene_classes():
    # mean brightness (unstretched) per block: water 36, forest 57.5 (54-61 with the checkerboard), cropland 82,
    # bare field 118, building 144 -> cuts at 45, 70, 100, 130 give exactly the five reference blocks
    sc = block_scene(64)
    seg = obia.segment(sc.raster, "multi_threshold", layer="brightness", thresholds=[45, 70, 100, 130], rescale=False)
    assert same_partition(seg.labels, sc.ref_labels)
    assert seg.params["thresholds"] == [45.0, 70.0, 100.0, 130.0]
    # multi-Otsu finds its own cuts; they must give back 5 classes with the right number of thresholds
    auto = obia.segment(sc.raster, "multi_threshold", layer="brightness", n_classes=4, rescale=False)
    assert len(auto.params["thresholds"]) == 3


def test_multi_threshold_on_named_layers():
    sc = block_scene(64)
    # NDVI = (nir-red)/(nir+red): water ~0.4, cropland 0.62, bare ~-0.04, forest ~0.65, building ~-0.3
    ndvi = obia.spectral_layers(sc.raster)["ndvi"]
    seg = obia.segment(sc.raster, "multi_threshold", layer="ndvi", thresholds=[0.0, 0.5])
    cls = np.digitize(ndvi, [0.0, 0.5])
    ref = np.zeros(cls.shape, int)
    n = 0
    for k in np.unique(cls):
        comp, c = ndi.label(cls == k)
        ref[comp > 0] = comp[comp > 0] + n
        n += c
    assert same_partition(seg.labels, ref)
    assert seg.params["layer"] == "ndvi" and seg.params["thresholds"] == [0.0, 0.5]
    assert obia.segment(sc.raster, "multi_threshold", layer="nir", n_classes=2).n_objects >= 2
    assert obia.segment(sc.raster, "multi_threshold", layer=3, n_classes=2).n_objects >= 2
    with pytest.raises(ValueError, match="unknown layer"):
        obia.segment(sc.raster, "multi_threshold", layer="swir9")
    with pytest.raises(ValueError, match="out of range"):
        obia.segment(sc.raster, "multi_threshold", layer=9)
    no_nir = obia.RasterData(sc.raster.image[:3], ["blue", "green", "red"], sc.raster.transform, sc.raster.crs)
    with pytest.raises(ValueError, match="ndvi"):
        obia.segment(no_nir, "multi_threshold", layer="ndvi")


# ================================================================================================ contrast split
def naive_best_threshold(vals, steps, min_contrast=0.0, min_size=1):
    """Edge-contrast threshold for one object given as a 2-D array with NaN outside it (loops over pixel pairs)."""
    H, W = vals.shape
    ok = ~np.isnan(vals)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pairs = []
    for i in range(H):
        for j in range(W):
            if not ok[i, j]:
                continue
            if j + 1 < W and ok[i, j + 1]:
                pairs.append((vals[i, j], vals[i, j + 1]))
            if i + 1 < H and ok[i + 1, j]:
                pairs.append((vals[i, j], vals[i + 1, j]))
    best, best_t = -np.inf, None
    for k in range(1, steps + 1):
        t = lo + (hi - lo) * k / (steps + 1)
        straddle = [abs(a - b) for a, b in pairs if min(a, b) <= t < max(a, b)]
        dark = int((vals[ok] <= t).sum())
        bright = int(ok.sum()) - dark
        if not straddle or dark < min_size or bright < min_size:
            continue
        c = float(np.mean(straddle))
        if c >= min_contrast and c > best + 1e-12:
            best, best_t = c, t
    return best_t


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("steps", [1, 5, 32])
def test_single_split_equals_naive_edge_contrast(seed, steps):
    rng = np.random.default_rng(seed)
    lay = np.round(rng.uniform(0, 100, (7, 8)), 1)
    lay[2:5, 2:6] += 80                                                       # a bright patch
    t = naive_best_threshold(lay, steps)
    expected = ndi.label(lay <= t)[0] + 0, ndi.label(lay > t)[0]
    ref = np.zeros(lay.shape, int)
    ref[lay <= t] = expected[0][lay <= t]
    ref[lay > t] = expected[1][lay > t] + expected[0].max()
    got = contrast_split(lay, levels=1, steps=steps)
    assert np.array_equal(canon(ref), canon(got))


def test_contrast_split_three_stripes_by_hand():
    # columns 20 | 100 | 200.  Thresholds between 20 and 100 cut only the 20|100 edge (contrast 80), thresholds
    # between 100 and 200 cut only the 100|200 edge (contrast 100) -> first cut {20,100} | {200}, second cut 20 | 100.
    lay = np.zeros((6, 9))
    lay[:, 3:6] = 100
    lay[:, 6:] = 200
    lay[:, :3] = 20
    one = contrast_split(lay, levels=1)
    assert one.max() == 2 and (one[:, :6] == one[0, 0]).all() and (one[:, 6:] == one[0, 8]).all()
    two = contrast_split(lay, levels=2)
    assert two.tolist() == [[1, 1, 1, 2, 2, 2, 3, 3, 3]] * 6
    assert contrast_split(lay, levels=5).max() == 3                          # flat objects cannot be split further


def test_contrast_split_stopping_rules():
    lay = np.zeros((6, 9))
    lay[:, 3:6] = 100
    lay[:, 6:] = 200
    lay[:, :3] = 20
    # best contrast of the first cut is 100: a floor above it prevents any split, one below allows it
    assert contrast_split(lay, levels=1, min_contrast=100.5).max() == 1
    assert contrast_split(lay, levels=1, min_contrast=100.0).max() == 2
    # min_size: one isolated bright pixel cannot be a side of its own when each side needs 2 pixels
    dot = np.zeros((5, 5))
    dot[2, 2] = 50
    assert contrast_split(dot, levels=1, min_size=1).max() == 2
    assert contrast_split(dot, levels=1, min_size=2).max() == 1
    # a constant layer has nothing to split
    assert contrast_split(np.full((4, 4), 3.0), levels=3).max() == 1


def test_contrast_split_objects_are_connected_and_respect_init():
    lay = np.zeros((6, 6))
    lay[:, 0] = 100
    lay[:, 5] = 100                                                          # two bright columns that do not touch
    out = contrast_split(lay, levels=1)
    assert out.max() == 3 and out[0, 0] != out[0, 5]                          # the two bright patches are two objects
    init = np.zeros((6, 6), int)
    init[:, :3] = 1
    init[:, 3:] = 2
    lay2 = np.zeros((6, 6))
    lay2[:3, :] = 10                                                          # splits both halves horizontally
    lay2[3:, :] += 90
    got = contrast_split(lay2, levels=1, init=init)
    assert got.max() == 4 and got[0, 0] != got[0, 5] and got[0, 0] != got[5, 0]
    init0 = init.copy()
    init0[:, 3:] = 0                                                          # label 0 = outside, never touched
    got = contrast_split(lay2, levels=1, init=init0)
    assert (got[:, 3:] == 0).all() and got[:, :3].min() > 0


def test_contrast_split_rejects_bad_input():
    lay = np.zeros((3, 3))
    for kw in ({"levels": 0}, {"steps": 0}, {"min_size": 0}, {"init": np.ones((2, 2), int)}):
        with pytest.raises(ValueError):
            contrast_split(lay, **kw)
    with pytest.raises(ValueError):
        contrast_split(np.zeros(4))


def test_contrast_split_on_the_block_scene_water_is_cut_from_the_bright_blocks():
    # NIR (raw): water 12 is by far the darkest block, every other block is >= 90.  The first cut therefore has to
    # separate the water block from everything else (edge contrast ~80-160 there, <= 54 elsewhere).
    sc = block_scene(64)
    seg = obia.segment(sc.raster, "contrast_split", layer="nir", levels=1, rescale=False)
    water = seg.labels[:32, :32]
    assert len(np.unique(water)) == 1
    assert seg.labels[0, 0] not in np.unique(seg.labels[seg.labels != seg.labels[0, 0]])
    assert seg.n_objects == 2


# ======================================================================================= wrappers of library code
def _prep(rd):
    from pygeovision.obia.io import rescale_bands

    return rescale_bands(rd.image)


def _hwc(img):
    return np.moveaxis(img, 0, -1) / 255.0


def test_felzenszwalb_wrapper_equals_skimage():
    from skimage.segmentation import felzenszwalb

    sc = block_scene(64)
    ref = felzenszwalb(_hwc(_prep(sc.raster)), scale=40.0, sigma=0.8, min_size=20, channel_axis=-1)
    got = obia.segment(sc.raster, "felzenszwalb", scale=40.0, sigma=0.8, min_size=20).labels
    ref_conn = np.zeros(ref.shape, int)
    n = 0
    for v in np.unique(ref):
        comp, c = ndi.label(ref == v)
        ref_conn[comp > 0] = comp[comp > 0] + n
        n += c
    assert same_partition(ref_conn, got)


def test_slic_and_quickshift_wrappers_equal_skimage():
    from skimage.segmentation import quickshift, slic

    sc = block_scene(64)
    img = _hwc(_prep(sc.raster))

    def connected(lab):
        out = np.zeros(lab.shape, int)
        n = 0
        for v in np.unique(lab):
            comp, c = ndi.label(lab == v)
            out[comp > 0] = comp[comp > 0] + n
            n += c
        return out

    got = obia.segment(sc.raster, "slic", n_segments=60, compactness=0.1, sigma=1.0).labels
    ref = slic(img, n_segments=60, compactness=0.1, sigma=1.0, channel_axis=-1, start_label=1)
    assert same_partition(connected(ref), got)
    got = obia.segment(sc.raster, "quickshift", kernel_size=3, max_dist=6.0, ratio=0.5).labels
    ref = quickshift(img, kernel_size=3, max_dist=6.0, ratio=0.5, sigma=0.0, convert2lab=False, channel_axis=-1)
    assert same_partition(connected(ref), got)


def test_chessboard_by_hand():
    # 5 rows x 7 cols, tiles of 2: tile rows {0,1}{2,3}{4}, tile cols {0,1}{2,3}{4,5}{6}  -> 3 x 4 = 12 tiles
    lab = obia.segment(np.zeros((1, 5, 7)) + np.arange(35).reshape(5, 7), "chessboard", size=2).labels
    assert lab.max() == 12
    assert lab[0].tolist() == [1, 1, 2, 2, 3, 3, 4]
    assert lab[4].tolist() == [9, 9, 10, 10, 11, 11, 12]
    assert lab[1].tolist() == lab[0].tolist() and lab[2].tolist() == [5, 5, 6, 6, 7, 7, 8]


# ============================================================================================ through segment()
NEW_METHODS = {
    "quadtree": {"scale": 20.0},
    "multi_threshold": {"n_classes": 3},
    "contrast_split": {"levels": 3, "min_contrast": 10.0},
    "spectral_difference": {"max_difference": 12.0},
}


@pytest.mark.parametrize("method", sorted(NEW_METHODS))
def test_new_methods_mask_invalid_pixels_and_are_connected(method):
    sc = block_scene(64)
    img = sc.raster.image.copy()
    img[:, 5:9, 5:9] = np.nan                                                # a "cloud"
    rd = obia.RasterData(img, sc.raster.band_names, sc.raster.transform, sc.raster.crs)
    seg = obia.segment(rd, method, **NEW_METHODS[method])
    assert (seg.labels[5:9, 5:9] == 0).all() and (seg.labels[seg.valid] > 0).all()
    for oid in range(1, seg.n_objects + 1):
        assert ndi.label(seg.labels == oid)[1] == 1                          # every object is one connected piece


@pytest.mark.parametrize("method", sorted(NEW_METHODS))
def test_new_methods_are_registered_and_record_their_parameters(method):
    assert method in obia.METHODS
    seg = obia.segment(block_scene(32).raster, method, **NEW_METHODS[method])
    for k, v in NEW_METHODS[method].items():
        assert seg.params[k] == v and seg.method == method


def test_spectral_difference_and_multires_strategies_through_segment():
    sc = block_scene(32)
    for strategy in STRATEGIES:
        a = obia.segment(sc.raster, "multires", scale=20.0, strategy=strategy)
        assert a.params["strategy"] == strategy and a.n_objects >= 5
        b = obia.segment(sc.raster, "spectral_difference", max_difference=15.0, strategy=strategy)
        assert b.n_objects >= 5
    with pytest.raises(ValueError, match="strategy"):
        obia.segment(sc.raster, "multires", strategy="nope")
    # blue/green/red only (NIR carries the forest checkerboard): blocks differ by >= 30 there, pixels inside a block by <= 1;
    # unstretched, because the default percentile stretch would clip the small building (1.6% of the pixels)
    seg = obia.segment(sc.raster, "spectral_difference", max_difference=10.0, init="pixels", band_weights=[1, 1, 1, 0],
                       rescale=False)
    assert same_partition(seg.labels, sc.ref_labels)
