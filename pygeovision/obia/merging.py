"""Region merging with a scale parameter (multiresolution-style segmentation).

Idea
----
Start from many tiny objects (single pixels, or a fine over-segmentation) and
repeatedly merge the pair of neighbouring objects whose union increases
"heterogeneity" the least.  Stop when the cheapest available merge would cost
more than ``scale**2``.  A small scale therefore gives many small objects, a
large scale gives few large ones - the single most important OBIA knob.

Heterogeneity follows Baatz & Schäpe (2000), the criterion behind the
"multiresolution segmentation" taught in most OBIA courses::

    h_color  = sum_b w_b * ( n_m*sd_m,b - (n_1*sd_1,b + n_2*sd_2,b) )
    h_cmpct  = n_m*l_m/sqrt(n_m)  - ( n_1*l_1/sqrt(n_1) + n_2*l_2/sqrt(n_2) )
    h_smooth = n_m*l_m/b_m        - ( n_1*l_1/b_1       + n_2*l_2/b_2 )
    h_shape  = w_cmpct*h_cmpct + (1 - w_cmpct)*h_smooth
    f        = w_color*h_color + (1 - w_color)*h_shape        merge if f < scale**2

(``n`` pixel count, ``sd`` standard deviation, ``l`` perimeter in pixel edges,
``b`` perimeter of the bounding box; ``_1``/``_2`` the two objects, ``_m`` their
union.)

What this is NOT
----------------
This is an independent implementation of the published criterion, not a
re-implementation of eCognition's proprietary algorithm.  Two differences
matter for teaching:

* eCognition grows objects in passes using *local mutual best fitting*; here
  the globally cheapest merge is always performed next ("best-first").
* By default the process starts from a fine SLIC over-segmentation instead of
  single pixels (single pixels are slow in pure Python).  ``init="pixels"``
  gives the textbook behaviour for small images.

Scale values are therefore *comparable in meaning* to textbook values (bands
are stretched to 0-255) but will not reproduce eCognition's object boundaries
exactly.  Say so when comparing against eCognition results.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

from ._adjacency import border_pairs, image_edge_lengths


class RegionMerger:
    """Incremental region merging.

    Because merging is monotone in the threshold, one merger can be run to
    several increasing scales and a snapshot taken at each - this is how the
    nested *levels* of an object hierarchy and the scale diagnostics are
    produced cheaply.

    Parameters
    ----------
    image : (bands, rows, cols) array, ideally already rescaled to 0-255.
    init_labels : (rows, cols) int array of the starting objects.
    color_weight : ``w_color`` in [0, 1].  1.0 = spectral criterion only.
    compactness_weight : ``w_cmpct`` in [0, 1] (ignored if color_weight = 1).
    band_weights : per-band ``w_b`` (default: equal, summing to 1).
    """

    def __init__(self, image, init_labels, color_weight=0.9,
                 compactness_weight=0.5, band_weights=None):
        if not 0.0 <= color_weight <= 1.0:
            raise ValueError("color_weight must be in [0, 1]")
        if not 0.0 <= compactness_weight <= 1.0:
            raise ValueError("compactness_weight must be in [0, 1]")
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 3:
            raise ValueError("image must be (bands, rows, cols)")
        self.B, self.H, self.W = img.shape
        init = np.asarray(init_labels)
        if init.shape != (self.H, self.W):
            raise ValueError("init_labels shape does not match the image")

        # relabel to 0..N-1
        uniq, inv = np.unique(init, return_inverse=True)
        self._init = inv.reshape(self.H, self.W).astype(np.int64)
        N = len(uniq)
        self.N = N
        flat = self._init.ravel()

        bw = np.ones(self.B) if band_weights is None else np.asarray(band_weights, float)
        if bw.shape != (self.B,) or (bw < 0).any() or bw.sum() == 0:
            raise ValueError("band_weights must be non-negative, one per band")
        self.bw = bw / bw.sum()
        self.wc = float(color_weight)
        self.wk = float(compactness_weight)

        self.n = np.bincount(flat, minlength=N).astype(np.float64)
        self.s = np.empty((N, self.B))
        self.ss = np.empty((N, self.B))
        for b in range(self.B):
            v = img[b].ravel()
            self.s[:, b] = np.bincount(flat, weights=v, minlength=N)
            self.ss[:, b] = np.bincount(flat, weights=v * v, minlength=N)
        self.sig = self._sigma(self.n[:, None], self.s, self.ss)

        # bounding boxes
        sl = ndimage.find_objects(self._init + 1)
        self.r0 = np.array([x[0].start for x in sl])
        self.r1 = np.array([x[0].stop for x in sl])
        self.c0 = np.array([x[1].start for x in sl])
        self.c1 = np.array([x[1].stop for x in sl])

        # adjacency + perimeter
        a, b_, ln = border_pairs(self._init)
        self.adj: list[dict[int, int]] = [dict() for _ in range(N)]
        per = image_edge_lengths(self._init, N).astype(np.float64)
        for x, y, l in zip(a.tolist(), b_.tolist(), ln.tolist()):
            self.adj[x][y] = l
            self.adj[y][x] = l
            per[x] += l
            per[y] += l
        self.l = per

        self.alive = np.ones(N, dtype=bool)
        self.ver = np.zeros(N, dtype=np.int64)
        self.parent = np.arange(N)
        self.scale_reached = 0.0
        self.n_merges = 0

        self._heap: list[tuple] = []
        for x, y, l in zip(a.tolist(), b_.tolist(), ln.tolist()):
            self._push(x, y, l)

    # ------------------------------------------------------------------ math
    @staticmethod
    def _sigma(n, s, ss):
        mean = s / n
        return np.sqrt(np.maximum(ss / n - mean * mean, 0.0))

    def _bbox_perim(self, r0, r1, c0, c1):
        return 2.0 * ((r1 - r0) + (c1 - c0))

    def _cost(self, a: int, b: int, shared: int) -> float:
        na, nb = self.n[a], self.n[b]
        nm = na + nb
        sig_m = self._sigma(nm, self.s[a] + self.s[b], self.ss[a] + self.ss[b])
        h_color = float(np.dot(self.bw, nm * sig_m - na * self.sig[a] - nb * self.sig[b]))
        if self.wc >= 1.0:
            return h_color
        lm = self.l[a] + self.l[b] - 2.0 * shared
        r0, r1 = min(self.r0[a], self.r0[b]), max(self.r1[a], self.r1[b])
        c0, c1 = min(self.c0[a], self.c0[b]), max(self.c1[a], self.c1[b])
        bm = self._bbox_perim(r0, r1, c0, c1)
        ba = self._bbox_perim(self.r0[a], self.r1[a], self.c0[a], self.c1[a])
        bb = self._bbox_perim(self.r0[b], self.r1[b], self.c0[b], self.c1[b])
        h_cmpct = nm * lm / math.sqrt(nm) - (na * self.l[a] / math.sqrt(na)
                                              + nb * self.l[b] / math.sqrt(nb))
        h_smooth = nm * lm / bm - (na * self.l[a] / ba + nb * self.l[b] / bb)
        h_shape = self.wk * h_cmpct + (1.0 - self.wk) * h_smooth
        return self.wc * h_color + (1.0 - self.wc) * h_shape

    def _push(self, a: int, b: int, shared: int):
        if a > b:
            a, b = b, a
        heapq.heappush(self._heap, (self._cost(a, b, shared), a, b,
                                    int(self.ver[a]), int(self.ver[b])))

    # --------------------------------------------------------------- merging
    def _merge(self, a: int, b: int):
        shared = self.adj[a].pop(b)
        self.adj[b].pop(a)
        self.n[a] += self.n[b]
        self.s[a] += self.s[b]
        self.ss[a] += self.ss[b]
        self.sig[a] = self._sigma(self.n[a], self.s[a], self.ss[a])
        self.l[a] = self.l[a] + self.l[b] - 2.0 * shared
        self.r0[a] = min(self.r0[a], self.r0[b])
        self.r1[a] = max(self.r1[a], self.r1[b])
        self.c0[a] = min(self.c0[a], self.c0[b])
        self.c1[a] = max(self.c1[a], self.c1[b])
        for nb, sh in self.adj[b].items():
            del self.adj[nb][b]
            tot = self.adj[a].get(nb, 0) + sh
            self.adj[a][nb] = tot
            self.adj[nb][a] = tot
        self.adj[b] = {}
        self.alive[b] = False
        self.parent[b] = a
        self.ver[a] += 1
        self.n_merges += 1
        for nb, sh in self.adj[a].items():
            self._push(a, nb, sh)

    def run(self, scale: float) -> "RegionMerger":
        """Merge until the cheapest remaining merge costs >= ``scale**2``."""
        if scale < 0:
            raise ValueError("scale must be >= 0")
        thr = float(scale) ** 2
        heap = self._heap
        while heap:
            cost, a, b, va, vb = heap[0]
            if not (self.alive[a] and self.alive[b]
                    and self.ver[a] == va and self.ver[b] == vb):
                heapq.heappop(heap)
                continue
            if cost >= thr:
                break
            heapq.heappop(heap)
            self._merge(a, b)
        self.scale_reached = max(self.scale_reached, float(scale))
        return self

    # --------------------------------------------------------------- results
    @property
    def n_objects(self) -> int:
        return int(self.alive.sum())

    def labels(self) -> np.ndarray:
        """Current objects as an int32 raster numbered 1..n_objects."""
        root = self.parent.copy()
        # path compression until fixed point
        while True:
            nxt = root[root]
            if np.array_equal(nxt, root):
                break
            root = nxt
        alive_ids = np.flatnonzero(self.alive)
        remap = np.zeros(self.N, dtype=np.int64)
        remap[alive_ids] = np.arange(1, len(alive_ids) + 1)
        return remap[root][self._init].astype(np.int32)
