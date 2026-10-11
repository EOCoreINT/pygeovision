"""Terrain hydrology on a DEM grid: depression filling, D8 flow, flow accumulation, streams, Strahler order, HAND.

Plain NumPy / Python (no extra dependencies). The algorithms are the standard ones:

* **Priority-Flood + Epsilon** (Barnes, Lehman & Mulla 2014): fills depressions *and* resolves flats in one pass, so
  every cell ends up with at least one strictly lower neighbour and the flow network has no cycles.
* **D8 steepest descent** (O'Callaghan & Mark 1984) on the conditioned DEM.
* **Flow accumulation** by processing cells from high to low.
* **Strahler stream order** (Strahler 1957).
* **HAND -- Height Above the Nearest Drainage** (Rennó et al. 2008): elevation above the stream cell a cell drains to.

Cost is a few Python operations per cell (about a minute for 5 million cells); ``check_size`` refuses grids that
would not fit in memory and says how to shrink them.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)
#: (d_row, d_col, distance in cells)
NEIGHBOURS = ((-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2))


def check_size(shape: tuple[int, int], max_cells: int) -> None:
    n = shape[0] * shape[1]
    if n > max_cells:
        raise ValueError(f"the area has {n / 1e6:.1f} million cells; the limit for the flow analysis is {max_cells / 1e6:.1f} "
                         f"million. Use a smaller bounding box or a coarser resolution_m.")


def priority_flood_epsilon(dem: np.ndarray) -> np.ndarray:
    """Depression-filled copy of ``dem`` in which no cell is a pit or a flat (NaN cells are outlets).

    Cells on the grid border, and cells next to NaN, are where water can leave.
    """
    h, w = dem.shape
    flat = dem.astype("float64").ravel().tolist()
    nan = np.isnan(dem).ravel().tolist()
    out = list(flat)
    closed = bytearray(h * w)
    heap: list[tuple[float, int]] = []
    for i in range(h * w):
        if nan[i]:
            closed[i] = 1
    nextafter = math.nextafter
    inf = math.inf
    for r in range(h):
        for c in range(w):
            i = r * w + c
            if closed[i]:
                continue
            edge = r == 0 or c == 0 or r == h - 1 or c == w - 1
            if not edge:
                for dr, dc, _ in NEIGHBOURS:
                    if nan[(r + dr) * w + c + dc]:
                        edge = True
                        break
            if edge:
                closed[i] = 1
                heapq.heappush(heap, (flat[i], i))
    while heap:
        z, i = heapq.heappop(heap)
        r, c = divmod(i, w)
        for dr, dc, _ in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= h or cc >= w:
                continue
            j = rr * w + cc
            if closed[j]:
                continue
            closed[j] = 1
            zn = flat[j]
            if zn <= z:
                zn = nextafter(z, inf)
            out[j] = zn
            heapq.heappush(heap, (zn, j))
    return np.asarray(out, dtype="float64").reshape(h, w)


def d8_targets(filled: np.ndarray, cell_size: float, chunk_rows: int = 512) -> np.ndarray:
    """Flat index of the cell each cell drains to (steepest strictly-downhill neighbour), -1 if it leaves the grid.

    NaN cells get -1. A cell with no lower valid neighbour (only possible on the border or next to NaN) drains out.
    """
    h, w = filled.shape
    target = np.full(h * w, -1, dtype=np.int64)
    padded = np.pad(filled, 1, constant_values=np.nan)
    for r0 in range(0, h, chunk_rows):
        r1 = min(h, r0 + chunk_rows)
        z = padded[1 + r0:1 + r1, 1:-1]
        best = np.zeros(z.shape)
        arg = np.full(z.shape, -1, dtype=np.int64)
        for dr, dc, dist in NEIGHBOURS:
            zn = padded[1 + r0 + dr:1 + r1 + dr, 1 + dc:1 + dc + w]
            with np.errstate(invalid="ignore"):
                drop = (z - zn) / (dist * cell_size)
                better = np.isfinite(zn) & (zn < z) & (drop > best)
            best = np.where(better, drop, best)
            rows = np.arange(r0, r1)[:, None] + dr
            cols = np.arange(w)[None, :] + dc
            arg = np.where(better, rows * w + cols, arg)
        target[r0 * w:r1 * w] = arg.ravel()
    target[np.isnan(filled).ravel()] = -1
    return target


def flow_accumulation(filled: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Number of cells (including itself) that drain through each cell."""
    h, w = filled.shape
    flat = np.nan_to_num(filled.ravel(), nan=-np.inf)
    order = np.argsort(-flat, kind="stable").tolist()
    tgt = target.tolist()
    acc = [1.0] * (h * w)
    for i in order:
        t = tgt[i]
        if t >= 0:
            acc[t] += acc[i]
    out = np.asarray(acc, dtype="float64").reshape(h, w)
    out[np.isnan(filled)] = np.nan
    return out


def strahler_order(target: np.ndarray, stream: np.ndarray, high_to_low: np.ndarray) -> np.ndarray:
    """Strahler order of every stream cell (0 elsewhere).

    ``stream`` is a flat boolean array, ``high_to_low`` the cell indices sorted from highest to lowest elevation
    (so every cell is visited after all the cells that drain into it).
    """
    n = target.size
    tgt = target.tolist()
    is_stream = stream.tolist()
    max_in = [0] * n
    n_max = [0] * n
    order = [0] * n
    for i in high_to_low.tolist():
        if not is_stream[i]:
            continue
        o = 1 if max_in[i] == 0 else (max_in[i] if n_max[i] == 1 else max_in[i] + 1)
        order[i] = o
        t = tgt[i]
        if t >= 0 and is_stream[t]:
            if o > max_in[t]:
                max_in[t], n_max[t] = o, 1
            elif o == max_in[t]:
                n_max[t] += 1
    return np.asarray(order, dtype="int16")


def height_above_drainage(filled: np.ndarray, target: np.ndarray, stream: np.ndarray) -> np.ndarray:
    """HAND: elevation of each cell above the stream cell it drains into (0 on streams, NaN if it never reaches one)."""
    h, w = filled.shape
    flat = np.nan_to_num(filled.ravel(), nan=np.inf)
    low_to_high = np.argsort(flat, kind="stable").tolist()
    tgt = target.tolist()
    is_stream = stream.tolist()
    base = [-1] * (h * w)                     # index of the stream cell each cell drains to
    for i in low_to_high:
        if is_stream[i]:
            base[i] = i
        else:
            t = tgt[i]
            if t >= 0:
                base[i] = base[t]
    base_a = np.asarray(base, dtype=np.int64)
    zf = filled.ravel()
    out = np.full(h * w, np.nan)
    ok = base_a >= 0
    out[ok] = zf[ok] - zf[base_a[ok]]
    out[np.isnan(zf)] = np.nan
    return out.reshape(h, w)


def stream_length_m(target: np.ndarray, stream: np.ndarray, width: int, cell_size: float) -> float:
    """Total length of stream, measuring each stream cell's step to the cell it drains to."""
    idx = np.nonzero(stream & (target >= 0))[0]
    if idx.size == 0:
        return 0.0
    t = target[idx]
    dr, dc = (t // width) - (idx // width), (t % width) - (idx % width)
    return float(np.hypot(dr, dc).sum() * cell_size)
