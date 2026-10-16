"""Classifying image objects.

Three families, all taking the feature table from :mod:`.features` and all
returning a :class:`ClassificationResult`:

* :class:`RuleBasedClassifier` - ordered ``(class, condition)`` rules; the first
  rule an object satisfies wins.  Transparent, and ``explain()`` shows which
  conditions held, which is invaluable when teaching.
* :class:`FuzzyClassifier` - each class is described by membership functions of
  features (0 = not at all, 1 = fully); an object gets the class with the
  highest combined membership.  Also reports *classification stability*
  (best minus second-best membership): low values flag objects the rule set
  cannot separate well.
* :class:`ObjectClassifier` - supervised learning from sample objects:
  random forest, k-nearest-neighbour (the classic "nearest neighbour" of OBIA
  software), SVM, or a decision tree whose rules can be printed.

Rule strings are evaluated with ``pandas.DataFrame.eval`` against the feature
columns.  Treat rule sets as trusted code, not as untrusted user input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UNCLASSIFIED = "unclassified"


# ----------------------------------------------------------------------- result
@dataclass
class ClassificationResult:
    """Class per object plus the evidence behind it."""

    classes: pd.Series                     # index obj_id -> class name
    class_names: list[str]
    method: str
    scores: pd.DataFrame | None = None     # membership / probability per class
    stability: pd.Series | None = None     # best - second best score
    extra: dict = field(default_factory=dict)

    def codes(self) -> dict[str, int]:
        """``{class name: raster code}``; 0 is reserved for unclassified."""
        return {c: i + 1 for i, c in enumerate(self.class_names)}

    def to_raster(self, labels: np.ndarray) -> np.ndarray:
        """Paint each object's class code onto the label raster (uint8)."""
        labels = np.asarray(labels).astype(np.int64)
        lut = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
        codes = self.codes()
        for oid, cname in self.classes.items():
            lut[int(oid)] = codes.get(cname, 0)
        return lut[labels]

    def to_vector(self, labels, transform=None, crs=None, features: pd.DataFrame | None = None):
        """Classified objects as a GeoDataFrame (``class`` column, scores, stability)."""
        from .vector import labels_to_gdf
        attr = pd.DataFrame({"class": self.classes})
        if self.stability is not None:
            attr["stability"] = self.stability
        if self.scores is not None:
            attr = attr.join(self.scores.add_prefix("score_"))
        if features is not None:
            attr = attr.join(features, rsuffix="_feat")
        return labels_to_gdf(labels, transform, crs, attr)

    def summary(self, area: pd.Series | None = None) -> pd.DataFrame:
        """Object count (and area, if given) per class."""
        df = pd.DataFrame({"class": self.classes})
        out = df.groupby("class").size().rename("n_objects").to_frame()
        if area is not None:
            out["area"] = pd.Series(area).reindex(df.index).groupby(df["class"]).sum()
            out["area_share"] = out["area"] / out["area"].sum()
        return out


def _finish(classes, names, method, scores=None, extra=None) -> ClassificationResult:
    classes = pd.Series(classes)
    classes.name = "class"
    stab = None
    if scores is not None and scores.shape[1] >= 2:
        srt = np.sort(scores.to_numpy(), axis=1)
        stab = pd.Series(srt[:, -1] - srt[:, -2], index=scores.index, name="stability")
    elif scores is not None and scores.shape[1] == 1:
        stab = scores.iloc[:, 0].rename("stability")
    return ClassificationResult(classes, list(names), method, scores, stab, extra or {})


# ----------------------------------------------------------------------- rules
class RuleBasedClassifier:
    """Ordered rules, first match wins.

    >>> rules = [("water", "mean_nir < 40"),
    ...          ("vegetation", "mean_ndvi > 0.4"),
    ...          ("building", "brightness > 120 and compactness > 0.6")]
    >>> RuleBasedClassifier(rules).classify(features)        # doctest: +SKIP
    """

    def __init__(self, rules, default: str = UNCLASSIFIED):
        self.rules = [(str(c), str(e)) for c, e in rules]
        if not self.rules:
            raise ValueError("need at least one rule")
        self.default = default

    @property
    def class_names(self):
        return list(dict.fromkeys(c for c, _ in self.rules))

    def _eval(self, features: pd.DataFrame, expr: str) -> pd.Series:
        try:
            res = features.eval(expr, engine="python")
        except Exception as exc:  # unknown column, bad syntax, ...
            cols = ", ".join(map(str, features.columns[:25]))
            raise ValueError(f"cannot evaluate rule {expr!r}: {exc}\n"
                             f"available columns include: {cols}") from exc
        res = pd.Series(res, index=features.index)
        if res.dtype != bool:
            raise ValueError(f"rule {expr!r} must give True/False, got {res.dtype}")
        return res

    def classify(self, features: pd.DataFrame) -> ClassificationResult:
        out = pd.Series(self.default, index=features.index, dtype=object)
        todo = pd.Series(True, index=features.index)
        for cname, expr in self.rules:
            hit = self._eval(features, expr) & todo
            out[hit] = cname
            todo &= ~hit
        return _finish(out, self.class_names, "rules")

    def explain(self, features: pd.DataFrame, obj_id) -> pd.DataFrame:
        """For one object: every rule, whether its condition held, and which one decided.

        Because the first matching rule wins, several conditions can be true
        while only one ``decides``; seeing that is the point of this method.
        """
        row = features.loc[[obj_id]]
        rows, decided = [], False
        for cname, expr in self.rules:
            ok = bool(self._eval(row, expr).iloc[0])
            rows.append({"class": cname, "rule": expr, "condition_true": ok,
                         "decides": ok and not decided})
            decided = decided or ok
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------- fuzzy
class Membership:
    """A function mapping a feature value to a degree of membership in [0, 1]."""

    def __call__(self, x) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class Trapezoid(Membership):
    """0 below ``a``, rises to 1 at ``b``, stays 1 until ``c``, falls to 0 at ``d``."""

    def __init__(self, a, b, c, d):
        if not a <= b <= c <= d:
            raise ValueError("need a <= b <= c <= d")
        self.pts = (a, b, c, d)

    def __call__(self, x):
        a, b, c, d = self.pts
        x = np.asarray(x, dtype=float)
        up = np.where(b > a, np.clip((x - a) / max(b - a, 1e-12), 0, 1), (x >= a).astype(float))
        dn = np.where(d > c, np.clip((d - x) / max(d - c, 1e-12), 0, 1), (x <= d).astype(float))
        return np.where((x >= b) & (x <= c), 1.0, np.minimum(up, dn))


class Triangle(Trapezoid):
    """Peak of 1 at ``b``, zero at ``a`` and ``c``."""

    def __init__(self, a, b, c):
        super().__init__(a, b, b, c)


class LargerThan(Membership):
    """0 up to ``a``, linear rise to 1 at ``b``, 1 beyond (``a == b`` = hard threshold)."""

    def __init__(self, a, b=None):
        b = a if b is None else b
        if b < a:
            raise ValueError("need a <= b")
        self.a, self.b = a, b

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if self.b == self.a:
            return (x > self.a).astype(float)
        return np.clip((x - self.a) / (self.b - self.a), 0, 1)


class SmallerThan(Membership):
    """1 up to ``a``, linear fall to 0 at ``b``, 0 beyond (``a == b`` = hard threshold)."""

    def __init__(self, a, b=None):
        b = a if b is None else b
        if b < a:
            raise ValueError("need a <= b")
        self.a, self.b = a, b

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if self.b == self.a:
            return (x < self.a).astype(float)
        return np.clip((self.b - x) / (self.b - self.a), 0, 1)


class Gaussian(Membership):
    """Bell curve: 1 at ``mean``, falling off with ``sigma``."""

    def __init__(self, mean, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.mean, self.sigma = mean, sigma

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)


class FuzzyClassifier:
    """Class descriptions built from membership functions.

    Parameters
    ----------
    descriptions : ``{class: {feature column: Membership}}``
    combine : how the per-feature memberships of one class are merged -
        ``"min"`` (fuzzy AND, default: every condition must hold),
        ``"product"``, or ``"mean"`` (compensating: a weak feature can be
        offset by strong ones).
    min_membership : best membership below this -> unclassified.
    """

    def __init__(self, descriptions, combine: str = "min", min_membership: float = 0.1,
                 default: str = UNCLASSIFIED):
        if combine not in ("min", "product", "mean"):
            raise ValueError("combine must be 'min', 'product' or 'mean'")
        if not descriptions:
            raise ValueError("need at least one class description")
        self.desc = descriptions
        self.combine = combine
        self.min_membership = float(min_membership)
        self.default = default

    @property
    def class_names(self):
        return list(self.desc)

    def memberships(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for cname, feats in self.desc.items():
            parts = []
            for fname, fn in feats.items():
                if fname not in features.columns:
                    raise KeyError(f"class {cname!r} uses feature {fname!r}, which is not in the "
                                   f"feature table (have e.g. {list(features.columns[:15])})")
                m = fn(features[fname].to_numpy())
                parts.append(np.nan_to_num(np.asarray(m, dtype=float), nan=0.0))
            arr = np.vstack(parts)
            cols[cname] = (arr.min(0) if self.combine == "min"
                           else arr.prod(0) if self.combine == "product" else arr.mean(0))
        return pd.DataFrame(cols, index=features.index)

    def classify(self, features: pd.DataFrame) -> ClassificationResult:
        sc = self.memberships(features)
        best = sc.idxmax(axis=1)
        top = sc.max(axis=1)
        best = best.where(top >= self.min_membership, self.default).astype(object)
        return _finish(best, self.class_names, f"fuzzy-{self.combine}", sc)


# ------------------------------------------------------------------ supervised
_DEFAULT_EXCLUDE = ("centroid_row", "centroid_col", "centroid_x", "centroid_y", "parent_id")


class ObjectClassifier:
    """Supervised classification of objects from labelled sample objects.

    ``method``: ``"rf"`` random forest, ``"knn"`` nearest neighbour,
    ``"svm"`` support-vector machine, ``"tree"`` decision tree (readable with
    :meth:`export_rules`).  kNN and SVM standardise features first.  Location
    columns (centroids) are excluded by default so the model cannot "learn the
    map"; pass ``feature_columns`` to choose features explicitly.
    """

    def __init__(self, method: str = "rf", feature_columns=None, exclude=_DEFAULT_EXCLUDE,
                 random_state: int = 0, **model_params):
        if method not in ("rf", "knn", "svm", "tree"):
            raise ValueError("method must be 'rf', 'knn', 'svm' or 'tree'")
        self.method = method
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.exclude = tuple(exclude)
        self.random_state = random_state
        self.model_params = model_params
        self.model_ = None
        self.columns_: list[str] = []
        self.medians_: pd.Series | None = None

    def _build(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        p = self.model_params
        if self.method == "rf":
            return RandomForestClassifier(**{"n_estimators": 300, "random_state": self.random_state,
                                             "n_jobs": 1, **p})
        if self.method == "tree":
            return DecisionTreeClassifier(**{"random_state": self.random_state, "max_depth": 4, **p})
        if self.method == "knn":
            return make_pipeline(StandardScaler(), KNeighborsClassifier(**{"n_neighbors": 3, **p}))
        return make_pipeline(StandardScaler(), SVC(**{"probability": True,
                                                      "random_state": self.random_state, **p}))

    def _select(self, features: pd.DataFrame) -> list[str]:
        if self.feature_columns:
            missing = [c for c in self.feature_columns if c not in features.columns]
            if missing:
                raise KeyError(f"feature columns not found: {missing}")
            return self.feature_columns
        num = features.select_dtypes(include=[np.number]).columns
        return [c for c in num if c not in self.exclude and features[c].notna().any()]

    def _matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[self.columns_].replace([np.inf, -np.inf], np.nan)
        return X.fillna(self.medians_)

    def fit(self, features: pd.DataFrame, samples: pd.Series) -> "ObjectClassifier":
        s = pd.Series(samples).dropna()
        s = s[s.index.isin(features.index)]
        if s.nunique() < 2:
            raise ValueError("need samples from at least two classes")
        self.columns_ = self._select(features)
        feats = features[self.columns_].replace([np.inf, -np.inf], np.nan)
        self.medians_ = feats.median()
        X = feats.fillna(self.medians_).loc[s.index]
        self.model_ = self._build().fit(X.to_numpy(), s.astype(str).to_numpy())
        self.n_samples_ = len(s)
        return self

    @property
    def class_names(self) -> list[str]:
        est = self.model_[-1] if hasattr(self.model_, "steps") else self.model_
        return [str(c) for c in est.classes_]

    def predict(self, features: pd.DataFrame) -> ClassificationResult:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        X = self._matrix(features).to_numpy()
        proba = self.model_.predict_proba(X)
        names = self.class_names
        sc = pd.DataFrame(proba, index=features.index, columns=names)
        return _finish(sc.idxmax(axis=1).astype(object), names, f"supervised-{self.method}", sc)

    @property
    def feature_importances_(self) -> pd.Series | None:
        est = self.model_[-1] if hasattr(self.model_, "steps") else self.model_
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=self.columns_).sort_values(ascending=False)
        return None

    def export_rules(self) -> str:
        """Human-readable if/else rules (decision tree only)."""
        if self.method != "tree" or self.model_ is None:
            raise RuntimeError("export_rules() needs a fitted method='tree' classifier")
        from sklearn.tree import export_text
        return export_text(self.model_, feature_names=self.columns_)

    def cross_validate(self, features: pd.DataFrame, samples: pd.Series, k: int = 5) -> dict:
        """Stratified k-fold on the sample objects.

        Samples drawn from neighbouring objects are spatially correlated, so
        this is optimistic; use an independent reference for the real accuracy.
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from .accuracy import confusion_report
        s = pd.Series(samples).dropna()
        s = s[s.index.isin(features.index)].astype(str)
        cols = self._select(features)
        feats = features[cols].replace([np.inf, -np.inf], np.nan)
        X = feats.fillna(feats.median()).loc[s.index].to_numpy()
        kk = int(min(k, s.value_counts().min()))
        if kk < 2:
            raise ValueError("every class needs at least 2 samples for cross-validation")
        pred = cross_val_predict(self._build(), X, s.to_numpy(),
                                 cv=StratifiedKFold(kk, shuffle=True, random_state=self.random_state))
        rep = confusion_report(s, pd.Series(pred, index=s.index))
        rep["folds"] = kk
        return rep


# ---------------------------------------------------------------- sample tools
def samples_from_geometries(labels, gdf, class_column: str, transform=None,
                            min_coverage: float | None = None, crs=None) -> pd.Series:
    """Give each image object the reference class that covers most of it.

    Polygons (training areas or reference data) are rasterised onto the label
    grid.  An object is labelled with the dominant reference class if that class
    covers at least ``min_coverage`` of the object (default 0.5 for polygons;
    for point samples any hit counts).  Objects the reference does not reach
    are left out, and label 0 (no data) is never a sample.

    ``crs`` is the coordinate system of the label raster. When it is given and the geometries
    carry a different one, they are reprojected; if they have none, a ``ValueError`` asks you to
    set it. Geometries that lie wholly outside the raster extent raise ``ValueError`` too: that is
    nearly always a coordinate-system mix-up (degrees against metres), not an empty result.
    """
    from .vector import rasterize_geometries
    lab = np.asarray(labels).astype(np.int64)
    if crs is not None:
        if getattr(gdf, "crs", None) is None:
            raise ValueError("the reference geometries have no coordinate system; set one with "
                             "gdf.set_crs(...) so they can be placed on the image")
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
    if transform is not None and len(gdf):
        x0, y0 = transform * (0, 0)
        x1, y1 = transform * (lab.shape[1], lab.shape[0])
        minx, maxx, miny, maxy = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        gx0, gy0, gx1, gy1 = gdf.total_bounds
        if gx1 < minx or gx0 > maxx or gy1 < miny or gy0 > maxy:
            raise ValueError(
                f"the reference geometries (x {gx0:.5g}..{gx1:.5g}, y {gy0:.5g}..{gy1:.5g}) lie outside the image "
                f"(x {minx:.5g}..{maxx:.5g}, y {miny:.5g}..{maxy:.5g}). This is usually a coordinate-system mismatch: "
                "pass crs=<the image's CRS> or reproject the geometries with gdf.to_crs(...).")
    ref, codes = rasterize_geometries(gdf, lab.shape, transform, class_column)
    if min_coverage is None:
        pts = all(g.geom_type in ("Point", "MultiPoint") for g in gdf.geometry if g is not None)
        min_coverage = 0.0 if pts else 0.5
    inv = {v: k for k, v in codes.items()}
    K = max(codes.values(), default=0) + 1
    m = ref > 0
    if not m.any():
        return pd.Series(dtype=object, name="class")
    key = lab[m] * K + ref[m]
    uniq, cnt = np.unique(key, return_counts=True)
    df = pd.DataFrame({"obj": uniq // K, "code": uniq % K, "n": cnt})
    best = df.sort_values(["obj", "n"], ascending=[True, False]).drop_duplicates("obj").set_index("obj")
    area = np.bincount(lab.ravel(), minlength=int(lab.max()) + 1)
    best = best[best.index > 0]                     # label 0 = no data, never an object
    best["cov"] = best["n"] / area[best.index.to_numpy()]
    best = best[best["cov"] >= min_coverage]
    s = best["code"].map(inv).astype(object)
    s.index.name = "obj_id"
    s.name = "class"
    return s


def samples_from_raster(labels, class_raster, mapping: dict | None = None, min_purity: float = 0.6,
                        nodata: int = 0) -> pd.DataFrame:
    """Turn an existing land-cover map (ESA WorldCover, Dynamic World, an older classification) into
    *weak* training samples: each object gets the class that most of its pixels carry.

    ``class_raster`` is a 2-D integer raster on the same grid as ``labels``. ``mapping`` translates its
    codes to your class names (``{10: "forest", 40: "farmland", ...}``); codes that are not listed are
    "something else" - they never win, but they do count against purity. Pixels equal to ``nodata`` are
    ignored altogether. An object is kept only if the winning class covers at least ``min_purity`` of
    its mapped-or-unmapped pixels, so objects that straddle two classes in the map are left out rather
    than given a coin-flip label.

    Returns a DataFrame indexed by ``obj_id`` with ``class``, ``purity`` and ``n_px``. These labels
    inherit every error of the source map: use them to train, not to measure accuracy.
    """
    lab = np.asarray(labels).astype(np.int64)
    ras = np.asarray(class_raster)
    if lab.shape != ras.shape:
        raise ValueError(f"labels {lab.shape} and class raster {ras.shape} must be on the same grid")
    if not 0.0 < min_purity <= 1.0:
        raise ValueError("min_purity must be in (0, 1]")
    raw = ras.astype(np.int64)
    uniq_codes = np.unique(raw[raw != nodata])
    if mapping is None:
        mapping = {int(c): str(c) for c in uniq_codes}
    names = sorted(set(mapping.values()))
    cls_of = {int(c): names.index(n) + 1 for c, n in mapping.items()}   # 1..K, 0 = unmapped
    K = len(names) + 1
    code = np.zeros(raw.shape, np.int64)
    for c, k in cls_of.items():
        code[raw == c] = k
    m = (lab > 0) & (raw != nodata)
    if not m.any():
        return pd.DataFrame(columns=["class", "purity", "n_px"], index=pd.Index([], name="obj_id"))
    key = lab[m] * K + code[m]
    uniq, cnt = np.unique(key, return_counts=True)
    df = pd.DataFrame({"obj": uniq // K, "code": uniq % K, "n": cnt})
    total = df.groupby("obj")["n"].transform("sum")
    df["purity"] = df["n"] / total
    df["n_px"] = total
    best = df[df["code"] > 0].sort_values(["obj", "n"], ascending=[True, False]).drop_duplicates("obj").set_index("obj")
    best = best[best["purity"] >= min_purity]
    out = pd.DataFrame({"class": best["code"].map(lambda k: names[int(k) - 1]).astype(object),
                        "purity": best["purity"], "n_px": best["n_px"].astype(int)})
    out.index.name = "obj_id"
    return out
