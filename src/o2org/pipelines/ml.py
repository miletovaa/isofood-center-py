"""Standalone machine-learning techniques — ported from the reference
lcms pipeline (`s02pca.py`/`s03rforest.py`), generalized to any wide
DataFrame (rows = samples, columns = features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as _PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from . import preprocessing


@dataclass
class PCAResult:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: np.ndarray
    model: _PCA


def pca(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_components: int = 2,
    scale: Literal["pareto", "standard", None] = "pareto",
    seed: int = 42,
) -> PCAResult:
    """PCA on a wide DataFrame (sample-level PCA if rows = samples,
    feature-level PCA if you pass a transposed frame — rows = features).

    Args:
        df: Wide DataFrame.
        columns: Columns to include; defaults to all of `df`'s columns.
        n_components: Number of components (clamped to what the data allows).
        scale: "pareto" (default, standard for metabolomics-style data),
            "standard" (z-score), or None to skip scaling.
        seed: Random state for the PCA solver.
    """
    columns = columns or list(df.columns)
    X = df[columns].apply(pd.to_numeric, errors="coerce")

    if scale == "pareto":
        X = preprocessing.pareto_scale(X, columns=columns)[columns]
    elif scale == "standard":
        mu, sd = X.mean(), X.std(ddof=1).replace(0, np.nan)
        X = ((X - mu) / sd).fillna(0)
    elif scale is None:
        X = X.fillna(0)
    else:
        raise ValueError(f"scale must be 'pareto', 'standard', or None, got {scale!r}")

    n_components = max(1, min(n_components, X.shape[1], X.shape[0]))
    model = _PCA(n_components=n_components, random_state=seed)
    scores = model.fit_transform(X.values)

    pc_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=pc_cols, index=df.index)
    loadings_df = pd.DataFrame(model.components_.T, index=columns, columns=pc_cols)
    return PCAResult(scores=scores_df, loadings=loadings_df, explained_variance_ratio=model.explained_variance_ratio_, model=model)


def hotellings_t2_outliers(scores_df: pd.DataFrame, pcs: tuple[str, str] = ("PC1", "PC2"), alpha: float = 0.95) -> pd.Series:
    """Hotelling's T² outlier flags on a 2-PC score plane.

    Args:
        scores_df: PCA scores (e.g. `PCAResult.scores`).
        pcs: Which two score columns to use.
        alpha: Confidence level for the T² critical value.

    Returns:
        Boolean Series (True = outlier), same index as `scores_df`.
    """
    S = scores_df[list(pcs)].to_numpy()
    n, a = S.shape
    eig = S.var(axis=0, ddof=1)
    t2 = (S[:, 0] ** 2) / eig[0] + (S[:, 1] ** 2) / eig[1]
    f_crit = f_dist.ppf(alpha, a, n - a)
    t2_crit = (a * (n - 1) / (n - a)) * f_crit
    return pd.Series(t2 > t2_crit, index=scores_df.index, name="is_outlier")


@dataclass
class RFResult:
    model: RandomForestClassifier
    y_true: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    target_names: tuple[str, str]
    feature_names: list[str]


def random_forest_classify(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    positive_class,
    negative_class,
    n_estimators: int = 100,
    max_depth: int = 5,
    n_splits: int = 5,
    seed: int = 42,
) -> RFResult:
    """Binary Random Forest classification with stratified cross-validated
    predictions (direct port of the reference lcms pipeline's RF step).

    Args:
        df: Wide DataFrame.
        feature_cols: Feature columns.
        label_col: Column holding the class label.
        positive_class: Label value treated as the positive class (1).
        negative_class: Label value treated as the negative class (0). Rows
            with any other label are excluded.
        n_estimators: Number of trees.
        max_depth: Max tree depth.
        n_splits: Number of stratified CV folds.
        seed: Random state (splitter and classifier).
    """
    mask = df[label_col].isin([positive_class, negative_class])
    X = df.loc[mask, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = (df.loc[mask, label_col] == positive_class).astype(int).values

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced", random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
    y_pred = cross_val_predict(rf, X, y, cv=skf, method="predict")
    rf.fit(X, y)

    return RFResult(
        model=rf, y_true=y, y_proba=y_proba, y_pred=y_pred,
        target_names=(str(negative_class), str(positive_class)), feature_names=list(feature_cols),
    )


def stratified_kfold_predict(X, y, model, n_splits: int = 5, seed: int = 42, method: str = "predict_proba"):
    """Stratified k-fold cross-validated predictions for an arbitrary sklearn
    estimator — the manual building block behind `random_forest_classify`.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=skf, method=method)


def label_encode(labels) -> tuple[np.ndarray, LabelEncoder]:
    """Encode a categorical label column to integers.

    Returns:
        `(encoded_values, fitted_encoder)`.
    """
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder


def tune_threshold(y_true, y_proba, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Precision/recall/F1 for both classes across a sweep of decision
    thresholds on the positive-class probability.
    """
    from sklearn.metrics import precision_recall_fscore_support

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    rows = []
    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        rows.append({
            "threshold": t,
            "precision_0": precision[0], "recall_0": recall[0], "f1_0": f1[0],
            "precision_1": precision[1], "recall_1": recall[1], "f1_1": f1[1],
        })
    return pd.DataFrame(rows)


def mdi_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Mean Decrease in Impurity feature importances, sorted descending."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).rename_axis("feature").reset_index(name="importance")


def elbow_cutoff(importances_sorted) -> tuple[float, int]:
    """Elbow-point detection via the max-perpendicular-distance method:
    finds the point on a sorted-descending curve farthest from the straight
    line between its first and last point.

    Args:
        importances_sorted: 1D array-like of importances, sorted descending
            (e.g. `mdi_importance(...)["importance"].values`).

    Returns:
        `(cutoff_value, elbow_index)`.
    """
    y_vals = np.asarray(importances_sorted, dtype=float)
    n = len(y_vals)
    x0, y0 = 0, y_vals[0]
    x1, y1 = n - 1, y_vals[-1]

    numerator = np.abs((y1 - y0) * np.arange(n) - (x1 - x0) * y_vals + x1 * y0 - y1 * x0)
    denominator = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    distances = numerator / denominator if denominator else numerator

    idx = int(np.argmax(distances))
    return float(y_vals[idx]), idx


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: KMeans
    inertia: float


def kmeans_cluster(
    df: pd.DataFrame, columns: list[str], k: int = 3, scale: Literal["standard", "pareto", None] = "standard", seed: int = 42
) -> ClusterResult:
    """K-means clustering over a feature subspace — unsupervised exploration
    for compositional clusters not captured by an existing group label.
    """
    X = df[columns].apply(pd.to_numeric, errors="coerce")
    if scale == "standard":
        mu, sd = X.mean(), X.std(ddof=1).replace(0, np.nan)
        X = ((X - mu) / sd).fillna(0)
    elif scale == "pareto":
        X = preprocessing.pareto_scale(X, columns=columns)[columns]
    else:
        X = X.fillna(0)

    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X.values)
    return ClusterResult(labels=labels, model=model, inertia=float(model.inertia_))


@dataclass
class LDAResult:
    model: LinearDiscriminantAnalysis
    scores: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    encoder: LabelEncoder


def lda_classify(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> LDAResult:
    """Linear Discriminant Analysis classification — the practical
    substitute for a Bayesian mixing model (MixSIAR has no Python port) on
    small isotope-style feature sets, or a lighter-weight alternative to
    Random Forest generally.
    """
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y, encoder = label_encode(df[label_col])

    n_components = min(len(encoder.classes_) - 1, X.shape[1])
    n_components = max(n_components, 1)
    model = LinearDiscriminantAnalysis(n_components=n_components)
    scores = model.fit_transform(X.values, y)
    y_pred = model.predict(X.values)

    score_cols = [f"LD{i + 1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=score_cols, index=df.index)
    scores_df[label_col] = df[label_col].values
    return LDAResult(model=model, scores=scores_df, y_true=y, y_pred=y_pred, encoder=encoder)


# --------------------------------------------------------------------------
# Added after the O2/ISOFood publication survey (method_statistics.csv). The
# chemometrics methods ISOFood papers use most — LDA/QDA, PLS-DA/OPLS-DA,
# SIMCA, HCA, cross-validation/ROC/permutation validation — and the ML
# models that show up across O2 (ANN, decision trees, random forest, kNN,
# SVM, gradient boosting, logistic regression, feature importance/XAI).
#
# Every classifier is a plain sklearn estimator, so they all go through the
# same `classify()` / `compare_classifiers()` / `permutation_test()` path and
# can be dropped into your own sklearn code too.
# --------------------------------------------------------------------------

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def vip_scores(pls_model) -> np.ndarray:
    """Variable Importance in Projection for a fitted `PLSRegression`
    (VIP > 1 is the usual "important feature" rule of thumb)."""
    t, w, q = pls_model.x_scores_, pls_model.x_weights_, pls_model.y_loadings_
    p, h = w.shape
    ss = np.diag(t.T @ t @ q.T @ q).reshape(h)
    w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(p * (w_norm ** 2 @ ss) / ss.sum())


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """PLS-DA: PLS regression onto one-hot class indicators, predicting the
    class with the highest response. Scale inputs first (see `classify`)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression

        self.classes_ = np.unique(y)
        Y = (np.asarray(y)[:, None] == self.classes_[None, :]).astype(float)
        n_comp = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=n_comp, scale=False).fit(X, Y)
        return self

    def decision_function(self, X):
        return self.pls_.predict(X)

    def predict_proba(self, X):
        # not calibrated probabilities — a softmax of the PLS response, good
        # enough for ROC curves and ranking
        z = self.decision_function(X)
        z = np.exp(5 * (z - z.max(axis=1, keepdims=True)))
        return z / z.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def transform(self, X):
        return self.pls_.transform(X)


class SIMCAClassifier(ClassifierMixin, BaseEstimator):
    """SIMCA (Soft Independent Modelling of Class Analogy): one PCA model per
    class; a sample is assigned to the class whose model reconstructs it best
    (smallest residual distance scaled by that class's own residual spread).
    This is the "hard" assignment variant, so it fits the sklearn API."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = []
        for c in self.classes_:
            Xc = X[y == c]
            n_comp = max(1, min(self.n_components, Xc.shape[0] - 1, Xc.shape[1]))
            model = _PCA(n_components=n_comp).fit(Xc)
            resid = ((Xc - model.inverse_transform(model.transform(Xc))) ** 2).sum(axis=1)
            self.models_.append((model, max(resid.mean(), 1e-12)))
        return self

    def class_distances(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([
            ((X - m.inverse_transform(m.transform(X))) ** 2).sum(axis=1) / s0 for m, s0 in self.models_
        ])

    def predict_proba(self, X):
        inv = 1.0 / (self.class_distances(X) + 1e-12)
        return inv / inv.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmin(self.class_distances(X), axis=1)]


#: Name -> factory for every classifier `classify()` knows. `scaled=True`
#: means it gets a StandardScaler in front (distance/variance-based models).
CLASSIFIERS: dict[str, tuple[str, Any, bool]] = {}


def _register_classifiers(seed: int = 42) -> dict:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "lda": ("Linear discriminant analysis", lambda: LinearDiscriminantAnalysis(), True),
        "qda": ("Quadratic discriminant analysis", lambda: QuadraticDiscriminantAnalysis(reg_param=0.1), True),
        "plsda": ("PLS-DA", lambda: PLSDAClassifier(n_components=2), True),
        "simca": ("SIMCA", lambda: SIMCAClassifier(n_components=2), True),
        "random_forest": ("Random forest", lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed), False),
        "tree": ("Decision tree (CART)", lambda: DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed), False),
        "knn": ("k-nearest neighbours", lambda: KNeighborsClassifier(n_neighbors=5), True),
        "svm": ("Support vector machine (RBF)", lambda: SVC(probability=True, class_weight="balanced", random_state=seed), True),
        "logistic": ("Logistic regression", lambda: LogisticRegression(max_iter=5000, class_weight="balanced"), True),
        # plain GradientBoosting, not HistGradientBoosting: the latter's
        # min_samples_leaf=20 can't split the 10-50-sample sets typical here
        "gbm": ("Gradient boosting", lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed), False),
        "ann": ("Neural network (MLP)", lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=seed), True),
    }


CLASSIFIERS.update(_register_classifiers())


def make_classifier(name: str, seed: int = 42, scale: bool | None = None, **params):
    """Build one of `CLASSIFIERS` by short name, wrapped with a StandardScaler
    when appropriate. Extra `params` are passed to the estimator
    (e.g. `make_classifier("knn", n_neighbors=3)`)."""
    registry = _register_classifiers(seed)
    if name not in registry:
        raise ValueError(f"unknown classifier {name!r}; choose from {sorted(registry)}")
    _, factory, default_scaled = registry[name]
    est = factory()
    if params:
        est.set_params(**params)
    return make_pipeline(StandardScaler(), est) if (default_scaled if scale is None else scale) else est


def _xy(df: pd.DataFrame, feature_cols: list[str], label_col: str, groups_col: str | None = None):
    sub = df.dropna(subset=[label_col])
    X = sub[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())  # cheap, leak-free-enough default; impute properly upstream
    groups = sub[groups_col].astype(str).values if groups_col else None
    return X, sub[label_col].astype(str).values, groups


def _cv(y, n_splits: int, seed: int, groups=None):
    """Stratified k-fold — or stratified *group* k-fold when `groups` is
    given, so every row of one sample (replicates, time points) lands in the
    same fold and can't leak from training into test."""
    from sklearn.model_selection import StratifiedGroupKFold

    if groups is not None:
        per_class = pd.DataFrame({"y": y, "g": groups}).groupby("y")["g"].nunique().min()
        n_splits = int(min(n_splits, per_class))
        if n_splits < 2:
            raise ValueError("every class needs at least 2 distinct samples for grouped cross-validation")
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    smallest = pd.Series(y).value_counts().min()
    n_splits = int(min(n_splits, smallest))
    if n_splits < 2:
        raise ValueError("every class needs at least 2 samples for cross-validation")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


@dataclass
class ClassifyResult:
    name: str
    model: Any                # fitted on all data
    feature_names: list[str]
    classes: np.ndarray
    y_true: np.ndarray        # string labels
    y_pred: np.ndarray        # cross-validated predictions
    y_proba: np.ndarray | None  # cross-validated class probabilities (columns = classes)
    metrics: dict
    confusion: pd.DataFrame
    report: pd.DataFrame


def classify(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model: str | Any = "random_forest",
    n_splits: int = 5,
    seed: int = 42,
    groups_col: str | None = None,
) -> ClassifyResult:
    """Cross-validated multi-class classification with any model.

    Args:
        model: A name from `CLASSIFIERS` ("lda", "qda", "plsda", "simca",
            "random_forest", "tree", "knn", "svm", "logistic", "gbm", "ann")
            or any sklearn estimator/pipeline of your own.
        n_splits: Stratified CV folds (lowered automatically to the smallest
            class size).
        groups_col: Column identifying the physical sample when it has
            several rows (technical replicates, time points). Rows sharing a
            value are kept in the same fold — without this, accuracy is
            inflated because the model is tested on samples it has seen.

    Returns:
        `ClassifyResult` — metrics (accuracy, balanced accuracy, macro F1,
        and ROC AUC when probabilities are available) come from the
        cross-validated predictions, never from resubstitution.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score, roc_auc_score)

    X, y, groups = _xy(df, feature_cols, label_col, groups_col)
    est = make_classifier(model, seed=seed) if isinstance(model, str) else model
    name = model if isinstance(model, str) else type(model).__name__
    cv = _cv(y, n_splits, seed, groups)

    y_pred = cross_val_predict(clone(est), X, y, cv=cv, groups=groups)
    y_proba = None
    try:
        y_proba = cross_val_predict(clone(est), X, y, cv=cv, groups=groups, method="predict_proba")
    except (AttributeError, ValueError):
        pass
    fitted = clone(est).fit(X, y)
    classes = np.unique(y)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "n_samples": len(y), "n_classes": len(classes), "cv_folds": cv.get_n_splits(),
        "grouped_cv": groups is not None,
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = (roc_auc_score(y, y_proba[:, 1]) if len(classes) == 2
                                  else roc_auc_score(y, y_proba, multi_class="ovr", labels=classes))
        except ValueError:
            pass

    confusion = pd.DataFrame(confusion_matrix(y, y_pred, labels=classes),
                             index=pd.Index(classes, name="true"), columns=pd.Index(classes, name="predicted"))
    report = pd.DataFrame(classification_report(y, y_pred, output_dict=True, zero_division=0)).T
    return ClassifyResult(name=name, model=fitted, feature_names=list(feature_cols), classes=classes,
                          y_true=y, y_pred=y_pred, y_proba=y_proba, metrics=metrics, confusion=confusion, report=report)


def compare_classifiers(
    df: pd.DataFrame, feature_cols: list[str], label_col: str, models: list[str] | None = None,
    n_splits: int = 5, seed: int = 42, groups_col: str | None = None,
) -> pd.DataFrame:
    """Run `classify()` for several models on the same CV folds and return
    one row of metrics per model, best balanced accuracy first. Models that
    fail on this data (e.g. QDA with tiny classes) get an `error` instead."""
    models = models or list(CLASSIFIERS)
    rows = []
    for m in models:
        try:
            res = classify(df, feature_cols, label_col, model=m, n_splits=n_splits, seed=seed, groups_col=groups_col)
            rows.append({"model": m, "label": CLASSIFIERS.get(m, (m,))[0], **res.metrics})
        except Exception as exc:  # noqa: BLE001 — one bad model shouldn't sink the comparison
            rows.append({"model": m, "label": CLASSIFIERS.get(m, (m,))[0], "error": str(exc)})
    out = pd.DataFrame(rows)
    if "balanced_accuracy" in out:
        out = out.sort_values("balanced_accuracy", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def permutation_test(
    df: pd.DataFrame, feature_cols: list[str], label_col: str, model: str | Any = "plsda",
    n_permutations: int = 200, n_splits: int = 5, seed: int = 42, groups_col: str | None = None,
) -> dict:
    """Is the cross-validated accuracy better than chance? Shuffles the labels
    `n_permutations` times (the standard validation for PLS-DA/LDA models).

    Returns:
        `{"score", "permutation_scores", "pvalue"}` (balanced accuracy).
    """
    from sklearn.model_selection import cross_val_score

    X, y, groups = _xy(df, feature_cols, label_col, groups_col)
    est = make_classifier(model, seed=seed) if isinstance(model, str) else model

    def score(labels):
        cv = _cv(labels, n_splits, seed, groups)
        return cross_val_score(clone(est), X, labels, groups=groups, cv=cv, scoring="balanced_accuracy").mean()

    rng = np.random.default_rng(seed)
    if groups is None:
        shuffles = (rng.permutation(y) for _ in range(n_permutations))
    else:
        # shuffle labels between whole samples, not between rows of a sample
        # (sklearn's permutation_test_score shuffles *within* groups, which
        # leaves one-label-per-sample data unchanged)
        g_label = pd.Series(y, index=groups).groupby(level=0).first()
        shuffles = (pd.Series(rng.permutation(g_label.values), index=g_label.index).loc[groups].values
                    for _ in range(n_permutations))
    real = score(y)
    perm = np.array([score(lab) for lab in shuffles])
    p = (np.sum(perm >= real) + 1) / (n_permutations + 1)
    return {"score": real, "permutation_scores": perm, "pvalue": p}


def feature_importance(
    result: ClassifyResult, df: pd.DataFrame, label_col: str, n_repeats: int = 20, seed: int = 42,
) -> pd.DataFrame:
    """Model-agnostic permutation feature importance (drop in balanced
    accuracy when a feature is shuffled) — works for every classifier,
    unlike MDI. Note it is computed on the training data; use it to rank
    features, not to estimate performance."""
    from sklearn.inspection import permutation_importance

    X, y, _ = _xy(df, result.feature_names, label_col)
    imp = permutation_importance(result.model, X, y, n_repeats=n_repeats, random_state=seed, scoring="balanced_accuracy")
    return (pd.DataFrame({"feature": result.feature_names, "importance": imp.importances_mean, "std": imp.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


@dataclass
class PLSDAResult:
    model: Any
    scores: pd.DataFrame
    vip: pd.DataFrame
    explained_x: np.ndarray
    cv: ClassifyResult


def plsda(
    df: pd.DataFrame, feature_cols: list[str], label_col: str, n_components: int = 2, n_splits: int = 5, seed: int = 42,
    groups_col: str | None = None,
) -> PLSDAResult:
    """PLS-DA with autoscaling: score plot coordinates, VIP scores, and
    cross-validated classification metrics in one call."""
    cv_result = classify(df, feature_cols, label_col, model=make_pipeline(StandardScaler(), PLSDAClassifier(n_components)),
                         n_splits=n_splits, seed=seed, groups_col=groups_col)
    cv_result.name = "plsda"
    X, y, _ = _xy(df, feature_cols, label_col)
    scaler, clf = cv_result.model.steps[0][1], cv_result.model.steps[1][1]
    Xs = scaler.transform(X)
    t = clf.transform(Xs)
    cols = [f"LV{i + 1}" for i in range(t.shape[1])]
    scores = pd.DataFrame(t, columns=cols, index=X.index)
    scores[label_col] = y
    total_var = (Xs ** 2).sum()
    explained = np.array([
        ((np.outer(t[:, i], clf.pls_.x_loadings_[:, i])) ** 2).sum() / total_var for i in range(t.shape[1])
    ])
    vip = (pd.DataFrame({"feature": feature_cols, "vip": vip_scores(clf.pls_)})
           .sort_values("vip", ascending=False).reset_index(drop=True))
    return PLSDAResult(model=cv_result.model, scores=scores, vip=vip, explained_x=explained, cv=cv_result)


@dataclass
class HCAResult:
    linkage: np.ndarray
    labels: np.ndarray
    sample_labels: list[str]
    method: str
    metric: str


def hierarchical_cluster(
    df: pd.DataFrame, columns: list[str], n_clusters: int = 3, method: str = "ward", metric: str = "euclidean",
    scale: Literal["standard", "pareto", None] = "standard", sample_labels: pd.Series | None = None,
) -> HCAResult:
    """Hierarchical cluster analysis (HCA) of samples. `method` is any
    scipy linkage method ("ward", "average", "complete", "single")."""
    from scipy.cluster.hierarchy import fcluster, linkage

    X = df[columns].apply(pd.to_numeric, errors="coerce")
    if scale == "standard":
        X = ((X - X.mean()) / X.std(ddof=1).replace(0, np.nan)).fillna(0)
    elif scale == "pareto":
        X = preprocessing.pareto_scale(X, columns=columns)[columns]
    else:
        X = X.fillna(0)
    Z = linkage(X.values, method=method, metric=metric)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    names = [str(v) for v in (sample_labels if sample_labels is not None else df.index)]
    return HCAResult(linkage=Z, labels=labels, sample_labels=names, method=method, metric=metric)


def cluster_agreement(true_labels, cluster_labels) -> dict:
    """How well unsupervised clusters recover known groups (adjusted Rand
    index, normalized mutual information)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {"adjusted_rand": adjusted_rand_score(true_labels, cluster_labels),
            "nmi": normalized_mutual_info_score(true_labels, cluster_labels)}


def pls_regression(
    df: pd.DataFrame, feature_cols: list[str], y_col: str, n_components: int = 2, n_splits: int = 5, seed: int = 42,
) -> dict:
    """PLS regression (PLSR) predicting a continuous property, with
    cross-validated R² (Q²) and RMSECV."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    sub = df.dropna(subset=[y_col])
    X = sub[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = pd.to_numeric(sub[y_col], errors="coerce").values
    n_comp = max(1, min(n_components, X.shape[1], X.shape[0] - 2))
    est = make_pipeline(StandardScaler(), PLSRegression(n_components=n_comp))
    y_cv = cross_val_predict(est, X, y, cv=KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)).ravel()
    fitted = est.fit(X, y)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "model": fitted, "y_true": y, "y_cv": y_cv,
        "q2": 1 - ((y - y_cv) ** 2).sum() / ss_tot,
        "r2": fitted.score(X, y),
        "rmsecv": float(np.sqrt(((y - y_cv) ** 2).mean())),
        "vip": pd.DataFrame({"feature": feature_cols, "vip": vip_scores(fitted.steps[1][1])}).sort_values("vip", ascending=False),
    }
