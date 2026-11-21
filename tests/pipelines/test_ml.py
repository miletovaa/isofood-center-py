import numpy as np
import pandas as pd
import pytest

from o2org.pipelines import ml


def _classification_df(n_per_class=15, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n_per_class, 3))
    b = rng.normal(4, 1, (n_per_class, 3))
    df = pd.DataFrame(np.concatenate([a, b]), columns=["f1", "f2", "f3"])
    df["group"] = ["a"] * n_per_class + ["b"] * n_per_class
    return df


def test_pca_scores_have_requested_number_of_components():
    df = _classification_df()
    result = ml.pca(df, ["f1", "f2", "f3"], n_components=2)
    assert result.scores.shape == (30, 2)
    assert result.loadings.shape == (3, 2)
    assert len(result.explained_variance_ratio) == 2


def test_pca_clamps_n_components_to_available_features():
    df = pd.DataFrame({"only_feature": [1.0, 2.0, 3.0]})
    result = ml.pca(df, ["only_feature"], n_components=5)
    assert result.scores.shape[1] == 1


def test_hotellings_t2_outliers_flags_a_clear_outlier():
    rng = np.random.default_rng(0)
    scores = pd.DataFrame({"PC1": rng.normal(0, 1, 50), "PC2": rng.normal(0, 1, 50)})
    scores.loc[len(scores)] = [50.0, 50.0]  # extreme outlier
    outliers = ml.hotellings_t2_outliers(scores)
    assert outliers.iloc[-1]
    assert outliers.sum() < 5  # shouldn't flag the whole cloud


def test_random_forest_classify_separates_well_separated_classes():
    df = _classification_df(n_per_class=20)
    result = ml.random_forest_classify(df, ["f1", "f2", "f3"], "group", positive_class="b", negative_class="a")
    accuracy = (result.y_pred == result.y_true).mean()
    assert accuracy > 0.9
    assert result.target_names == ("a", "b")


def test_label_encode_round_trips():
    encoded, encoder = ml.label_encode(["a", "b", "a", "c"])
    assert list(encoder.inverse_transform(encoded)) == ["a", "b", "a", "c"]


def test_tune_threshold_covers_requested_thresholds():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.6, 0.9]
    result = ml.tune_threshold(y_true, y_proba, thresholds=np.array([0.3, 0.5, 0.7]))
    assert len(result) == 3
    assert set(result.columns) >= {"threshold", "precision_0", "recall_1", "f1_1"}


def test_mdi_importance_sums_to_one_and_is_sorted_descending():
    df = _classification_df()
    rf = ml.random_forest_classify(df, ["f1", "f2", "f3"], "group", "b", "a")
    importances = ml.mdi_importance(rf.model, ["f1", "f2", "f3"])
    assert importances["importance"].sum() == pytest.approx(1.0)
    assert (importances["importance"].diff().dropna() <= 0).all()


def test_elbow_cutoff_finds_the_obvious_break():
    # a clear elbow where values drop sharply after the first few
    importances = np.array([0.5, 0.45, 0.4, 0.02, 0.01, 0.01, 0.01])
    cutoff, idx = ml.elbow_cutoff(importances)
    assert idx == 3
    assert cutoff == pytest.approx(0.02)


def test_elbow_cutoff_on_uniform_importances_picks_an_endpoint():
    importances = np.array([0.25, 0.25, 0.25, 0.25])
    cutoff, idx = ml.elbow_cutoff(importances)
    assert idx in (0, len(importances) - 1)  # a straight line has no interior elbow


def test_kmeans_cluster_recovers_two_separated_blobs():
    df = _classification_df(n_per_class=25)
    result = ml.kmeans_cluster(df, ["f1", "f2", "f3"], k=2)
    # cluster assignment should align with the two well-separated blobs (up to label swap)
    first_half, second_half = result.labels[:25], result.labels[25:]
    assert len(set(first_half)) == 1
    assert len(set(second_half)) == 1
    assert first_half[0] != second_half[0]


def test_lda_classify_separates_well_separated_classes():
    df = _classification_df(n_per_class=20)
    result = ml.lda_classify(df, ["f1", "f2", "f3"], "group")
    accuracy = (result.encoder.inverse_transform(result.y_pred) == df["group"].values).mean()
    assert accuracy > 0.9
