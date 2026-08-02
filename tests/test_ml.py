import numpy as np
import pandas as pd
import pytest

from o2org.analysis import ml


def test_pca_reduces_to_requested_components():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 5)), columns=list("abcde"))
    result = ml.pca(df, n_components=2)
    assert result.components.shape == (50, 2)
    assert list(result.components.columns) == ["PC1", "PC2"]
    assert len(result.explained_variance_ratio) == 2


def test_kmeans_cluster_assigns_k_labels():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=0, scale=0.2, size=(20, 2))
    blob_b = rng.normal(loc=10, scale=0.2, size=(20, 2))
    df = pd.DataFrame(np.vstack([blob_a, blob_b]), columns=["x", "y"])
    result = ml.kmeans_cluster(df, k=2)
    assert len(set(result.labels)) == 2


def test_linear_regression_recovers_linear_relationship():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y": 3 * x + 1 + rng.normal(scale=0.01, size=200)})
    result = ml.linear_regression(df, target="y", features=["x"])
    assert result.r2 > 0.95


def test_pca_accepts_raw_list_of_dicts_like_resource_list_returns():
    records = [{"a": float(i), "b": float(i * 2), "c": float(i * 3)} for i in range(30)]
    result = ml.pca(records, n_components=2)
    assert result.components.shape == (30, 2)


def test_pca_plot_hue_accepts_a_column_not_used_by_pca_itself(monkeypatch):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    df["group"] = (["x"] * 20) + (["y"] * 20)  # PCA never sees this column
    result = ml.pca(df, columns=["a", "b", "c"], n_components=2)

    captured = {}
    monkeypatch.setattr(ml.viz, "pca_scatter", lambda data, hue=None, ax=None: captured.update(data=data, hue=hue))
    result.plot(hue="group")

    assert "group" in captured["data"].columns
    assert captured["data"]["group"].tolist() == df["group"].tolist()


def test_classify_separates_clearly_separable_classes():
    rng = np.random.default_rng(0)
    class_a = rng.normal(loc=0, scale=0.5, size=(50, 2))
    class_b = rng.normal(loc=5, scale=0.5, size=(50, 2))
    df = pd.DataFrame(np.vstack([class_a, class_b]), columns=["x", "y"])
    df["label"] = [0] * 50 + [1] * 50
    result = ml.classify(df, target="label", features=["x", "y"])
    assert result.accuracy > 0.9


def test_classify_raises_clear_error_when_target_has_only_one_class():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "label": [True, True, True]})
    with pytest.raises(ValueError, match="only one distinct value"):
        ml.classify(df, target="label", features=["x", "y"])
