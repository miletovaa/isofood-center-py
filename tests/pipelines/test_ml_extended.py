"""Tests for the classifiers / chemometrics added after the publication survey."""
import numpy as np
import pandas as pd
import pytest

from isofood_center.pipelines import ml, viz


@pytest.fixture
def three_class_df():
    rng = np.random.default_rng(0)
    centers = {"SI": (0, 0, 0, 0), "HR": (3, 0, 1, 0), "IT": (0, 3, 0, 1)}
    rows = []
    for label, c in centers.items():
        for _ in range(15):
            rows.append({"label": label, **{f"f{i}": c[i] + rng.normal(0, 0.6) for i in range(4)}})
    return pd.DataFrame(rows)


FEATURES = ["f0", "f1", "f2", "f3"]


@pytest.mark.parametrize("model", sorted(ml.CLASSIFIERS))
def test_every_registered_classifier_separates_clear_classes(three_class_df, model):
    res = ml.classify(three_class_df, FEATURES, "label", model=model)
    assert res.metrics["balanced_accuracy"] > 0.8, model
    assert res.confusion.values.sum() == 45
    assert list(res.classes) == ["HR", "IT", "SI"]


def test_classify_accepts_custom_estimator(three_class_df):
    from sklearn.naive_bayes import GaussianNB

    res = ml.classify(three_class_df, FEATURES, "label", model=GaussianNB())
    assert res.name == "GaussianNB" and res.metrics["accuracy"] > 0.8


def test_classify_lowers_folds_to_smallest_class(three_class_df):
    small = pd.concat([three_class_df, three_class_df.iloc[:3].assign(label="tiny")])
    assert ml.classify(small, FEATURES, "label", model="lda", n_splits=5).metrics["cv_folds"] == 3


def test_compare_classifiers_reports_errors_instead_of_raising(three_class_df):
    out = ml.compare_classifiers(three_class_df, FEATURES, "label", models=["lda", "knn", "does_not_exist"])
    assert set(out["model"]) == {"lda", "knn", "does_not_exist"}
    assert out.loc[out.model == "does_not_exist", "error"].notna().all()
    assert out.iloc[0]["balanced_accuracy"] >= out.iloc[1]["balanced_accuracy"]


def test_plsda_scores_vip_and_cv(three_class_df):
    res = ml.plsda(three_class_df, FEATURES, "label", n_components=2)
    assert res.scores.shape == (45, 3)
    assert set(res.vip["feature"]) == set(FEATURES)
    # f0 and f1 carry the class separation
    assert set(res.vip["feature"].head(2)) == {"f0", "f1"}
    assert res.cv.metrics["balanced_accuracy"] > 0.8


def test_permutation_test_significant_for_real_signal_not_for_noise(three_class_df):
    real = ml.permutation_test(three_class_df, FEATURES, "label", model="lda", n_permutations=30)
    assert real["pvalue"] < 0.05
    shuffled = three_class_df.assign(label=np.random.default_rng(1).permutation(three_class_df["label"].values))
    noise = ml.permutation_test(shuffled, FEATURES, "label", model="lda", n_permutations=30)
    assert noise["pvalue"] > 0.05


def test_feature_importance_ranks_informative_features(three_class_df):
    res = ml.classify(three_class_df, FEATURES, "label", model="random_forest")
    imp = ml.feature_importance(res, three_class_df, "label", n_repeats=5)
    assert set(imp["feature"].head(2)) == {"f0", "f1"}


def test_hierarchical_cluster_recovers_groups(three_class_df):
    hca = ml.hierarchical_cluster(three_class_df, ["f0", "f1"], n_clusters=3)
    assert len(set(hca.labels)) == 3
    assert ml.cluster_agreement(three_class_df["label"], hca.labels)["adjusted_rand"] > 0.7
    assert viz.dendrogram_plot(hca).data


def test_pls_regression_predicts_continuous_target():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (60, 5))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["y"] = 2 * df["x0"] - df["x1"] + rng.normal(0, 0.2, 60)
    res = ml.pls_regression(df, [f"x{i}" for i in range(5)], "y")
    assert res["q2"] > 0.9
    assert set(res["vip"]["feature"].head(2)) == {"x0", "x1"}


def test_new_figures_build(three_class_df):
    from isofood_center.pipelines import stats

    res = ml.classify(three_class_df, FEATURES, "label", model="lda")
    r, p = stats.correlation_matrix(three_class_df, FEATURES)
    pls = ml.plsda(three_class_df, FEATURES, "label")
    for fig in [
        viz.confusion_heatmap(res.confusion),
        viz.correlation_heatmap(r, p),
        viz.scores_scatter(pls.scores, "label"),
        viz.importance_bar(pls.vip, value_col="vip", threshold=1),
        viz.model_comparison_plot(ml.compare_classifiers(three_class_df, FEATURES, "label", models=["lda", "knn"])),
        viz.regression_plot(three_class_df, "f0", "f2", group_col="label"),
        viz.interaction_plot(three_class_df.assign(year=[2024, 2025] * 22 + [2024]), "f0", "label", "year"),
        viz.scree_plot(ml.pca(three_class_df, FEATURES, n_components=4)),
        viz.permutation_hist({"score": 0.9, "permutation_scores": np.random.rand(20), "pvalue": 0.01}),
    ]:
        assert fig.data


def test_grouped_cv_keeps_replicates_together_and_permutation_works(three_class_df):
    # duplicate every sample (e.g. two time points) — rows of one sample must share a fold
    dup = pd.concat([three_class_df.assign(sample=range(45)), three_class_df.assign(sample=range(45))], ignore_index=True)
    res = ml.classify(dup, FEATURES, "label", model="lda", groups_col="sample")
    assert res.metrics["grouped_cv"] and res.confusion.values.sum() == 90
    perm = ml.permutation_test(dup, FEATURES, "label", model="lda", n_permutations=20, groups_col="sample")
    assert perm["pvalue"] < 0.1
    assert perm["permutation_scores"].mean() < 0.6  # labels really were shuffled between samples


def test_grouped_cv_needs_two_samples_per_class(three_class_df):
    with pytest.raises(ValueError):
        ml.classify(three_class_df.assign(sample=three_class_df["label"]), FEATURES, "label", model="lda", groups_col="sample")
