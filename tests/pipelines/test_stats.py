import numpy as np
import pandas as pd
import pytest

from o2org.pipelines import stats


def _grouped_df():
    return pd.DataFrame({
        "group": ["a"] * 5 + ["b"] * 5,
        "feat": [1.0, 2.0, 1.5, 2.2, 1.8, 10.0, 11.0, 9.0, 10.5, 9.5],
    })


def test_descriptive_stats_reports_mean_median_std_n_per_group():
    result = stats.descriptive_stats(_grouped_df(), "feat", "group")
    row_a = result.loc[result["group"] == "a"].iloc[0]
    assert row_a["n"] == 5
    assert row_a["mean"] == pytest.approx(1.7)


def test_rsd_is_std_over_mean_times_100():
    df = pd.DataFrame({"feat": [10.0, 10.0, 10.0]})
    assert stats.rsd(df, ["feat"])["feat"] == pytest.approx(0.0)
    df2 = pd.DataFrame({"feat": [8.0, 10.0, 12.0]})
    expected = df2["feat"].std() / df2["feat"].mean() * 100
    assert stats.rsd(df2, ["feat"])["feat"] == pytest.approx(expected)


def test_shapiro_by_group_skips_groups_with_fewer_than_three_values():
    df = pd.DataFrame({"group": ["a", "a", "b"], "feat": [1.0, 2.0, 3.0]})
    result = stats.shapiro_by_group(df, "feat", "group")
    assert result.empty  # neither group has >= 3 values


def test_shapiro_by_group_flags_normal_and_nonnormal():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": ["normal"] * 30 + ["skewed"] * 30,
        "feat": np.concatenate([rng.normal(0, 1, 30), rng.exponential(1, 30) ** 3]),
    })
    result = stats.shapiro_by_group(df, "feat", "group", alpha=0.01).set_index("group")
    assert result.loc["normal", "is_normal"]
    assert not result.loc["skewed", "is_normal"]


def test_qq_regression_returns_high_r_for_normal_data():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 200)
    slope, intercept, r = stats.qq_regression(values)
    assert r > 0.98


def test_kruskal_wallis_detects_group_difference():
    result = stats.kruskal_wallis(_grouped_df(), "feat", "group")
    assert result["pvalue"] < 0.01


def test_anova_detects_group_difference():
    result = stats.anova(_grouped_df(), "feat", "group")
    assert result["pvalue"] < 0.01


def test_bonferroni_correction_is_more_conservative_than_bh():
    pvals = [0.001, 0.02, 0.03, 0.5]
    bonf = stats.bonferroni_correction(pvals)
    bh = stats.benjamini_hochberg(pvals)
    assert (bonf >= bh).all()


def test_test_all_features_marks_significant_and_nonsignificant_features():
    df = _grouped_df()
    df["noise"] = np.random.default_rng(1).normal(0, 1, 10)
    result = stats.test_all_features(df, ["feat", "noise"], "group", alpha=0.05)
    sig = result.set_index("feature")["significant"]
    assert sig["feat"]
    assert not sig["noise"]


def test_test_all_features_picks_anova_when_normal_kruskal_otherwise():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "group": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        "normal_feat": np.concatenate([rng.normal(0, 1, 20), rng.normal(3, 1, 20), rng.normal(1, 1, 20)]),
        "skewed_feat": np.concatenate([rng.exponential(1, 20), rng.exponential(3, 20), rng.exponential(2, 20)]),
    })
    result = stats.test_all_features(df, ["normal_feat", "skewed_feat"], "group", test="auto").set_index("feature")
    assert result.loc["normal_feat", "test"] == "anova"
    assert result.loc["skewed_feat", "test"] == "kruskal"


def test_test_all_features_uses_welch_or_mannwhitney_for_two_groups():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "group": ["a"] * 20 + ["b"] * 20,
        "normal_feat": np.concatenate([rng.normal(0, 1, 20), rng.normal(3, 1, 20)]),
        "skewed_feat": np.concatenate([rng.exponential(1, 20), rng.exponential(3, 20)]),
    })
    result = stats.test_all_features(df, ["normal_feat", "skewed_feat"], "group", test="auto").set_index("feature")
    assert result.loc["normal_feat", "test"] == "welch"
    assert result.loc["skewed_feat", "test"] == "mannwhitney"


def test_test_all_features_returns_empty_frame_for_no_valid_features():
    df = pd.DataFrame({"group": ["a"], "feat": [1.0]})  # only one group present
    result = stats.test_all_features(df, ["feat"], "group")
    assert result.empty
