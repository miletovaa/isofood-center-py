"""Tests for the tests added for docs/choosing_a_test.md Results A, C-G and post-hoc."""
import numpy as np
import pandas as pd
import pytest

from isofood_center.pipelines import stats


@pytest.fixture
def three_groups():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "group": ["a"] * 15 + ["b"] * 15 + ["c"] * 15,
        "x": np.concatenate([rng.normal(0, 1, 15), rng.normal(0.1, 1, 15), rng.normal(4, 1, 15)]),
    })


def test_welch_and_mann_whitney_detect_shift():
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "x": list(range(10)) + list(range(20, 30))})
    assert stats.welch_ttest(df, "x", "g")["pvalue"] < 0.001
    assert stats.mann_whitney(df, "x", "g")["pvalue"] < 0.001


def test_welch_rejects_three_groups(three_groups):
    with pytest.raises(ValueError):
        stats.welch_ttest(three_groups, "x", "group")


def test_effect_size_two_and_many_groups(three_groups):
    two = three_groups[three_groups.group != "b"]
    assert abs(stats.effect_size(two, "x", "group")["cohens_d"]) > 2
    assert 0.5 < stats.effect_size(three_groups, "x", "group")["eta_squared"] <= 1


@pytest.mark.parametrize("fn", [stats.tukey_hsd, stats.dunn_test])
def test_posthoc_finds_only_the_shifted_group(three_groups, fn):
    res = fn(three_groups, "x", "group", alpha=0.01)
    sig = {frozenset((r.group1, r.group2)) for r in res.itertuples() if r.significant}
    assert sig == {frozenset("ac"), frozenset("bc")}


def test_posthoc_auto_picks_tukey_for_normal_data(three_groups):
    assert stats.posthoc(three_groups, "x", "group")["test"].iloc[0] == "tukey"


def test_levene_flags_unequal_variance():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 40, "x": np.r_[rng.normal(0, 1, 40), rng.normal(0, 6, 40)]})
    assert stats.levene_test(df, "x", "g")["pvalue"] < 0.01


def _paired(n_conditions):
    rng = np.random.default_rng(3)
    rows = []
    for s in range(12):
        base = rng.normal(10, 2)
        for c in range(n_conditions):
            rows.append({"subject": s, "time": f"t{c}", "x": base + 1.5 * c + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


def test_paired_test_detects_within_subject_change_and_drops_incomplete():
    df = _paired(2)
    df = pd.concat([df, pd.DataFrame([{"subject": 99, "time": "t0", "x": 1.0}])])
    res = stats.paired_test(df, "x", "subject", "time")
    assert res["n_pairs"] == 12
    assert res["pvalue"] < 0.001
    assert res["test"] in {"paired_t", "wilcoxon"}


@pytest.mark.parametrize("test", ["rm_anova", "friedman"])
def test_repeated_measures(test):
    res = stats.repeated_measures_test(_paired(4), "x", "subject", "time", test=test)
    assert res["pvalue"] < 0.001 and res["n_subjects"] == 12


@pytest.mark.parametrize("test", ["anova", "art"])
def test_two_way_anova_finds_main_effect_not_interaction(test):
    rng = np.random.default_rng(4)
    rows = [{"region": r, "year": y, "zn": (5 if r == "coast" else 0) + rng.normal(0, 1)}
            for r in ("coast", "mountain") for y in (2024, 2025) for _ in range(12)]
    res = stats.two_way_anova(pd.DataFrame(rows), "zn", "region", "year", test=test).set_index("effect")
    assert res.loc["region", "significant"]
    assert not res.loc["year", "significant"]
    assert not res.loc["region:year", "significant"]


def test_correlation_auto_and_matrix():
    rng = np.random.default_rng(5)
    fe = rng.normal(0, 1, 50)
    df = pd.DataFrame({"fe": fe, "zn": fe * 2 + rng.normal(0, 0.3, 50), "noise": rng.normal(0, 1, 50)})
    res = stats.correlation(df["fe"], df["zn"])
    assert res["method"] == "pearson" and res["r"] > 0.9
    r, p = stats.correlation_matrix(df, method="spearman")
    assert r.loc["fe", "zn"] > 0.9 and p.loc["fe", "zn"] < 0.001
    assert p.loc["fe", "noise"] > 0.01
    assert np.allclose(np.diag(r), 1)


def test_linear_regression_recovers_slope():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 10, 60)
    df = pd.DataFrame({"x": x, "y": 3 * x + 1 + rng.normal(0, 0.5, 60)})
    model = stats.linear_regression(df, "y", "x")
    table = stats.regression_table(model).set_index("term")
    assert table.loc["x", "estimate"] == pytest.approx(3, abs=0.1)
    assert model.rsquared > 0.95


def test_mixed_model_fits():
    rng = np.random.default_rng(7)
    rows = [{"batch": b, "region": r, "y": b * 0.5 + (2 if r == "a" else 0) + rng.normal(0, 0.5)}
            for b in range(6) for r in "ab" for _ in range(5)]
    fit = stats.mixed_model(pd.DataFrame(rows), "y ~ region", "batch")
    assert fit.pvalues["region[T.b]"] < 0.001


def test_contingency_switches_to_fisher_for_sparse_2x2():
    sparse = pd.DataFrame({"origin": ["coast"] * 6 + ["mountain"] * 6, "pollen": ["yes"] * 5 + ["no"] + ["no"] * 5 + ["yes"]})
    assert stats.contingency_test(sparse, "origin", "pollen")["test"] == "fisher"
    big = pd.DataFrame({"origin": ["coast"] * 60 + ["mountain"] * 60,
                        "pollen": ["yes"] * 45 + ["no"] * 15 + ["yes"] * 15 + ["no"] * 45})
    res = stats.contingency_test(big, "origin", "pollen")
    assert res["test"] == "chi2" and res["pvalue"] < 0.001
