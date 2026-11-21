import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from o2org.pipelines import ml, viz


@pytest.fixture
def grouped_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "group": ["a"] * 10 + ["b"] * 10,
        "feat1": rng.normal(0, 1, 20),
        "feat2": rng.normal(5, 1, 20),
    })


def test_box_plot_has_one_trace_per_group_per_feature(grouped_df):
    fig = viz.box_plot(grouped_df, ["feat1", "feat2"], "group")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # 2 features x 2 groups
    assert all(isinstance(trace, go.Box) for trace in fig.data)


def test_median_bar_plot_has_one_bar_trace_per_group_per_feature(grouped_df):
    fig = viz.median_bar_plot(grouped_df, ["feat1", "feat2"], "group")
    assert len(fig.data) == 4
    assert all(isinstance(trace, go.Bar) for trace in fig.data)


def test_kde_density_plot_has_one_line_per_group_per_feature(grouped_df):
    fig = viz.kde_density_plot(grouped_df, ["feat1", "feat2"], "group")
    assert len(fig.data) == 4
    assert all(trace.mode == "lines" for trace in fig.data)


def test_pca_score_scatter_one_trace_per_group(grouped_df):
    pca_result = ml.pca(grouped_df, ["feat1", "feat2"])
    fig = viz.pca_score_scatter(pca_result, group=grouped_df["group"])
    assert len(fig.data) == 2
    assert f"PC1" in fig.layout.xaxis.title.text


def test_pca_loadings_plot_has_a_marker_trace(grouped_df):
    pca_result = ml.pca(grouped_df, ["feat1", "feat2"])
    fig = viz.pca_loadings_plot(pca_result, top_n=2)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers+text"


def test_pca_biplot_combines_scores_and_loadings(grouped_df):
    pca_result = ml.pca(grouped_df, ["feat1", "feat2"])
    fig = viz.pca_biplot(pca_result, group=grouped_df["group"])
    # 2 group score traces + 1 loadings trace
    assert len(fig.data) == 3


def test_rsd_comparison_plot_one_trace_per_stage():
    stages = {"Raw": pd.Series([1.0, 2.0]), "Normalized": pd.Series([0.5, 1.0])}
    fig = viz.rsd_comparison_plot(stages)
    assert len(fig.data) == 2


def test_imputed_counts_plot_sorts_descending():
    counts = pd.Series({"low": 1, "high": 10, "mid": 5})
    fig = viz.imputed_counts_plot(counts)
    assert list(fig.data[0].x) == ["high", "mid", "low"]


def test_roc_plot_reports_auc_in_trace_name():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.6, 0.9]
    fig = viz.roc_plot(y_true, y_proba)
    assert "AUC" in fig.data[0].name


def test_confusion_matrix_table_is_a_plain_dataframe():
    table = viz.confusion_matrix_table([0, 0, 1, 1], [0, 1, 1, 1], target_names=("neg", "pos"))
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["neg", "pos"]
    assert list(table.index) == ["neg", "pos"]


def test_mdi_elbow_plot_marks_the_elbow_index():
    importances = np.array([0.5, 0.3, 0.1, 0.05])
    fig = viz.mdi_elbow_plot(importances, elbow_idx=1, cutoff=0.3)
    assert len(fig.layout.shapes) == 1  # the vertical elbow marker


def test_top_features_barh_only_includes_features_above_cutoff():
    df = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [0.5, 0.3, 0.05]})
    fig = viz.top_features_barh(df, cutoff=0.2)
    assert sorted(fig.data[0].y) == ["a", "b"]


def test_isotope_biplot_one_trace_per_group(grouped_df):
    df = grouped_df.rename(columns={"feat1": "d13c", "feat2": "d15n"})
    fig = viz.isotope_biplot(df, "d13c", "d15n", group_col="group")
    assert len(fig.data) == 2


def test_van_krevelen_plot_uses_o_c_and_h_c_axes(grouped_df):
    df = grouped_df.rename(columns={"feat1": "h_c", "feat2": "o_c"})
    fig = viz.van_krevelen_plot(df, group_col="group")
    assert fig.layout.xaxis.title.text == "O/C"
    assert fig.layout.yaxis.title.text == "H/C"


def test_save_figure_writes_both_png_and_html(tmp_path):
    pytest.importorskip("kaleido")
    from o2org.pipelines.utils import save_figure

    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    stem = save_figure(fig, tmp_path / "myplot")
    assert (tmp_path / "myplot.png").exists()
    assert (tmp_path / "myplot.html").exists()
    assert stem == tmp_path / "myplot"
