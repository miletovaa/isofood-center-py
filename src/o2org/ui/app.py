"""o2org analysis UI (Streamlit).

Run with `o2org-ui` or `streamlit run src/o2org/ui/app.py`.

Every calculation is a call into `o2org.pipelines` — the UI only collects
choices and draws results. Each section has a "Python code" expander with the
equivalent library calls, so anything done here can be repeated or extended
in a notebook.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

try:
    import o2org  # noqa: F401
except ImportError:  # running `streamlit run src/o2org/ui/app.py` from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from o2org.pipelines import ml, preprocessing, stats, viz
from o2org.ui.example_data import example_dataset

st.set_page_config(page_title="o2org analysis", page_icon="🧪", layout="wide")

RECORD_TYPES = {
    "Stable isotopes": ("result_stable_isotopes", "details_analyte"),
    "Elemental composition": ("result_elemental_composition", "details_element"),
    "MK GC-MS": ("result_mk_gc_ms", "details_compound_id"),
    "MK GC-IRMS": ("result_mk_gc_irms", "details_compound_id"),
    "VOC GC-MS": ("result_voc_gc_ms", "details_compound_id"),
    "VOC GC-IRMS": ("result_voc_gc_irms", "details_compound_id"),
}


# ---------------------------------------------------------------- helpers


def show_code(code: str) -> None:
    with st.expander("Python code for this step"):
        st.code(code.strip(), language="python")


def download(df: pd.DataFrame, name: str, key: str) -> None:
    st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=f"{name}.csv",
                       mime="text/csv", key=key)


def fmt_p(p: float) -> str:
    return "n/a" if p is None or np.isnan(p) else (f"{p:.2e}" if p < 0.001 else f"{p:.4f}")


def verdict(p: float, alpha: float) -> str:
    return "significant" if p < alpha else "not significant"


def plot(fig) -> None:
    fig.update_layout(width=None, autosize=True)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def read_upload(name: str, raw: bytes, sheet: str | int | None) -> pd.DataFrame:
    buf = io.BytesIO(raw)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf, sheet_name=sheet or 0)
    sep = "\t" if name.lower().endswith((".tsv", ".txt")) else None
    return pd.read_csv(buf, sep=sep, engine="python")


@st.cache_data(show_spinner="Loading from org_dma…")
def load_from_db(base_url: str, email: str, password: str, project_id: str, record_type: str, analyte_col: str,
                 grouping_cols: tuple[str, ...]) -> pd.DataFrame:
    import o2org
    from o2org.pipelines.utils import extract, pivot_wide

    kwargs = {k: v for k, v in dict(base_url=base_url, email=email, password=password).items() if v}
    client = o2org.connect(**kwargs)
    df_long = extract(client, record_type, project_id)
    index_cols = [c for c in ("sample_id", *grouping_cols) if c in df_long.columns]
    return pivot_wide(df_long, index_cols=index_cols, analyte_col=analyte_col, value_col="details_value")


def prepare(df: pd.DataFrame, features: list[str], missing: str, transform: str) -> pd.DataFrame:
    out = df.copy()
    out[features] = out[features].apply(pd.to_numeric, errors="coerce")
    if missing == "Drop rows with missing values":
        out = out.dropna(subset=features)
    elif missing == "Replace with column median":
        out[features] = out[features].fillna(out[features].median())
    elif missing == "Replace with ½ × column minimum (LOD/2)":
        out[features] = out[features].fillna(out[features].min() / 2)
    if transform == "log10":
        out = preprocessing.log10_transform(out, features)
    return out


# ---------------------------------------------------------------- sidebar: data

st.sidebar.title("🧪 o2org analysis")
source = st.sidebar.radio("Data source", ["Example dataset", "Upload a file", "org_dma database"])

raw: pd.DataFrame | None = None
load_code = ""
if source == "Example dataset":
    raw = example_dataset()
    st.sidebar.caption("Synthetic honey-provenance data (made-up values) — 60 jars, 3 regions, 2 years, fresh vs. 6 months.")
    load_code = "from o2org.ui import example_dataset\ndf = example_dataset()"
elif source == "Upload a file":
    up = st.sidebar.file_uploader("CSV, TSV or Excel — one row per sample, one column per feature",
                                  type=["csv", "tsv", "txt", "xlsx", "xls"])
    sheet = None
    if up is not None and up.name.lower().endswith((".xlsx", ".xls")):
        sheet = st.sidebar.text_input("Sheet name (blank = first sheet)") or None
    if up is not None:
        raw = read_upload(up.name, up.getvalue(), sheet)
        reader = "read_excel" if up.name.lower().endswith((".xlsx", ".xls")) else "read_csv"
        load_code = f'import pandas as pd\ndf = pd.{reader}("{up.name}")'
else:
    with st.sidebar.form("db"):
        st.caption("Leave blank to use ~/.o2org/config.ini or O2ORG_* environment variables.")
        base_url = st.text_input("Base URL")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        project_id = st.text_input("Project id")
        family = st.selectbox("Result family", list(RECORD_TYPES))
        grouping_cols = st.multiselect("Sample attributes to keep",
                                       ["sample_group", "sample_subgroup", "matrix_group", "storage_condition", "project_id",
                                        "instrument", "performed_at"], default=["sample_group", "sample_subgroup", "matrix_group"])
        submitted = st.form_submit_button("Load")
    if submitted and project_id:
        st.session_state["db_args"] = (base_url, email, password, project_id, *RECORD_TYPES[family], tuple(grouping_cols))
    if "db_args" in st.session_state:
        try:
            raw = load_from_db(*st.session_state["db_args"])
        except Exception as exc:  # noqa: BLE001 — surface any connection/API error to the user
            st.sidebar.error(f"Could not load: {exc}")
        args = st.session_state["db_args"]
        load_code = f"""import o2org
from o2org.pipelines.utils import extract, pivot_wide

client = o2org.connect()  # reads ~/.o2org/config.ini
df_long = extract(client, "{args[4]}", project_id={args[3]!r})
df = pivot_wide(df_long, index_cols={['sample_id', *args[6]]!r}, analyte_col="{args[5]}")"""

if raw is None or raw.empty:
    st.title("o2org analysis")
    st.info("Pick a data source in the sidebar to begin. Not sure? Start with the **example dataset**.")
    st.markdown(
        "This interface covers the statistical and machine-learning methods most used in JSI O2 / ISOFood "
        "publications: group comparisons (t-test, Mann-Whitney, ANOVA, Kruskal-Wallis + post-hoc), paired and "
        "two-factor designs, correlation and regression, PCA, cluster analysis, and classification "
        "(LDA, PLS-DA, SIMCA, random forest, kNN, SVM, neural networks…) with cross-validation."
    )
    st.stop()

# ---------------------------------------------------------------- sidebar: setup

numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and raw[c].nunique() > 2]
# id-like columns (a distinct value on most rows) are no use as a grouping factor
categorical_cols = [c for c in raw.columns
                    if (c not in numeric_cols or raw[c].nunique() <= 12) and raw[c].nunique() < max(13, 0.5 * len(raw))]

st.sidebar.subheader("Samples")
filter_col = st.sidebar.selectbox("Only use rows where…", ["(all rows)", *categorical_cols])
data = raw
filter_code = ""
if filter_col != "(all rows)":
    levels = sorted(raw[filter_col].dropna().unique(), key=str)
    keep = st.sidebar.multiselect(f"{filter_col} is", levels, default=levels[:1])
    data = raw[raw[filter_col].isin(keep)]
    filter_code = f"df = df[df[{filter_col!r}].isin({list(keep)!r})]"

default_group = next((c for c in ("sample_group", "region", "group") if c in categorical_cols), categorical_cols[0] if categorical_cols else None)
if default_group is None:
    st.error("No categorical column found to group samples by.")
    st.stop()
group = st.sidebar.selectbox("Group samples by", categorical_cols, index=categorical_cols.index(default_group))

skip = {"sample_id", "id", group}
feature_options = [c for c in numeric_cols if c not in skip]
features = st.sidebar.multiselect("Features (measured variables)", feature_options, default=feature_options)
if not features:
    st.warning("Select at least one feature in the sidebar.")
    st.stop()

st.sidebar.subheader("Preprocessing")
missing = st.sidebar.selectbox("Missing values", ["Replace with column median", "Replace with ½ × column minimum (LOD/2)",
                                                  "Drop rows with missing values", "Leave as missing"])
transform = st.sidebar.selectbox("Transform", ["none", "log10"], help="log10 often makes concentration data closer to normal.")
alpha = st.sidebar.select_slider("Significance level α", [0.001, 0.01, 0.05, 0.1], value=0.01)
correction = st.sidebar.selectbox("Multiple-testing correction", ["fdr_bh", "bonferroni", None],
                                  format_func=lambda c: {"fdr_bh": "Benjamini-Hochberg (FDR)", "bonferroni": "Bonferroni", None: "None"}[c])

df = prepare(data, features, missing, transform).dropna(subset=[group])
prep_code = "\n".join(filter(None, [
    load_code, filter_code,
    f"features = {features!r}",
    {"Drop rows with missing values": "df = df.dropna(subset=features)",
     "Replace with column median": "df[features] = df[features].fillna(df[features].median())",
     "Replace with ½ × column minimum (LOD/2)": "df[features] = df[features].fillna(df[features].min() / 2)"}.get(missing, ""),
    "from o2org.pipelines import preprocessing\ndf = preprocessing.log10_transform(df, features)" if transform == "log10" else "",
]))

# ---------------------------------------------------------------- main

st.title("o2org analysis")
counts = df[group].value_counts()
st.caption(f"{len(df)} samples · {len(features)} features · grouped by **{group}**: "
           + ", ".join(f"{g} ({n})" for g, n in counts.items()))

tabs = st.tabs(["📋 Data", "⚖️ Compare groups", "📈 Correlation & regression", "🧭 PCA & clustering",
                "🤖 Classification", "🔢 Categories", "📖 Which test?"])

# ---- Data
with tabs[0]:
    st.subheader("Data preview")
    st.dataframe(df, use_container_width=True, height=300)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary per group")
        feat = st.selectbox("Feature", features, key="desc_feat")
        desc = stats.descriptive_stats(df, feat, group)
        desc["rsd_%"] = desc["std"] / desc["mean"] * 100
        st.dataframe(desc, use_container_width=True)
    with c2:
        st.subheader("Missing values (before preprocessing)")
        miss = data[features].apply(pd.to_numeric, errors="coerce").isna().sum()
        st.dataframe(miss[miss > 0].rename("missing").to_frame() if miss.any() else pd.DataFrame({"missing": []}),
                     use_container_width=True)
        if not miss.any():
            st.caption("No missing values.")
    plot(viz.kde_density_plot(df, [feat], group))
    show_code(prep_code + f"\n\nfrom o2org.pipelines import stats\nstats.descriptive_stats(df, {feat!r}, {group!r})")

# ---- Compare groups
with tabs[1]:
    design = st.radio("Experimental design", ["Independent groups", "Two grouping factors", "Same samples measured repeatedly"],
                      horizontal=True, key="design",
                      help="See the 'Which test?' tab if you're unsure.")

    if design == "Independent groups":
        n_groups = df[group].nunique()
        test_choice = st.selectbox(
            "Test", ["auto", "welch", "mannwhitney"] if n_groups == 2 else ["auto", "anova", "kruskal"],
            format_func=lambda t: {"auto": "Automatic (Shapiro-Wilk decides)", "welch": "Welch's t-test",
                                   "mannwhitney": "Mann-Whitney U", "anova": "One-way ANOVA", "kruskal": "Kruskal-Wallis"}[t])
        res = stats.test_all_features(df, features, group, test=test_choice, alpha=alpha, correction=correction)
        if res.empty:
            st.warning("Need at least two groups with data.")
        else:
            st.markdown(f"**{int(res['significant'].sum())} of {len(res)} features** differ between groups "
                        f"(α = {alpha}, corrected p-values).")
            st.dataframe(res.style.format({"statistic": "{:.3f}", "pvalue": "{:.2e}", "pvalue_corrected": "{:.2e}"})
                         .apply(lambda r: ["background-color: rgba(0,204,150,.15)" if r["significant"] else ""] * len(r), axis=1),
                         use_container_width=True)
            download(res, "group_tests", "dl_tests")

            sig = res.loc[res["significant"], "feature"].tolist()
            chosen = st.multiselect("Show box plots for", features, default=sig[:6] or features[:3])
            if chosen:
                plot(viz.box_plot(df, chosen, group))

            st.subheader("One feature in detail")
            f = st.selectbox("Feature", res["feature"].tolist(), key="detail_feat")
            c1, c2, c3 = st.columns(3)
            norm = stats.shapiro_by_group(df, f, group, alpha=alpha)
            lev = stats.levene_test(df, f, group)
            eff = stats.effect_size(df, f, group)
            c1.markdown("**Normality (Shapiro-Wilk)**")
            c1.dataframe(norm, use_container_width=True, hide_index=True)
            c2.markdown("**Equal variances (Levene)**")
            c2.write(f"p = {fmt_p(lev['pvalue'])} → variances {'differ' if lev['pvalue'] < alpha else 'look similar'}")
            c3.markdown("**Effect size**")
            c3.write("  \n".join(f"{k.replace('_', ' ')} = {float(v):.3f}" for k, v in eff.items()))
            post_code = ""
            if n_groups >= 3:
                st.markdown("**Post-hoc: which groups differ?**")
                ph = stats.posthoc(df, f, group, alpha=alpha)
                st.caption(f"{'Tukey HSD' if ph['test'].iloc[0] == 'tukey' else 'Dunn (Bonferroni-adjusted)'} — chosen by the normality check.")
                st.dataframe(ph, use_container_width=True, hide_index=True)
                post_code = f"\nstats.posthoc(df, {f!r}, {group!r}, alpha={alpha})  # Tukey HSD or Dunn"
            show_code(prep_code + f"""

from o2org.pipelines import stats, viz
tests = stats.test_all_features(df, features, {group!r}, test={test_choice!r}, alpha={alpha}, correction={correction!r})
viz.box_plot(df, tests.loc[tests.significant, "feature"].tolist(), {group!r}).show()
stats.shapiro_by_group(df, {f!r}, {group!r})
stats.levene_test(df, {f!r}, {group!r})
stats.effect_size(df, {f!r}, {group!r}){post_code}""")

    elif design == "Two grouping factors":
        others = [c for c in categorical_cols if c != group]
        if not others:
            st.info("Need a second categorical column.")
        else:
            factor_b = st.selectbox("Second factor", others)
            f = st.selectbox("Feature", features, key="tw_feat")
            test2 = st.selectbox("Test", ["auto", "anova", "art"],
                                 format_func=lambda t: {"auto": "Automatic (residual normality decides)", "anova": "Two-way ANOVA",
                                                        "art": "Aligned Rank Transform ANOVA (non-parametric)"}[t])
            try:
                tw = stats.two_way_anova(df, f, group, factor_b, test=test2, alpha=alpha)
                st.dataframe(tw, use_container_width=True, hide_index=True)
                inter = tw.loc[tw["effect"].str.contains(":"), "significant"].any()
                if inter:
                    st.warning("The interaction is significant: the effect of one factor depends on the level of the other. "
                               "Describe effects per level rather than overall.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not fit the model: {exc}")
            plot(viz.interaction_plot(df, f, group, factor_b))
            show_code(prep_code + f"\n\nstats.two_way_anova(df, {f!r}, {group!r}, {factor_b!r}, test={test2!r})\n"
                                  f"viz.interaction_plot(df, {f!r}, {group!r}, {factor_b!r}).show()")

    else:
        st.caption("Uses one row per sample × measurement. Tip: don't filter out the time-point column in the sidebar.")
        cols = list(df.columns)
        subj = st.selectbox("Sample identifier", cols, index=cols.index("sample_id") if "sample_id" in cols else 0)
        cond_options = [c for c in categorical_cols if c != subj]
        cond = st.selectbox("Measurement / time point column", cond_options,
                            index=cond_options.index("storage") if "storage" in cond_options else 0)
        f = st.selectbox("Feature", features, key="pair_feat")
        n_cond = df[cond].nunique()
        try:
            if n_cond == 2:
                r = stats.paired_test(df, f, subj, cond, alpha=alpha)
                name = {"paired_t": "Paired t-test", "wilcoxon": "Wilcoxon signed-rank"}[r["test"]]
                st.metric(name, f"p = {fmt_p(r['pvalue'])}", f"mean change {r['mean_diff']:+.3g}")
                st.write(f"{r['n_pairs']} samples measured under both {r['conditions'][0]!r} and {r['conditions'][1]!r} → "
                         f"**{verdict(r['pvalue'], alpha)}**.")
                code = f"stats.paired_test(df, {f!r}, subject_col={subj!r}, condition_col={cond!r})"
            else:
                r = stats.repeated_measures_test(df, f, subj, cond, alpha=alpha)
                name = {"rm_anova": "Repeated-measures ANOVA", "friedman": "Friedman test"}[r["test"]]
                st.metric(name, f"p = {fmt_p(r['pvalue'])}")
                st.write(f"{r['n_subjects']} complete samples across {len(r['conditions'])} time points → **{verdict(r['pvalue'], alpha)}**.")
                code = f"stats.repeated_measures_test(df, {f!r}, subject_col={subj!r}, condition_col={cond!r})"
            plot(viz.interaction_plot(df.assign(_all="all samples"), f, cond, "_all"))
            show_code(prep_code + "\n\n" + code)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{exc}")

# ---- Correlation & regression
with tabs[2]:
    method = st.selectbox("Correlation", ["spearman", "pearson", "kendall"],
                          format_func=lambda m: {"spearman": "Spearman (any monotonic trend)", "pearson": "Pearson (linear)",
                                                 "kendall": "Kendall's τ"}[m])
    if len(features) >= 2:
        r, p = stats.correlation_matrix(df, features, method=method, correction=correction)
        plot(viz.correlation_heatmap(r, p, alpha=alpha, title=f"{method.title()} correlation (only significant cells labelled)"))

        st.subheader("Two features")
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("x", features, key="corr_x")
        y = c2.selectbox("y", [f for f in features if f != x], key="corr_y")
        colour = c3.checkbox("Colour by group", value=True)
        cr = stats.correlation(df[x], df[y], method="auto", alpha=alpha)
        st.write(f"{cr['method'].title()} r = **{cr['r']:.3f}**, p = {fmt_p(cr['pvalue'])} (n = {cr['n']}) → {verdict(cr['pvalue'], alpha)}. "
                 "Correlation is not causation — both may depend on region, batch, etc.")
        plot(viz.regression_plot(df, x, y, group_col=group if colour else None))

        st.subheader("Multiple linear regression")
        target = st.selectbox("Predict", features, key="reg_y")
        preds = st.multiselect("from", [f for f in features if f != target], default=[f for f in features if f != target][:3])
        if preds:
            model = stats.linear_regression(df, target, preds)
            st.write(f"R² = **{model.rsquared:.3f}**, adjusted R² = {model.rsquared_adj:.3f}, model p = {fmt_p(model.f_pvalue)}")
            st.dataframe(stats.regression_table(model), use_container_width=True, hide_index=True)
        show_code(prep_code + f"""

r, p = stats.correlation_matrix(df, features, method={method!r})
viz.correlation_heatmap(r, p).show()
stats.correlation(df[{x!r}], df[{y!r}])
model = stats.linear_regression(df, {target!r}, {preds!r})
print(model.summary())""")
    else:
        st.info("Select at least two features.")

# ---- PCA & clustering
with tabs[3]:
    if len(features) < 2:
        st.info("Select at least two features.")
    else:
        c1, c2 = st.columns(2)
        scale = c1.selectbox("Scaling", ["standard", "pareto", None],
                             format_func=lambda s: {"standard": "Autoscale (z-score)", "pareto": "Pareto", None: "None"}[s])
        n_comp = c2.slider("Components", 2, min(10, len(features)), min(4, len(features)))
        pca_res = ml.pca(df, features, n_components=n_comp, scale=scale)
        pcs = [c for c in pca_res.scores.columns]
        c1, c2 = st.columns(2)
        px_ = c1.selectbox("x axis", pcs, index=0)
        py_ = c2.selectbox("y axis", pcs, index=1)
        g = df[group]
        plot(viz.pca_biplot(pca_res, group=g, pcs=(px_, py_)))
        c1, c2 = st.columns(2)
        with c1:
            plot(viz.scree_plot(pca_res))
        with c2:
            out = ml.hotellings_t2_outliers(pca_res.scores, pcs=(px_, py_))
            st.markdown(f"**Hotelling's T² outliers (95%)**: {int(out.sum())}")
            if out.any():
                st.dataframe(df.loc[out[out].index, [c for c in ("sample_id", group) if c in df.columns]], hide_index=True)
            st.markdown("**Loadings**")
            st.dataframe(pca_res.loadings.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Cluster analysis")
        c1, c2, c3 = st.columns(3)
        algo = c1.selectbox("Method", ["Hierarchical (HCA)", "k-means"])
        k = c2.slider("Number of clusters", 2, 10, int(min(max(df[group].nunique(), 2), 10)))
        if algo.startswith("Hier"):
            link = c3.selectbox("Linkage", ["ward", "average", "complete", "single"])
            labels_col = "sample_id" if "sample_id" in df.columns else None
            hca = ml.hierarchical_cluster(df, features, n_clusters=k, method=link, scale=scale,
                                          sample_labels=(df[labels_col].astype(str) + " · " + df[group].astype(str)) if labels_col else df[group])
            plot(viz.dendrogram_plot(hca))
            clusters = hca.labels
            ccode = f"hca = ml.hierarchical_cluster(df, features, n_clusters={k}, method={link!r})\nviz.dendrogram_plot(hca).show()"
        else:
            km = ml.kmeans_cluster(df, features, k=k, scale=scale)
            clusters = km.labels + 1
            ccode = f"km = ml.kmeans_cluster(df, features, k={k})"
        agree = ml.cluster_agreement(df[group], clusters)
        st.write(f"Agreement between clusters and **{group}**: adjusted Rand = {agree['adjusted_rand']:.2f}, "
                 f"NMI = {agree['nmi']:.2f} (1 = identical, 0 = unrelated).")
        st.dataframe(pd.crosstab(df[group], pd.Series(clusters, index=df.index, name="cluster")), use_container_width=True)
        show_code(prep_code + f"""

from o2org.pipelines import ml, viz
pca = ml.pca(df, features, n_components={n_comp}, scale={scale!r})
viz.pca_biplot(pca, group=df[{group!r}]).show()
ml.hotellings_t2_outliers(pca.scores)
{ccode}
ml.cluster_agreement(df[{group!r}], clusters)""")

# ---- Classification
with tabs[4]:
    st.caption(f"Predict **{group}** from the selected features. All scores are cross-validated "
               "(stratified k-fold) — never measured on the samples the model was trained on.")
    names = list(ml.CLASSIFIERS)
    labels = {k: v[0] for k, v in ml.CLASSIFIERS.items()}
    c1, c2 = st.columns(2)
    folds = c1.slider("Cross-validation folds", 2, 10, 5)
    id_options = ["(every row is a different sample)"] + [c for c in df.columns if c not in features and c != group]
    repeated_id = next((c for c in ("sample_id", "sample", "jar_id") if c in df.columns and df[c].duplicated().any()), None)
    gsel = c2.selectbox("Rows that share this id are the same sample", id_options,
                        index=id_options.index(repeated_id) if repeated_id else 0,
                        help="Replicates or time points of one sample are kept in the same fold. "
                             "Otherwise the model is tested on samples it has already seen and accuracy is inflated.")
    gcol = None if gsel == id_options[0] else gsel
    if gcol is None and "sample_id" in df.columns and df["sample_id"].duplicated().any():
        st.warning("Some samples have several rows — scores will be optimistic unless you pick the sample id column.")

    st.subheader("Compare models")
    chosen_models = st.multiselect("Models", names, default=["lda", "plsda", "simca", "random_forest", "knn", "svm"],
                                   format_func=labels.get)
    if st.button("Run comparison", type="primary"):
        with st.spinner("Cross-validating…"):
            st.session_state["comparison"] = ml.compare_classifiers(df, features, group, models=chosen_models, n_splits=folds, groups_col=gcol)
    if "comparison" in st.session_state:
        comp = st.session_state["comparison"]
        st.dataframe(comp, use_container_width=True, hide_index=True)
        if "balanced_accuracy" in comp:
            plot(viz.model_comparison_plot(comp))
        download(comp, "model_comparison", "dl_comp")

    st.subheader("One model in detail")
    m = st.selectbox("Model", names, index=names.index("plsda"), format_func=labels.get)
    try:
        with st.spinner("Training…"):
            res = ml.classify(df, features, group, model=m, n_splits=folds, groups_col=gcol)
        mt = res.metrics
        c = st.columns(4)
        c[0].metric("Accuracy", f"{mt['accuracy']:.1%}")
        c[1].metric("Balanced accuracy", f"{mt['balanced_accuracy']:.1%}")
        c[2].metric("Macro F1", f"{mt['f1_macro']:.2f}")
        c[3].metric("ROC AUC", f"{mt['roc_auc']:.2f}" if "roc_auc" in mt else "–")
        c1, c2 = st.columns(2)
        with c1:
            plot(viz.confusion_heatmap(res.confusion))
        with c2:
            st.dataframe(res.report.style.format("{:.2f}"), use_container_width=True)
            if len(res.classes) == 2 and res.y_proba is not None:
                plot(viz.roc_plot((res.y_true == res.classes[1]).astype(int), res.y_proba[:, 1]))

        if m == "plsda":
            pls = ml.plsda(df, features, group, n_splits=folds, groups_col=gcol)
            c1, c2 = st.columns(2)
            with c1:
                ev = pls.explained_x * 100
                plot(viz.scores_scatter(pls.scores, group, title="PLS-DA scores",
                                        x_label=f"LV1 ({ev[0]:.1f}% of X)", y_label=f"LV2 ({ev[1]:.1f}% of X)" if len(ev) > 1 else None))
            with c2:
                plot(viz.importance_bar(pls.vip, value_col="vip", threshold=1.0, title="VIP scores (VIP > 1 = important)"))
        elif m == "lda":
            lda = ml.lda_classify(df, features, group)
            plot(viz.scores_scatter(lda.scores, group, title="LDA scores"))

        c1, c2 = st.columns(2)
        if c1.button("Feature importance (permutation)"):
            with st.spinner("Shuffling features…"):
                st.session_state["imp"] = (m, ml.feature_importance(res, df, group, n_repeats=10))
        if c2.button("Permutation test (is it better than chance?)"):
            with st.spinner("Refitting on shuffled labels…"):
                st.session_state["perm"] = (m, ml.permutation_test(df, features, group, model=m, n_permutations=100, n_splits=folds, groups_col=gcol))
        if st.session_state.get("imp", (None,))[0] == m:
            plot(viz.importance_bar(st.session_state["imp"][1], title=f"Permutation importance — {labels[m]}"))
        if st.session_state.get("perm", (None,))[0] == m:
            perm = st.session_state["perm"][1]
            plot(viz.permutation_hist(perm))
            st.write(f"p = {perm['pvalue']:.3f} → the model is "
                     f"{'**better than chance**' if perm['pvalue'] < 0.05 else '**not** clearly better than chance'}.")
        show_code(prep_code + f"""

from o2org.pipelines import ml, viz
ml.compare_classifiers(df, features, {group!r}, models={chosen_models!r}, n_splits={folds}, groups_col={gcol!r})
res = ml.classify(df, features, {group!r}, model={m!r}, n_splits={folds}, groups_col={gcol!r})
res.metrics, res.confusion
viz.confusion_heatmap(res.confusion).show()
ml.feature_importance(res, df, {group!r})
ml.permutation_test(df, features, {group!r}, model={m!r}, groups_col={gcol!r})
# PLS-DA scores & VIP:  pls = ml.plsda(df, features, {group!r}); pls.vip
# your own sklearn model works too:  ml.classify(df, features, {group!r}, model=MyEstimator())""")
    except Exception as exc:  # noqa: BLE001
        st.error(f"{labels[m]} could not be trained on this data: {exc}")

    with st.expander("Predict a continuous property instead (PLS regression)"):
        tgt = st.selectbox("Target", features, key="plsr_y")
        xs = [f for f in features if f != tgt]
        if xs:
            pr = ml.pls_regression(df, xs, tgt, n_splits=folds)
            st.write(f"R² = {pr['r2']:.3f} · Q² (cross-validated) = **{pr['q2']:.3f}** · RMSECV = {pr['rmsecv']:.3g}")
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter(x=pr["y_true"], y=pr["y_cv"], mode="markers"))
            lo, hi = float(np.min(pr["y_true"])), float(np.max(pr["y_true"]))
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
            fig.update_layout(title="Measured vs. cross-validated prediction", xaxis_title="measured", yaxis_title="predicted")
            plot(fig)

# ---- Categories
with tabs[5]:
    cats = [c for c in raw.columns if raw[c].nunique() <= 12]
    if len(cats) < 2:
        st.info("Need two categorical columns (e.g. region and a presence/absence marker).")
    else:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Rows", cats, index=cats.index(group) if group in cats else 0)
        b = c2.selectbox("Columns", [c for c in cats if c != a])
        ct = stats.contingency_test(df, a, b)
        st.dataframe(ct["table"], use_container_width=True)
        name = "Fisher's exact test" if ct["test"] == "fisher" else "Chi-square test"
        st.write(f"**{name}**: p = {fmt_p(ct['pvalue'])} → {verdict(ct['pvalue'], alpha)}.")
        if ct.get("warning"):
            st.warning("Some expected counts are below 5, so the chi-square p-value is approximate (Fisher's test needs a 2×2 table).")
        show_code(prep_code + f"\n\nstats.contingency_test(df, {a!r}, {b!r})")

# ---- Guide
with tabs[6]:
    guide = Path(__file__).resolve().parents[3] / "docs" / "choosing_a_test.md"
    if guide.exists():
        st.markdown(guide.read_text(encoding="utf-8"))
    else:
        st.markdown("See `docs/choosing_a_test.md` in the o2org repository.")
