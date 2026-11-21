# o2org

A Python client for the o2_org_dma metabolomics database: log in with your
email/password and pull any table (compounds, projects, samples,
experiments, ...) into Python, or write records back.

No SQL, no `requests` calls, no boilerplate — just Python data.

**New here?** [`examples/o2org_walkthrough.ipynb`](examples/o2org_walkthrough.ipynb)
runs through everything below end to end — connecting, retrieving data, and
error handling — against a real org_dma instance, with every cell's output
saved so you can read it without running anything. Open it with
`pip install -e ".[notebook]"` then `jupyter notebook examples/`.

> **Status:** this library is a REST API client. It expects org_dma to expose
> the auth + CRUD endpoints described in [API contract](#api-contract-what-org_dma-needs-to-expose) below.
> The routes `samples`, `experiment-records` (including the project-scoped,
> pipelines-supporting listing and the record-type schema endpoint) are
> implemented on the Laravel side as of this writing — check that section if
> you're pointing this client at an older org_dma checkout.

## Installation

Requires Python 3.9+.

```bash
pip install -e .        # from a checkout of this repo
```

(Once published, this will just be `pip install o2org`.)

This pulls in `requests` and `pandas` — nothing else to install by hand.

## Connecting

Use a config file, so you never type a password in a script:

```ini
# ~/.o2org/config.ini
[default]
base_url = https://your-o2-org-dma-instance.example
email = you@example.com
password = your-password
```


```python
import o2org

client = o2org.connect()
```

or pass the parameters directly in your script:

```python
import o2org

client = o2org.connect(
    base_url="https://your-o2-org-dma-instance.example",
    email="you@example.com",
    password="your-password",
)
```

<!-- Use it as a context manager to log out automatically when you're done:

```python
with o2org.connect() as client:
    compounds = client.compounds.list()
``` -->

## Retrieving data

This is what you'll use most. Every entity in the database is available the
same way — as an attribute on `client`, named after the table
(`client.compounds`, `client.projects`, `client.samples`,
`client.experiments`, `client.diseases`, `client.biomarkers`,
`client.ontologies`, `client.sources`, `client.taxonomies`,
`client.compound_synonyms`, `client.retention_indices`,
`client.compound_disease_associations`, `client.compound_ontologies`,
`client.compound_biomarkers`, `client.project_compounds`,
`client.experiment_records`, `client.samplings`).

```python
# list / search — any keyword becomes a filter, but which keys are recognized
# (and whether they match exactly or partially) is up to your server's own
# filtering implementation. org_dma's Compound model uses the
# mehradsadeghi/laravel-filter-querystring package: exact match is
# canonical_name="Glucose", partial/substring match needs like="canonical_name,glucose".
# An unrecognized key (e.g. name=) is silently ignored, not an error — so a
# "filtered" list() the same size as the whole table usually means the key
# didn't match, not that there's no data.
client.compounds.list()
client.compounds.list(canonical_name="D-Glucose")
client.compounds.list(like="canonical_name,glucose")
client.compounds.list(all_pages=True)

client.projects.list()
client.projects.get(5)

client.project_compounds.list(project_id=5)  # nested under a project

# a single record
client.compounds.get(42)
```

`list()` is lazy — calling it never touches the network by itself. What
happens next depends on what you do with the result:

```python
compounds = client.compounds.list(like="canonical_name,acid")

compounds.count()   # one cheap request — reads the server's own total, doesn't fetch records
compounds.pages     # same: one cheap request, or None if the response wasn't paginated
compounds.get()     # fetches every matching record (see paging behavior below)
compounds.names()   # needs the actual records, so this fetches everything too
```

This matters because a filtered search on a big table can match thousands of
rows across hundreds of pages — `acid` matches over 30,000 compounds in a
78,000-row real table. `.count()`/`.pages` never walk those pages just to
hand you a number; they read it straight from the server's pagination
metadata in a single request. `.get()` (or iterating/indexing the result
directly) is what actually fetches the records, and returns a `RecordList` —
a plain list of dicts with `.count()` and `.names()` added, so once you have
the records, counting/naming them doesn't need pandas either.

That fetch walks every page and returns every match by default when you filter
— a filter is supposed to narrow things down, so "give me everything that
matches" is the reasonable default. An *unfiltered* `client.compounds.list()`
only fetches the first page by default instead, since "give me the whole
table" could mean tens of thousands of rows; pass `all_pages=True` there if
you really want all of it. Nested lists (`project_compounds`,
`experiment_records`) always default to fetching every page, since they're
already scoped to one parent.

Fetch a specific page with `page=`; that also disables the "fetch everything"
default (you asked for one page, so you get one page) — `.count()`/`.pages`
are unaffected either way, since they always report the query's true total:

```python
client.compounds.list(page=2)
```

`project_compounds` and `experiment_records` are nested under their parent (a
project, an experiment), so their `list()` takes a parent id:

```python
client.project_compounds.list(project_id=5)
```

## Other procedures: creating & updating records

Less common than retrieval, but supported:

```python
# create
client.compounds.create(canonical_name="D-Glucose", molecular_formula="C6H12O6")

# update
client.compounds.update(42, molecular_formula="C6H12O6")
```

Importing a whole spreadsheet? `bulk_create` inserts a list of dicts (or a
DataFrame) row by row and reports which ones failed instead of stopping at the
first bad row:

```python
import pandas as pd

rows = pd.read_excel("new_compounds.xlsx")
result = client.compounds.bulk_create(rows)
print(f"created {len(result['created'])}, failed {len(result['errors'])}")
for row, error in result["errors"]:
    print(row, "->", error)
```

`project_compounds` and `experiment_records` also take a parent id when
creating (they're read the same way as everything else via
`.list(project_id=...)` / `.list(experiment_id=...)` above, but written to a
nested route):

```python
client.project_compounds.create(project_id=5, custom_name="Peak 12", mz=180.06, rt=4.2)
# .get()/.update() address the record directly — no parent id needed:
client.project_compounds.update(99, is_mapped=True)
```

There is no `.delete()` on any resource — this client does not support
deleting records.

## Pipelines

`o2org.pipelines` adds domain-specific statistical/ML pipelines on top of the
REST client — one pair of `stats_pipeline()`/`ml_pipeline()` functions per
result family (stable isotopes, elemental composition, MK GC-MS/IRMS, VOC
GC-MS/IRMS), plus every underlying transform/test/plot as a standalone
function you can call yourself. It's an optional extra:

```bash
pip install -e ".[pipelines]"   # numpy, scipy, scikit-learn, statsmodels, plotly, kaleido
```

Every pipeline starts the same way — extracting one record_type for one
project out of `experiment_records.details` (EAV JSON) and pivoting it from
long (one row per analyte/compound) to wide (one row per sample):

```python
from o2org.pipelines.utils import extract, pivot_wide

df_long = extract(client, "result_stable_isotopes", project_id=42)
df_wide = pivot_wide(df_long, index_cols=["sample_id", "sample_group"], analyte_col="details_analyte")
```

The pipelines call this for you — and one call already returns the *full*
report: sample counts, per-analyte p-values and verdicts, and every figure,
all rendered inline as the last expression in a Jupyter cell:

```python
from o2org.pipelines import isotopes

isotopes.stats_pipeline(client, project_id=42, grouping="sample_group")
```

It's still a plain dict underneath, so picking out one piece works exactly
like you'd expect — nothing about the report display changes how you use it programmatically:

```python
result = isotopes.stats_pipeline(client, project_id=42)
result["tests"]                        # per-analyte Kruskal-Wallis/ANOVA + FDR correction
result["significant_analytes"]         # analytes below the significance threshold
result["figures"]["box"].show()        # any individual figure, on its own
```

Pass `include_figures=False` to skip building/embedding plots entirely (data
and test results are unaffected) — faster for headless/batch use.

| Family | Module | Stats pipeline | ML pipeline |
|---|---|---|---|
| Stable isotopes | `isotopes` | Shapiro → Kruskal-Wallis/ANOVA → correction → box/KDE plots, isotope biplot (δ13C vs δ15N) | LDA (default) or Random Forest baseline for provenance classification |
| Elemental composition | `elemental` | Same stats chain per element, plus H/C, O/C, N/C molar ratios → Van Krevelen plot | K-means clustering over the ratio space (default), or Random Forest |
| MK GC-MS / MK GC-IRMS | `mk_gcms` | Same stats chain per fatty acid | IS/median normalize → batch-wise LOD/2 impute → RSD diagnostics → RF + stratified CV, ROC, MDI importance + elbow cutoff |
| VOC GC-MS / VOC GC-IRMS | `voc_gcms` | Row-threshold filter → LOQ/2 impute → Shapiro (raw + log10) → correction → box/median-bar/KDE → sample- and compound-level PCA | Same normalize/impute/RF chain as `mk_gcms`, at full strength (elbow-cutoff feature selection matters here — VOC panels can run to hundreds of compounds) |

`grouping` (which sample attribute defines the groups being compared) accepts
any real column but is typically one of `sample_group`, `sample_subgroup`,
`matrix_group`, or `project_id` — see `o2org.pipelines.utils.GROUPING_CANDIDATES`.

The individual building blocks are plain functions, meant to be called
directly if a pipeline's defaults don't fit:

```python
from o2org.pipelines import stats, ml, viz, preprocessing

stats.shapiro_by_group(df_wide, "d13c", "sample_group")
stats.kruskal_wallis(df_wide, "d13c", "sample_group")
preprocessing.lod_half_impute(df_wide, ["d13c", "d15n"], batch_col="instrument")
pca_result = ml.pca(df_wide, columns=["d13c", "d15n"])
viz.pca_biplot(pca_result, group=df_wide["sample_group"])
```

Below, `df_wide` is any wide DataFrame — rows are samples, one column per
feature (analyte/element/compound), plus a group column — like the one
`pivot_wide()` produces above. Everything here works the same whether you got
it from `extract()`/`pivot_wide()` or built it yourself.

### Normality & group comparison (`stats`)

```python
from o2org.pipelines import stats

stats.descriptive_stats(df_wide, "d13c", "sample_group")   # mean/median/std/n per group
stats.shapiro_by_group(df_wide, "d13c", "sample_group")     # Shapiro-Wilk normality, one row per group
stats.qq_regression(df_wide["d13c"])                        # (slope, intercept, r) — secondary normality check
stats.rsd(df_wide, ["d13c", "d15n"])                         # relative standard deviation (%) per feature

stats.kruskal_wallis(df_wide, "d13c", "sample_group")        # non-parametric group comparison
stats.anova(df_wide, "d13c", "sample_group")                 # parametric group comparison (when normality holds)

# multiple-testing correction over a batch of p-values
stats.bonferroni_correction([0.001, 0.02, 0.03, 0.5])
stats.benjamini_hochberg([0.001, 0.02, 0.03, 0.5])

# the per-feature driver every pipeline's stats_pipeline() calls: picks
# Kruskal-Wallis vs. ANOVA per feature (based on Shapiro normality), then
# corrects across all of them at once
stats.test_all_features(df_wide, ["d13c", "d15n"], "sample_group", correction="fdr_bh")
```

### Preprocessing & normalization (`preprocessing`)

```python
from o2org.pipelines import preprocessing

preprocessing.log10_transform(df_wide, ["d13c", "d15n"])
preprocessing.pareto_scale(df_wide, ["d13c", "d15n"])                       # mean-center ÷ √std, the standard pre-PCA scaling
preprocessing.is_normalize(df_wide, is_column="d13c", columns=["d15n"])     # ratio to an internal-standard feature
preprocessing.batch_median_normalize(df_wide, ["d13c", "d15n"], batch_col="instrument")
preprocessing.zero_nan_threshold(df_wide, ["d13c", "d15n"], threshold=0.2)  # drop features too often missing

imputed, imputed_counts = preprocessing.lod_half_impute(df_wide, ["d13c", "d15n"], batch_col="instrument")
outlier_mask = preprocessing.grubbs_outliers(df_wide["d13c"].values)        # iterative Grubbs' test
```

### Dimensionality reduction, classification & clustering (`ml`)

```python
from o2org.pipelines import ml

pca_result = ml.pca(df_wide, columns=["d13c", "d15n"])       # .scores, .loadings, .explained_variance_ratio
ml.hotellings_t2_outliers(pca_result.scores)                  # outlier flags on the PC1/PC2 plane

rf = ml.random_forest_classify(df_wide, ["d13c", "d15n"], "sample_group", positive_class="food", negative_class="environment")
ml.tune_threshold(rf.y_true, rf.y_proba)                      # precision/recall/F1 across a threshold sweep

# the building blocks random_forest_classify() is composed from, if you want
# a different estimator or label set than the binary positive/negative shape:
from sklearn.ensemble import RandomForestClassifier
y_encoded, encoder = ml.label_encode(df_wide["sample_group"])
ml.stratified_kfold_predict(df_wide[["d13c", "d15n"]], y_encoded, RandomForestClassifier())

importances = ml.mdi_importance(rf.model, ["d13c", "d15n"])   # Mean Decrease in Impurity, sorted descending
cutoff, elbow_idx = ml.elbow_cutoff(importances["importance"].values)

ml.kmeans_cluster(df_wide, ["d13c", "d15n"], k=3)              # unsupervised exploration
ml.lda_classify(df_wide, ["d13c", "d15n"], "sample_group")     # lighter-weight alternative to Random Forest
```

### Visualization (`viz`)

Every function returns a Plotly `go.Figure` (hover tooltips included) — call
`.show()` in a notebook, or save with `o2org.pipelines.utils.save_figure(fig, "path/to/name")`
(writes both a `.png` and an interactive `.html`).

```python
from o2org.pipelines import viz

viz.box_plot(df_wide, ["d13c", "d15n"], "sample_group")
viz.median_bar_plot(df_wide, ["d13c", "d15n"], "sample_group")
viz.kde_density_plot(df_wide, ["d13c", "d15n"], "sample_group")

viz.pca_score_scatter(pca_result, group=df_wide["sample_group"])
viz.pca_loadings_plot(pca_result)
viz.pca_biplot(pca_result, group=df_wide["sample_group"])

viz.roc_plot(rf.y_true, rf.y_proba)
viz.threshold_sensitivity_plot(ml.tune_threshold(rf.y_true, rf.y_proba))
viz.confusion_matrix_table(rf.y_true, rf.y_pred, rf.target_names)   # a DataFrame, not a figure
viz.mdi_elbow_plot(importances["importance"].values, elbow_idx, cutoff)
viz.top_features_barh(importances, cutoff)

viz.imputed_counts_plot(imputed_counts)
viz.rsd_comparison_plot({"Raw": stats.rsd(df_wide, ["d13c"]), "Normalized": stats.rsd(imputed, ["d13c"])})

viz.isotope_biplot(df_wide, "d13c", "d15n", group_col="sample_group")   # family-specific: isotopes
viz.van_krevelen_plot(df_wide, group_col="sample_group")                # family-specific: elemental
```

### Methods from the O2 / ISOFood publication survey

The `stats` and `ml` modules cover the statistical and ML methods most often
used in JSI O2 publications (a survey of ~1,450 OpenAlex records; ISOFood
papers lean on ANOVA/Kruskal-Wallis, PCA, LDA, PLS-DA/OPLS-DA, SIMCA and
cross-validation; the wider department on correlation, regression and
t-tests). Which test fits which experimental design is explained in
[`docs/choosing_a_test.md`](docs/choosing_a_test.md).

```python
from o2org.pipelines import stats, ml, viz

# group comparisons — two groups → Welch / Mann-Whitney, 3+ → ANOVA / Kruskal-Wallis
stats.test_all_features(df_wide, features, "sample_group")
stats.posthoc(df_wide, "fe", "sample_group")            # Tukey HSD or Dunn
stats.effect_size(df_wide, "fe", "sample_group")
stats.levene_test(df_wide, "fe", "sample_group")
stats.two_way_anova(df_wide, "zn", "region", "year")    # or ART ANOVA when not normal
stats.paired_test(df, "d18o", "sample_id", "storage")   # paired t / Wilcoxon
stats.repeated_measures_test(df, "d13c", "sample_id", "month")  # RM-ANOVA / Friedman
stats.contingency_test(df, "region", "pollen_marker")   # chi-square / Fisher

# correlation & regression
r, p = stats.correlation_matrix(df_wide, features, method="spearman")
model = stats.linear_regression(df_wide, "zn", ["fe", "mn"]); stats.regression_table(model)
stats.mixed_model(df_wide, "d13c ~ region", group_col="batch")   # random intercept per batch

# classification — every score is cross-validated
ml.compare_classifiers(df_wide, features, "sample_group",
                       models=["lda", "qda", "plsda", "simca", "random_forest", "tree",
                               "knn", "svm", "logistic", "gbm", "ann"])
res = ml.classify(df_wide, features, "sample_group", model="plsda", groups_col="sample_id")
res.metrics, res.confusion
ml.permutation_test(df_wide, features, "sample_group", model="plsda")  # better than chance?
ml.feature_importance(res, df_wide, "sample_group")   # model-agnostic permutation importance
pls = ml.plsda(df_wide, features, "sample_group"); pls.vip, pls.scores
ml.classify(df_wide, features, "sample_group", model=MyOwnSklearnEstimator())  # any estimator works

# unsupervised & regression
hca = ml.hierarchical_cluster(df_wide, features, n_clusters=3); viz.dendrogram_plot(hca)
ml.cluster_agreement(df_wide["sample_group"], hca.labels)
ml.pls_regression(df_wide, features, "age_months")   # PLSR with Q² / RMSECV
```

`groups_col=` matters whenever one physical sample has several rows
(technical replicates, time points): those rows are kept in the same
cross-validation fold, otherwise the model is tested on samples it has
already seen and the accuracy is too optimistic.

New plots: `viz.correlation_heatmap`, `viz.regression_plot`,
`viz.interaction_plot`, `viz.scree_plot`, `viz.dendrogram_plot`,
`viz.scores_scatter` (LDA/PLS-DA), `viz.confusion_heatmap`,
`viz.importance_bar` (VIP/permutation importance),
`viz.model_comparison_plot`, `viz.permutation_hist`.

## User interface

For point-and-click analysis there's a Streamlit app on top of the same
functions:

```bash
pip install -e ".[ui]"
o2org-ui            # or: python -m o2org.ui
```

Load a CSV/Excel file, a project straight from org_dma, or the bundled
synthetic example dataset; pick the grouping column and features; then work
through the tabs — data overview, group comparisons (independent, two-factor,
paired/repeated), correlation & regression, PCA & clustering, classification
(model comparison, confusion matrix, VIP, permutation test), categorical
tests, and the "which test?" guide. Every tab has a **Python code** expander
showing the equivalent library calls, so an analysis started in the UI can
be continued in a notebook.

## Handling errors

```python
from o2org import AuthenticationError, NotFoundError, ValidationError, APIError

try:
    client.compounds.create(canonical_name="")
except ValidationError as e:
    print(e.payload)   # the field-level errors the server returned
except NotFoundError:
    print("no such record")
except AuthenticationError:
    print("login expired — call client.login(email, password) again")
```

## API contract: what org_dma needs to expose

This client assumes standard Laravel API-resource routes. Implementing these
on the org_dma side is what makes the client work end to end:

| Purpose | Method & path | Notes |
|---|---|---|
| Login | `POST /api/login` | body `{email, password}` → `{"token": "...", "user": {...}}` (Sanctum) |
| Logout | `POST /api/logout` | invalidates the token |
| Current user | `GET /api/me` | |
| List / search | `GET /api/{resource}` | query params are filters; supports either a bare array, `{"data": [...]}`, or a paginated `{"data": [...], "meta": {...}}`. Laravel's default paginator meta (`current_page`, `last_page`, `total`, ...) is exactly what `.count()`/`.pages` read — without a `total` field, those fall back to counting the one page fetched |
| Fetch one | `GET /api/{resource}/{id}` | |
| Create | `POST /api/{resource}` | |
| Update | `PUT /api/{resource}/{id}` | |

`{resource}` is the kebab-case table name (`compounds`, `project-compounds`,
`compound-disease-associations`, etc). Every route above except login should
sit behind `auth:sanctum` middleware. `project-compounds`, `samples`, and
`experiment-records` additionally need their nested index/store routes
(`GET`/`POST /api/projects/{project}/compounds`,
`GET`/`POST /api/projects/{project}/samples`,
`GET`/`POST /api/experiments/{experiment}/records`) alongside the shallow
show/update routes. This client has no delete operation, so a `destroy` route
isn't required.

`o2org.pipelines` needs two more routes beyond the standard CRUD shape above:

| Purpose | Method & path | Notes |
|---|---|---|
| Records for a project, by record_type | `GET /api/projects/{project}/experiment-records?record_type=...` | The one call every pipeline's `extract()` is built on — every matching `experiment_record`, with its sample's `sample_group`/`sample_subgroup`/`matrix_group`/`storage_condition`/`project_id` already joined in server-side |
| EAV field schema for a record_type | `GET /api/record-types/{recordType}/schema` | `{record_type, label, family, fields: [...]}` — what `extract()` uses to flatten `details` JSON into typed columns without hardcoding a copy of the schema |

## Development

```bash
pip install -e ".[dev,pipelines,ui]"
pytest
```
