# o2org

A Python toolkit for the o2_org_dma metabolomics database, for two kinds of work:

1. **Data retrieval** — log in with your email/password and pull any table
   (compounds, projects, samples, experiments, ...) into Python for analysis or reporting.
2. **Analysis shortcuts** — one-line normalization, statistics, machine
   learning, visualization, and whole pipelines.

No SQL, no `requests` calls, no boilerplate — just Python data and plots.

**New here?** [`examples/o2org_walkthrough.ipynb`](examples/o2org_walkthrough.ipynb)
runs through everything below end to end — connecting, retrieving data, error
handling, and the full analysis pipeline — against a small in-memory fake
server (org_dma's real API isn't live yet), with every cell's output saved so
you can read it without running anything. Open it with
`pip install -e ".[notebook]"` then `jupyter notebook examples/`.

> **Status:** this library is a REST API client. It expects org_dma to expose
> the auth + CRUD endpoints described in [API contract](#api-contract-what-org_dma-needs-to-expose) below.
> As of this writing those endpoints (Sanctum login, full create/update
> routes) are not yet built on the Laravel side — see that section for exactly
> what's needed.

## Installation

Requires Python 3.9+.

```bash
pip install -e .        # from a checkout of this repo
```

(Once published, this will just be `pip install o2org`.)

This pulls in `requests`, `pandas`, `numpy`, `scipy`, `scikit-learn`, and
`matplotlib` — nothing else to install by hand.

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
directly, or passing it straight into `Dataset`/`quick_report`/anything in
`o2org.analysis`) is what actually fetches the records, and returns a
`RecordList` — a plain list of dicts with `.count()` and `.names()` added, so
once you have the records, counting/naming them doesn't need pandas either.

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

Feed the result straight into anything in `o2org.analysis` (see below) —
still lazy or already fetched, either way there's no separate "convert to a
DataFrame" step to remember.

`project_compounds` and `experiment_records` are nested under their parent (a
project, an experiment), so their `list()` takes a parent id:

```python
client.project_compounds.list(project_id=5)
```

## Analysis shortcuts

The whole point of this half of the library is that you never have to touch
scikit-learn or matplotlib directly — or even pandas. Every function below
accepts data in whatever shape you retrieved it: the still-lazy result of
`client.compounds.list()` (fetched here, since "I want to analyze this" is
unambiguous), a single dict from `client.compounds.get(42)`, a DataFrame, or a
CSV you loaded yourself. Each one inspects what it was given and converts
internally, so there's nothing to get wrong by picking the "df" version
instead of the "dict" version — there's only one version.

### The one-liner: `quick_report`

```python
from o2org.analysis import quick_report

data = client.compounds.list()

quick_report(data)                       # summary table + correlation heatmap + PCA plot
quick_report(data, target="mz")          # ...plus a regression predicting mz
quick_report(data, target="is_mapped")   # ...plus a classifier, if the target looks categorical
```

That's the entire pipeline: cleaning-aware summary, visualization, and a
model, in one call.

### Step by step: `Dataset`

When you want to control the steps yourself, `Dataset` wraps retrieved data
and chains cleanly:

```python
from o2org.analysis import Dataset

ds = Dataset(client.compounds.list())
ds.impute(method="mean")          # fill missing values
ds.normalize(method="zscore")     # scale numeric columns (or "minmax", "robust")
ds.summary()                      # per-column stats: dtype, missing, mean, std, ...
ds.correlation()                  # correlation matrix + a heatmap
ds.pca(n_components=2)            # PCA + a scatter plot of PC1 vs PC2
ds.cluster(k=3)                   # k-means + a colored scatter plot
ds.regression(target="mz")        # linear regression + predicted-vs-actual plot
ds.classify(target="is_mapped")   # logistic regression (or model="random_forest")
```

Every step that cleans data (`impute`, `normalize`, `encode`) returns `ds`
itself, so they chain: `Dataset(data).impute().normalize()`. Every step that
*analyzes* data returns a result object instead — inspect it, or call
`.plot()` again with different options:

```python
result = ds.pca(plot=False)
result.explained_variance_ratio
result.components               # DataFrame of PC1, PC2, ...
result.plot(hue="sample_group")
```

Statistical tests are one call each, no `scipy` required:

```python
ds.ttest(group_col="sample_group", value_col="mz")
ds.anova(group_col="sample_group", value_col="mz")
ds.chi_square(col1="sample_type", col2="storage_condition")
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
sit behind `auth:sanctum` middleware. `project-compounds` and
`experiment-records` additionally need their nested index/store routes
(`GET`/`POST /api/projects/{project}/compounds`,
`GET`/`POST /api/experiments/{experiment}/records`) alongside the shallow
show/update routes. This client has no delete operation, so a `destroy` route
isn't required.

## Development

```bash
pip install -e ".[dev]"
pytest
```
