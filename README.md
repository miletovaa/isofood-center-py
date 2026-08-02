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
> As of this writing those endpoints (Sanctum login, full create/update
> routes) are not yet built on the Laravel side — see that section for exactly
> what's needed.

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
