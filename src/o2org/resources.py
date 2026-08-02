"""One class per o2_org_dma entity, over standard Laravel apiResource routes:

    GET    /api/{endpoint}            index   (list, optionally paginated)
    GET    /api/{endpoint}/{id}       show
    POST   /api/{endpoint}            store   (create — see "other procedures" below)
    PUT    /api/{endpoint}/{id}       update  (see "other procedures" below)

`list()`/`get()` — retrieving data — are the methods you'll use for almost
everything; `create()`/`update()`/`bulk_create()` exist for the less common
case of writing data back and are grouped at the bottom of each class on
purpose. This client does not support deleting records.

`list()` accepts arbitrary keyword args as query-string filters (e.g.
client.compounds.list(name="glucose")) — but which keys the server actually
recognizes, and whether they're exact or partial matches, is up to however
that server implements filtering (e.g. org_dma's Compound model uses the
`mehradsadeghi/laravel-filter-querystring` package, where exact match is
`canonical_name=Glucose` and partial/substring match needs the package's own
`like=canonical_name,glucose` syntax — a plain `name=glucose` is silently
ignored, not an error). An unrecognized filter key typically doesn't error;
it's just dropped, so a "filtered" list() that comes back the same size as
the whole table usually means the filter key didn't match anything the
server recognizes, not that there's no matching data. Check your server's
own filter contract rather than assuming a key works.

`list()` is lazy: calling it never touches the network by itself. It returns
a ListQuery, and what happens next depends on what you do with it:

    client.compounds.list(like="canonical_name,acid").count()   # one cheap request
    client.compounds.list(like="canonical_name,acid").pages     # one cheap request
    client.compounds.list(like="canonical_name,acid").get()     # fetches everything
    for c in client.compounds.list(like="canonical_name,acid"): ...   # fetches everything

`.count()`/`.pages` read the server's own pagination metadata from a single
request — they never walk every page just to hand you a number, which matters
a lot on a filter that matches thousands of rows across a huge table.
Anything that actually needs the records (`.get()`, iterating, indexing,
`==`, `repr()`) fetches and caches the full result the same way a
materialized call always did — every page is walked and combined whenever
that's safe to assume: a filtered search on a top-level resource (a filter
narrows the result, so "give me every match" is the reasonable default) and
any nested-resource list (already scoped to one parent). An unfiltered
top-level list() only fetches the first page by default instead, since "give
me the whole table" could mean tens of thousands of rows. Pass
all_pages=True/False, or a specific page=N, to override either default
explicitly.

The fully-fetched result is a RecordList — a plain list of dicts with
`.count()` and `.names()` added (e.g. client.compounds.list().get().names()).

A few entities (project compounds, experiment records) are nested under a
parent for listing/creation but addressed directly by id for show/update
("shallow nesting"), matching the one nested route already in org_dma's
routes/api.php (projects/{project}/compounds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
from typing import List as _List  # avoids clashing with the `list()` method name below

import pandas as pd

if TYPE_CHECKING:
    from .client import OrgClient

Record = dict[str, Any]

#: Field names tried, in order, when a record's "name" is asked for — covers
#: every entity in the schema without needing a per-resource mapping (compounds
#: use canonical_name, project_compounds use custom_name, most others use name).
_NAME_FIELDS: tuple[str, ...] = ("name", "canonical_name", "custom_name", "title")


def _pick_name(record: Record) -> Any:
    for field in _NAME_FIELDS:
        if field in record and record[field] is not None:
            return record[field]
    raise KeyError(
        f"No name-like field found on this record (tried {_NAME_FIELDS!r}): {record!r}"
    )


class RecordList(list):
    """What a fully-fetched `.list()` query holds: a `list[dict]` of records,
    plus a couple of read-only conveniences. Behaves exactly like a plain
    list otherwise (indexing, iteration, `len()`, equality with a plain
    list, ...).

    Attributes:
        pages: Total number of pages available from the server, if the
            response was paginated; None if it wasn't (a bare array, or an
            API Resource collection with no pagination meta).
    """

    pages: int | None = None

    def count(self, *value: Any) -> int:
        """Number of records (`client.projects.list().get().count()`).

        Matches `list.count(value)` if `value` is given — the number of times
        it appears in the list — so this stays a drop-in `list` otherwise.
        """
        if not value:
            return len(self)
        return super().count(*value)

    def names(self) -> list[Any]:
        """The name of each record: whichever of `name` / `canonical_name` /
        `custom_name` / `title` is present on it (checked in that order).

        Returns:
            One value per record, in the same order as the list.

        Raises:
            KeyError: A record has none of the recognized name-like fields.
        """
        return [_pick_name(record) for record in self]


def _unwrap_list(data: Any) -> tuple[list[Record], Record | None]:
    """Return (items, meta) from any of the three shapes described above."""
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict) and "data" in data:
        return data["data"], data.get("meta")
    return ([data] if data else []), None


def _unwrap_item(data: Any) -> Any:
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        return data["data"]
    return data


class ListQuery:
    """The lazy result of a `.list()` call — see the module docstring above.

    Nothing is fetched at construction time. `.count()`/`.pages` make one
    cheap request and read the server's pagination metadata; everything else
    that actually needs the records (`.get()`, iterating, indexing, `==`,
    `repr()`) fetches and caches the full `RecordList` on first use.
    """

    def __init__(self, fetch_page: Callable[[int | None], tuple[list[Record], Record | None]],
                 all_pages: bool) -> None:
        self._fetch_page = fetch_page
        self._all_pages = all_pages
        self._materialized: RecordList | None = None
        self._cheap: tuple[list[Record], Record | None] | None = None

    def get(self) -> RecordList:
        """Fetch every record this query matches (walking every page if
        `all_pages` is in effect) and return it as a `RecordList`. Cached —
        calling this again returns the same result without refetching.
        """
        if self._materialized is None:
            items, meta = self._cheap if self._cheap is not None else self._fetch_page(None)
            record_list = RecordList(items)
            record_list.pages = meta.get("last_page") if meta else None
            while self._all_pages and meta and meta.get("current_page", 1) < meta.get("last_page", 1):
                next_page = meta["current_page"] + 1
                more, meta = self._fetch_page(next_page)
                record_list.extend(more)
                record_list.pages = meta.get("last_page") if meta else record_list.pages
            self._materialized = record_list
        return self._materialized

    def count(self) -> int:
        """Total number of matching records — one cheap request, no matter
        how many pages that total spans (reads the server's own pagination
        metadata instead of fetching and counting every page).
        """
        if self._materialized is not None:
            return len(self._materialized)
        items, meta = self._cheap_fetch()
        if meta and meta.get("total") is not None:
            return int(meta["total"])
        return len(items)  # no pagination meta - this one response IS everything

    @property
    def pages(self) -> int | None:
        """Total number of pages available, or None if the response wasn't
        paginated. Same one-request cost as `.count()`.
        """
        if self._materialized is not None:
            return self._materialized.pages
        _, meta = self._cheap_fetch()
        return meta.get("last_page") if meta else None

    def names(self) -> list[Any]:
        """The name of each matching record — see `RecordList.names()`.
        Needs the actual records, so this fetches everything like `.get()` does.
        """
        return self.get().names()

    def _cheap_fetch(self) -> tuple[list[Record], Record | None]:
        if self._cheap is None:
            self._cheap = self._fetch_page(None)
        return self._cheap

    def __iter__(self):
        return iter(self.get())

    def __len__(self) -> int:
        return len(self.get())

    def __getitem__(self, key):
        return self.get()[key]

    def __eq__(self, other: object) -> bool:
        return self.get() == other

    def __repr__(self) -> str:
        return repr(self.get())


class Resource:
    """Generic CRUD wrapper over one REST endpoint. Attached to a client as
    e.g. `client.compounds` — see `attach_resources()` below for the full list.
    """

    endpoint: str | None = None

    def __init__(self, client: OrgClient, endpoint: str | None = None) -> None:
        """Args:
            client: The connected `OrgClient` to send requests through.
            endpoint: URL segment for this resource, e.g. "compounds". Falls
                back to the class attribute `endpoint` if omitted (used by
                subclasses that hardcode it, like `ProjectCompounds`).
        """
        self.client = client
        self.endpoint = endpoint or self.endpoint

    def _path(self, id: Any = None) -> str:
        return f"/api/{self.endpoint}/{id}" if id is not None else f"/api/{self.endpoint}"

    # -- retrieving data (the common case) -----------------------------------

    def list(self, all_pages: bool | None = None, page: int | None = None, **params: Any) -> ListQuery:
        """List/search records. Lazy — see the module docstring for what
        triggers a request and what stays cheap.

        Args:
            all_pages: Whether `.get()`/iterating walks every page of a
                paginated response and returns the combined result. Defaults
                to True when you pass a filter (e.g. `name="glucose"`) and no
                explicit `page` — a filtered search should give you every
                match, not just page 1 of them. Defaults to False for an
                unfiltered call (fetching every row of a huge table by
                accident would be a footgun) and whenever `page` is given.
                Pass it explicitly to override either default. Never affects
                `.count()`/`.pages`, which always report the query's true
                total regardless of this flag.
            page: Which page `.get()`/iterating fetches (only meaningful when
                `all_pages` ends up False).
            **params: Arbitrary query-string filters, e.g. `name="glucose"`.

        Returns:
            A `ListQuery`. Use `.count()`/`.pages` for a cheap single-request
            answer, or `.get()` (or just iterate/index it) to fetch every
            matching record.
        """
        if all_pages is None:
            all_pages = page is None and bool(params)

        def fetch_page(requested_page: int | None) -> tuple[list[Record], Record | None]:
            page_params = {**params, "page": requested_page} if requested_page is not None else (params or None)
            return _unwrap_list(self.client.get(self._path(), params=page_params))

        query = ListQuery(fetch_page, all_pages)
        if page is not None:
            query._cheap = fetch_page(page)  # so .get() starts from the requested page, not page 1
        return query

    def get(self, id: Any) -> Record:
        """Fetch a single record by id.

        Args:
            id: The record's id.

        Returns:
            The record as a dict.

        Raises:
            NotFoundError: No record exists with that id.
        """
        return _unwrap_item(self.client.get(self._path(id)))

    # -- other procedures: writing data ---------------------------------------

    def create(self, **fields: Any) -> Record:
        """Create a new record.

        Args:
            **fields: Field values for the new record, e.g. `canonical_name="Glucose"`.

        Returns:
            The created record, as returned by the server.

        Raises:
            ValidationError: The server rejected the payload (HTTP 422).
        """
        return _unwrap_item(self.client.post(self._path(), json=fields))

    def update(self, id: Any, **fields: Any) -> Record:
        """Update an existing record.

        Args:
            id: The record's id.
            **fields: Field values to update, e.g. `molecular_formula="C6H12O6"`.

        Returns:
            The updated record, as returned by the server.

        Raises:
            NotFoundError: No record exists with that id.
            ValidationError: The server rejected the payload (HTTP 422).
        """
        return _unwrap_item(self.client.put(self._path(id), json=fields))

    def bulk_create(self, records: _List[Record] | pd.DataFrame) -> dict[str, Any]:
        """Create many records, one request per row.

        Never raises on a single bad row — reports which ones failed instead
        of aborting midway, which is what you want for a spreadsheet import.

        Args:
            records: A list of dicts, or a DataFrame (one row per record).

        Returns:
            `{"created": [...], "errors": [(row, error), ...]}` — `created`
            holds the successfully created records; `errors` pairs each failed
            row with the `APIError` it raised.
        """
        from .exceptions import APIError

        rows = records.to_dict(orient="records") if isinstance(records, pd.DataFrame) else records

        created, errors = [], []
        for row in rows:
            try:
                created.append(self.create(**row))
            except APIError as exc:
                errors.append((row, exc))
        return {"created": created, "errors": errors}


class _NestedResource(Resource):
    """Base for entities whose index/store route is nested under a parent, but
    whose show/update route is the shallow top-level endpoint.
    """

    parent_endpoint: str | None = None
    nested_path_name: str | None = None

    def _nested_path(self, parent_id: Any) -> str:
        return f"/api/{self.parent_endpoint}/{parent_id}/{self.nested_path_name}"

    # -- retrieving data (the common case) -----------------------------------

    def list(self, parent_id: Any, all_pages: bool | None = None, page: int | None = None, **params: Any) -> ListQuery:
        """List/search records belonging to one parent. Lazy — see the module
        docstring for what triggers a request and what stays cheap.

        Args:
            parent_id: The parent record's id (e.g. a project id).
            all_pages: Whether `.get()`/iterating walks every page. Defaults
                to True (this is already scoped to one parent, so
                "everything" is bounded) unless `page` is given, which
                defaults it to False. Pass it explicitly to override either
                default. Never affects `.count()`/`.pages`.
            page: Which page `.get()`/iterating fetches.
            **params: Arbitrary query-string filters.

        Returns:
            A `ListQuery` — see `Resource.list`.
        """
        if all_pages is None:
            all_pages = page is None

        def fetch_page(requested_page: int | None) -> tuple[list[Record], Record | None]:
            page_params = {**params, "page": requested_page} if requested_page is not None else (params or None)
            return _unwrap_list(self.client.get(self._nested_path(parent_id), params=page_params))

        query = ListQuery(fetch_page, all_pages)
        if page is not None:
            query._cheap = fetch_page(page)
        return query

    # -- other procedures: writing data ---------------------------------------

    def create(self, parent_id: Any, **fields: Any) -> Record:
        """Create a new record under a parent.

        Args:
            parent_id: The parent record's id (e.g. a project id).
            **fields: Field values for the new record.

        Returns:
            The created record, as returned by the server.
        """
        return _unwrap_item(self.client.post(self._nested_path(parent_id), json=fields))

    def bulk_create(self, parent_id: Any, records: _List[Record] | pd.DataFrame) -> dict[str, Any]:
        """Create many records under a parent, one request per row.

        Args:
            parent_id: The parent record's id (e.g. a project id).
            records: A list of dicts, or a DataFrame (one row per record).

        Returns:
            `{"created": [...], "errors": [(row, error), ...]}`, as in `Resource.bulk_create`.
        """
        from .exceptions import APIError

        rows = records.to_dict(orient="records") if isinstance(records, pd.DataFrame) else records

        created, errors = [], []
        for row in rows:
            try:
                created.append(self.create(parent_id, **row))
            except APIError as exc:
                errors.append((row, exc))
        return {"created": created, "errors": errors}


class ProjectCompounds(_NestedResource):
    """`client.project_compounds` — nested under a project for list/create,
    addressed directly by id for get/update.
    """

    endpoint = "project-compounds"       # used for get/update
    parent_endpoint = "projects"
    nested_path_name = "compounds"

    def list(self, project_id: Any, all_pages: bool | None = None, page: int | None = None, **params: Any) -> ListQuery:
        """List/search the compounds attached to one project. Lazy — see
        `_NestedResource.list`.

        Args:
            project_id: The project's id.
            all_pages: Whether `.get()`/iterating walks every page. Defaults to True unless `page` is given.
            page: Which page `.get()`/iterating fetches.
            **params: Arbitrary query-string filters.
        """
        return super().list(project_id, all_pages=all_pages, page=page, **params)

    def create(self, project_id: Any, **fields: Any) -> Record:
        """Attach a new compound record to a project.

        Args:
            project_id: The project's id.
            **fields: Field values for the new record, e.g. `mz=180.06, rt=4.2`.
        """
        return super().create(project_id, **fields)

    def bulk_create(self, project_id: Any, records: _List[Record] | pd.DataFrame) -> dict[str, Any]:
        """Attach many compound records to a project, one request per row.

        Args:
            project_id: The project's id.
            records: A list of dicts, or a DataFrame (one row per record).
        """
        return super().bulk_create(project_id, records)


class ExperimentRecords(_NestedResource):
    """`client.experiment_records` — nested under an experiment for
    list/create, addressed directly by id for get/update.
    """

    endpoint = "experiment-records"      # used for get/update
    parent_endpoint = "experiments"
    nested_path_name = "records"

    def list(self, experiment_id: Any, all_pages: bool | None = None, page: int | None = None, **params: Any) -> ListQuery:
        """List/search the records attached to one experiment. Lazy — see
        `_NestedResource.list`.

        Args:
            experiment_id: The experiment's id.
            all_pages: Whether `.get()`/iterating walks every page. Defaults to True unless `page` is given.
            page: Which page `.get()`/iterating fetches.
            **params: Arbitrary query-string filters.
        """
        return super().list(experiment_id, all_pages=all_pages, page=page, **params)

    def create(self, experiment_id: Any, **fields: Any) -> Record:
        """Attach a new record to an experiment.

        Args:
            experiment_id: The experiment's id.
            **fields: Field values for the new record.
        """
        return super().create(experiment_id, **fields)

    def bulk_create(self, experiment_id: Any, records: _List[Record] | pd.DataFrame) -> dict[str, Any]:
        """Attach many records to an experiment, one request per row.

        Args:
            experiment_id: The experiment's id.
            records: A list of dicts, or a DataFrame (one row per record).
        """
        return super().bulk_create(experiment_id, records)


# endpoint (URL segment) for every remaining entity in the current schema
_SIMPLE_RESOURCES = {
    "compounds": "compounds",
    "sources": "sources",
    "projects": "projects",
    "compound_synonyms": "compound-synonyms",
    "retention_indices": "retention-indices",
    "diseases": "diseases",
    "compound_disease_associations": "compound-disease-associations",
    "ontologies": "ontologies",
    "compound_ontologies": "compound-ontologies",
    "biomarkers": "biomarkers",
    "compound_biomarkers": "compound-biomarkers",
    "taxonomies": "taxonomies",
    "samples": "samples",
    "samplings": "samplings",
    "experiments": "experiments",
}


def attach_resources(client: OrgClient) -> None:
    """Attach one Resource instance per entity as an attribute on the client,
    e.g. client.compounds, client.projects, client.project_compounds.
    """
    for attr, endpoint in _SIMPLE_RESOURCES.items():
        setattr(client, attr, Resource(client, endpoint))
    client.project_compounds = ProjectCompounds(client)
    client.experiment_records = ExperimentRecords(client)
