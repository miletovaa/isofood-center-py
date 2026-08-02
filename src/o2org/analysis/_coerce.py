"""Internal: every public function in o2org.analysis accepts data exactly as
it comes back from a Resource — a list of dicts from `.list()`, a single dict
from `.get()` — as well as a plain DataFrame or a Dataset. Calling this at the
top of each function means nobody has to remember to convert first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import pandas as pd

if TYPE_CHECKING:
    from ..resources import ListQuery
    from .dataset import Dataset

#: What every public analysis function accepts as its "data" argument: a
#: DataFrame, a Dataset, a single record (dict), a list of records, or a
#: still-lazy ListQuery straight from `.list()` — i.e. exactly what a
#: Resource's `.get()`/`.list()` hands back, fetched or not.
AnalysisInput = Union["pd.DataFrame", "Dataset", "ListQuery", dict[str, Any], list[dict[str, Any]]]


def _coerce_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Laravel (and many other JSON APIs) serialize decimal-cast columns as
    strings (e.g. "180.1600000000" for what's really a float) so precision
    survives the trip through JSON. Left alone, that means numeric_columns()
    — and everything built on it: correlation, PCA, clustering, regression —
    silently ignores those columns as "text". Convert any non-numeric column
    back to a real numeric dtype when *every* non-null value in it parses as
    a number, so numeric analysis picks it up the way you'd expect.

    Checks "not already numeric/bool" rather than "dtype is object", since
    pandas 3's default string dtype ("str"/StringDtype) isn't `object` either.
    """
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if series.notna().sum() == 0:
            continue  # all-missing - vacuously "numeric", but that's not useful to anyone
        if series.map(lambda v: isinstance(v, (dict, list))).any():
            continue  # nested structures (e.g. a "compound" relation) - leave as-is
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.isna().sum() == series.isna().sum():
            df[col] = coerced
    return df


def as_dataframe(data: AnalysisInput) -> pd.DataFrame:
    """Coerce `data` (see `AnalysisInput`) into a plain DataFrame. A lazy
    `ListQuery` (an un-materialized `.list()` result) is fetched here — this
    is the point where "I want the actual records" is unambiguous.
    """
    from ..resources import ListQuery  # deferred: avoids importing resources at package init
    from .dataset import Dataset  # deferred: dataset.py imports this module too

    if isinstance(data, Dataset):
        return data.df
    if isinstance(data, ListQuery):
        data = data.get()
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return _coerce_numeric_strings(pd.DataFrame([data]))
    return _coerce_numeric_strings(pd.DataFrame(data))
