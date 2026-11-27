"""The grouping-variable candidates every pipeline's `grouping=` kwarg picks
from — not enforced (any real column name works), just what's typically
meaningful for org_dma data.
"""

GROUPING_CANDIDATES = ("sample_group", "sample_subgroup", "matrix_group", "project_id")
