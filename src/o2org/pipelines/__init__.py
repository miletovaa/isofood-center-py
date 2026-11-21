"""Domain-specific analysis pipelines for org_dma's four result families.

Each family module (`isotopes`, `elemental`, `mk_gcms`, `voc_gcms`) exposes
`stats_pipeline(client, project_id, ...)` and `ml_pipeline(client, project_id, ...)`.
The building blocks they're composed from — extraction/pivoting, individual
statistical tests, ML techniques, and plots — are in `utils`, `stats`, `ml`,
and `viz` respectively, and are meant to be called directly too.

    import o2org
    from o2org.pipelines import isotopes

    client = o2org.connect()
    result = isotopes.stats_pipeline(client, project_id=42)
    result["figures"]["box"].show()
"""

from . import elemental, isotopes, ml, mk_gcms, preprocessing, stats, viz, voc_gcms
from .utils import extract

__all__ = [
    "isotopes",
    "elemental",
    "mk_gcms",
    "voc_gcms",
    "preprocessing",
    "stats",
    "ml",
    "viz",
    "extract",
]
