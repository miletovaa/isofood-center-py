"""Figure export: every pipeline plot is a Plotly `go.Figure` — save it as
both a static PNG (via kaleido) and an interactive HTML file.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def save_figure(fig: go.Figure, path: str | Path) -> Path:
    """Write `fig` to `<path>.png` and `<path>.html` (extension on `path` is
    ignored/replaced).

    Args:
        fig: The figure to save.
        path: Destination path; parent directories are created if missing.

    Returns:
        The path stem the two files were written under.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    fig.write_html(stem.with_suffix(".html"))
    fig.write_image(stem.with_suffix(".png"))
    return stem
