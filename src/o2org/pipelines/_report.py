"""Internal: the self-describing report object every `stats_pipeline()`/
`ml_pipeline()` returns.

It's a plain `dict` underneath — every existing key (`result["tests"]`,
`result["wide"]`, `result["figures"]`, ...) works exactly as before — but the
object also renders as a full report (summary text + p-value tables +
inline figures) when it's the last expression in a Jupyter cell, or via
`print(result)` in a plain terminal. So a single `isotopes.stats_pipeline(...)`
call, with nothing extra, already shows the whole analysis; picking out one
piece via `result["tests"]` still works exactly the same for programmatic use.
"""

from __future__ import annotations

import html as _html
from typing import Any

import plotly.graph_objects as go


class PipelineResult(dict):
    """Dict-like pipeline output with a rich, auto-rendering report.

    Args:
        data: Everything the pipeline computed — becomes the dict contents,
            unchanged (so `result["tests"]`, `result["wide"]`, etc. keep working).
        title: Report heading, e.g. "isotopes.stats_pipeline — project 12".
        summary_lines: Plain-English lines (sample counts, per-feature
            significance, notes on anything skipped) shown at the top of
            the report, in order.
        figures: Name -> `go.Figure` to embed in the report, in display
            order. Only figures that were actually built (not `None`) should
            be included.
    """

    def __init__(
        self, data: dict[str, Any], *, title: str, summary_lines: list[str], figures: dict[str, go.Figure] | None = None,
    ) -> None:
        super().__init__(data)
        self._title = title
        self._summary_lines = summary_lines
        self._figures = {name: fig for name, fig in (figures or {}).items() if fig is not None}

    def _repr_html_(self) -> str:
        parts = [f"<h3>{_html.escape(self._title)}</h3>", "<ul>"]
        parts += [f"<li>{_html.escape(line)}</li>" for line in self._summary_lines]
        parts.append("</ul>")

        for key in ("tests", "normality"):
            df = self.get(key)
            if df is not None and hasattr(df, "empty") and not df.empty:
                parts.append(f"<p><b>{_html.escape(key)}</b></p>")
                parts.append(df.to_html(index=False))

        for i, (name, fig) in enumerate(self._figures.items()):
            parts.append(f"<p><b>{_html.escape(name)}</b></p>")
            parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))

        return "\n".join(parts)

    def __repr__(self) -> str:
        lines = [self._title, "=" * len(self._title)]
        lines += [f"- {line}" for line in self._summary_lines]
        if self._figures:
            lines.append(f"({len(self._figures)} figure(s): {', '.join(self._figures)} — view in Jupyter for the full report)")
        return "\n".join(lines)


def describe_group_tests(
    tests_df, wide_df, grouping: str, alpha: float, feature_label: str = "feature", label_map: dict[Any, str] | None = None,
) -> list[str]:
    """Plain-English summary lines for a `stats.test_all_features()` result:
    sample counts per group, how many features came out significant, and a
    verdict line per feature. Shared across every family's `stats_pipeline`.

    Args:
        label_map: Optional feature-id -> display-name mapping (e.g. VOC
            compound_id -> canonical_name) used only for the printed lines —
            the underlying `tests_df` keeps the raw ids.
    """
    label_map = label_map or {}
    counts = wide_df[grouping].value_counts()
    lines = [f"{len(wide_df)} samples: " + ", ".join(f"{group} ({n})" for group, n in counts.items())]

    if tests_df.empty:
        lines.append(f"Too few samples/groups for a {feature_label} comparison — need at least 2 groups with data.")
        return lines

    n_sig = int(tests_df["significant"].sum())
    lines.append(f"{n_sig} of {len(tests_df)} {feature_label}s show a significant difference between groups (α={alpha}, corrected).")
    for _, row in tests_df.iterrows():
        verdict = "significant" if row["significant"] else "not significant"
        name = label_map.get(row["feature"], row["feature"])
        lines.append(f"{name}: {verdict} ({row['test']}, p={row['pvalue']:.2e} → corrected p={row['pvalue_corrected']:.2e})")
    return lines
