"""Point-and-click interface for o2org's statistics & ML toolbox.

    pip install -e ".[ui]"
    o2org-ui                      # or: python -m o2org.ui

Everything the UI does is a call into `o2org.pipelines` (stats, ml, viz,
preprocessing), and each tab shows the equivalent Python code, so an
analysis started here can be continued in a notebook.
"""

from .example_data import example_dataset

__all__ = ["example_dataset", "main"]


def main() -> None:
    """Console entry point: `streamlit run` the bundled app."""
    import sys
    from pathlib import Path

    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('The UI needs Streamlit: pip install "o2org[ui]"') from exc

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
    sys.exit(stcli.main())
