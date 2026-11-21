"""Smoke-test the Streamlit app end to end on the bundled example dataset."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "o2org" / "ui" / "app.py")


def _run(**setup):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    for fn in setup.values():
        fn(at)
        at.run()
    return at


def test_app_runs_on_example_dataset_without_errors():
    at = _run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == "o2org analysis"


def test_app_with_fresh_filter_and_all_designs():
    def fresh_only(at):
        at.sidebar.selectbox[0].set_value("storage")

    at = _run(filter=fresh_only)
    assert not at.exception, at.exception
    for design in ["Two grouping factors", "Independent groups"]:
        at.radio(key="design").set_value(design).run()
        assert not at.exception, (design, at.exception)


def test_paired_design_and_model_comparison():
    at = _run()
    at.radio(key="design").set_value("Same samples measured repeatedly").run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Run comparison").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
