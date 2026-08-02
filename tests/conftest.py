import matplotlib
import pytest

matplotlib.use("Agg")  # headless: tests must never pop up a window

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
