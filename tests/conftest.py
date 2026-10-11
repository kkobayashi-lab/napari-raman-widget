"""Keep one QApplication alive across all widget tests."""
import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def session_qapplication():
    # Creating/destroying multiple applications invalidates Qt-backed core signals.
    app = QApplication.instance() or QApplication([])
    yield app
