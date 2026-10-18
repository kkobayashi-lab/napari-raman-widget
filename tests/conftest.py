"""Keep one QApplication alive across all widget tests."""
import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def isolated_mda_timing_settings(tmp_path, monkeypatch):
    """Tests must neither read nor overwrite the microscope's learned timing."""
    from qtpy.QtCore import QSettings
    from napari_raman_widget import mda_timing

    path = str(tmp_path / "timing.ini")
    monkeypatch.setattr(
        mda_timing, "_timing_settings",
        lambda: QSettings(path, QSettings.IniFormat),
    )


@pytest.fixture(scope="session", autouse=True)
def session_qapplication():
    # Creating/destroying multiple applications invalidates Qt-backed core signals.
    app = QApplication.instance() or QApplication([])
    yield app
