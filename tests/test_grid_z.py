"""Grid focus capture without microscope hardware."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from qtpy.QtWidgets import QApplication, QLineEdit, QMessageBox, QWidget

from napari_raman_widget.widget import HardwareWidget


@pytest.mark.parametrize("corner", ["top_left", "bottom_right"])
def test_corner_capture_records_xy_and_grid_z(corner):
    app = QApplication.instance() or QApplication([])
    widget = HardwareWidget.__new__(HardwareWidget)
    QWidget.__init__(widget)
    widget.core = Mock()
    widget.core.getXYPosition.return_value = (12.5, -3.25)
    widget.core.getPosition.return_value = 45.1234
    widget.status = Mock()
    for name in (
        "grid_tl_x_input", "grid_tl_y_input", "grid_br_x_input",
        "grid_br_y_input", "grid_z_input",
    ):
        setattr(widget, name, QLineEdit(widget))
    try:
        widget._capture_grid_corner(corner)
        prefix = "grid_tl" if corner == "top_left" else "grid_br"
        assert getattr(widget, f"{prefix}_x_input").text() == "12.500"
        assert getattr(widget, f"{prefix}_y_input").text() == "-3.250"
        assert widget.grid_z_input.text() == "45.1234"
        widget.core.getPosition.return_value = -8.25
        widget._capture_grid_z()
        assert widget.grid_z_input.text() == "-8.2500"
    finally:
        widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("z_text,tilt,response,continues,warns", [
    ("", False, QMessageBox.Cancel, False, True),
    ("  ", False, QMessageBox.Ok, True, True),
    ("0", False, QMessageBox.Cancel, True, False),
    ("12.5", False, QMessageBox.Cancel, True, False),
    ("", True, QMessageBox.Cancel, True, False),
])
def test_add_grid_warns_only_without_declared_z(z_text, tilt, response, continues, warns):
    widget = HardwareWidget(SimpleNamespace(window=SimpleNamespace()))
    widget.core = Mock()
    widget.core.getImageWidth.return_value = 512
    widget.core.getImageHeight.return_value = 512
    widget.default_engine = Mock()
    widget.grid_z_input.setText(z_text)
    if tilt:
        for point in ((0, 0, 1), (10, 0, 2), (0, 10, 3)):
            widget._add_grid_tilt_row(point)
        widget.grid_tilt_check.setChecked(True)
    try:
        with patch("napari_raman_widget.widget.QMessageBox.warning", return_value=response) as warning, \
             patch("napari_raman_widget.widget.LogWindow", side_effect=RuntimeError("generation reached")) as log:
            if continues:
                with pytest.raises(RuntimeError, match="generation reached"):
                    widget.run_grid_selection()
            else:
                widget.run_grid_selection()
                log.assert_not_called()
            assert warning.call_count == int(warns)
            if warns:
                assert warning.call_args.args[-1] == QMessageBox.Cancel
            assert widget._grid_regions == []
            widget.core.register_mda_engine.assert_not_called()
    finally:
        widget.core = None
        widget.deleteLater()
        QApplication.instance().processEvents()
