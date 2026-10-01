"""Grid focus capture without microscope hardware."""

from unittest.mock import Mock

import pytest
from qtpy.QtWidgets import QApplication, QLineEdit, QWidget

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
