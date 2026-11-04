from unittest.mock import patch

import pytest
from qtpy.QtWidgets import QMessageBox
from useq import GridFromEdges, GridRowsColumns, MDASequence, Position
from useq._point_visiting import OrderMode

from napari_raman_widget.raman_grid_editor import point_region
from raman_mda_engine.grid_plan import GRID_KEY, grid_indices, grid_position, iter_grid_events
from test_unified_mda import owner, qt_app


@pytest.mark.parametrize("rows,columns", [(rows, columns) for rows in range(1, 9) for columns in range(1, 9)])
@pytest.mark.parametrize("order", list(OrderMode))
def test_compact_orders_match_native_mda(rows, columns, order):
    assert [grid_indices(index, rows, columns, order.value) for index in range(rows * columns)] == list(order.generate_indices(rows, columns))


def test_point_bounds_have_no_imaging_fov_offset():
    region = point_region("bounds", dict(left=10, right=110, top=80, bottom=20),
                          (30, 25), (999, 999), "center", "row_wise")
    assert region["x"] == [10, 110, 5]
    assert region["y"] == [80, 20, 4]
    sequence = MDASequence(metadata={GRID_KEY: dict(version=1, regions=[region])})
    assert (grid_position(sequence, 0).x, grid_position(sequence, 0).y) == (10, 80)
    assert (grid_position(sequence, 19).x, grid_position(sequence, 19).y) == (110, 20)


@pytest.mark.parametrize("relative_to,expected", [("center", ([80, 120, 3], [210, 190, 2])),
                                                 ("top_left", ([100, 140, 3], [200, 180, 2]))])
def test_native_row_column_origin(relative_to, expected):
    region = point_region("number", dict(rows=2, columns=3), (20, 20),
                          (100, 200), relative_to, "column_wise_snake")
    assert (region["x"], region["y"]) == expected


def test_native_area_uses_point_extents():
    region = point_region("area", dict(width=100, height=60), (30, 25),
                          (100, 200), "center", "spiral")
    assert region["x"] == [50, 150, 5]
    assert region["y"] == [230, 170, 4]


@pytest.mark.parametrize("owner", [True], indirect=True)
def test_add_native_grids_preserves_positions_selection_and_saved_settings(owner):
    widget, panel, app = owner
    panel.setValue(MDASequence(stage_positions=[Position(x=123, y=456)], channels=["BF"]))
    panel.grid_modes.selector.setCurrentIndex(1)
    table = panel.stage_positions.value()
    native = panel.grid_plan
    editor = panel.grid_modes.editor
    assert not native._mode_number_radio.isHidden()
    assert native.overlap.isHidden()
    assert not native.isAncestorOf(widget.grid_xrange_input)
    assert panel.raman_selection_tab.isAncestorOf(widget._raman_settings_panel)
    editor.x_spacing.setValue(10)
    editor.y_spacing.setValue(20)
    widget.grid_z_input.setText("0")
    native.setValue(GridRowsColumns(rows=2, columns=3, mode="spiral"))
    widget.run_grid_sel_btn.click()
    assert len(widget._grid_regions) == 1
    native.setValue(GridFromEdges(left=0, right=30, top=40, bottom=0, mode="row_wise"))
    widget.run_grid_sel_btn.click()
    assert len(widget._grid_regions) == 2
    assert panel.stage_positions.value() == table
    assert widget.selection_results is None
    saved = panel.value()
    assert len(list(iter_grid_events(saved))) == 18
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert panel.stage_positions.value() == table
    assert native.mode().value == "bounds"
    assert editor.x_spacing.value() == 10
    assert editor.y_spacing.value() == 20
    assert panel.value().metadata[GRID_KEY] == saved.metadata[GRID_KEY]


@pytest.mark.parametrize("owner", [True], indirect=True)
def test_native_grid_z_warning_and_tilt(owner):
    widget, panel, app = owner
    panel.grid_modes.selector.setCurrentIndex(1)
    panel.grid_plan.setValue(GridRowsColumns(rows=2, columns=2))
    widget.grid_z_input.clear()
    with patch("napari_raman_widget.raman_grid_editor.QMessageBox.warning", return_value=QMessageBox.Cancel) as warning:
        widget.run_grid_sel_btn.click()
        warning.assert_called_once()
        assert not widget._grid_regions
    for point in ((0, 0, 1), (100, 0, 2), (0, 100, 3)):
        widget._add_grid_tilt_row(point)
    widget.grid_tilt_check.setChecked(True)
    with patch("napari_raman_widget.raman_grid_editor.QMessageBox.warning") as warning:
        widget.run_grid_sel_btn.click()
        warning.assert_not_called()
    assert widget._grid_regions[0]["sequence"].metadata[GRID_KEY]["regions"][0]["tilt"]["degree"] == 1
    saved = panel.value()
    widget._clear_grid_tilt_rows()
    widget.grid_tilt_check.setChecked(False)
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert widget.grid_tilt_table.rowCount() == 3
    assert widget.grid_tilt_check.isChecked()
