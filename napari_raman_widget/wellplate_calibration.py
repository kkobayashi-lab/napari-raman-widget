"""Preserve measured plate-to-stage geometry, including reversed stage axes."""
from types import MethodType

import numpy as np
from pydantic import field_validator
from useq import WellPlatePlan
from .plate_focus import PlateFocusCalibration


class FocusWellPlatePlan(WellPlatePlan):
    focus_height: PlateFocusCalibration | None = None


def plate_plan_model(data):
    return (CalibratedWellPlatePlan if "stage_affine_um" in data else
            FocusWellPlatePlan if "focus_height" in data else WellPlatePlan)


class CalibratedWellPlatePlan(FocusWellPlatePlan):
    # Maps (row, column, 1) to stage (X, Y), in micrometers.
    stage_affine_um: tuple[tuple[float, float, float], tuple[float, float, float]]

    @field_validator("stage_affine_um")
    @classmethod
    def valid_transform(cls, value):
        matrix = np.asarray(value)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-6:
            raise ValueError("Plate calibration has invalid or degenerate stage axes")
        return value

    @property
    def affine_transform(self):
        # useq internally orders coordinates (Y, X), in millimeters.
        matrix = np.eye(3)
        matrix[:2] = np.asarray(self.stage_affine_um)[::-1] / 1000
        return matrix


def fit_plate_transform(wells):
    """Fit the measured centers directly, without discarding axis signs or shear."""
    indices = np.asarray([[row, col, 1] for row, col in wells], dtype=float)
    centers = np.asarray(list(wells.values()), dtype=float)
    if len(wells) < 3 or not np.isfinite(centers).all():
        raise ValueError("Calibrate at least three non-collinear wells")
    coefficients, _, rank, _ = np.linalg.lstsq(indices, centers, rcond=None)
    if rank != 3:
        raise ValueError("Calibrate at least three non-collinear wells")
    matrix = coefficients.T
    if abs(np.linalg.det(matrix[:, :2])) < 1e-6:
        raise ValueError("Plate calibration has degenerate stage axes")
    return tuple(tuple(float(v) for v in row) for row in matrix)


def extend_plate_mapping(wizard):
    """Repair the native calibration preview and keep its full fit in the plan."""
    from qtpy.QtWidgets import QFileDialog

    calibration = wizard.calibration_page.widget
    native_value = calibration.value
    native_set_value = calibration.setValue
    native_update_info = calibration._update_info
    saved_matrix = None

    install_plate_navigation_view(calibration._plate_test, calibration._calibrated_wells)

    # Imported calibrated plans can also reach the selection view. Always draw
    # its logical layout, independently of the saved stage transform.
    selection_view = wizard.plate_page.widget._view
    native_selection_draw = selection_view.drawPlate

    def draw_selection(_self, plan):
        if isinstance(plan, WellPlatePlan):
            plan = WellPlatePlan(plate=plan.plate, a1_center_xy=(0, 0),
                                 selected_wells=plan.selected_wells)
        native_selection_draw(plan)

    selection_view.drawPlate = MethodType(draw_selection, selection_view)

    def value(_self):
        plan = native_value()
        if plan is None:
            return None
        matrix = saved_matrix
        if len(calibration._calibrated_wells) >= 3:
            matrix = fit_plate_transform(calibration._calibrated_wells)
        if matrix is None:
            return plan
        return CalibratedWellPlatePlan(
            **plan.model_dump(), stage_affine_um=matrix,
        ).replace(a1_center_xy=(matrix[0][2], matrix[1][2]))

    def update_info(_self):
        native_update_info()
        try:
            plan = value(calibration)
        except ValueError as error:
            calibration._info.setText(str(error))
            return
        if isinstance(plan, CalibratedWellPlatePlan):
            matrix = np.asarray(plan.stage_affine_um)
            row, column = matrix[:, 0], matrix[:, 1]
            calibration._info.setText(
                "<strong>Plate calibrated with measured stage directions.</strong><br>"
                f"A1 center [mm]: ({matrix[0, 2] / 1000:.3f}, {matrix[1, 2] / 1000:.3f})<br>"
                f"Column step (ΔX, ΔY) [mm]: ({column[0] / 1000:.3f}, {column[1] / 1000:.3f}); "
                f"row step [mm]: ({row[0] / 1000:.3f}, {row[1] / 1000:.3f})"
            )

    def set_value(_self, plan):
        nonlocal saved_matrix
        saved_matrix = getattr(plan, "stage_affine_um", None)
        # Native setValue leaves the previous plate's measured wells behind.
        calibration._calibrated_wells.clear()
        native_set_value(plan)

    calibration.value = MethodType(value, calibration)
    calibration.setValue = MethodType(set_value, calibration)
    calibration._update_info = MethodType(update_info, calibration)

    def redraw(calibrated):
        if calibrated:
            calibration._plate_test.drawPlate(calibration.value())
        else:
            calibration._plate_test.clear()

    calibration.calibrationChanged.connect(redraw)

    def load(_self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(wizard, "Load Well Plate Plan", "", "JSON (*.json)")
        if path:
            import json
            from pathlib import Path
            data = json.loads(Path(path).read_text())
            model = plate_plan_model(data)
            wizard.setValue(model.model_validate(data))

    # The load button already holds the old bound method; reconnect it as well.
    wizard.load = MethodType(load, wizard)
    button = wizard.button(wizard.WizardButton.CustomButton2)
    button.clicked.disconnect()
    button.clicked.connect(wizard.load)


def install_plate_navigation_view(view, calibrated_wells=()):
    """Display logical wells with calibrated stage coordinates for click targets."""
    # Draw in logical plate coordinates. Stage coordinates are stored only in
    # the items' click data, so reversed stage axes never flip the displayed board.
    native_draw = view.drawPlate
    native_markers = view._add_preset_positions_items
    displayed_plan = None
    view.setDrawLabels(False)
    view.setDrawWellEdgeSpots(True)

    def draw_plate(_self, plan):
        nonlocal displayed_plan
        from pymmcore_widgets.useq_widgets._well_plate_widget import DATA_POSITION
        from qtpy.QtCore import Qt

        if not isinstance(plan, WellPlatePlan):
            plan = WellPlatePlan(plate=plan, a1_center_xy=(0, 0))
        displayed_plan = plan
        nominal = WellPlatePlan(plate=plan.plate, a1_center_xy=(0, 0),
                                selected_wells=plan.selected_wells)
        native_draw(nominal)
        for position in plan.all_well_positions:
            item = view._well_items[(position.plate_row, position.plate_col)]
            item.setData(DATA_POSITION, position)
            item.setToolTip(f"{position.name} stage XY [µm]: ({position.x:.0f}, {position.y:.0f})")
        for index in calibrated_wells:
            view.setWellColor(*index, Qt.green)

    view.drawPlate = MethodType(draw_plate, view)

    def add_markers(_self, rect, position, nominal):
        from pymmcore_widgets.useq_widgets._well_plate_widget import DATA_POSITION

        start = len(view._well_edge_spots)
        native_markers(rect, position, nominal)
        center = displayed_plan.all_well_positions[
            position.plate_row * nominal.plate.columns + position.plate_col]
        # Convert logical edge offsets into measured stage directions. Normalize
        # each direction so the manufacturer radius remains the physical radius.
        directions = displayed_plan.affine_transform[:2, :2][::-1]
        directions = directions / np.linalg.norm(directions, axis=0)
        screen_center = rect.center()
        for item in view._well_edge_spots[start:]:
            point = item.rect().center()
            delta = directions @ np.array([point.y() - screen_center.y(), point.x() - screen_center.x()])
            stage_position = center.replace(x=center.x + delta[0], y=center.y + delta[1])
            item.setData(DATA_POSITION, stage_position)
            item.setToolTip(f"{stage_position.name} stage XY [µm]: "
                            f"({stage_position.x:.0f}, {stage_position.y:.0f})")

    view._add_preset_positions_items = MethodType(add_markers, view)

    def double_click(_self, event):
        from pymmcore_widgets.useq_widgets._well_plate_widget import DATA_POSITION
        from qtpy.QtWidgets import QGraphicsView

        if event is not None:
            # Labels sit above the well outline; skip items with no click data.
            for item in view.items(event.pos()):
                position = item.data(DATA_POSITION)
                if position is not None:
                    view.positionDoubleClicked.emit(position)
                    break
        QGraphicsView.mouseDoubleClickEvent(view, event)

    view.mouseDoubleClickEvent = MethodType(double_click, view)
