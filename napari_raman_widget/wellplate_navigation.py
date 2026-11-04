"""Manual XY navigation using a saved, calibrated well-plate plan."""
import json
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget
from pymmcore_widgets.useq_widgets._well_plate_widget import WellPlateView

from .wellplate_calibration import plate_plan_model, install_plate_navigation_view


class WellPlateNavigationPanel(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.plan = None
        self._current_well = None
        self._position_timer = QTimer(self)
        self._position_timer.setInterval(250)
        self._position_timer.timeout.connect(self._refresh_stage_well)
        layout = QVBoxLayout(self)
        self.load_button = QPushButton("Load well plate config…")
        self.load_button.clicked.connect(self.load_config)
        layout.addWidget(self.load_button)
        self.config_label = QLabel("Load a well-plate JSON saved from the position definitions.")
        self.config_label.setWordWrap(True)
        layout.addWidget(self.config_label)
        instructions = QLabel("Double-click a well center or boundary dot to move XY. Z stays unchanged.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.view = WellPlateView(self)
        self.view.setMinimumHeight(260)
        self.view.setDragMode(WellPlateView.DragMode.NoDrag)
        self.view.setSelectionMode(WellPlateView.SelectionMode.NoSelection)
        self.view.setSelectedColor(Qt.transparent)
        install_plate_navigation_view(self.view)
        self.view.setEnabled(False)
        self.view.positionDoubleClicked.connect(self.move_to_position)
        layout.addWidget(self.view)
        self.status = QLabel("No plate loaded.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_stage_well()
        self._position_timer.start()

    def hideEvent(self, event):
        self._position_timer.stop()
        super().hideEvent(event)

    def _refresh_stage_well(self):
        if self.plan is None:
            return
        well = None
        core = self.owner.core
        try:
            if (core is not None and not core.mda.is_running()
                    and core.getXYStageDevice()):
                xy = core.getXYPosition()
                # Map stage offsets into fractions of the well's two radii.
                offsets = (xy - self._centers) @ self._to_well_units
                if self.plan.plate.circular_wells:
                    distances = np.linalg.norm(offsets, axis=1)
                else:
                    distances = np.max(np.abs(offsets), axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] <= 1 + 1e-9:
                    p = self.plan.all_well_positions[nearest]
                    well = (p.plate_row, p.plate_col)
        except Exception:
            # The stage may be disconnected or temporarily unavailable.
            pass
        if well != self._current_well:
            if self._current_well is not None:
                self.view.setWellColor(*self._current_well, None)
            if well is not None:
                self.view.setWellColor(*well, Qt.green)
            self._current_well = well

    def load_config(self, path=None):
        if not isinstance(path, (str, Path)):
            path, _ = QFileDialog.getOpenFileName(self, "Load Well Plate Config", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            model = plate_plan_model(data)
            plan = model.model_validate(data)
            if not np.isfinite(plan.affine_transform).all():
                raise ValueError("Plate coordinates must be finite")
            self._centers = np.array([(p.x, p.y) for p in plan.all_well_positions])
            # As for the clickable boundary dots, use calibrated directions
            # with the physical well height and width (in micrometers).
            directions = plan.affine_transform[:2, :2][::-1]
            radii = np.asarray(plan.plate.well_size[::-1]) * 500
            axes = directions / np.linalg.norm(directions, axis=0) * radii
            self._to_well_units = np.linalg.inv(axes).T
            self.view.drawPlate(plan)
        except Exception as error:
            self.plan = None
            self.view.clear()
            self.view.setEnabled(False)
            self.config_label.setText("No valid plate loaded.")
            self.status.setText(f"Could not load plate: {error}")
            return
        self.plan = plan
        self._current_well = None
        if self.isVisible():
            self._refresh_stage_well()
        self.config_label.setText(str(Path(path)))
        self.view.setEnabled(True)
        self.status.setText(f"Loaded {plan.plate.rows} × {plan.plate.columns} plate. Ready for XY navigation.")

    def move_to_position(self, position):
        if self.plan is None:
            return
        core = self.owner.core
        try:
            if core is None or not core.getXYStageDevice():
                raise RuntimeError("Connect an XY stage before navigating")
            if core.mda.is_running() or getattr(self.owner, "_raman_mda_pending", False):
                raise RuntimeError("Wait for acquisition to finish before navigating")
            x, y = float(position.x), float(position.y)
            if not np.isfinite([x, y]).all():
                raise ValueError("Target coordinates must be finite")
            core.setXYPosition(x, y)
            self.status.setText(f"Move requested: {position.name}, XY ({x:.1f}, {y:.1f}) µm")
        except Exception as error:
            self.status.setText(f"Could not move stage: {error}")
