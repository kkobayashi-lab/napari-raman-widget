"""Manual XY navigation using a saved, calibrated well-plate plan."""
import json
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget
from pymmcore_widgets.useq_widgets._well_plate_widget import WellPlateView
from useq import WellPlatePlan

from .wellplate_calibration import CalibratedWellPlatePlan, install_plate_navigation_view


class WellPlateNavigationPanel(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.plan = None
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
        install_plate_navigation_view(self.view)
        self.view.setEnabled(False)
        self.view.positionDoubleClicked.connect(self.move_to_position)
        layout.addWidget(self.view)
        self.status = QLabel("No plate loaded.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def load_config(self, path=None):
        if not isinstance(path, (str, Path)):
            path, _ = QFileDialog.getOpenFileName(self, "Load Well Plate Config", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            model = CalibratedWellPlatePlan if "stage_affine_um" in data else WellPlatePlan
            plan = model.model_validate(data)
            if not np.isfinite(plan.affine_transform).all():
                raise ValueError("Plate coordinates must be finite")
            self.view.drawPlate(plan)
        except Exception as error:
            self.plan = None
            self.view.clear()
            self.view.setEnabled(False)
            self.config_label.setText("No valid plate loaded.")
            self.status.setText(f"Could not load plate: {error}")
            return
        self.plan = plan
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
