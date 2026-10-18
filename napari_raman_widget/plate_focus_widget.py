"""Manual focus capture inside the well-plate calibration page."""
from types import MethodType

from qtpy.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton,
                           QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .plate_focus import PlateFocusCalibration


class PlateFocusWidget(QWidget):
    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.samples = []
        layout = QVBoxLayout(self)
        info = QLabel("Move XY, focus manually, then Record current XYZ. Sample corners and interior "
                      "of the acquisition area using the same objective and focus reference. "
                      "Recalibrate after moving the plate or changing objective. Units: µm.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.enabled = QCheckBox("Use fitted Z before autofocus / acquisition")
        self.enabled.setToolTip("Overrides manual position Z values, even when Include Z is off. "
                                "Autofocus refines the initial height; later visits reuse that result.")
        layout.addWidget(self.enabled)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Stage X (µm)", "Stage Y (µm)", "Focus Z (µm)"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        for label, callback in [("Record current XYZ", self.record),
                                ("Remove selected", self.remove), ("Clear", self.clear)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh()

    def value(self):
        return PlateFocusCalibration(samples=tuple(self.samples), enabled=self.enabled.isChecked())

    def setValue(self, value):
        value = value or PlateFocusCalibration()
        self.samples = list(value.samples)
        self.enabled.setChecked(value.enabled)
        self.refresh()

    def record(self):
        try:
            core = self.owner.core
            if core is None or not core.getXYStageDevice() or not core.getFocusDevice():
                raise ValueError("Connect XY and focus stages first")
            if core.mda.is_running() or getattr(self.owner, "_raman_mda_pending", False):
                raise ValueError("Wait for acquisition to finish")
            core.waitForDevice(core.getXYStageDevice())
            core.waitForDevice(core.getFocusDevice())
            sample = (float(core.getXPosition()), float(core.getYPosition()), float(core.getPosition()))
            PlateFocusCalibration(samples=(sample,))
            # A second measurement at the same location replaces the old focus.
            self.samples = [p for p in self.samples if p[:2] != sample[:2]] + [sample]
            self.refresh()
        except Exception as error:
            self.status.setText(str(error))

    def remove(self):
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.samples.pop(row)
        self.refresh()

    def clear(self):
        self.setValue(None)

    def refresh(self):
        self.table.setRowCount(len(self.samples))
        for row, sample in enumerate(self.samples):
            for col, value in enumerate(sample):
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.3f}"))
        try:
            fit = self.value().fit()
            self.status.setText(f"{len(self.samples)} points; automatic degree {fit.degree} (0–3). "
                                f"Fit RMS: {fit.rms:.3f} µm; leave-one-out RMS: {fit.validation_rms:.3f} µm. "
                                "Acquisition positions must lie inside the sampled area.")
        except ValueError as error:
            self.status.setText(str(error))


def install_focus_calibration(wizard, owner):
    page = wizard.calibration_page
    calibration = page.widget
    tabs = QTabWidget(page)
    page.layout().removeWidget(calibration)
    tabs.addTab(calibration, "XY calibration")
    focus = PlateFocusWidget(owner, tabs)
    tabs.addTab(focus, "Z height model")
    page.layout().addWidget(tabs)
    wizard.focus_height_widget = focus
    native_value, native_set_value = calibration.value, calibration.setValue

    def value(_self):
        plan = native_value()
        if plan is not None:
            from .wellplate_calibration import FocusWellPlatePlan
            if not isinstance(plan, FocusWellPlatePlan):
                plan = FocusWellPlatePlan(**plan.model_dump())
            plan = plan.replace(focus_height=focus.value())
        return plan

    def set_value(_self, plan):
        focus.setValue(getattr(plan, "focus_height", None))
        native_set_value(plan)

    calibration.value = MethodType(value, calibration)
    calibration.setValue = MethodType(set_value, calibration)
