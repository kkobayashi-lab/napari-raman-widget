"""Capture, inspect and accept well boundaries before fitting a well center."""
from datetime import datetime, timezone
from types import MethodType

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from .well_boundary import BoundaryCapture, boundary_to_stage, detect_boundary, fit_well_boundary


class BoundaryWellWidget(QWidget):
    calibrationChanged = Signal(bool)

    def __init__(self, owner, well_name, radius, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.radius = radius
        self.pending = None
        self.accepted = []
        self.fit = None
        self._well_center = None
        self.well_label = QLabel(f"Well {well_name}")
        self.status = QLabel("Focus a boundary section, then capture. Accept views from at least 3 sides.")
        self.status.setWordWrap(True)
        self.capture_button = QPushButton("Add current XYZ / Capture")
        self.accept_button = QPushButton("Accept boundary")
        self.retake_button = QPushButton("Retake at current XYZ")
        self.discard_button = QPushButton("Discard pending")
        self.remove_button = QPushButton("Remove selected view")
        self.apply_button = QPushButton("Use fitted well center")
        self.overlay = QCheckBox("Show detected boundary")
        self.overlay.setChecked(True)
        self.views = QListWidget()
        self.views.setMaximumHeight(85)
        self.figure = Figure(figsize=(5, 3), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(360, 230)
        self.ax = self.figure.add_subplot(121)
        self.overview = self.figure.add_subplot(122)
        self.selector = RectangleSelector(self.ax, self._select_roi, button=[1],
                                          minspanx=16, minspany=16, spancoords="pixels")
        from .well_boundary_guidance import BoundaryGuidance
        self.guide = BoundaryGuidance(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.well_label)
        hint = QLabel("Draw a rectangle on the captured image to retry detection in that region.\n"
                      "Capture stops live imaging. Z is saved for this view, not used as sample focus.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.canvas)
        layout.addWidget(self.overlay)
        row = QHBoxLayout()
        for button in (self.accept_button, self.retake_button, self.discard_button):
            row.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.views)
        layout.addWidget(self.remove_button)
        layout.addWidget(self.guide)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.status)
        self.capture_button.clicked.connect(self.capture)
        self.retake_button.clicked.connect(self.capture)
        self.accept_button.clicked.connect(self.accept)
        self.discard_button.clicked.connect(self.discard)
        self.remove_button.clicked.connect(self.remove)
        self.apply_button.clicked.connect(self.apply_center)
        self.overlay.toggled.connect(self.redraw)
        self.views.currentRowChanged.connect(self.redraw)
        self._buttons()

    def wellCenter(self):
        return self._well_center

    def _buttons(self):
        pending = self.pending is not None
        self.capture_button.setEnabled(not pending)
        self.accept_button.setEnabled(pending and self.pending.stage_xy is not None)
        self.retake_button.setEnabled(pending)
        self.discard_button.setEnabled(pending)
        self.apply_button.setEnabled(self.fit is not None and not pending)
        self.remove_button.setEnabled(bool(self.accepted) and not pending)
        self.guide.buttons()
        if self.guide.seed is not None and not self.guide.verified:
            self.apply_button.setEnabled(False)
        if self.guide.moving:
            for control in (self.capture_button, self.accept_button, self.retake_button,
                            self.discard_button, self.remove_button, self.apply_button, self.views):
                control.setEnabled(False)
        else:
            self.views.setEnabled(True)

    def hideEvent(self, event):
        if self.guide.moving:
            self.guide.stop_motion()
        super().hideEvent(event)

    def capture(self):
        if self.guide.moving:
            return
        # Invalidate an old pending image first: a failed retake cannot be accepted.
        self.pending = None
        try:
            core = self.owner.core
            if core is None or core.mda.is_running():
                raise ValueError("Connect the microscope and stop MDA before capturing")
            objective = self.owner._ensure_current_vandermonde()
            model = self.owner.vandermonde
            if objective is None or model is None:
                raise ValueError("Load the current objective's pixel-to-stage calibration")
            from .grid_calibration import vandermonde_fov
            from cns_control.utils import stop_live_acquisitions
            vandermonde_fov(model, core.getImageWidth(), core.getImageHeight())
            stop_live_acquisitions(core)
            core.waitForSystem()
            xyz = (*core.getXYPosition(), core.getPosition())
            if not np.isfinite(xyz).all():
                raise ValueError("Stage XYZ must be finite")
            camera = core.getCameraDevice()
            metadata = dict(well=self.well_label.text(), objective=objective,
                            camera=camera, exposure_ms=core.getExposure(),
                            binning=core.getProperty(camera, "Binning"),
                            roi=tuple(core.getROI()),
                            captured_at=datetime.now(timezone.utc).isoformat())
            reference = np.array([self.owner.sel_cx_input.value(), self.owner.sel_cy_input.value()])
            core.snapImage()
            image = np.array(core.getImage(), copy=True)
            after = np.array([*core.getXYPosition(), core.getPosition()])
            if not np.allclose(xyz, after, rtol=0, atol=0.1):
                raise ValueError("Stage moved during capture; stop moving and retake")
            self.pending = BoundaryCapture(image, xyz, metadata,
                                            (np.array(model[0], copy=True), model[1]), reference)
            self._detect()
        except Exception as error:
            self.status.setText(f"Capture failed: {error}")
        self._buttons()
        self.redraw()

    def _detect(self, roi=None):
        capture = self.pending
        capture.pixels_yx = capture.stage_xy = None
        try:
            capture.pixels_yx = detect_boundary(capture.image, roi)
            capture.stage_xy = boundary_to_stage(capture.pixels_yx, capture.xyz,
                                                 capture.reference_xy, capture.model)
            self.status.setText(f"Pending XYZ {capture.xyz}: check the cyan boundary, then Accept or Retake.")
        except ValueError as error:
            self.status.setText(f"Detection needs review: {error}. Draw a region or retake.")

    def _select_roi(self, start, end):
        if self.pending is None or None in (start.xdata, start.ydata, end.xdata, end.ydata):
            return
        height, width = self.pending.image.shape
        x0, x1 = np.clip(sorted([start.xdata, end.xdata]), 0, width).astype(int)
        y0, y1 = np.clip(sorted([start.ydata, end.ydata]), 0, height).astype(int)
        self._detect((y0, x0, y1, x1))
        self._buttons()
        self.redraw()

    def accept(self):
        if self.guide.moving or self.pending is None or self.pending.stage_xy is None:
            return
        if not self.guide.accept(self.pending):
            return
        self.accepted.append(self.pending)
        x, y, z = self.pending.xyz
        self.views.addItem(f"View {len(self.accepted)}: X {x:.1f}, Y {y:.1f}, Z {z:.1f} µm")
        self.pending = None
        self.views.setCurrentRow(len(self.accepted)-1)
        self._refit()

    def discard(self):
        self.pending = None
        self.status.setText("Pending capture discarded; accepted views are unchanged.")
        self._buttons()
        self.redraw()

    def remove(self):
        row = self.views.currentRow()
        if self.pending is not None or row < 0:
            return
        self.guide.clear()
        self.accepted.pop(row)
        self.views.takeItem(row)
        self._refit()

    def _refit(self):
        self.fit = None
        if self._well_center is not None:
            self._well_center = None
            self.calibrationChanged.emit(False)
        try:
            self.fit = fit_well_boundary(self.accepted, self.radius)
            center, radius, rms = self.fit
            self.status.setText(f"Center XY ({center[0]:.1f}, {center[1]:.1f}) µm; "
                                f"diameter {2*radius:.1f} µm; boundary RMS {rms:.1f} µm. "
                                "Review, then Use fitted well center. RMS is not center uncertainty.")
        except ValueError as error:
            self.status.setText(str(error))
        if self.guide.seed is not None and self.guide.index == 4 and self.fit is None:
            self.guide.info.setText("The four sides did not produce a valid fit. Review the accepted views; end guidance to remove or add measurements.")
        self._buttons()
        self.redraw()

    def apply_center(self):
        if (self.fit is not None and self.pending is None and not self.guide.moving
                and (self.guide.seed is None or self.guide.verified)):
            self._well_center = self.fit[0]
            self.calibrationChanged.emit(True)
            self.status.setText("Well center applied to plate calibration. Captured Z remains image metadata.")

    def redraw(self, *_):
        self.ax.clear()
        self.overview.clear()
        row = self.views.currentRow()
        capture = self.pending or (self.accepted[row] if 0 <= row < len(self.accepted) else None)
        if capture is not None:
            self.ax.imshow(capture.image, cmap="gray")
            if self.overlay.isChecked() and capture.pixels_yx is not None:
                self.ax.plot(capture.pixels_yx[:, 1], capture.pixels_yx[:, 0], color="cyan", lw=1)
        self.ax.set_title("Pending review" if self.pending else "Accepted view", fontsize=9)
        self.ax.set_axis_off()
        for capture in self.accepted:
            self.overview.plot(*capture.stage_xy.T, lw=1)
        if self.fit:
            center, radius, _ = self.fit
            theta = np.linspace(0, 2*np.pi, 200)
            self.overview.plot(center[0]+radius*np.cos(theta), center[1]+radius*np.sin(theta), "--", lw=1)
            self.overview.plot(*center, "+")
        self.guide.draw(capture)
        self.overview.set_title("Accepted boundaries", fontsize=9)
        self.overview.set_xlabel("Stage X (µm)", fontsize=8)
        self.overview.set_ylabel("Stage Y (µm)", fontsize=8)
        self.overview.tick_params(labelsize=7)
        self.overview.set_aspect("equal", adjustable="datalim")
        self.canvas.draw_idle()


def install_boundary_calibration(wizard, owner):
    """Use boundary capture for circular wells; preserve native rectangular mode."""
    calibration = wizard.calibration_page.widget
    if getattr(calibration, "_boundary_capture_installed", False):
        return
    calibration._boundary_capture_installed = True
    native_factory = calibration._get_or_create_well_calibration_widget

    def factory(_self, idx):
        if idx in calibration._calibration_widgets:
            return calibration._calibration_widgets[idx]
        plate = calibration._current_plate
        if plate is None or not plate.circular_wells:
            return native_factory(idx)
        widget = BoundaryWellWidget(owner, plate.all_well_names[idx],
                                    plate.well_size[0]*500, calibration)
        widget.calibrationChanged.connect(calibration._on_well_calibration_changed)
        calibration._calibration_widgets[idx] = widget
        calibration._calibration_widget_stack.addWidget(widget)
        return widget

    calibration._get_or_create_well_calibration_widget = MethodType(factory, calibration)
    # The wizard may already have created A1 before this extension was installed.
    for idx, widget in list(calibration._calibration_widgets.items()):
        if calibration._current_plate.circular_wells:
            calibration._calibration_widgets.pop(idx)
            calibration._calibration_widget_stack.removeWidget(widget)
            widget.deleteLater()
    calibration._on_plate_selection_changed()
