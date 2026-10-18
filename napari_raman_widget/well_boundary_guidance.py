"""Assisted XY sampling of four sides, with a separate fifth-view check."""
import time

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .well_boundary import boundary_to_stage


def initial_geometry(capture, inward_xy, radius):
    """Estimate the normal locally; choose its sign from a user interior click."""
    pixels = capture.pixels_yx[:, ::-1]
    middle = np.median(pixels, axis=0)
    _, singular, axes = np.linalg.svd(pixels-middle, full_matrices=False)
    if singular[0] <= 0 or singular[1] > singular[0]*0.35:
        raise ValueError("Select a short, smooth boundary section for the starting direction")
    normal = axes[1]
    projection = np.dot(np.asarray(inward_xy)-middle, normal)
    if abs(projection) < 5:
        raise ValueError("Click farther inside the well, away from the boundary")
    normal *= np.sign(projection)
    image_pair = np.array([middle, middle + normal*10])
    stage_pair = boundary_to_stage(image_pair[:, ::-1], capture.xyz,
                                   capture.reference_xy, capture.model)
    inward = stage_pair[1]-stage_pair[0]
    length = np.linalg.norm(inward)
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError("Invalid calibrated boundary direction")
    inward /= length
    center = stage_pair[0] + radius*inward
    angle = np.arctan2(-inward[1], -inward[0])
    return center, angle, image_pair


def sampling_target(center, radius, angle, index):
    """Three more quadrants, followed by a held-out 45 degree location."""
    if index not in (1, 2, 3, 4):
        raise ValueError("No further target")
    theta = angle + (index*np.pi/2 if index < 4 else np.pi/4)
    direction = np.array([np.cos(theta), np.sin(theta)])
    return np.asarray(center) + radius*direction, direction


class BoundaryGuidance(QGroupBox):
    def __init__(self, host):
        super().__init__("Assisted four-side collection", host)
        self.host = host
        self.seed = None
        self.center = None
        self.angle = None
        self.arrow = None
        self.index = 1
        self.arrived = False
        self.verified = False
        self.verification = None
        self.target = None
        self.search_index = 0
        self.moving = False
        self.motion_core = None
        self.motion_device = None
        self.choose = QPushButton("Choose inward side")
        self.choose.setCheckable(True)
        self.next = QPushButton("Confirm direction / Move to side 2")
        self.search = QPushButton("Search next radial offset")
        self.stop = QPushButton("Stop XY")
        self.reset = QPushButton("End guidance")
        self.info = QLabel("Accept one boundary image. Choose inward side, then click inside the well on that image.")
        self.info.setWordWrap(True)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        for button in (self.choose, self.next, self.search):
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.stop)
        row.addWidget(self.reset)
        layout.addLayout(row)
        layout.addWidget(self.info)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.choose.toggled.connect(self.choose_mode)
        self.next.clicked.connect(self.move_next)
        self.search.clicked.connect(self.search_next)
        self.stop.clicked.connect(self.stop_motion)
        self.reset.clicked.connect(self.clear)
        host.canvas.mpl_connect("button_press_event", self.clicked)

    def choose_mode(self, enabled):
        self.host.selector.set_active(not enabled)
        if enabled:
            self.info.setText("Click the illuminated interior on the accepted image. The yellow arrow must point inward.")

    def clicked(self, event):
        if not self.choose.isChecked() or event.inaxes is not self.host.ax or event.button != 1:
            return
        row = self.host.views.currentRow()
        if self.host.pending is not None or row < 0 or self.moving:
            return
        try:
            seed = self.host.accepted[row]
            center, angle, arrow = initial_geometry(seed, [event.xdata, event.ydata], self.host.radius)
            self.seed, self.center, self.angle, self.arrow = seed, center, angle, arrow
            self.index, self.arrived, self.verified = 1, False, False
            self.verification = None
            self.choose.setChecked(False)
            self.info.setText("Check the yellow inward arrow, then confirm the move. XY only; focus Z yourself at each stop.")
        except ValueError as error:
            self.info.setText(str(error))
        self.host._buttons()
        self.host.redraw()

    def buttons(self):
        ready = self.host.pending is None and not self.moving
        self.choose.setEnabled(ready and bool(self.host.accepted))
        self.next.setEnabled(ready and self.seed is not None and not self.arrived
                             and self.index <= 4 and (self.index < 4 or self.host.fit is not None))
        self.search.setEnabled(ready and self.arrived and self.search_index < 6)
        self.stop.setEnabled(self.moving)
        self.reset.setEnabled(self.seed is not None or self.moving)
        if self.index == 1:
            self.next.setText("Confirm direction / Move to side 2")
        else:
            self.next.setText("Move to check location" if self.index == 4 else f"Move to side {self.index+1}")

    def validate_context(self):
        core = self.host.owner.core
        if core is None or core.mda.is_running():
            raise ValueError("Connect the microscope and stop MDA before moving")
        owner = self.host.owner
        objective = owner._ensure_current_vandermonde()
        meta = self.seed.metadata
        if (objective != meta["objective"] or core.getCameraDevice() != meta["camera"]
                or core.getProperty(meta["camera"], "Binning") != meta["binning"]
                or tuple(core.getROI()) != meta["roi"]
                or owner.vandermonde is None or owner.vandermonde[1] != self.seed.model[1]
                or not np.array_equal(owner.vandermonde[0], self.seed.model[0])
                or not np.array_equal([owner.sel_cx_input.value(), owner.sel_cy_input.value()], self.seed.reference_xy)):
            raise ValueError("Camera/objective calibration changed; end guidance and capture a new starting view")
        return core

    def geometry(self):
        if self.host.fit is not None:
            return np.asarray(self.host.fit[0]), self.host.fit[1]
        return self.center, self.host.radius

    def move_next(self):
        if self.moving or self.seed is None or self.arrived or self.host.pending is not None or self.index > 4:
            return
        if self.index == 4 and self.host.fit is None:
            return
        center, radius = self.geometry()
        self.target, self.direction = sampling_target(center, radius, self.angle, self.index)
        self.search_index = 0
        self.move(self.target)

    def search_next(self):
        if not self.arrived or self.moving or self.host.pending is not None or self.search_index >= 6:
            return
        from .grid_calibration import vandermonde_fov
        h, w = self.seed.image.shape
        # Six overlapping positions, bounded to one FOV or 20% of the radius.
        extent = min(min(vandermonde_fov(self.seed.model, w, h)), self.host.radius*0.2)
        offsets = [1/3, -1/3, 2/3, -2/3, 1, -1]
        offset = offsets[self.search_index]*extent
        self.search_index += 1
        self.move(self.target + offset*self.direction)

    def move(self, boundary_xy):
        try:
            core = self.validate_context()
            h, w = self.seed.image.shape
            # Place the predicted boundary at image center, even if the aiming
            # reference differs from the image center.
            bias = boundary_to_stage([[h/2, w/2]], (0, 0, 0),
                                     self.seed.reference_xy, self.seed.model)[0]
            destination = np.asarray(boundary_xy)-bias
            current = np.asarray(core.getXYPosition())
            if (not np.isfinite(destination).all() or not np.isfinite(current).all()
                    or np.linalg.norm(destination-current) > 2.5*self.host.radius
                    or np.linalg.norm(destination-np.asarray(self.seed.xyz[:2])) > 2.5*self.host.radius):
                raise ValueError("Target is outside the local well search range; reposition or restart guidance")
            from cns_control.utils import stop_live_acquisitions
            stop_live_acquisitions(core)
            device = core.getXYStageDevice()
            if core.deviceBusy(device):
                raise ValueError("XY stage is already moving")
            self.motion_core, self.motion_device = core, device
            self.destination = destination
            self.deadline = time.monotonic()+30
            self.moving = True
            core.setXYPosition(*map(float, destination))
            self.timer.start()
            self.info.setText(f"Moving XY to ({destination[0]:.1f}, {destination[1]:.1f}) µm. Z unchanged.")
        except Exception as error:
            if self.moving:
                self.stop_motion()
            self.info.setText(f"Move failed: {error}")
        self.host._buttons()

    def poll(self):
        try:
            if time.monotonic() > self.deadline:
                raise RuntimeError("XY move timed out")
            if self.motion_core.deviceBusy(self.motion_device):
                return
            if not np.allclose(self.motion_core.getXYPosition(), self.destination, rtol=0, atol=2):
                raise RuntimeError("XY stage did not reach the requested position")
            self.timer.stop()
            self.moving, self.arrived = False, True
            self.info.setText("Focus the rim, then Capture and Accept. If the rim is absent, use Search next radial offset (6 positions maximum).")
        except Exception as error:
            self.stop_motion()
            self.info.setText(f"Movement stopped: {error}")
        self.host._buttons()

    def stop_motion(self):
        self.timer.stop()
        if self.moving:
            try:
                self.motion_core.stop(self.motion_device)
            except Exception as error:
                self.info.setText(f"Stage stop failed: {error}. Use the hardware stop.")
                # Keep controls locked and retain a retryable Stop button.
                return
        self.moving, self.arrived = False, False
        self.info.setText("XY movement stopped. No image was accepted.")
        self.host._buttons()

    def clear(self):
        self.stop_motion()
        if self.moving:
            return
        self.seed = self.center = self.arrow = self.verification = None
        self.index, self.arrived, self.verified = 1, False, False
        self.choose.setChecked(False)
        self.info.setText("Guidance ended. Accepted boundary views remain available.")
        self.host._buttons()
        self.host.redraw()

    def accept(self, capture):
        """Return False for a rejected capture or a separate validation capture."""
        if self.seed is None:
            return True
        if not self.arrived:
            self.info.setText("Move to the next guided target before accepting, or end guidance for manual collection.")
            return False
        try:
            self.validate_context()
            if (any(capture.metadata.get(key) != self.seed.metadata[key]
                    for key in ("objective", "camera", "binning", "roi"))
                    or capture.model[1] != self.seed.model[1]
                    or not np.array_equal(capture.model[0], self.seed.model[0])
                    or not np.array_equal(capture.reference_xy, self.seed.reference_xy)):
                raise ValueError("This image used different camera calibration; retake it")
            center, radius = self.geometry()
            measured = capture.stage_xy.mean(axis=0)-center
            if np.linalg.norm(measured) < 1 or np.dot(measured/np.linalg.norm(measured), self.direction) < np.cos(np.deg2rad(30)):
                raise ValueError("This boundary is not near the requested side; retake or end guidance")
            if self.index == 4:
                residual = np.linalg.norm(capture.stage_xy-center, axis=1)-radius
                rms = float(np.sqrt(np.mean(residual**2)))
                if rms > max(5, radius*0.01):
                    raise ValueError(f"Independent check disagrees by {rms:.1f} µm RMS; inspect or retake")
                self.verification, self.verified = capture, True
                self.host.pending = None
                self.index, self.arrived = 5, False
                self.info.setText(f"Independent check passed: {rms:.1f} µm RMS. This view was excluded from the fit. Use fitted well center.")
                self.host._buttons()
                self.host.redraw()
                return False
            self.index += 1
            self.arrived = False
            self.info.setText("Boundary accepted. Move to the next side, focus, and capture.")
            return True
        except (ValueError, RuntimeError) as error:
            self.info.setText(str(error))
            return False

    def draw(self, capture):
        if capture is self.seed and self.arrow is not None:
            start, end = self.arrow
            tip = start+(end-start)*6
            self.host.ax.annotate("", xy=tip, xytext=start,
                                  arrowprops=dict(arrowstyle="->", color="yellow", lw=2))
            self.host.ax.annotate("Inward", tip, xytext=(4, 4), textcoords="offset points",
                                  color="yellow", fontsize=8)
        if self.seed is not None:
            center, radius = self.geometry()
            for i in range(1, 5):
                target, _ = sampling_target(center, radius, self.angle, i)
                self.host.overview.plot(*target, "x", color="orange")
                self.host.overview.annotate("Check" if i == 4 else str(i+1), target, fontsize=8)
        if self.verification is not None:
            self.host.overview.plot(*self.verification.stage_xy.T, color="green", lw=2)
