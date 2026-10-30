"""Event-driven focus position beside napari's cursor status."""

import math

from qtpy.QtCore import Slot
from qtpy.QtWidgets import QLabel


class StageStatusLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._core = None
        self._z_device = ""
        self._z = None
        self.setToolTip(
            "Last reported Z stage position in micrometers. "
            "Updates when the microscope reports a position change."
        )
        self.setContentsMargins(6, 0, 12, 0)
        self._render()

    def bind(self, core):
        self.unbind()
        self._core = core
        core.events.stagePositionChanged.connect(self._on_z)
        core.events.systemConfigurationLoaded.connect(self._refresh)
        self._refresh()

    def unbind(self):
        if self._core is not None:
            for signal, callback in (
                (self._core.events.stagePositionChanged, self._on_z),
                (self._core.events.systemConfigurationLoaded, self._refresh),
            ):
                try:
                    signal.disconnect(callback)
                except (RuntimeError, ValueError, TypeError):
                    pass
        self._core = None
        self._z_device = ""
        self._z = None
        self._render()

    @Slot()
    def _refresh(self):
        self._z = None
        self._z_device = ""
        try:
            self._z_device = self._core.getFocusDevice()
            if self._z_device:
                self._z = self._core.getPosition(self._z_device)
        except Exception:
            pass
        self._render()

    @Slot(str, float)
    def _on_z(self, device, z):
        if self._core is not None and device == self._z_device:
            self._z = z
            self._render()

    def _render(self):
        value = (
            f"{self._z:.2f}"
            if self._z is not None and math.isfinite(self._z)
            else "--"
        )
        self.setText(f"Z: {value} µm")
