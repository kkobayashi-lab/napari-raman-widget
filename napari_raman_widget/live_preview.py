"""Manage live-preview channel controls, exposures, and image calibration."""

from math import isfinite

from pymmcore_widgets import ChannelGroupWidget, ChannelWidget, ExposureWidget
from qtpy.QtCore import QObject, Qt, Slot


def ensure_preview_layer(viewer, core):
    """Create the live feed's image layer without starting camera acquisition."""
    if "preview" in viewer.layers:
        return viewer.layers["preview"]
    width, height = int(core.getImageWidth()), int(core.getImageHeight())
    if width <= 0 or height <= 0:
        return None

    import numpy as np

    pixel_size = core.getPixelSizeUm() or 1.0
    bit_depth = int(core.getImageBitDepth())
    dtype = np.uint8 if bit_depth <= 8 else np.uint16 if bit_depth <= 16 else np.uint32
    layer = viewer.add_image(
        np.zeros((height, width), dtype=dtype),
        name="preview",
        metadata={"mode": "preview"},
        scale=(pixel_size, pixel_size),
        contrast_limits=(0, 2 ** max(1, bit_depth) - 1),
    )
    viewer.reset_view()
    return layer


class LivePreviewChannelLock(QObject):
    def __init__(self, window, core):
        super().__init__(window)
        self._window = window
        self._core = core
        self._saved = {}
        self._connections = (
            (core.events.continuousSequenceAcquisitionStarted, self._started),
            (core.events.sequenceAcquisitionStopped, self._sync),
            (core.events.systemConfigurationLoaded, self._sync),
        )
        for signal, slot in self._connections:
            signal.connect(slot)
        window.destroyed.connect(self.disconnect)
        self._sync()

    @Slot()
    def _started(self):
        self._set_locked(True)

    @Slot()
    def _sync(self):
        self._set_locked(self._core.isSequenceRunning())

    def _set_locked(self, locked):
        if locked:
            for kind in (ChannelWidget, ChannelGroupWidget):
                for widget in self._window.findChildren(kind):
                    if widget not in self._saved:
                        # Preserve explicit disabling, independently of ancestors.
                        self._saved[widget] = not widget.testAttribute(
                            Qt.WA_ForceDisabled
                        )
                        widget.setEnabled(False)
        else:
            for widget, enabled in self._saved.items():
                widget.setEnabled(enabled)
            self._saved.clear()

    @Slot()
    def disconnect(self):
        connections = self._connections
        self._connections = ()
        for signal, slot in connections:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                # Qt may already have disconnected or deleted objects on shutdown.
                pass


def install_live_preview_channel_lock(window, core):
    """Install once per Micro-Manager window, including an already-live core."""
    lock = getattr(window, "_raman_live_channel_lock", None)
    if lock is None:
        lock = LivePreviewChannelLock(window, core)
        window._raman_live_channel_lock = lock
    if getattr(window, "_raman_channel_exposures", None) is None:
        window._raman_channel_exposures = PreviewChannelExposures(window, core)
    return lock


class PreviewChannelExposures(QObject):
    """Remember preview edits separately from exposures used by MDA."""

    def __init__(self, window, core):
        super().__init__(window)
        self._core = core
        self._exposures = {}
        self._connections = [
            (core.events.configSet, self._channel_changed),
            (core.events.channelGroupChanged, self._group_changed),
            (core.events.systemConfigurationLoaded, self._reset),
        ]
        # ExposureWidget blocks spinBox signals when synchronizing from the core.
        # Listen to edits here so MDA, autofocus, and preset changes cannot replace
        # the user's preview values with their own camera exposures.
        for widget in window.findChildren(ExposureWidget):
            def edited(value, widget=widget):
                self._remember(widget._camera, value)
            self._connections.append((widget.spinBox.valueChanged, edited))
        for signal, slot in self._connections:
            signal.connect(slot)
        window.destroyed.connect(self.disconnect)
        self._group_changed(core.getChannelGroup())

    def _remember(self, camera, exposure):
        core = self._core
        if core.mda.is_running() or camera != core.getCameraDevice():
            return
        group = core.getChannelGroup()
        channel = core.getCurrentConfig(group) if group else ""
        if channel and isfinite(exposure) and exposure > 0:
            self._exposures[group, channel, camera] = float(exposure)

    def _channel_changed(self, group, channel):
        core = self._core
        if core.mda.is_running() or group != core.getChannelGroup() or not channel:
            return
        camera = core.getCameraDevice()
        if not camera:
            return
        key = group, channel, camera
        exposure = self._exposures.get(key)
        if exposure is None:
            for device, prop, value in core.getConfigData(group, channel):
                if device == camera and prop == "Exposure":
                    try:
                        candidate = float(value)
                    except (TypeError, ValueError):
                        continue
                    if isfinite(candidate) and candidate > 0:
                        exposure = candidate
                        break
        if exposure is None:
            exposure = core.getExposure()
        if isfinite(exposure) and exposure > 0:
            core.setExposure(exposure)
            self._exposures[key] = float(exposure)

    def _group_changed(self, group):
        if group:
            self._channel_changed(group, self._core.getCurrentConfig(group))

    def _reset(self):
        self._exposures.clear()
        self._group_changed(self._core.getChannelGroup())

    @Slot()
    def disconnect(self):
        connections, self._connections = self._connections, []
        for signal, slot in connections:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass


class PreviewScaleBar(QObject):
    """Show the pixel-size table's current calibration in the preview canvas."""

    def __init__(self, window, viewer, core):
        super().__init__(window)
        self._viewer = viewer
        self._core = core
        self._connections = (
            (core.events.pixelSizeChanged, self._sync),
            (core.events.configSet, self._sync),
            (core.events.systemConfigurationLoaded, self._sync),
            (viewer.layers.events.inserted, self._sync),
            (viewer.layers.events.removed, self._sync),
        )
        for signal, slot in self._connections:
            signal.connect(slot)
        viewer.scale_bar.unit = "um"
        viewer.scale_bar.position = "bottom_right"
        viewer.scale_bar.box = True
        window.destroyed.connect(self.disconnect)
        self._sync()

    def _sync(self, *_args):
        pixel_size = float(self._core.getPixelSizeUm())
        valid = isfinite(pixel_size) and pixel_size > 0
        preview = self._viewer.layers["preview"] if "preview" in self._viewer.layers else None
        if preview is not None:
            preview.scale = (pixel_size, pixel_size) if valid else (1.0, 1.0)
        self._viewer.scale_bar.visible = valid and preview is not None

    @Slot()
    def disconnect(self):
        connections, self._connections = self._connections, ()
        for signal, slot in connections:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass


def install_preview_scale_bar(window, viewer, core):
    if getattr(window, "_raman_preview_scale_bar", None) is None:
        window._raman_preview_scale_bar = PreviewScaleBar(window, viewer, core)
    else:
        window._raman_preview_scale_bar._sync()
