"""Keep channel controls locked while Micro-Manager streams preview images."""

from pymmcore_widgets import ChannelGroupWidget, ChannelWidget
from qtpy.QtCore import QObject, Qt, Slot


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
        for signal, slot in self._connections:
            signal.disconnect(slot)
        self._connections = ()


def install_live_preview_channel_lock(window, core):
    """Install once per Micro-Manager window, including an already-live core."""
    lock = getattr(window, "_raman_live_channel_lock", None)
    if lock is None:
        lock = LivePreviewChannelLock(window, core)
        window._raman_live_channel_lock = lock
    return lock
