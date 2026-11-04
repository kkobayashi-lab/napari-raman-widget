from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QComboBox, QPushButton, QVBoxLayout, QWidget

from napari_raman_widget import live_preview


class CoreEvents(QObject):
    continuousSequenceAcquisitionStarted = Signal()
    sequenceAcquisitionStopped = Signal(str)
    systemConfigurationLoaded = Signal()


class ChannelControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        QVBoxLayout(self).addWidget(QComboBox())


class ChannelGroupControl(QComboBox):
    pass


@pytest.fixture
def preview(monkeypatch):
    monkeypatch.setattr(live_preview, "ChannelWidget", ChannelControl)
    monkeypatch.setattr(live_preview, "ChannelGroupWidget", ChannelGroupControl)
    window = QWidget()
    channels = [ChannelControl(window), ChannelControl(window)]
    group = ChannelGroupControl(window)
    stop = QPushButton(window)
    core = SimpleNamespace(events=CoreEvents(), isSequenceRunning=Mock(return_value=False))
    yield window, channels, group, stop, core
    lock = getattr(window, "_raman_live_channel_lock", None)
    if lock is not None:
        lock.disconnect()
    window.close()


def test_live_preview_locks_all_channel_controls_and_restores_selection(preview):
    window, channels, group, stop, core = preview
    for channel in channels:
        combo = channel.findChild(QComboBox)
        combo.addItems(["BF", "GFP"])
        combo.setCurrentText("GFP")
    live_preview.install_live_preview_channel_lock(window, core)
    core.isSequenceRunning.return_value = True
    core.events.continuousSequenceAcquisitionStarted.emit()
    core.events.continuousSequenceAcquisitionStarted.emit()
    assert all(not channel.isEnabled() for channel in channels)
    assert not group.isEnabled()
    assert stop.isEnabled()
    core.isSequenceRunning.return_value = False
    core.events.sequenceAcquisitionStopped.emit("Camera")
    assert all(channel.isEnabled() for channel in channels)
    assert group.isEnabled()
    assert all(channel.findChild(QComboBox).currentText() == "GFP" for channel in channels)


def test_install_during_preview_preserves_disabled_controls(preview):
    window, channels, group, stop, core = preview
    group.setEnabled(False)
    core.isSequenceRunning.return_value = True
    lock = live_preview.install_live_preview_channel_lock(window, core)
    assert live_preview.install_live_preview_channel_lock(window, core) is lock
    assert all(not channel.isEnabled() for channel in channels)
    # A stop for another camera must not unlock a still-running preview.
    core.events.sequenceAcquisitionStopped.emit("OtherCamera")
    assert not channels[0].isEnabled()
    core.isSequenceRunning.return_value = False
    core.events.sequenceAcquisitionStopped.emit("Camera")
    assert all(channel.isEnabled() for channel in channels)
    assert not group.isEnabled()


def test_rebuilt_channel_combo_stays_disabled_until_preview_stops(preview):
    window, channels, group, stop, core = preview
    live_preview.install_live_preview_channel_lock(window, core)
    core.isSequenceRunning.return_value = True
    core.events.continuousSequenceAcquisitionStarted.emit()
    replacement = QComboBox(channels[0])
    assert not replacement.isEnabled()
    core.isSequenceRunning.return_value = False
    core.events.systemConfigurationLoaded.emit()
    assert replacement.isEnabled()


def test_disconnect_removes_core_callbacks(preview):
    window, channels, group, stop, core = preview
    lock = live_preview.install_live_preview_channel_lock(window, core)
    lock.disconnect()
    lock.disconnect()
    core.events.continuousSequenceAcquisitionStarted.emit()
    assert all(channel.isEnabled() for channel in channels)
