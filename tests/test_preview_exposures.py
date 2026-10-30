"""Exercise preview exposure restoration with real Qt widgets and a fake camera."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import QObject, Signal

for name in ("QAction", "QActionGroup", "QUndoCommand", "QUndoStack"):
    if not hasattr(QtGui, name):
        setattr(QtGui, name, getattr(QtWidgets, name))

spec = importlib.util.spec_from_file_location(
    "preview_under_test",
    Path(__file__).parents[1] / "napari_raman_widget" / "live_preview.py",
)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class Events(QObject):
    configSet = Signal(str, str)
    channelGroupChanged = Signal(str)
    systemConfigurationLoaded = Signal()
    exposureChanged = Signal(str, float)


class Core:
    def __init__(self):
        self.events = Events()
        self.channel = "BF"
        self.group = "Channel"
        self.exposure = 10.0
        self.running = False
        self.mda = SimpleNamespace(is_running=lambda: self.running)
        self.defaults = {"BF": "10", "GFP": "200", "Unknown": "nan"}

    def getCameraDevice(self):
        return "Camera"

    def getLoadedDevices(self):
        return ["Camera"]

    def getChannelGroup(self):
        return self.group

    def getCurrentConfig(self, group):
        return self.channel

    def getConfigData(self, group, channel):
        return [("Camera", "Exposure", self.defaults[channel])]

    def getExposure(self, *_args):
        return self.exposure

    def setExposure(self, value):
        self.exposure = value
        self.events.exposureChanged.emit("Camera", value)

    def switch(self, channel):
        self.channel = channel
        # Presets can change hardware exposure before configSet is emitted.
        if channel != "Unknown":
            self.setExposure(float(self.defaults[channel]))
        self.events.configSet.emit(self.group, channel)


class PreviewExposureTests(unittest.TestCase):
    def test_defaults_edits_acquisition_reload_and_cleanup(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = QtWidgets.QWidget()
        core = Core()
        widget = preview.ExposureWidget(parent=window, mmcore=core)
        controller = preview.PreviewChannelExposures(window, core)
        widget.spinBox.setValue(35)
        core.switch("GFP")
        self.assertEqual(widget.spinBox.value(), 200)
        widget.spinBox.setValue(450)
        core.switch("BF")
        self.assertEqual(core.exposure, 35)
        core.switch("GFP")
        self.assertEqual(core.exposure, 450)
        core.running = True
        core.switch("BF")
        core.setExposure(999)
        core.running = False
        core.switch("GFP")
        core.switch("BF")
        self.assertEqual(core.exposure, 35)
        core.events.configSet.emit("Objective", "10x")
        self.assertEqual(core.exposure, 35)
        core.switch("Unknown")
        self.assertEqual(core.exposure, 35)
        core.switch("BF")
        core.events.systemConfigurationLoaded.emit()
        self.assertEqual(core.exposure, 10)
        core.switch("GFP")
        self.assertEqual(core.exposure, 200)
        controller.disconnect()
        core.switch("BF")
        self.assertEqual(core.exposure, 10)
        window.close()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
