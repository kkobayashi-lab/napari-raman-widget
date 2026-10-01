"""Wheel-driven focus control, without connecting to microscope hardware."""

import unittest
from unittest.mock import Mock

from qtpy.QtCore import QPoint, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QApplication, QPushButton, QSlider, QWidget

from napari_raman_widget.widget import HardwareWidget


class TestStageZScroll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Use the real Qt event filter without building unrelated hardware UI.
        self.widget = HardwareWidget.__new__(HardwareWidget)
        QWidget.__init__(self.widget)
        self.widget._napari_window = QWidget()
        self.widget._stage_scroll_canvas = QWidget()
        self.widget.drag_stage_btn = QPushButton()
        self.widget.drag_stage_btn.setCheckable(True)
        self.widget.drag_stage_btn.setChecked(True)
        self.widget.stage_z_scroll_slider = QSlider(Qt.Horizontal)
        self.widget.stage_z_scroll_slider.setRange(1, 100)
        self.widget.stage_z_scroll_slider.setValue(10)
        self.widget.status = Mock()
        self.core = Mock()
        self.core.mda.is_running.return_value = False
        self.core.getFocusDevice.return_value = "Z"
        self.core.deviceBusy.return_value = False
        self.widget.core = self.core

    def tearDown(self):
        for child in (
            self.widget._napari_window,
            self.widget._stage_scroll_canvas,
            self.widget.drag_stage_btn,
            self.widget.stage_z_scroll_slider,
            self.widget,
        ):
            child.deleteLater()
        self.app.processEvents()

    def wheel(self, delta, target=None):
        event = QWheelEvent(
            QPointF(20, 20), QPointF(20, 20), QPoint(), QPoint(0, delta),
            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
        )
        event.ignore()
        consumed = self.widget.eventFilter(
            target if target is not None else self.widget._stage_scroll_canvas,
            event,
        )
        return consumed, event

    def test_direction_fractional_notches_and_slider_rate(self):
        for rate, delta, expected in ((10, 120, 1), (25, -240, -5), (1, 60, .05)):
            with self.subTest(rate=rate, delta=delta):
                self.widget.stage_z_scroll_slider.setValue(rate)
                consumed, event = self.wheel(delta)
                self.assertTrue(consumed)
                self.assertTrue(event.isAccepted())
                self.core.setRelativePosition.assert_called_with("Z", expected)

    def test_disarmed_wheel_preserves_viewer_behavior(self):
        self.widget.drag_stage_btn.setChecked(False)
        self.assertFalse(self.wheel(120)[0])
        self.core.setRelativePosition.assert_not_called()

    def test_wheel_outside_canvas_does_not_move_stage(self):
        self.assertFalse(self.wheel(120, self.widget._napari_window)[0])
        self.core.setRelativePosition.assert_not_called()

    def test_horizontal_or_zero_delta_does_not_move(self):
        self.assertTrue(self.wheel(0)[0])
        self.core.setRelativePosition.assert_not_called()

    def test_busy_stage_drops_input(self):
        self.core.deviceBusy.return_value = True
        self.assertTrue(self.wheel(120)[0])
        self.core.setRelativePosition.assert_not_called()
        self.core.deviceBusy.return_value = False
        self.wheel(120)
        self.core.setRelativePosition.assert_called_once_with("Z", 1.0)

    def test_acquisition_disarms_without_moving(self):
        self.core.mda.is_running.return_value = True
        self.assertTrue(self.wheel(120)[0])
        self.assertFalse(self.widget.drag_stage_btn.isChecked())
        self.core.setRelativePosition.assert_not_called()

    def test_disconnect_disarms_without_moving(self):
        self.widget.core = None
        self.assertTrue(self.wheel(120)[0])
        self.assertFalse(self.widget.drag_stage_btn.isChecked())
        self.core.setRelativePosition.assert_not_called()

    def test_missing_focus_stage_disarms(self):
        self.core.getFocusDevice.return_value = ""
        self.wheel(120)
        self.assertFalse(self.widget.drag_stage_btn.isChecked())
        self.core.setRelativePosition.assert_not_called()
        self.assertIn("no Z focus stage", self.widget.status.setText.call_args.args[0])

    def test_hardware_error_disarms_and_reports_failure(self):
        self.core.setRelativePosition.side_effect = RuntimeError("move rejected")
        self.wheel(120)
        self.assertFalse(self.widget.drag_stage_btn.isChecked())
        self.assertIn("move rejected", self.widget.status.setText.call_args.args[0])
