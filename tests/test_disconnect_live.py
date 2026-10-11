from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from napari_raman_widget.widget import HardwareWidget


def test_disconnect_stops_live_before_tearing_down_controller():
    core = Mock()
    core.mda.is_running.return_value = False
    controller = Mock()
    controller.disconnect.side_effect = RuntimeError("end of test")
    widget = SimpleNamespace(
        core=core, main_window=None, _acquisition_controller=controller,
        status=Mock(),
    )
    order = Mock()
    order.attach_mock(controller.disconnect, "disconnect")
    with patch("cns_control.utils.stop_live_acquisitions") as stop:
        order.attach_mock(stop, "stop")
        with pytest.raises(RuntimeError, match="end of test"):
            HardwareWidget.disconnect(widget)
    assert order.mock_calls == [call.stop(core), call.disconnect()]


def test_disconnect_keeps_connection_when_live_stop_fails():
    widget = SimpleNamespace(
        core=Mock(), main_window=None, _acquisition_controller=None, status=Mock(),
    )
    core = widget.core
    with patch("cns_control.utils.stop_live_acquisitions", side_effect=RuntimeError("busy")):
        HardwareWidget.disconnect(widget)
    assert widget.core is core
    core.unloadAllDevices.assert_not_called()
    assert "disconnect failed" in widget.status.setText.call_args.args[0]


def test_running_mda_is_stopped_before_disconnect_is_retried():
    widget = SimpleNamespace(
        core=Mock(), _acquisition_controller=Mock(), status=Mock(),
    )
    widget.core.mda.is_running.return_value = True
    with patch("cns_control.utils.stop_live_acquisitions") as stop:
        HardwareWidget.disconnect(widget)
    widget._acquisition_controller.stop.assert_called_once_with()
    widget._acquisition_controller.disconnect.assert_not_called()
    stop.assert_not_called()
