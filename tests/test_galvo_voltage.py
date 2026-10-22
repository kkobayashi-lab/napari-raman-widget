from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtWidgets import QDoubleSpinBox

from napari_raman_widget.widget import HardwareWidget


def make_widget():
    writes, statuses = [], []
    controls = [QDoubleSpinBox(), QDoubleSpinBox()]
    for control in controls:
        control.setDecimals(4)
        control.setRange(-1.6, 1.6)
    widget = SimpleNamespace(
        core=SimpleNamespace(mda=SimpleNamespace(is_running=lambda: False)),
        daq=SimpleNamespace(set_galvo_position=lambda v: writes.append(v.copy())),
        galvo_x_input=controls[0],
        galvo_y_input=controls[1],
        status=SimpleNamespace(setText=statuses.append),
        _pt_to_volts=lambda pt: np.array([[0.2345, -0.6789]]),
    )
    return widget, writes, statuses


def test_calibrated_default_and_manual_apply():
    widget, writes, statuses = make_widget()
    HardwareWidget._reset_galvo_voltage_inputs(widget)
    assert widget.galvo_x_input.value() == 0.2345
    assert widget.galvo_y_input.value() == -0.6789
    assert writes == []
    widget.galvo_x_input.setValue(0.1234)
    assert writes == []
    HardwareWidget.apply_galvo_voltage(widget)
    np.testing.assert_allclose(writes, [[0.1234, -0.6789]])
    assert "position set" in statuses[-1]
    widget._pt_to_volts = lambda pt: np.array([[-0.1, 0.2]])
    HardwareWidget._reset_galvo_voltage_inputs(widget)
    assert widget.galvo_x_input.value() == -0.1
    assert widget.galvo_y_input.value() == 0.2
    assert len(writes) == 1


@pytest.mark.parametrize("running", [True, lambda: True])
def test_active_mda_blocks_manual_voltage(running):
    widget, writes, statuses = make_widget()
    widget.core.mda.is_running = running
    HardwareWidget.apply_galvo_voltage(widget)
    assert writes == []
    assert "stop the active MDA" in statuses[-1]


@pytest.mark.parametrize("voltage", [float("nan"), 1.7, -1.7])
def test_invalid_voltage_never_reaches_daq(voltage):
    widget, writes, statuses = make_widget()
    widget.galvo_x_input = SimpleNamespace(value=lambda: voltage)
    HardwareWidget.apply_galvo_voltage(widget)
    assert writes == []
    assert "failed" in statuses[-1]


def test_disconnected_apply_is_reported():
    widget, writes, statuses = make_widget()
    widget.daq = None
    HardwareWidget.apply_galvo_voltage(widget)
    assert writes == []
    assert "not connected" in statuses[-1]
