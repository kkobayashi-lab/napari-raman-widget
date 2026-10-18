import numpy as np
import pytest
from unittest.mock import Mock

from napari_raman_widget.plate_focus import PlateFocusCalibration
from napari_raman_widget.plate_focus_widget import PlateFocusWidget
from napari_raman_widget.wellplate_calibration import FocusWellPlatePlan, plate_plan_model


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_degree_selection_and_prediction(degree):
    x, y = np.meshgrid(np.linspace(-2, 2, 6), np.linspace(-2, 2, 6))
    z = 100 + (2*x + y if degree else 0*x)
    if degree >= 2:
        z += x*x + 2*x*y
    if degree >= 3:
        z += x**3 - y**3
    samples = np.column_stack([x.ravel()*1000+1e6, y.ravel()*1000-2e6, z.ravel()])
    calibration = PlateFocusCalibration(samples=tuple(map(tuple, samples)), enabled=True)
    fit = calibration.fit()
    assert fit.degree == degree
    np.testing.assert_allclose(calibration.heights(samples[:, :2]), z.ravel(), atol=1e-8)
    plan = FocusWellPlatePlan(plate="96-well", a1_center_xy=(0, 0), focus_height=calibration)
    data = plan.model_dump(mode="json")
    assert plate_plan_model(data).model_validate(data).focus_height == calibration


def test_invalid_geometry_and_extrapolation():
    with pytest.raises(ValueError, match="single line"):
        PlateFocusCalibration(samples=[(i, i, i) for i in range(5)]).fit()
    calibration = PlateFocusCalibration(samples=[(0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 4)])
    with pytest.raises(ValueError, match="coverage"):
        calibration.heights([(2, 2)])
    with pytest.raises(ValueError, match="finite"):
        PlateFocusCalibration(samples=[(0, 1, float("nan"))])


def test_capture_replace_remove_and_acquisition_guard():
    owner = Mock()
    owner._raman_mda_pending = False
    owner.core.mda.is_running.return_value = False
    owner.core.getXPosition.return_value = 10
    owner.core.getYPosition.return_value = 20
    owner.core.getPosition.return_value = 30
    widget = PlateFocusWidget(owner)
    widget.record()
    owner.core.getPosition.return_value = 31
    widget.record()
    assert widget.samples == [(10, 20, 31)]
    owner.core.mda.is_running.return_value = True
    owner.core.getXPosition.return_value = 40
    widget.record()
    assert len(widget.samples) == 1
    assert "finish" in widget.status.text()
    widget.table.selectRow(0)
    widget.remove()
    assert widget.samples == []
    widget.close()
