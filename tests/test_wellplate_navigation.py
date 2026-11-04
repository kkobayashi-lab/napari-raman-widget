from unittest.mock import Mock

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget
from useq import WellPlatePlan
from pymmcore_widgets.useq_widgets._well_plate_widget import DATA_POSITION

from napari_raman_widget.wellplate_calibration import CalibratedWellPlatePlan
from napari_raman_widget.wellplate_navigation import WellPlateNavigationPanel


@pytest.fixture
def panel():
    owner = QWidget()
    owner.core = Mock()
    owner.core.mda.is_running.return_value = False
    widget = WellPlateNavigationPanel(owner)
    yield widget
    owner.close()
    owner.deleteLater()


@pytest.mark.parametrize("calibrated", [False, True])
def test_loaded_centers_and_edges_follow_stage_transform(panel, tmp_path, calibrated):
    plan = WellPlatePlan(plate="96-well", a1_center_xy=(100, 200), rotation=25)
    if calibrated:
        plan = CalibratedWellPlatePlan(**plan.model_dump(),
            stage_affine_um=((0, -9000, 100), (9000, 0, 200)))
    path = tmp_path / "plate.json"
    path.write_text(plan.model_dump_json())
    panel.load_config(path)
    assert panel.plan.all_well_positions == plan.all_well_positions
    center = plan.all_well_positions[0]
    item = panel.view._well_items[(0, 0)]
    assert item.data(DATA_POSITION) == center
    # Every preset lies at the center or on the physical well boundary.
    positions = [item.data(DATA_POSITION) for item in panel.view._well_edge_spots]
    positions = [p for p in positions if p.plate_row == 0 and p.plate_col == 0]
    assert len(positions) >= 4
    radii = [np.hypot(p.x - center.x, p.y - center.y) for p in positions]
    assert max(radii) == pytest.approx(3200)
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication
    panel.owner.show()
    panel.show()
    panel.resize(700, 500)
    QApplication.processEvents()
    marker = panel.view._well_edge_spots[1]
    target = marker.data(DATA_POSITION)
    click = panel.view.mapFromScene(marker.rect().center())
    QTest.mouseDClick(panel.view.viewport(), Qt.LeftButton, pos=click)
    panel.owner.core.setXYPosition.assert_called_once_with(target.x, target.y)
    panel.owner.core.setPosition.assert_not_called()


def test_invalid_reload_disables_old_targets(panel, tmp_path):
    path = tmp_path / "plate.json"
    path.write_text(WellPlatePlan(plate="96-well", a1_center_xy=(1, 2)).model_dump_json())
    panel.load_config(path)
    target = panel.plan.all_well_positions[0]
    path.write_text('{"invalid": true}')
    panel.load_config(path)
    assert panel.plan is None
    assert not panel.view.isEnabled()
    panel.move_to_position(target)
    panel.owner.core.setXYPosition.assert_not_called()


def test_live_highlight_follows_stage_readback(panel, tmp_path):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest

    plan = CalibratedWellPlatePlan(plate="96-well", a1_center_xy=(100, 200),
        stage_affine_um=((0, -9000, 100), (9000, 0, 200)))
    path = tmp_path / "plate.json"
    path.write_text(plan.model_dump_json())
    a1, a2 = plan.all_well_positions[:2]
    core = panel.owner.core
    core.getXYPosition.return_value = (a1.x, a1.y)
    panel.load_config(path)
    panel.owner.show()
    first = panel.view._well_items[(0, 0)]
    second = panel.view._well_items[(0, 1)]
    assert first.brush().color() == Qt.green
    panel.move_to_position(a2)
    assert first.brush().color() == Qt.green
    core.getXYPosition.return_value = (a2.x + 3200, a2.y)
    QTest.qWait(400)
    assert second.brush().color() == Qt.green
    assert first.brush().color() == Qt.transparent
    core.getXYPosition.return_value = ((a1.x + a2.x) / 2, a1.y)
    panel._refresh_stage_well()
    assert second.brush().color() == Qt.transparent


def test_navigation_rejects_disconnected_and_acquiring_stage(panel, tmp_path):
    path = tmp_path / "plate.json"
    path.write_text(WellPlatePlan(plate="96-well", a1_center_xy=(1, 2)).model_dump_json())
    panel.load_config(path)
    target = panel.plan.all_well_positions[0]
    core = panel.owner.core
    core.mda.is_running.return_value = True
    panel.move_to_position(target)
    core.setXYPosition.assert_not_called()
    panel.owner.core = None
    panel.move_to_position(target)
    assert "Connect an XY stage" in panel.status.text()
