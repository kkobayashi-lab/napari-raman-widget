from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from napari_raman_widget.well_boundary import (
    boundary_to_stage, detect_boundary, fit_well_boundary,
)
from napari_raman_widget.well_boundary_widget import BoundaryWellWidget


def wall_image():
    y, x = np.indices((200, 240))
    return np.where(x < 45 + y*0.12, 0, 160).astype(np.uint16)


def test_user_boundary_image_follows_interface_not_dark_region_or_cells():
    image = np.asarray(Image.open(Path(__file__).parent / "fixtures/well_boundary.png").convert("L"))
    points = detect_boundary(image)
    # Manually inspected landmarks on the supplied screenshot, including its notch.
    for y, x in [(50, 118), (150, 100), (250, 85), (285, 74), (350, 75), (450, 62)]:
        point = points[np.argmin(abs(points[:, 0]-y))]
        assert point[1] == pytest.approx(x, abs=3)
    assert np.ptp(points[:, 0]) > 480
    assert points[:, 1].max() < 130


@pytest.mark.parametrize("turns", range(4))
def test_boundary_detection_all_orientations(turns):
    image = np.rot90(wall_image(), turns)
    points = detect_boundary(image)
    assert len(points) > 150
    # Points must lie at a strong intensity interface, not a dark skeleton.
    iy, ix = np.rint(points).astype(int).T
    neighborhood = np.stack([image[iy+dy, ix+dx] for dy in [-2, 2] for dx in [-2, 2]])
    assert (np.ptp(neighborhood, axis=0) > 100).mean() > 0.95


def test_flat_image_and_tiny_roi_rejected():
    with pytest.raises(ValueError):
        detect_boundary(np.zeros((100, 100)))
    with pytest.raises(ValueError):
        detect_boundary(wall_image(), (0, 0, 4, 4))


def test_pixel_transform_sign_rotation_and_nonlinear_terms():
    coefficients = np.array([[33, 44], [-2, 0], [0, 3], [0.1, 0], [0, 0], [0, 0.2]])
    result = boundary_to_stage([[22, 13]], (100, 200, 4), [10, 20], (coefficients, 2))
    assert result[0] == pytest.approx([103.6, 189.2])


def arcs(angles=(0, 120, 240)):
    center = np.array([12000, -3000])
    captures = []
    for angle in angles:
        theta = np.deg2rad(np.linspace(angle-3, angle+3, 100))
        points = center + 3000*np.column_stack([np.cos(theta), np.sin(theta)])
        captures.append(SimpleNamespace(stage_xy=points))
    return captures


def test_fit_center_and_reject_clustered_or_wrong_radius():
    center, radius, rms = fit_well_boundary(arcs(), 3000)
    assert center == pytest.approx([12000, -3000], abs=0.01)
    assert radius == pytest.approx(3000, abs=0.01)
    assert rms < 0.01
    with pytest.raises(ValueError, match="clustered"):
        fit_well_boundary(arcs((0, 10, 20)), 3000)
    with pytest.raises(ValueError, match="diameter"):
        fit_well_boundary(arcs(), 1000)


@pytest.fixture
def widget(monkeypatch):
    monkeypatch.setattr("cns_control.utils.stop_live_acquisitions", lambda core: None)
    core = Mock()
    core.mda.is_running.return_value = False
    core.getXYPosition.return_value = (100, 200)
    core.getPosition.return_value = 30
    core.getImageWidth.return_value = 240
    core.getImageHeight.return_value = 200
    core.getROI.return_value = (0, 0, 240, 200)
    core.getImage.return_value = wall_image()
    owner = SimpleNamespace(core=core, _ensure_current_vandermonde=lambda: "10x",
                            vandermonde=(np.array([[0, 0], [0, 1], [1, 0]]), 1),
                            sel_cx_input=Mock(value=lambda: 120),
                            sel_cy_input=Mock(value=lambda: 100))
    widget = BoundaryWellWidget(owner, "A1", 3000)
    yield widget
    widget.close()
    widget.deleteLater()


def test_capture_review_accept_retake_and_owned_image(widget):
    widget.capture()
    assert widget.pending is not None
    assert widget.pending.xyz == (100, 200, 30)
    assert widget.accepted == []
    assert widget.wellCenter() is None
    widget.owner.core.getImage.return_value[:] = 0
    assert widget.pending.image.max() == 160  # saved buffer does not mutate
    widget.accept()
    assert len(widget.accepted) == 1
    assert widget.pending is None
    widget.owner.core.getImage.return_value = wall_image()
    widget.capture()
    widget.owner.core.snapImage.side_effect = RuntimeError("camera failed")
    widget.capture()  # failed retake must invalidate previous pending detection
    assert widget.pending is None
    assert not widget.accept_button.isEnabled()
    assert len(widget.accepted) == 1


def test_failed_detection_can_be_retaken_but_not_accepted(widget):
    widget.owner.core.getImage.return_value = np.zeros((200, 240))
    widget.capture()
    assert widget.pending is not None
    assert widget.pending.stage_xy is None
    assert not widget.accept_button.isEnabled()
    assert widget.retake_button.isEnabled()
    widget.accept()
    assert not widget.accepted


def test_capture_rejects_mda_and_stage_motion(widget):
    widget.owner.core.mda.is_running.return_value = True
    widget.capture()
    widget.owner.core.snapImage.assert_not_called()
    widget.owner.core.mda.is_running.return_value = False
    widget.owner.core.getXYPosition.side_effect = [(0, 0), (1, 0)]
    widget.capture()
    assert widget.pending is None
    assert "moved" in widget.status.text()


def test_fit_requires_explicit_apply_and_removal_invalidates(widget):
    widget.accepted = arcs()
    widget._refit()
    assert widget.wellCenter() is None
    widget.apply_center()
    assert widget.wellCenter() == pytest.approx([12000, -3000])
    widget.accepted.pop()
    widget._refit()
    assert widget.wellCenter() is None


def test_native_plate_selection_keeps_captures_associated_with_well(widget):
    from pymmcore_widgets.hcs._plate_calibration_widget import PlateCalibrationWidget
    from napari_raman_widget.well_boundary_widget import install_boundary_calibration

    plate = PlateCalibrationWidget(mmcore=widget.owner.core)
    plate.setValue("96-well")
    wizard = SimpleNamespace(calibration_page=SimpleNamespace(widget=plate))
    install_boundary_calibration(wizard, widget.owner)
    first = plate._current_calibration_widget()
    assert isinstance(first, BoundaryWellWidget)
    first.capture()
    first.accept()
    plate._plate_view.setSelectedIndices([(0, 1)])
    second = plate._current_calibration_widget()
    assert second is not first
    assert not second.accepted
    plate._plate_view.setSelectedIndices([(0, 0)])
    assert plate._current_calibration_widget() is first
    assert len(first.accepted) == 1
    first.fit = ((123., 456.), 3000, 0)
    first.apply_center()
    assert plate._calibrated_wells[(0, 0)] == (123, 456)
    first.remove()
    assert (0, 0) not in plate._calibrated_wells
    plate.setValue("24-well")
    assert not plate._current_calibration_widget().accepted
    plate.close()
    plate.deleteLater()


def test_boundary_centers_feed_existing_plate_mapping(widget):
    from pymmcore_widgets.hcs._hcs_wizard import HCSWizard
    from pymmcore_plus import CMMCorePlus
    from napari_raman_widget.wellplate_calibration import extend_plate_mapping
    from napari_raman_widget.well_boundary_widget import install_boundary_calibration

    wizard = HCSWizard(mmcore=CMMCorePlus.instance())
    extend_plate_mapping(wizard)
    install_boundary_calibration(wizard, widget.owner)
    plate = wizard.calibration_page.widget
    plate.setValue("96-well")
    for index, center in [((0, 0), (1000, 2000)), ((0, 11), (-98000, 2000)),
                          ((7, 0), (1000, 65000))]:
        plate._plate_view.setSelectedIndices([index])
        current = plate._current_calibration_widget()
        current.fit = (center, 3000, 0)
        current.apply_center()
    assert wizard._calibrated
    plan = plate.value()
    np.testing.assert_allclose(plan.stage_affine_um, [[0, -9000, 1000], [9000, 0, 2000]], atol=1e-8)
    wizard.close()
    wizard.deleteLater()


def start_guidance(widget):
    widget.capture()
    widget.accept()
    widget.guide.choose.setChecked(True)
    widget.guide.clicked(SimpleNamespace(inaxes=widget.ax, button=1, xdata=180, ydata=100))
    core = widget.owner.core
    core.deviceBusy.return_value = False
    core.setXYPosition.side_effect = lambda x, y: setattr(core.getXYPosition, "return_value", (x, y))
    return widget.guide, core


def test_inward_direction_uses_calibrated_stage_axes(widget):
    from napari_raman_widget.well_boundary_guidance import initial_geometry, sampling_target
    guide, _ = start_guidance(widget)
    seed = guide.seed
    center, angle, _ = initial_geometry(seed, [180, 100], 3000)
    # Camera x maps to negative stage X in this calibration.
    assert center[0] < seed.xyz[0]-2500
    opposite, _ = sampling_target(center, 3000, angle, 2)
    assert np.linalg.norm(opposite-seed.stage_xy.mean(axis=0)) == pytest.approx(6000, abs=10)
    seed.model = (np.array([[0, 0], [-2, 0], [0, 3]]), 1)
    center, _, _ = initial_geometry(seed, [180, 100], 3000)
    assert center[1] < seed.xyz[1]-2500


def test_guided_move_is_xy_only_waits_for_stage_and_limits_search(widget):
    guide, core = start_guidance(widget)
    guide.move_next()
    assert guide.moving
    assert not widget.capture_button.isEnabled()
    core.setPosition.assert_not_called()
    core.deviceBusy.return_value = True
    guide.poll()
    assert guide.moving
    core.deviceBusy.return_value = False
    guide.poll()
    assert guide.arrived and not guide.moving
    assert widget.capture_button.isEnabled()
    for _ in range(6):
        guide.search_next()
        guide.poll()
    count = core.setXYPosition.call_count
    guide.search_next()
    assert core.setXYPosition.call_count == count == 7
    assert not guide.search.isEnabled()
    assert len(widget.accepted) == 1


def test_guidance_blocks_changed_settings_mda_and_large_moves(widget):
    guide, core = start_guidance(widget)
    core.mda.is_running.return_value = True
    guide.move_next()
    core.setXYPosition.assert_not_called()
    core.mda.is_running.return_value = False
    core.getProperty.return_value = "changed binning"
    guide.move_next()
    core.setXYPosition.assert_not_called()
    core.getProperty.return_value = guide.seed.metadata["binning"]
    guide.move(np.array([1e8, 1e8]))
    core.setXYPosition.assert_not_called()


def test_stop_and_timeout_keep_pending_measurements_out_of_fit(widget):
    guide, core = start_guidance(widget)
    guide.move_next()
    guide.stop_motion()
    core.stop.assert_called_once_with(core.getXYStageDevice())
    assert not guide.moving and not guide.arrived
    guide.move_next()
    guide.deadline = 0
    guide.poll()
    assert not guide.moving
    assert "timed out" in guide.info.text()
    assert len(widget.accepted) == 1


def test_independent_check_excluded_from_fit_and_required_for_apply(widget):
    from copy import deepcopy
    guide, core = start_guidance(widget)
    guide.index = 4
    guide.center = np.array([100, 200])
    widget.fit = ((100, 200), 3000, 0)
    guide.angle = 0
    guide.move_next()
    guide.poll()
    check = deepcopy(guide.seed)
    theta = np.deg2rad(np.linspace(43, 47, 100))
    check.stage_xy = [100, 200] + 3000*np.column_stack([np.cos(theta), np.sin(theta)])
    widget.pending = check
    widget.apply_center()
    assert widget.wellCenter() is None
    old_fit = widget.fit
    # Deepcopy mock-valued metadata is not an actual camera setting change.
    check.metadata = dict(guide.seed.metadata)
    widget.accept()
    assert guide.verified
    assert widget.pending is None
    assert len(widget.accepted) == 1
    assert widget.fit is old_fit
    widget.apply_center()
    assert widget.wellCenter() == (100, 200)


def test_incorrect_check_remains_pending(widget):
    from copy import copy
    guide, core = start_guidance(widget)
    guide.index, guide.angle = 4, 0
    widget.fit = ((100, 200), 3000, 0)
    guide.move_next()
    guide.poll()
    check = copy(guide.seed)
    check.stage_xy = np.tile([100+3200/np.sqrt(2), 200+3200/np.sqrt(2)], (100, 1))
    widget.pending = check
    widget.accept()
    assert not guide.verified
    assert widget.pending is check
    assert "disagrees" in guide.info.text()


def test_full_guided_collection_refines_targets_and_checks_fifth_view(widget):
    from copy import copy
    guide, core = start_guidance(widget)
    center, radius, angle = guide.center.copy(), widget.radius, guide.angle
    # Exact synthetic circle measurements, in physical stage coordinates.
    def view_at(theta):
        view = copy(guide.seed)
        angles = np.linspace(theta-0.03, theta+0.03, 80)
        view.stage_xy = center + radius*np.column_stack([np.cos(angles), np.sin(angles)])
        return view
    widget.accepted[0].stage_xy = view_at(angle).stage_xy
    for index in (1, 2, 3):
        guide.move_next()
        guide.poll()
        assert guide.arrived
        widget.pending = view_at(angle+index*np.pi/2)
        widget.accept()
        assert len(widget.accepted) == index+1
        assert not guide.arrived
    np.testing.assert_allclose(widget.fit[0], center, atol=1e-6)
    assert not widget.apply_button.isEnabled()
    guide.move_next()
    guide.poll()
    widget.pending = view_at(angle+np.pi/4)
    widget.accept()
    assert guide.verified
    assert len(widget.accepted) == 4
    assert widget.apply_button.isEnabled()
    core.setPosition.assert_not_called()
