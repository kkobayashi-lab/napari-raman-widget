from types import SimpleNamespace
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pytest
from qtpy import QtGui, QtWidgets
from qtpy.QtWidgets import QApplication, QDockWidget, QLabel, QPushButton, QWidget
from pymmcore_plus import CMMCorePlus
from useq import MDASequence, Position

from napari_raman_widget.unified_mda import (
    apply_prepared_positions, build_extension_controls, install_unified_mda,
)


def test_automatic_autofocus_controls_persist(owner):
    from napari_raman_widget.unified_mda import read_extensions
    widget, panel, _app = owner
    widget.mda_af_axes["p"].setChecked(True)
    widget.mda_auto_af_check.setChecked(True)
    widget.mda_auto_af_radius.setValue(35)
    assert not widget.choose_af_references_btn.isEnabled()
    saved = panel.value()
    widget.mda_auto_af_check.setChecked(False)
    assert widget.choose_af_references_btn.isEnabled()
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    ext = read_extensions(widget)
    assert ext.autofocus.automatic_positions
    assert ext.autofocus.automatic_min_radius_px == 35
    assert widget.mda_auto_af_radius.isEnabled()


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def owner(qt_app):
    app = qt_app
    for name in ("QAction", "QActionGroup", "QUndoCommand", "QUndoStack"):
        if not hasattr(QtGui, name):
            setattr(QtGui, name, getattr(QtWidgets, name))
    from napari_micromanager._gui_objects._mda_widget import MultiDWidget
    core = CMMCorePlus()
    core.loadSystemConfiguration(str(Path(__file__).parents[2] / "raman-mda-engine/tests/test-config.cfg"))
    widget = QWidget()
    widget.core = core
    widget.browse_cellpose_model = lambda *_: None
    widget.browse_tracking_cfg = lambda: None
    widget.open_acquisition_viewer = lambda: None
    widget._toggle_autofocus_fields = lambda *_: None
    widget._toggle_seg_track_fields = lambda *_: None
    widget._on_raman_mda_finished = lambda *_: None
    core.mda.events.sequenceFinished.connect(widget._on_raman_mda_finished)
    widget.status = QLabel()
    widget.apply_mda_positions_btn = QPushButton()
    widget._raman_mda_pending = False
    widget._scheduled_raman_mda = None
    widget._finish_live_mda_timing = lambda *_: None
    widget.tf_path = QtWidgets.QLineEdit("aiming.json")
    widget.vandermonde_path = "stage.json"
    widget.sel_repeats_input = QtWidgets.QSpinBox()
    widget.sel_repeats_input.setValue(1)
    build_extension_controls(widget)
    window = QWidget()
    old = MultiDWidget(parent=window, mmcore=core)
    dock = QDockWidget("MDA", window)
    dock.setWidget(old)
    window._dock_widgets = {"MDA": dock}
    widget.main_window = window
    panel = install_unified_mda(widget)
    yield widget, panel, app
    widget._acquisition_controller.disconnect()
    panel._disconnect()
    panel.control_btns._disconnect()
    window.close()
    widget.close()
    app.processEvents()


def test_dense_grid_is_listed_in_positions_without_expanding_points(owner):
    from raman_mda_engine.grid_plan import GRID_KEY

    widget, panel, app = owner
    native_table = panel.stage_positions.table()
    grid = {"version": 1, "regions": [
        {"x": [0, 100, 10001], "y": [0, 200, 20001], "z": None},
        {"x": [20, 30, 2], "y": [40, 50, 3], "z": 7},
    ]}
    panel.setValue(MDASequence(channels=["BF"], metadata={GRID_KEY: grid}))
    assert panel.tab_wdg.currentWidget() is panel.stage_positions
    assert panel.stage_positions.isAncestorOf(panel.dense_grid_table)
    assert not panel.dense_positions.isHidden()
    assert native_table.isHidden()
    assert panel.dense_grid_table.rowCount() == 2
    assert panel.dense_grid_table.item(0, 1).text() == "10,001 × 20,001 = 200,030,001"
    assert panel.dense_grid_table.item(0, 4).text() == "At run start"
    assert panel.dense_grid_table.item(1, 4).text() == "7"
    saved = panel.value()
    assert saved.stage_positions == ()
    assert saved.metadata[GRID_KEY] == grid
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert panel.dense_grid_table.rowCount() == 2
    assert panel.value().metadata[GRID_KEY] == grid
    panel.lock(True)
    assert not panel.detach_grid_btn.isEnabled()
    panel.lock(False)
    assert panel.detach_grid_btn.isEnabled()
    panel.detach_grid_btn.click()
    assert GRID_KEY not in panel.value().metadata
    assert panel.dense_positions.isHidden()
    assert not native_table.isHidden()
    assert panel.stage_positions.af_per_position.isHidden()
    panel.setValue(MDASequence(stage_positions=[Position(x=3, y=4)], channels=["BF"]))
    assert panel.value().stage_positions[0].x == 3


def test_install_is_idempotent_and_extension_settings_round_trip(owner):
    widget, panel, app = owner
    assert install_unified_mda(widget) is panel
    sequence = MDASequence(channels=[{"config": "BF", "exposure": 10}],
                           metadata={"external": {"preserved": True}})
    panel.setValue(sequence)
    widget.mda_segment_every_input.setValue(3)
    widget.mda_delay_input.setValue(12)
    saved = panel.value()
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert widget.mda_segment_every_input.value() == 3
    assert widget.mda_delay_input.value() == 12
    assert panel.value().metadata["external"] == {"preserved": True}


@pytest.mark.parametrize("dense", [False, True])
def test_grid_shutter_checkbox_round_trip_and_event_behavior(owner, dense):
    from raman_mda_engine.grid_plan import GRID_KEY, iter_grid_events

    widget, panel, app = owner
    metadata = {GRID_KEY: {"version": 1, "regions": [
        {"x": [0, 1, 2], "y": [0, 0, 1], "z": 0, "snake_axis": None}]}} if dense else {}
    sequence = MDASequence(
        channels=["BF"], metadata=metadata,
        grid_plan=None if dense else {"rows": 1, "columns": 2},
        time_plan={"loops": 2, "interval": 0},
    )
    panel.setValue(sequence)
    checkbox = panel.keep_shutter_open.leave_open_g
    assert not checkbox.isChecked()
    checkbox.setChecked(True)
    saved = panel.value()
    assert saved.keep_shutter_open_across == (("p",) if dense else ("g",))
    events = list(iter_grid_events(saved))
    assert [event.keep_shutter_open for event in events] == [True, False, True, False]
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert checkbox.isChecked()
    assert panel.value().keep_shutter_open_across == saved.keep_shutter_open_across
    # Toggling grid leaves the existing Z/time shutter choices intact.
    panel.keep_shutter_open.leave_open_z.setChecked(True)
    panel.keep_shutter_open.leave_open_t.setChecked(True)
    checkbox.setChecked(False)
    assert panel.value().keep_shutter_open_across == ("z", "t")
    panel.setValue(sequence)
    assert not checkbox.isChecked()


def test_grid_uses_vandermonde_scale_overlap_and_current_objective(owner):
    widget, panel, app = owner
    widget.vandermonde = (np.array([[12, 13], [0, -0.3], [0.2, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "6"
    panel.setValue(MDASequence(channels=["BF"], grid_plan={"rows": 2, "columns": 3, "overlap": 10}))
    plan = panel.value().grid_plan
    width, height = widget.core.getImageWidth(), widget.core.getImageHeight()
    assert plan.fov_width == pytest.approx(width * 0.2)
    assert plan.fov_height == pytest.approx(height * 0.3)
    points = list(plan)
    xs = sorted(set(point.x for point in points))
    ys = sorted(set(point.y for point in points))
    assert np.diff(xs) == pytest.approx([width * 0.2 * 0.9] * 2)
    assert np.diff(ys) == pytest.approx([height * 0.3 * 0.9])
    assert "objective 6" in panel.grid_calibration_status.text()
    # Model refresh is authoritative even if imported FOV values are different.
    saved = MDASequence.model_validate_json(panel.value().model_dump_json())
    panel.setValue(saved)
    widget.vandermonde = (np.array([[0, 0], [0, 0.4], [0.5, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "7"
    assert panel.value().grid_plan.fov_width == pytest.approx(width * 0.5)
    assert "objective 7" in panel.grid_calibration_status.text()
    # An objective without a model cannot keep the previous objective's scale.
    widget.vandermonde = None
    assert panel.value().grid_plan.fov_width == pytest.approx(width * widget.core.getPixelSizeUm())
    assert "Micro-Manager" in panel.grid_calibration_status.text()


@pytest.mark.parametrize("definition", [
    {"width": 600, "height": 500, "overlap": 20},
    {"left": -300, "right": 300, "bottom": -250, "top": 250, "overlap": 20},
])
def test_grid_area_and_bounds_use_calibrated_field_count(owner, definition):
    widget, panel, app = owner
    widget.vandermonde = (np.array([[0, 0], [0, 0.3], [0.2, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "6"
    panel.setValue(MDASequence(grid_plan=definition))
    plan = panel.value().grid_plan
    expected = MDASequence(grid_plan={
        **definition,
        "fov_width": widget.core.getImageWidth() * 0.2,
        "fov_height": widget.core.getImageHeight() * 0.3,
    }).grid_plan
    assert list(plan) == list(expected)
    # Objective updates refresh the preview without requiring a manual edit.
    widget.vandermonde = (np.array([[0, 0], [0, 0.6], [0.4, 0]]), 1)
    widget.core.events.propertyChanged.emit("Objective", "State", "6")
    assert "Vandermonde" in panel.grid_calibration_status.text()
    assert panel.grid_plan._fov_width == pytest.approx(widget.core.getImageWidth() * 0.4)
    assert len(list(panel.value().grid_plan)) < len(list(plan))


def test_grid_vandermonde_opt_out_is_saved_and_uses_core_calibration(owner):
    widget, panel, app = owner
    widget.vandermonde = (np.array([[0, 0], [0, 0.3], [0.2, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "6"
    panel.setValue(MDASequence(grid_plan={"rows": 2, "columns": 2}))
    panel.grid_use_vandermonde.setChecked(False)
    saved = panel.value()
    assert saved.grid_plan.fov_width == pytest.approx(widget.core.getImageWidth() * widget.core.getPixelSizeUm())
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert not panel.grid_use_vandermonde.isChecked()
    assert panel.value().grid_plan.fov_width == saved.grid_plan.fov_width


def test_applying_selection_preserves_experiment_channels_z_and_time(owner):
    widget, panel, app = owner
    original = MDASequence(channels=[{"config": "BF", "exposure": 23}],
                           z_plan={"relative": [-2, 0, 2]},
                           time_plan={"loops": 5, "interval": 10})
    panel.setValue(original)
    source = SimpleNamespace(name="cell:A", role="cell", _img_shape=(100, 100),
                             get_mda_points=lambda event, transform=True: np.array([[0.5, 0.5]]))
    widget.selection_results = {
        "new_seq": MDASequence(stage_positions=[Position(x=3, y=4)]),
        "sources": [source], "autofocus_p": [], "autofocus_object": "None",
    }
    apply_prepared_positions(widget)
    value = panel.value()
    assert value.channels == original.channels
    assert value.z_plan == original.z_plan
    assert value.time_plan == original.time_plan
    assert len(panel.targets) == 1
    assert value.stage_positions[0].x == 3


def test_legacy_channel_offsets_import_to_native_channels(owner):
    widget, panel, app = owner
    sequence = MDASequence(channels=["BF", {"config": "RM", "exposure": 100}],
                           metadata={"channel_z_offsets_um": {"RM": -5}, "raman": {"z": [0]}})
    panel.setValue(sequence)
    assert panel.value().channels[1].z_offset == -5
    assert widget.mda_rz_input.text() == "0"


def test_unchecked_axes_disable_laser_and_ignore_imported_hardware_plan(owner):
    widget, panel, app = owner
    panel.setValue(MDASequence(channels=["BF"], autofocus_plan={"axes": ("p",)}))
    widget.collector = None
    widget.transformer = None
    widget.mda_af_mode_combo.setCurrentText("laser")
    widget.mda_afp_input.setText("invalid custom positions")
    widget.mda_delay_input.setValue(60)
    controller = widget._acquisition_controller
    with patch.object(widget.core, "isSequenceRunning", return_value=False), \
         patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok) as review:
        controller.run()
    assert controller.pending[0].extensions.autofocus.mode == "off"
    assert "Autofocus: off" in review.call_args.args[2]
    assert widget.mda_af_mode_combo.currentText() == "laser"
    controller.stop()
    panel.setValue(MDASequence(channels=["BF"]))
    widget.mda_af_mode_combo.setCurrentText("laser")
    widget.mda_af_axes["p"].setChecked(True)
    widget.mda_delay_input.setValue(60)
    with patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok):
        controller.run()
    assert controller.pending[0].autofocus_positions == ()
    assert controller.pending[0].extensions.autofocus.mode == "off"
    controller.stop()



def test_delayed_cancel_does_not_create_writer_or_touch_engine(owner):
    widget, panel, app = owner
    widget.collector = None
    widget.transformer = None
    widget.mda_delay_input.setValue(60)
    panel.setValue(MDASequence())
    widget.mda_delay_input.setValue(60)
    controller = widget._acquisition_controller
    previous = widget.core.mda.engine
    with patch.object(widget.core, "isSequenceRunning", return_value=False), \
         patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok):
        controller.run()
    assert controller.pending is not None
    assert controller.writer is None
    assert widget.core.mda.engine is previous
    controller.stop()
    assert controller.pending is None
    assert widget.core.mda.engine is previous


def test_imported_hardware_autofocus_never_dispatches_device_actions(owner):
    from useq import HardwareAutofocus
    widget, panel, app = owner
    widget.collector = None
    widget.transformer = None
    panel.setValue(MDASequence(stage_positions=[Position(x=0, z=0)], channels=["BF"],
                              autofocus_plan={"axes": ("p",)}))
    widget.mda_delay_input.setValue(60)
    controller = widget._acquisition_controller
    assert widget.mda_af_mode_combo.currentText() == "laser"
    assert widget.mda_af_mode_combo.findText("hardware") == -1
    # This simulator config has no autofocus device, so its native editor disables
    # the axes. Supply the plan a configured microscope returns to the controller.
    native_sequence = MDASequence(stage_positions=[Position(x=0, y=0, z=0)],
                                  channels=["BF"], axis_order=("p", "c"),
                                  autofocus_plan={"axes": ("p",)},
                                  metadata=panel.value().metadata)
    with patch.object(panel, "value", return_value=native_sequence), \
         patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok):
        controller.run()
    assert controller.pending is not None
    sequence = controller.pending[0].sequence
    assert not any(isinstance(event.action, HardwareAutofocus) for event in sequence)
    controller.stop()
    assert controller.pending is None



@pytest.mark.parametrize("camera", [False, True])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("frames", [1, 2])
def test_simulated_raman_run_streams_data_and_restores_engine(owner, tmp_path, camera, dense, frames):
    import h5py
    from raman_mda_engine import fakeAcquirer
    widget, panel, app = owner
    widget.core.defineConfig("Channel", "RM", "Dichroic", "Label", "Q505LP")
    widget.collector = fakeAcquirer()
    widget.collector.get_wavenumbers = lambda: np.arange(1340)
    widget.transformer = SimpleNamespace(BF_to_volts=lambda pts, max_volts: np.asarray(pts) * max_volts)
    widget.mm_config = str(Path(__file__).parents[2] / "raman-mda-engine/tests/test-config.cfg")
    widget.sel_cx_input = QtWidgets.QSpinBox()
    widget.sel_cy_input = QtWidgets.QSpinBox()
    widget.sel_r_input = QtWidgets.QSpinBox()
    widget._connect_raman_visualization = lambda engine: None
    widget._start_live_mda_timing = lambda count: None
    channels = [{"config": "RM", "exposure": 100}]
    if camera:
        channels.insert(0, {"config": "BF", "exposure": 1})
    panel.setValue(MDASequence(channels=channels))
    widget.mda_imgp_input.setText("all" if camera else "none")
    source = SimpleNamespace(name="cell:A", role="cell", _img_shape=(512, 512),
                             get_mda_points=lambda event, transform=True: np.array([[0.5, 0.5]]))
    widget.selection_results = {"selection_type": "grid", "new_seq": MDASequence(stage_positions=[Position(x=0, y=0, z=0)]),
                                "sources": [source], "autofocus_p": [], "autofocus_object": "None"}
    if dense:
        widget.selection_results["selection_type"] = "grid"
        widget.selection_results["new_seq"] = MDASequence(metadata={"raman_grid": {
            "version": 1, "regions": [{"x": [0, 1, 2], "y": [0, 0, 1], "z": 0,
                                        "tilt": None, "snake_axis": None}]}})
    widget.mda_repeats_input.setValue(frames)
    if not dense:
        widget.sel_repeats_input.setValue(frames)
    apply_prepared_positions(widget)
    assert widget.mda_repeats_input.value() == frames
    if dense:
        assert panel.value().stage_positions == ()
        assert len(panel.targets[0]["points"]) == 1
        assert "2 virtual positions" in panel.grid_summary.text()
    assert widget.mda_imgp_input.text() == ("all" if camera else "none")
    panel.update_summary()
    assert widget.mda_collection_summary.text().startswith(
        f"{(2 if dense else 1) * frames} spectral collections;")
    widget.mda_pre_acq_check.setChecked(False)
    panel.save_info.setValue({"save_dir": str(tmp_path), "save_name": "run", "format": "tiff-sequence", "should_save": True})
    controller = widget._acquisition_controller
    previous = widget.core.mda.engine
    with patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok):
        controller.run()
    deadline = time.monotonic() + 10
    while controller.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not controller.busy
    assert widget.core.mda.engine is previous
    assert (tmp_path / "run/useq-sequence.json").is_file()
    with h5py.File(tmp_path / "run/raman.h5") as store:
        assert store["raman/spectra"].shape[0] == (2 if dense else 1) * frames
        assert store["raman/points_yx"].shape[0] == (2 if dense else 1) * frames
        assert store.attrs["completion_status"] == "completed"
        if dense:
            saved = MDASequence.model_validate_json((tmp_path / "run/useq-sequence.json").read_text())
            assert saved.stage_positions == ()
            assert saved.metadata["raman_grid"]["regions"][0]["x"][2] == 2
    assert bool(list((tmp_path / "run").rglob("*.tiff"))) == camera
    if dense:
        from napari_raman_widget.large_dataset_viewing import AcquisitionIndex
        acquisition = AcquisitionIndex.build(tmp_path / "run")
        np.testing.assert_allclose(acquisition.stage_xyz_um, [[0, 0, 0], [1, 0, 0]])
        assert len(acquisition.raman_files) == 2
        if not camera:
            from napari_raman_widget.acquisition_viewer import LargeAcquisitionViewerWindow
            viewer = LargeAcquisitionViewerWindow(imaging_folder=str(tmp_path / "run"))
            viewer.index_acquisition()
            assert viewer.acquisition is not None, viewer.status_label.text()
            assert not viewer.channel_combo.isEnabled()
            viewer.show_selection()
            assert viewer.status_label.text().startswith("Showing Raman"), viewer.status_label.text()
            assert viewer.spectrum_ax.lines
            assert viewer.image_ax.get_title() == "No camera measurements saved"
            viewer.close()
        panel.setValue(MDASequence.model_validate_json(panel.value().model_dump_json()))
        assert panel.value().stage_positions == ()
        assert panel.targets[0]["fixed"]
        panel.detach_grid()
        assert "raman_grid" not in panel.value().metadata


@pytest.mark.parametrize("grid", [False, True])
def test_simulated_camera_only_uses_run_folder_without_raman(owner, tmp_path, grid):
    import h5py
    from napari_raman_widget.large_dataset_viewing import AcquisitionIndex

    widget, panel, app = owner
    widget.collector = None
    widget.transformer = None
    widget._start_live_mda_timing = lambda count: None
    panel.setValue(MDASequence(channels=[{"config": "BF", "exposure": 1}],
                              grid_plan={"rows": 1, "columns": 2} if grid else None))
    panel.save_info.setValue({"save_dir": str(tmp_path), "save_name": "camera",
                             "format": "tiff-sequence", "should_save": True})
    controller = widget._acquisition_controller
    previous = widget.core.mda.engine
    with patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok):
        controller.run()
    deadline = time.monotonic() + 10
    while controller.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not controller.busy
    assert widget.core.mda.engine is previous
    assert controller.writer is None
    run = tmp_path / "camera"
    assert (run / "useq-sequence.json").is_file()
    assert list(run.rglob("*.tiff"))
    with h5py.File(run / "raman.h5") as store:
        assert store.attrs["completion_status"] == "completed"
        assert store["raman/events"].shape[0] == 0
    acquisition = AcquisitionIndex.build(run)
    assert len(acquisition.image_keys) == (2 if grid else 1)
    assert acquisition.load_image(acquisition.image_keys[0]).shape == (512, 512)


@pytest.mark.parametrize("fmt", ["ome-zarr", "ome-tiff", "tiff-sequence"])
def test_save_format_is_hidden_and_imported_formats_use_run_folder(owner, tmp_path, fmt):
    widget, panel, app = owner
    panel.save_info.setValue({"save_dir": str(tmp_path), "save_name": "camera",
                             "format": fmt, "should_save": True})
    panel.setValue(MDASequence.model_validate_json(panel.value().model_dump_json()))
    panel.lock(True)
    panel.lock(False)
    combo = panel.save_info._writer_combo
    assert combo.isHidden()
    assert not combo.isEnabled()
    assert combo.currentText() == "tiff-sequence"
    assert panel.save_info.currentPath() == tmp_path / "camera"
    assert panel.save_info.name_label.text() == "Subfolder:"
    assert combo.findText(fmt) >= 0
    assert "one run folder" in panel.output_description.text()


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("camera_positions", ["all", "none", "0"])
def test_grid_handoff_preserves_mda_frames_per_point(owner, dense, camera_positions):
    widget, panel, _app = owner
    widget.mda_repeats_input.setValue(3)
    widget.mda_af_axes["p"].setChecked(True)
    widget.mda_afp_input.setText("none")
    widget.mda_imgp_input.setText(camera_positions)
    source = SimpleNamespace(name="cells", role="cell", _img_shape=(512, 512),
                             get_mda_points=lambda event, transform=True: np.array([[0.5, 0.5]]))
    sequence = MDASequence(stage_positions=[Position(x=0, y=0, z=0)])
    if dense:
        sequence = MDASequence(metadata={"raman_grid": {"version": 1, "regions": [
            {"x": [0, 1, 2], "y": [0, 0, 1], "z": 0, "tilt": None, "snake_axis": None}]}})
    widget.selection_results = {"selection_type": "grid", "new_seq": sequence,
                                "sources": [source], "autofocus_p": [], "autofocus_object": "None"}
    apply_prepared_positions(widget)
    assert widget.mda_repeats_input.value() == 3
    assert widget.mda_af_mode_combo.currentText() == "laser"
    assert widget.mda_af_axes["p"].isChecked()
    assert widget.mda_afp_input.text() == "none"
    assert widget.mda_imgp_input.text() == camera_positions
    assert panel.targets[0]["points"]["0"] == [[0.5, 0.5]]


def test_autofocus_tab_axes_gate_settings_and_round_trip(owner):
    widget, panel, app = owner
    assert panel.af_axis.isHidden()
    assert panel.stage_positions.af_per_position.isHidden()
    assert panel.tab_wdg.tabText(panel.tab_wdg.indexOf(widget._mda_af_panel)) == "Autofocus"
    assert widget.mda_af_mode_combo.count() == 1
    assert widget.mda_af_mode_combo.currentText() == "laser"
    assert not widget._mda_af_settings.isEnabled()
    widget.mda_af_axes["g"].setChecked(True)
    widget.mda_refocus_input.setValue(3)
    assert widget._mda_af_settings.isEnabled()
    saved = panel.value()
    assert saved.autofocus_plan is None
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert widget.mda_af_axes["g"].isChecked()
    assert widget.mda_refocus_input.value() == 3
    widget.mda_af_axes["g"].setChecked(False)
    assert not widget._mda_af_settings.isEnabled()


def test_camera_only_autofocus_references_round_trip_without_cells(owner):
    from napari_raman_widget.autofocus_references import apply_references
    from napari_raman_widget.experiment import identify_positions, prepare_acquisition
    from napari_raman_widget.unified_mda import read_extensions

    widget, panel, app = owner
    panel.setValue(identify_positions(MDASequence(stage_positions=[Position(x=1, y=2, z=0)], channels=["BF"])))
    widget.mda_af_axes["p"].setChecked(True)
    sequence = panel.value()
    source = SimpleNamespace(name="autofocus references", role="autofocus", _img_shape=(512, 512),
                             get_mda_points=lambda event, transform=True: np.array([[0.4, 0.6]]))
    widget._autofocus_reference_selection = (sequence, [source])
    apply_references(widget)
    saved = panel.value()
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    prepared = prepare_acquisition(panel.value(), read_extensions(widget), panel.targets)
    assert prepared.extensions.autofocus.mode == "laser"
    assert tuple(prepared.autofocus_positions) == (0,)
    assert not prepared.raman_enabled
    assert [s.role for s in prepared.sources] == ["autofocus"]
    np.testing.assert_allclose(prepared.sources[0].get_mda_points(next(prepared.sequence.iter_events())), [[0.4, 0.6]])
    assert saved.channels == sequence.channels
    assert [(p.x, p.y, p.z) for p in saved.stage_positions] == [(p.x, p.y, p.z) for p in sequence.stage_positions]


def test_cell_selection_preserves_independent_references_and_focus_policy(owner):
    from napari_raman_widget.autofocus_references import apply_references
    from napari_raman_widget.experiment import identify_positions, prepare_acquisition
    from napari_raman_widget.unified_mda import read_extensions

    widget, panel, app = owner
    panel.setValue(identify_positions(MDASequence(stage_positions=[Position(x=1, y=2)], channels=["BF"])))
    widget.mda_af_axes["p"].setChecked(True)
    widget.mda_refocus_input.setValue(3)
    sequence = panel.value()
    focus = SimpleNamespace(name="autofocus references", role="autofocus", _img_shape=(100, 100),
                            get_mda_points=lambda event, transform=True: np.array([[0.3, 0.7]]))
    widget._autofocus_reference_selection = (sequence, [focus])
    apply_references(widget)
    cell = SimpleNamespace(name="cells", role="cell", _img_shape=(100, 100),
                           get_mda_points=lambda event, transform=True: np.array([[0.5, 0.5]]))
    widget.selection_results = {"new_seq": sequence.replace(stage_positions=[sequence.stage_positions[0]] * 2),
                                "sources": [cell], "autofocus_object": "None", "autofocus_p": []}
    apply_prepared_positions(widget)
    apply_prepared_positions(widget)
    prepared = prepare_acquisition(panel.value(), read_extensions(widget), panel.targets)
    assert prepared.extensions.autofocus.every == 3
    assert prepared.extensions.autofocus.axes == ("p",)
    assert tuple(prepared.autofocus_positions) == (0, 1)
    assert [s.role for s in prepared.sources] == ["cell", "autofocus"]
    for p in (0, 1):
        from useq import MDAEvent
        np.testing.assert_allclose(prepared.sources[1].get_mda_points(MDAEvent(index={"p": p})), [[0.3, 0.7]])


@pytest.mark.parametrize("failure", [False, True])
def test_reference_preview_uses_camera_and_restores_engine_and_preset(owner, failure):
    from napari_raman_widget.autofocus_references import choose_references
    from napari_raman_widget.experiment import identify_positions

    widget, panel, app = owner
    widget.core.defineConfig("Channel", "RM", "Dichroic", "Label", "Q505LP")
    panel.setValue(identify_positions(MDASequence(stage_positions=[Position(x=0, y=0, z=0)],
                                                channels=["RM", "BF"], time_plan={"loops": 3, "interval": 2})))
    original = panel.value()
    previous = widget.core.mda.engine
    widget.default_engine = previous
    widget._prepare_for_selection = lambda: None
    widget.viewer = SimpleNamespace(layers=[])
    widget.core.setConfig("Channel", "RM")
    def acquire(preview):
        if failure:
            raise RuntimeError("preview failed")
        widget.viewer.layers.append(SimpleNamespace(metadata={"useq_sequence": preview, "acquisition_complete": True}))
        return SimpleNamespace(is_alive=lambda: False)
    with patch.object(widget.core, "run_mda", side_effect=acquire) as run, \
         patch("qtpy.QtWidgets.QInputDialog.getItem", return_value=("Acquire configured imaging", True)), \
         patch("napari_raman_widget.autofocus_references.ReferenceEditor") as create:
        if failure:
            with pytest.raises(RuntimeError, match="preview failed"):
                choose_references(widget)
            create.assert_not_called()
        else:
            choose_references(widget)
            create.assert_called_once()
    preview = run.call_args.args[0]
    assert [c.config for c in preview.channels] == ["BF"]
    assert preview.time_plan == original.time_plan
    assert widget.core.mda.engine is previous
    assert widget.core.getCurrentConfig("Channel") == "RM"
    assert panel.value().channels == original.channels
    assert panel.value().time_plan == original.time_plan
    assert panel.value().stage_positions == original.stage_positions


def test_simulated_camera_only_run_autofocuses_without_cell_layers(owner, tmp_path):
    from raman_mda_engine import RamanEngine, fakeAcquirer
    from napari_raman_widget.autofocus_references import apply_references
    from napari_raman_widget.experiment import identify_positions

    widget, panel, app = owner
    widget.collector = fakeAcquirer()
    widget.transformer = SimpleNamespace(BF_to_volts=lambda pts, max_volts: np.asarray(pts) * max_volts)
    widget.vandermonde = (np.zeros((1, 2)), 0)
    widget._ensure_current_vandermonde = lambda: "1"
    widget.mm_config = str(Path(__file__).parents[2] / "raman-mda-engine/tests/test-config.cfg")
    widget.sel_cx_input = QtWidgets.QSpinBox()
    widget.sel_cy_input = QtWidgets.QSpinBox()
    widget.sel_r_input = QtWidgets.QSpinBox()
    widget._connect_raman_visualization = lambda engine: None
    widget._start_live_mda_timing = lambda count: None
    panel.setValue(identify_positions(MDASequence(stage_positions=[Position(x=0, y=0, z=0)],
                                                channels=[{"config": "BF", "exposure": 1}])))
    focus = SimpleNamespace(name="autofocus references", role="autofocus", _img_shape=(512, 512),
                            get_mda_points=lambda event, transform=True: np.array([[0.5, 0.5]]))
    widget._autofocus_reference_selection = (panel.value(), [focus])
    apply_references(widget)
    widget.mda_af_axes["p"].setChecked(True)
    panel.save_info.setValue({"save_dir": str(tmp_path), "save_name": "focus", "format": "tiff-sequence", "should_save": True})
    previous = widget.core.mda.engine
    with patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok), \
         patch.object(RamanEngine, "autofocus_w_raman", return_value=(0, 0, None)) as focus_run, \
         patch.object(RamanEngine, "record_raman") as collect:
        widget._acquisition_controller.run()
        deadline = time.monotonic() + 10
        while widget._acquisition_controller.busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
    focus_run.assert_called_once()
    collect.assert_not_called()
    assert not widget._acquisition_controller.busy
    assert widget.core.mda.engine is previous
    assert list((tmp_path / "focus").rglob("*.tiff"))
    import h5py
    with h5py.File(tmp_path / "focus/raman.h5") as store:
        assert "spectra" not in store["raman"]


def test_wellplate_uses_current_model_for_wizard_import_and_positions(owner):
    from useq import WellPlatePlan

    widget, panel, app = owner
    widget.vandermonde = (np.array([[12, 13], [0, -0.3], [0.2, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "6"
    table = panel.stage_positions
    wizard = table._hcs
    imported = WellPlatePlan(
        plate="96-well", a1_center_xy=(1000, 2000), rotation=5,
        selected_wells=((0, 0), (0, 1)),
        well_points_plan={"rows": 1, "columns": 2, "overlap": 10,
                          "fov_width": 999, "fov_height": 999},
    )
    # Materialize useq's cache so the test catches stale generated positions.
    list(imported)
    table.setValue(imported)
    width, height = widget.core.getImageWidth(), widget.core.getImageHeight()
    plan = wizard.value()
    assert plan.well_points_plan.fov_width == pytest.approx(width * 0.2)
    assert plan.well_points_plan.fov_height == pytest.approx(height * 0.3)
    positions = list(panel.value().stage_positions)
    assert len(positions) == 4
    assert positions[1].x - positions[0].x == pytest.approx(width * 0.2 * 0.9)
    assert table._plate_plan.a1_center_xy == (1000, 2000)
    assert table._plate_plan.rotation == 5
    assert "Vandermonde" in panel.wellplate_calibration_status.text()
    # Objective changes refresh both the wizard preview and the position table.
    widget.vandermonde = (np.array([[0, 0], [0, 0.4], [0.5, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "7"
    widget.core.events.propertyChanged.emit("Objective", "State", "7")
    assert wizard.points_plan_page.widget.value().fov_width == pytest.approx(width * 0.5)
    positions = list(panel.value().stage_positions)
    assert positions[1].x - positions[0].x == pytest.approx(width * 0.5 * 0.9)
    assert "Vandermonde" in panel.wellplate_calibration_status.text()
    # A missing calibration must not silently use the core's pixel size.
    widget.vandermonde = None
    assert wizard.value() is None
    with pytest.raises(ValueError, match="current-objective Vandermonde"):
        table.value()


def test_plate_focus_roundtrip_and_generated_z(owner, tmp_path):
    from napari_raman_widget.plate_focus import PlateFocusCalibration
    from napari_raman_widget.wellplate_calibration import FocusWellPlatePlan

    widget, panel, _ = owner
    widget.vandermonde = (np.array([[0, 0], [0, .3], [.2, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "6"
    calibration = PlateFocusCalibration(samples=[
        (x, y, 100+x*.001+y*.002)
        for x in (-20000, 0, 20000) for y in (-20000, 0, 20000)], enabled=True)
    plan = FocusWellPlatePlan(plate="96-well", a1_center_xy=(1000, 2000),
                             selected_wells=((0, 0), (0, 1)), focus_height=calibration)
    table = panel.stage_positions
    table.setValue(plan)
    saved = panel.value()
    assert saved.metadata["plate_focus_height_applied"]
    for p in saved.stage_positions:
        assert p.z == pytest.approx(100+p.x*.001+p.y*.002)
    path = str(tmp_path / "plate.json")
    table._hcs.save(path)
    table._hcs.focus_height_widget.clear()
    table._hcs.load(path)
    assert table._hcs.value().focus_height == calibration
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert panel.value().metadata["plate_focus_height_applied"]
    assert table._plate_plan.focus_height == calibration
    assert table._hcs.focus_height_widget.value() == calibration
    from useq import ZTopBottom
    panel.z_plan.setValue(ZTopBottom(top=10, bottom=0, step=1))
    panel.tab_wdg.setChecked(panel.z_plan, True)
    from napari_raman_widget.experiment import prepare_acquisition
    from napari_raman_widget.unified_mda import read_extensions
    with pytest.raises(ValueError, match="relative Z stacks"):
        prepare_acquisition(panel.value(), read_extensions(widget), [])
    panel.tab_wdg.setChecked(panel.z_plan, False)
    table.setValue(plan.replace(focus_height=calibration.model_copy(update={"enabled": False})))
    assert "plate_focus_height_applied" not in panel.value().metadata


def test_wellplate_xy_locked_z_editable_and_persisted(owner):
    from useq import WellPlatePlan

    widget, panel, _app = owner
    widget.vandermonde = (np.array([[0, 0], [0, 0.3], [0.2, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "6"
    table = panel.stage_positions
    plan = WellPlatePlan(plate="96-well", a1_center_xy=(1000, 2000),
                         selected_wells=((0, 0), (0, 1)))
    table.setValue(plan)
    panel.value()  # Refresh calibrated field size before editing.
    grid = table.table()
    assert not table.include_z.isHidden()
    table.include_z.setChecked(True)
    assert not grid.isColumnHidden(grid.indexOf(table.Z))
    assert not grid.isColumnHidden(grid.indexOf(table._z_btn_col))
    for row in range(grid.rowCount()):
        assert not grid.cellWidget(row, grid.indexOf(table.X)).isEnabled()
        assert not grid.cellWidget(row, grid.indexOf(table.Y)).isEnabled()
        assert grid.cellWidget(row, grid.indexOf(table.Z)).isEnabled()
    grid.cellWidget(0, grid.indexOf(table.Z)).setValue(123.5)
    widget.core.setZPosition(234.5)
    table._set_z_from_core(1)
    saved = panel.value()
    assert [p.z for p in saved.stage_positions] == [123.5, 234.5]
    assert [e.z_pos for e in saved.iter_events()] == [123.5, 234.5]
    panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
    assert table._plate_plan is not None
    assert [p.z for p in panel.value().stage_positions] == [123.5, 234.5]
    assert not grid.cellWidget(0, grid.indexOf(table.X)).isEnabled()
    widget.vandermonde = (np.array([[0, 0], [0, 0.4], [0.5, 0]]), 1)
    widget.core.events.propertyChanged.emit("Objective", "State", "7")
    assert [p.z for p in panel.value().stage_positions] == [123.5, 234.5]
    table.include_z.setChecked(False)
    assert all(p.z is None for p in panel.value().stage_positions)


def test_cellvis_plate_default_and_test_well_radius(owner):
    from useq import WellPlate, WellPlatePlan
    from napari_raman_widget.grid_calibration import CELLVIS_96_PLATE

    widget, panel, app = owner
    wizard = panel.stage_positions._hcs
    plate = wizard.plate_page.widget.value().plate
    assert wizard.plate_page.widget.plate_name.currentText() == CELLVIS_96_PLATE
    assert (plate.rows, plate.columns) == (8, 12)
    assert plate.well_spacing == (9.0, 9.0)
    assert plate.well_size == (6.0, 6.0)
    assert WellPlate.from_str("96-well").well_size == (6.4, 6.4)
    # Test the actual Test Well callback without moving hardware.
    calibration = wizard.calibration_page.widget
    center = (1200, 3400)
    calibration._current_calibration_widget().setWellCenter(center)
    destinations = []
    with patch.object(calibration, "_move_to_xy_position", destinations.append):
        calibration._move_to_test_position()
    assert len(destinations) == 1
    point = destinations[0]
    assert np.hypot(point.x - center[0], point.y - center[1]) == pytest.approx(3000)
    # The custom dimensions survive a saved plan round-trip.
    plan = WellPlatePlan(plate=plate, a1_center_xy=center)
    restored = WellPlatePlan.model_validate_json(plan.model_dump_json())
    assert restored.plate.well_size == (6.0, 6.0)


@pytest.mark.parametrize("column_step,row_step", [
    ((-9000, 66), (-66, -9000)),  # reversed X as in the reported stage geometry
    ((9000, 0), (0, -9000)),
    ((0, 9000), (9000, 0)),  # rotated axes
])
def test_hcs_measured_mapping_preview_acquisition_and_save(owner, tmp_path, column_step, row_step):
    from napari_raman_widget.wellplate_calibration import CalibratedWellPlatePlan
    from pymmcore_widgets.useq_widgets._well_plate_widget import DATA_POSITION

    widget, panel, app = owner
    widget.vandermonde = (np.array([[0, 0], [0, 0.3], [0.2, 0]]), 1)
    widget._ensure_current_vandermonde = lambda: "6"
    wizard = panel.stage_positions._hcs
    calibration = wizard.calibration_page.widget
    origin = np.array([71630, 30830])

    def center(row, col):
        return origin + np.array(row_step) * row + np.array(column_step) * col

    # Exercise the real callback invoked by recording well centers in the wizard.
    for index in ((0, 0), (0, 11), (7, 0)):
        calibration._plate_view.setSelectedIndices([index])
        calibration._current_calibration_widget().setWellCenter(tuple(center(*index)))
    assert wizard._calibrated
    target = center(3, 5)  # D6
    preview = calibration._plate_test._well_items[(3, 5)].data(DATA_POSITION)
    assert (preview.x, preview.y) == pytest.approx(target)
    assert "Column step" in calibration._info.text()
    # The board stays in logical order while click data retains stage coordinates.
    view = calibration._plate_test
    a1 = view._well_items[(0, 0)].rect().center()
    a2 = view._well_items[(0, 1)].rect().center()
    b1 = view._well_items[(1, 0)].rect().center()
    assert a2.x() > a1.x()
    assert b1.y() > a1.y()
    assert a2.y() == a1.y()
    assert b1.x() == a1.x()
    assert len(view._well_edge_spots) == 96 * 5
    markers = [item for item in view._well_edge_spots if item.data(DATA_POSITION).name == "D6"]
    assert len(markers) == 5
    directions = np.column_stack((row_step, column_step)).astype(float)
    directions /= np.linalg.norm(directions, axis=0)
    offsets = [(0, 0), (0, -3000), (0, 3000), (-3000, 0), (3000, 0)]
    for marker, offset in zip(markers, offsets):
        position = marker.data(DATA_POSITION)
        assert (position.x, position.y) == pytest.approx(target + directions @ offset)
        assert "stage XY" in marker.toolTip()
    # Exercise the center and all edge dots in the logical, unflipped layout.
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    wizard.show()
    calibration._tab_wdg.setCurrentIndex(1)
    app.processEvents()
    with patch.object(calibration._mmc, "setXYPosition") as move:
        for marker in markers:
            move.reset_mock()
            click_position = view.mapFromScene(marker.rect().center())
            QTest.mouseDClick(view.viewport(), Qt.LeftButton, pos=click_position)
            position = marker.data(DATA_POSITION)
            move.assert_called_once_with(position.x, position.y)
    wizard.plate_page.widget.setCurrentSelection((3, 5))
    plan = wizard.value()
    assert isinstance(plan, CalibratedWellPlatePlan)
    assert (list(plan)[0].x, list(plan)[0].y) == pytest.approx(target)
    # The actual MDA event must agree with the test view.
    panel.stage_positions.setValue(plan)
    sequence = panel.value()
    event = next(sequence.iter_events())
    assert (event.x_pos, event.y_pos) == pytest.approx(target)
    saved = MDASequence.model_validate_json(sequence.model_dump_json())
    panel.setValue(saved)
    event = next(panel.value().iter_events())
    assert (event.x_pos, event.y_pos) == pytest.approx(target)
    assert isinstance(panel.stage_positions._plate_plan, CalibratedWellPlatePlan)
    # HCS's own Save/Load must also retain the full mapping.
    path = str(tmp_path / "plate.json")
    wizard.save(path)
    wizard.load(path)
    restored = wizard.value()
    assert isinstance(restored, CalibratedWellPlatePlan)
    assert restored.stage_affine_um == plan.stage_affine_um
    selection = wizard.plate_page.widget._view
    a1 = selection._well_items[(0, 0)].rect().center()
    a2 = selection._well_items[(0, 1)].rect().center()
    b1 = selection._well_items[(1, 0)].rect().center()
    assert a2.x() > a1.x() and a2.y() == a1.y()
    assert b1.y() > a1.y() and b1.x() == a1.x()
    assert wizard.plate_page.widget.currentSelection() == ((3, 5),)

    # Changing plate type must discard measured centers from the previous plate.
    wizard.plate_page.widget.plate_name.setCurrentText("24-well")
    assert not calibration._calibrated_wells
    assert not wizard._calibrated
