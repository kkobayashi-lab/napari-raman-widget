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


def test_hardware_autofocus_overrides_laser_requirements_and_review(owner):
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
    assert controller.pending[0].extensions.autofocus.mode == "hardware"
    assert "Hardware autofocus: True; custom autofocus: off" in review.call_args.args[2]
    assert widget.mda_af_mode_combo.currentText() == "laser"
    controller.stop()
    panel.setValue(MDASequence(channels=["BF"]))
    widget.mda_af_mode_combo.setCurrentText("laser")
    with pytest.raises(ValueError, match="autofocus targets"):
        controller.run()


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


def test_native_autofocus_does_not_require_duplicate_mode_selection(owner):
    from useq import HardwareAutofocus
    widget, panel, app = owner
    widget.collector = None
    widget.transformer = None
    panel.setValue(MDASequence(stage_positions=[Position(x=0, z=0)], channels=["BF"],
                              autofocus_plan={"axes": ("p",)}))
    widget.mda_delay_input.setValue(60)
    controller = widget._acquisition_controller
    assert widget.mda_af_mode_combo.currentText() == "off"
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
    assert any(isinstance(event.action, HardwareAutofocus) for event in sequence)
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
    # Grids default to no camera measurement; explicitly request it here.
    widget.mda_imgp_input.setText("all" if camera else "none")
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


def test_simulated_camera_only_uses_native_writer_without_raman(owner, tmp_path):
    widget, panel, app = owner
    widget.collector = None
    widget.transformer = None
    widget._start_live_mda_timing = lambda count: None
    panel.setValue(MDASequence(channels=[{"config": "BF", "exposure": 1}]))
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
    assert list(tmp_path.rglob("*.tif")) or list(tmp_path.rglob("*.tiff"))
    assert not list(tmp_path.rglob("raman.h5"))


@pytest.mark.parametrize("dense", [False, True])
def test_grid_handoff_preserves_mda_frames_per_point(owner, dense):
    widget, panel, _app = owner
    widget.mda_repeats_input.setValue(3)
    widget.mda_af_mode_combo.setCurrentText("cell")
    widget.mda_afp_input.setText("none")
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
    assert widget.mda_af_mode_combo.currentText() == "cell"
    assert widget.mda_afp_input.text() == "none"
    assert panel.targets[0]["points"]["0"] == [[0.5, 0.5]]
