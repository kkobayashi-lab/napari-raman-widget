from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication, QSpinBox
from useq import MDASequence

from napari_raman_widget.galvo_grid_editor import GalvoGridEditor
from napari_raman_widget.experiment import prepare_acquisition, ExperimentExtensions
from napari_raman_widget.mda_timing import estimate_mda_time
from napari_raman_widget.raman_grid_image import RamanGridImage
from raman_mda_engine.grid_plan import GRID_KEY, iter_grid_events


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor(app):
    repeats = QSpinBox()
    repeats.setRange(1, 100)
    owner = SimpleNamespace(_saved_galvo_center=[.2, -.3], mda_repeats_input=repeats)
    widget = GalvoGridEditor(SimpleNamespace(owner=owner, panel=SimpleNamespace(valueChanged=Mock())))
    yield widget
    widget.close()


def test_controls_round_trip_offsets_and_saved_center(editor):
    editor.ranges[0].setValue(.3)
    editor.ranges[1].setValue(.2)
    editor.counts[0].setValue(3)
    editor.counts[1].setValue(2)
    settings = editor.settings()
    plan = editor.plan()
    np.testing.assert_allclose(plan["regions"][0]["x"], [-.1, .5, 3])
    editor.ranges[0].setValue(.9)
    editor.restore(settings)
    assert editor.settings() == settings
    editor.modes.owner._saved_galvo_center = [.4, -.2]
    editor.changed()
    np.testing.assert_allclose(editor.plan()["regions"][0]["x"], [.1, .7, 3])
    assert "0.4000" in editor.center.text()


def test_prepare_without_camera_targets_and_timing(editor):
    editor.counts[0].setValue(3)
    editor.counts[1].setValue(2)
    sequence = MDASequence(metadata={GRID_KEY: editor.plan()}, channels=[dict(config="RM", exposure=100)])
    ext = ExperimentExtensions()
    ext.raman.batch = True
    ext.raman.repeats = 2
    ext.autofocus.mode = "laser"
    ext.tracking.enabled = True
    prepared = prepare_acquisition(sequence, ext, [])
    assert not prepared.extensions.raman.batch
    assert prepared.extensions.autofocus.mode == "off"
    assert not prepared.extensions.tracking.enabled
    assert ext.autofocus.mode == "laser" and ext.tracking.enabled
    assert all(event.z_pos is None for event in iter_grid_events(prepared.sequence))
    assert not prepared.sources
    estimate = estimate_mda_time(prepared.sequence, prepared.sources, 100, [0], [], point_repeats=2)
    assert estimate.raman_points == 12
    assert estimate.raman_events == 6
    editor.modes.owner._saved_galvo_center = None
    bad = sequence.replace(metadata={GRID_KEY: editor.plan()})
    with pytest.raises(ValueError, match="Save the galvo center"):
        prepare_acquisition(bad, ext, [])


def test_voltage_map_keeps_native_shape_and_snake_pixel_order(editor, app):
    editor.counts[0].setValue(3)
    editor.counts[1].setValue(2)
    sequence = MDASequence(metadata={GRID_KEY: editor.plan()}, channels=["RM"])
    image = RamanGridImage(sequence, 1003, SimpleNamespace(get_wavenumbers=lambda n: np.array([1003.])),
                           frame_shape=(100, 200))
    for i, event in enumerate(iter_grid_events(sequence)):
        image.add(event, np.array([[i], [i + 2]]))
    data = image._render_map((0, 0, 0, 0), (2, 3))
    np.testing.assert_allclose(data, [[4, 5, 6], [3, 2, 1]])
    class Viewer:
        def add_image(self, data, **kw):
            return SimpleNamespace(data=data, **kw)
        def reset_view(self):
            pass
    image.show(Viewer(), Mock(), save=False)
    layer = next(iter(image.layers.values()))
    assert layer.data.shape == (100, 200)
    assert "stage_grid_um" not in layer.metadata
    assert layer.metadata["display_coordinates"] == "preview canvas (not camera registration)"


def test_live_map_matches_camera_extent_and_exports_native_voltages(editor, app, tmp_path):
    import napari
    import tifffile
    editor.counts[0].setValue(3)
    editor.counts[1].setValue(2)
    sequence = MDASequence(metadata={GRID_KEY: editor.plan()}, channels=["RM"])
    viewer = napari.Viewer(show=False)
    try:
        camera = viewer.add_image(np.zeros((120, 160)), scale=(.25, .25))
        image = RamanGridImage(sequence, 1003, SimpleNamespace(get_wavenumbers=lambda n: np.array([1003.])),
                               frame_shape=(120, 160), pixel_size=.25, output_dir=tmp_path)
        image.start(viewer, Mock(), sequence)
        layer = next(iter(image.layers.values()))
        np.testing.assert_allclose(layer.extent.world, camera.extent.world)
        events = list(iter_grid_events(sequence))
        for i, event in enumerate(events):
            image.add(event, np.array([[100 + i]]))
        app.processEvents()
        assert image.error is None
        assert layer.visible
        assert layer.data.shape == camera.data.shape
        assert layer.data[0, 0] == 103
        assert layer.data[-1, -1] == 100  # Both axes flipped only for display.
        assert layer.metadata["display_flip_xy"] is True
        assert layer.contrast_limits[1] >= 105
        image.show(viewer, Mock(), reference_layer=camera)
        saved = tifffile.imread(next(tmp_path.glob('*.tif')))
        np.testing.assert_array_equal(saved, [[100, 101, 102], [105, 104, 103]])
        export = image._galvo_export_layer(layer, (0, 0, 0, 0))
        assert export.metadata["display_coordinates"] == "galvo voltage Y/X (V)"
        assert export.metadata["display_flip_xy"] is False
        np.testing.assert_allclose(export.scale, [.2, .1])
        np.testing.assert_allclose(export.translate, [-.4, .1])
    finally:
        viewer.close()


def test_mda_grid_mode_switch_and_round_trip(app, monkeypatch):
    from qtpy import QtGui, QtWidgets
    for name in ("QAction", "QActionGroup", "QUndoCommand", "QUndoStack"):
        if not hasattr(QtGui, name):
            monkeypatch.setattr(QtGui, name, getattr(QtWidgets, name), raising=False)
    from superqt.iconify import QIconifyIcon
    monkeypatch.setattr(QIconifyIcon, "addKey", lambda *a, **k: None)
    from pymmcore_plus import CMMCorePlus
    from napari_micromanager._gui_objects._mda_widget import MultiDWidget
    from napari_raman_widget.unified_mda import build_extension_controls, install_unified_mda
    core = CMMCorePlus()
    core.loadSystemConfiguration(str(Path(__file__).parents[2] / "raman-mda-engine/tests/test-config.cfg"))
    owner = QtWidgets.QWidget()
    owner.core = core
    for name in ("browse_cellpose_model", "browse_tracking_cfg", "open_acquisition_viewer",
                 "_toggle_autofocus_fields", "_toggle_seg_track_fields", "_on_raman_mda_finished",
                 "_finish_live_mda_timing"):
        setattr(owner, name, lambda *a: None)
    core.mda.events.sequenceFinished.connect(owner._on_raman_mda_finished)
    owner.status = QtWidgets.QLabel()
    owner.apply_mda_positions_btn = QtWidgets.QPushButton()
    owner._raman_mda_pending = False
    owner._scheduled_raman_mda = None
    owner.tf_path = QtWidgets.QLineEdit("aiming.json")
    owner.vandermonde_path = "stage.json"
    owner.sel_repeats_input = QSpinBox()
    owner.sel_repeats_input.setValue(1)
    owner._saved_galvo_center = [.2, -.3]
    build_extension_controls(owner)
    window = QtWidgets.QWidget()
    old = MultiDWidget(parent=window, mmcore=core)
    dock = QtWidgets.QDockWidget("MDA", window)
    dock.setWidget(old)
    window._dock_widgets = {"MDA": dock}
    owner.main_window = window
    panel = install_unified_mda(owner)
    try:
        modes = panel.grid_modes
        modes.selector.setCurrentIndex(modes.selector.findData("galvo"))
        modes.galvo.counts[0].setValue(3)
        saved = panel.value()
        assert saved.metadata[GRID_KEY]["backend"] == "galvo"
        assert saved.stage_positions == ()
        panel.setValue(MDASequence.model_validate_json(saved.model_dump_json()))
        assert modes.mode == "galvo"
        assert panel.tab_wdg.currentWidget() is panel.grid_plan
        assert not panel.tab_wdg.isChecked(panel.stage_positions)
        assert not panel.stage_positions.isEnabled()
        assert modes.galvo.counts[0].value() == 3
        modes.selector.setCurrentIndex(modes.selector.findData("raman"))
        assert modes.mode == "raman"
        assert panel.stage_positions.isEnabled()
        modes.selector.setCurrentIndex(modes.selector.findData("galvo"))
        assert modes.galvo.counts[0].value() == 3
        assert panel.value().metadata[GRID_KEY]["backend"] == "galvo"
        owner._saved_galvo_center = None
        panel.update_summary()
        assert "Save the galvo center" in owner.mda_collection_summary.text()
    finally:
        owner._acquisition_controller.disconnect()
        panel._disconnect()
        panel.control_btns._disconnect()
        window.close()
        owner.close()
