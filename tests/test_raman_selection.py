from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from useq import MDASequence, Position

from napari_raman_widget.raman_selection import selected_positions, target, RamanEditor
from napari_raman_widget.experiment import prepare_acquisition, ExperimentExtensions
from test_unified_mda import owner, qt_app


@pytest.fixture(autouse=True)
def local_icons(monkeypatch):
    # UI behavior does not depend on downloading decorative icons.
    from superqt.iconify import QIconifyIcon
    monkeypatch.setattr(QIconifyIcon, "addKey", lambda *args, **kwargs: None)


def survey():
    return MDASequence(stage_positions=[Position(x=100, y=200, z=10)], channels=["BF"], axis_order="tpcz")


MODEL = (np.array([[0, 0], [0, 2], [3, 0]]), 1)
ROWS = [[0, 0, 0, 0, 0, 12, 18], [0, 0, 0, 0, 0, 8, 14]]


def test_stage_positions_and_optional_indexed_provenance():
    positions = selected_positions(survey(), ROWS, (20, 30), MODEL, dataset="survey")
    assert [(p.x, p.y, p.z) for p in positions] == [(91, 196, 10), (103, 204, 10)]
    value = target(positions[0])
    assert value["source_image"]["index"] == dict(t=0, p=0, g=0, c=0, z=0)
    assert value["source_image"]["dataset"] == "survey"
    np.testing.assert_allclose(value["points"], [[10/30, 15/20]])
    plain = selected_positions(survey(), ROWS, (20, 30), MODEL, provenance=False)
    assert "source_image" not in target(plain[0])
    restored = MDASequence.model_validate_json(MDASequence(stage_positions=positions).model_dump_json())
    assert target(restored.stage_positions[0]) == value


def test_batch_and_centered_run_reorder_by_saved_position_index():
    positions = selected_positions(survey(), ROWS, (20, 30), batch=True)
    assert len(positions) == 1
    assert (positions[0].x, positions[0].y) == (100, 200)
    positions += selected_positions(survey(), ROWS[:1], (20, 30), MODEL)
    seq = MDASequence(stage_positions=positions[::-1], channels=[dict(config="RM", exposure=100), "BF"])
    plan = prepare_acquisition(seq, ExperimentExtensions(), [], image_shape=(30, 20))
    assert plan.sequence.metadata["raman_positions"]["0"]["mode"] == "centered"
    assert plan.sequence.metadata["raman_positions"]["1"]["mode"] == "integrated_batch"
    assert len(plan.sources[0]._points_by_position[1]) == 2
    assert len(list(plan.sequence)) == 4
    assert [c.config for c in plan.sequence.channels] == ["RM", "BF"]
    from napari_raman_widget.raman_selection import restore_position_targets
    restored = restore_position_targets(MDASequence.model_validate_json(plan.sequence.model_dump_json()))
    assert target(restored.stage_positions[1])["mode"] == "integrated_batch"
    assert "raman_positions" not in restored.metadata
    recompiled = prepare_acquisition(plan.sequence, ExperimentExtensions(), [], image_shape=(30, 20))
    assert recompiled.sequence.metadata["raman_positions"]["1"]["mode"] == "integrated_batch"


def test_direct_positions_need_no_survey_and_validate_selection():
    seq = survey().replace(channels=[dict(config="RM", exposure=100)])
    plan = prepare_acquisition(seq, ExperimentExtensions(), [], image_shape=(30, 20))
    assert "source_image" not in plan.sequence.metadata["raman_positions"]["0"]
    with pytest.raises(ValueError, match="calibration"):
        selected_positions(survey(), ROWS, (20, 30))
    with pytest.raises(ValueError, match="inside"):
        selected_positions(survey(), [[0, 0, 0, 0, 0, 20, 1]], (20, 30), MODEL)


@pytest.mark.parametrize("append", [False, True])
def test_editor_apply_and_cancel_preserves_channels(owner, append):
    import napari
    widget, panel, app = owner
    panel.setValue(survey())
    seq = panel.value()
    widget._ensure_current_vandermonde = lambda: MODEL
    widget.vandermonde = MODEL
    source = SimpleNamespace(data=np.zeros((1, 1, 1, 1, 1, 20, 30, 3)), name="survey", rgb=True,
                             scale=(1,)*7, contrast_limits=(0, 1), metadata={"useq_sequence": seq})
    with patch.object(napari.Viewer, "show", lambda *_: None):
        editor = RamanEditor(widget, seq, source)
        editor.points.data = np.asarray(ROWS)
        editor.destination.setCurrentIndex(int(append))
        editor.apply()
        result = panel.value()
        assert len(result.stage_positions) == 2 + int(append)
        assert [c.config for c in result.channels] == ["BF"]
        assert target(result.stage_positions[-1])["mode"] == "centered"
        editor = RamanEditor(widget, result, source)
        editor.viewer.close()
        assert panel.value().stage_positions == result.stage_positions


def test_legacy_off_center_targets_are_rejected_before_hardware(owner):
    from napari_raman_widget.experiment import capture_targets
    widget, panel, app = owner
    widget.core.defineConfig("Channel", "RM", "Dichroic", "Label", "Q505LP")
    panel.setValue(survey().replace(channels=[dict(config="RM", exposure=100)]))
    source = SimpleNamespace(name="cell", role="cell", _img_shape=(512, 512),
                             get_mda_points=lambda e, transform=True: np.array([[.1, .2]]))
    panel.targets = capture_targets(panel.value(), [source])
    with patch.object(widget.core, "run_mda") as run, patch(
        "napari_raman_widget.unified_mda.QMessageBox.question",
        side_effect=AssertionError("Off-center targets reached acquisition review"),
    ):
        with pytest.raises(ValueError, match="centered stage"):
            widget._acquisition_controller.run()
        run.assert_not_called()


def test_saved_mixed_run_has_indexed_metadata_and_batch_repeats(tmp_path):
    import h5py
    from test_streaming_storage import _FakeCore
    from raman_mda_engine import RamanAcquisitionWriter
    from napari_raman_widget.large_dataset_viewing import AcquisitionIndex, RamanKey
    positions = selected_positions(survey(), ROWS[:1], (20, 30), MODEL, provenance=False)
    positions += selected_positions(survey(), ROWS, (20, 30), batch=True, dataset="survey")
    ext = ExperimentExtensions()
    ext.raman.repeats = 2
    plan = prepare_acquisition(MDASequence(stage_positions=positions, channels=[dict(config="RM", exposure=100)]),
                               ext, [], image_shape=(30, 20))
    core = _FakeCore()
    writer = RamanAcquisitionWriter(tmp_path / "run", core=core, exact_path=True, image_positions=[])
    core.mda.events.sequenceStarted.emit(plan.sequence, {})
    for event in plan.sequence:
        points = plan.sources[0].get_mda_points(event)
        if event.index["p"] == 0:
            points = np.repeat(points, 2, axis=0)
        writer._save_raman(event, np.ones((2, 3)), points, ["Raman positions"] * len(points), 100)
    writer.close()
    acquisition = AcquisitionIndex.build(writer.path)
    centered = acquisition.load_raman_metadata(RamanKey(0, 0, 0))
    batch = acquisition.load_raman_metadata(RamanKey(0, 1, 0))
    assert not centered["batch"] and "source_image" not in centered["selection"]
    assert batch["batch"] and batch["repeats"] == 2 and batch["c"] == 0
    assert batch["selection"]["source_image"]["dataset"] == "survey"
    assert len(acquisition.raman_cells(1)) == 1  # two spectra are repeats of the whole batch
    assert acquisition.raman_cells(1)[0].repeat_count == 2
    np.testing.assert_allclose(acquisition.load_locations(RamanKey(0, 0, 0)), [[.5, .5], [.5, .5]])
    with h5py.File(writer.path / "raman.h5", "r") as handle:
        assert handle.attrs["schema_version"] == 2
        assert handle["positions/image_source_position"][:].tolist() == [-1, -1]


@pytest.mark.parametrize("batch", [False, True])
def test_engine_only_steers_galvo_for_integrated_batch(batch):
    from unittest.mock import MagicMock
    from useq import MDAEvent
    from raman_mda_engine import RamanEngine
    engine = object.__new__(RamanEngine)
    engine._batch = False
    engine._segment_and_track = False
    engine._image_x = engine._image_y = 100
    engine._max_volt = 1.6
    engine._point_repeats = 2
    engine._default_rm_exp = 100
    engine._is_raman_event = lambda e: False
    engine.try_set_config = MagicMock()
    engine._transformer = MagicMock()
    engine._transformer.BF_to_volts.side_effect = lambda points, **kw: points
    engine._spectra_collector = MagicMock()
    engine._spectra_collector.collect_spectra_pts_batch.return_value = np.ones((1, 3))
    engine.raman_events = MagicMock()
    engine.aiming_sources = [SimpleNamespace(role="cell", name="points",
        get_mda_points=lambda e: np.array([[.1, .2], [.3, .4]]))]
    seq = MDASequence(metadata={"raman_positions": {"0": {"mode": "integrated_batch" if batch else "centered"}}})
    event = MDAEvent(index={"p": 0}, channel={"config": "RM", "group": "Channel"}, sequence=seq)
    if batch:
        engine.record_raman(event, pre_acq=False)
        assert engine._spectra_collector.collect_spectra_pts_batch.call_count == 2
        emitted = engine.raman_events.ramanSpectraReady.emit.call_args.args
        assert emitted[1].shape == (2, 3)
        assert emitted[2].shape == (2, 2)
    else:
        with pytest.raises(ValueError, match="centered stage"):
            engine.record_raman(event, pre_acq=False)
        engine.try_set_config.assert_not_called()
        engine._spectra_collector.collect_spectra_pts.assert_not_called()


def test_selected_positions_execute_camera_and_raman_at_same_stage(owner, tmp_path):
    import time
    import h5py
    from pathlib import Path
    from qtpy import QtWidgets
    from raman_mda_engine import fakeAcquirer
    widget, panel, app = owner
    widget.core.defineConfig("Channel", "RM", "Dichroic", "Label", "Q505LP")
    widget.collector = fakeAcquirer()
    widget.collector.get_wavenumbers = lambda: np.arange(1340)
    widget.transformer = SimpleNamespace(BF_to_volts=lambda pts, max_volts: np.asarray(pts) * max_volts)
    widget.mm_config = str(Path(__file__).parents[2] / "raman-mda-engine/tests/test-config.cfg")
    for name in ("sel_cx_input", "sel_cy_input", "sel_r_input"):
        setattr(widget, name, QtWidgets.QSpinBox())
    widget._connect_raman_visualization = lambda engine: None
    widget._start_live_mda_timing = lambda count: None
    rows = [[0, 0, 0, 0, 0, 260, 260]]
    positions = selected_positions(survey(), rows, (512, 512), MODEL)
    positions += selected_positions(survey(), rows, (512, 512), batch=True)
    panel.setValue(MDASequence(stage_positions=positions, channels=[dict(config="BF", exposure=1), dict(config="RM", exposure=100)]))
    widget.mda_pre_acq_check.setChecked(False)
    widget.mda_repeats_input.setValue(2)
    panel.save_info.setValue(dict(save_dir=str(tmp_path), save_name="selected", format="tiff-sequence", should_save=True))
    camera_xy = []
    def frame(image, event, metadata):
        camera_xy.append((event.x_pos, event.y_pos))
    widget.core.mda.events.frameReady.connect(frame)
    controller = widget._acquisition_controller
    with patch("napari_raman_widget.unified_mda.QMessageBox.question", return_value=QtWidgets.QMessageBox.Ok):
        controller.run()
    deadline = time.monotonic() + 10
    while controller.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    widget.core.mda.events.frameReady.disconnect(frame)
    assert not controller.busy
    assert camera_xy == [(88, 192), (100, 200)]
    with h5py.File(tmp_path / "selected/raman.h5") as handle:
        events = handle["raman/events"][:]
        assert events["p"].tolist() == [0, 1]
        assert events["batch"].tolist() == [False, True]
        assert events["spectra_count"].tolist() == [2, 2]
        np.testing.assert_allclose(events["stage_xyz_um"][:, :2], camera_xy, atol=.02)
        assert handle["raman/repeat_index"][:].tolist() == [0, 1, 0, 1]
        assert handle.attrs["completion_status"] == "completed"


@pytest.mark.parametrize("batch", [False, True])
def test_inherit_current_reference_at_physical_location(batch):
    from napari_raman_widget.autofocus_references import reference, with_reference
    from napari_raman_widget.raman_selection import restore_position_targets
    image = survey()
    parent = with_reference(image.stage_positions[0], dict(
        point=[14/30, 20/20], image_shape=[30, 20], stage=[100, 200, 10]))
    current = image.replace(stage_positions=[parent])
    positions = selected_positions(image, ROWS, (20, 30), MODEL, batch=batch,
                                   provenance=False, reference_sequence=current)
    for pos in positions:
        assert reference(pos)["stage_xy_um"] == [85, 192]
        assert reference(pos)["stage"] == [pos.x, pos.y, pos.z]
        assert "source_image" not in target(pos)
    positions = [p.replace(sequence=p.sequence.replace(metadata={**p.sequence.metadata, "position_identity": str(i)}))
                 for i, p in enumerate(positions)]
    seq = MDASequence(stage_positions=positions[::-1], channels=["BF"])
    plan = prepare_acquisition(seq, ExperimentExtensions(), [])
    assert plan.sequence.metadata["autofocus_stage_xy"]["0"] == [85, 192]
    restored = restore_position_targets(MDASequence.model_validate_json(plan.sequence.model_dump_json()))
    assert reference(restored.stage_positions[0])["stage_xy_um"] == [85, 192]
    assert prepare_acquisition(restored, ExperimentExtensions(), []).sequence.metadata["autofocus_stage_xy"]["0"] == [85, 192]
    assert "stage_xy_um" not in reference(parent)
    unset = selected_positions(current, ROWS, (20, 30), MODEL, reference_sequence=image)
    assert all(reference(p) is None for p in unset)


def test_later_images_set_missing_reference_and_preserve_inherited(owner):
    import napari
    from napari_raman_widget.autofocus_references import reference, with_reference, ReferenceEditor, camera_sequence
    widget, panel, app = owner
    positions = selected_positions(survey(), ROWS, (20, 30), MODEL)
    positions[0] = with_reference(positions[0], dict(point=[14/30, 1], image_shape=[30, 20],
        stage=[positions[0].x, positions[0].y, positions[0].z], stage_xy_um=[85, 192]))
    panel.setValue(MDASequence(stage_positions=positions, channels=["BF"]))
    seq = panel.value()
    source = SimpleNamespace(data=np.zeros((1, 2, 1, 1, 1, 20, 30, 3)), name="Raman images", rgb=True,
        scale=(1,)*7, contrast_limits=(0, 1), metadata={"useq_sequence": camera_sequence(seq)})
    with patch.object(napari.Viewer, "show", lambda *_: None):
        editor = ReferenceEditor(widget, seq, source)
        editor.points.add([0, 1, 0, 0, 0, 10, 12])
        editor.apply()
        result = panel.value()
        assert reference(result.stage_positions[0])["stage_xy_um"] == [85, 192]
        assert reference(result.stage_positions[1])["point"] == [10/30, 12/20]
        editor = ReferenceEditor(widget, result, source)
        editor.points.add([0, 0, 0, 0, 0, 9, 11])
        editor.apply()
        assert "stage_xy_um" not in reference(panel.value().stage_positions[0])


@pytest.mark.parametrize("rgb", [False, True])
def test_acquired_bf_then_reference_then_fractional_raman_positions(owner, tmp_path, rgb):
    import napari
    from napari_raman_widget.autofocus_references import reference, ReferenceEditor, camera_sequence
    widget, panel, app = owner
    panel.setValue(survey())
    image_sequence = prepare_acquisition(panel.value(), ExperimentExtensions(), []).sequence
    source = SimpleNamespace(data=np.zeros((1, 1, 1, 1, 1, 20, 30) + ((3,) if rgb else ())), name="MDA (BF)", rgb=rgb,
        scale=(1,)*7, contrast_limits=(0, 1), metadata={"useq_sequence": image_sequence})
    model = (np.array([[0.01234567, 0.07654321], [0.1234567, 2.7654321], [3.1234567, 0.2345678]]), 1)
    widget._ensure_current_vandermonde = lambda: model
    widget.vandermonde = model
    with patch.object(napari.Viewer, "show", lambda *_: None):
        editor = ReferenceEditor(widget, panel.value(), source)
        editor.points.add([0, 0, 0, 0, 0, 14, 20])
        editor.apply()
        assert reference(panel.value().stage_positions[0]) is not None
        editor = RamanEditor(widget, panel.value(), source)
        editor.points.data = np.asarray(ROWS)
        editor.apply()
        result = panel.value()
        assert all(reference(p) is not None for p in result.stage_positions)
        assert "2 set" in panel._reference_summary.text()
        inherited_xy = reference(result.stage_positions[0])["stage_xy_um"]
        panel.stage_positions.save(tmp_path / "raman-positions.json")
        panel.stage_positions.load(tmp_path / "raman-positions.json")
        result = panel.value()
        assert reference(result.stage_positions[0])["stage_xy_um"] == inherited_xy
        source.metadata = {"useq_sequence": camera_sequence(result)}
        source.data = np.zeros((1, 2, 1, 1, 1, 20, 30) + ((3,) if rgb else ()))
        editor = ReferenceEditor(widget, result, source)
        editor.points.add([0, 0, 0, 0, 0, 9, 11])
        editor.apply()
        assert reference(panel.value().stage_positions[0])["point"] == [9/30, 11/20]
        # Even the smallest representable coordinate edit invalidates a reference.
        table = panel.stage_positions.table()
        x = table.cellWidget(0, table.indexOf(panel.stage_positions.X))
        old = x.value()
        x.setValue(old + 10 ** -x.decimals())
        assert reference(panel.value().stage_positions[0]) is None
        x.setValue(old)
        assert reference(panel.value().stage_positions[0]) is None
        assert reference(panel.value().stage_positions[1]) is not None
