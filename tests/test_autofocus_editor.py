from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from useq import MDASequence, Position

from test_unified_mda import owner, qt_app
from napari_raman_widget.autofocus_references import (
    reference, with_reference, reference_records, match_positions, camera_sequence,
    ReferenceEditor, choose_references,
)
from napari_raman_widget.experiment import ExperimentExtensions, prepare_acquisition, position_id


def configured(panel):
    panel.setValue(MDASequence(stage_positions=[Position(x=1, y=2, z=3), Position(x=4, y=5, z=6)], channels=["BF"]))
    seq = panel.value()
    pos = seq.stage_positions[0]
    pos = with_reference(pos, dict(point=[0.25, 0.75], image_shape=[20, 30], stage=[pos.x, pos.y, pos.z]))
    panel.setValue(seq.replace(stage_positions=[pos, seq.stage_positions[1]]))
    return panel.value()


def test_position_reference_lifecycle_and_column(owner, tmp_path):
    widget, panel, app = owner
    seq = configured(panel)
    table = panel.stage_positions.table()
    assert table.item(0, panel._reference_column).text() == "● Set"
    assert table.item(1, panel._reference_column).text() == "— Skipped"
    assert "1 set" in panel._reference_summary.text()
    # Native rename preserves metadata identity and reference.
    table.item(0, table.indexOf(panel.stage_positions.NAME)).setText("renamed")
    renamed = panel.value()
    assert position_id(renamed.stage_positions[0]) == position_id(seq.stage_positions[0])
    assert reference(renamed.stage_positions[0])
    panel.stage_positions.save(tmp_path / "positions.json")
    panel.stage_positions.load(tmp_path / "positions.json")
    assert reference(panel.value().stage_positions[0])
    panel.setValue(renamed.replace(stage_positions=renamed.stage_positions[::-1]))
    assert reference(panel.value().stage_positions[1])
    # Moving, then moving back, must not resurrect a discarded reference.
    x = table.cellWidget(1, table.indexOf(panel.stage_positions.X))
    old = x.value()
    x.setValue(old + 1)
    assert reference(panel.value().stage_positions[1]) is None
    x.setValue(old)
    assert reference(panel.value().stage_positions[1]) is None
    table.removeRow(1)
    assert not any(r.get("role") == "autofocus" for r in panel.targets)


def test_only_referenced_fovs_autofocus(owner):
    widget, panel, app = owner
    seq = configured(panel)
    ext = ExperimentExtensions()
    ext.autofocus.mode = "laser"
    plan = prepare_acquisition(seq, ext, panel.targets)
    assert plan.autofocus_positions == (0,)
    moved = seq.replace(stage_positions=[seq.stage_positions[0].replace(x=99), seq.stage_positions[1]])
    plan = prepare_acquisition(moved, ext, panel.targets)
    assert plan.autofocus_positions == ()
    assert plan.extensions.autofocus.mode == "off"


def test_image_matching_and_camera_plan(owner):
    widget, panel, app = owner
    seq = configured(panel).replace(channels=["BF", {"config": "RM", "exposure": 100}],
                                   time_plan={"loops": 2, "interval": 0}, z_plan={"range": 2, "step": 1}, axis_order="tpcz")
    preview = camera_sequence(seq)
    assert [c.config for c in preview.channels] == ["BF"]
    assert preview.time_plan == seq.time_plan
    assert preview.z_plan == seq.z_plan
    assert len(list(preview)) == 12
    assert {e.channel.config for e in preview} == {"BF"}
    assert match_positions(seq.replace(stage_positions=seq.stage_positions[::-1]), preview) == [1, 0]
    with pytest.raises(ValueError, match="does not match"):
        match_positions(seq.replace(stage_positions=[seq.stage_positions[0].replace(z=99)]), preview)


def test_editor_apply_cancel_and_channel_navigation(owner):
    import napari
    widget, panel, app = owner
    seq = configured(panel).replace(channels=["BF", "BF"])
    data = np.zeros((1, 2, 1, 2, 1, 20, 30, 3), dtype=np.uint16)
    source = SimpleNamespace(data=data, name="MDA (BF)", rgb=True,
                             scale=(1, 1, 1, 1, 1, 0.5, 0.5), contrast_limits=(0, 100),
                             metadata={"useq_sequence": seq})
    with patch.object(napari.Viewer, "show", lambda *_: None):
        editor = ReferenceEditor(widget, seq, source)
        widget._autofocus_editor = editor
        # Add replaces the old reference in this FOV.
        editor.points.add([0, 0, 0, 0, 0, 10, 12])
        assert len(editor.points.data) == 1
        editor.viewer.dims.set_current_step(3, 1)
        assert editor.points.data[0, 3] == 1
        assert reference(panel.value().stage_positions[0])["point"] == [0.25, 0.75]
        editor.apply()
        assert reference(panel.value().stage_positions[0])["point"] == [10 / 30, 12 / 20]
        editor = ReferenceEditor(widget, panel.value(), source)
        widget._autofocus_editor = editor
        editor.points.data = np.empty((0, 7))
        editor.viewer.close()  # Cancel must retain the saved point.
        assert reference(panel.value().stage_positions[0])
        editor = ReferenceEditor(widget, panel.value(), source)
        widget._autofocus_editor = editor
        editor.points.data = np.empty((0, 7))
        editor.apply()
        assert reference(panel.value().stage_positions[0]) is None


def test_existing_source_does_not_acquire(owner):
    widget, panel, app = owner
    seq = configured(panel)
    layer = SimpleNamespace(metadata={"useq_sequence": seq})
    with patch("napari_raman_widget.autofocus_references.ReferenceEditor") as editor, patch.object(widget.core, "run_mda") as run:
        choose_references(widget, image_layer=layer)
        run.assert_not_called()
        editor.assert_called_once()


def test_editor_rejects_changed_positions(owner):
    widget, panel, app = owner
    seq = configured(panel)
    panel.setValue(seq.replace(stage_positions=[seq.stage_positions[0].replace(x=99), seq.stage_positions[1]]))
    editor = ReferenceEditor.__new__(ReferenceEditor)
    editor.owner, editor.sequence = widget, seq
    from qtpy.QtWidgets import QLabel
    editor.error = QLabel()
    editor.apply()
    assert "Positions changed" in editor.error.text()
    assert reference(panel.value().stage_positions[0]) is None
