from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from useq import MDASequence

from napari_raman_widget.raman_batches import sample_region, batch_positions
from napari_raman_widget.raman_selection import RamanEditor, target, restore_position_targets
from napari_raman_widget.experiment import prepare_acquisition, ExperimentExtensions, METADATA_KEY
from test_raman_selection import survey, local_icons, MODEL
from test_unified_mda import owner, qt_app


def vertices(points, plane=(0, 0, 0, 0, 0)):
    return np.column_stack([np.tile(plane, (len(points), 1)), points])


RECT = vertices([[2, 3], [16, 3], [16, 25], [2, 25]])


@pytest.mark.parametrize("kind", ["rectangle", "ellipse", "polygon"])
@pytest.mark.parametrize("count", [1, 25, 123])
def test_sampling_exact_reproducible_interior(kind, count):
    points = sample_region(RECT, kind, count, (20, 30))
    assert points.shape == (count, 7)
    assert len(np.unique(points, axis=0)) == count
    assert np.all(points[:, -2:] > [2, 3])
    assert np.all(points[:, -2:] < [16, 25])
    np.testing.assert_array_equal(points, sample_region(RECT, kind, count, (20, 30)))
    if kind == "ellipse":
        assert np.all(np.sum(((points[:, -2:] - [9, 14]) / [7, 11])**2, axis=1) < 1)


def test_concave_and_rotated_shapes():
    concave = vertices([[1, 1], [15, 1], [15, 5], [5, 5], [5, 20], [1, 20]])
    points = sample_region(concave, "polygon", 79, (20, 30))[:, -2:]
    assert np.all((points[:, 0] < 5) | (points[:, 1] < 5))
    theta = .6
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    box = np.array([[-6, -3], [6, -3], [6, 3], [-6, 3]]) @ rotation + [10, 15]
    points = sample_region(vertices(box), "ellipse", 37, (20, 30))[:, -2:]
    local = (points - [10, 15]) @ rotation.T
    assert np.all(np.sum((local / [6, 3])**2, axis=1) < 1)


def test_invalid_regions():
    with pytest.raises(ValueError, match="lines and paths"):
        sample_region(RECT, "path", 25, (20, 30))
    bad = RECT.copy()
    bad[1, 0] = 1
    with pytest.raises(ValueError, match="one acquired image plane"):
        sample_region(bad, "polygon", 25, (20, 30))
    with pytest.raises(ValueError, match="inside"):
        sample_region(RECT, "rectangle", 25, (10, 10))
    with pytest.raises(ValueError, match="nonzero"):
        sample_region(np.zeros((4, 7)), "rectangle", 25, (20, 30))


def test_separate_batches_same_plane_survive_execution_and_json():
    source = survey()
    batches = [dict(id=str(i), name=f"Batch {i}", vertices=RECT.tolist(), shape_type="rectangle",
                    count=n, points=sample_region(RECT, "rectangle", n, (20, 30)).tolist())
               for i, n in enumerate([7, 19])]
    positions = batch_positions(source, batches, (20, 30), MODEL)
    plan = prepare_acquisition(MDASequence(stage_positions=positions[::-1], channels=[dict(config="RM", exposure=100)]),
                               ExperimentExtensions(), [], image_shape=(30, 20))
    assert len(list(plan.sequence)) == 2
    assert [len(plan.sources[0]._points_by_position[p]) for p in (0, 1)] == [19, 7]
    from napari_raman_widget.mda_timing import estimate_mda_time
    estimate = estimate_mda_time(plan.sequence, plan.sources, 100, [0], [], point_repeats=3)
    assert estimate.raman_points == 6  # three spectra per batch, independent of sampling count
    restored = restore_position_targets(MDASequence.model_validate_json(plan.sequence.model_dump_json()))
    assert target(restored.stage_positions[0])["batch_region"]["id"] == "1"
    assert target(restored.stage_positions[1])["batch_region"]["vertices"] == RECT.tolist()


def test_native_shapes_edit_delete_apply_and_restore(owner):
    import napari
    widget, panel, app = owner
    widget._ensure_current_vandermonde = lambda: MODEL
    widget.vandermonde = MODEL
    panel.setValue(survey())
    seq = panel.value()
    source = SimpleNamespace(data=np.zeros((1, 1, 1, 1, 1, 20, 30)), name="survey", rgb=False,
                             scale=(1,)*7, contrast_limits=(0, 1), metadata={"useq_sequence": seq})
    with patch.object(napari.Viewer, "show", lambda *_: None):
        editor = RamanEditor(widget, seq, source)
        editor.batch.setChecked(True)
        editor.shapes.add([RECT, RECT], shape_type=["rectangle", "ellipse"])
        editor.refresh_batches()
        editor._change_count(0, 7)
        editor._change_count(1, 19)
        assert len(editor.preview.data) == 26
        assert len({b["id"] for b in editor._batches}) == 2
        # Native shape movement regenerates preview without changing count or identity.
        identity = editor._batches[1]["id"]
        moved = RECT.copy()
        moved[:, -1] += 1
        editor.shapes.data = [RECT, moved]
        editor.refresh_batches()
        assert editor._batches[1]["id"] == identity
        editor.apply()
        result = panel.value()
        assert len(result.stage_positions) == 2, editor.error.text()
        assert [target(p)["batch_region"]["count"] for p in result.stage_positions] == [7, 19]
        editor = RamanEditor(widget, result, source)
        assert editor.batch.isChecked()
        assert len(editor.preview.data) == 26
        editor.shapes.selected_data = {0}
        editor.shapes.remove_selected()
        editor.refresh_batches()
        assert editor._batches[0]["id"] == identity
        assert editor._batches[0]["count"] == 19
        editor.default_count.setValue(11)
        editor.shapes.add(RECT, shape_type="rectangle")
        editor.refresh_batches()
        assert [b["count"] for b in editor._batches] == [19, 11]
        assert editor._batches[1]["id"] != identity
        editor.viewer.close()
        assert panel.value().stage_positions == result.stage_positions


def test_raman_controls_are_in_selection_tab(owner):
    widget, panel, app = owner
    tab = next(panel.tab_wdg.widget(i) for i in range(panel.tab_wdg.count())
               if panel.tab_wdg.tabText(i) == "Raman selection")
    assert tab.isAncestorOf(widget.mda_repeats_input)
    assert not widget._mda_extension_panel.isAncestorOf(widget.mda_repeats_input)
    widget.mda_repeats_input.setValue(3)
    assert panel.value().metadata[METADATA_KEY]["extensions"]["raman"]["repeats"] == 3


def test_batches_move_stage_and_rebase_galvo_points():
    first = vertices([[2, 3], [6, 3], [6, 9], [2, 9]])
    second = first.copy()
    second[:, -2:] += [6, 12]
    batches = [dict(id=str(i), name=f"Batch {i}", vertices=v.tolist(), shape_type="rectangle",
                    count=7, points=sample_region(v, "rectangle", 7, (20, 30)).tolist())
               for i, v in enumerate([first, second])]
    positions = batch_positions(survey(), batches, (20, 30), MODEL)
    assert [(p.x, p.y, p.z) for p in positions] == [(127, 212, 10), (91, 200, 10)]
    for position, batch, center in zip(positions, batches, ([4, 6], [10, 18])):
        selection = target(position)
        sampled = np.asarray(batch["points"])[:, -2:]
        aimed = np.asarray(selection["points"]) * [30, 20]
        np.testing.assert_allclose(aimed, sampled - center + [10, 15])
        # Stage translation plus the new aiming offset preserves physical targets
        # under the affine pixel-to-stage calibration used in this test.
        expected_xy = [100, 200] - (sampled - [10, 15])[:, ::-1] * [3, 2]
        actual_xy = [position.x, position.y] - (aimed - [10, 15])[:, ::-1] * [3, 2]
        np.testing.assert_allclose(actual_xy, expected_xy)
        assert selection["source_image"]["pixels_yx"] == sampled.tolist()
        assert selection["batch_region"]["center_yx"] == center
        assert selection["batch_region"]["stage_centered"]
    with pytest.raises(ValueError, match="calibration"):
        batch_positions(survey(), batches, (20, 30))


def test_polygon_centroid_is_independent_of_point_count_and_winding():
    from napari_raman_widget.raman_batches import region_center
    triangle = vertices([[2, 2], [8, 2], [2, 14]])
    np.testing.assert_allclose(region_center(triangle, "polygon"), [4, 6])
    np.testing.assert_allclose(region_center(triangle[::-1], "polygon"), [4, 6])
    for count in (1, 19):
        batch = dict(id="triangle", name="Triangle", vertices=triangle.tolist(), shape_type="polygon",
                     count=count, points=sample_region(triangle, "polygon", count, (20, 30)).tolist())
        position = batch_positions(survey(), [batch], (20, 30), MODEL, provenance=False)[0]
        assert (position.x, position.y) == (127, 212)
        assert "source_image" not in target(position)


def test_batch_hover_and_picking_follow_all_image_axes(owner):
    """Exercise the status-checker path, including empty slices and rebuilt data."""
    import napari
    widget, panel, app = owner
    panel.setValue(survey())
    seq = panel.value()
    source = SimpleNamespace(data=np.zeros((3, 3, 3, 3, 3, 20, 30)), name="survey", rgb=False,
                             scale=(1,)*7, contrast_limits=(0, 1), metadata={"useq_sequence": seq})
    with patch.object(napari.Viewer, "show", lambda *_: None):
        editor = RamanEditor(widget, seq, source)
        try:
            editor.batch.setChecked(True)
            viewer = editor.viewer
            viewer.mouse_over_canvas = True
            for axis in range(5):
                viewer.dims.set_point(axis, 0)
            for kind in ("ellipse", "rectangle", "polygon"):
                for axis in range(5):
                    other = RECT.copy()
                    other[:, axis] = 1
                    # Replacing data recreates napari's internal ShapeList.
                    editor.shapes.data = []
                    editor.shapes.add([RECT, other], shape_type=kind)
                    editor.shapes.selected_data = set()
                    editor.shapes.mode = "select"
                    for plane, expected in ((0, 0), (1, 1), (2, None), (0, 0)):
                        viewer.dims.set_point(axis, plane)
                        # Avoid the exact center, a shared triangulation vertex.
                        cursor = [0.] * 5 + [9.2, 14.3]
                        cursor[axis] = plane
                        viewer.cursor.position = tuple(cursor)
                        assert viewer._calc_status_from_cursor() is not None
                        assert editor.shapes.get_value(cursor, world=True) == (expected, None), (
                            kind, axis, plane, editor.shapes._data_view.slice_key)
                        viewer.cursor.position = tuple(cursor[:5] + [1016.82322, 1195.51166])
                        assert viewer._calc_status_from_cursor() is not None
                        assert editor.shapes.get_value(viewer.cursor.position, world=True) == (None, None)
        finally:
            editor.viewer.close()
