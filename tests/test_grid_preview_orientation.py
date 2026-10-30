"""Asymmetric spatial patterns catch transposes, mirroring, and snake errors."""
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from useq import MDASequence

from napari_raman_widget.raman_grid_image import RamanGridImage
from raman_mda_engine.grid_plan import GRID_KEY, grid_position, iter_grid_events
from raman_mda_engine.galvo_grid import galvo_point


@pytest.mark.parametrize("order", ["row_wise", "column_wise", "row_wise_snake", "column_wise_snake", "spiral"])
@pytest.mark.parametrize("reverse_x,reverse_y", [(False, False), (False, True), (True, False), (True, True)])
def test_stage_preview_matches_camera_orientation_for_all_traversals(order, reverse_x, reverse_y):
    app = QApplication.instance() or QApplication([])
    x = [30, 10, 3] if reverse_x else [10, 30, 3]
    y = [40, 20, 2] if reverse_y else [20, 40, 2]
    sequence = MDASequence(channels=["RM"], metadata={GRID_KEY: dict(version=1, regions=[dict(x=x, y=y, order=order)])})
    model = RamanGridImage(sequence, 1000, SimpleNamespace(get_wavenumbers=lambda n: [1000]))
    for event in iter_grid_events(sequence):
        position = grid_position(sequence, event.index["p"])
        model.add(event, [[position.x + 100 * position.y]])
    assert model.error is None
    expected = np.array([[4030, 4020, 4010], [2030, 2020, 2010]])
    np.testing.assert_array_equal(model._render_map((0, 0, 0, 0), (2, 3)), expected)
    # Rebuilding at camera resolution must preserve corner orientation.
    np.testing.assert_array_equal(model._render_map((0, 0, 0, 0), (4, 6)),
                                  expected.repeat(2, axis=0).repeat(2, axis=1))


@pytest.mark.parametrize("order", ["row_wise", "column_wise", "row_wise_snake", "column_wise_snake", "spiral"])
def test_galvo_preview_uses_same_xy_display_flip(order):
    app = QApplication.instance() or QApplication([])
    plan = dict(version=1, backend="galvo", center_volts=[0, 0],
                regions=[dict(x=[-.1, .1, 3], y=[-.2, .2, 2], order=order)])
    sequence = MDASequence(channels=["RM"], metadata={GRID_KEY: plan})
    model = RamanGridImage(sequence, 1000, SimpleNamespace(get_wavenumbers=lambda n: [1000]))
    for event in iter_grid_events(sequence):
        (x, y), _ = galvo_point(sequence, event.index["p"])
        model.add(event, [[x + 100 * y]])
    assert model.error is None
    np.testing.assert_allclose(model._render_map((0, 0, 0, 0), (2, 3)),
                               [[20.1, 20, 19.9], [-19.9, -20, -20.1]])
