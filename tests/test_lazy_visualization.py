from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtCore import QCoreApplication
from useq import MDAEvent, MDASequence


def _load_lazy_module():
    path = (
        Path(__file__).parents[1]
        / "napari_raman_widget"
        / "lazy_visualization.py"
    )
    spec = importlib.util.spec_from_file_location("lazy_visualization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        self.slots.remove(slot)


class _Core:
    def __init__(self):
        events = type(
            "Events",
            (),
            {
                "sequenceStarted": _Signal(),
                "frameReady": _Signal(),
                "sequenceFinished": _Signal(),
            },
        )()
        self.mda = type("MDA", (), {"events": events})()

    def getImageHeight(self):
        return 120

    def getImageWidth(self):
        return 180

    def getBytesPerPixel(self):
        return 2

    def getPixelSizeUm(self):
        return 0


class _Layer:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.visible = True
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class _Dims:
    def __init__(self):
        self.axis_labels = ["y", "x"]
        self.current_step = [0, 0]


class _Viewer:
    def __init__(self):
        self.dims = _Dims()
        self.layers = []

    def add_image(self, data, **kwargs):
        layer = _Layer(data, kwargs["metadata"])
        self.layers.append(layer)
        ndim = data.ndim - (1 if data.shape[-1] in (3, 4) else 0)
        self.dims.axis_labels = [""] * ndim
        self.dims.current_step = [0] * ndim
        return layer


def test_raman_and_camera_share_one_lazy_channel_stack():
    app = QCoreApplication.instance() or QCoreApplication([])
    lazy = _load_lazy_module()
    core = _Core()
    viewer = _Viewer()
    handler = lazy.LazyMDAViewer(core, viewer)
    sequence = MDASequence(
        axis_order="pcz",
        channels=["RM", "BF"],
        stage_positions=[(0, 0, 0)],
        z_plan={"relative": [0]},
    )
    events = list(sequence)
    raman_event, camera_event = events[0], events[1]

    handler._on_mda_started(sequence)
    assert viewer.layers == []

    handler.add_raman_spectrum(
        raman_event,
        np.array(
            [
                [0.0, 1.0, 0.25],
                [0.2, 0.8, 0.35],
                [1.0, 0.1, 0.75],
            ]
        ),
        np.array([500.0, 1000.0, 1500.0]),
        points=np.array([[0.2, 0.3], [0.2, 0.3], [0.8, 0.7]]),
        which=["cells", "cells", "cells"],
    )
    handler._on_mda_frame(
        np.full((120, 180), 17, dtype=np.uint16),
        camera_event,
    )

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.metadata["channel_labels"] == ["RM", "BF"]
    assert viewer.dims.axis_labels == ["t", "p", "g", "c", "z", "y", "x"]
    assert np.max(layer.data[0, 0, 0, 0, 0]) == np.iinfo(np.uint16).max
    np.testing.assert_array_equal(
        layer.data[0, 0, 0, 1, 0],
        np.full((120, 180, 3), 17, dtype=np.uint16),
    )
    assert app is not None


@pytest.mark.parametrize("order", ["tpcz", "cpzt", "ctz"])
def test_dense_grid_preserves_time_position_channel_and_z_results(order):
    from raman_mda_engine.grid_plan import GRID_KEY

    lazy = _load_lazy_module()
    expected_axes = ("t", "p", "g", "c", "z")
    viewer = _Viewer()
    handler = lazy.LazyMDAViewer(_Core(), viewer)
    sequence = MDASequence(
        axis_order=order, channels=["RM", "BF"],
        time_plan={"loops": 2, "interval": 0}, z_plan={"relative": [0, 1]},
        metadata={GRID_KEY: {"version": 1, "regions": [
            {"x": [0, 1, 2], "y": [0, 1, 2], "z": 0, "snake_axis": None}]}}
    )
    handler._spectrum_image = lambda spectra, wn, shape, dtype, **kwargs: np.full(shape, spectra[0], dtype=dtype)
    handler._on_mda_started(sequence)
    first = MDAEvent(sequence=sequence, index={"p": 0, "c": 0, "z": 0, "t": 0})
    last = MDAEvent(sequence=sequence, index={"p": 3, "c": 0, "z": 0, "t": 0})
    try:
        handler.add_raman_spectrum(first, [11])
        handler.add_raman_spectrum(last, [29])
        axes, array, layer = handler._run_layer
        assert axes == expected_axes
        assert viewer.dims.axis_labels == [*expected_axes, "y", "x"]
        assert array.shape[axes.index("p")] == 4
        for axis in ("t", "c", "z"):
            assert array.shape[axes.index(axis)] == 2
        assert viewer.dims.current_step[axes.index("p")] == 3
        np.testing.assert_array_equal(array[handler._event_index(first, axes)], 11)
        np.testing.assert_array_equal(array[handler._event_index(last, axes)], 29)
        later_time = MDAEvent(sequence=sequence, index={"p": 3, "c": 0, "z": 0, "t": 1})
        handler.add_raman_spectrum(later_time, [44])
        assert viewer.dims.current_step[axes.index("t")] == 1
        later_z = MDAEvent(sequence=sequence, index={"p": 3, "c": 0, "z": 1, "t": 1})
        handler.add_raman_spectrum(later_z, [55])
        assert viewer.dims.current_step[axes.index("z")] == 1
        camera = MDAEvent(sequence=sequence, index={"p": 3, "c": 1, "z": 1, "t": 1})
        handler._on_mda_frame(np.full((120, 180), 7, dtype=np.uint16), camera)
        assert len(viewer.layers) == 1
        np.testing.assert_array_equal(array[handler._event_index(camera, axes)], 7)
        handler._on_mda_finished(sequence)
        # Each time/position/channel/Z combination remains independently browsable.
        for event, value in ((first, 11), (last, 29), (later_time, 44), (later_z, 55), (camera, 7)):
            handler._show_event(event, axes)
            assert tuple(viewer.dims.current_step[:-2]) == handler._event_index(event, axes)
            np.testing.assert_array_equal(array[handler._event_index(event, axes)], value)
    finally:
        handler._cleanup()


def test_different_run_orders_align_shared_sliders_and_zero_z_range_is_single_plane():
    from raman_mda_engine.grid_plan import GRID_KEY

    lazy = _load_lazy_module()
    handler = lazy.LazyMDAViewer(_Core(), _Viewer())
    handler._spectrum_image = lambda spectra, wn, shape, dtype, **kwargs: np.full(shape, 7, dtype=dtype)
    try:
        for order in ("tpcz", "ctz"):
            sequence = MDASequence(
                axis_order=order, channels=["BF", "RM"],
                time_plan={"loops": 1, "interval": 0}, z_plan={"range": 0, "step": 0.5},
                metadata={GRID_KEY: {"version": 1, "regions": [
                    {"x": [0, 2, 3], "y": [0, 2, 3], "z": 0, "snake_axis": None}]}}
            )
            handler._on_mda_started(sequence)
            handler.add_raman_spectrum(MDAEvent(sequence=sequence, index={"p": 8, "c": 1, "t": 0, "z": 0}), [7])
        assert len(handler.viewer.layers) == 2
        for layer in handler.viewer.layers:
            assert layer.data.shape[:5] == (1, 9, 1, 2, 1)
        assert handler.viewer.dims.axis_labels == ["t", "p", "g", "c", "z", "y", "x"]
        assert handler.viewer.dims.current_step[:5] == [0, 8, 0, 1, 0]
    finally:
        handler._cleanup()


def test_napari_shared_ranges_do_not_turn_grid_positions_into_time():
    from napari.components import ViewerModel
    from raman_mda_engine.grid_plan import GRID_KEY

    lazy = _load_lazy_module()
    viewer = ViewerModel()
    handler = lazy.LazyMDAViewer(_Core(), viewer)
    handler._spectrum_image = lambda spectra, wn, shape, dtype, **kwargs: np.full(shape, 7, dtype=dtype)
    try:
        for order, z_range in (("tpcz", 0), ("ctz", 0), ("cpzt", 2)):
            sequence = MDASequence(
                axis_order=order, channels=["BF", "RM"],
                time_plan={"loops": 1, "interval": 0}, z_plan={"range": z_range, "step": 1},
                metadata={GRID_KEY: {"version": 1, "regions": [
                    {"x": [0, 2, 3], "y": [0, 2, 3], "z": 0, "snake_axis": None}]}}
            )
            handler._on_mda_started(sequence)
            handler.add_raman_spectrum(MDAEvent(sequence=sequence, index={"p": 8, "c": 1, "t": 0, "z": z_range}), [7])
            assert viewer.dims.axis_labels == ("t", "p", "g", "c", "z", "y", "x")
            assert viewer.dims.range[0][1] == 0
            assert viewer.dims.range[1][1] == 8
            assert viewer.dims.range[4][1] == z_range
    finally:
        viewer.layers.clear()
        handler._cleanup()


def test_million_position_grid_viewer_remains_chunked_and_unexpanded():
    from raman_mda_engine.grid_plan import GRID_KEY

    lazy = _load_lazy_module()
    handler = lazy.LazyMDAViewer(_Core(), _Viewer())
    sequence = MDASequence(channels=["RM"], metadata={GRID_KEY: {
        "version": 1, "regions": [
            {"x": [0, 100, 1000], "y": [0, 100, 1000], "z": 0, "snake_axis": None}]}})
    try:
        handler._on_mda_started(sequence)
        assert handler.viewer.layers == []
        axes, array, layer = handler._create_stack(MDAEvent(sequence=sequence, index={"p": 999999}))
        assert sequence.stage_positions == ()
        assert array.shape[axes.index("p")] == 1_000_000
        assert array.chunks[axes.index("p")] == 1
    finally:
        handler._cleanup()


def test_repeated_frames_at_same_point_are_averaged():
    lazy = _load_lazy_module()
    spectra = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0],
        ]
    )
    points = np.array([[0.25, 0.5], [0.25, 0.5], [0.75, 0.5]])

    traces, grouped_points, counts = (
        lazy.LazyMDAViewer._group_spectra_by_point(spectra, points)
    )

    np.testing.assert_allclose(
        traces,
        [[2.0, 3.0, 4.0], [10.0, 20.0, 30.0]],
    )
    np.testing.assert_allclose(
        grouped_points,
        [[0.25, 0.5], [0.75, 0.5]],
    )
    np.testing.assert_array_equal(counts, [2, 1])


def test_matplotlib_spectrum_renderer_uses_rgb_without_hardware():
    lazy = _load_lazy_module()

    image = lazy.LazyMDAViewer._spectrum_image(
        np.array(
            [
                [1.0, 4.0, 2.0],
                [4.0, 1.0, 3.0],
            ]
        ),
        None,
        (180, 260, 3),
        np.dtype("uint8"),
        points=np.array([[0.2, 0.3], [0.7, 0.8]]),
    )

    assert image.shape == (180, 260, 3)
    assert image.dtype == np.uint8
    assert image.max() == 255
    assert np.any(
        (image[..., 0] != image[..., 1])
        | (image[..., 1] != image[..., 2])
    )
