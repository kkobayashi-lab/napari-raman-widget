from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from qtpy.QtCore import QCoreApplication
from useq import MDASequence


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
    assert viewer.dims.axis_labels == ["p", "c", "z", "y", "x"]
    assert np.max(layer.data[0, 0, 0]) == np.iinfo(np.uint16).max
    np.testing.assert_array_equal(
        layer.data[0, 1, 0],
        np.full((120, 180, 3), 17, dtype=np.uint16),
    )
    assert app is not None


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
