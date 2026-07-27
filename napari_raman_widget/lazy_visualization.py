"""Low-overhead, data-driven MDA visualizations for napari."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import numpy as np
import zarr
from superqt.utils import ensure_main_thread


class LazyMDAViewer:
    """Create one MDA layer on first data, including Raman spectrum planes."""

    def __init__(self, core, viewer):
        self._mmc = core
        self.viewer = viewer
        self._mda_running = False
        self._sequence = None
        self._run_layer = None
        self._frame_shape = None
        self._stores = []
        events = core.mda.events
        self._connections = [
            (events.sequenceStarted, self._on_mda_started),
            (events.frameReady, self._on_mda_frame),
            (events.sequenceFinished, self._on_mda_finished),
        ]
        for signal, slot in self._connections:
            signal.connect(slot)

    def _cleanup(self):
        for signal, slot in self._connections:
            with contextlib.suppress(TypeError, RuntimeError):
                signal.disconnect(slot)
        for array, tmpdir in self._stores:
            with contextlib.suppress(Exception):
                array.store.close()
            with contextlib.suppress(Exception):
                tmpdir.cleanup()
        self._stores.clear()

    @ensure_main_thread
    def _on_mda_started(self, sequence, *_metadata):
        # Do not allocate arrays or add layers here. Acquisition starts
        # immediately; the common detector stack materializes on first data.
        self._sequence = sequence
        self._run_layer = None
        self._frame_shape = None
        self._mda_running = True

    def _storage_axes(self, sequence):
        sizes = getattr(sequence, "sizes", {})
        return tuple(
            axis
            for axis in getattr(sequence, "axis_order", ())
            if sizes.get(axis, 0)
        )

    def _camera_dtype(self):
        try:
            return np.dtype(f"u{int(self._mmc.getBytesPerPixel())}")
        except Exception:
            return np.dtype("uint16")

    def _camera_shape(self):
        try:
            height = int(self._mmc.getImageHeight())
            width = int(self._mmc.getImageWidth())
        except Exception:
            height = width = 0
        return (height or 512, width or 512)

    def _create_stack(self, event, *, image=None):
        sequence = event.sequence or self._sequence
        axes = self._storage_axes(sequence)
        if image is None:
            frame_shape = self._camera_shape()
            dtype = self._camera_dtype()
        else:
            image = np.asarray(image)
            frame_shape = image.shape
            dtype = image.dtype
        self._frame_shape = frame_shape
        shape = tuple(int(sequence.sizes[axis]) for axis in axes) + frame_shape
        chunks = (1,) * len(axes) + frame_shape
        tmpdir = tempfile.TemporaryDirectory()
        array = zarr.open(
            str(Path(tmpdir.name) / "frames.zarr"),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
        )
        self._stores.append((array, tmpdir))

        channel_labels = [
            channel.config for channel in getattr(sequence, "channels", ())
        ]
        metadata = {
            "useq_sequence": sequence,
            "channel_labels": channel_labels,
            "lazy_mda_visualization": True,
            "raman_channel": "RM",
        }
        is_rgb = len(frame_shape) == 3 and frame_shape[-1] in (3, 4)
        scale = [1.0] * (len(shape) - (1 if is_rgb else 0))
        pixel_size = float(self._mmc.getPixelSizeUm())
        if pixel_size:
            scale[-2:] = [pixel_size, pixel_size]
        channel_summary = ", ".join(channel_labels) or "Camera"
        layer = self.viewer.add_image(
            array,
            name=f"MDA ({channel_summary})",
            visible=True,
            scale=scale,
            metadata=metadata,
        )
        axis_labels = [*axes, "y", "x"]
        if len(self.viewer.dims.axis_labels) == len(axis_labels):
            self.viewer.dims.axis_labels = axis_labels
        return axes, array, layer

    def _event_index(self, event, axes):
        return tuple(event.index.get(axis, 0) for axis in axes)

    def _show_event(self, event, axes):
        current = list(self.viewer.dims.current_step)
        for axis_number, value in enumerate(self._event_index(event, axes)):
            if axis_number < len(current):
                current[axis_number] = value
        self.viewer.dims.current_step = current

    @ensure_main_thread
    def _on_mda_frame(self, image, event, *_metadata):
        image = np.asarray(image)
        if self._run_layer is None:
            self._run_layer = self._create_stack(event, image=image)
        axes, array, layer = self._run_layer
        index = self._event_index(event, axes)
        array[index] = image
        layer.refresh()
        self._show_event(event, axes)

    @staticmethod
    def _spectrum_image(spectra, wavenumbers, shape, dtype):
        """Rasterize a spectrum into an existing camera-sized image plane."""
        spectra = np.asarray(spectra)
        if spectra.ndim == 1:
            values = spectra
        else:
            values = np.mean(spectra.reshape(-1, spectra.shape[-1]), axis=0)
        values = np.asarray(values, dtype=np.float64)

        height, width = shape[:2]
        canvas = np.zeros((height, width), dtype=dtype)
        if values.size < 2 or height < 4 or width < 4:
            return canvas

        if wavenumbers is None:
            x_values = np.arange(values.size, dtype=np.float64)
        else:
            x_values = np.asarray(wavenumbers, dtype=np.float64)
            if x_values.ndim != 1 or x_values.size != values.size:
                x_values = np.arange(values.size, dtype=np.float64)
        finite = np.isfinite(x_values) & np.isfinite(values)
        if np.count_nonzero(finite) < 2:
            return canvas
        x_values = x_values[finite]
        values = values[finite]
        order = np.argsort(x_values)
        x_values = x_values[order]
        values = values[order]

        left = max(1, width // 20)
        right = max(left + 1, width - left - 1)
        top = max(1, height // 20)
        bottom = max(top + 1, height - top - 1)
        columns = np.arange(left, right + 1)
        sampled = np.interp(
            np.linspace(x_values[0], x_values[-1], columns.size),
            x_values,
            values,
        )
        y_min = float(np.min(sampled))
        y_max = float(np.max(sampled))
        if y_max == y_min:
            normalized = np.full(sampled.shape, 0.5)
        else:
            normalized = (sampled - y_min) / (y_max - y_min)
        rows = bottom - np.rint(normalized * (bottom - top)).astype(int)

        if np.issubdtype(np.dtype(dtype), np.integer):
            line_value = np.iinfo(dtype).max
        else:
            line_value = 1.0
        axis_value = line_value // 4 if np.issubdtype(
            np.dtype(dtype), np.integer
        ) else 0.25
        canvas[bottom, left:right + 1] = axis_value
        canvas[top:bottom + 1, left] = axis_value
        canvas[rows, columns] = line_value
        canvas[np.maximum(top, rows - 1), columns] = line_value

        if len(shape) == 3:
            rgb = np.zeros(shape, dtype=dtype)
            rgb[..., 1] = canvas
            rgb[..., 2] = canvas
            return rgb
        return canvas

    @ensure_main_thread
    def add_raman_spectrum(self, event, spectra, wavenumbers=None):
        """Put one acquired spectrum into the RM plane of the shared MDA stack."""
        if self._run_layer is None:
            self._run_layer = self._create_stack(event)
        axes, array, layer = self._run_layer
        image = self._spectrum_image(
            spectra,
            wavenumbers,
            self._frame_shape,
            array.dtype,
        )
        array[self._event_index(event, axes)] = image
        layer.refresh()
        self._show_event(event, axes)

    @ensure_main_thread
    def _on_mda_finished(self, _sequence):
        self._mda_running = False


def install_lazy_mda_viewer(main_window, core, viewer):
    """Replace napari-micromanager's eager MDA viewer for this window."""
    core_link = getattr(main_window, "_core_link", None)
    if core_link is None:
        return None
    old_handler = getattr(core_link, "_mda_handler", None)
    if isinstance(old_handler, LazyMDAViewer):
        return old_handler
    if old_handler is not None:
        old_handler._cleanup()
    handler = LazyMDAViewer(core, viewer)
    core_link._mda_handler = handler
    return handler
