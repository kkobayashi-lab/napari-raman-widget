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
        return (height or 512, width or 512, 3)

    def _create_stack(self, event, *, image=None):
        sequence = event.sequence or self._sequence
        axes = self._storage_axes(sequence)
        if image is None:
            frame_shape = self._camera_shape()
            dtype = self._camera_dtype()
        else:
            image = np.asarray(image)
            frame_shape = (*image.shape[:2], 3)
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
        scale = [1.0] * (len(shape) - 1)
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
        array[index] = self._camera_image_rgb(image, array.dtype)
        layer.refresh()
        self._show_event(event, axes)

    @staticmethod
    def _camera_image_rgb(image, dtype):
        """Return a camera frame as RGB without changing its intensity."""
        image = np.asarray(image, dtype=dtype)
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] >= 3:
            return image[..., :3]
        raise ValueError(f"Unsupported camera frame shape: {image.shape}")

    @staticmethod
    def _group_spectra_by_point(spectra, points=None):
        """Average repeated spectral frames collected at the same point."""
        spectra = np.asarray(spectra)
        if spectra.ndim == 1:
            frames = spectra[None, :]
        else:
            frames = spectra.reshape(-1, spectra.shape[-1])
        frames = np.asarray(frames, dtype=np.float64)

        if points is None:
            points_array = np.empty((0, 2), dtype=float)
        else:
            points_array = np.atleast_2d(np.asarray(points, dtype=float))
        if len(frames) == 1 or len(points_array) != len(frames):
            return frames, [None] * len(frames), np.ones(len(frames), dtype=int)

        grouped_indices = []
        grouped_points = []
        for frame_index, point in enumerate(points_array):
            for group_index, known_point in enumerate(grouped_points):
                if np.allclose(point, known_point, rtol=0, atol=1e-6):
                    grouped_indices[group_index].append(frame_index)
                    break
            else:
                grouped_points.append(point.copy())
                grouped_indices.append([frame_index])

        averaged = np.stack(
            [np.mean(frames[indices], axis=0) for indices in grouped_indices]
        )
        counts = np.asarray([len(indices) for indices in grouped_indices])
        return averaged, grouped_points, counts

    @staticmethod
    def _convert_rgb_dtype(image, dtype):
        """Convert an Agg uint8 RGB buffer to the MDA stack's dtype."""
        dtype = np.dtype(dtype)
        image = np.asarray(image)
        if dtype == np.dtype("uint8"):
            return image
        normalized = image.astype(np.float64) / 255.0
        if np.issubdtype(dtype, np.integer):
            return np.rint(normalized * np.iinfo(dtype).max).astype(dtype)
        if np.issubdtype(dtype, np.floating):
            return normalized.astype(dtype)
        return (normalized >= 0.5).astype(dtype)

    @classmethod
    def _spectrum_image(
        cls,
        spectra,
        wavenumbers,
        shape,
        dtype,
        *,
        points=None,
        which=None,
        title=None,
    ):
        """Render point-grouped Raman spectra using off-screen Matplotlib."""
        # Keep Matplotlib lazy: camera-only runs never import it.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        traces, grouped_points, counts = cls._group_spectra_by_point(
            spectra, points
        )
        spectrum_length = traces.shape[-1]

        if wavenumbers is None:
            x_values = np.arange(spectrum_length, dtype=np.float64)
            x_label = "Detector pixel"
        else:
            x_values = np.asarray(wavenumbers, dtype=np.float64)
            if x_values.ndim != 1 or x_values.size != spectrum_length:
                x_values = np.arange(spectrum_length, dtype=np.float64)
                x_label = "Detector pixel"
            else:
                x_label = "Raman shift (cm$^{-1}$)"

        height, width = shape[:2]
        dpi = 100
        figure = Figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor="#181818",
        )
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        axis.set_facecolor("#181818")

        point_sources = [None] * len(grouped_points)
        if points is not None and which is not None:
            points_array = np.atleast_2d(np.asarray(points, dtype=float))
            source_names = [str(name) for name in which]
            if len(points_array) == len(source_names):
                for point_index, point in enumerate(grouped_points):
                    if point is None:
                        continue
                    matches = np.all(
                        np.isclose(
                            points_array,
                            point,
                            rtol=0,
                            atol=1e-6,
                        ),
                        axis=1,
                    )
                    names = list(
                        dict.fromkeys(
                            source_names[index]
                            for index in np.flatnonzero(matches)
                        )
                    )
                    if names:
                        point_sources[point_index] = "/".join(names)

        for trace_index, (trace, point, count) in enumerate(
            zip(traces, grouped_points, counts)
        ):
            finite = np.isfinite(x_values) & np.isfinite(trace)
            if np.count_nonzero(finite) < 2:
                continue
            label = f"Point {trace_index + 1}"
            if point is not None:
                label += f" ({point[0]:.3f}, {point[1]:.3f})"
            if point_sources[trace_index]:
                label += f" [{point_sources[trace_index]}]"
            if count > 1:
                label += f", mean of {count}"
            axis.plot(
                x_values[finite],
                trace[finite],
                linewidth=1.25,
                label=label,
            )

        axis.set_xlabel(x_label, color="white")
        axis.set_ylabel("Intensity", color="white")
        if title:
            axis.set_title(title, color="white")
        axis.tick_params(colors="white", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#b0b0b0")
        axis.grid(color="white", alpha=0.12, linewidth=0.5)
        if len(traces) > 1 or np.any(counts > 1):
            legend = axis.legend(
                loc="best",
                fontsize=7,
                facecolor="#202020",
                edgecolor="#808080",
            )
            for text in legend.get_texts():
                text.set_color("white")
        figure.tight_layout(pad=0.7)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        return cls._convert_rgb_dtype(rgb, dtype)

    @ensure_main_thread
    def add_raman_spectrum(
        self,
        event,
        spectra,
        wavenumbers=None,
        *,
        points=None,
        which=None,
    ):
        """Put one acquired spectrum into the RM plane of the shared MDA stack."""
        if self._run_layer is None:
            self._run_layer = self._create_stack(event)
        axes, array, layer = self._run_layer
        index = getattr(event, "index", {})
        title = (
            f"Raman t={index.get('t', 0)}, "
            f"p={index.get('p', 0)}, z={index.get('z', 0)}, "
            f"c={index.get('c', 0)}"
        )
        image = self._spectrum_image(
            spectra,
            wavenumbers,
            self._frame_shape,
            array.dtype,
            points=points,
            which=which,
            title=title,
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
