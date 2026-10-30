"""Persistent, single-point visual reference for the Raman focus."""

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QSettings


class RamanFocusMarker:
    KEY = "raman/focus_yx"

    def __init__(self, viewer, settings):
        self.viewer = viewer
        self.settings = settings
        self.layer = None
        self._updating = False
        self._point = None
        try:
            point = np.asarray(settings.value(self.KEY), dtype=float)
            if point.shape == (2,) and np.isfinite(point).all():
                self._point = point
        except (TypeError, ValueError):
            pass
        self._waiting_for_image = self._point is None
        if self._point is None:
            self._point = self._image_center()
        self._ensure_layer()
        viewer.layers.events.inserted.connect(self._image_inserted)

    def _image_center(self):
        # Prefer the camera frame over other images already open in the viewer.
        layers = sorted(self.viewer.layers, key=lambda layer: layer.name != "preview")
        for layer in layers:
            if isinstance(layer, Image):
                data = layer.data[0] if layer.multiscale else layer.data
                shape = data.shape[:-1] if layer.rgb else data.shape
                center = (np.asarray(shape, dtype=float) - 1) / 2
                self._waiting_for_image = False
                return np.asarray(layer.data_to_world(center)[-2:])
        # Default camera frame until the first image arrives.
        return np.array([511.5, 671.5])

    def _ensure_layer(self):
        if self.layer is not None and self.layer in self.viewer.layers:
            return
        selected = list(self.viewer.layers.selection)
        active = self.viewer.layers.selection.active
        self.layer = self.viewer.add_points(
            [self._point], name="Raman focus", ndim=2,
            symbol="cross", size=18, face_color="yellow",
            metadata={"raman_focus_marker": True},
        )
        self.layer.mode = "select"
        self.layer.events.data.connect(self._data_changed)
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.update(selected)
        self.viewer.layers.selection.active = active

    def _data_changed(self, event=None):
        if self._updating:
            return
        data = np.asarray(self.layer.data)
        if len(data) and np.isfinite(data[-1]).all():
            self._point = data[-1].copy()
            self._waiting_for_image = False
        # Adding replaces the reference; deleting restores it.
        if data.shape != (1, 2) or not np.isfinite(data).all():
            self._set_point()

    def _set_point(self):
        self._updating = True
        try:
            self.layer.data = np.array([self._point])
        finally:
            self._updating = False

    def _image_inserted(self, event):
        if isinstance(event.value, Image):
            if self._waiting_for_image:
                self._point = self._image_center()
                self._set_point()
            self._raise_above_images()

    def _raise_above_images(self):
        layers = self.viewer.layers
        if self.layer is not None and self.layer in layers:
            index = layers.index(self.layer)
            if any(isinstance(layer, Image) for layer in layers[index + 1:]):
                # Napari draws later entries above earlier ones. The preview
                # can be created or replaced after hardware connection.
                layers.move(index, len(layers))

    def show(self):
        self._ensure_layer()
        self.layer.visible = True
        self._raise_above_images()

    def center(self):
        self._waiting_for_image = True
        self._point = self._image_center()
        self._ensure_layer()
        self._set_point()
        self.show()

    def save(self):
        self._ensure_layer()
        self.settings.setValue(self.KEY, self._point.tolist())
        self.settings.sync()
        if self.settings.status() != QSettings.NoError:
            raise OSError("Could not write Raman focus settings")
        self._waiting_for_image = False
        return self._point.copy()
