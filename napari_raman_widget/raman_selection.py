"""Position-owned Raman selections, with optional indexed image provenance."""
from copy import deepcopy

import numpy as np
from useq import MDASequence, Position

TARGET_KEY = "raman_target"


def target(position):
    return (position.sequence.metadata if position.sequence else {}).get(TARGET_KEY)


def selected_positions(image_sequence, rows, shape, model=None, *, batch=False, provenance=True, dataset=None):
    """Convert viewer t,p,g,c,z,y,x points to explicit acquisition positions."""
    rows = np.asarray(rows, dtype=float).reshape((-1, 7))
    shape = np.asarray(shape, dtype=float)
    if not len(rows):
        raise ValueError("Select at least one Raman point")
    if (not np.isfinite(rows).all() or np.any(rows[:, :5] != np.round(rows[:, :5]))
            or np.any(rows[:, :5] < 0) or np.any(rows[:, -2:] < 0)
            or np.any(rows[:, -2:] >= shape)):
        raise ValueError("Place Raman points inside an acquired image plane")
    keys = {tuple(row[:5].astype(int)) for row in rows}
    events = {}
    for event in image_sequence:
        key = tuple(event.index.get(a, 0) for a in "tpgcz")
        if key in keys:
            events[key] = event
        if len(events) == len(keys):
            break
    if set(events) != keys:
        raise ValueError("A selected plane is not present in the source acquisition")
    groups = {}
    for i, row in enumerate(rows):
        key = tuple(row[:5].astype(int))
        groups.setdefault(key if batch else (i, key), []).append(row)
    positions = []
    for group in groups.values():
        key = tuple(group[0][:5].astype(int))
        event = events[key]
        if event.channel and event.channel.config == "RM":
            raise ValueError("Select Raman targets on a camera image channel")
        xyz = [event.x_pos, event.y_pos, event.z_pos]
        if any(v is None or not np.isfinite(v) for v in xyz):
            raise ValueError("The source image needs recorded stage XYZ coordinates")
        pixels = np.asarray(group)[:, -2:]
        if not batch:
            if model is None:
                raise ValueError("Load the pixel-to-stage calibration before applying Raman points")
            coefficients, degree = model
            x, y = (pixels[0] - shape / 2)[::-1]
            terms = [x**j * y**(n-j) for n in range(int(degree)+1) for j in range(n+1)]
            offset = np.asarray(terms) @ np.asarray(coefficients)
            if offset.shape != (2,) or not np.isfinite(offset).all():
                raise ValueError("Invalid pixel-to-stage calibration")
            xyz[:2] = (np.asarray(xyz[:2]) - offset).tolist()
        # Runtime aiming convention is pixel (y,x) / (width,height).
        sampling = pixels if batch else np.asarray([shape / 2])
        value = dict(mode="integrated_batch" if batch else "centered",
                     points=(sampling / shape[::-1]).tolist(), image_shape=shape[::-1].tolist())
        if provenance:
            value["source_image"] = dict(index=dict(zip("tpgcz", key)), pixels_yx=pixels.tolist(),
                                         stage_xyz_um=[event.x_pos, event.y_pos, event.z_pos],
                                         image_shape_yx=shape.tolist())
            if dataset:
                value["source_image"]["dataset"] = str(dataset)
            if model is not None:
                value["source_image"]["pixel_to_stage"] = dict(coefficients=np.asarray(model[0]).tolist(), degree=int(model[1]))
        positions.append(Position(x=xyz[0], y=xyz[1], z=xyz[2],
                                  name=f"Raman {'batch' if batch else 'point'} {len(positions)+1}",
                                  sequence=MDASequence(metadata={TARGET_KEY: value})))
    return positions


def position_records(sequence, image_shape):
    """Build detached sources indexed by this run's p; no target UUIDs."""
    shape = np.asarray(image_shape, dtype=float)  # width, height
    center = (shape[::-1] / 2 / shape).tolist()
    record = dict(id="raman-positions", name="Raman positions", role="cell", indexed=True,
                  image_shape=shape.tolist(), points={}, anchors={})
    policies = {}
    for p, pos in enumerate(sequence.stage_positions):
        value = deepcopy(target(pos) or dict(mode="centered", points=[center], image_shape=shape.tolist()))
        if value["mode"] not in ("centered", "integrated_batch"):
            raise ValueError("Unknown Raman acquisition mode")
        # Centered targets stay centered even if camera dimensions changed.
        points = [center] if value["mode"] == "centered" else value["points"]
        if value["mode"] == "integrated_batch" and list(value["image_shape"]) != shape.tolist():
            raise ValueError("Batch image dimensions changed; select batch points again")
        values = np.asarray(points, dtype=float).reshape((-1, 2))
        pixels = values * shape
        if not len(values) or not np.isfinite(values).all() or np.any(pixels < 0) or np.any(pixels >= shape[::-1]):
            raise ValueError("Raman sampling points must be inside the image")
        record["points"][str(p)] = record["anchors"][str(p)] = points
        policies[str(p)] = value
    return record, policies


def install_raman_tab(panel, owner):
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget
    from .autofocus_references import choose_references
    tab = QWidget(panel)
    layout = QVBoxLayout(tab)
    text = QLabel("Select Raman points on existing images or acquire configured imaging.\n"
                  "Ordinary points become centered stage positions. Integrated batches use the galvo.\n"
                  "Add RM in Channels for spectra; add camera channels only for fresh images.")
    text.setWordWrap(True)
    layout.addWidget(text)
    button = QPushButton("Select Raman points…")
    def choose():
        try:
            choose_references(owner, raman=True)
        except Exception as error:
            owner.status.setText(f"Status: Raman selection — {error}")
    button.clicked.connect(choose)
    layout.addWidget(button)
    layout.addStretch()
    QTabWidget.addTab(panel.tab_wdg, tab, "Raman")


class RamanEditor:
    def __init__(self, owner, sequence, image_layer):
        import napari
        from qtpy.QtCore import QObject, QEvent
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox, QComboBox
        self.owner, self.sequence, self.image_layer = owner, sequence, image_layer
        self.image_sequence = image_layer.metadata["useq_sequence"]
        self.shape = tuple(image_layer.data.shape[-3:-1] if image_layer.rgb else image_layer.data.shape[-2:])
        self.viewer = napari.Viewer(title="Select Raman points")
        self.viewer.add_image(image_layer.data, name=image_layer.name, rgb=image_layer.rgb,
                              scale=image_layer.scale, contrast_limits=image_layer.contrast_limits,
                              metadata=dict(image_layer.metadata))
        self.viewer.dims.axis_labels = tuple("tpgczyx")
        self.points = self.viewer.add_points(np.empty((0, 7)), ndim=7, name="Raman selections",
                                             face_color="orange", size=12, scale=image_layer.scale,
                                             out_of_slice_display=False)
        self.points.mode = "add"
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.addWidget(QLabel("Click targets on each image plane. Select and delete to remove points.\n"
                               "Batch mode integrates all selected points on each plane into one spectrum."))
        self.batch = QCheckBox("Integrated batch (galvo)")
        self.provenance = QCheckBox("Attach optional source-image metadata")
        self.provenance.setChecked(True)
        self.destination = QComboBox()
        self.destination.addItems(["Replace acquisition positions", "Append positions"])
        for widget in (self.batch, self.provenance, self.destination):
            layout.addWidget(widget)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        def summary(*_):
            channels = ", ".join(c.config for c in owner._unified_mda.value().channels) or "none"
            self.summary.setText(f"{len(self.points.data)} selected points. Active channels: {channels}")
        self.points.events.data.connect(summary)
        summary()
        apply = QPushButton("Apply Raman positions")
        cancel = QPushButton("Cancel")
        apply.clicked.connect(self.apply)
        cancel.clicked.connect(self.viewer.close)
        layout.addWidget(apply)
        layout.addWidget(cancel)
        self.viewer.window.add_dock_widget(controls, area="right", name="Raman")
        editor = self
        class CloseFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Close:
                    editor.owner._raman_editor = None
                return False
        self.close_filter = CloseFilter(self.viewer.window._qt_window)
        self.viewer.window._qt_window.installEventFilter(self.close_filter)

    def apply(self):
        from .autofocus_references import _require_idle
        from .experiment import METADATA_KEY
        try:
            _require_idle(self.owner)
            panel = self.owner._unified_mda
            current = panel.value()
            if current.stage_positions != self.sequence.stage_positions:
                raise ValueError("Positions changed while selecting; reopen the selector")
            model = None
            if not self.batch.isChecked():
                self.owner._ensure_current_vandermonde()
                model = self.owner.vandermonde
            positions = selected_positions(self.image_sequence, self.points.data, self.shape, model,
                batch=self.batch.isChecked(), provenance=self.provenance.isChecked(),
                dataset=self.image_layer.metadata.get("dataset_path"))
            if self.destination.currentIndex() == 1:
                positions = [*current.stage_positions, *positions]
            metadata = deepcopy(current.metadata)
            metadata.pop("raman_grid", None)
            meta = metadata.setdefault(METADATA_KEY, {})
            meta["targets"] = []  # Position-owned targets are compiled at run time.
            meta.pop("wellplate_plan", None)
            panel.setValue(current.replace(stage_positions=positions, grid_plan=None, metadata=metadata))
            self.owner.status.setText(f"Status: applied {len(positions)} positions; configure acquisition channels in Channels")
            self.viewer.close()
        except (ValueError, RuntimeError) as error:
            self.error.setText(str(error))
