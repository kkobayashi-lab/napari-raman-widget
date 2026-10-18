"""Position-owned Raman selections, with optional indexed image provenance."""
from copy import deepcopy

import numpy as np
from useq import MDASequence, Position

TARGET_KEY = "raman_target"


def target(position):
    return (position.sequence.metadata if position.sequence else {}).get(TARGET_KEY)


def restore_position_targets(sequence):
    """Restore position settings from a saved execution snapshot's p table."""
    policies = sequence.metadata.get("raman_positions", {})
    saved_references = sequence.metadata.get("autofocus_position_references", {})
    if not policies and not saved_references:
        return sequence
    positions = []
    for p, pos in enumerate(sequence.stage_positions):
        if not target(pos) and str(p) in policies:
            sub = pos.sequence or MDASequence()
            pos = pos.replace(sequence=sub.replace(metadata={**sub.metadata, TARGET_KEY: deepcopy(policies[str(p)])}))
        if str(p) in saved_references:
            from .autofocus_references import with_reference, REFERENCE_KEY
            if not pos.sequence or REFERENCE_KEY not in pos.sequence.metadata:
                pos = with_reference(pos, saved_references[str(p)])
        positions.append(pos)
    metadata = dict(sequence.metadata)
    metadata.pop("raman_positions", None)
    metadata.pop("autofocus_position_references", None)
    return sequence.replace(stage_positions=positions, metadata=metadata)


def selected_positions(image_sequence, rows, shape, model=None, *, batch=False, provenance=True, dataset=None, reference_sequence=None, batch_center=None):
    """Convert viewer t,p,g,c,z,y,x points to explicit acquisition positions."""
    from .autofocus_references import reference, with_reference, source_stage_xy
    from .experiment import position_id
    parents = image_sequence.stage_positions if reference_sequence is None else reference_sequence.stage_positions
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
    from raman_mda_engine.grid_plan import iter_grid_events
    for event in iter_grid_events(image_sequence):
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
        center = np.asarray(batch_center, dtype=float) if batch_center is not None else pixels[0]
        if batch_center is not None and (center.shape != (2,) or not np.isfinite(center).all()
                                        or np.any(center < 0) or np.any(center >= shape)):
            raise ValueError("Batch center must be inside the source image")
        if not batch or batch_center is not None:
            if model is None:
                raise ValueError("Load the pixel-to-stage calibration before applying Raman points")
            coefficients, degree = model
            x, y = (center - shape / 2)[::-1]
            terms = [x**j * y**(n-j) for n in range(int(degree)+1) for j in range(n+1)]
            offset = np.asarray(terms) @ np.asarray(coefficients)
            if offset.shape != (2,) or not np.isfinite(offset).all():
                raise ValueError("Invalid pixel-to-stage calibration")
            xyz[:2] = (np.asarray(xyz[:2]) - offset).tolist()
        # Runtime aiming convention is pixel (y,x) / (width,height).
        sampling = pixels if batch else np.asarray([shape / 2])
        if batch and batch_center is not None:
            sampling = pixels - center + shape / 2
            if np.any(sampling < 0) or np.any(sampling >= shape):
                raise ValueError("Batch extends outside the aiming field after centering; use a smaller region")
        value = dict(mode="integrated_batch" if batch else "centered",
                     points=(sampling / shape[::-1]).tolist(), image_shape=shape[::-1].tolist())
        if provenance:
            value["source_image"] = dict(index=dict(zip("tpgcz", map(int, key))), pixels_yx=pixels.tolist(),
                                         stage_xyz_um=[event.x_pos, event.y_pos, event.z_pos],
                                         image_shape_yx=shape.tolist())
            if dataset:
                value["source_image"]["dataset"] = str(dataset)
            if model is not None:
                value["source_image"]["pixel_to_stage"] = dict(coefficients=np.asarray(model[0]).tolist(), degree=int(model[1]))
        positions.append(Position(x=xyz[0], y=xyz[1], z=xyz[2],
                                  name=f"Raman {'batch' if batch else 'point'} {len(positions)+1}",
                                  sequence=MDASequence(metadata={TARGET_KEY: value})))
        source_p = key[1]
        if source_p < len(image_sequence.stage_positions):
            original = image_sequence.stage_positions[source_p]
            matches = [p for p in parents if (p.x, p.y, p.z) == (original.x, original.y, original.z)
                       and position_id(p) == position_id(original)]
            parent = matches[0] if len(matches) == 1 else None
            ref = reference(parent) if parent is not None else None
            if ref:
                inherited = deepcopy(ref)
                if "stage_xy_um" not in inherited:
                    if model is None:
                        raise ValueError("Load the pixel-to-stage calibration to inherit autofocus references")
                    inherited["stage_xy_um"] = source_stage_xy(dict(
                        pixel_to_stage=dict(coefficients=np.asarray(model[0]).tolist(), degree=int(model[1])),
                        stage_xyz_um=[parent.x, parent.y, parent.z], image_shape_yx=ref["image_shape"][::-1]),
                        np.asarray(ref["point"]) * ref["image_shape"])
                inherited["stage"] = xyz
                positions[-1] = with_reference(positions[-1], inherited)
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
        value.update(points=points, image_shape=shape.tolist())
        policies[str(p)] = value
    return record, policies


def install_raman_tab(panel, owner):
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget
    from .autofocus_references import choose_references
    tab = QWidget(panel)
    layout = QVBoxLayout(tab)
    text = QLabel("Select Raman points or draw batch regions on existing images or configured imaging.\n"
                  "Ordinary points become centered stage positions. Integrated batches use the galvo.\n"
                  "Add RM in Channels for spectra; add camera channels only for fresh images.")
    text.setWordWrap(True)
    layout.addWidget(text)
    button = QPushButton("Select Raman points / batches…")
    def choose():
        try:
            choose_references(owner, raman=True)
        except Exception as error:
            owner.status.setText(f"Status: Raman selection — {error}")
    button.clicked.connect(choose)
    layout.addWidget(button)
    layout.addWidget(owner._raman_settings_panel)
    owner._raman_settings_panel.show()
    layout.addStretch()
    QTabWidget.addTab(panel.tab_wdg, tab, "Raman selection")


class RamanEditor:
    def __init__(self, owner, sequence, image_layer):
        import napari
        from qtpy.QtCore import QObject, QEvent, QTimer
        from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox, QComboBox,
                                   QSpinBox, QTableWidget, QHeaderView)
        self.owner, self.sequence, self.image_layer = owner, sequence, image_layer
        self.image_sequence = image_layer.metadata["useq_sequence"]
        self.shape = tuple(image_layer.data.shape[-3:-1] if image_layer.rgb else image_layer.data.shape[-2:])
        self.viewer = napari.Viewer(title="Raman selection — points and batches")
        self.viewer.add_image(image_layer.data, name=image_layer.name, rgb=image_layer.rgb,
                              scale=image_layer.scale, contrast_limits=image_layer.contrast_limits,
                              metadata=dict(image_layer.metadata))
        self.viewer.dims.axis_labels = tuple("tpgczyx")
        self.points = self.viewer.add_points(np.empty((0, 7)), ndim=7, name="Raman selections",
                                             face_color="orange", size=12, scale=image_layer.scale,
                                             out_of_slice_display=False)
        self.points.mode = "add"
        from .batch_shapes import BatchShapes
        self.shapes = self.viewer.add_layer(BatchShapes(
            ndim=7, name="Raman batches", scale=image_layer.scale,
            edge_color="cyan", face_color=[0, 1, 1, .15],
            features={"batch_id": np.array([], dtype=object), "count": np.array([], dtype=int)},
            feature_defaults={"batch_id": "", "count": 25}))
        self.preview = self.viewer.add_points(
            np.empty((0, 7)), ndim=7, name="Batch sampling preview", size=5,
            face_color="cyan", scale=image_layer.scale, out_of_slice_display=False)
        self.preview.editable = False
        self.shapes.visible = self.preview.visible = False
        self.viewer.layers.selection.active = self.points
        self._batches = []
        self._refreshing = False
        controls = QWidget()
        layout = QVBoxLayout(controls)
        help_text = QLabel("Individual points: click targets on each image plane.\n"
                          "Batches: draw rectangles, ellipses, or polygons using napari's Shapes tools.\n"
                          "The stage moves to each shape's center; the galvo samples within that batch.\n"
                          "Each shape produces one integrated spectrum per repeat. RM exposure is for the whole batch.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.batch = QCheckBox("Integrated batch (galvo)")
        self.default_count = QSpinBox()
        self.default_count.setRange(1, 10000)
        self.default_count.setValue(25)
        self.default_count.setPrefix("Points per new batch: ")
        self.default_count.valueChanged.connect(self._set_default_count)
        self.batch_table = QTableWidget(0, 3)
        self.batch_table.setHorizontalHeaderLabels(["Batch", "Region", "Points"])
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.cellClicked.connect(self._select_batch)
        self.default_count.setVisible(False)
        self.batch_table.setVisible(False)
        self.provenance = QCheckBox("Attach optional source-image metadata")
        self.provenance.setChecked(True)
        self.destination = QComboBox()
        self.destination.addItems(["Replace acquisition positions", "Append positions"])
        for widget in (self.batch, self.provenance, self.destination):
            layout.addWidget(widget)
        layout.addWidget(self.default_count)
        layout.addWidget(self.batch_table)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self._preview_timer = QTimer(controls)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self.refresh_batches)
        self.shapes.events.data.connect(self._schedule_preview)
        self.shapes.events.features.connect(self._schedule_preview)
        self.points.events.data.connect(self._update_summary)
        self.batch.toggled.connect(self._toggle_batch)
        self._update_summary()
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
                    editor._preview_timer.stop()
                    editor.owner._raman_editor = None
                return False
        self.close_filter = CloseFilter(self.viewer.window._qt_window)
        self.viewer.window._qt_window.installEventFilter(self.close_filter)
        self._restore_batches()

    def _set_default_count(self, value):
        self.shapes.feature_defaults["count"] = value

    def _select_batch(self, row, _column):
        self.viewer.layers.selection.active = self.shapes
        self.shapes.mode = "select"
        self.shapes.selected_data = {row}
        vertices = self.shapes.data[row]
        for axis in range(5):
            self.viewer.dims.set_point(axis, self.shapes.data_to_world(vertices[0])[axis])

    def _toggle_batch(self, enabled):
        self.points.visible = not enabled
        self.shapes.visible = self.preview.visible = enabled
        self.default_count.setVisible(enabled)
        self.batch_table.setVisible(enabled)
        self.viewer.layers.selection.active = self.shapes if enabled else self.points
        if enabled:
            self.shapes.mode = "add_polygon"
        self._update_summary()

    def _schedule_preview(self, *_):
        if not self._refreshing:
            self._preview_timer.start()

    def _update_summary(self, *_):
        channels = ", ".join(c.config for c in self.owner._unified_mda.value().channels) or "none"
        selection = (f"{len(self._batches)} batches, {len(self.preview.data)} sampling points"
                     if self.batch.isChecked() else f"{len(self.points.data)} individual points")
        self.summary.setText(f"{selection}. Active channels: {channels}")

    def _change_count(self, row, count):
        features = self.shapes.features.copy()
        features.loc[row, "count"] = count
        self.shapes.features = features
        self.refresh_batches()

    def refresh_batches(self):
        from uuid import uuid4
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QTableWidgetItem, QSpinBox
        from .raman_batches import sample_region
        self._preview_timer.stop()
        self._refreshing = True
        self._batches = []
        errors = []
        try:
            features = self.shapes.features.copy()
            seen = set()
            self.batch_table.setRowCount(len(self.shapes.data))
            for row, (vertices, kind) in enumerate(zip(self.shapes.data, self.shapes.shape_type)):
                identity = str(features.loc[row, "batch_id"] or "")
                if not identity or identity in seen:
                    identity = str(uuid4())
                    features.loc[row, "batch_id"] = identity
                seen.add(identity)
                count = int(features.loc[row, "count"])
                for column, value in enumerate((f"Batch {row + 1}", kind)):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.batch_table.setItem(row, column, item)
                spin = self.batch_table.cellWidget(row, 2)
                if spin is None:
                    spin = QSpinBox()
                    spin.setRange(1, 10000)
                    spin.setKeyboardTracking(False)
                    spin.valueChanged.connect(lambda n, r=row: self._change_count(r, n))
                    self.batch_table.setCellWidget(row, 2, spin)
                spin.blockSignals(True)
                spin.setValue(count)
                spin.blockSignals(False)
                try:
                    points = sample_region(vertices, kind, count, self.shape)
                    self._batches.append(dict(id=identity, name=f"Raman batch {row + 1}",
                        shape_type=kind, vertices=np.asarray(vertices).tolist(), count=count,
                        source_sequence_uid=str(self.image_sequence.uid), points=points.tolist()))
                except ValueError as error:
                    errors.append(f"Batch {row + 1}: {error}")
            self.shapes.features = features
            # New shapes always receive the default, rather than a selected shape's count/id.
            self.shapes.feature_defaults["count"] = self.default_count.value()
            self.shapes.feature_defaults["batch_id"] = ""
            self.preview.data = (np.concatenate([b["points"] for b in self._batches])
                                 if self._batches else np.empty((0, 7)))
            self.error.setText("\n".join(errors))
            self._update_summary()
        finally:
            self._refreshing = False
        return not errors

    def _restore_batches(self):
        regions = [target(p).get("batch_region") for p in self.sequence.stage_positions if target(p)]
        regions = [r for r in regions if r and r.get("source_sequence_uid") == str(self.image_sequence.uid)]
        if not regions:
            return
        self.shapes.add([r["vertices"] for r in regions], shape_type=[r["shape_type"] for r in regions])
        self.shapes.features = {"batch_id": [r["id"] for r in regions], "count": [r["count"] for r in regions]}
        self.refresh_batches()
        self.batch.setChecked(True)

    def apply(self):
        from .autofocus_references import _require_idle
        from .experiment import METADATA_KEY
        try:
            _require_idle(self.owner)
            panel = self.owner._unified_mda
            current = panel.value()
            if current.stage_positions != self.sequence.stage_positions:
                raise ValueError("Positions changed while selecting; reopen the selector")
            if self.owner._ensure_current_vandermonde() is None:
                raise ValueError("Load the current objective's pixel-to-stage calibration")
            model = self.owner.vandermonde
            options = dict(provenance=self.provenance.isChecked(),
                           dataset=self.image_layer.metadata.get("dataset_path"), reference_sequence=current)
            if self.batch.isChecked():
                from .raman_batches import batch_positions
                if not self.refresh_batches():
                    return
                positions = batch_positions(self.image_sequence, self._batches, self.shape, model, **options)
                applied_points = self.preview.data
            else:
                positions = selected_positions(self.image_sequence, self.points.data, self.shape, model, **options)
                applied_points = self.points.data
            if self.destination.currentIndex() == 1:
                if current.grid_plan or "raman_grid" in current.metadata:
                    raise ValueError("Replace grid positions, or expand the grid before appending Raman positions")
                positions = [*current.stage_positions, *positions]
            metadata = deepcopy(current.metadata)
            metadata.pop("raman_grid", None)
            meta = metadata.setdefault(METADATA_KEY, {})
            meta["targets"] = []  # Position-owned targets are compiled at run time.
            meta.setdefault("extensions", {}).setdefault("raman", {})["camera_positions"] = "all"
            meta.pop("wellplate_plan", None)
            panel.setValue(current.replace(stage_positions=positions, grid_plan=None, metadata=metadata))
            if hasattr(self.owner, "viewer"):
                self.owner.viewer.add_points(applied_points.copy(), ndim=7, name="Selected Raman targets",
                    face_color="orange", size=12, scale=self.image_layer.scale,
                    out_of_slice_display=False, metadata={"raman_selection": True})
            self.owner.status.setText(f"Status: applied {len(positions)} positions; configure acquisition channels in Channels")
            self.viewer.close()
        except (ValueError, RuntimeError) as error:
            self.error.setText(str(error))
