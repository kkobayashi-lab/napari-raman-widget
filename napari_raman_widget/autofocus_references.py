"""Select autofocus references without creating or changing Raman cell targets."""
from copy import deepcopy

import numpy as np
from useq import MDASequence

from .experiment import capture_targets, position_id


def preserve_references(old_sequence, new_sequence, records, source_positions=None):
    """Carry references to unchanged FOVs, including repeated batch positions.

    Shifted, cell-centered FOVs need a new reference; pixel coordinates from
    their parent image must not silently be used in a different FOV.
    """
    old_positions = list(old_sequence.stage_positions)
    result = []
    for record in records:
        if record.get("role") != "autofocus":
            continue
        if record.get("fixed"):
            continue  # Grid anchors belong to that grid, not a different draft.
        mapped = deepcopy(record)
        mapped.update(position_ids=[], points={}, anchors={})
        for p, destination_position in enumerate(new_sequence.stage_positions):
            position = source_positions[p] if source_positions is not None else destination_position
            matches = [i for i, old in enumerate(old_positions)
                       if (old.x, old.y, old.z) == (position.x, position.y, position.z)
                       and position_id(old) == position_id(position)]
            if not matches:
                matches = [i for i, old in enumerate(old_positions)
                           if (old.x, old.y, old.z) == (position.x, position.y, position.z)
                           and (old.name or "").split(" [id:")[0] == (position.name or "").split(" [id:")[0]]
            if len(matches) != 1:
                continue
            identity = position_id(old_positions[matches[0]])
            if identity not in record["position_ids"]:
                continue
            source_p = str(record["position_ids"].index(identity))
            destination = str(len(mapped["position_ids"]))
            mapped["position_ids"].append(position_id(destination_position))
            for key in ("points", "anchors"):
                mapped[key][destination] = deepcopy(record.get(key, record["points"]).get(source_p, []))
        if mapped["position_ids"]:
            result.append(mapped)
    return result


def _require_idle(owner):
    if owner._unified_mda is None:
        raise ValueError("Connect the microscope first")
    if owner._acquisition_controller.busy or owner.core.mda.is_running():
        raise ValueError("Stop the acquisition before editing autofocus references")


# Position sub-sequence metadata survives native table edits and position-file saves.
REFERENCE_KEY = "autofocus_reference"


def reference(position):
    value = (position.sequence.metadata if position.sequence else {}).get(REFERENCE_KEY)
    if value and value["stage"] == [position.x, position.y, position.z]:
        return value
    return None


def with_reference(position, value):
    sub = position.sequence or MDASequence()
    metadata = dict(sub.metadata)
    metadata.pop(REFERENCE_KEY, None)
    if value is not None:
        metadata[REFERENCE_KEY] = deepcopy(value)
    return position.replace(sequence=sub.replace(metadata=metadata))


def attach_references(sequence, records):
    positions = list(sequence.stage_positions)
    for i, pos in enumerate(positions):
        value = reference(pos)
        for record in records:
            if record.get("role") != "autofocus" or record.get("fixed"):
                continue
            ids = record["position_ids"]
            if position_id(pos) in ids:
                points = record.get("anchors", record["points"]).get(str(ids.index(position_id(pos))), [])
                if value and "stage_xy_um" in value and points == [value["point"]]:
                    continue  # Legacy point records cannot replace an absolute stage reference.
                value = ({"point": points[0], "image_shape": record["image_shape"],
                          "stage": [pos.x, pos.y, pos.z]} if points else None)
        positions[i] = with_reference(pos, value)
    return sequence.replace(stage_positions=positions)


def reference_records(sequence):
    record = dict(id="position-autofocus", name="autofocus references", role="autofocus",
                  image_shape=[1, 1], position_ids=[], points={}, anchors={})
    for i, pos in enumerate(sequence.stage_positions):
        ref = reference(pos)
        if ref:
            record["image_shape"] = ref["image_shape"]
        points = [ref["point"]] if ref else []
        record["position_ids"].append(position_id(pos))
        record["points"][str(i)] = points
        record["anchors"][str(i)] = points
    return [record] if any(reference(p) for p in sequence.stage_positions) else []


def source_stage_xy(source, pixel_yx):
    """Map a survey pixel to absolute stage XY, without a shifted-FOV approximation."""
    model = source["pixel_to_stage"]
    x, y = (np.asarray(pixel_yx) - np.asarray(source["image_shape_yx"]) / 2)[::-1]
    terms = [x**j * y**(n-j) for n in range(int(model["degree"])+1) for j in range(n+1)]
    result = np.asarray(source["stage_xyz_um"][:2]) - np.asarray(terms) @ np.asarray(model["coefficients"])
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("Invalid source-image pixel-to-stage calibration")
    return result.tolist()


def match_positions(sequence, image_sequence):
    """Return image indices, rejecting stale or ambiguous FOV associations."""
    result = []
    for pos in sequence.stage_positions:
        matches = [i for i, old in enumerate(image_sequence.stage_positions)
                   if position_id(pos) is not None and position_id(pos) == position_id(old)
                   and (pos.x, pos.y, pos.z) == (old.x, old.y, old.z)]
        if len(matches) != 1:
            raise ValueError("This acquisition does not match the current FOVs. Acquire configured imaging again.")
        result.append(matches[0])
    return result


def camera_sequence(sequence):
    channels = [c for c in sequence.channels if c.config != "RM"]
    if not channels:
        raise ValueError("Configure at least one camera channel before acquiring reference images")
    if any(p.sequence and p.sequence.model_dump(exclude={"uid", "metadata", "autofocus_plan"},
                                               exclude_defaults=True) for p in sequence.stage_positions):
        raise ValueError("Expand position sub-sequences before selecting autofocus references")
    from .experiment import execution_positions
    return execution_positions(sequence.replace(channels=channels, autofocus_plan=None, metadata={}))


def choose_references(owner, *, image_layer=None, position_index=0, raman=False):
    """Choose an image source, then edit references in a dedicated napari viewer."""
    _require_idle(owner)
    from qtpy.QtWidgets import QInputDialog
    panel = owner._unified_mda
    sequence = panel.value()
    if not raman and ("raman_grid" in sequence.metadata or sequence.grid_plan):
        raise ValueError("Grid autofocus uses its existing grid anchor; select explicit FOVs for custom references")
    if not sequence.stage_positions and not (raman and (sequence.grid_plan or "raman_grid" in sequence.metadata)):
        raise ValueError("Add imaging positions before selecting points")
    existing = getattr(owner, "_raman_editor" if raman else "_autofocus_editor", None)
    if existing is not None:
        existing.viewer.window._qt_window.raise_()
        if not raman:
            existing.viewer.dims.set_current_step(1, existing.indices[position_index])
        return
    if image_layer is None:
        layers = [layer for layer in reversed(list(owner.viewer.layers))
                  if layer.metadata.get("lazy_mda_visualization")
                  and layer.metadata.get("acquisition_complete", False)]
        options = ["Acquire configured imaging"] + [f"Use existing acquisition: {layer.name}" for layer in layers]
        choice, ok = QInputDialog.getItem(panel, "Raman image source" if raman else "Autofocus image source", "Image source", options, 1 if layers else 0, False)
        if not ok:
            return
        if choice != options[0]:
            image_layer = layers[options.index(choice) - 1]
        else:
            preview = camera_sequence(sequence)
            previous_engine = owner.core.mda.engine
            presets = {c.group or "Channel": owner.core.getCurrentConfig(c.group or "Channel") for c in preview.channels}
            auto_shutter, shutter_open = owner.core.getAutoShutter(), owner.core.getShutterOpen()
            try:
                owner._prepare_for_selection()
                owner.core.register_mda_engine(owner.default_engine)
                from qtpy.QtCore import QEventLoop, QTimer, Qt
                from qtpy.QtWidgets import QProgressDialog
                progress = QProgressDialog("Acquiring reference images…", "Cancel", 0, 0, panel)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.canceled.connect(owner.core.mda.cancel)
                try:
                    thread = owner.core.run_mda(preview)
                    loop = QEventLoop()
                    timer = QTimer(progress)
                    timer.timeout.connect(lambda: loop.quit() if not thread.is_alive() else None)
                    timer.start(50)
                    progress.show()
                    loop.exec()
                    timer.stop()
                    if progress.wasCanceled():
                        raise ValueError("Reference imaging canceled; no references were changed")
                finally:
                    # QProgressDialog.close() emits canceled even on success.
                    # Disconnect before cleanup so it cannot cancel the next MDA.
                    progress.canceled.disconnect(owner.core.mda.cancel)
                    progress.close()
            finally:
                owner.core.register_mda_engine(previous_engine)
                for group, preset in presets.items():
                    if preset:
                        owner.core.setConfig(group, preset)
                        owner.core.waitForConfig(group, preset)
                owner.core.setAutoShutter(auto_shutter)
                owner.core.setShutterOpen(shutter_open)
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            image_layer = next((layer for layer in reversed(list(owner.viewer.layers))
                                if layer.metadata.get("useq_sequence") is preview), None)
            if image_layer is None or not image_layer.metadata.get("acquisition_complete"):
                raise ValueError("Reference imaging did not complete; no references were changed")
    if raman:
        from .raman_selection import RamanEditor
        owner._raman_editor = RamanEditor(owner, sequence, image_layer)
        return
    match_positions(sequence, image_layer.metadata["useq_sequence"])
    owner._autofocus_editor = ReferenceEditor(owner, sequence, image_layer, position_index)


class ReferenceEditor:
    def __init__(self, owner, sequence, image_layer, position_index=0):
        import napari
        from qtpy.QtCore import QObject, QEvent
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
        self.owner, self.sequence = owner, sequence
        self.indices = match_positions(sequence, image_layer.metadata["useq_sequence"])
        self.image_sequence = image_layer.metadata["useq_sequence"]
        self.viewer = napari.Viewer(title="Set autofocus references")
        self.viewer.add_image(image_layer.data, name=image_layer.name, rgb=image_layer.rgb,
                              scale=image_layer.scale, contrast_limits=image_layer.contrast_limits,
                              metadata=dict(image_layer.metadata))
        self.viewer.dims.axis_labels = ("t", "p", "g", "c", "z", "y", "x")
        self.shape = tuple(image_layer.data.shape[-3:-1] if image_layer.rgb else image_layer.data.shape[-2:])
        # Engine sources normalize (y, x) by (width, height), including legacy records.
        self.source_shape = self.shape[::-1]
        data = []
        for i, pos in enumerate(sequence.stage_positions):
            ref = reference(pos)
            if ref:
                y, x = np.asarray(ref["point"]) * ref["image_shape"]
                if ref.get("stage_xy_um") is not None:
                    continue  # A survey reference cannot be drawn as a local image pixel.
                data.append([0, self.indices[i], 0, 0, 0, y, x])
        self.points = self.viewer.add_points(np.asarray(data).reshape((-1, 7)), ndim=7,
                                             name="Autofocus references", face_color="springgreen",
                                             size=12, scale=image_layer.scale,
                                             out_of_slice_display=False)
        self.points.mode = "add"
        self._updating = False
        self.points.events.data.connect(self.changed)
        self.viewer.dims.events.current_step.connect(self.slice_changed)
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.addWidget(QLabel("One reference per FOV. No reference means autofocus is skipped.\n"
                               "Add a point to replace that FOV's reference; select and delete to remove it.\n"
                               "Inherited stage references are retained unless replaced by a point on these images."))
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        apply = QPushButton("Apply")
        cancel = QPushButton("Cancel")
        layout.addWidget(apply)
        layout.addWidget(cancel)
        apply.clicked.connect(self.apply)
        cancel.clicked.connect(self.viewer.close)
        self.viewer.window.add_dock_widget(controls, area="right", name="Autofocus")
        editor = self
        class CloseFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Close:
                    editor.owner._autofocus_editor = None
                return False
        self.close_filter = CloseFilter(self.viewer.window._qt_window)
        self.viewer.window._qt_window.installEventFilter(self.close_filter)
        self.viewer.dims.set_current_step(1, self.indices[position_index])
        self.changed()

    def slice_changed(self, *_):
        # References belong to FOVs, and remain visible across time/channel/Z.
        data = self.points.data.copy()
        if len(data):
            for axis in (0, 2, 3, 4):
                data[:, axis] = self.viewer.dims.current_step[axis]
            self.points.data = data

    def changed(self, *_):
        if self._updating:
            return
        # New points replace old points in the same FOV, regardless of channel/time/Z.
        rows = {}
        for row in self.points.data:
            rows[int(round(row[1]))] = row
        data = np.asarray(list(rows.values())).reshape((-1, 7))
        if len(data) != len(self.points.data):
            self._updating = True
            self.points.data = data
            self._updating = False
        count = sum(index in rows or bool((reference(pos) or {}).get("stage_xy_um"))
                    for index, pos in zip(self.indices, self.sequence.stage_positions))
        self.summary.setText(f"Autofocus: {count} set · {len(self.indices) - count} skipped")

    def apply(self):
        try:
            _require_idle(self.owner)
            current = self.owner._unified_mda.value()
            if {(position_id(p), p.x, p.y, p.z) for p in current.stage_positions} != {
                    (position_id(p), p.x, p.y, p.z) for p in self.sequence.stage_positions}:
                raise ValueError("Positions changed while selecting. Close and reopen the selector.")
            rows = {int(round(row[1])): row[-2:] for row in self.points.data}
            positions = []
            for pos in current.stage_positions:
                index = match_positions(MDASequence(stage_positions=[pos]), self.image_sequence)[0]
                point = rows.get(index)
                if point is not None and (not np.isfinite(point).all() or np.any(point < 0) or np.any(point >= self.shape)):
                    raise ValueError("Place references inside the image")
                value = None if point is None else dict(point=(point / self.source_shape).tolist(),
                    image_shape=list(self.source_shape), stage=[pos.x, pos.y, pos.z])
                if point is None and (reference(pos) or {}).get("stage_xy_um") is not None:
                    value = reference(pos)
                positions.append(with_reference(pos, value))
            self.owner._unified_mda.setValue(current.replace(stage_positions=positions))
            self.owner._unified_mda.update_summary()
            self.owner.status.setText("Status: autofocus references applied")
            self.viewer.close()
        except ValueError as error:
            self.error.setText(str(error))


def apply_references(owner):
    """Compatibility adapter for prepared point sources."""
    _require_idle(owner)
    sequence, sources = owner._autofocus_reference_selection
    current = owner._unified_mda.value()
    match_positions(current, sequence)
    records = capture_targets(sequence, sources)
    if any(len(points) > 1 for record in records for points in record["anchors"].values()):
        raise ValueError("Keep one autofocus reference per FOV")
    owner._unified_mda.setValue(attach_references(current, records))
    owner._unified_mda.update_summary()


def reference_action(owner, action):
    try:
        action(owner)
    except Exception as error:
        owner.status.setText(f"Status: autofocus references — {error}")
