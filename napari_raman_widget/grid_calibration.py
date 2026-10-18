"""Use the objective's measured pixel-to-stage scale in the native MDA grid."""
from contextlib import suppress
from types import MethodType

import numpy as np
from qtpy.QtWidgets import QCheckBox, QLabel


CELLVIS_96_PLATE = "Cellvis 96-well (6.0 mm)"


def vandermonde_fov(model, width, height):
    """Return field dimensions from the polynomial's Jacobian at zero offset.

    Calibration terms are 1, y, x, y², xy, x², ... . Vector lengths retain
    scale when the camera is rotated relative to the stage. A native grid
    represents a rectangular field, so nonlinear distortion is approximated
    by the scale at the calibrated center.
    """
    coefficients, degree = model
    coefficients = np.asarray(coefficients, dtype=float)
    if (not isinstance(degree, (int, np.integer)) or degree < 1
            or coefficients.shape != ((degree + 1) * (degree + 2) // 2, 2)
            or not np.isfinite(coefficients).all()):
        raise ValueError("Invalid pixel-to-stage Vandermonde calibration")
    scales = np.linalg.norm(coefficients[[2, 1]], axis=1)
    fov = scales * np.asarray([width, height], dtype=float)
    if not np.isfinite(fov).all() or np.any(fov <= 0):
        raise ValueError("Calibration must have positive finite X and Y scales")
    if abs(np.linalg.det(coefficients[[2, 1]])) <= np.prod(scales) * 1e-6:
        raise ValueError("Calibration's X and Y directions are degenerate")
    return tuple(float(value) for value in fov)


def extend_wellplate_calibration(table, owner):
    """Use the current objective's model for HCS previews and generated positions."""
    from superqt.utils import signals_blocked
    from useq import register_well_plates
    from .wellplate_calibration import CalibratedWellPlatePlan, extend_plate_mapping

    # Register before constructing the wizard, which populates its plate selector
    # from useq's registry. Keep the generic 6.4 mm preset available for other plates.
    # Specs: https://www.cellvis.com/_96-well-glass-bottom-plates_/products_by_category.php?cat_id=11
    register_well_plates({CELLVIS_96_PLATE: {
        "rows": 8, "columns": 12, "well_spacing": 9.0,
        "well_size": 6.0, "circular_wells": True,
    }})
    new_wizard = table._hcs_wizard is None
    wizard = table._hcs
    extend_plate_mapping(wizard)
    from .well_boundary_widget import install_boundary_calibration
    install_boundary_calibration(wizard, owner)
    from .plate_focus_widget import install_focus_calibration
    install_focus_calibration(wizard, owner)
    if new_wizard and table._plate_plan is None:
        wizard.plate_page.widget.plate_name.setCurrentText(CELLVIS_96_PLATE)
    page = wizard.points_plan_page
    status = QLabel(page)
    status.setWordWrap(True)
    page.layout().addWidget(status)
    native_value = table.value
    native_set_value = table.setValue
    native_editable = table._set_position_table_editable
    refreshing = False

    def editable(_self, state):
        if state:
            # Upstream disconnects this slot unconditionally when unlocking.
            from qtpy.QtCore import Qt
            with suppress(TypeError, RuntimeError):
                table.table().cellDoubleClicked.connect(
                    table._show_pos_editing_dialog, Qt.ConnectionType.UniqueConnection)
        native_editable(state)
        if not state:
            table.include_z.show()
            table._on_include_z_toggled(table.include_z.isChecked())
            # Z can be edited without converting the calibrated XY positions.
            with suppress(TypeError, RuntimeError):
                table.table().cellDoubleClicked.disconnect(table._show_pos_editing_dialog)

    def set_value(_self, positions):
        with signals_blocked(table):
            native_set_value(positions)
            if table._plate_plan is not None:
                editable(table, False)
        table.valueChanged.emit()

    table._set_position_table_editable = MethodType(editable, table)
    table.setValue = MethodType(set_value, table)

    def field_size(_self=None):
        ensure = getattr(owner, "_ensure_current_vandermonde", None)
        objective = ensure() if ensure else None
        model = getattr(owner, "vandermonde", None)
        try:
            if objective is None or model is None:
                raise ValueError("Well plate requires a current-objective Vandermonde calibration")
            fov = vandermonde_fov(model, owner.core.getImageWidth(), owner.core.getImageHeight())
        except ValueError as error:
            status.setText(str(error))
            return None, None
        status.setText(
            f"Well plate FOV: {fov[0]:g} × {fov[1]:g} µm — "
            f"Vandermonde, objective {objective} (center scale). "
            "Camera binning must match the calibration."
        )
        return fov

    # Existing pixel-size/configuration callbacks now call our field-size provider.
    page._get_fov_size = MethodType(field_size, page)

    def refresh(*_):
        nonlocal refreshing
        if refreshing:
            return
        refreshing = True
        try:
            page._on_px_size_changed()
            plan = table._plate_plan
            fov = field_size()
            if plan is not None and None not in fov:
                points = plan.well_points_plan
                if (points.fov_width, points.fov_height) != fov:
                    from pymmcore_widgets.useq_widgets import PositionTable
                    old_z = {p.name: row.z for p, row in
                             zip(plan, PositionTable.value(table, False, False))}
                    include_z = table.include_z.isChecked()
                    # Reconstruct the plan: useq caches its generated image positions.
                    points = points.model_copy(update=dict(zip(("fov_width", "fov_height"), fov)))
                    table._plate_plan = plan.replace(well_points_plan=points)
                    with signals_blocked(table):
                        table._update_table_positions(table._plate_plan)
                        for row, position in enumerate(table._plate_plan):
                            if position.name in old_z:
                                table.table().setRowData(row, {"z": old_z[position.name]})
                        table.include_z.setChecked(include_z)
        finally:
            refreshing = False

    def value(_self, *args, **kwargs):
        refresh()
        if table._plate_plan is not None and None in field_size():
            raise ValueError(status.text())
        result = native_value(*args, **kwargs)
        if table._plate_plan is not None:
            focus = getattr(table._plate_plan, "focus_height", None)
            if focus is not None and focus.enabled:
                positions = tuple(result)
                heights = focus.heights([(p.x, p.y) for p in positions])
                return tuple(p.replace(z=float(z)) for p, z in zip(positions, heights))
            # Keep the plan's calibrated XY, but read focus heights from the UI.
            from pymmcore_widgets.useq_widgets import PositionTable
            rows = PositionTable.value(table, False, False)
            include_z = table.include_z.isChecked() or kwargs.get("exclude_hidden_cols") is False
            if len(args) > 1 and args[1] is False:
                include_z = True
            return tuple(p.replace(z=row.z if include_z else None)
                         for p, row in zip(result, rows))
        # Native MDASequence serializes WellPlatePlan as its base type. Emit real
        # positions; the application saves the full HCS plan in its metadata.
        return tuple(result) if isinstance(result, CalibratedWellPlatePlan) else result

    def wizard_value(_self):
        refresh()
        if None in field_size():
            return None
        calibrated = wizard.calibration_page.widget.value()
        if not wizard._calibrated or calibrated is None:
            return None
        selected = wizard.plate_page.widget.value()
        if selected.plate != calibrated.plate:
            return None
        return calibrated.replace(selected_wells=selected.selected_wells,
                                  well_points_plan=page.widget.value())

    table.value = MethodType(value, table)
    wizard.value = MethodType(wizard_value, wizard)
    connections = []

    def connect(signal, callback):
        signal.connect(callback)
        connections.append((signal, callback))

    path_control = getattr(owner, "sel_vdm_path", None)
    if path_control is not None:
        connect(path_control.textChanged, refresh)

    def property_changed(device, prop, *_):
        if device == "Objective" or (device == owner.core.getCameraDevice() and prop == "Binning"):
            refresh()

    connect(owner.core.events.propertyChanged, property_changed)

    def cleanup(*_):
        for signal, callback in connections:
            with suppress(TypeError, RuntimeError):
                signal.disconnect(callback)
        connections.clear()

    table.destroyed.connect(cleanup)
    refresh()
    return status


def extend_grid_calibration(widget, owner):
    """Override grid FOV on every read, including preview, save, and Run."""
    native_value = widget.value
    use_model = QCheckBox("Use Vandermonde calibration for grid", widget)
    use_model.setChecked(True)
    use_model.setToolTip(
        "Uses the current objective's pixel-to-stage model at the calibrated center. "
        "Camera binning must match the binning used for that calibration."
    )
    status = QLabel(widget)
    status.setWordWrap(True)
    widget.widget().layout().addWidget(use_model)
    widget.widget().layout().addWidget(status)
    refreshing = False

    def value(_self):
        nonlocal refreshing
        if refreshing:
            return native_value()
        refreshing = True
        try:
            width, height = owner.core.getImageWidth(), owner.core.getImageHeight()
            pixel_size = owner.core.getPixelSizeUm()
            fov = (width * pixel_size or None, height * pixel_size or None)
            message = "Grid FOV uses Micro-Manager pixel size."
            if use_model.isChecked():
                ensure = getattr(owner, "_ensure_current_vandermonde", None)
                objective = ensure() if ensure else None
                model = getattr(owner, "vandermonde", None)
                if objective is not None and model is not None:
                    try:
                        fov = vandermonde_fov(model, width, height)
                        message = (f"Grid FOV: {fov[0]:g} × {fov[1]:g} µm — "
                                   f"Vandermonde, objective {objective} (center scale).")
                    except ValueError as error:
                        message = f"{error}. Grid FOV uses Micro-Manager pixel size."
                else:
                    message = "No current-objective Vandermonde calibration; grid uses Micro-Manager pixel size."
            status.setText(message)
            # Avoid emitting recursive valueChanged signals while reading a plan.
            widget._fov_width, widget._fov_height = fov
            return native_value()
        finally:
            refreshing = False

    widget.value = MethodType(value, widget)
    use_model.toggled.connect(lambda *_: widget._on_change())
    connections = []

    def connect(signal, callback):
        signal.connect(callback)
        connections.append((signal, callback))

    path_control = getattr(owner, "sel_vdm_path", None)
    if path_control is not None:
        connect(path_control.textChanged, lambda *_: widget._on_change())

    def property_changed(device, prop, *_):
        if device == "Objective" or (device == owner.core.getCameraDevice() and prop == "Binning"):
            widget._on_change()

    connect(owner.core.events.propertyChanged, property_changed)

    def cleanup(*_):
        for signal, callback in connections:
            with suppress(TypeError, RuntimeError):
                signal.disconnect(callback)
        connections.clear()

    widget.destroyed.connect(cleanup)
    return use_model, status
