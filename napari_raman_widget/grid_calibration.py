"""Use the objective's measured pixel-to-stage scale in the native MDA grid."""
from contextlib import suppress
from types import MethodType

import numpy as np
from qtpy.QtWidgets import QCheckBox, QLabel


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
