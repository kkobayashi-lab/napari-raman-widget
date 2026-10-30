"""Point sampling using the native MDA geometry and traversal controls."""
import math

from qtpy.QtWidgets import QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QTableWidgetItem, QWidget
from useq import MDASequence

from .experiment import SnapshotSource
from raman_mda_engine.grid_plan import GRID_KEY


def point_region(kind, geometry, spacing, origin, relative_to, order):
    """Translate native geometry into compact point-center bounds."""
    dx, dy = spacing
    if not all(math.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("Spacing must be finite and positive")
    if kind == "number":
        nx, ny = geometry["columns"], geometry["rows"]
        width, height = (nx - 1) * dx, (ny - 1) * dy
    elif kind == "area":
        width, height = geometry["width"], geometry["height"]
        nx, ny = math.ceil(width / dx) + 1, math.ceil(height / dy) + 1
    else:
        left, right = sorted((geometry["left"], geometry["right"]))
        bottom, top = sorted((geometry["bottom"], geometry["top"]))
        nx, ny = math.ceil((right - left) / dx) + 1, math.ceil((top - bottom) / dy) + 1
        return dict(x=[left, right, nx], y=[top, bottom, ny], order=order, z=None, tilt=None)
    left, top = origin
    if relative_to == "center":
        left -= width / 2
        top += height / 2
    return dict(x=[left, left + width, nx], y=[top, top - height, ny], order=order, z=None, tilt=None)


class RamanGridEditor:
    def __init__(self, modes):
        self.modes = modes
        self.owner = owner = modes.owner
        self.native = modes.panel.grid_plan
        self.spacing_panel = QWidget()
        form = QFormLayout(self.spacing_panel)
        self.x_spacing = QDoubleSpinBox()
        self.y_spacing = QDoubleSpinBox()
        for label, control in (("X spacing", self.x_spacing), ("Y spacing", self.y_spacing)):
            control.setRange(0.01, 1_000_000)
            control.setDecimals(3)
            control.setValue(50)
            control.setSuffix(" µm")
            form.addRow(label, control)
            control.valueChanged.connect(self.preview)
        modes.imaging.layout().insertWidget(3, self.spacing_panel)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        modes.raman_layout.addWidget(self.summary)
        self.shift = QDoubleSpinBox()
        self.shift.setRange(-10000, 10000)
        self.shift.setDecimals(1)
        self.shift.setValue(1003)
        self.shift.setSuffix(" cm⁻¹")
        self.shift.setToolTip("Use the nearest calibrated spectral bin; average repeated spectra at each grid point.")
        display_form = QFormLayout()
        display_form.addRow("Raman shift", self.shift)
        modes.raman_layout.addLayout(display_form)
        self.shift.valueChanged.connect(lambda *_: modes.panel.valueChanged.emit())
        if hasattr(owner, "_raman_grid_panel"):
            focus = QFormLayout()
            focus.addRow("Grid Z (µm)", owner.grid_z_input)
            owner.grid_z_input.setPlaceholderText("Use acquisition-start Z")
            owner.grid_z_input.setToolTip("Blank leaves Z unspecified. Tilt correction overrides the constant Z.")
            focus.addRow(owner.grid_capture_z_btn)
            modes.raman_layout.addLayout(focus)
            modes.raman_layout.addWidget(owner.grid_tilt_check)
            modes.raman_layout.addWidget(owner._grid_tilt_help)
            degree_form = QFormLayout()
            degree_form.addRow(owner._grid_tilt_degree_label, owner.grid_tilt_degree_input)
            modes.raman_layout.addLayout(degree_form)
            modes.raman_layout.addWidget(owner.grid_tilt_table)
            tilt_buttons = QHBoxLayout()
            for control in (owner.grid_tilt_capture_btn, owner.grid_tilt_add_btn,
                            owner.grid_tilt_remove_btn, owner.grid_tilt_clear_btn):
                tilt_buttons.addWidget(control)
            modes.raman_layout.addLayout(tilt_buttons)
            modes.raman_layout.addWidget(owner.grid_tilt_fit_label)
            owner._toggle_grid_tilt_fields(owner.grid_tilt_check.isChecked())
            modes.raman_layout.addWidget(QLabel("Raman grids to acquire with Run MDA:"))
            modes.raman_layout.addWidget(owner.grid_region_list)
            modes.raman_layout.addWidget(owner.grid_region_summary)
            buttons = QHBoxLayout()
            for button in (owner.run_grid_sel_btn, owner.grid_remove_btn, owner.grid_clear_btn):
                buttons.addWidget(button)
            modes.raman_layout.addLayout(buttons)
            owner.run_grid_sel_btn.clicked.disconnect(owner.run_grid_selection)
            owner.run_grid_sel_btn.clicked.connect(self.add)
        self.native.valueChanged.connect(self.preview)
        modes.repeats.valueChanged.connect(self.preview)
        self.preview()

    def definition(self, origin=(0, 0)):
        geometry = self.native._stack.currentWidget().value()
        return point_region(self.native.mode().value, geometry,
                            (self.x_spacing.value(), self.y_spacing.value()), origin,
                            self.native.relative_to.currentEnum().value,
                            self.native.order.currentEnum().value)

    def settings(self):
        settings = dict(plan=self.native.value().model_dump(mode="json"),
                        spacing=[self.x_spacing.value(), self.y_spacing.value()],
                        shift_cm=self.shift.value())
        owner = self.owner
        if hasattr(owner, "grid_tilt_table"):
            table = owner.grid_tilt_table
            settings["focus"] = dict(z=owner.grid_z_input.text(),
                tilt=owner.grid_tilt_check.isChecked(), degree=owner.grid_tilt_degree_input.value(),
                references=[[table.item(row, column).text() if table.item(row, column) else ""
                             for column in range(3)] for row in range(table.rowCount())])
        return settings

    def restore(self, settings):
        self.shift.setValue((settings or {}).get("shift_cm", 1003))
        if not settings:
            return
        self.native.setValue(MDASequence(grid_plan=settings["plan"]).grid_plan)
        self.x_spacing.setValue(settings["spacing"][0])
        self.y_spacing.setValue(settings["spacing"][1])
        owner = self.owner
        if "focus" in settings and hasattr(owner, "grid_tilt_table"):
            focus = settings["focus"]
            owner.grid_z_input.setText(focus["z"])
            owner.grid_tilt_degree_input.setValue(focus["degree"])
            table = owner.grid_tilt_table
            table.setRowCount(len(focus["references"]))
            for row, values in enumerate(focus["references"]):
                for column, value in enumerate(values):
                    table.setItem(row, column, QTableWidgetItem(value))
            owner.grid_tilt_check.setChecked(focus["tilt"])

    def preview(self, *_args):
        try:
            region = self.definition()
            nx, ny = region["x"][2], region["y"][2]
            self.summary.setText(f"{ny:,} rows × {nx:,} columns = {nx * ny:,} points; "
                                 f"{nx * ny * self.modes.repeats.value():,} spectra per plane/time point.\n"
                                 "Area and boundary modes include both endpoints, reducing spacing if needed.")
        except (ValueError, OverflowError) as error:
            self.summary.setText(str(error))

    def set_visible(self, raman):
        self.spacing_panel.setVisible(raman)
        self.native._mode_number_radio.setText("Rows / Columns" if raman else "Fields of View")
        for control in (self.native.row_col_wdg.rows, self.native.row_col_wdg.columns):
            control.setSuffix(" points" if raman else " fields")
        self.native.overlap.setVisible(not raman)
        self.native._bottom_stuff._form_layout.labelForField(self.native.overlap).setVisible(not raman)
        self.modes.panel.grid_use_vandermonde.setVisible(not raman)
        self.modes.panel.grid_calibration_status.setVisible(not raman)

    def add(self):
        owner = self.owner
        controller = getattr(owner, "_acquisition_controller", None)
        if self.modes.mode != "raman" or (controller and controller.busy):
            return
        try:
            origin = owner.core.getXYPosition() if self.native.mode().value != "bounds" else (0, 0)
            region = self.definition(origin)
            if owner.grid_tilt_check.isChecked():
                from cns_control.utils import _fit_tilt_surface
                degree = owner.grid_tilt_degree_input.value()
                center, scale, coefficients, _error = _fit_tilt_surface(owner._grid_tilt_reference_points(), degree)
                region["tilt"] = dict(degree=degree, origin=center.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist())
            elif owner.grid_z_input.text().strip():
                region["z"] = float(owner.grid_z_input.text())
                if not math.isfinite(region["z"]):
                    raise ValueError("Grid Z must be finite")
            elif QMessageBox.warning(self.modes.panel, "Grid Z not specified",
                    "No Z position is specified and tilt correction is off.\n\n"
                    "Enter or capture a focused Z, or enable tilt correction.\n\n"
                    "Add this grid without an explicit Z position?",
                    QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel) != QMessageBox.Ok:
                return
            width, height = owner.core.getImageWidth(), owner.core.getImageHeight()
            if width <= 0 or height <= 0:
                raise ValueError("A valid camera field is required for centered Raman aiming")
            center = [height / 2 / width, width / 2 / height]
            sources = [SnapshotSource(dict(id=f"grid-{role}", name=f"Grid {role}", role=role,
                image_shape=[width, height], fixed=True, position_ids=[],
                points={"0": [center]}, anchors={"0": [center]})) for role in ("cell", "autofocus")]
            nx, ny = region["x"][2], region["y"][2]
            owner._grid_regions.append(dict(
                sequence=MDASequence(metadata={GRID_KEY: {"version": 1, "regions": [region]}}),
                sources=sources, source_settings=dict(fov_x=width / 2, fov_y=height / 2, dense=True),
                summary=f"{ny} × {nx} points; X {region['x'][0]:g}..{region['x'][1]:g}; "
                        f"Y {region['y'][0]:g}..{region['y'][1]:g}; {region['order']}"))
            owner._refresh_grid_region_list()
            owner.grid_region_list.setCurrentRow(len(owner._grid_regions) - 1)
            owner.status.setText("Status: Raman grid added — ready for Run MDA")
        except (ValueError, RuntimeError, OverflowError) as error:
            QMessageBox.warning(self.modes.panel, "Cannot add Raman grid", str(error))
