"""Application-owned MDA dock and its shared acquisition controller."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType

from qtpy.QtCore import QObject, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QLabel, QLayout, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget, QHBoxLayout, QTabWidget,
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QTabBar,
)

from .experiment import (
    AutofocusSettings, ExperimentExtensions, METADATA_KEY, RamanSettings,
    TrackingSettings, capture_targets, identify_positions, prepare_acquisition,
)
from .mda_timing import estimate_mda_time, format_duration
from .ui_helpers import make_collapsible
from .grid_calibration import extend_grid_calibration, extend_wellplate_calibration
from raman_mda_engine.grid_plan import GRID_KEY, grid_count


def build_extension_controls(owner):
    """Create controls before microscope connection; attach them to MDA later."""
    panel = QWidget(owner)
    panel.hide()
    layout = QVBoxLayout(panel)
    owner._mda_extension_panel = panel
    owner._unified_mda = None
    owner._acquisition_controller = None

    def section(title):
        group = make_collapsible(title, expanded=title == "Run status")
        form = QFormLayout(group)
        # Qt5's form layout retains row spacing when children are hidden.
        form.setSizeConstraint(QLayout.SetNoConstraint)
        def resize_section(expanded):
            group.setMaximumHeight(16_777_215 if expanded else group.fontMetrics().height() + 14)
            group.updateGeometry()
        group.toggled.connect(resize_section)
        resize_section(group.isChecked())
        layout.addWidget(group)
        group.toggled.connect(lambda *_: owner._reapply_toggles() if hasattr(owner, "_reapply_toggles") else None)
        return form

    def field(form, name, label, value, minimum=0, maximum=1_000_000):
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(minimum, maximum)
            widget.setValue(value)
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setDecimals(2)
            widget.setValue(value)
        else:
            widget = QLineEdit(str(value))
        label_widget = QLabel(label)
        form.addRow(label_widget, widget)
        setattr(owner, name, widget)
        return label_widget

    owner._raman_settings_panel = QWidget(owner)
    owner._raman_settings_panel.hide()
    raman = QFormLayout(owner._raman_settings_panel)
    owner._raman_settings_form = raman
    raman.addRow(QLabel("RM channel exposure is milliseconds per collection.\n"
                         "Individual: per aiming point. Batch: per integrated spectrum."))
    owner.mda_batch_check = QCheckBox("Integrated batch collection")
    # Retain the legacy value for imported experiments; new selections own mode.
    owner.mda_batch_check.setParent(owner._raman_settings_panel)
    owner.mda_batch_check.hide()
    owner.mda_pre_acq_check = QCheckBox("Preposition galvo before collection")
    owner.mda_pre_acq_check.setChecked(True)
    raman.addRow(owner.mda_pre_acq_check)
    field(raman, "mda_repeats_input", "Frames per Raman point / batch", 1, minimum=1)
    owner.mda_repeats_input.setToolTip("Consecutive spectra at each point before moving; independent of MDA time points.")
    if hasattr(owner, "_update_grid_size_preview"):
        owner.mda_repeats_input.valueChanged.connect(owner._update_grid_size_preview)
    field(raman, "mda_rz_input", "Raman planes (all / middle / indices)", "all")
    field(raman, "mda_imgp_input", "Camera positions (all / none / indices)", "all")
    owner.mda_collection_summary = QLabel("Add RM to the MDA channel table to collect spectra.")
    owner.mda_collection_summary.setWordWrap(True)
    raman.addRow(owner.mda_collection_summary)

    owner._mda_af_panel = QWidget(owner)
    owner._mda_af_panel.hide()
    af_layout = QVBoxLayout(owner._mda_af_panel)
    axis_row = QHBoxLayout()
    axis_row.addWidget(QLabel("Use autofocus on axis:"))
    owner.mda_af_axes = {}
    for axis, description in (("p", "stage position"), ("t", "timepoint"), ("g", "grid tile")):
        checkbox = QCheckBox(axis)
        checkbox.setToolTip(f"Run laser autofocus when the {description} changes.")
        owner.mda_af_axes[axis] = checkbox
        axis_row.addWidget(checkbox)
        checkbox.toggled.connect(lambda *_: update_autofocus_controls(owner))
    axis_row.addStretch()
    af_layout.addLayout(axis_row)
    af_layout.addWidget(QLabel("Select an axis to enable autofocus. No axes selected = autofocus off."))
    from .autofocus_references import choose_references, reference_action
    owner.choose_af_references_btn = QPushButton("Set autofocus references…")
    owner.choose_af_references_btn.clicked.connect(lambda: reference_action(owner, choose_references))
    af_layout.addWidget(owner.choose_af_references_btn)
    af_layout.addWidget(QLabel("No reference for a FOV means autofocus is skipped there."))
    owner._mda_af_settings = QWidget()
    af_layout.addWidget(owner._mda_af_settings)
    af = QFormLayout(owner._mda_af_settings)
    af.setContentsMargins(0, 0, 0, 0)
    owner.mda_af_mode_combo = QComboBox()
    owner.mda_af_mode_combo.addItem("laser")
    af.addRow("Custom mode", owner.mda_af_mode_combo)
    field(af, "mda_afp_input", "Positions (all / none / indices)", "all")
    field(af, "mda_refocus_input", "Refocus every N timepoints (0 = initial only)", 0)
    owner._af_range_label = field(af, "mda_af_range_input", "Search range (± µm)", 6.0, minimum=0.1)
    owner._search_pts_label = field(af, "mda_search_pts_input", "Search points", 8, minimum=2, maximum=500)
    owner._fine_range_label = field(af, "mda_fine_range_input", "Laser fine range (± µm)", 1.5, minimum=0.1)
    owner._fine_pts_label = field(af, "mda_fine_pts_input", "Laser fine search points", 8, minimum=2, maximum=500)
    af_layout.addStretch()
    update_autofocus_controls(owner)

    tracking = section("Segmentation and tracking")
    field(tracking, "mda_seg_track_check", "Segment and track", False)
    field(tracking, "mda_segment_every_input", "Update every N timepoints (0 = initial only)", 0)
    owner.mda_seg_ch_combo = QComboBox()
    owner.mda_seg_ch_combo.addItem("BF")
    owner._seg_ch_label = QLabel("Segmentation channel")
    tracking.addRow(owner._seg_ch_label, owner.mda_seg_ch_combo)
    owner._seg_scale_label = field(tracking, "mda_seg_scale_input", "Image rescale factor", 2.0, minimum=0.1, maximum=100)
    owner.mda_seg_model_combo = QComboBox()
    owner.mda_seg_model_combo.addItem("cyto2", "cyto2")
    owner._seg_model_label = QLabel("Cellpose model")
    tracking.addRow(owner._seg_model_label, owner.mda_seg_model_combo)
    owner.mda_seg_model_browse = QPushButton("Choose model weights…")
    owner.mda_seg_model_browse.clicked.connect(lambda: owner.browse_cellpose_model(owner.mda_seg_model_combo))
    tracking.addRow(owner.mda_seg_model_browse)
    owner._seg_diameter_label = field(tracking, "mda_seg_diameter_input", "Cell diameter (px, 0 = model default)", 15.0, maximum=500)
    owner.mda_seg_crop_combo = QComboBox()
    owner.mda_seg_crop_combo.addItems(["True", "False"])
    owner._seg_crop_label = QLabel("Crop around mask")
    tracking.addRow(owner._seg_crop_label, owner.mda_seg_crop_combo)
    owner._seg_track_cfg_label = field(tracking, "mda_track_cfg_input", "Tracking configuration", "particle_config.json")
    owner._seg_track_cfg_browse = QPushButton("Choose tracking configuration…")
    owner._seg_track_cfg_browse.clicked.connect(owner.browse_tracking_cfg)
    tracking.addRow(owner._seg_track_cfg_browse)

    run = section("Run status")
    field(run, "mda_delay_input", "Start delay (s)", 0.0)
    field(run, "mda_overwrite_check", "Overwrite acquisition folder", False)
    owner.mda_live_eta_label = QLabel("Live ETA: --")
    owner.mda_live_eta_label.setWordWrap(True)
    run.addRow(owner.mda_live_eta_label)
    owner.view_acquisition_btn = QPushButton("Open saved acquisition")
    owner.view_acquisition_btn.clicked.connect(owner.open_acquisition_viewer)
    run.addRow(owner.view_acquisition_btn)
    # Path mirror for the existing offline viewer, not an editable output setting.
    owner.mda_dir_input = QLineEdit("data/run", panel)
    owner.mda_dir_input.hide()
    owner.mda_af_mode_combo.currentTextChanged.connect(owner._toggle_autofocus_fields)
    owner.mda_seg_track_check.toggled.connect(owner._toggle_seg_track_fields)
    owner._toggle_autofocus_fields("laser")
    owner._toggle_seg_track_fields(False)
    layout.addStretch()
    return panel


def update_autofocus_controls(owner):
    owner._mda_af_settings.setEnabled(any(box.isChecked() for box in owner.mda_af_axes.values()))


def read_extensions(owner):
    return ExperimentExtensions(
        raman=RamanSettings(owner.mda_rz_input.text(), owner.mda_imgp_input.text(),
                            owner.mda_batch_check.isChecked(), owner.mda_pre_acq_check.isChecked(),
                            owner.mda_repeats_input.value()),
        autofocus=AutofocusSettings(owner.mda_af_mode_combo.currentText(), owner.mda_afp_input.text(),
                                    owner.mda_refocus_input.value(), owner.mda_af_range_input.value(),
                                    owner.mda_search_pts_input.value(), owner.mda_fine_range_input.value(),
                                    owner.mda_fine_pts_input.value(),
                                    tuple(axis for axis, box in owner.mda_af_axes.items() if box.isChecked())),
        tracking=TrackingSettings(owner.mda_seg_track_check.isChecked(), owner.mda_segment_every_input.value(),
                                  owner.mda_seg_ch_combo.currentText(), owner.mda_seg_scale_input.value(),
                                  str(owner.mda_seg_model_combo.currentData() or owner.mda_seg_model_combo.currentText()),
                                  owner.mda_seg_diameter_input.value(), owner.mda_seg_crop_combo.currentText() == "True",
                                  owner.mda_track_cfg_input.text()),
        delay=owner.mda_delay_input.value(), overwrite=owner.mda_overwrite_check.isChecked(),
    )


def write_extensions(owner, ext):
    mapping = {
        "mda_rz_input": ext.raman.planes, "mda_imgp_input": ext.raman.camera_positions,
        "mda_batch_check": ext.raman.batch, "mda_pre_acq_check": ext.raman.pre_acq,
        "mda_repeats_input": ext.raman.repeats,
        "mda_af_mode_combo": "laser",
        "mda_afp_input": ext.autofocus.positions, "mda_refocus_input": ext.autofocus.every,
        "mda_af_range_input": ext.autofocus.search_range, "mda_search_pts_input": ext.autofocus.search_points,
        "mda_fine_range_input": ext.autofocus.fine_range, "mda_fine_pts_input": ext.autofocus.fine_points,
        "mda_seg_track_check": ext.tracking.enabled, "mda_segment_every_input": ext.tracking.every,
        "mda_seg_ch_combo": ext.tracking.channel, "mda_seg_scale_input": ext.tracking.scale,
        "mda_seg_model_combo": ext.tracking.model, "mda_seg_diameter_input": ext.tracking.diameter,
        "mda_seg_crop_combo": str(ext.tracking.crop), "mda_track_cfg_input": ext.tracking.config,
        "mda_delay_input": ext.delay, "mda_overwrite_check": ext.overwrite,
    }
    for name, value in mapping.items():
        widget = getattr(owner, name)
        if isinstance(widget, QComboBox):
            if widget.findText(str(value)) < 0:
                widget.addItem(str(value), str(value))
            widget.setCurrentText(str(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(value)
        elif isinstance(widget, QLineEdit):
            widget.setText(value)
        else:
            widget.setValue(value)

    for axis, checkbox in owner.mda_af_axes.items():
        checkbox.setChecked(ext.autofocus.mode == "laser" and axis in ext.autofocus.axes)
    update_autofocus_controls(owner)


class AcquisitionController(QObject):
    """Own a detached run and its temporary engine/writers through completion."""
    def __init__(self, owner, panel):
        super().__init__(panel)
        self.owner = owner
        self.panel = panel
        self.pending = None
        self.previous_engine = None
        self.writer = None
        self.active_sequence = None
        self._hardware_snapshot = None
        self.timer = QTimer(panel)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_from_timer)
        self.cancelled = False
        self._finishing = False
        owner.core.mda.events.sequenceCanceled.connect(self.on_cancel)
        owner.core.mda.events.sequenceFinished.connect(self.finish)

    @property
    def busy(self):
        return self.pending is not None or self.previous_engine is not None

    def run(self):
        owner = self.owner
        if self.busy or owner.core.mda.is_running():
            raise ValueError("An acquisition is already scheduled or running")
        sequence = self.panel.value()
        ext = read_extensions(owner)
        prepared = prepare_acquisition(sequence, ext, self.panel.targets,
                                       image_shape=(owner.core.getImageWidth(), owner.core.getImageHeight()))
        ext = prepared.extensions
        if prepared.raman_enabled and not ext.raman.batch and "raman_positions" not in prepared.sequence.metadata:
            import numpy as np
            width, height = owner.core.getImageWidth(), owner.core.getImageHeight()
            for source in prepared.sources:
                if source.role == "cell" and any(
                    len(points) and not np.allclose(points * [width, height] / [height, width], .5)
                    for points in source._points_by_position.values()
                ):
                    raise ValueError("Non-batch Raman requires centered stage positions. Use MDA → Raman to apply selections.")
        needs_raman_hardware = prepared.raman_enabled or ext.autofocus.mode in ("laser", "cell", "glass", "quartz")
        if needs_raman_hardware and (owner.collector is None or owner.transformer is None):
            raise ValueError("This experiment requires connected Raman hardware and calibration")
        if owner.core.isSequenceRunning() or getattr(owner, "_live_raman_worker", None) is not None:
            raise ValueError("Stop live acquisition before starting MDA")
        stage_model = None
        if ext.autofocus.mode == "laser":
            if owner._ensure_current_vandermonde() is None or owner.vandermonde is None:
                raise ValueError("Laser autofocus requires current pixel-to-stage calibration")
            stage_model = deepcopy(owner.vandermonde)
        if ext.tracking.enabled:
            # Fail before arming a run for missing explicitly selected weights/config.
            owner._selected_cellpose_model(owner.mda_seg_model_combo)
            if not Path(ext.tracking.config).is_file():
                raise ValueError("Tracking configuration file does not exist")
        estimate = estimate_mda_time(prepared.sequence, prepared.sources, prepared.raman_exposure,
                                     prepared.raman_planes, prepared.image_positions,
                                     point_repeats=ext.raman.repeats, batch=ext.raman.batch)
        requires_save = prepared.raman_enabled or ext.tracking.enabled or GRID_KEY in prepared.sequence.metadata
        output = self.panel.save_info.currentPath().expanduser().resolve() if self.panel.save_info.isChecked() else None
        if requires_save and output is None:
            raise ValueError("Enable Save Acquisition for Raman or tracked runs")
        if output is not None and not ext.overwrite:
            output = self.panel.get_next_available_path(output)
        if output is not None and output.exists() and not output.is_dir():
            raise ValueError("Acquisition output must be a folder")
        message = (f"Output: {output or 'not saved'}\n"
                   f"Camera frames: {estimate.imaging_frames:,}\nRaman collections: {estimate.raman_points:,}\n"
                   f"Estimated duration: {format_duration(estimate.duration_seconds)}\n"
                   f"Start delay: {format_duration(ext.delay)}\n"
                   f"Autofocus: {ext.autofocus.mode}; axes: {', '.join(ext.autofocus.axes) or 'none'}\n"
                   f"Tracking: {ext.tracking.enabled}\n\n"
                   + ("Timing uses remembered overhead plus current exposure; live ETA adapts to this run." if estimate.learned_overheads else "Autofocus, tracking, and motion add time. Their internal images are not saved as camera channels."))
        if ext.overwrite and output is not None and output.exists():
            message += f"\n\nOverwrite permanently replaces all contents of {output}."
        if QMessageBox.question(self.panel, "Review acquisition", message,
                                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel) != QMessageBox.Ok:
            return
        # Modal review pumps Qt events: do not start against a changed microscope.
        if owner.core.mda.is_running() or owner.core is not self.panel._mmc:
            raise ValueError("Microscope state changed during review")
        self._hardware_snapshot = (owner.collector, deepcopy(owner.transformer))
        self.pending = (prepared, output, estimate, stage_model)
        self.panel.lock(True)
        owner._scheduled_raman_mda = {"sequence": prepared.sequence}
        owner.mda_live_eta_label.setText("Live ETA: waiting for scheduled start")
        if ext.delay:
            self.timer.start(min(int(ext.delay * 1000), 2_147_483_647))
        else:
            self.start_pending()

    def _start_from_timer(self):
        try:
            self.start_pending()
        except Exception as error:
            self.owner.status.setText(f"Status: MDA failed — {error}")

    def start_pending(self):
        if self.pending is None:
            return
        owner = self.owner
        prepared, output, estimate, stage_model = self.pending
        self.pending = None
        owner._scheduled_raman_mda = None
        ext = prepared.extensions
        collector, transformer = self._hardware_snapshot
        if GRID_KEY in prepared.sequence.metadata:
            metadata = deepcopy(prepared.sequence.metadata)
            regions = metadata[GRID_KEY]["regions"]
            if any(region.get("z") is None and not region.get("tilt") for region in regions):
                start_z = owner.core.getZPosition()
                for region in regions:
                    if region.get("z") is None and not region.get("tilt"):
                        region["z"] = start_z
                prepared = replace(prepared, sequence=prepared.sequence.replace(metadata=metadata))
        self.active_sequence = prepared.sequence
        if output is not None:
            prepared = replace(prepared, sequence=prepared.sequence.replace(
                metadata={**prepared.sequence.metadata, "dataset_path": str(output)}))
            self.active_sequence = prepared.sequence
        self.previous_engine = owner.core.mda.engine
        try:
            from pymmcore_plus.mda import MDAEngine
            custom = prepared.raman_enabled or ext.tracking.enabled or GRID_KEY in prepared.sequence.metadata or ext.autofocus.mode not in ("off", "hardware") or ext.raman.camera_positions != "all"
            if custom:
                from raman_mda_engine import RamanEngine
                engine = RamanEngine(
                    mmc=owner.core, spectra_collector=collector, transformer=transformer,
                    default_rm_exp=prepared.raman_exposure, sources=list(prepared.sources),
                    batch=ext.raman.batch, pre_acq=ext.raman.pre_acq,
                    autofocus=ext.autofocus.mode not in ("off", "hardware"),
                    autofocus_object=ext.autofocus.mode, autofocus_p=prepared.autofocus_positions,
                    image_p=prepared.image_positions, skip_imaging_for_same_pos=True,
                    segment_and_track=ext.tracking.enabled, refocus_every=ext.autofocus.every,
                    segment_every=ext.tracking.every, segment_channel=ext.tracking.channel,
                    scale=ext.tracking.scale, cellpose_model=ext.tracking.model,
                    cell_diameter=ext.tracking.diameter or None, segment_crop=ext.tracking.crop,
                    tracking_config=ext.tracking.config, autofocus_search_range=ext.autofocus.search_range,
                    search_pts=ext.autofocus.search_points, fine_search_range=ext.autofocus.fine_range,
                    fine_search_pts=ext.autofocus.fine_points, stage_centering_model=stage_model,
                    image_x=owner.core.getImageWidth(), image_y=owner.core.getImageHeight(),
                    config_file=owner.mm_config, point_repeats=ext.raman.repeats,
                    autofocus_axes=ext.autofocus.axes,
                    circle_center=(owner.sel_cx_input.value(), owner.sel_cy_input.value()),
                    circle_radius=owner.sel_r_input.value(),
                )
                owner._connect_raman_visualization(engine)
            else:
                engine = MDAEngine(owner.core)
            owner.core.register_mda_engine(engine)
            if output is not None:
                from raman_mda_engine import RamanAcquisitionWriter
                self.writer = RamanAcquisitionWriter(
                    output, core=owner.core,
                    wavenumbers=collector.get_wavenumbers() if prepared.raman_enabled else None,
                    image_positions=prepared.image_positions, batch=ext.raman.batch, overwrite=ext.overwrite,
                    save_tracking_images=False,
                    exact_path=True,
                )
                owner.mda_writer = self.writer
            owner._raman_mda_pending = True
            owner._raman_mda_writer = self.writer
            owner._start_live_mda_timing(estimate.acquisition_count)
            owner.mda_dir_input.setText(str(output or ""))
            self.cancelled = False
            owner.core.run_mda(prepared.sequence, output=None)
            owner.status.setText("Status: MDA started")
        except Exception:
            self.finish(reason="failed")
            raise

    def stop(self):
        if self.pending is not None:
            self.timer.stop()
            self.pending = None
            self.owner._scheduled_raman_mda = None
            self.panel.lock(False)
            self.owner._finish_live_mda_timing("canceled before start")
        else:
            self.owner.core.mda.cancel()

    def on_cancel(self, *_args):
        self.cancelled = True

    def finish(self, *_args, reason=None):
        if self.previous_engine is None or self._finishing:
            return
        self._finishing = True
        owner = self.owner
        try:
            if reason is None:
                reason = "canceled" if self.cancelled else "completed"
                status = getattr(owner.core.mda, "status", None)
                if callable(status):
                    finished = getattr(status(), "finish_reason", None)
                    if finished is not None:
                        reason = str(getattr(finished, "value", finished))
            reason_text = reason.lower()
            reason = "failed" if "error" in reason_text or "fail" in reason_text else (
                "canceled" if "cancel" in reason_text else "completed"
            )
            try:
                if self.writer is not None:
                    if reason == "completed":
                        self.writer._on_mda_finished(self.active_sequence)
                    else:
                        self.writer.close(status=reason)
                    reason = self.writer.completion_status
            finally:
                self.writer = None
                self.active_sequence = None
                owner.core.register_mda_engine(self.previous_engine)
                self.previous_engine = None
                owner._raman_mda_pending = False
                owner._raman_mda_writer = None
                self.panel.lock(False)
                owner._finish_live_mda_timing(reason)
                owner.status.setText(f"Status: MDA {reason}")
        finally:
            self._finishing = False

    def disconnect(self):
        self.stop() if self.busy else None
        self.owner.core.mda.events.sequenceCanceled.disconnect(self.on_cancel)
        self.owner.core.mda.events.sequenceFinished.disconnect(self.finish)


def install_unified_mda(owner):
    """Replace this window's MDA contents; contain private upstream integration here."""
    from napari_micromanager._gui_objects._mda_widget import MultiDWidget
    from raman_mda_engine.utils import get_mda_widget_from_napari

    if owner._unified_mda is not None:
        return owner._unified_mda

    class UnifiedMDAWidget(MultiDWidget):
        def __init__(self, **kwargs):
            self.targets = []
            self.calibration = {}
            self._extra_metadata = {}
            self._owner = owner
            self._ready = False
            self._locked = False
            super().__init__(**kwargs)
            extend_z_editor(self.z_plan)
            extend_shutter_editor(self.keep_shutter_open)
            self.grid_use_vandermonde, self.grid_calibration_status = extend_grid_calibration(self.grid_plan, owner)
            self.wellplate_calibration_status = extend_wellplate_calibration(self.stage_positions, owner)
            self.af_axis.hide()
            self.stage_positions.af_per_position.setChecked(False)
            self.stage_positions.af_per_position.hide()
            from .autofocus_positions import install_reference_column
            install_reference_column(self, owner)
            from .grid_modes import GridModes
            self.grid_modes = GridModes(self, owner)
            from .raman_selection import install_raman_tab
            install_raman_tab(self, owner)
            QTabWidget.addTab(self.tab_wdg, owner._mda_af_panel, "Autofocus")
            owner._mda_af_panel.show()
            self.extensions_scroll = QScrollArea(self)
            self.extensions_scroll.setWidgetResizable(True)
            self.extensions_scroll.setMaximumHeight(360)
            self.extensions_scroll.setWidget(owner._mda_extension_panel)
            self.layout().insertWidget(self.layout().count() - 1, self.extensions_scroll)
            self.output_description = QLabel()
            self.output_description.setWordWrap(True)
            self.layout().insertWidget(1, self.output_description)
            self.dense_positions = QWidget(self.stage_positions)
            dense_layout = QVBoxLayout(self.dense_positions)
            dense_layout.setContentsMargins(0, 0, 0, 0)
            self.grid_summary = QLabel()
            self.grid_summary.setWordWrap(True)
            dense_layout.addWidget(self.grid_summary)
            self.dense_grid_table = QTableWidget(0, 5)
            self.dense_grid_table.setMaximumHeight(160)
            self.dense_grid_table.setHorizontalHeaderLabels(
                ["Raman grid", "Points (X × Y)", "X range (µm)", "Y range (µm)", "Z (µm)"])
            self.dense_grid_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.dense_grid_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.dense_grid_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.dense_grid_table.horizontalHeader().setStretchLastSection(True)
            dense_layout.addWidget(self.dense_grid_table)
            self.detach_grid_btn = QPushButton("Remove active Raman grid")
            self.detach_grid_btn.clicked.connect(self.detach_grid)
            dense_layout.addWidget(self.detach_grid_btn)
            self.grid_modes.raman_layout.insertWidget(0, self.dense_positions)
            self.dense_positions.hide()
            self._displayed_grid = None
            # Keep upstream writers available, but expose one folder layout here.
            self.save_info._writer_combo.currentTextChanged.connect(self._use_run_folder)
            self._use_run_folder()
            self.save_info.layout().addWidget(self.save_info.save_name, 1, 1, 1, 3)
            owner._mda_extension_panel.show()
            self._ready = True
            self.control_btns.cancel_btn.released.disconnect()
            self.control_btns.cancel_btn.released.connect(lambda: owner._acquisition_controller.stop())
            self.valueChanged.connect(self.update_summary)
            for control in [*owner._mda_extension_panel.findChildren(QWidget), *owner._mda_af_panel.findChildren(QWidget),
                            *owner._raman_settings_panel.findChildren(QWidget)]:
                for signal in ("valueChanged", "textChanged", "toggled", "currentTextChanged"):
                    if hasattr(control, signal):
                        getattr(control, signal).connect(lambda *_: self.valueChanged.emit())
                        break

        def _use_run_folder(self, *_args):
            combo = self.save_info._writer_combo
            combo.hide()
            combo.setEnabled(False)
            if combo.currentText() != "tiff-sequence":
                combo.setCurrentText("tiff-sequence")

        def value(self):
            seq = super().value()
            if not self._ready:
                return seq
            from .autofocus_positions import synchronize_reference_positions
            seq = synchronize_reference_positions(self, seq)
            from .autofocus_references import reference_records
            self.targets = [r for r in self.targets if r.get("role") != "autofocus" or r.get("fixed")]
            if GRID_KEY not in seq.metadata:
                self.targets += reference_records(seq)
            metadata = {**deepcopy(self._extra_metadata), **deepcopy(seq.metadata)}
            metadata[METADATA_KEY] = {
                "grid_mode": self.grid_modes.mode,
                "schema_version": 1, "extensions": read_extensions(owner).to_dict(),
                "targets": deepcopy([r for r in self.targets if r.get("role") != "autofocus" or r.get("fixed")]), "calibration": {
                    **deepcopy(self.calibration),
                    "grid_use_vandermonde": self.grid_use_vandermonde.isChecked(),
                },
            }
            plate_plan = self.stage_positions._plate_plan
            if plate_plan is not None:
                metadata[METADATA_KEY]["wellplate_plan"] = plate_plan.model_dump(mode="json")
            self.grid_modes.compile_regions(metadata)
            if self.grid_modes.mode == "raman":
                metadata[METADATA_KEY]["raman_grid_editor"] = self.grid_modes.editor.settings()
            # CoreConnectedMDAWidget substitutes the current stage position when
            # the native table is empty. Dense grids must retain an empty table.
            if GRID_KEY in metadata:
                metadata[METADATA_KEY]["grid_position_list"] = [
                    position.model_dump(mode="json") for position in self.stage_positions.value()
                ]
                metadata[METADATA_KEY]["grid_positions_enabled"] = bool(self.tab_wdg.isChecked(self.stage_positions))
                # Dense grids traverse virtual stage positions rather than g.
                shutter_axes = tuple("p" if axis == "g" else axis
                                     for axis in seq.keep_shutter_open_across)
                return seq.replace(metadata=metadata, stage_positions=(), grid_plan=None,
                                   keep_shutter_open_across=shutter_axes)
            return seq.replace(metadata=metadata, grid_plan=None if self.grid_modes.mode == "raman" else seq.grid_plan)

        def setValue(self, value):
            from .raman_selection import restore_position_targets
            value = restore_position_targets(value)
            from .autofocus_references import attach_references
            value = attach_references(identify_positions(value), value.metadata.get(METADATA_KEY, {}).get("targets", []))
            self._extra_metadata = deepcopy(value.metadata)
            meta = value.metadata.get(METADATA_KEY, {})
            if meta.get("schema_version", 1) != 1:
                raise ValueError("Unsupported experiment settings version")
            if self._ready:
                self.grid_modes.restore(value)
                self.grid_use_vandermonde.setChecked(meta.get("calibration", {}).get("grid_use_vandermonde", True))
            # Import legacy offsets into native channel definitions exactly once.
            offsets = value.metadata.get("channel_z_offsets_um", {})
            if offsets:
                value = value.replace(channels=[c.replace(z_offset=offsets.get(c.config, c.z_offset)) for c in value.channels])
            if GRID_KEY in value.metadata:
                if "grid_position_list" in meta:
                    value = value.replace(stage_positions=meta["grid_position_list"])
                value = value.replace(keep_shutter_open_across=tuple(
                    "g" if axis == "p" else axis for axis in value.keep_shutter_open_across))
            value = value.replace(autofocus_plan=None, stage_positions=[
                p.replace(sequence=p.sequence.replace(autofocus_plan=None)) if p.sequence else p
                for p in value.stage_positions
            ])
            if self._ready:
                had_plate = self.stage_positions._plate_plan is not None
                self.stage_positions._plate_plan = None
                if had_plate:
                    self.stage_positions._set_position_table_editable(True)
            ready, self._ready = self._ready, False
            try:
                super().setValue(value)
                if hasattr(self, "grid_modes") and self.grid_modes.mode == "raman":
                    self.grid_modes.editor.restore(meta.get("raman_grid_editor"))
                if GRID_KEY in value.metadata and "grid_positions_enabled" in meta:
                    self.tab_wdg.setChecked(self.stage_positions, meta["grid_positions_enabled"])
            finally:
                self._ready = ready
            if self._ready:
                if plate_data := meta.get("wellplate_plan"):
                    from useq import WellPlatePlan
                    from .wellplate_calibration import CalibratedWellPlatePlan
                    model = CalibratedWellPlatePlan if "stage_affine_um" in plate_data else WellPlatePlan
                    plate_plan = model.model_validate(plate_data)
                    # A prepared-position handoff can carry old metadata. Restore
                    # HCS editing only when the plan still describes these positions.
                    if [(p.x, p.y) for p in plate_plan] == [(p.x, p.y) for p in value.stage_positions]:
                        self.stage_positions.setValue(plate_plan)
                        self.stage_positions.setValue(value.stage_positions)
                self.targets = deepcopy(meta.get("targets", []))
                self.calibration = deepcopy(meta.get("calibration", {}))
                ext = ExperimentExtensions.from_dict(meta.get("extensions", {}))
                if not meta and value.metadata.get("raman"):
                    planes = value.metadata["raman"].get("z", [0])
                    ext.raman.planes = planes if isinstance(planes, str) else ",".join(map(str, planes))
                self.grid_modes.restore_regions(value)
                write_extensions(owner, ext)
                self.update_summary()

        def run_mda(self):
            try:
                owner._acquisition_controller.run()
            except Exception as error:
                owner.status.setText(f"Status: cannot start MDA — {error}")
                QMessageBox.warning(self, "Cannot start acquisition", str(error))

        def lock(self, locked):
            self._locked = locked
            self._enable_widgets(not locked)
            self.control_btns.run_btn.setVisible(not locked)
            self.control_btns.cancel_btn.setVisible(locked)
            self.control_btns.pause_btn.setVisible(locked and owner._acquisition_controller.pending is None)
            owner.apply_mda_positions_btn.setEnabled(not locked)
            owner.choose_af_references_btn.setEnabled(not locked)
            self.detach_grid_btn.setEnabled(not locked)
            if hasattr(owner, "apply_grid_mda_btn"):
                owner.apply_grid_mda_btn.setEnabled(not locked and bool(owner._grid_regions))
            self.update_summary()

        def detach_grid(self):
            if owner._acquisition_controller.busy:
                return
            sequence = self.value()
            metadata = deepcopy(sequence.metadata)
            metadata.pop(GRID_KEY, None)
            metadata[METADATA_KEY]["targets"] = []
            if hasattr(owner, "_reset_grid_regions"):
                owner._reset_grid_regions()
            # Dense mode maps the native grid shutter axis to virtual positions.
            shutter_axes = tuple("g" if axis == "p" else axis
                                 for axis in sequence.keep_shutter_open_across)
            self.setValue(sequence.replace(metadata=metadata,
                                           keep_shutter_open_across=shutter_axes))
            self.update_summary()

        def update_summary(self, *_args):
            if not self._ready:
                return
            sequence = self.value()
            from .autofocus_positions import refresh_reference_column
            refresh_reference_column(self, owner)
            dense = GRID_KEY in sequence.metadata
            self.grid_summary.setVisible(dense)
            self.detach_grid_btn.setVisible(dense)
            self._duration_label.setVisible(not dense)
            controller = getattr(owner, "_acquisition_controller", None)
            editable = not self._locked and not (controller and controller.busy)
            self.stage_positions.setEnabled(editable)
            self.grid_plan.setEnabled(editable)
            self.raman_selection_tab.setEnabled(editable)
            self.grid_modes.refresh()
            from .raman_selection import target
            position_owned = any(target(p) for p in sequence.stage_positions)
            owner.mda_batch_check.setEnabled(editable and not position_owned)
            owner.mda_batch_check.setToolTip("Selected Raman positions carry their own centered or integrated-batch mode."
                                           if position_owned else "Integrate legacy aiming points into one spectrum.")
            self.update_dense_positions(sequence)
            if dense:
                count = grid_count(sequence)
                self.grid_summary.setText(f"Dense Raman grid: {count:,} virtual positions. "
                                          "Each row represents a region. Points are generated during acquisition.")
            try:
                plan = prepare_acquisition(sequence, read_extensions(owner), self.targets)
                self._use_run_folder()
                self.output_description.setText(
                    "Save destination is one run folder: camera OME-TIFF, Raman HDF5, and experiment settings."
                )
                estimate = estimate_mda_time(plan.sequence, plan.sources, plan.raman_exposure,
                                             plan.raman_planes, plan.image_positions,
                                             point_repeats=plan.extensions.raman.repeats,
                                             batch=plan.extensions.raman.batch)
                if dense:
                    self.grid_summary.setText(
                        f"Dense Raman grid: {count:,} virtual positions. "
                        f"Estimated duration: {format_duration(estimate.duration_seconds)} "
                        + ("(remembered overhead + current exposure). Positions are generated during acquisition." if estimate.learned_overheads else "plus autofocus/tracking/motion. Positions are generated during acquisition.")
                    )
                owner.mda_collection_summary.setText(
                    f"{estimate.raman_points:,} spectral collections; {estimate.imaging_frames:,} camera frames; "
                    f"estimated {format_duration(estimate.duration_seconds)} " + ("(remembered overhead + current exposure)." if estimate.learned_overheads else "plus autofocus/tracking/motion.")
                )
            except (ValueError, TypeError) as error:
                owner.mda_collection_summary.setText(str(error))

        def update_dense_positions(self, sequence):
            plan = sequence.metadata.get(GRID_KEY)
            dense = plan is not None
            self.dense_positions.setVisible(dense)
            if hasattr(owner, "grid_region_list"):
                self.dense_grid_table.hide()
                self.detach_grid_btn.hide()
            if plan == self._displayed_grid:
                return
            self._displayed_grid = deepcopy(plan)
            regions = plan["regions"] if dense else []
            self.dense_grid_table.setRowCount(len(regions))
            for row, region in enumerate(regions):
                nx, ny = region["x"][2], region["y"][2]
                z = ("Tilt surface" if region.get("tilt") else
                     "At run start" if region.get("z") is None else f"{region['z']:g}")
                values = [f"Grid {row + 1}", f"{nx:,} × {ny:,} = {nx * ny:,}",
                          *[f"{region[axis][0]:g} → {region[axis][1]:g}" for axis in ("x", "y")], z]
                for column, text in enumerate(values):
                    self.dense_grid_table.setItem(row, column, QTableWidgetItem(text))

    old = get_mda_widget_from_napari(owner.main_window)
    value = old.value()
    dock = owner.main_window._dock_widgets["MDA"]
    panel = UnifiedMDAWidget(parent=owner.main_window, mmcore=owner.core)
    panel.setValue(value)
    dock.setWidget(panel)
    old._disconnect()
    old.control_btns._disconnect()
    old.deleteLater()
    owner._unified_mda = panel
    owner._acquisition_controller = AcquisitionController(owner, panel)
    owner.core.mda.events.sequenceFinished.disconnect(owner._on_raman_mda_finished)
    panel.update_summary()
    return panel


def extend_shutter_editor(widget):
    """Add grid traversal to the native Z/time shutter controls."""
    native_value = widget.value
    native_set_value = widget.setValue
    widget.leave_open_g = QCheckBox("g", widget)
    widget.leave_open_g.setToolTip("Keep the shutter open between neighboring grid tiles.")
    widget.layout().insertWidget(widget.layout().count() - 1, widget.leave_open_g)

    def value(_self):
        axes = native_value()
        if widget.leave_open_g.isChecked() and widget.leave_open_g.isEnabled():
            axes += ("g",)
        return axes

    def set_value(_self, axes):
        native_set_value(axes)
        widget.leave_open_g.setChecked("g" in axes)

    widget.value = MethodType(value, widget)
    widget.setValue = MethodType(set_value, widget)
    widget.leave_open_g.toggled.connect(widget.valueChanged)


def extend_z_editor(widget):
    """Support imported irregular planes inside the native MDA Z tab."""
    from useq import ZAbsolutePositions, ZRelativePositions, ZRangeAround, ZTopBottom

    native_value = widget.value
    native_set_value = widget.setValue
    explicit = QGroupBox("Use explicit planes instead of range settings", widget)
    explicit.setCheckable(True)
    explicit.setChecked(False)
    form = QFormLayout(explicit)
    mode = QComboBox()
    mode.addItems(["Relative offsets", "Absolute positions"])
    coordinates = QLineEdit("0")
    form.addRow("Mode", mode)
    form.addRow("Planes (µm, comma-separated)", coordinates)
    widget.layout().addWidget(explicit)

    def value(_self):
        if not explicit.isChecked():
            return native_value()
        try:
            points = [float(p.strip()) for p in coordinates.text().split(",")]
        except ValueError as error:
            raise ValueError("Explicit Z planes must contain numbers") from error
        if mode.currentIndex() == 0:
            return ZRelativePositions(relative=points)
        return ZAbsolutePositions(absolute=points)

    def set_value(_self, plan):
        is_explicit = isinstance(plan, (ZRelativePositions, ZAbsolutePositions))
        if is_explicit:
            relative = isinstance(plan, ZRelativePositions)
            points = list(plan.positions())
            native_set_value(ZRangeAround(range=0, step=1) if relative else
                             ZTopBottom(top=max(points), bottom=min(points), step=1))
            mode.setCurrentIndex(0 if relative else 1)
            coordinates.setText(", ".join(f"{p:g}" for p in points))
        else:
            native_set_value(plan)
        explicit.setChecked(is_explicit)

    def toggle(checked):
        for i in range(widget.layout().count()):
            item = widget.layout().itemAt(i)
            child = item.widget()
            if child is not None and child is not explicit:
                child.setEnabled(not checked)
        widget.valueChanged.emit(widget.value())

    widget.value = MethodType(value, widget)
    widget.setValue = MethodType(set_value, widget)
    explicit.toggled.connect(toggle)
    def changed(*_args):
        try:
            widget.valueChanged.emit(widget.value())
        except ValueError:
            # Intermediate edits can be invalid; Run performs final validation.
            pass

    mode.currentTextChanged.connect(changed)
    coordinates.textChanged.connect(changed)


def apply_prepared_positions(owner):
    if owner._unified_mda is None or owner.selection_results is None:
        owner.status.setText("Status: connect and prepare targets first")
        return
    if owner._acquisition_controller.busy:
        owner.status.setText("Status: stop the acquisition before applying targets")
        return
    panel = owner._unified_mda
    selection = owner.selection_results
    old_sequence = panel.value()
    dense = GRID_KEY in selection["new_seq"].metadata
    positions = () if dense else identify_positions(selection["new_seq"]).stage_positions
    reference_positions = positions
    # A repeated batch FOV needs distinct saved identities; the reference can
    # still be shared by its copies at the same physical coordinates.
    if not dense:
        import uuid
        from .experiment import position_id
        seen = set()
        independent = []
        for position in positions:
            identity = position_id(position)
            if identity in seen:
                new_id = uuid.uuid4().hex
                sub = position.sequence
                if sub:
                    sub = sub.replace(metadata={**sub.metadata, "position_identity": new_id})
                position = position.replace(name=(position.name or "Position").split(" [id:")[0], sequence=sub)
            seen.add(position_id(position))
            independent.append(position)
        positions = independent
    sequence = panel.value().replace(stage_positions=positions, grid_plan=None)
    if dense:
        from useq import MDASequence, Position
        records = capture_targets(MDASequence(stage_positions=[Position()]), selection["sources"])
        for record in records:
            record["fixed"] = True
            record["position_ids"] = []
    else:
        records = capture_targets(sequence, selection["sources"])
        from .autofocus_references import preserve_references
        records.extend(preserve_references(old_sequence, sequence, panel.targets, reference_positions))
    ext = read_extensions(owner)
    # Autofocus policy is configured independently in MDA.
    ext.raman.batch = bool(selection.get("batch", False))
    ext.raman.pre_acq = bool(selection.get("pre_acq", True))
    # Grid preparation owns geometry; retain MDA's requested frame count.
    if selection.get("selection_type") != "grid":
        ext.raman.repeats = owner.sel_repeats_input.value()
    # Keep MDA's camera position policy so selected BF/other camera channels
    # are acquired at the requested grid positions.
    metadata = deepcopy(sequence.metadata)
    metadata.pop(GRID_KEY, None)
    if dense:
        metadata[GRID_KEY] = deepcopy(selection["new_seq"].metadata[GRID_KEY])
    metadata[METADATA_KEY] = {
        "grid_mode": "raman" if selection.get("selection_type") == "grid" else panel.grid_modes.mode,
        "schema_version": 1, "extensions": ext.to_dict(), "targets": records,
        "calibration": {"aiming": owner.tf_path.text(), "pixel_to_stage": str(owner.vandermonde_path or "")},
    }
    panel.setValue(sequence.replace(metadata=metadata))
    panel.update_summary()
    owner.main_window._dock_widgets["MDA"].show()
    owner.status.setText(f"Status: attached dense grid ({grid_count(selection['new_seq']):,} virtual positions)"
                         if dense else f"Status: applied {len(positions)} prepared positions to MDA")
