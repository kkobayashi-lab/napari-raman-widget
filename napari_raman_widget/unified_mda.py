"""Application-owned MDA dock and its shared acquisition controller."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType

from qtpy.QtCore import QObject, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QLabel, QLayout, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .experiment import (
    AutofocusSettings, ExperimentExtensions, METADATA_KEY, RamanSettings,
    TrackingSettings, capture_targets, identify_positions, prepare_acquisition,
)
from .mda_timing import estimate_mda_time, format_duration
from .ui_helpers import make_collapsible
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

    raman = section("Raman acquisition")
    owner._raman_settings_form = raman
    raman.addRow(QLabel("RM channel exposure is milliseconds per collection.\n"
                         "Individual: per aiming point. Batch: per integrated spectrum."))
    owner.mda_batch_check = QCheckBox("Integrated batch collection")
    raman.addRow(owner.mda_batch_check)
    owner.mda_pre_acq_check = QCheckBox("Preposition galvo before collection")
    owner.mda_pre_acq_check.setChecked(True)
    raman.addRow(owner.mda_pre_acq_check)
    field(raman, "mda_repeats_input", "Frames per point", 1, minimum=1)
    field(raman, "mda_rz_input", "Raman planes (all / middle / indices)", "all")
    field(raman, "mda_imgp_input", "Camera positions (all / none / indices)", "all")
    owner.mda_collection_summary = QLabel("Add RM to the MDA channel table to collect spectra.")
    owner.mda_collection_summary.setWordWrap(True)
    raman.addRow(owner.mda_collection_summary)

    af = section("Custom autofocus")
    af.addRow(QLabel("Hardware autofocus on axis takes priority over these custom settings."))
    owner.mda_af_mode_combo = QComboBox()
    owner.mda_af_mode_combo.addItems(["off", "software", "laser", "cell", "glass", "quartz"])
    af.addRow("Custom mode", owner.mda_af_mode_combo)
    field(af, "mda_afp_input", "Positions (all / none / indices)", "all")
    field(af, "mda_refocus_input", "Refocus every N timepoints (0 = initial only)", 0)
    owner._af_range_label = field(af, "mda_af_range_input", "Search range (± µm)", 6.0, minimum=0.1)
    owner._search_pts_label = field(af, "mda_search_pts_input", "Search points", 8, minimum=2, maximum=500)
    owner._fine_range_label = field(af, "mda_fine_range_input", "Laser fine range (± µm)", 1.5, minimum=0.1)
    owner._fine_pts_label = field(af, "mda_fine_pts_input", "Laser fine search points", 8, minimum=2, maximum=500)

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
    field(run, "mda_overwrite_check", "Overwrite combined acquisition folder", False)
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
    owner._toggle_autofocus_fields("off")
    owner._toggle_seg_track_fields(False)
    layout.addStretch()
    return panel


def read_extensions(owner):
    return ExperimentExtensions(
        raman=RamanSettings(owner.mda_rz_input.text(), owner.mda_imgp_input.text(),
                            owner.mda_batch_check.isChecked(), owner.mda_pre_acq_check.isChecked(),
                            owner.mda_repeats_input.value()),
        autofocus=AutofocusSettings(owner.mda_af_mode_combo.currentText(), owner.mda_afp_input.text(),
                                    owner.mda_refocus_input.value(), owner.mda_af_range_input.value(),
                                    owner.mda_search_pts_input.value(), owner.mda_fine_range_input.value(),
                                    owner.mda_fine_pts_input.value()),
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
        "mda_af_mode_combo": "off" if ext.autofocus.mode == "hardware" else ext.autofocus.mode,
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
        native_af = sequence.autofocus_plan or any(
            p.sequence and p.sequence.autofocus_plan for p in sequence.stage_positions
        )
        prepared = prepare_acquisition(sequence, ext, self.panel.targets)
        ext = prepared.extensions
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
        combined = prepared.raman_enabled or ext.tracking.enabled or GRID_KEY in prepared.sequence.metadata
        output = self.panel.save_info.currentPath().expanduser().resolve() if self.panel.save_info.isChecked() else None
        if combined and output is None:
            raise ValueError("Enable Save Acquisition for Raman or tracked runs")
        if output is not None and not ext.overwrite:
            output = self.panel.get_next_available_path(output)
        if output is not None and combined and output.exists() and not output.is_dir():
            raise ValueError("Combined output must be a folder")
        if ext.overwrite and not combined:
            raise ValueError("Overwrite is available only for combined acquisition folders")
        message = (f"Output: {output or 'not saved'}\n"
                   f"Camera frames: {estimate.imaging_frames:,}\nRaman collections: {estimate.raman_points:,}\n"
                   f"Estimated duration: {format_duration(estimate.duration_seconds)}\n"
                   f"Start delay: {format_duration(ext.delay)}\n"
                   f"Hardware autofocus: {bool(native_af)}; custom autofocus: {'off' if native_af else ext.autofocus.mode}\n"
                   f"Tracking: {ext.tracking.enabled}\n\n"
                   "Autofocus, tracking, and motion add time. Their internal images are not saved as camera channels.")
        if ext.overwrite and output.exists():
            message += f"\n\nOverwrite permanently replaces all contents of {output}."
        if QMessageBox.question(self.panel, "Review acquisition", message,
                                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel) != QMessageBox.Ok:
            return
        # Modal review pumps Qt events: do not start against a changed microscope.
        if owner.core.mda.is_running() or owner.core is not self.panel._mmc:
            raise ValueError("Microscope state changed during review")
        self._hardware_snapshot = (owner.collector, deepcopy(owner.transformer))
        self.pending = (prepared, output, estimate, stage_model, combined)
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
        prepared, output, estimate, stage_model, combined = self.pending
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
                    circle_center=(owner.sel_cx_input.value(), owner.sel_cy_input.value()),
                    circle_radius=owner.sel_r_input.value(),
                )
                owner._connect_raman_visualization(engine)
            else:
                engine = MDAEngine(owner.core)
            owner.core.register_mda_engine(engine)
            if combined:
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
            owner.core.run_mda(prepared.sequence, output=None if combined else output)
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
            super().__init__(**kwargs)
            extend_z_editor(self.z_plan)
            extend_shutter_editor(self.keep_shutter_open)
            self.extensions_scroll = QScrollArea(self)
            self.extensions_scroll.setWidgetResizable(True)
            self.extensions_scroll.setMaximumHeight(360)
            self.extensions_scroll.setWidget(owner._mda_extension_panel)
            self.layout().insertWidget(self.layout().count() - 1, self.extensions_scroll)
            self.output_description = QLabel()
            self.output_description.setWordWrap(True)
            self.layout().insertWidget(1, self.output_description)
            self.grid_summary = QLabel()
            self.grid_summary.setWordWrap(True)
            self.layout().insertWidget(2, self.grid_summary)
            self.detach_grid_btn = QPushButton("Detach dense Raman grid")
            self.detach_grid_btn.clicked.connect(self.detach_grid)
            self.layout().insertWidget(3, self.detach_grid_btn)
            self._native_format = self.save_info._writer_combo.currentText()
            owner._mda_extension_panel.show()
            self._ready = True
            self.control_btns.cancel_btn.released.disconnect()
            self.control_btns.cancel_btn.released.connect(lambda: owner._acquisition_controller.stop())
            self.valueChanged.connect(self.update_summary)
            for control in owner._mda_extension_panel.findChildren(QWidget):
                for signal in ("valueChanged", "textChanged", "toggled", "currentTextChanged"):
                    if hasattr(control, signal):
                        getattr(control, signal).connect(lambda *_: self.valueChanged.emit())
                        break

        def value(self):
            seq = super().value()
            if not self._ready:
                return seq
            metadata = {**deepcopy(self._extra_metadata), **deepcopy(seq.metadata)}
            metadata[METADATA_KEY] = {
                "schema_version": 1, "extensions": read_extensions(owner).to_dict(),
                "targets": deepcopy(self.targets), "calibration": deepcopy(self.calibration),
            }
            # CoreConnectedMDAWidget substitutes the current stage position when
            # the native table is empty. Dense grids must retain an empty table.
            if GRID_KEY in metadata:
                # Dense grids traverse virtual stage positions rather than g.
                shutter_axes = tuple("p" if axis == "g" else axis
                                     for axis in seq.keep_shutter_open_across)
                return seq.replace(metadata=metadata, stage_positions=(), grid_plan=None,
                                   keep_shutter_open_across=shutter_axes)
            return seq.replace(metadata=metadata)

        def setValue(self, value):
            self._extra_metadata = deepcopy(value.metadata)
            meta = value.metadata.get(METADATA_KEY, {})
            if meta.get("schema_version", 1) != 1:
                raise ValueError("Unsupported experiment settings version")
            # Import legacy offsets into native channel definitions exactly once.
            offsets = value.metadata.get("channel_z_offsets_um", {})
            if offsets:
                value = value.replace(channels=[c.replace(z_offset=offsets.get(c.config, c.z_offset)) for c in value.channels])
            if GRID_KEY in value.metadata:
                value = value.replace(keep_shutter_open_across=tuple(
                    "g" if axis == "p" else axis for axis in value.keep_shutter_open_across))
            super().setValue(value)
            if self._ready:
                self.targets = deepcopy(meta.get("targets", []))
                self.calibration = deepcopy(meta.get("calibration", {}))
                ext = ExperimentExtensions.from_dict(meta.get("extensions", {}))
                if not meta and value.metadata.get("raman"):
                    planes = value.metadata["raman"].get("z", [0])
                    ext.raman.planes = planes if isinstance(planes, str) else ",".join(map(str, planes))
                write_extensions(owner, ext)

        def run_mda(self):
            try:
                owner._acquisition_controller.run()
            except Exception as error:
                owner.status.setText(f"Status: cannot start MDA — {error}")
                QMessageBox.warning(self, "Cannot start acquisition", str(error))

        def lock(self, locked):
            self._enable_widgets(not locked)
            self.control_btns.run_btn.setVisible(not locked)
            self.control_btns.cancel_btn.setVisible(locked)
            self.control_btns.pause_btn.setVisible(locked and owner._acquisition_controller.pending is None)
            owner.apply_mda_positions_btn.setEnabled(not locked)
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
            self.setValue(sequence.replace(metadata=metadata))
            self.update_summary()

        def update_summary(self, *_args):
            if not self._ready:
                return
            sequence = self.value()
            dense = GRID_KEY in sequence.metadata
            self.grid_summary.setVisible(dense)
            self.detach_grid_btn.setVisible(dense)
            self._duration_label.setVisible(not dense)
            controller = getattr(owner, "_acquisition_controller", None)
            editable = not (controller and controller.busy)
            self.stage_positions.setEnabled(editable and not dense)
            self.grid_plan.setEnabled(editable and not dense)
            if dense:
                count = grid_count(sequence)
                self.grid_summary.setText(f"Dense Raman grid: {count:,} virtual positions. "
                                          "Positions are generated during acquisition; the native position table stays empty.")
            try:
                plan = prepare_acquisition(sequence, read_extensions(owner), self.targets)
                combined = plan.raman_enabled or plan.extensions.tracking.enabled or dense
                combo = self.save_info._writer_combo
                if combined:
                    if combo.isEnabled():
                        self._native_format = combo.currentText()
                    combo.setEnabled(False)
                    if combo.currentText() != "tiff-sequence":
                        combo.setCurrentText("tiff-sequence")
                elif not combo.isEnabled():
                    combo.setCurrentText(self._native_format)
                combo.setEnabled(not combined)
                self.output_description.setText(
                    "Save destination is one run folder: camera OME-TIFF, Raman HDF5, and experiment settings."
                    if combined else "Camera data uses the selected MDA save format."
                )
                estimate = estimate_mda_time(plan.sequence, plan.sources, plan.raman_exposure,
                                             plan.raman_planes, plan.image_positions,
                                             point_repeats=plan.extensions.raman.repeats,
                                             batch=plan.extensions.raman.batch)
                if dense:
                    self.grid_summary.setText(
                        f"Dense Raman grid: {count:,} virtual positions. "
                        f"Estimated duration: {format_duration(estimate.duration_seconds)} "
                        "plus autofocus/tracking/motion. Positions are generated during acquisition."
                    )
                owner.mda_collection_summary.setText(
                    f"{estimate.raman_points:,} spectral collections; {estimate.imaging_frames:,} camera frames; "
                    f"estimated {format_duration(estimate.duration_seconds)} plus autofocus/tracking/motion."
                )
            except (ValueError, TypeError) as error:
                owner.mda_collection_summary.setText(str(error))

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
    dense = GRID_KEY in selection["new_seq"].metadata
    positions = () if dense else identify_positions(selection["new_seq"]).stage_positions
    sequence = panel.value().replace(stage_positions=positions, grid_plan=None)
    if dense:
        from useq import MDASequence, Position
        records = capture_targets(MDASequence(stage_positions=[Position()]), selection["sources"])
        for record in records:
            record["fixed"] = True
            record["position_ids"] = []
    else:
        records = capture_targets(sequence, selection["sources"])
    ext = read_extensions(owner)
    # Grid preparation must not overwrite MDA's autofocus mode or position policy.
    if selection.get("selection_type") != "grid":
        ext.autofocus.mode = str(selection.get("autofocus_object", "off")).lower()
        if ext.autofocus.mode == "none":
            ext.autofocus.mode = "off"
        ext.autofocus.positions = ",".join(map(str, selection.get("autofocus_p", []))) or "none"
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
        "schema_version": 1, "extensions": ext.to_dict(), "targets": records,
        "calibration": {"aiming": owner.tf_path.text(), "pixel_to_stage": str(owner.vandermonde_path or "")},
    }
    panel.setValue(sequence.replace(metadata=metadata))
    panel.update_summary()
    owner.main_window._dock_widgets["MDA"].show()
    owner.status.setText(f"Status: attached dense grid ({grid_count(selection['new_seq']):,} virtual positions)"
                         if dense else f"Status: applied {len(positions)} prepared positions to MDA")
