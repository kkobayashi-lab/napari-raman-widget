"""The main HardwareWidget: a dockable napari panel for the CNS Raman rig."""
import os
import time
import uuid
from pathlib import Path

import xarray as xr
import napari
import numpy as np
from qtpy.QtCore import QEvent, QTimer, Qt, QUrl, Slot
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QMessageBox
)
from .acquisition_viewer import LargeAcquisitionViewerWindow
from .field_help import apply_tooltips
from .log_window import LogWindow, _StdoutRedirector
from .lazy_visualization import install_lazy_mda_viewer
from .plot_windows import (
    CalibrationPlotWindow, GridScanPlotWindow, ReferenceSpectraWindow,
    SpectrumWindow,
)
from .ui_helpers import make_collapsible


DEFAULT_LIGHTFIELD_CONFIG = (
    r"C:\Users\spraman\Documents\LightField\Experiments\RamanConfocal.lfe"
)
DEFAULT_BEAM_CENTER_XY = (512.0, 512.0)
ND_FILTER_DEVICE = "DigitalIO"
ND_FILTER_MASK = 1 << 1  # Dev1/port0/line1
STAGE_DRAG_INTERVAL_MS = 100
STAGE_DRAG_DEAD_ZONE_PX = 10.0
STAGE_DRAG_FULL_SPEED_PX = 100.0


def _enable_raman_mda_options():
    """Expose channel-major order and position shutter control in the MDA editor."""
    from pymmcore_widgets.useq_widgets import _mda_sequence

    _mda_sequence.ALLOWED_ORDERS.add("cpzt")

    base = _mda_sequence.KeepShutterOpen
    if getattr(base, "_raman_position_axis", False):
        return

    class KeepShutterOpenAcrossPosition(base):
        _raman_position_axis = True

        def __init__(self, parent=None):
            super().__init__(parent)
            self.leave_open_p = QCheckBox("p")
            self.layout().insertWidget(1, self.leave_open_p)
            self.leave_open_p.toggled.connect(self.valueChanged)

        def value(self):
            axes = super().value()
            if self.leave_open_p.isChecked() and self.leave_open_p.isEnabled():
                return ("p", *axes)
            return axes

        def setValue(self, value):
            super().setValue(value)
            self.leave_open_p.setChecked("p" in value)

    _mda_sequence.KeepShutterOpen = KeepShutterOpenAcrossPosition


def _parse_raman_z_indices(text):
    """Parse Raman Z indices, with ``None``/``Off`` disabling spectra."""
    stripped = text.strip()
    if stripped.lower() in {"none", "off"}:
        return []
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    if not parts:
        raise ValueError("Raman z indices is empty")
    try:
        return [int(part) for part in parts]
    except ValueError:
        raise ValueError(
            f"Raman z indices contains non-integer entries: {text!r}"
        )


def _raman_free_autofocus_allowed(autofocus_enabled, autofocus_object):
    """Allow autofocus when the main acquisition has no Raman channel.

    Laser, cell, glass, and quartz modes may still use the Raman hardware
    during autofocus; "Raman-free" here only means that the main acquisition
    does not record Raman spectra.
    """
    supported = {"laser", "software", "cell", "glass", "quartz"}
    return not autofocus_enabled or autofocus_object in supported


def _spatial_yx(point):
    """Return the final Y/X coordinates from one napari point."""
    point = np.asarray(point)
    if point.ndim != 1 or point.size < 2:
        raise ValueError(
            "Expected one napari point with at least Y and X coordinates"
        )
    return point[-2:]


class HardwareWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.core = None
        self.collector = None
        self.daq = None
        self.transformer = None
        self.vandermonde = None
        self.vandermonde_objective = None
        self.vandermonde_path = None
        self.default_engine = None
        self.calibration_ds = None
        self.calibrator = None
        self.selector = None
        self.scan_ds = None
        self.main_window = None
        self.selection_results = None
        self.mda_channel_rows = []
        self.mda_writer = None
        self._mda_completion_events = None
        self._raman_mda_pending = False
        self._raman_mda_canceled = False
        self._raman_mda_writer = None
        self._raman_visualization_engine = None
        self._lazy_mda_viewer = None
        self.px2stage_picker = None
        self.px2stage_xy = None
        self.px2stage_objective = None
        self.mm_config = None
        self._shutter_return_channel = "BF"
        self._stage_drag_active = False
        self._stage_drag_anchor_yx = None
        self._stage_drag_current_yx = None
        self._stage_drag_timer = QTimer(self)
        self._stage_drag_timer.setInterval(STAGE_DRAG_INTERVAL_MS)
        self._stage_drag_timer.timeout.connect(self._stage_drag_tick)
        outer = QVBoxLayout()

        self.manual_link = QLabel(
            '<a href="manual" '
            'style="color:white; font-weight:bold; text-decoration:none;">'
            'Help</a>'
        )
        self.manual_link.setAlignment(Qt.AlignRight)
        self.manual_link.setOpenExternalLinks(False)
        self.manual_link.setToolTip("Open the user manual")
        self.manual_link.linkActivated.connect(self.open_user_manual)

        outer.addWidget(self.manual_link)

        # ================= LOADING SECTION =================
        loading_box = make_collapsible("Loading", expanded=True)
        loading_layout = QVBoxLayout()

        loading_layout.addWidget(QLabel("Micro-Manager config (.cfg):"))
        cfg_row = QHBoxLayout()
        self.cfg_path = QLineEdit()
        self.cfg_path.setText(r"C:\Users\spraman\Desktop\config\exp_1ms_polysterenebeads_125mW_02_18_bakkk_v2.cfg")
        self.cfg_path.setPlaceholderText(r"C:\Users\spraman\Desktop\config\exp_1ms_polysterenebeads_125mW_02_18_bakkk_v2.cfg")
        cfg_browse = QPushButton("...")
        cfg_browse.setFixedWidth(30)
        cfg_browse.clicked.connect(self.browse_cfg)
        cfg_row.addWidget(self.cfg_path)
        cfg_row.addWidget(cfg_browse)
        loading_layout.addLayout(cfg_row)

        loading_layout.addWidget(QLabel("Transformer model (.json):"))
        tf_row = QHBoxLayout()
        self.tf_path = QLineEdit()
        self.tf_path.setText(r"C:\Users\spraman\Desktop\config\model_2026-07-16.json")
        self.tf_path.setPlaceholderText(r"C:\Users\spraman\Desktop\config\model_2026-07-16.json")
        tf_browse = QPushButton("...")
        tf_browse.setFixedWidth(30)
        tf_row.addWidget(self.tf_path)
        tf_row.addWidget(tf_browse)
        loading_layout.addLayout(tf_row)

        loading_layout.addWidget(QLabel("LightField experiment:"))
        self.lightfield_config = QLineEdit()
        self.lightfield_config.setText(DEFAULT_LIGHTFIELD_CONFIG)
        self.lightfield_config.setPlaceholderText(DEFAULT_LIGHTFIELD_CONFIG)
        loading_layout.addWidget(self.lightfield_config)

        loading_layout.addWidget(QLabel("Vandermonde model (.json):"))
        vdm_row = QHBoxLayout()
        self.sel_vdm_path = QLineEdit()
        self.sel_vdm_path.setText(r"C:\Users\spraman\Desktop\config\2026_07_23_vandermonde_model.json")
        self.sel_vdm_path.setPlaceholderText("vandermonde_model.json")
        vdm_browse = QPushButton("...")
        vdm_browse.setFixedWidth(30)
        vdm_browse.clicked.connect(self.browse_vandermonde)
        vdm_row.addWidget(self.sel_vdm_path)
        vdm_row.addWidget(vdm_browse)
        loading_layout.addLayout(vdm_row)

        objective_row = QHBoxLayout()
        objective_row.addWidget(QLabel("Objective index:"))
        self.objective_combo = QComboBox()
        self.objective_combo.addItem("(connect first)")
        self.objective_combo.setEnabled(False)
        objective_row.addWidget(self.objective_combo)
        loading_layout.addLayout(objective_row)

        loading_layout.addWidget(
            QLabel("Output folder (optional, applied on connect):")
        )
        out_row = QHBoxLayout()
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("(current directory)")
        out_browse = QPushButton("...")
        out_browse.setFixedWidth(30)
        out_browse.clicked.connect(self.browse_out)
        out_row.addWidget(self.out_path)
        out_row.addWidget(out_browse)
        loading_layout.addLayout(out_row)

        self.connect_btn = QPushButton("Connect hardware")
        self.disconnect_btn = QPushButton("Disconnect")
        self.reload_tf_btn = QPushButton("Reload transformer")
        self.disconnect_btn.setEnabled(False)
        self.reload_tf_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect)
        self.disconnect_btn.clicked.connect(self.disconnect)
        self.reload_tf_btn.clicked.connect(self.reload_transformer)

        loading_layout.addWidget(self.connect_btn)
        loading_layout.addWidget(self.disconnect_btn)
        loading_layout.addWidget(self.reload_tf_btn)

        loading_box.setLayout(loading_layout)
        outer.addWidget(loading_box)

        # ================= HARDWARE CONTROL SECTION =================
        hardware_box = make_collapsible("Hardware Control", expanded=False)
        hardware_layout = QVBoxLayout()

        click_row = QHBoxLayout()
        self.click_center_btn = QPushButton("Click to center")
        self.click_center_btn.setCheckable(True)
        self.click_center_btn.toggled.connect(self._toggle_click_to_center)
        self.click_laser_btn = QPushButton("Click to point laser")
        self.click_laser_btn.setCheckable(True)
        self.click_laser_btn.toggled.connect(self._toggle_click_to_laser)
        click_row.addWidget(self.click_center_btn)
        click_row.addWidget(self.click_laser_btn)
        hardware_layout.addLayout(click_row)

        drag_row = QHBoxLayout()
        self.drag_stage_btn = QPushButton("Drag image to move stage")
        self.drag_stage_btn.setCheckable(True)
        self.drag_stage_btn.toggled.connect(self._toggle_stage_drag)
        self.stage_drag_speed_input = QDoubleSpinBox()
        self.stage_drag_speed_input.setRange(1.0, 1000.0)
        self.stage_drag_speed_input.setValue(800.0)
        self.stage_drag_speed_input.setDecimals(1)
        self.stage_drag_speed_input.setSuffix(" um/s max")
        drag_row.addWidget(self.drag_stage_btn)
        drag_row.addWidget(self.stage_drag_speed_input)
        hardware_layout.addLayout(drag_row)

        shutter_row = QHBoxLayout()
        shutter_row.addWidget(QLabel("Laser shutter:"))
        self.open_shutter_btn = QPushButton("Open (RM)")
        self.close_shutter_btn = QPushButton("Close")
        self.open_shutter_btn.clicked.connect(
            lambda _checked=False: self._set_laser_shutter(True)
        )
        self.close_shutter_btn.clicked.connect(
            lambda _checked=False: self._set_laser_shutter(False)
        )
        shutter_row.addWidget(self.open_shutter_btn)
        shutter_row.addWidget(self.close_shutter_btn)
        hardware_layout.addLayout(shutter_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("ND filter:"))
        self.open_filter_btn = QPushButton("Open (remove)")
        self.close_filter_btn = QPushButton("Close (insert)")
        self.open_filter_btn.clicked.connect(
            lambda _checked=False: self._set_nd_filter(True)
        )
        self.close_filter_btn.clicked.connect(
            lambda _checked=False: self._set_nd_filter(False)
        )
        filter_row.addWidget(self.open_filter_btn)
        filter_row.addWidget(self.close_filter_btn)
        hardware_layout.addLayout(filter_row)

        hardware_box.setLayout(hardware_layout)
        outer.addWidget(hardware_box)

        # ================= COLLECT SPECTRUM SECTION =================
        raman_box = make_collapsible(
            "Collect spectra using points layer", expanded=False
        )
        raman_layout = QVBoxLayout()

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure (ms):"))
        self.exposure_input = QDoubleSpinBox()
        self.exposure_input.setRange(1, 1_000_000)
        self.exposure_input.setValue(1000)
        self.exposure_input.setDecimals(1)
        exp_row.addWidget(self.exposure_input)
        raman_layout.addLayout(exp_row)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("N (repeats, >=2):"))
        self.n_input = QSpinBox()
        self.n_input.setRange(2, 1000)
        self.n_input.setValue(2)
        n_row.addWidget(self.n_input)
        raman_layout.addLayout(n_row)
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save as (blank = don't save):"))
        self.collect_save_input = QLineEdit()
        self.collect_save_input.setPlaceholderText("filename (no extension)")
        save_row.addWidget(self.collect_save_input)
        raman_layout.addLayout(save_row)

        self.collect_btn = QPushButton("Collect spectra at last point")
        self.collect_btn.clicked.connect(self.collect_raman)
        raman_layout.addWidget(self.collect_btn)

        raman_box.setLayout(raman_layout)
        outer.addWidget(raman_box)

        # ================= LASER AIMING CALIBRATION SECTION =================
        calib_box = make_collapsible("Laser aiming calibration", expanded=False)
        calib_layout = QVBoxLayout()

        cal_n_row = QHBoxLayout()
        cal_n_row.addWidget(QLabel("N (repeats):"))
        self.cal_n_input = QSpinBox()
        self.cal_n_input.setRange(1, 1000)
        self.cal_n_input.setValue(2)
        cal_n_row.addWidget(self.cal_n_input)
        calib_layout.addLayout(cal_n_row)

        cal_exp_row = QHBoxLayout()
        cal_exp_row.addWidget(QLabel("Exposure (ms):"))
        self.cal_exp_input = QDoubleSpinBox()
        self.cal_exp_input.setRange(1, 1_000_000)
        self.cal_exp_input.setValue(100)
        self.cal_exp_input.setDecimals(1)
        cal_exp_row.addWidget(self.cal_exp_input)
        calib_layout.addLayout(cal_exp_row)

        cal_volts_row = QHBoxLayout()
        cal_volts_row.addWidget(QLabel("Max volts:"))
        self.cal_volts_input = QDoubleSpinBox()
        self.cal_volts_input.setRange(0.01, 10.0)
        self.cal_volts_input.setValue(1.6)
        self.cal_volts_input.setDecimals(2)
        self.cal_volts_input.setSingleStep(0.1)
        cal_volts_row.addWidget(self.cal_volts_input)
        calib_layout.addLayout(cal_volts_row)

        cal_grid_row = QHBoxLayout()
        cal_grid_row.addWidget(QLabel("Grid size:"))
        self.cal_grid_input = QSpinBox()
        self.cal_grid_input.setRange(2, 100)
        self.cal_grid_input.setValue(10)
        cal_grid_row.addWidget(self.cal_grid_input)
        calib_layout.addLayout(cal_grid_row)

        cal_thres_row = QHBoxLayout()
        cal_thres_row.addWidget(QLabel("Threshold:"))
        self.cal_thres_input = QDoubleSpinBox()
        self.cal_thres_input.setRange(0.0, 100.0)
        self.cal_thres_input.setValue(10)
        self.cal_thres_input.setDecimals(2)
        self.cal_thres_input.setSingleStep(0.1)
        cal_thres_row.addWidget(self.cal_thres_input)
        calib_layout.addLayout(cal_thres_row)

        self.calibrate_btn = QPushButton("Run calibration")
        self.calibrate_btn.clicked.connect(self.run_calibration)
        calib_layout.addWidget(self.calibrate_btn)

        # --- recalibration (shown when checked) ---
        self.recal_check = QCheckBox("Recalibration")
        self.recal_check.setChecked(False)
        self.recal_check.toggled.connect(self._toggle_recal_fields)
        calib_layout.addWidget(self.recal_check)

        self._recal_help = QLabel(
            "Opens manual selector on the last calibration dataset.\n"
            "Click points, Enter to advance, Backspace to go back,\n"
            "R to reset, N to mark as NaN. Close window when done,\n"
            "then click Save to write the new model."
        )
        self._recal_help.setWordWrap(True)
        calib_layout.addWidget(self._recal_help)
        model_name_row = QHBoxLayout()
        self._recal_model_name_label = QLabel("Model name:")
        model_name_row.addWidget(self._recal_model_name_label)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("model_2026-01-08")
        model_name_row.addWidget(self.model_name_input)
        calib_layout.addLayout(model_name_row)
        self.open_selector_btn = QPushButton("Open manual selector")
        self.open_selector_btn.clicked.connect(self.open_selector)
        calib_layout.addWidget(self.open_selector_btn)
        self.save_model_btn = QPushButton("Save recalibrated model")
        self.save_model_btn.clicked.connect(self.save_recalibration)
        calib_layout.addWidget(self.save_model_btn)

        # collect recal widgets so they can be hidden as a group
        self._recal_widgets = [
            self._recal_help,
            self._recal_model_name_label, self.model_name_input,
            self.open_selector_btn, self.save_model_btn,
        ]
        self._toggle_recal_fields(False)   # hidden until checked

        calib_box.setLayout(calib_layout)
        outer.addWidget(calib_box)

        # ================= COLLECT REFERENCE SPECTRA SECTION =================
        ref_box = make_collapsible("Axial background scan", expanded=False)
        ref_layout = QVBoxLayout()

        ref_name_row = QHBoxLayout()
        ref_name_row.addWidget(QLabel("Name:"))
        self.ref_name_input = QLineEdit()
        self.ref_name_input.setPlaceholderText("testing")
        ref_name_row.addWidget(self.ref_name_input)
        ref_layout.addLayout(ref_name_row)

        ref_exp_row = QHBoxLayout()
        ref_exp_row.addWidget(QLabel("Exposure (ms):"))
        self.ref_exp_input = QDoubleSpinBox()
        self.ref_exp_input.setRange(1, 1_000_000)
        self.ref_exp_input.setValue(1000)
        self.ref_exp_input.setDecimals(1)
        ref_exp_row.addWidget(self.ref_exp_input)
        ref_layout.addLayout(ref_exp_row)

        ref_n_row = QHBoxLayout()
        ref_n_row.addWidget(QLabel("N (spectra per z):"))
        self.ref_n_input = QSpinBox()
        self.ref_n_input.setRange(1, 1000)
        self.ref_n_input.setValue(5)
        ref_n_row.addWidget(self.ref_n_input)
        ref_layout.addLayout(ref_n_row)

        ref_range_row = QHBoxLayout()
        ref_range_row.addWidget(QLabel("Search range (um):"))
        self.ref_range_input = QDoubleSpinBox()
        self.ref_range_input.setRange(0.1, 1000.0)
        self.ref_range_input.setValue(10)
        self.ref_range_input.setDecimals(1)
        ref_range_row.addWidget(self.ref_range_input)
        ref_layout.addLayout(ref_range_row)

        ref_pts_row = QHBoxLayout()
        ref_pts_row.addWidget(QLabel("Search pts:"))
        self.ref_pts_input = QSpinBox()
        self.ref_pts_input.setRange(2, 500)
        self.ref_pts_input.setValue(20)
        ref_pts_row.addWidget(self.ref_pts_input)
        ref_layout.addLayout(ref_pts_row)

        self.ref_collect_btn = QPushButton("Collect reference spectra")
        self.ref_collect_btn.clicked.connect(self.collect_reference)
        ref_layout.addWidget(self.ref_collect_btn)

        ref_box.setLayout(ref_layout)
        outer.addWidget(ref_box)

        # ================= SPATIAL MAPPING SECTION =================
        scan_box = make_collapsible("Spatial mapping", expanded=False)
        scan_layout = QVBoxLayout()

        scan_layout.addWidget(QLabel(
            "Draw a rectangle in a Shapes layer first, then run."
        ))

        scan_name_row = QHBoxLayout()
        scan_name_row.addWidget(QLabel("File name:"))
        self.scan_name_input = QLineEdit()
        self.scan_name_input.setPlaceholderText("scan_label")
        scan_name_row.addWidget(self.scan_name_input)
        scan_layout.addLayout(scan_name_row)

        scan_exp_row = QHBoxLayout()
        scan_exp_row.addWidget(QLabel("Raman exposure (ms):"))
        self.scan_exp_input = QDoubleSpinBox()
        self.scan_exp_input.setRange(1, 1_000_000)
        self.scan_exp_input.setValue(1000)
        self.scan_exp_input.setDecimals(1)
        scan_exp_row.addWidget(self.scan_exp_input)
        scan_layout.addLayout(scan_exp_row)

        scan_n_row = QHBoxLayout()
        scan_n_row.addWidget(QLabel("N (grid side):"))
        self.scan_n_input = QSpinBox()
        self.scan_n_input.setRange(2, 500)
        self.scan_n_input.setValue(20)
        scan_n_row.addWidget(self.scan_n_input)
        scan_layout.addLayout(scan_n_row)

        scan_z_row = QHBoxLayout()
        scan_z_row.addWidget(QLabel("Z offset (um):"))
        self.scan_z_input = QDoubleSpinBox()
        self.scan_z_input.setRange(-1000, 1000)
        self.scan_z_input.setValue(4)
        self.scan_z_input.setDecimals(2)
        self.scan_z_input.setSingleStep(0.5)
        scan_z_row.addWidget(self.scan_z_input)
        scan_layout.addLayout(scan_z_row)

        # Z-scan option
        self.scan_zscan_check = QCheckBox("Z-scan (multi-plane Raman)")
        self.scan_zscan_check.setChecked(False)
        self.scan_zscan_check.toggled.connect(self._toggle_zscan_fields)
        scan_layout.addWidget(self.scan_zscan_check)

        self.scan_zrange_row = QHBoxLayout()
        self._zscan_range_label = QLabel("Z range (+/- um):")
        self.scan_zrange_row.addWidget(self._zscan_range_label)
        self.scan_zrange_input = QDoubleSpinBox()
        self.scan_zrange_input.setRange(0.1, 1000.0)
        self.scan_zrange_input.setValue(5.0)
        self.scan_zrange_input.setDecimals(1)
        self.scan_zrange_input.setSingleStep(0.5)
        self.scan_zrange_row.addWidget(self.scan_zrange_input)
        scan_layout.addLayout(self.scan_zrange_row)

        self.scan_zsteps_row = QHBoxLayout()
        self._zscan_steps_label = QLabel("Z steps:")
        self.scan_zsteps_row.addWidget(self._zscan_steps_label)
        self.scan_zsteps_input = QSpinBox()
        self.scan_zsteps_input.setRange(2, 500)
        self.scan_zsteps_input.setValue(10)
        self.scan_zsteps_row.addWidget(self.scan_zsteps_input)
        scan_layout.addLayout(self.scan_zsteps_row)

        # Hide z-scan fields initially
        self._zscan_range_label.setVisible(False)
        self.scan_zrange_input.setVisible(False)
        self._zscan_steps_label.setVisible(False)
        self.scan_zsteps_input.setVisible(False)

        scan_layout.addWidget(QLabel(
            "Extra channels (BF is always captured before & after):"
        ))
        self.channel_rows_layout = QVBoxLayout()
        scan_layout.addLayout(self.channel_rows_layout)
        self.channel_rows = []

        self.add_channel_btn = QPushButton("+ Add channel")
        self.add_channel_btn.clicked.connect(self._add_channel_row)
        scan_layout.addWidget(self.add_channel_btn)

        self.scan_btn = QPushButton("Run grid scan")
        self.scan_btn.clicked.connect(self.run_grid_scan)
        scan_layout.addWidget(self.scan_btn)

        scan_box.setLayout(scan_layout)
        outer.addWidget(scan_box)

        # ================= GENERATE STAGE GRID SECTION =================
        grid_box = make_collapsible("Generate stage grid", expanded=False)
        grid_layout = QVBoxLayout()

        grid_layout.addWidget(QLabel(
            "Creates a grid from the current center or two captured corners,\n"
            "each carrying the same single fixed point (non-batch)."
        ))

        definition_row = QHBoxLayout()
        definition_row.addWidget(QLabel("Grid definition:"))
        self.grid_definition_combo = QComboBox()
        self.grid_definition_combo.addItem("Centered on current XY", "center")
        self.grid_definition_combo.addItem(
            "Top-left + bottom-right", "corners"
        )
        definition_row.addWidget(self.grid_definition_combo)
        grid_layout.addLayout(definition_row)

        grid_af_row = QHBoxLayout()
        grid_af_row.addWidget(QLabel("Autofocus object:"))
        self.grid_af_combo = QComboBox()
        self.grid_af_combo.addItems(
            ["None", "laser", "software", "quartz", "glass", "cell"]
        )
        grid_af_row.addWidget(self.grid_af_combo)
        grid_layout.addLayout(grid_af_row)
        # let the grid combo also drive the MDA autofocus-field visibility
        self.grid_af_combo.currentTextChanged.connect(self._toggle_autofocus_fields)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Grid setup channel:"))
        self.grid_channel_combo = QComboBox()
        self.grid_channel_combo.addItem(
            "Raman (pre-scan)", (None, False)
        )
        self.grid_channel_combo.addItem(
            "Raman (compact, no pre-scan)", (None, True)
        )
        channel_row.addWidget(self.grid_channel_combo)
        grid_layout.addLayout(channel_row)

        # Fixed point in image (pixel) coordinates -- same point at every FOV.
        fovx_row = QHBoxLayout()
        fovx_row.addWidget(QLabel("FOV x (px):"))
        self.grid_fovx_input = QSpinBox()
        self.grid_fovx_input.setRange(0, 100000)
        self.grid_fovx_input.setValue(510)
        fovx_row.addWidget(self.grid_fovx_input)
        grid_layout.addLayout(fovx_row)

        fovy_row = QHBoxLayout()
        fovy_row.addWidget(QLabel("FOV y (px):"))
        self.grid_fovy_input = QSpinBox()
        self.grid_fovy_input.setRange(0, 100000)
        self.grid_fovy_input.setValue(510)
        fovy_row.addWidget(self.grid_fovy_input)
        grid_layout.addLayout(fovy_row)

        # Stage grid extent / spacing (real stage units).
        xr_row = QHBoxLayout()
        self._grid_xrange_label = QLabel("X range (+/- um):")
        xr_row.addWidget(self._grid_xrange_label)
        self.grid_xrange_input = QDoubleSpinBox()
        self.grid_xrange_input.setRange(0.0, 1_000_000)
        self.grid_xrange_input.setValue(100.0)
        self.grid_xrange_input.setDecimals(2)
        self.grid_xrange_input.setSingleStep(10.0)
        xr_row.addWidget(self.grid_xrange_input)
        grid_layout.addLayout(xr_row)

        yr_row = QHBoxLayout()
        self._grid_yrange_label = QLabel("Y range (+/- um):")
        yr_row.addWidget(self._grid_yrange_label)
        self.grid_yrange_input = QDoubleSpinBox()
        self.grid_yrange_input.setRange(0.0, 1_000_000)
        self.grid_yrange_input.setValue(100.0)
        self.grid_yrange_input.setDecimals(2)
        self.grid_yrange_input.setSingleStep(10.0)
        yr_row.addWidget(self.grid_yrange_input)
        grid_layout.addLayout(yr_row)

        self._grid_corner_help = QLabel(
            "Move to each corner and capture its XY, or type the coordinates."
        )
        grid_layout.addWidget(self._grid_corner_help)

        tl_row = QHBoxLayout()
        self._grid_tl_label = QLabel("Top-left (X, Y):")
        tl_row.addWidget(self._grid_tl_label)
        self.grid_tl_x_input = QLineEdit()
        self.grid_tl_x_input.setPlaceholderText("X")
        self.grid_tl_y_input = QLineEdit()
        self.grid_tl_y_input.setPlaceholderText("Y")
        self.grid_capture_tl_btn = QPushButton("Capture current XY")
        self.grid_capture_tl_btn.clicked.connect(
            lambda _checked=False: self._capture_grid_corner("top_left")
        )
        tl_row.addWidget(self.grid_tl_x_input)
        tl_row.addWidget(self.grid_tl_y_input)
        tl_row.addWidget(self.grid_capture_tl_btn)
        grid_layout.addLayout(tl_row)

        br_row = QHBoxLayout()
        self._grid_br_label = QLabel("Bottom-right (X, Y):")
        br_row.addWidget(self._grid_br_label)
        self.grid_br_x_input = QLineEdit()
        self.grid_br_x_input.setPlaceholderText("X")
        self.grid_br_y_input = QLineEdit()
        self.grid_br_y_input.setPlaceholderText("Y")
        self.grid_capture_br_btn = QPushButton("Capture current XY")
        self.grid_capture_br_btn.clicked.connect(
            lambda _checked=False: self._capture_grid_corner("bottom_right")
        )
        br_row.addWidget(self.grid_br_x_input)
        br_row.addWidget(self.grid_br_y_input)
        br_row.addWidget(self.grid_capture_br_btn)
        grid_layout.addLayout(br_row)

        self._grid_center_widgets = [
            self._grid_xrange_label, self.grid_xrange_input,
            self._grid_yrange_label, self.grid_yrange_input,
        ]
        self._grid_corner_widgets = [
            self._grid_corner_help,
            self._grid_tl_label, self.grid_tl_x_input, self.grid_tl_y_input,
            self.grid_capture_tl_btn,
            self._grid_br_label, self.grid_br_x_input, self.grid_br_y_input,
            self.grid_capture_br_btn,
        ]
        self.grid_definition_combo.currentIndexChanged.connect(
            self._toggle_grid_definition_fields
        )
        self._toggle_grid_definition_fields()

        sampling_row = QHBoxLayout()
        sampling_row.addWidget(QLabel("Grid sampling:"))
        self.grid_sampling_combo = QComboBox()
        self.grid_sampling_combo.addItem("Maximum spacing", "spacing")
        self.grid_sampling_combo.addItem(
            "Point count (including endpoints)", "count"
        )
        sampling_row.addWidget(self.grid_sampling_combo)
        grid_layout.addLayout(sampling_row)

        xs_row = QHBoxLayout()
        self._grid_xstep_label = QLabel("X max spacing (um):")
        xs_row.addWidget(self._grid_xstep_label)
        self.grid_xstep_input = QDoubleSpinBox()
        self.grid_xstep_input.setRange(0.01, 1_000_000)
        self.grid_xstep_input.setValue(50.0)
        self.grid_xstep_input.setDecimals(2)
        self.grid_xstep_input.setSingleStep(5.0)
        xs_row.addWidget(self.grid_xstep_input)
        grid_layout.addLayout(xs_row)

        ys_row = QHBoxLayout()
        self._grid_ystep_label = QLabel("Y max spacing (um):")
        ys_row.addWidget(self._grid_ystep_label)
        self.grid_ystep_input = QDoubleSpinBox()
        self.grid_ystep_input.setRange(0.01, 1_000_000)
        self.grid_ystep_input.setValue(50.0)
        self.grid_ystep_input.setDecimals(2)
        self.grid_ystep_input.setSingleStep(5.0)
        ys_row.addWidget(self.grid_ystep_input)
        grid_layout.addLayout(ys_row)

        xcount_row = QHBoxLayout()
        self._grid_xcount_label = QLabel("X points:")
        xcount_row.addWidget(self._grid_xcount_label)
        self.grid_xcount_input = QSpinBox()
        self.grid_xcount_input.setRange(1, 100000)
        self.grid_xcount_input.setValue(5)
        xcount_row.addWidget(self.grid_xcount_input)
        grid_layout.addLayout(xcount_row)

        ycount_row = QHBoxLayout()
        self._grid_ycount_label = QLabel("Y points:")
        ycount_row.addWidget(self._grid_ycount_label)
        self.grid_ycount_input = QSpinBox()
        self.grid_ycount_input.setRange(1, 100000)
        self.grid_ycount_input.setValue(5)
        ycount_row.addWidget(self.grid_ycount_input)
        grid_layout.addLayout(ycount_row)

        self._grid_spacing_widgets = [
            self._grid_xstep_label, self.grid_xstep_input,
            self._grid_ystep_label, self.grid_ystep_input,
        ]
        self._grid_count_widgets = [
            self._grid_xcount_label, self.grid_xcount_input,
            self._grid_ycount_label, self.grid_ycount_input,
        ]
        self.grid_sampling_combo.currentIndexChanged.connect(
            self._toggle_grid_sampling_fields
        )
        self._toggle_grid_sampling_fields()

        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Scan order:"))
        self.grid_scan_order_combo = QComboBox()
        self.grid_scan_order_combo.addItem("Raster", None)
        self.grid_scan_order_combo.addItem("Snake (X fast)", "x")
        self.grid_scan_order_combo.addItem("Snake (Y fast)", "y")
        order_row.addWidget(self.grid_scan_order_combo)
        grid_layout.addLayout(order_row)

        # Number of identical points placed per position (>=2 for the DAQ,
        # which needs at least 2 samples per channel).
        reps_row = QHBoxLayout()
        reps_row.addWidget(QLabel("Repeats (points per position, >=2):"))
        self.grid_repeats_input = QSpinBox()
        self.grid_repeats_input.setRange(2, 1000)
        self.grid_repeats_input.setValue(2)
        reps_row.addWidget(self.grid_repeats_input)
        grid_layout.addLayout(reps_row)

        preview_row = QHBoxLayout()
        self.grid_size_label = QLabel()
        self.grid_size_label.setStyleSheet("font-weight: bold;")
        self.grid_size_label.setWordWrap(True)
        self.grid_size_btn = QPushButton("Grid details...")
        self.grid_size_btn.clicked.connect(self._show_grid_size_preview)
        preview_row.addWidget(self.grid_size_label)
        preview_row.addWidget(self.grid_size_btn)
        grid_layout.addLayout(preview_row)

        self.grid_tilt_check = QCheckBox(
            "Correct sample tilt with a fitted Z surface"
        )
        self.grid_tilt_check.setChecked(False)
        grid_layout.addWidget(self.grid_tilt_check)

        self._grid_tilt_help = QLabel(
            "Move to a grid location, focus Z, then capture XYZ. Use at least "
            "3, 6, or 10 well-distributed points for degree 1, 2, or 3. Table "
            "values are editable. For centered grids, return to the intended "
            "center before generating."
        )
        self._grid_tilt_help.setWordWrap(True)
        grid_layout.addWidget(self._grid_tilt_help)

        tilt_degree_row = QHBoxLayout()
        self._grid_tilt_degree_label = QLabel("Vandermonde degree:")
        tilt_degree_row.addWidget(self._grid_tilt_degree_label)
        self.grid_tilt_degree_input = QSpinBox()
        self.grid_tilt_degree_input.setRange(1, 3)
        self.grid_tilt_degree_input.setValue(1)
        tilt_degree_row.addWidget(self.grid_tilt_degree_input)
        grid_layout.addLayout(tilt_degree_row)

        self.grid_tilt_table = QTableWidget(0, 3)
        self.grid_tilt_table.setHorizontalHeaderLabels(["X", "Y", "Focused Z"])
        self.grid_tilt_table.horizontalHeader().setStretchLastSection(True)
        self.grid_tilt_table.setMaximumHeight(150)
        grid_layout.addWidget(self.grid_tilt_table)

        tilt_buttons = QHBoxLayout()
        self.grid_tilt_capture_btn = QPushButton("Capture current XYZ")
        self.grid_tilt_add_btn = QPushButton("Add row")
        self.grid_tilt_remove_btn = QPushButton("Remove selected")
        self.grid_tilt_clear_btn = QPushButton("Clear")
        tilt_buttons.addWidget(self.grid_tilt_capture_btn)
        tilt_buttons.addWidget(self.grid_tilt_add_btn)
        tilt_buttons.addWidget(self.grid_tilt_remove_btn)
        tilt_buttons.addWidget(self.grid_tilt_clear_btn)
        grid_layout.addLayout(tilt_buttons)

        self.grid_tilt_fit_label = QLabel("Tilt fit: need at least 3 points")
        self.grid_tilt_fit_label.setWordWrap(True)
        grid_layout.addWidget(self.grid_tilt_fit_label)

        self._grid_tilt_widgets = [
            self._grid_tilt_help,
            self._grid_tilt_degree_label, self.grid_tilt_degree_input,
            self.grid_tilt_table,
            self.grid_tilt_capture_btn, self.grid_tilt_add_btn,
            self.grid_tilt_remove_btn, self.grid_tilt_clear_btn,
            self.grid_tilt_fit_label,
        ]
        self.grid_tilt_check.toggled.connect(self._toggle_grid_tilt_fields)
        self.grid_tilt_capture_btn.clicked.connect(
            self._capture_grid_tilt_point
        )
        self.grid_tilt_add_btn.clicked.connect(
            lambda _checked=False: self._add_grid_tilt_row()
        )
        self.grid_tilt_remove_btn.clicked.connect(
            self._remove_selected_grid_tilt_rows
        )
        self.grid_tilt_clear_btn.clicked.connect(
            self._clear_grid_tilt_rows
        )
        self.grid_tilt_table.itemChanged.connect(
            self._update_grid_tilt_fit_preview
        )
        self.grid_tilt_degree_input.valueChanged.connect(
            self._update_grid_tilt_fit_preview
        )
        self._toggle_grid_tilt_fields(False)

        for control in (
            self.grid_definition_combo,
            self.grid_sampling_combo,
            self.grid_scan_order_combo,
        ):
            control.currentIndexChanged.connect(self._update_grid_size_preview)
        for control in (
            self.grid_xrange_input, self.grid_yrange_input,
            self.grid_xstep_input, self.grid_ystep_input,
            self.grid_xcount_input, self.grid_ycount_input,
            self.grid_repeats_input,
        ):
            control.valueChanged.connect(self._update_grid_size_preview)
        for control in (
            self.grid_tl_x_input, self.grid_tl_y_input,
            self.grid_br_x_input, self.grid_br_y_input,
        ):
            control.textChanged.connect(self._update_grid_size_preview)
        self._update_grid_size_preview()

        self.run_grid_sel_btn = QPushButton("Generate grid")
        self.run_grid_sel_btn.clicked.connect(self.run_grid_selection)
        grid_layout.addWidget(self.run_grid_sel_btn)

        grid_box.setLayout(grid_layout)
        outer.addWidget(grid_box)

        # ================= AUTOMATED CELL SELECTION SECTION =================
        self.sel_box = make_collapsible("Automated cell selection", expanded=False)
        sel_box = self.sel_box
        sel_layout = QVBoxLayout()

        sel_layout.addWidget(QLabel("Mask region (shared by both buttons):"))

        cy_row = QHBoxLayout()
        cy_row.addWidget(QLabel("Center Y:"))
        self.sel_cy_input = QSpinBox()
        self.sel_cy_input.setRange(0, 100000)
        self.sel_cy_input.setValue(510)
        cy_row.addWidget(self.sel_cy_input)
        sel_layout.addLayout(cy_row)

        cx_row = QHBoxLayout()
        cx_row.addWidget(QLabel("Center X:"))
        self.sel_cx_input = QSpinBox()
        self.sel_cx_input.setRange(0, 100000)
        self.sel_cx_input.setValue(510)
        cx_row.addWidget(self.sel_cx_input)
        sel_layout.addLayout(cx_row)

        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Radius:"))
        self.sel_r_input = QSpinBox()
        self.sel_r_input.setRange(1, 100000)
        self.sel_r_input.setValue(100)
        r_row.addWidget(self.sel_r_input)
        sel_layout.addLayout(r_row)

        mask_btn_row = QHBoxLayout()
        self.add_mask_btn = QPushButton("Add mask")
        self.add_mask_btn.clicked.connect(self.add_mask)
        mask_btn_row.addWidget(self.add_mask_btn)
        sel_layout.addLayout(mask_btn_row)

        sel_layout.addWidget(QLabel("Automated point selection:"))

        af_row = QHBoxLayout()
        af_row.addWidget(QLabel("Autofocus object:"))
        self.sel_af_combo = QComboBox()
        self.sel_af_combo.addItems(
            ["laser", "software", "quartz", "glass", "cell", "None"]
        )
        af_row.addWidget(self.sel_af_combo)
        sel_layout.addLayout(af_row)

        npf_row = QHBoxLayout()
        npf_row.addWidget(QLabel("N per FOV:"))
        self.sel_npf_input = QSpinBox()
        self.sel_npf_input.setRange(1, 1000)
        self.sel_npf_input.setValue(6)
        npf_row.addWidget(self.sel_npf_input)
        sel_layout.addLayout(npf_row)

        # Center-cell mode: split each FOV into one new stage position per
        # detected cell, each shifted so that cell sits exactly at center.
        self.sel_center_cell_check = QCheckBox(
            "Center cell (split FOV into one position per cell, centered)"
        )
        self.sel_center_cell_check.setChecked(False)
        sel_layout.addWidget(self.sel_center_cell_check)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Aiming pattern:"))
        self.sel_shape_combo = QComboBox()
        self.sel_shape_combo.addItems(["Square", "Circle"])
        shape_row.addWidget(self.sel_shape_combo)
        sel_layout.addLayout(shape_row)

        sq_size_row = QHBoxLayout()
        sq_size_row.addWidget(QLabel("Pattern size (px):"))
        self.sel_sqsize_input = QDoubleSpinBox()
        self.sel_sqsize_input.setRange(0.0, 10000.0)
        self.sel_sqsize_input.setValue(30)
        self.sel_sqsize_input.setDecimals(1)
        self.sel_sqsize_input.setSingleStep(1.0)
        sq_size_row.addWidget(self.sel_sqsize_input)
        sel_layout.addLayout(sq_size_row)

        sq_n_row = QHBoxLayout()
        sq_n_row.addWidget(QLabel("N_x (subpoints):"))
        self.sel_sqn_input = QSpinBox()
        self.sel_sqn_input.setRange(1, 100)
        self.sel_sqn_input.setValue(1)
        sq_n_row.addWidget(self.sel_sqn_input)
        sel_layout.addLayout(sq_n_row)


        bkd_row = QHBoxLayout()
        bkd_row.addWidget(QLabel("Background distance (px):"))
        self.sel_bkd_input = QDoubleSpinBox()
        self.sel_bkd_input.setRange(0.0, 1_000_000.0)
        self.sel_bkd_input.setValue(80)
        self.sel_bkd_input.setDecimals(1)
        bkd_row.addWidget(self.sel_bkd_input)
        sel_layout.addLayout(bkd_row)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Integrated batch collection:"))
        self.sel_batch_combo = QComboBox()
        self.sel_batch_combo.addItems(["False", "True"])
        batch_row.addWidget(self.sel_batch_combo)
        sel_layout.addLayout(batch_row)

        sel_cp_row = QHBoxLayout()
        sel_cp_row.addWidget(QLabel("Cellpose model:"))
        self.sel_cellpose_combo = QComboBox()
        try:
            from cellpose import models as _cp_models
            _sel_model_names = list(_cp_models.MODEL_NAMES)
        except Exception:
            _sel_model_names = ["cyto2"]
        self.sel_cellpose_combo.addItems(_sel_model_names)
        if "cyto2" in _sel_model_names:
            self.sel_cellpose_combo.setCurrentText("cyto2")
        sel_cp_row.addWidget(self.sel_cellpose_combo)
        sel_layout.addLayout(sel_cp_row)

        self.run_selection_btn = QPushButton("Run automated selection")
        self.run_selection_btn.clicked.connect(self.run_automated_selection)
        sel_layout.addWidget(self.run_selection_btn)

        manual_row = QHBoxLayout()
        self.run_manual_btn = QPushButton("Manual selection")
        self.run_manual_btn.clicked.connect(self.run_manual_selection)
        self.center_manual_btn = QPushButton("Center clicked cells")
        self.center_manual_btn.clicked.connect(self.center_manual_cells)
        manual_row.addWidget(self.run_manual_btn)
        manual_row.addWidget(self.center_manual_btn)
        sel_layout.addLayout(manual_row)

        sel_box.setLayout(sel_layout)
        outer.addWidget(sel_box)

        # ================= RUN RAMAN MDA SECTION =================
        mda_box = make_collapsible("Run Raman MDA", expanded=False)
        mda_layout = QVBoxLayout()

        mda_layout.addWidget(QLabel(
            "Run automated cell selection first; this uses its sources "
            "& autofocus_p."
        ))

        mda_dir_row = QHBoxLayout()
        mda_dir_row.addWidget(QLabel("Writer output dir:"))
        self.mda_dir_input = QLineEdit()
        self.mda_dir_input.setText("data/run")
        mda_dir_row.addWidget(self.mda_dir_input)
        mda_layout.addLayout(mda_dir_row)

        afp_row = QHBoxLayout()
        afp_row.addWidget(QLabel("Autofocus positions (comma-sep):"))
        self.mda_afp_input = QLineEdit()
        self.mda_afp_input.setText("")
        self.mda_afp_input.setPlaceholderText("(blank = from selection)")
        afp_row.addWidget(self.mda_afp_input)
        mda_layout.addLayout(afp_row)

        imgp_row = QHBoxLayout()
        imgp_row.addWidget(QLabel("Imaging positions (comma-sep):"))
        self.mda_imgp_input = QLineEdit()
        self.mda_imgp_input.setText("")
        self.mda_imgp_input.setPlaceholderText("(blank = same as autofocus p)")
        imgp_row.addWidget(self.mda_imgp_input)
        mda_layout.addLayout(imgp_row)

        af_range_row = QHBoxLayout()
        self._af_range_label = QLabel("Autofocus search range:")
        af_range_row.addWidget(self._af_range_label)
        self.mda_af_range_input = QDoubleSpinBox()
        self.mda_af_range_input.setRange(0.1, 1000)
        self.mda_af_range_input.setValue(6)
        self.mda_af_range_input.setDecimals(2)
        self.mda_af_range_input.setSingleStep(0.5)
        af_range_row.addWidget(self.mda_af_range_input)
        mda_layout.addLayout(af_range_row)

        # Coarse autofocus search points (all autofocus objects)
        search_pts_row = QHBoxLayout()
        self._search_pts_label = QLabel("Autofocus search pts:")
        search_pts_row.addWidget(self._search_pts_label)
        self.mda_search_pts_input = QSpinBox()
        self.mda_search_pts_input.setRange(2, 500)
        self.mda_search_pts_input.setValue(8)
        search_pts_row.addWidget(self.mda_search_pts_input)
        mda_layout.addLayout(search_pts_row)

        # Laser autofocus FINE scan range
        fine_range_row = QHBoxLayout()
        self._fine_range_label = QLabel("Laser fine search range (+/- um):")
        fine_range_row.addWidget(self._fine_range_label)
        self.mda_fine_range_input = QDoubleSpinBox()
        self.mda_fine_range_input.setRange(0.1, 1000)
        self.mda_fine_range_input.setValue(1.5)
        self.mda_fine_range_input.setDecimals(2)
        self.mda_fine_range_input.setSingleStep(0.5)
        fine_range_row.addWidget(self.mda_fine_range_input)
        mda_layout.addLayout(fine_range_row)

        # Laser autofocus FINE scan points
        fine_pts_row = QHBoxLayout()
        self._fine_pts_label = QLabel("Laser fine search pts:")
        fine_pts_row.addWidget(self._fine_pts_label)
        self.mda_fine_pts_input = QSpinBox()
        self.mda_fine_pts_input.setRange(2, 500)
        self.mda_fine_pts_input.setValue(8)
        fine_pts_row.addWidget(self.mda_fine_pts_input)
        mda_layout.addLayout(fine_pts_row)

        # Show/hide autofocus fields based on the selected autofocus object.
        self.sel_af_combo.currentTextChanged.connect(self._toggle_autofocus_fields)
        self._toggle_autofocus_fields(self.sel_af_combo.currentText())

        # Segment-and-track toggle (independent of autofocus)
        self.mda_seg_track_check = QCheckBox(
            "Segment and track (update aiming)"
        )
        self.mda_seg_track_check.setChecked(False)
        self.mda_seg_track_check.toggled.connect(self._toggle_seg_track_fields)
        mda_layout.addWidget(self.mda_seg_track_check)
        # Seg-track options (shown only when the box is checked)
        seg_ch_row = QHBoxLayout()
        self._seg_ch_label = QLabel("Segment channel:")
        seg_ch_row.addWidget(self._seg_ch_label)
        self.mda_seg_ch_combo = QComboBox()
        self.mda_seg_ch_combo.addItem("BF")
        seg_ch_row.addWidget(self.mda_seg_ch_combo)
        mda_layout.addLayout(seg_ch_row)
        seg_scale_row = QHBoxLayout()
        self._seg_scale_label = QLabel("Image rescale factor:")
        seg_scale_row.addWidget(self._seg_scale_label)
        self.mda_seg_scale_input = QDoubleSpinBox()
        self.mda_seg_scale_input.setRange(0.1, 100.0)
        self.mda_seg_scale_input.setValue(2.0)
        self.mda_seg_scale_input.setDecimals(1)
        self.mda_seg_scale_input.setSingleStep(0.5)
        seg_scale_row.addWidget(self.mda_seg_scale_input)
        mda_layout.addLayout(seg_scale_row)
        
        # Cellpose model dropdown
        seg_model_row = QHBoxLayout()
        self._seg_model_label = QLabel("Cellpose model:")
        seg_model_row.addWidget(self._seg_model_label)
        self.mda_seg_model_combo = QComboBox()
        try:
            from cellpose import models as _cp_models
            model_names = list(_cp_models.MODEL_NAMES)
        except Exception:
            model_names = ["cyto2"]
        self.mda_seg_model_combo.addItems(model_names)
        if "cyto2" in model_names:
            self.mda_seg_model_combo.setCurrentText("cyto2")
        seg_model_row.addWidget(self.mda_seg_model_combo)
        mda_layout.addLayout(seg_model_row)
        # Crop-around-mask dropdown
        seg_crop_row = QHBoxLayout()
        self._seg_crop_label = QLabel("Crop image around mask:")
        seg_crop_row.addWidget(self._seg_crop_label)
        self.mda_seg_crop_combo = QComboBox()
        self.mda_seg_crop_combo.addItems(["True", "False"])
        seg_crop_row.addWidget(self.mda_seg_crop_combo)
        mda_layout.addLayout(seg_crop_row)
        # Tracking config file
        seg_track_cfg_row = QHBoxLayout()
        self._seg_track_cfg_label = QLabel("Tracking config (.json):")
        seg_track_cfg_row.addWidget(self._seg_track_cfg_label)
        self.mda_track_cfg_input = QLineEdit()
        self.mda_track_cfg_input.setText("particle_config.json")
        self.mda_track_cfg_input.setPlaceholderText("particle_config.json")
        self._seg_track_cfg_browse = QPushButton("...")
        self._seg_track_cfg_browse.setFixedWidth(30)
        self._seg_track_cfg_browse.clicked.connect(self.browse_tracking_cfg)
        seg_track_cfg_row.addWidget(self.mda_track_cfg_input)
        seg_track_cfg_row.addWidget(self._seg_track_cfg_browse)
        mda_layout.addLayout(seg_track_cfg_row)
        # hidden until seg-track is checked
        self._toggle_seg_track_fields(False)

        mda_exp_row = QHBoxLayout()
        mda_exp_row.addWidget(QLabel("Exposure per cell (ms):"))
        self.mda_exp_input = QDoubleSpinBox()
        self.mda_exp_input.setRange(1, 1_000_000)
        self.mda_exp_input.setValue(1000)
        self.mda_exp_input.setDecimals(1)
        mda_exp_row.addWidget(self.mda_exp_input)
        mda_layout.addLayout(mda_exp_row)

        loops_row = QHBoxLayout()
        loops_row.addWidget(QLabel("Loops (time points):"))
        self.mda_loops_input = QSpinBox()
        self.mda_loops_input.setRange(1, 1_000_000)
        self.mda_loops_input.setValue(100)
        loops_row.addWidget(self.mda_loops_input)
        mda_layout.addLayout(loops_row)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Interval (s):"))
        self.mda_interval_input = QDoubleSpinBox()
        self.mda_interval_input.setRange(0.0, 1_000_000)
        self.mda_interval_input.setValue(600)
        self.mda_interval_input.setDecimals(1)
        interval_row.addWidget(self.mda_interval_input)
        mda_layout.addLayout(interval_row)

        # Refocus / re-segment cadence: run autofocus AND segment-and-track
        # only every Nth timepoint (1 = every timepoint).
        refocus_row = QHBoxLayout()
        refocus_row.addWidget(QLabel("Refocus & segment every (timepoints):"))
        self.mda_refocus_input = QSpinBox()
        self.mda_refocus_input.setRange(1, 1_000_000)
        self.mda_refocus_input.setValue(1)
        refocus_row.addWidget(self.mda_refocus_input)
        mda_layout.addLayout(refocus_row)

        zrel_row = QHBoxLayout()
        zrel_row.addWidget(QLabel("Z relative (comma-sep um):"))
        self.mda_zrel_input = QLineEdit()
        self.mda_zrel_input.setText("0, 4")
        self.mda_zrel_input.setPlaceholderText("e.g. 0, 3.33")
        zrel_row.addWidget(self.mda_zrel_input)
        mda_layout.addLayout(zrel_row)

        rz_row = QHBoxLayout()
        rz_row.addWidget(QLabel("Raman z indices (None = off):"))
        self.mda_rz_input = QLineEdit()
        self.mda_rz_input.setText("0")
        self.mda_rz_input.setPlaceholderText("e.g. 0, 1 or None")
        rz_row.addWidget(self.mda_rz_input)
        mda_layout.addLayout(rz_row)

        mda_layout.addWidget(QLabel(
            "Acquisition channels (channel / exposure / Z offset):"
        ))
        self.mda_channel_rows_layout = QVBoxLayout()
        mda_layout.addLayout(self.mda_channel_rows_layout)

        self.mda_add_channel_btn = QPushButton("+ Add channel")
        self.mda_add_channel_btn.clicked.connect(
            lambda: self._add_mda_channel_row()
        )
        mda_layout.addWidget(self.mda_add_channel_btn)

        mda_btns_row = QHBoxLayout()
        self.run_mda_btn = QPushButton("Run Raman MDA")
        self.run_mda_btn.clicked.connect(self.run_raman_mda)
        self.stop_mda_btn = QPushButton("Stop")
        self.stop_mda_btn.clicked.connect(self.stop_raman_mda)
        mda_btns_row.addWidget(self.run_mda_btn, 3)
        mda_btns_row.addWidget(self.stop_mda_btn, 1)

        mda_layout.addLayout(mda_btns_row)

        # --- separator ---
        sep = QLabel("-" * 45)
        sep.setAlignment(Qt.AlignCenter)
        mda_layout.addWidget(sep)

        self.view_acquisition_btn = QPushButton("Open saved acquisition")
        self.view_acquisition_btn.clicked.connect(
            self.open_acquisition_viewer
        )
        mda_layout.addWidget(self.view_acquisition_btn)
 
        # --- pixel-to-stage calibration ---
        self.px2stage_check = QCheckBox("Pixel-to-stage calibration")
        self.px2stage_check.setChecked(False)
        self.px2stage_check.toggled.connect(self._toggle_px2stage_fields)
        mda_layout.addWidget(self.px2stage_check)

        self._px2s_help = QLabel(
            "Pick the same feature in each grid position, then fit a\n"
            "pixel->stage Vandermonde model. Uses stage XY from the\n"
            "dataset's useq_sequence attribute."
        )
        self._px2s_help.setWordWrap(True)
        mda_layout.addWidget(self._px2s_help)

        px2s_ds_row = QHBoxLayout()
        self._px2s_ds_label = QLabel("Dataset (.zarr):")
        px2s_ds_row.addWidget(self._px2s_ds_label)
        self.px2stage_ds_path = QLineEdit()
        self.px2stage_ds_path.setPlaceholderText("data/dataset/ds_run_7.zarr")
        self._px2s_ds_browse = QPushButton("...")
        self._px2s_ds_browse.setFixedWidth(30)
        self._px2s_ds_browse.clicked.connect(self.browse_px2stage_ds)
        px2s_ds_row.addWidget(self.px2stage_ds_path)
        px2s_ds_row.addWidget(self._px2s_ds_browse)
        mda_layout.addLayout(px2s_ds_row)

        px2s_deg_row = QHBoxLayout()
        self._px2s_deg_label = QLabel("Vandermonde degree:")
        px2s_deg_row.addWidget(self._px2s_deg_label)
        self.px2stage_degree_input = QSpinBox()
        self.px2stage_degree_input.setRange(1, 5)
        self.px2stage_degree_input.setValue(1)
        px2s_deg_row.addWidget(self.px2stage_degree_input)
        mda_layout.addLayout(px2s_deg_row)

        self.px2stage_pick_btn = QPushButton("Pick points...")
        self.px2stage_pick_btn.clicked.connect(self.open_pixel_stage_picker)
        mda_layout.addWidget(self.px2stage_pick_btn)

        px2s_name_row = QHBoxLayout()
        self._px2s_name_label = QLabel("Model file:")
        px2s_name_row.addWidget(self._px2s_name_label)
        self.px2stage_name_input = QLineEdit()
        self.px2stage_name_input.setText("vandermonde_model.json")
        px2s_name_row.addWidget(self.px2stage_name_input)
        mda_layout.addLayout(px2s_name_row)

        self.px2stage_save_btn = QPushButton("Fit && save model")
        self.px2stage_save_btn.clicked.connect(self.fit_and_save_pixel_stage)
        mda_layout.addWidget(self.px2stage_save_btn)

        # collect the px2stage widgets so they can be hidden as a group
        self._px2stage_widgets = [
            self._px2s_help,
            self._px2s_ds_label, self.px2stage_ds_path, self._px2s_ds_browse,
            self._px2s_deg_label, self.px2stage_degree_input,
            self.px2stage_pick_btn,
            self._px2s_name_label, self.px2stage_name_input,
            self.px2stage_save_btn,
        ]
        # hidden until the box is checked
        self._toggle_px2stage_fields(False)
 
        mda_box.setLayout(mda_layout)
        outer.addWidget(mda_box)

        # make_collapsible re-shows ALL descendants on expand, which clobbers
        # our conditional field hiding -- re-apply it after any box expands.
        for _box in (calib_box, scan_box, self.sel_box, mda_box):
            _box.toggled.connect(lambda checked: self._reapply_toggles())

        # from .chat_panel import ChatPanel
        # self.chat_panel = ChatPanel(self)
        # outer.addWidget(self.chat_panel)
        
        outer.addStretch()

        # # ================= LIVE STAGE POSITION =================
        # self.pos_label = QLabel("Stage:  X --  Y --")
        # self.pos_label.setStyleSheet(
        #     "QLabel { border-top: 1px solid palette(mid); padding: 4px; "
        #     "font-family: monospace; }"
        # )
        # outer.addWidget(self.pos_label)

        # ================= STATUS BAR (bottom) =================
        self.status = QLabel("Status: disconnected")
        self.status.setStyleSheet(
            "QLabel { border-top: 1px solid palette(mid); padding: 4px; }"
        )
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        # Wrap everything in a scroll area so the panel doesn't get cut off
        # when many sections are expanded.
        inner = QWidget()
        inner.setLayout(outer)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(scroll)

        # Keep references to pop-up windows so they don't get garbage collected.
        self._plot_windows = []
        # attach hover help text to every field
        apply_tooltips(self)
        self._napari_window = getattr(self.viewer.window, "_qt_window", None)
        if self._napari_window is not None:
            self._napari_window.installEventFilter(self)
        QApplication.instance().aboutToQuit.connect(
            self._disconnect_on_shutdown
        )
        # Temporarily disable the napari widget's live X/Y polling.
        # self._pos_timer = QTimer(self)
        # self._pos_timer.setInterval(500)  # ms
        # self._pos_timer.timeout.connect(self._update_position_label)
        # self._pos_timer.start()

    def open_user_manual(self, _link=None):
        pdf_path = (
            Path(__file__).resolve().parent
            / "resources"
            / "napari-raman-widget-manual.pdf"
        )

        if not pdf_path.exists():
            QMessageBox.warning(
                self,
                "Manual not found",
                f"The user manual could not be found:\n{pdf_path}",
            )
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path)))

    # -------- file pickers --------
    def browse_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Micro-Manager config", "",
            "Config files (*.cfg);;All files (*)",
        )
        if path:
            self.cfg_path.setText(path)

    def browse_tf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select transformer model", "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.tf_path.setText(path)

    def browse_out(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select output folder", ""
        )
        if path:
            self.out_path.setText(path)

    def browse_vandermonde(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Vandermonde model", "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.sel_vdm_path.setText(path)
            if self.core is not None:
                self.status.setText(
                    f"Status: Vandermonde path changed"
                    f"{self._load_vandermonde()}"
                )

    def browse_tracking_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select tracking config", "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.mda_track_cfg_input.setText(path)

    # -------- helpers --------
    def _get_image_xy(self):
        """Return (X_size, Y_size) from the camera via the core if connected,
        else from the first image layer in the viewer. Falls back to
        (1344, 1024) if neither is available."""
        if self.core is not None:
            try:
                return int(self.core.getImageWidth()), int(self.core.getImageHeight())
            except Exception:
                pass
        from napari.layers import Image
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                shape = layer.data.shape
                Y, X = shape[-2], shape[-1]
                return int(X), int(Y)
        return 1344, 1024
    
    def _set_objective_combo(
        self, objectives, current=None,
        placeholder="(no objective choices)",
    ):
        """Replace objective choices without moving hardware."""
        self.objective_combo.blockSignals(True)
        try:
            self.objective_combo.clear()
            if objectives:
                self.objective_combo.addItems(objectives)
                if current in objectives:
                    self.objective_combo.setCurrentText(current)
                self.objective_combo.setEnabled(True)
            else:
                self.objective_combo.addItem(placeholder)
                self.objective_combo.setEnabled(False)
        finally:
            self.objective_combo.blockSignals(False)

    def _refresh_objective_combo(self):
        """Display the current objective device state index."""
        if self.core is None:
            self._set_objective_combo(
                [], placeholder="(connect first)"
            )
            return None
        objective = self._current_objective()
        if objective is None:
            self._set_objective_combo(
                [], placeholder="(index unavailable)"
            )
            return None
        self._set_objective_combo([objective], current=objective)
        self.objective_combo.setEnabled(False)
        return objective

    def _current_objective(self):
        """Read the objective's numeric state without moving the turret."""
        if self.core is None:
            return None
        try:
            objective = str(int(self.core.getState("Objective")))
        except Exception as e:
            print(f"[objective index] {e}")
            return None

        if self.objective_combo.currentText() != objective:
            self._set_objective_combo([objective], current=objective)
            self.objective_combo.setEnabled(False)
        return objective

    def _load_vandermonde(self):
        """Load the model entry for the current objective state index."""
        path = self.sel_vdm_path.text().strip()
        if not path:
            self.vandermonde = None
            self.vandermonde_objective = None
            self.vandermonde_path = None
            return " (no vandermonde)"
        objective = self._current_objective()
        if not objective:
            self.vandermonde = None
            self.vandermonde_objective = None
            self.vandermonde_path = None
            return " (no objective selected)"
        try:
            from cns_control.vandermonde import load_vandermonde_model
            C, degree = load_vandermonde_model(
                path, objective=objective
            )
            self.vandermonde = (C, degree)
            self.vandermonde_objective = objective
            self.vandermonde_path = path
            return (
                f" (vandermonde {objective}, deg={degree} OK)"
            )
        except Exception as e:
            self.vandermonde = None
            self.vandermonde_objective = None
            self.vandermonde_path = None
            return f" (vandermonde load failed: {e})"

    def _ensure_current_vandermonde(self):
        """Reload whenever hardware and cached objective identities differ."""
        objective = self._current_objective()
        if not objective:
            self.vandermonde = None
            self.vandermonde_objective = None
            self.vandermonde_path = None
            return None
        if (
            self.vandermonde is None
            or self.vandermonde_objective != objective
            or self.vandermonde_path != self.sel_vdm_path.text().strip()
        ):
            self._load_vandermonde()
        return objective

    def _make_point_transformer(self, size_px, n):
        """Build the selected aiming transformer from a pixel size.

        Converts px -> normalized using image width. Square uses it as edge
        length, Circle as radius.
        """
        from raman_mda_engine.aiming.transformers import Square, Circle
        img_x, _ = self._get_image_xy()
        length = float(size_px) / float(img_x)
        n = max(1, int(n))
        if self.sel_shape_combo.currentText() == "Circle":
            return Circle(length, n)
        return Square(length, n)

    def _pt_to_volts(self, pt):
        X, Y = self._get_image_xy()
        pt = _spatial_yx(pt)
        return self.transformer.BF_to_volts(
            (pt.reshape(1, -1)) / np.array([Y, X]),
            max_volts=1.6,
        )

    def _aim_beam_at_pixel(self, x, y):
        """Move the galvos to one calibrated camera pixel and hold there."""
        if self.daq is None or self.transformer is None:
            raise RuntimeError("DAQ and transformer must be connected")

        volts = np.asarray(
            self._pt_to_volts(np.array([y, x], dtype=float)),
            dtype=float,
        )
        if volts.shape != (1, 2) or not np.all(np.isfinite(volts)):
            raise ValueError(
                "Transformer must return one finite X/Y voltage pair"
            )

        self.daq.galvo.stop()
        self.daq.galvo.write(
            np.ascontiguousarray(volts[0]),
            auto_start=True,
        )
        return volts[0]

    def _find_points_layer(self):
        """Return the active Points layer, or the most recent one."""
        active = self.viewer.layers.selection.active
        if isinstance(active, napari.layers.Points):
            return active
        return next(
            (
                layer
                for layer in reversed(self.viewer.layers)
                if isinstance(layer, napari.layers.Points)
            ),
            None,
        )

    def _parse_float_list(self, text, label="list"):
        """Parse a comma-separated string of floats."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{label} is empty")
        try:
            return [float(p) for p in parts]
        except ValueError:
            raise ValueError(
                f"{label} contains non-numeric entries: {text!r}"
            )

    def _parse_int_list(self, text, label="list"):
        """Parse a comma-separated string of ints."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{label} is empty")
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise ValueError(
                f"{label} contains non-integer entries: {text!r}"
            )

    def _update_position_label(self):
        """Poll the stage for X/Y and update the live readout label.

        Z is deliberately NOT read here: reading the focus device on a timer can
        contend with the engine while it drives Z during acquisition. X/Y live
        only. Never raises; transient failures keep the last good reading."""
        if self.core is None:
            self.pos_label.setText("Stage:  X --  Y --")
            return
        try:
            x = self.core.getXPosition()
            y = self.core.getYPosition()
            self.pos_label.setText(
                f"Stage:  X {x:9.2f}  Y {y:9.2f}"
            )
        except Exception:
            # transient failure (device busy, reload in progress) -- leave the
            # last good reading up rather than flickering an error.
            pass

    def _toggle_zscan_fields(self, checked):
        """Show/hide the z-scan range and steps fields."""
        self._zscan_range_label.setVisible(checked)
        self.scan_zrange_input.setVisible(checked)
        self._zscan_steps_label.setVisible(checked)
        self.scan_zsteps_input.setVisible(checked)

    def _toggle_seg_track_fields(self, checked):
        """Show/hide the segment-channel and rescale fields."""
        self._seg_ch_label.setVisible(checked)
        self.mda_seg_ch_combo.setVisible(checked)
        self._seg_scale_label.setVisible(checked)
        self.mda_seg_scale_input.setVisible(checked)
        self._seg_model_label.setVisible(checked)
        self.mda_seg_model_combo.setVisible(checked)
        self._seg_crop_label.setVisible(checked)
        self.mda_seg_crop_combo.setVisible(checked)
        self._seg_track_cfg_label.setVisible(checked)
        self.mda_track_cfg_input.setVisible(checked)
        self._seg_track_cfg_browse.setVisible(checked)

    def _toggle_px2stage_fields(self, checked):
        """Show/hide the pixel-to-stage calibration fields."""
        for w in self._px2stage_widgets:
            w.setVisible(checked)

    def _toggle_recal_fields(self, checked):
        """Show/hide the recalibration fields."""
        for w in self._recal_widgets:
            w.setVisible(checked)

    def _toggle_grid_definition_fields(self, _index=None):
        """Show only the controls used by the selected grid definition."""
        use_corners = self.grid_definition_combo.currentData() == "corners"
        for widget in self._grid_center_widgets:
            widget.setVisible(not use_corners)
        for widget in self._grid_corner_widgets:
            widget.setVisible(use_corners)

    def _toggle_grid_sampling_fields(self, _index=None):
        """Show spacing or point-count controls for the selected grid mode."""
        use_count = self.grid_sampling_combo.currentData() == "count"
        for widget in self._grid_spacing_widgets:
            widget.setVisible(not use_count)
        for widget in self._grid_count_widgets:
            widget.setVisible(use_count)

    def _grid_size_preview(self):
        """Calculate grid dimensions without moving hardware or running MDA."""
        if self.grid_definition_combo.currentData() == "corners":
            x1 = float(self.grid_tl_x_input.text())
            y1 = float(self.grid_tl_y_input.text())
            x2 = float(self.grid_br_x_input.text())
            y2 = float(self.grid_br_y_input.text())
            x_span, y_span = abs(x2 - x1), abs(y2 - y1)
        else:
            x_span = 2.0 * float(self.grid_xrange_input.value())
            y_span = 2.0 * float(self.grid_yrange_input.value())

        if self.grid_sampling_combo.currentData() == "count":
            nx = int(self.grid_xcount_input.value())
            ny = int(self.grid_ycount_input.value())
        else:
            x_step = float(self.grid_xstep_input.value())
            y_step = float(self.grid_ystep_input.value())
            nx = 1 if x_span == 0 else int(np.ceil(x_span / x_step)) + 1
            ny = 1 if y_span == 0 else int(np.ceil(y_span / y_step)) + 1

        x_spacing = 0.0 if nx == 1 else x_span / (nx - 1)
        y_spacing = 0.0 if ny == 1 else y_span / (ny - 1)
        positions = nx * ny
        repeats = int(self.grid_repeats_input.value())
        return nx, ny, positions, repeats, x_span, y_span, x_spacing, y_spacing

    def _update_grid_size_preview(self, _value=None):
        """Refresh the compact grid-size readout as controls change."""
        try:
            nx, ny, positions, repeats, *_rest = self._grid_size_preview()
            order = self.grid_scan_order_combo.currentText()
            self.grid_size_label.setText(
                f"Grid: {nx} x {ny} = {positions:,} positions; "
                f"{positions * repeats:,} acquisition points; {order}"
            )
        except ValueError:
            self.grid_size_label.setText(
                "Grid: enter or capture both corner positions"
            )

    def _show_grid_size_preview(self):
        """Open a quick summary of the currently defined grid."""
        try:
            (nx, ny, positions, repeats, x_span, y_span,
             x_spacing, y_spacing) = self._grid_size_preview()
        except ValueError:
            QMessageBox.warning(
                self, "Grid details",
                "Enter or capture both top-left and bottom-right positions."
            )
            return
        QMessageBox.information(
            self,
            "Grid details",
            f"Grid shape: {nx} x {ny}\n"
            f"Scan order: {self.grid_scan_order_combo.currentText()}\n"
            f"Stage positions: {positions:,}\n"
            f"Points per position: {repeats}\n"
            f"Total acquisition points: {positions * repeats:,}\n\n"
            f"X span: {x_span:.3f} um; spacing: {x_spacing:.3f} um\n"
            f"Y span: {y_span:.3f} um; spacing: {y_spacing:.3f} um",
        )

    def _toggle_grid_tilt_fields(self, checked):
        """Show or hide the grid tilt-reference workflow."""
        for widget in self._grid_tilt_widgets:
            widget.setVisible(checked)
        if checked:
            self._update_grid_tilt_fit_preview()

    def _add_grid_tilt_row(self, xyz=None):
        """Append an editable XYZ reference row."""
        row = self.grid_tilt_table.rowCount()
        self.grid_tilt_table.insertRow(row)
        values = ("", "", "") if xyz is None else xyz
        for column, value in enumerate(values):
            text = "" if value == "" else f"{float(value):.4f}"
            self.grid_tilt_table.setItem(row, column, QTableWidgetItem(text))
        self._update_grid_tilt_fit_preview()

    def _capture_grid_tilt_point(self):
        """Capture the current focused stage XYZ as a tilt reference."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        try:
            x, y = self.core.getXYPosition()
            z = self.core.getPosition()
            self._add_grid_tilt_row((x, y, z))
            self.status.setText(
                f"Status: captured tilt reference X {x:.3f}, "
                f"Y {y:.3f}, Z {z:.3f}"
            )
        except Exception as e:
            self.status.setText(f"Status: couldn't capture tilt point -- {e}")

    def _remove_selected_grid_tilt_rows(self):
        """Remove selected tilt-reference rows from bottom to top."""
        rows = {index.row() for index in self.grid_tilt_table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.grid_tilt_table.removeRow(row)
        self._update_grid_tilt_fit_preview()

    def _clear_grid_tilt_rows(self):
        """Remove every tilt reference and reset the fit readout."""
        self.grid_tilt_table.setRowCount(0)
        self._update_grid_tilt_fit_preview()

    def _grid_tilt_reference_points(self):
        """Read and validate the editable XYZ reference table."""
        points = []
        for row in range(self.grid_tilt_table.rowCount()):
            values = []
            for column in range(3):
                item = self.grid_tilt_table.item(row, column)
                if item is None or not item.text().strip():
                    raise ValueError(f"tilt reference row {row + 1} is incomplete")
                values.append(float(item.text()))
            points.append(values)
        return np.asarray(points, dtype=float)

    def _update_grid_tilt_fit_preview(self, _item=None):
        """Fit the current reference table and report degree-aware RMSE."""
        try:
            points = self._grid_tilt_reference_points()
            degree = int(self.grid_tilt_degree_input.value())
            required = (degree + 1) * (degree + 2) // 2
            if len(points) < required:
                self.grid_tilt_fit_label.setText(
                    f"Degree {degree} fit: need at least {required} points "
                    f"({len(points)} defined)"
                )
                return
            from cns_control.utils import _fit_tilt_surface
            _origin, _scale, _coefficients, rmse = _fit_tilt_surface(
                points, degree
            )
            self.grid_tilt_fit_label.setText(
                f"Degree {degree} tilt fit: RMSE {rmse:.4f} um"
            )
        except Exception as e:
            self.grid_tilt_fit_label.setText(f"Tilt fit: {e}")

    def _capture_grid_corner(self, corner):
        """Copy the current stage XY into one of the corner input pairs."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        try:
            x, y = self.core.getXYPosition()
            if corner == "top_left":
                x_input, y_input, name = (
                    self.grid_tl_x_input, self.grid_tl_y_input, "top-left"
                )
            else:
                x_input, y_input, name = (
                    self.grid_br_x_input, self.grid_br_y_input, "bottom-right"
                )
            x_input.setText(f"{x:.3f}")
            y_input.setText(f"{y:.3f}")
            self.status.setText(
                f"Status: captured {name} at X {x:.3f}, Y {y:.3f}"
            )
        except Exception as e:
            self.status.setText(f"Status: couldn't capture grid corner -- {e}")

    def _toggle_autofocus_fields(self, method):
        """Show/hide the MDA autofocus fields based on the chosen object.

        - None  : hide everything (no autofocus).
        - laser : show coarse range/pts AND the laser fine range/pts.
        - other : show coarse range/pts only (fine is laser-specific).
        """
        method = (method or "").lower()
        has_autofocus = method not in ("none", "")
        is_laser = method == "laser"

        # coarse fields: shown for any real autofocus object
        self._af_range_label.setVisible(has_autofocus)
        self.mda_af_range_input.setVisible(has_autofocus)
        self._search_pts_label.setVisible(has_autofocus)
        self.mda_search_pts_input.setVisible(has_autofocus)

        # fine fields: laser only
        self._fine_range_label.setVisible(is_laser)
        self.mda_fine_range_input.setVisible(is_laser)
        self._fine_pts_label.setVisible(is_laser)
        self.mda_fine_pts_input.setVisible(is_laser)

    def _reapply_toggles(self):
        self._toggle_seg_track_fields(self.mda_seg_track_check.isChecked())
        self._toggle_zscan_fields(self.scan_zscan_check.isChecked())
        self._toggle_autofocus_fields(self.sel_af_combo.currentText())
        self._toggle_px2stage_fields(self.px2stage_check.isChecked())
        self._toggle_recal_fields(self.recal_check.isChecked())
        self._toggle_grid_definition_fields()
        self._toggle_grid_sampling_fields()
        self._toggle_grid_tilt_fields(self.grid_tilt_check.isChecked())

    # -------- channel row helpers --------
    def _available_channels(self):
        """Query MM for available channels in the 'Channel' group, excluding BF."""
        if self.core is None:
            return []
        try:
            channels = list(self.core.getAvailableConfigs("Channel"))
            return [c for c in channels if c != "BF"]
        except Exception:
            return []

    def _add_channel_row(self, *, channel=None, exposure=500.0):
        """Append a new channel row to the spatial-mapping section."""
        row = QHBoxLayout()
        combo = QComboBox()
        combo.setEditable(False)
        available = self._available_channels()
        if available:
            combo.addItems(available)
            if channel and channel in available:
                combo.setCurrentText(channel)
        else:
            combo.addItem("(connect first)")
            combo.setEnabled(False)

        exp_spin = QDoubleSpinBox()
        exp_spin.setRange(1, 1_000_000)
        exp_spin.setValue(exposure)
        exp_spin.setDecimals(1)
        exp_spin.setSuffix(" ms")

        remove_btn = QPushButton("x")
        remove_btn.setFixedWidth(30)

        row.addWidget(combo, 2)
        row.addWidget(exp_spin, 1)
        row.addWidget(remove_btn)
        self.channel_rows_layout.addLayout(row)

        entry = {
            "row": row, "combo": combo, "exp": exp_spin, "remove": remove_btn,
        }
        self.channel_rows.append(entry)
        remove_btn.clicked.connect(lambda: self._remove_channel_row(entry))

    def _remove_channel_row(self, entry):
        """Remove a channel row from the layout and the bookkeeping list."""
        while entry["row"].count():
            item = entry["row"].takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.channel_rows_layout.removeItem(entry["row"])
        if entry in self.channel_rows:
            self.channel_rows.remove(entry)

    def _add_mda_channel_row(
        self, *, channel=None, exposure=10.0, z_offset=0.0
    ):
        """Append a channel row to the MDA section."""
        row = QHBoxLayout()
        combo = QComboBox()
        combo.setEditable(False)
        try:
            available_all = (
                list(self.core.getAvailableConfigs("Channel"))
                if self.core else []
            )
        except Exception:
            available_all = []
        if available_all:
            combo.addItems(available_all)
            if channel is None:
                used = {
                    entry["combo"].currentText()
                    for entry in self.mda_channel_rows
                }
                if "RM" in available_all and "RM" not in used:
                    channel = "RM"
                else:
                    channel = next(
                        (name for name in available_all if name not in used),
                        available_all[0],
                    )
            if channel and channel in available_all:
                combo.setCurrentText(channel)
        else:
            combo.addItem("(connect first)")
            combo.setEnabled(False)

        exp_spin = QDoubleSpinBox()
        exp_spin.setRange(1, 1_000_000)
        exp_spin.setValue(exposure)
        exp_spin.setDecimals(1)
        exp_spin.setSuffix(" ms")

        offset_spin = QDoubleSpinBox()
        offset_spin.setRange(-1000, 1000)
        offset_spin.setValue(z_offset)
        offset_spin.setDecimals(2)
        offset_spin.setSingleStep(0.1)
        offset_spin.setSuffix(" um")
        offset_spin.setToolTip(
            "Added directly to the focused Z position for this channel"
        )

        remove_btn = QPushButton("x")
        remove_btn.setFixedWidth(30)

        row.addWidget(combo, 2)
        row.addWidget(exp_spin, 1)
        row.addWidget(offset_spin, 1)
        row.addWidget(remove_btn)
        self.mda_channel_rows_layout.addLayout(row)

        entry = {
            "row": row,
            "combo": combo,
            "exp": exp_spin,
            "offset": offset_spin,
            "remove": remove_btn,
        }
        self.mda_channel_rows.append(entry)
        combo.currentTextChanged.connect(
            lambda text, spin=exp_spin: self._set_mda_channel_exposure_state(
                text, spin
            )
        )
        self._set_mda_channel_exposure_state(combo.currentText(), exp_spin)
        remove_btn.clicked.connect(lambda: self._remove_mda_channel_row(entry))

    @staticmethod
    def _set_mda_channel_exposure_state(channel, exposure_widget):
        """RM uses the dedicated Raman exposure setting, not camera exposure."""
        is_raman = channel == "RM"
        exposure_widget.setEnabled(not is_raman)
        exposure_widget.setToolTip(
            "Uses 'Exposure per cell' above"
            if is_raman else "Camera exposure for this channel"
        )

    def _remove_mda_channel_row(self, entry):
        while entry["row"].count():
            item = entry["row"].takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.mda_channel_rows_layout.removeItem(entry["row"])
        if entry in self.mda_channel_rows:
            self.mda_channel_rows.remove(entry)

    def _refresh_grid_channel_combo(self, available_channels):
        """Keep Raman first while refreshing available hardware channels."""
        current = self.grid_channel_combo.currentData()
        self.grid_channel_combo.blockSignals(True)
        self.grid_channel_combo.clear()
        self.grid_channel_combo.addItem(
            "Raman (pre-scan)", (None, False)
        )
        self.grid_channel_combo.addItem(
            "Raman (compact, no pre-scan)", (None, True)
        )
        for channel in available_channels:
            self.grid_channel_combo.addItem(channel, (channel, False))

        index = self.grid_channel_combo.findData(current)
        if index >= 0:
            self.grid_channel_combo.setCurrentIndex(index)
        self.grid_channel_combo.setEnabled(True)
        self.grid_channel_combo.blockSignals(False)


    def _refresh_channel_combos(self):
        """Repopulate every channel combo with the current MM channel list."""
        available_no_bf = self._available_channels()
        try:
            available_all = (
                list(self.core.getAvailableConfigs("Channel"))
                if self.core else []
            )
        except Exception:
            available_all = []

        self._refresh_grid_channel_combo(available_all)

        for entry in self.channel_rows:
            combo = entry["combo"]
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if available_no_bf:
                combo.addItems(available_no_bf)
                combo.setEnabled(True)
                if current in available_no_bf:
                    combo.setCurrentText(current)
            else:
                combo.addItem("(connect first)")
                combo.setEnabled(False)
            combo.blockSignals(False)

        for entry in self.mda_channel_rows:
            combo = entry["combo"]
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if available_all:
                combo.addItems(available_all)
                combo.setEnabled(True)
                if current in available_all:
                    combo.setCurrentText(current)
            else:
                combo.addItem("(connect first)")
                combo.setEnabled(False)
            combo.blockSignals(False)
            self._set_mda_channel_exposure_state(
                combo.currentText(), entry["exp"]
            )
        
        combo = self.mda_seg_ch_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if available_all:
            combo.addItems(available_all)
            combo.setCurrentText(current if current in available_all else "BF")
        else:
            combo.addItem("BF")
        combo.blockSignals(False)

    def _prepare_for_selection(self):
        """Stop live mode and reset Z/time plans while preserving axis order."""
        try:
            self.core.stopSequenceAcquisition()
        except Exception as e:
            print(f"[live mode stop] {e}")

        if self.main_window is None:
            print("[mda setup] no main_window -- can't configure MDA widget")
            return

        try:
            from raman_mda_engine.utils import get_mda_widget_from_napari

            mda_settings = get_mda_widget_from_napari(self.main_window)
        except Exception as e:
            print(f"[mda setup] couldn't locate MDA widget: {e}")
            return

        try:
            from useq import ZRangeAround, TIntervalLoops
            import datetime as _dt
            seq = mda_settings.value()
            new_seq = seq.replace(
                z_plan=ZRangeAround(range=0.0, step=1.0),
                time_plan=TIntervalLoops(
                    interval=_dt.timedelta(seconds=0), loops=1
                ),
            )
            if hasattr(mda_settings, "setValue"):
                mda_settings.setValue(new_seq)
                print(
                    "[mda setup] preserved "
                    f"axis_order={new_seq.axis_order}, z_plan, t=1loop OK"
                )
            else:
                print("[mda setup] no setValue method -- can't push sequence back")
                print(
                    "[mda setup] available: "
                    f"{[m for m in dir(mda_settings) if 'value' in m.lower() or 'set' in m.lower()][:10]}"
                )
        except Exception as e:
            print(f"[mda setup] failed to update sequence: {e}")

    # -------- dataset generation --------
    def open_acquisition_viewer(self):
        """Open the offline indexed viewer without querying hardware."""
        active_writer = getattr(self, "mda_writer", None)
        if active_writer is not None and not getattr(
            active_writer, "closed", True
        ):
            self.status.setText(
                "Status: finish or stop the active MDA before offline viewing"
            )
            return
        imaging_folder = self.mda_dir_input.text().strip()
        active_path = getattr(active_writer, "path", None)
        if active_path is not None:
            imaging_folder = str(active_path)
        imaging_path = Path(imaging_folder) if imaging_folder else None
        raman_folder = ""
        sequence_file = ""
        if imaging_path is not None:
            candidate = imaging_path / "raman.h5"
            if candidate.is_file():
                raman_folder = str(candidate)
            else:
                candidate = imaging_path / "raman"
                if candidate.is_dir():
                    raman_folder = str(candidate)
            candidate = imaging_path / "useq-sequence.json"
            if candidate.is_file():
                sequence_file = str(candidate)

        objective = self.objective_combo.currentText().strip()
        if not objective.isdigit():
            objective = self.vandermonde_objective

        window = LargeAcquisitionViewerWindow(
            imaging_folder=imaging_folder,
            raman_folder=raman_folder,
            sequence_file=sequence_file,
            vandermonde_model=self.sel_vdm_path.text().strip(),
            objective=objective,
            parent=self,
        )
        window.show()
        self._plot_windows.append(window)

    def _connect_mda_completion_events(self):
        """Connect completion handling once for the active MDA event source."""
        events = self.core.mda.events
        if events is self._mda_completion_events:
            return
        if self._mda_completion_events is not None:
            try:
                self._mda_completion_events.sequenceFinished.disconnect(
                    self._on_raman_mda_finished
                )
                self._mda_completion_events.sequenceCanceled.disconnect(
                    self._on_raman_mda_canceled
                )
            except Exception:
                pass
        events.sequenceCanceled.connect(self._on_raman_mda_canceled)
        events.sequenceFinished.connect(self._on_raman_mda_finished)
        self._mda_completion_events = events

    def _connect_raman_visualization(self, engine):
        """Connect one active Raman engine to the lazy napari spectrum view."""
        if engine is self._raman_visualization_engine:
            return
        if self._raman_visualization_engine is not None:
            try:
                self._raman_visualization_engine.raman_events.ramanSpectraReady.disconnect(
                    self._on_raman_spectra_ready
                )
            except Exception:
                pass
        engine.raman_events.ramanSpectraReady.connect(
            self._on_raman_spectra_ready
        )
        self._raman_visualization_engine = engine

    @Slot(object, object, object, object, object)
    def _on_raman_spectra_ready(
        self, event, spectra, points, which, _exposure
    ):
        """Add the spectrum image at RM's index in the shared MDA stack."""
        if self._lazy_mda_viewer is None:
            return
        try:
            wavenumbers = self.collector.get_wavenumbers(
                np.asarray(spectra).shape[-1]
            )
        except Exception:
            wavenumbers = None
        self._lazy_mda_viewer.add_raman_spectrum(
            event,
            spectra,
            wavenumbers,
            points=points,
            which=which,
        )

    def _on_raman_mda_canceled(self, _sequence):
        """Remember cancellation before the ensuing sequenceFinished signal."""
        if self._raman_mda_pending:
            self._raman_mda_canceled = True

    def _on_raman_mda_finished(self, _sequence):
        """Close the writer and report how the Raman MDA finished."""
        if not self._raman_mda_pending:
            return

        writer = self._raman_mda_writer
        self._raman_mda_pending = False
        self._raman_mda_writer = None

        reason = "canceled" if self._raman_mda_canceled else "completed"
        self._raman_mda_canceled = False
        status = getattr(self.core.mda, "status", None)
        if callable(status):
            finish_reason = getattr(status(), "finish_reason", None)
            if finish_reason is not None:
                reason = str(getattr(finish_reason, "value", finish_reason))
        reason_text = reason.casefold()
        if "cancel" in reason_text:
            reason = "canceled"
        elif "fail" in reason_text or "error" in reason_text:
            reason = "failed"
        else:
            reason = "completed"

        if writer is not None:
            writer.close(status=reason)
            writer_status = getattr(writer, "completion_status", reason)
            if writer_status != "completed":
                reason = writer_status
        if reason != "completed":
            self.status.setText(f"Status: MDA {reason}")
            return

        self.status.setText("Status: MDA finished OK")

    
    def browse_px2stage_ds(self):
        # zarr stores are directories
        path = QFileDialog.getExistingDirectory(
            self, "Select dataset (.zarr)", "data/dataset"
        )
        if path:
            self.px2stage_ds_path.setText(path)
 
    def open_pixel_stage_picker(self):
        """Open the frame-by-frame point picker on the selected dataset."""
        import json as _json
        path = self.px2stage_ds_path.text().strip()
        if not path:
            self.status.setText("Status: select a dataset (.zarr) first")
            return
        objective = self._current_objective()
        if not objective:
            self.status.setText(
                "Status: objective state index unavailable -- connect first"
            )
            return
        try:
            ds = xr.open_zarr(path)
            if "useq_sequence" not in ds.attrs:
                self.status.setText(
                    "Status: dataset has no useq_sequence attr -- "
                    "regenerate it with the current loader"
                )
                return
            seq = _json.loads(ds.attrs["useq_sequence"])
            self.px2stage_xy = np.array(
                [[s["x"], s["y"]] for s in seq["stage_positions"]]
            )
            # one frame per position: first t / c / z
            imgs = ds["image"].isel(t=0, c=0, z=0).values
            if len(imgs) != len(self.px2stage_xy):
                print(
                    f"[px2stage] warning: {len(imgs)} frames vs "
                    f"{len(self.px2stage_xy)} stage positions"
                )
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
            from cns_control.calibration import StagePointPicker
            plt.ion()
            self.px2stage_picker = StagePointPicker(imgs)
            self.px2stage_objective = objective
            plt.show()
            self.status.setText(
                f"Status: picker open ({len(imgs)} frames, "
                f"objective {objective}) -- click through, then Fit & save"
            )
        except Exception as e:
            self.status.setText(f"Status: picker failed -- {e}")
 
    def fit_and_save_pixel_stage(self):
        """Fit the centered Vandermonde model on picked points and save it."""
        if self.px2stage_picker is None or self.px2stage_xy is None:
            self.status.setText("Status: no picked points -- pick points first")
            return
        objective = self.px2stage_objective
        current_objective = self._current_objective()
        if not objective or current_objective != objective:
            self.status.setText(
                "Status: objective index changed after points were opened -- "
                "restore the calibration objective and Pick points again"
            )
            return
        log = LogWindow(title="Pixel-to-stage fit log")
        log.show()
        self._plot_windows.append(log)
        try:
            from cns_control.calibration import (
                apply_vandermonde, fit_vandermonde, save_vandermonde_model,
            )
            points = np.asarray(self.px2stage_picker.points, dtype=float)
            xy = self.px2stage_xy
            n = min(len(points), len(xy))
            points, xy = points[:n], xy[:n]
            valid = ~np.isnan(points).any(axis=1)
            degree = int(self.px2stage_degree_input.value())
            n_terms = (degree + 1) * (degree + 2) // 2
            if valid.sum() < n_terms:
                self.status.setText(
                    f"Status: need >= {n_terms} points for degree {degree}, "
                    f"got {valid.sum()}"
                )
                return
            with _StdoutRedirector(log):
                img_center = points[valid].mean(axis=0)
                xy_center = xy[valid].mean(axis=0)
                points_c = points[valid] - img_center
                xy_c = xy[valid] - xy_center
                print(f"Fitting on {valid.sum()}/{n} points")
                print(f"img_center={img_center}, xy_center={xy_center}")
                # RMSE comparison across degrees (offset domain)
                for deg in (1, 2, 3):
                    if valid.sum() < (deg + 1) * (deg + 2) // 2:
                        print(f"degree={deg}  (not enough points)")
                        continue
                    C_deg = fit_vandermonde(points_c, xy_c, deg)
                    res = xy_c - apply_vandermonde(points_c, C_deg, deg)
                    rmse = np.sqrt(np.mean(res**2, axis=0))
                    print(
                        f"degree={deg}  RMSE: x={rmse[0]:.4f}, y={rmse[1]:.4f}"
                    )
                C = fit_vandermonde(points_c, xy_c, degree)
                # sanity check: stage step for a 5 px x-offset
                test = apply_vandermonde(np.array([[5.0, 0.0]]), C, degree)[0]
                print(f"5px-x step -> stage (degree={degree}): {test}")
            # save dialog so the user can rename / choose location
            default_name = (
                self.px2stage_name_input.text().strip()
                or "vandermonde_model.json"
            )
            if not default_name.lower().endswith(".json"):
                default_name += ".json"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Vandermonde model", default_name,
                "JSON files (*.json);;All files (*)",
            )
            if not save_path:
                self.status.setText("Status: save cancelled")
                return
            save_vandermonde_model(
                save_path, C, degree,
                img_center=img_center, xy_center=xy_center,
                objective=objective,
            )
            self.px2stage_name_input.setText(save_path)
            # make it immediately usable by center-cell mode
            self.sel_vdm_path.setText(save_path)
            vdm_msg = self._load_vandermonde()
            log.append(f"\n--- saved {save_path} ---\n")
            self.status.setText(
                f"Status: Vandermonde model ({objective}, degree={degree}) "
                f"saved -> {save_path}{vdm_msg}"
            )
        except Exception as e:
            log.append(f"\n--- fit failed: {e} ---\n")
            self.status.setText(f"Status: pixel-to-stage fit failed -- {e}")

    # -------- loading actions --------
    def connect(self):
        self.status.setText("Status: connecting...")
        self.repaint()

        out = self.out_path.text().strip()
        if out:
            try:
                os.makedirs(out, exist_ok=True)
                os.chdir(out)
                print(f"[cwd] changed to {os.getcwd()}")
            except Exception as e:
                self.status.setText(
                    f"Status: couldn't cd to output folder -- {e}"
                )
                return

        try:
            from pymmcore_plus import CMMCorePlus
            from raman_control.princeton import (
                SpectraCollector as PrincetonSpectraCollector,
            )
            from cns_control.coordtransformer import CoordTransformer

            self.core = CMMCorePlus.instance()

            try:
                self.core.unloadAllDevices()
                time.sleep(0.5)
            except Exception:
                pass

            cfg = self.cfg_path.text().strip()
            if cfg:
                self.core.loadSystemConfiguration(cfg)
                self.mm_config = cfg
                try:
                    self.core.setConfig("Channel", "GFP")
                    time.sleep(1)
                    self.core.setConfig("Channel", "BF")
                except Exception as e:
                    print(f"[channel warm-up] {e}")

            try:
                # pymmcore-widgets uses the Qt 6 locations for these classes.
                # Expose their Qt 5 locations before napari imports the plugin.
                from qtpy import QtGui, QtWidgets

                for name in (
                    "QAction",
                    "QActionGroup",
                    "QUndoCommand",
                    "QUndoStack",
                ):
                    if not hasattr(QtGui, name):
                        setattr(QtGui, name, getattr(QtWidgets, name))

                _enable_raman_mda_options()
                result = self.viewer.window.add_plugin_dock_widget(
                    "napari-micromanager"
                )
                if isinstance(result, tuple) and len(result) >= 2:
                    self.main_window = result[1]
                    self._lazy_mda_viewer = install_lazy_mda_viewer(
                        self.main_window, self.core, self.viewer
                    )
            except Exception as e:
                print(f"[napari-micromanager load] {e}")

            lightfield_config = self.lightfield_config.text().strip()
            if not lightfield_config:
                lightfield_config = DEFAULT_LIGHTFIELD_CONFIG
            self.collector = PrincetonSpectraCollector(
                lightFieldConfig=lightfield_config
            )
            self.daq = self.collector.daq
            self.default_engine = self.core.mda.engine

            tf = self.tf_path.text().strip()
            if tf:
                self.transformer = CoordTransformer.from_json(tf)

            objective = self._refresh_objective_combo()
            vdm_msg = self._load_vandermonde()

            self._refresh_channel_combos()
            msg = "Status: connected OK (Princeton/LightField)"
            if not cfg:
                msg += " (no cfg loaded)"
            if not tf:
                msg += " (no transformer)"
            else:
                center_x, center_y = DEFAULT_BEAM_CENTER_XY
                try:
                    self._aim_beam_at_pixel(center_x, center_y)
                    msg += (
                        f" (beam centered at "
                        f"{center_x:.0f}, {center_y:.0f})"
                    )
                except Exception as e:
                    print(f"[beam centering] {e}")
                    msg += f" (beam centering failed: {e})"
            if not objective:
                msg += " (objective index unavailable)"
            msg += vdm_msg
            self.status.setText(msg)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.reload_tf_btn.setEnabled(True)
        except Exception as e:
            self.status.setText(f"Status: failed -- {e}")

    def reload_transformer(self):
        tf = self.tf_path.text().strip()
        if not tf:
            self.status.setText("Status: no transformer path set")
            return
        try:
            from cns_control.coordtransformer import CoordTransformer
            self.transformer = CoordTransformer.from_json(tf)
            vdm_msg = self._load_vandermonde()
            self.status.setText(f"Status: transformer reloaded OK{vdm_msg}")
        except Exception as e:
            self.status.setText(f"Status: transformer reload failed -- {e}")

    def disconnect(self):
        for button in (
            self.click_center_btn,
            self.click_laser_btn,
            self.drag_stage_btn,
        ):
            if button.isChecked():
                button.setChecked(False)
        self._stop_stage_drag()
        writer = self._raman_mda_writer or self.mda_writer
        if writer is not None and not getattr(writer, "closed", True):
            writer.disconnect()
        self._raman_mda_writer = None
        self._raman_mda_pending = False
        if self._raman_visualization_engine is not None:
            try:
                self._raman_visualization_engine.raman_events.ramanSpectraReady.disconnect(
                    self._on_raman_spectra_ready
                )
            except Exception:
                pass
            self._raman_visualization_engine = None
        try:
            if self.core is not None:
                from cns_control.utils import unload
                unload(self.core)
        except Exception as e:
            print(f"unload error: {e}")
        self.core = None

        try:
            if self.collector is not None:
                self.collector.close()
        except Exception as e:
            print(f"collector close error: {e}")
        self.collector = None
        self.daq = None
        self.transformer = None
        self.default_engine = None
        self.calibration_ds = None
        self.calibrator = None
        self.selector = None
        self.scan_ds = None
        self.main_window = None
        self.selection_results = None
        self.mda_writer = None
        self.px2stage_picker = None
        self.px2stage_xy = None
        self.px2stage_objective = None
        self.vandermonde = None
        self.vandermonde_objective = None
        self.vandermonde_path = None
        self._lazy_mda_viewer = None
        self._set_objective_combo([], placeholder="(connect first)")
        self.status.setText("Status: disconnected")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.reload_tf_btn.setEnabled(False)

    def _disconnect_on_shutdown(self):
        if self.core is not None or self.collector is not None:
            self.disconnect()

    def eventFilter(self, watched, event):
        if (
            watched is self._napari_window
            and event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_Escape
            and any(
                button.isChecked()
                for button in (
                    self.click_center_btn,
                    self.click_laser_btn,
                    self.drag_stage_btn,
                )
            )
        ):
            self.click_center_btn.setChecked(False)
            self.click_laser_btn.setChecked(False)
            self.drag_stage_btn.setChecked(False)
            self.status.setText("Status: viewer hardware control disabled")
            return True
        if (
            watched is self._napari_window
            and event.type() == QEvent.Close
            and (self.core is not None or self.collector is not None)
        ):
            reply = QMessageBox.question(
                self,
                "Hardware connected",
                "Hardware is still connected. Close napari and disconnect it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return True
            self.disconnect()
            return False
        return super().eventFilter(watched, event)

    # -------- raman collection --------
    def collect_raman(self):
        if self.collector is None or self.daq is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return

        points_layer = self._find_points_layer()
        if points_layer is None:
            self.status.setText("Status: no Points layer available")
            return

        points = np.asarray(points_layer.data)
        if len(points) == 0:
            self.status.setText(
                f"Status: Points layer '{points_layer.name}' is empty"
            )
            return

        exposure = float(self.exposure_input.value())
        N = int(self.n_input.value())

        # Keep acquisition separate from saving and plotting so a display
        # problem can never be mislabeled as a hardware collection failure.
        try:
            self.daq.galvo.stop()
            self.daq.galvo.start()

            pt = _spatial_yx(points[-1])
            volts = self._pt_to_volts(pt)
            spec = self.collector.collect_spectra_pts(
                np.tile(volts[0], (N, 1)), exposure
            )
            wavenumbers = self.collector.get_wavenumbers(spec.shape[-1])
        except Exception as e:
            self.status.setText(f"Status: collection failed -- {e}")
            return

        save_name = self.collect_save_input.text().strip()
        saved_msg = ""
        save_error = None
        if save_name:
            try:
                if not save_name.lower().endswith(".npy"):
                    save_name += ".npy"
                np.save(save_name, spec)
                axis_name = str(Path(save_name).with_suffix("")) + (
                    "_wavenumbers.npy"
                )
                np.save(axis_name, wavenumbers)
                saved_msg = f" -> {save_name}"
                print(f"Saved spectrum to {save_name}")
            except Exception as e:
                save_error = e

        plot_error = None
        try:
            win = SpectrumWindow(
                spec, wavenumbers=wavenumbers, title="Raman spectra"
            )
            win.show()
            self._plot_windows.append(win)
        except Exception as e:
            plot_error = e

        if save_error is not None or plot_error is not None:
            problems = []
            if save_error is not None:
                problems.append(f"save failed: {save_error}")
            if plot_error is not None:
                problems.append(f"plot failed: {plot_error}")
            self.status.setText(
                f"Status: collected {N}x{exposure:.0f}ms OK; "
                + "; ".join(problems)
            )
            return

        self.status.setText(
            f"Status: collected {N}x{exposure:.0f}ms OK{saved_msg}"
        )

    # -------- laser aiming calibration --------
    def run_calibration(self):
        if self.core is None or self.daq is None or self.collector is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return

        N = int(self.cal_n_input.value())
        exp = float(self.cal_exp_input.value())
        max_volts = float(self.cal_volts_input.value())
        grid = int(self.cal_grid_input.value())
        thres = float(self.cal_thres_input.value())

        log = LogWindow(title="Calibration log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: calibrating...")
        self.repaint()

        try:
            from cns_control.calibration import Calibrator

            self.calibrator = Calibrator(
                self.core, self.daq, self.transformer, self.collector,
                N=N, exp=exp, max_volts=max_volts,
            )
            with _StdoutRedirector(log):
                self.calibration_ds = self.calibrator.calibrate(
                    grid, thres=thres, plot=False
                )

            log.append("\n--- calibration complete ---\n")

            plot_win = CalibrationPlotWindow(
                self.calibration_ds, title="Calibration result"
            )
            plot_win.show()
            self._plot_windows.append(plot_win)

            self.status.setText("Status: calibration done OK")
        except Exception as e:
            log.append(f"\n--- calibration failed: {e} ---\n")
            self.status.setText(f"Status: calibration failed -- {e}")

    # -------- recalibration --------
    def open_selector(self):
        if self.calibration_ds is None:
            self.status.setText(
                "Status: no calibration dataset -- run calibration first"
            )
            return
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
            from cns_control.calibration import ManualImageSelector

            plt.ion()
            self.selector = ManualImageSelector(self.calibration_ds)
            plt.show()
            self.status.setText(
                "Status: selector open -- click through, then save"
            )
        except Exception as e:
            self.status.setText(f"Status: selector failed -- {e}")

    def save_recalibration(self):
        if self.selector is None:
            self.status.setText("Status: no selector -- open it first")
            return
        if self.calibrator is None:
            self.status.setText(
                "Status: no calibrator -- run calibration first"
            )
            return
        if self.calibration_ds is None:
            self.status.setText("Status: no calibration dataset")
            return

        model_name = self.model_name_input.text().strip()
        if not model_name:
            self.status.setText("Status: enter a model name")
            return

        try:
            from cns_control.coordtransformer import CoordTransformer

            selected_points = self.selector.selected_points
            self.calibrator.save_new_model(
                self.calibration_ds, selected_points, model_name
            )
            self.transformer = CoordTransformer.from_json(f"{model_name}.json")
            self.tf_path.setText(f"{model_name}.json")
            self.status.setText(f"Status: saved & loaded {model_name}.json OK")
        except Exception as e:
            self.status.setText(f"Status: save failed -- {e}")

    # -------- collect reference spectra --------
    def collect_reference(self):
        if self.core is None or self.daq is None or self.collector is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return
        points_layer = self._find_points_layer()
        if points_layer is None:
            self.status.setText("Status: no Points layer available")
            return
        points = np.asarray(points_layer.data)
        if len(points) == 0:
            self.status.setText(
                f"Status: Points layer '{points_layer.name}' is empty"
            )
            return

        name = self.ref_name_input.text().strip()
        if not name:
            self.status.setText("Status: enter a name for the reference")
            return

        exp = float(self.ref_exp_input.value())
        N = int(self.ref_n_input.value())
        search_range = float(self.ref_range_input.value())
        search_pts = int(self.ref_pts_input.value())

        self.status.setText("Status: collecting reference spectra...")
        self.repaint()

        log = LogWindow(title="Reference collection log")
        log.show()
        self._plot_windows.append(log)

        try:
            from cns_control.autofocus import autofocus_w_bkd

            pt = _spatial_yx(points[-1])
            volts = self._pt_to_volts(pt)
            volts_tiled = np.array([volts[0] for _ in range(N)])

            with _StdoutRedirector(log):
                focusZ, coarse_raman, all_raman = autofocus_w_bkd(
                    self.core, self.daq, self.collector, volts_tiled,
                    search_range=search_range,
                    search_pts=search_pts,
                    exposure=exp,
                )
            self.core.setZPosition(focusZ)

            zs = np.linspace(-search_range, search_range, search_pts)
            wavenumbers = self.collector.get_wavenumbers(
                all_raman.shape[-1]
            )

            win = ReferenceSpectraWindow(
                all_raman, zs, wavenumbers=wavenumbers,
                title=f"Reference spectra: {name}",
            )
            win.show()
            self._plot_windows.append(win)

            os.makedirs("reference", exist_ok=True)
            uid = str(uuid.uuid1())[:8]

            ds = xr.Dataset(
                {
                    "spec": (["z", "n", "wavenumber"], all_raman),
                },
                coords={
                    "z":        ("z",  zs),
                    "wavenumber": ("wavenumber", wavenumbers),
                    "x":        pt[1],
                    "y":        pt[0],
                    "exposure": exp,
                },
            )
            ds["wavenumber"].attrs.update(
                units="cm^-1", long_name="Raman shift"
            )

            zarr_path = f"reference/{name}_{uid}.zarr"
            ds.to_zarr(zarr_path)

            log.append(f"\n--- saved to {zarr_path} ---\n")
            self.status.setText(f"Status: reference saved ({zarr_path}) OK")

        except Exception as e:
            log.append(f"\n--- reference collection failed: {e} ---\n")
            self.status.setText(
                f"Status: reference collection failed -- {e}"
            )

    # -------- spatial mapping --------
    def run_grid_scan(self):
        if self.core is None or self.daq is None or self.collector is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return
        if len(self.viewer.layers) == 0:
            self.status.setText("Status: no layer to read shape from")
            return

        file_name = self.scan_name_input.text().strip()
        if not file_name:
            self.status.setText("Status: enter a file name")
            return

        exp = float(self.scan_exp_input.value())
        N = int(self.scan_n_input.value())
        z_offset = float(self.scan_z_input.value())
        do_zscan = self.scan_zscan_check.isChecked()

        if do_zscan:
            z_half = float(self.scan_zrange_input.value())
            z_steps = int(self.scan_zsteps_input.value())
            z_range = np.linspace(-z_half, z_half, z_steps)
        else:
            z_range = np.array([0.0])

        extra_channels = []
        seen = set()
        for entry in self.channel_rows:
            if not entry["combo"].isEnabled():
                continue
            ch = entry["combo"].currentText()
            if not ch or ch in seen:
                continue
            seen.add(ch)
            extra_channels.append((ch, float(entry["exp"].value())))

        log = LogWindow(title="Grid scan log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: grid scanning...")
        self.repaint()

        try:
            import xarray as xr
            from datetime import datetime

            with _StdoutRedirector(log):
                shapes = self.viewer.layers[-1]
                try:
                    shape0 = shapes.data[0]
                except Exception:
                    raise RuntimeError(
                        "Last layer has no shape data -- "
                        "draw a rectangle first."
                    )

                x_min = float(np.min(shape0[:, 0]))
                x_max = float(np.max(shape0[:, 0]))
                y_min = float(np.min(shape0[:, 1]))
                y_max = float(np.max(shape0[:, 1]))
                x = np.linspace(x_min, x_max, N)
                y = np.linspace(y_min, y_max, N)
                Xg, Yg = np.meshgrid(x, y)
                grid = np.column_stack([Xg.ravel(), Yg.ravel()])

                print(f"Grid: {N}x{N} = {grid.shape[0]} points")
                if do_zscan:
                    print(
                        f"Z-scan: {len(z_range)} planes, "
                        f"range [{z_range[0]:.1f}, {z_range[-1]:.1f}] um"
                    )

                self.core.setConfig("Channel", "BF")
                self.core.setExposure(10)
                BF = self.core.snap()

                extra_imgs = {}
                for ch, ch_exp in extra_channels:
                    print(f"Snapping {ch} at {ch_exp:.0f} ms")
                    self.core.setConfig("Channel", ch)
                    self.core.setExposure(ch_exp)
                    extra_imgs[ch] = self.core.snap()

                self.daq.galvo.stop()
                self.daq.galvo.start()
                currentz = self.core.getPosition()
                base_z = currentz - z_offset
                self.core.setConfig("Channel", "RM")
                # self.core.setShutterOpen("Fluoshutter", True)

                X_img, Y_img = self._get_image_xy()
                volts = self.transformer.BF_to_volts(
                    grid / np.array([Y_img, X_img]), max_volts=1.6
                )
                self.core.stopSequenceAcquisition()
                self.core.setExposure(1)

                all_specs = []
                all_BF_z = []
                for i, dz in enumerate(z_range):
                    self.core.setPosition(base_z + dz)
                    print(
                        f"  z-plane {i+1}/{len(z_range)}: "
                        f"dz={dz:+.2f} um, collecting "
                        f"{grid.shape[0]} spectra..."
                    )
                    specs = self.collector.collect_spectra_pts(volts, exp)
                    all_specs.append(specs)

                    if do_zscan:
                        # self.core.setShutterOpen("Fluoshutter", False)
                        self.core.setConfig("Channel", "BF")
                        self.core.setExposure(10)
                        BF_z = self.core.snap()
                        all_BF_z.append(BF_z)
                        self.core.setConfig("Channel", "RM")
                        # self.core.setShutterOpen("Fluoshutter", True)
                        self.core.setExposure(1)

                # self.core.setShutterOpen("Fluoshutter", False)
                self.core.setPosition(currentz)
                self.core.setConfig("Channel", "BF")
                self.core.setExposure(10)
                end_BF = self.core.snap()
                wavenumbers = self.collector.get_wavenumbers(
                    all_specs[0].shape[-1]
                )

                if do_zscan:
                    specs_stack = np.stack(all_specs, axis=0)
                    BF_stack = np.stack(all_BF_z, axis=0)
                    data_vars = {
                        "laser_pos": xr.DataArray(
                            volts, dims=("idx", "volt")
                        ),
                        "grid_pos": xr.DataArray(
                            grid, dims=("idx", "xy")
                        ),
                        "specs": xr.DataArray(
                            specs_stack, dims=("z", "idx", "wavenumber")
                        ),
                        "z_range": xr.DataArray(
                            z_range, dims=("z",)
                        ),
                        "BF": xr.DataArray(BF, dims=("Y", "X")),
                        "BF_z": xr.DataArray(
                            BF_stack, dims=("z", "Y", "X")
                        ),
                        "end_BF": xr.DataArray(end_BF, dims=("Y", "X")),
                    }
                    for ch, img in extra_imgs.items():
                        data_vars[ch] = xr.DataArray(img, dims=("Y", "X"))

                    ds = xr.Dataset(
                        data_vars, coords={"wavenumber": wavenumbers}
                    )
                    ds.attrs["time"] = str(datetime.now())
                    ds.attrs["raman_exposure_ms"] = exp
                    ds.attrs["z_offset"] = z_offset
                    ds.attrs["z_range_min"] = float(z_range[0])
                    ds.attrs["z_range_max"] = float(z_range[-1])
                    ds.attrs["z_steps"] = len(z_range)
                    ds.attrs["channel_exposures_ms"] = {
                        ch: ch_exp for ch, ch_exp in extra_channels
                    }

                    uid = uuid.uuid4().hex[:8]
                    zarr_name = f"grid_scan_z_{file_name}_{uid}.zarr"
                else:
                    data_vars = {
                        "laser_pos": xr.DataArray(
                            volts, dims=("idx", "volt")
                        ),
                        "grid_pos": xr.DataArray(
                            grid, dims=("idx", "volt")
                        ),
                        "specs": xr.DataArray(
                            all_specs[0], dims=("N", "wavenumber")
                        ),
                        "BF": xr.DataArray(BF, dims=("Y", "X")),
                        "end_BF": xr.DataArray(end_BF, dims=("Y", "X")),
                    }
                    for ch, img in extra_imgs.items():
                        data_vars[ch] = xr.DataArray(img, dims=("Y", "X"))

                    ds = xr.Dataset(
                        data_vars, coords={"wavenumber": wavenumbers}
                    )
                    ds.attrs["time"] = str(datetime.now())
                    ds.attrs["raman_exposure_ms"] = exp
                    ds.attrs["channel_exposures_ms"] = {
                        ch: ch_exp for ch, ch_exp in extra_channels
                    }

                    uid = uuid.uuid4().hex[:8]
                    zarr_name = f"grid_scan_data_{file_name}_{uid}.zarr"

                ds["wavenumber"].attrs.update(
                    units="cm^-1", long_name="Raman shift"
                )
                ds.to_zarr(zarr_name)
                print(f"Saved grid scan to {zarr_name}")

            self.scan_ds = ds

            win = GridScanPlotWindow(ds, title=f"Grid scan: {file_name}")
            win.show()
            self._plot_windows.append(win)

            self.status.setText(f"Status: grid scan saved -> {zarr_name}")
        except Exception as e:
            log.append(f"\n--- grid scan failed: {e} ---\n")
            self.status.setText(f"Status: grid scan failed -- {e}")

    # -------- automated cell selection --------
    def add_mask(self):
        """Add the masked overlay to the viewer using current center/radius."""
        try:
            from cns_control.utils import add_mask_with_hole
        except Exception as e:
            self.status.setText(f"Status: import failed -- {e}")
            return

        X, Y = self._get_image_xy()
        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        r = int(self.sel_r_input.value())

        try:
            add_mask_with_hole(
                self.viewer,
                image_size=(Y, X),
                circle_center=(cy, cx),
                circle_radius=r,
                small_circle_radius=10,
                color=(255, 0, 0),
                alpha=60,
                small_circle_color=(0, 255, 0),
                small_circle_alpha=255,
            )
            self.status.setText(
                f"Status: mask added at ({cx},{cy}) r={r} OK"
            )
        except Exception as e:
            self.status.setText(f"Status: add_mask failed -- {e}")

    def run_automated_selection(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.default_engine is None:
            self.status.setText("Status: no default engine -- reconnect")
            return

        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        r = int(self.sel_r_input.value())
        autofocus_object = self.sel_af_combo.currentText()
        N_per_fov = int(self.sel_npf_input.value())
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        bkd_thres = float(self.sel_bkd_input.value())
        batch = self.sel_batch_combo.currentText() == "True"

        center_cell = self.sel_center_cell_check.isChecked()
        vandermonde_model_path = self.sel_vdm_path.text().strip()
        cellpose_model = self.sel_cellpose_combo.currentText() or "cyto2"
        objective = self._current_objective() if center_cell else None

        if center_cell and not vandermonde_model_path:
            self.status.setText(
                "Status: Center cell mode requires a Vandermonde model (.json)"
            )
            return
        if center_cell and not objective:
            self.status.setText(
                "Status: Center cell mode requires an active objective"
            )
            return

        log = LogWindow(title="Automated selection log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: running automated selection...")
        self.repaint()

        try:
            from cns_control.utils import automated_point_selections

            with _StdoutRedirector(log):
                self._prepare_for_selection()
                self.core.register_mda_engine(self.default_engine)
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                sources, autofocus_p, new_seq = automated_point_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    N=N_per_fov + 1,
                    center=(cy, cx),
                    radius=r,
                    autofocus_object=autofocus_object,
                    bkd_thres=bkd_thres,
                    batch=batch,
                    center_cell=center_cell,
                    vandermonde_model_path=(
                        vandermonde_model_path if center_cell else None
                    ),
                    cellpose_model=cellpose_model,
                    objective=objective,
                )

            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
            }
            n_new = len(new_seq.stage_positions)
            log.append("\n--- selection complete ---\n")
            extra = (
                f" ({n_new} centered positions)" if center_cell else ""
            )
            self.status.setText(f"Status: automated selection done OK{extra}")
        except Exception as e:
            log.append(f"\n--- selection failed: {e} ---\n")
            self.status.setText(f"Status: selection failed -- {e}")


    def _toggle_click_to_center(self, checked):
        """Arm/disarm persistent click-to-center mode."""
        if checked:
            if self.click_laser_btn.isChecked():
                self.click_laser_btn.setChecked(False)
            if self.drag_stage_btn.isChecked():
                self.drag_stage_btn.setChecked(False)
            if self._click_center_cb not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self._click_center_cb)
            self.status.setText(
                "Status: click-to-center ARMED -- click a spot in the image"
            )
        else:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._click_center_cb)
            except ValueError:
                pass

    def _click_center_cb(self, viewer, event):
        """napari mouse callback: fires on press while armed."""
        if event.button != 1:          # left click only
            return
        yx = np.array(event.position[-2:], dtype=float)
        self._move_clicked_to_center(yx)

    def _toggle_click_to_laser(self, checked):
        """Arm/disarm one-shot click-to-point-laser mode."""
        if checked:
            if self.click_center_btn.isChecked():
                self.click_center_btn.setChecked(False)
            if self.drag_stage_btn.isChecked():
                self.drag_stage_btn.setChecked(False)
            if self._click_laser_cb not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self._click_laser_cb)
            self.status.setText(
                "Status: laser pointing ARMED -- click a spot in the image"
            )
        else:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._click_laser_cb)
            except ValueError:
                pass

    def _click_laser_cb(self, viewer, event):
        """napari mouse callback: aim the laser at one left-clicked pixel."""
        if event.button != 1:
            return
        yx = np.array(event.position[-2:], dtype=float)
        self.click_laser_btn.setChecked(False)
        try:
            volts = self._aim_beam_at_pixel(yx[1], yx[0])
            self.status.setText(
                f"Status: laser pointed at ({yx[0]:.0f},{yx[1]:.0f}) "
                f"[X={volts[0]:.3f}, Y={volts[1]:.3f} V]"
            )
        except Exception as e:
            self.status.setText(f"Status: laser pointing failed -- {e}")

    def _toggle_stage_drag(self, checked):
        """Enable or disable click-and-hold stage joystick mode."""
        if checked:
            if self.core is None:
                self.drag_stage_btn.setChecked(False)
                self.status.setText("Status: not connected")
                return
            if self.core.mda.is_running():
                self.drag_stage_btn.setChecked(False)
                self.status.setText(
                    "Status: stage drag unavailable while MDA is running"
                )
                return
            objective = self._ensure_current_vandermonde()
            if not objective or self.vandermonde is None:
                self.drag_stage_btn.setChecked(False)
                self.status.setText(
                    "Status: stage drag needs a Vandermonde calibration"
                )
                return
            if self.click_center_btn.isChecked():
                self.click_center_btn.setChecked(False)
            if self.click_laser_btn.isChecked():
                self.click_laser_btn.setChecked(False)
            if self._stage_drag_cb not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self._stage_drag_cb)
            self.status.setText(
                "Status: stage drag ARMED -- hold and drag the image"
            )
        else:
            self._stop_stage_drag()
            try:
                self.viewer.mouse_drag_callbacks.remove(self._stage_drag_cb)
            except ValueError:
                pass

    def _stage_drag_cb(self, viewer, event):
        """Use a held left-button drag as a velocity joystick."""
        if event.button != 1:
            return
        if hasattr(event, "handled"):
            event.handled = True
        self._stage_drag_anchor_yx = np.asarray(
            event.position[-2:], dtype=float
        )
        self._stage_drag_current_yx = self._stage_drag_anchor_yx.copy()
        self._stage_drag_active = True
        self._stage_drag_timer.start()
        try:
            yield
            while event.type == "mouse_move":
                self._stage_drag_current_yx = np.asarray(
                    event.position[-2:], dtype=float
                )
                if hasattr(event, "handled"):
                    event.handled = True
                yield
        finally:
            self._stop_stage_drag()
            if self.drag_stage_btn.isChecked():
                self.status.setText(
                    "Status: stage drag ARMED -- hold and drag the image"
                )

    def _stop_stage_drag(self):
        """Stop issuing stage moves for the current drag gesture."""
        self._stage_drag_timer.stop()
        self._stage_drag_active = False
        self._stage_drag_anchor_yx = None
        self._stage_drag_current_yx = None

    def _stage_drag_tick(self):
        """Issue one bounded relative move for the current drag direction."""
        if (
            not self._stage_drag_active
            or self._stage_drag_anchor_yx is None
            or self._stage_drag_current_yx is None
        ):
            return
        if self.core is None:
            self.drag_stage_btn.setChecked(False)
            self.status.setText("Status: stage drag stopped -- disconnected")
            return
        if self.core.mda.is_running():
            self.drag_stage_btn.setChecked(False)
            self.status.setText("Status: stage drag stopped -- MDA is running")
            return

        offset_yx = (
            self._stage_drag_current_yx - self._stage_drag_anchor_yx
        )
        distance_px = float(np.linalg.norm(offset_yx))
        if distance_px <= STAGE_DRAG_DEAD_ZONE_PX:
            return

        try:
            xy_stage = self.core.getXYStageDevice()
            if not xy_stage:
                raise RuntimeError("no XY stage is configured")
            if self.core.deviceBusy(xy_stage):
                return
            if self.vandermonde is None:
                raise RuntimeError("no Vandermonde calibration is loaded")

            from cns_control.utils import apply_vandermonde_model

            coefficients, degree = self.vandermonde
            offset_xy = offset_yx[::-1]
            stage_at_offset = np.asarray(
                apply_vandermonde_model(
                    offset_xy, coefficients, degree
                ),
                dtype=float,
            )
            stage_at_origin = np.asarray(
                apply_vandermonde_model(
                    np.zeros(2), coefficients, degree
                ),
                dtype=float,
            )
            stage_direction = stage_at_offset - stage_at_origin
            direction_norm = float(np.linalg.norm(stage_direction))
            if direction_norm == 0 or not np.isfinite(direction_norm):
                raise ValueError(
                    "Vandermonde calibration produced no finite direction"
                )

            speed_fraction = min(
                1.0,
                (distance_px - STAGE_DRAG_DEAD_ZONE_PX)
                / (
                    STAGE_DRAG_FULL_SPEED_PX
                    - STAGE_DRAG_DEAD_ZONE_PX
                ),
            )
            step_um = (
                float(self.stage_drag_speed_input.value())
                * speed_fraction
                * STAGE_DRAG_INTERVAL_MS
                / 1000.0
            )
            delta_xy = stage_direction / direction_norm * step_um
            self.core.setRelativeXYPosition(
                float(delta_xy[0]), float(delta_xy[1])
            )
            self.status.setText(
                "Status: dragging image "
                f"[dX={delta_xy[0]:+.2f}, dY={delta_xy[1]:+.2f} um]"
            )
        except Exception as e:
            self.drag_stage_btn.setChecked(False)
            self.status.setText(f"Status: stage drag failed -- {e}")

    def _set_laser_shutter(self, open_):
        """Control the shutter through the RM/imaging channel configs."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        try:
            if open_:
                current = self.core.getCurrentConfig("Channel")
                if current and current != "RM":
                    self._shutter_return_channel = current
                target = "RM"
            else:
                target = self._shutter_return_channel or "BF"
                if target == "RM":
                    target = "BF"
            self.core.setConfig("Channel", target)
            self.core.waitForConfig("Channel", target)
            state = "open (RM)" if open_ else f"closed ({target})"
            self.status.setText(f"Status: laser shutter {state}")
        except Exception as e:
            self.status.setText(f"Status: laser shutter failed -- {e}")

    def _set_nd_filter(self, open_):
        """Control the autofocus ND filter through Micro-Manager DigitalIO."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        try:
            if ND_FILTER_DEVICE not in self.core.getLoadedDevices():
                raise LookupError(
                    f"{ND_FILTER_DEVICE!r} is not loaded"
                )
            current = int(self.core.getState(ND_FILTER_DEVICE))
            desired = (
                current & ~ND_FILTER_MASK
                if open_
                else current | ND_FILTER_MASK
            )
            if desired != current:
                self.core.setState(ND_FILTER_DEVICE, desired)
                self.core.waitForDevice(ND_FILTER_DEVICE)
            state = "open (removed)" if open_ else "closed (inserted)"
            self.status.setText(f"Status: ND filter {state}")
        except Exception as e:
            self.status.setText(f"Status: ND filter failed -- {e}")

    def _move_clicked_to_center(self, yx):
        """Move the stage so the clicked pixel lands at (cy, cx)."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        objective = self._ensure_current_vandermonde()
        if not objective:
            self.status.setText(
                "Status: objective state index unavailable"
            )
            return
        if self.vandermonde is None:
            self.status.setText(
                f"Status: no Vandermonde calibration loaded for {objective}"
            )
            return
        try:
            from cns_control.utils import apply_vandermonde_model
            C, degree = self.vandermonde
            cy = int(self.sel_cy_input.value())
            cx = int(self.sel_cx_input.value())
            offset_yx = yx - np.array([cy, cx], dtype=float)
            offset_xy = np.array([offset_yx[1], offset_yx[0]])
            stage_dx, stage_dy = apply_vandermonde_model(offset_xy, C, degree)
            x, y = self.core.getXYPosition()
            self.core.setXYPosition(float(x - stage_dx), float(y - stage_dy))
            self.core.waitForSystem()
            self.status.setText(
                f"Status: moved ({yx[0]:.0f},{yx[1]:.0f}) -> center "
                f"({cy},{cx})  [dX={-stage_dx:.2f}, dY={-stage_dy:.2f} um]"
            )
        except Exception as e:
            self.status.setText(f"Status: click-to-center failed -- {e}")

    def run_manual_selection(self):
        """Create empty point-source layers for hand-clicking, mirroring the
        automated selection's (sources, autofocus_p, new_seq) contract.

        N cells-per-FOV is fixed up front by the 'N per FOV' spinbox. In batch
        mode positions are repeated N times and you click N cells per FOV; in
        non-batch mode you click freely."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.default_engine is None:
            self.status.setText("Status: no default engine -- reconnect")
            return

        autofocus_object = self.sel_af_combo.currentText()
        N_per_fov = int(self.sel_npf_input.value())
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        batch = self.sel_batch_combo.currentText() == "True"

        log = LogWindow(title="Manual selection log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: setting up manual selection...")
        self.repaint()

        try:
            from cns_control.utils import manual_point_selections

            with _StdoutRedirector(log):
                self._prepare_for_selection()
                self.core.register_mda_engine(self.default_engine)
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                sources, autofocus_p, new_seq = manual_point_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    N=N_per_fov,
                    autofocus_object=autofocus_object,
                    batch=batch,
                )

            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
            }
            hint = (
                f"click {N_per_fov} cell(s) per FOV"
                if batch else "click points freely"
            )
            log.append("\n--- manual layers ready ---\n")
            self.status.setText(
                f"Status: manual selection ready ({len(sources)} layer(s)) -- "
                f"{hint}, then Run Raman MDA"
            )
        except Exception as e:
            log.append(f"\n--- manual selection failed: {e} ---\n")
            self.status.setText(f"Status: manual selection failed -- {e}")

    def center_manual_cells(self):
        """Convert hand-clicked cells (from a non-batch manual selection)
        into one centered stage position per cell via the Vandermonde model."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.selection_results is None:
            self.status.setText(
                "Status: run Manual selection and click cells first"
            )
            return
        if self.sel_batch_combo.currentText() == "True":
            self.status.setText(
                "Status: centering requires a NON-batch manual selection"
            )
            return
        vandermonde_model_path = self.sel_vdm_path.text().strip()
        if not vandermonde_model_path:
            self.status.setText(
                "Status: set a Vandermonde model (.json) in Loading first"
            )
            return
        objective = self._current_objective()
        if not objective:
            self.status.setText(
                "Status: objective state index unavailable"
            )
            return
        autofocus_object = self.sel_af_combo.currentText()
        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        log = LogWindow(title="Center clicked cells log")
        log.show()
        self._plot_windows.append(log)
        self.status.setText("Status: centering clicked cells...")
        self.repaint()
        try:
            from cns_control.utils import center_manual_selections
            with _StdoutRedirector(log):
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                sources, autofocus_p, new_seq = center_manual_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    sources=self.selection_results["sources"],
                    vandermonde_model_path=vandermonde_model_path,
                    autofocus_object=autofocus_object,
                    center=(cy, cx),
                    objective=objective,
                )
            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
                "autofocus_object": autofocus_object,
                "batch": False,
            }
            n_new = len(new_seq.stage_positions)
            log.append("\n--- centering complete ---\n")
            self.status.setText(
                f"Status: centered {n_new} cell position(s) -- "
                "then Run Raman MDA"
            )
        except Exception as e:
            log.append(f"\n--- centering failed: {e} ---\n")
            self.status.setText(f"Status: centering failed -- {e}")

    def run_grid_selection(self):
        """Build a centered or corner-defined grid of stage positions."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.default_engine is None:
            self.status.setText("Status: no default engine -- reconnect")
            return

        fov_x = int(self.grid_fovx_input.value())
        fov_y = int(self.grid_fovy_input.value())
        x_range = float(self.grid_xrange_input.value())
        y_range = float(self.grid_yrange_input.value())
        x_step = float(self.grid_xstep_input.value())
        y_step = float(self.grid_ystep_input.value())
        x_count = y_count = None
        if self.grid_sampling_combo.currentData() == "count":
            x_count = int(self.grid_xcount_input.value())
            y_count = int(self.grid_ycount_input.value())
        repeats = int(self.grid_repeats_input.value())
        preview_channel, use_placeholder = self.grid_channel_combo.currentData()
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        autofocus_object = self.grid_af_combo.currentText()
        snake_axis = self.grid_scan_order_combo.currentData()
        tilt_degree = int(self.grid_tilt_degree_input.value())
        tilt_reference_points = None
        if self.grid_tilt_check.isChecked():
            try:
                tilt_reference_points = self._grid_tilt_reference_points()
                from cns_control.utils import _fit_tilt_surface
                _fit_tilt_surface(tilt_reference_points, tilt_degree)
            except Exception as e:
                self.status.setText(f"Status: {e}")
                return
        corner_positions = None
        if self.grid_definition_combo.currentData() == "corners":
            try:
                top_left = (
                    float(self.grid_tl_x_input.text()),
                    float(self.grid_tl_y_input.text()),
                )
                bottom_right = (
                    float(self.grid_br_x_input.text()),
                    float(self.grid_br_y_input.text()),
                )
                corner_positions = (top_left, bottom_right)
            except ValueError:
                self.status.setText(
                    "Status: enter or capture both grid-corner X/Y positions"
                )
                return

        log = LogWindow(title="Stage grid log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: generating stage grid...")
        self.repaint()

        try:
            from cns_control.utils import grid_point_selections

            with _StdoutRedirector(log):
                self._prepare_for_selection()
                self.core.register_mda_engine(self.default_engine)
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                sources, autofocus_p, new_seq = grid_point_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    fov_x=fov_x, fov_y=fov_y,
                    x_range=x_range, y_range=y_range,
                    x_step=x_step, y_step=y_step,
                    repeats=repeats,
                    preview_channel=preview_channel,
                    use_placeholder=use_placeholder,
                    corner_positions=corner_positions,
                    x_count=x_count, y_count=y_count,
                    tilt_reference_points=tilt_reference_points,
                    tilt_degree=tilt_degree,
                    snake_axis=snake_axis,
                    autofocus_object=autofocus_object,
                )

            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
                "autofocus_object": autofocus_object,
                "tilt_reference_points": tilt_reference_points,
                "tilt_degree": tilt_degree,
                "snake_axis": snake_axis,
                "batch": False,
                # Every grid position samples the same point at the center of
                # the FOV, so the discarded galvo-prepositioning acquisition
                # is unnecessary.
                "pre_acq": False,
            }
            n_pos = len(autofocus_p)
            if preview_channel is not None:
                ready_detail = f"after {preview_channel} preview"
            elif use_placeholder:
                ready_detail = "using compact Raman grid"
            else:
                ready_detail = "after Raman pre-scan"
            z_detail = (
                f"degree {tilt_degree} surface"
                if tilt_reference_points is not None else "default"
            )
            log.append(f"\n--- grid ready {ready_detail} ---\n")
            self.status.setText(
                f"Status: grid ready {ready_detail} ({n_pos} positions, "
                f"{repeats} pts each at ({fov_x},{fov_y}), "
                f"{self.grid_scan_order_combo.currentText()}, "
                f"Z={z_detail}"
                ") -- then Run Raman MDA"
            )
        except Exception as e:
            log.append(f"\n--- stage grid failed: {e} ---\n")
            self.status.setText(f"Status: stage grid failed -- {e}")

    # -------- run raman MDA --------
    def run_raman_mda(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.collector is None or self.transformer is None:
            self.status.setText(
                "Status: collector/transformer missing -- reconnect"
            )
            return
        if self.selection_results is None:
            self.status.setText(
                "Status: no selection results -- "
                "run automated selection first"
            )
            return

        sources = self.selection_results["sources"]
        autofocus_p = self.selection_results["autofocus_p"]
        new_seq = self.selection_results["new_seq"]
        image_p = autofocus_p
        afp_text = self.mda_afp_input.text().strip()
        if afp_text and afp_text.lower() != "none":
            try:
                autofocus_p = np.array(
                    self._parse_int_list(afp_text, "Autofocus positions")
                )
            except ValueError as e:
                self.status.setText(f"Status: {e}")
                return
            n_pos = len(new_seq.stage_positions)
            bad = [p for p in autofocus_p if p < 0 or p >= n_pos]
            if bad:
                self.status.setText(
                    f"Status: autofocus positions out of range {bad} "
                    f"(sequence has {n_pos} positions)"
                )
                return
            print(f"[autofocus_p] manual override: {autofocus_p.tolist()}")
        
        imgp_text = self.mda_imgp_input.text().strip()
        if imgp_text and imgp_text.lower() != "none":
            try:
                image_p = np.array(
                    self._parse_int_list(imgp_text, "Imaging positions")
                )
            except ValueError as e:
                self.status.setText(f"Status: {e}")
                return
            n_pos = len(new_seq.stage_positions)
            bad = [p for p in image_p if p < 0 or p >= n_pos]
            if bad:
                self.status.setText(
                    f"Status: imaging positions out of range {bad} "
                    f"(sequence has {n_pos} positions)"
                )
                return
            print(f"[image_p] manual override: {image_p.tolist()}")

        af_choice = self.selection_results.get(
            "autofocus_object", self.sel_af_combo.currentText()
        )
        autofocus_enabled = af_choice not in ("None", "none", "", None)
        autofocus_object = af_choice if autofocus_enabled else "laser"
        segment_and_track = self.mda_seg_track_check.isChecked()
        batch = self.selection_results.get(
            "batch", self.sel_batch_combo.currentText() == "True"
        )
        pre_acq = self.selection_results.get("pre_acq", True)
        stage_centering_model = None
        if autofocus_enabled and autofocus_object == "laser":
            objective = self._ensure_current_vandermonde()
            if objective is None or self.vandermonde is None:
                self.status.setText(
                    "Status: laser autofocus requires a valid "
                    "pixel-to-stage Vandermonde model for the current objective"
                )
                return
            coefficients, degree = self.vandermonde
            stage_centering_model = (
                np.asarray(coefficients, dtype=float).copy(),
                int(degree),
            )

        try:
            z_relative = self._parse_float_list(
                self.mda_zrel_input.text(), "Z relative"
            )
        except ValueError as e:
            self.status.setText(f"Status: {e}")
            return

        acquisition_channels = []
        seen_channels = set()
        for entry in self.mda_channel_rows:
            if not entry["combo"].isEnabled():
                continue
            channel_name = entry["combo"].currentText()
            if not channel_name or channel_name == "(connect first)":
                continue
            if channel_name in seen_channels:
                self.status.setText(
                    f"Status: duplicate acquisition channel {channel_name!r}"
                )
                return
            seen_channels.add(channel_name)
            acquisition_channels.append(
                (
                    channel_name,
                    float(entry["exp"].value()),
                    float(entry["offset"].value()),
                )
            )
        if not acquisition_channels:
            self.status.setText(
                "Status: add at least one acquisition channel "
                "(add RM for Raman)"
            )
            return

        raman_enabled = "RM" in seen_channels
        if raman_enabled:
            try:
                raman_z_indices = _parse_raman_z_indices(
                    self.mda_rz_input.text()
                )
            except ValueError as e:
                self.status.setText(f"Status: {e}")
                return
            if not raman_z_indices:
                self.status.setText(
                    "Status: RM is selected but Raman z indices is off"
                )
                return
        else:
            raman_z_indices = []

        if not raman_enabled and not _raman_free_autofocus_allowed(
            autofocus_enabled, autofocus_object
        ):
            self.status.setText(
                "Status: unsupported autofocus mode for Raman-free MDA: "
                f"{autofocus_object}"
            )
            return

        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        if (
            raman_enabled
            and batch
            and self._make_point_transformer(sq_size, sq_n).multiplier < 2
        ):
            self.status.setText(
                "Status: batch mode needs a pattern with >= 2 points "
                "(increase N)"
            )
            return

        out_dir = self.mda_dir_input.text().strip() or "data/run"
        os.makedirs(out_dir, exist_ok=True)
        af_range = float(self.mda_af_range_input.value())
        search_pts = int(self.mda_search_pts_input.value())
        fine_search_range = float(self.mda_fine_range_input.value())
        fine_search_pts = int(self.mda_fine_pts_input.value())
        total_exp = float(self.mda_exp_input.value())
        loops = int(self.mda_loops_input.value())
        interval = float(self.mda_interval_input.value())
        refocus_every = int(self.mda_refocus_input.value())
        segment_channel = self.mda_seg_ch_combo.currentText() or "BF"
        seg_scale = float(self.mda_seg_scale_input.value())
        cellpose_model = self.mda_seg_model_combo.currentText() or "cyto2"
        segment_crop = self.mda_seg_crop_combo.currentText() == "True"
        tracking_config = self.mda_track_cfg_input.text().strip() or "particle_config.json"
        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        circle_center=(cx,cy)
        # print(circle_center)
        circle_radius = int(self.sel_r_input.value())
        # print(circle_radius)

        log = LogWindow(title="Raman MDA log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: starting Raman MDA...")
        self.repaint()

        try:
            import datetime as _dt
            from useq import Channel, TIntervalLoops, ZRelativePositions
            from raman_mda_engine import RamanAcquisitionWriter, RamanEngine
            from cns_control.utils import set_up_new_seq

            try:
                img_x = int(self.core.getImageWidth())
                img_y = int(self.core.getImageHeight())
            except Exception:
                img_x, img_y = self._get_image_xy()

            with _StdoutRedirector(log):
                engine = RamanEngine(
                    mmc=self.core,
                    spectra_collector=self.collector,
                    transformer=self.transformer,
                    batch=batch,
                    pre_acq=pre_acq,
                    autofocus=autofocus_enabled,
                    autofocus_p=autofocus_p,
                    image_p=image_p,
                    autofocus_object=autofocus_object,
                    segment_and_track=segment_and_track,
                    scale=seg_scale,
                    segment_channel=segment_channel,
                    cellpose_model=cellpose_model,
                    segment_crop=segment_crop,
                    tracking_config=tracking_config,
                    autofocus_search_range=af_range,
                    search_pts=search_pts,
                    fine_search_range=fine_search_range,
                    fine_search_pts=fine_search_pts,
                    refocus_every=refocus_every,
                    image_x=img_x,
                    image_y=img_y,
                    skip_imaging_for_same_pos=True,
                    config_file = self.mm_config,
                    circle_center=circle_center,
                    circle_radius=circle_radius,
                    stage_centering_model=stage_centering_model,
                )

                self.core.register_mda_engine(engine)
                self._connect_raman_visualization(engine)

                previous_writer = self.mda_writer
                if previous_writer is not None and not getattr(
                    previous_writer, "closed", True
                ):
                    previous_writer.disconnect()
                self.mda_writer = RamanAcquisitionWriter(
                    out_dir,
                    core=self.core,
                    wavenumbers=self.collector.get_wavenumbers(),
                    image_positions=image_p,
                    batch=batch,
                )
                engine.aiming_sources = sources

                if raman_enabled:
                    point_transformer = self._make_point_transformer(
                        sq_size, sq_n
                    )
                    if batch:
                        final_seq = set_up_new_seq(
                            self.main_window, point_transformer, engine,
                            seq=new_seq, total_exposure=total_exp,
                            batch=batch, z_plan="middle",
                        )
                    else:
                        final_seq = set_up_new_seq(
                            self.main_window, point_transformer, engine,
                            seq=new_seq,
                            total_exposure=(
                                total_exp * point_transformer.multiplier
                            ),
                            batch=batch, z_plan="middle",
                        )
                else:
                    metadata = dict(new_seq.metadata)
                    metadata.pop("raman", None)
                    final_seq = new_seq.replace(
                        metadata=metadata,
                    )

                time_interval = _dt.timedelta(seconds=interval)
                if final_seq.time_plan is None:
                    new_time_plan = TIntervalLoops(
                        loops=loops,
                        interval=time_interval,
                    )
                else:
                    new_time_plan = final_seq.time_plan.replace(
                        loops=loops,
                        interval=time_interval,
                    )
                new_z_plan = ZRelativePositions(relative=z_relative)

                template = (
                    final_seq.channels[0]
                    if final_seq.channels
                    else Channel(config="BF")
                )
                acquisition_channel_objs = tuple(
                    template.replace(
                        config=channel_name,
                        exposure=(
                            None if channel_name == "RM" else channel_exposure
                        ),
                    )
                    for (
                        channel_name,
                        channel_exposure,
                        _channel_offset,
                    ) in acquisition_channels
                )

                metadata = dict(final_seq.metadata)
                metadata["channel_z_offsets_um"] = {
                    channel_name: channel_offset
                    for (
                        channel_name,
                        _channel_exposure,
                        channel_offset,
                    ) in acquisition_channels
                }
                final_seq = final_seq.replace(
                    time_plan=new_time_plan,
                    z_plan=new_z_plan,
                    channels=acquisition_channel_objs,
                    metadata=metadata,
                )

                if raman_enabled and "raman" in final_seq.metadata:
                    final_seq.metadata["raman"]["z"] = raman_z_indices
                elif raman_enabled:
                    print(
                        "[warn] final_seq.metadata has no 'raman' key; "
                        "skipping raman['z']"
                    )

                print(
                    f"Starting MDA: {loops} loops, interval={interval}s, "
                    f"axis_order={final_seq.axis_order}, "
                    f"z_rel={z_relative}, raman_z={raman_z_indices}, "
                    f"autofocus={autofocus_enabled} ({af_choice}), "
                    f"segment_and_track={segment_and_track}, "
                    f"search_pts={search_pts}, fine_range={fine_search_range}, "
                    f"fine_pts={fine_search_pts}, refocus_every={refocus_every}, "
                    f"image=({img_x}x{img_y}), "
                    "channels="
                    f"{[(ch, offset) for ch, _, offset in acquisition_channels]}"
                )
                print(
                    f"[debug] engine._autofocus={engine._autofocus}, "
                    f"engine._segment_and_track={engine._segment_and_track}"
                )
                self._connect_mda_completion_events()
                self._raman_mda_pending = True
                self._raman_mda_canceled = False
                self._raman_mda_writer = self.mda_writer
                self.core.run_mda(final_seq)

            log.append("\n--- MDA started ---\n")
            mode = "Raman MDA" if raman_enabled else "Raman-free MDA"
            self.status.setText(f"Status: {mode} started OK")
        except Exception as e:
            if self.mda_writer is not None and not getattr(
                self.mda_writer, "closed", True
            ):
                self.mda_writer.close(status="failed")
            self._raman_mda_pending = False
            self._raman_mda_writer = None
            log.append(f"\n--- MDA failed: {e} ---\n")
            self.status.setText(f"Status: MDA failed -- {e}")

    def stop_raman_mda(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        try:
            # cancel() requests a clean stop; the MDA finishes its current
            # event and then exits. For an immediate halt, also stop sequence
            # acq.
            self.core.mda.cancel()
            try:
                self.core.stopSequenceAcquisition()
            except Exception:
                pass
            self.status.setText("Status: stop requested OK")
        except Exception as e:
            self.status.setText(f"Status: stop failed -- {e}")
