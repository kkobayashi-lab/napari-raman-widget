"""Qt viewer for large saved Raman acquisitions.

The viewer indexes filenames once and only loads the selected Raman spectrum
and image tiles. It never connects to microscope hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .large_dataset_viewing import (
    AcquisitionIndex,
    _image_channel_label,
    _mean_raman_spectrum,
    _plot_raman_spectrum,
    find_image_for_raman_index,
    load_vandermonde_geometry,
    raman_stitched_preview,
)


def vandermonde_objectives(model_path) -> list[str]:
    """Return objective names without loading hardware or moving the turret."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    objectives = document.get("objectives")
    if isinstance(objectives, dict):
        return [str(name) for name in objectives]
    return []


class LargeAcquisitionViewerWindow(QMainWindow):
    """On-demand Raman spectrum and image viewer for large acquisitions."""

    def __init__(
        self,
        *,
        imaging_folder="",
        raman_folder="",
        sequence_file="",
        vandermonde_model="",
        objective=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Saved Raman acquisition viewer")
        self.resize(1300, 820)
        self.acquisition = None
        self.geometry = None
        self._auto_refresh_suspended = True

        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        central = QWidget()
        layout = QVBoxLayout(central)

        intro = QLabel(
            "Index a run folder (or legacy imaging/Raman sources). Only the requested "
            "spectrum and image tiles are read from disk."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        source_grid = QGridLayout()
        self.imaging_path = QLineEdit(str(imaging_folder or ""))
        self.raman_path = QLineEdit(str(raman_folder or ""))
        self.sequence_path = QLineEdit(str(sequence_file or ""))
        self.model_path = QLineEdit(str(vandermonde_model or ""))
        self.objective_combo = QComboBox()

        self._add_path_row(
            source_grid,
            0,
            "Imaging folder:",
            self.imaging_path,
            self._browse_imaging,
        )
        self._add_path_row(
            source_grid,
            1,
            "Raman source:",
            self.raman_path,
            self._browse_raman,
        )
        self._add_path_row(
            source_grid,
            2,
            "Sequence JSON:",
            self.sequence_path,
            self._browse_sequence,
        )
        self._add_path_row(
            source_grid,
            3,
            "Vandermonde model:",
            self.model_path,
            self._browse_model,
        )
        source_grid.addWidget(QLabel("Objective:"), 4, 0)
        source_grid.addWidget(self.objective_combo, 4, 1)
        layout.addLayout(source_grid)

        source_actions = QHBoxLayout()
        self.index_btn = QPushButton("Index acquisition")
        self.index_btn.clicked.connect(self.index_acquisition)
        source_actions.addWidget(self.index_btn)
        self.summary_label = QLabel("No acquisition indexed")
        self.summary_label.setWordWrap(True)
        source_actions.addWidget(self.summary_label, 1)
        layout.addLayout(source_actions)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Raman t:"))
        self.raman_t_combo = QComboBox()
        view_row.addWidget(self.raman_t_combo)
        view_row.addWidget(QLabel("Raman p/FOV:"))
        self.raman_index = QSpinBox()
        self.raman_index.setRange(0, 2_147_483_647)
        view_row.addWidget(self.raman_index)
        view_row.addWidget(QLabel("Raman z:"))
        self.raman_z_combo = QComboBox()
        view_row.addWidget(self.raman_z_combo)
        view_row.addWidget(QLabel("Cell index:"))
        self.cell_index = QSpinBox()
        self.cell_index.setRange(0, 0)
        view_row.addWidget(self.cell_index)
        view_row.addWidget(QLabel("Imaging channel:"))
        self.channel_combo = QComboBox()
        view_row.addWidget(self.channel_combo)
        self.stitched_check = QCheckBox("Show stitched image")
        self.stitched_check.setChecked(False)
        self.stitched_check.setToolTip(
            "Off is fastest. Enable to read and combine every image tile in "
            "the selected channel/Z plane."
        )
        view_row.addWidget(self.stitched_check)
        view_row.addWidget(QLabel("Preview edge:"))
        self.preview_size = QComboBox()
        for size in (1024, 2048, 4096):
            self.preview_size.addItem(str(size), size)
        self.preview_size.setCurrentText("2048")
        view_row.addWidget(self.preview_size)
        self.show_btn = QPushButton("Refresh")
        self.show_btn.setToolTip(
            "The viewer updates automatically; use this to refresh manually."
        )
        self.show_btn.clicked.connect(self.show_selection)
        view_row.addWidget(self.show_btn)
        layout.addLayout(view_row)
        self.raman_index.valueChanged.connect(self._update_raman_axes)
        self.raman_t_combo.currentIndexChanged.connect(
            self._update_raman_z_and_cells
        )
        self.raman_z_combo.currentIndexChanged.connect(
            self._update_cell_range
        )

        self.status_label = QLabel(
            "Choose folders and a Vandermonde objective, then index."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.figure = Figure(figsize=(12, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.spectrum_ax = self.figure.add_subplot(121)
        self.image_ax = self.figure.add_subplot(122)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setSingleShot(True)
        self._auto_refresh_timer.setInterval(100)
        self._auto_refresh_timer.timeout.connect(self.show_selection)
        self._connect_auto_refresh()

        self._set_view_controls_enabled(False)
        self.model_path.editingFinished.connect(self.refresh_objectives)
        self.refresh_objectives(preferred=objective)
        self._infer_related_paths()
        self._auto_refresh_suspended = False

    @staticmethod
    def _add_path_row(grid, row, label, line_edit, callback):
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(line_edit, row, 1)
        button = QPushButton("...")
        button.setFixedWidth(32)
        button.clicked.connect(callback)
        grid.addWidget(button, row, 2)

    def _set_view_controls_enabled(self, enabled):
        for widget in (
            self.raman_t_combo,
            self.raman_index,
            self.raman_z_combo,
            self.cell_index,
            self.channel_combo,
            self.stitched_check,
            self.preview_size,
            self.show_btn,
        ):
            widget.setEnabled(enabled)

    def _connect_auto_refresh(self):
        """Refresh the plots after any view selection changes."""
        for signal in (
            self.raman_t_combo.currentIndexChanged,
            self.raman_index.valueChanged,
            self.raman_z_combo.currentIndexChanged,
            self.cell_index.valueChanged,
            self.channel_combo.currentIndexChanged,
            self.stitched_check.toggled,
            self.preview_size.currentIndexChanged,
        ):
            signal.connect(self._schedule_auto_refresh)

    def _schedule_auto_refresh(self, *_args):
        """Coalesce rapid control changes into one spectrum/image load."""
        if (
            self._auto_refresh_suspended
            or self.acquisition is None
            or self.geometry is None
        ):
            return
        self._auto_refresh_timer.start()

    def _infer_related_paths(self):
        imaging_text = self.imaging_path.text().strip()
        if not imaging_text:
            return
        imaging = Path(imaging_text)
        run_folder = imaging.parent if imaging.name == "images" else imaging
        if not self.sequence_path.text().strip():
            sequence = run_folder / "useq-sequence.json"
            if sequence.exists():
                self.sequence_path.setText(str(sequence))
        if not self.raman_path.text().strip():
            h5_path = run_folder / "raman.h5"
            legacy_path = run_folder / "raman"
            if h5_path.is_file():
                self.raman_path.setText(str(h5_path))
            elif legacy_path.is_dir():
                self.raman_path.setText(str(legacy_path))

    def _browse_imaging(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select imaging folder",
            self.imaging_path.text().strip(),
        )
        if path:
            self.imaging_path.setText(path)
            self._infer_related_paths()

    def _browse_raman(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Raman HDF5",
            self.raman_path.text().strip(),
            "Raman HDF5 (*.h5);;All files (*)",
        )
        if not path:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select legacy Raman folder",
                self.raman_path.text().strip(),
            )
        if path:
            self.raman_path.setText(path)

    def _browse_sequence(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select useq sequence",
            self.sequence_path.text().strip(),
            "JSON (*.json)",
        )
        if path:
            self.sequence_path.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vandermonde model",
            self.model_path.text().strip(),
            "JSON (*.json)",
        )
        if path:
            self.model_path.setText(path)
            self.refresh_objectives()

    def refresh_objectives(self, preferred=None):
        current = (
            str(preferred)
            if preferred is not None
            else self.objective_combo.currentText()
        )
        self.objective_combo.clear()
        try:
            objectives = vandermonde_objectives(self.model_path.text().strip())
        except Exception as error:
            self.status_label.setText(
                f"Could not read Vandermonde objectives: {error}"
            )
            objectives = []
        if objectives:
            self.objective_combo.addItems(objectives)
            if current in objectives:
                self.objective_combo.setCurrentText(current)
        else:
            self.objective_combo.addItem("(single model)", None)

    def index_acquisition(self):
        """Index source filenames and populate lightweight view controls."""
        self._auto_refresh_timer.stop()
        self._auto_refresh_suspended = True
        self.index_btn.setEnabled(False)
        self._set_view_controls_enabled(False)
        self.status_label.setText("Indexing acquisition...")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        started = time.perf_counter()
        try:
            imaging = self.imaging_path.text().strip()
            raman = self.raman_path.text().strip()
            sequence = self.sequence_path.text().strip() or None
            model = self.model_path.text().strip()
            objective = self.objective_combo.currentData()
            if objective is None and self.objective_combo.count():
                text = self.objective_combo.currentText()
                objective = None if text == "(single model)" else text

            acquisition = AcquisitionIndex.build(
                imaging_folder=imaging,
                raman_folder=raman,
                sequence_file=sequence,
            )
            geometry = load_vandermonde_geometry(model, objective)
            raman_fovs = acquisition.raman_fov_source_positions
            if not raman_fovs:
                raise FileNotFoundError(
                    f"No Raman data files found in {acquisition.raman_folder}"
                )

            first_t = min(key.t for key in acquisition.image_files)
            channels = sorted(
                {
                    key.c
                    for key in acquisition.image_files
                    if key.t == first_t
                }
            )
            self.channel_combo.clear()
            for channel in channels:
                label = _image_channel_label(acquisition, channel)
                self.channel_combo.addItem(f"{label} (c={channel})", channel)
            self.acquisition = acquisition
            self.geometry = geometry
            self.raman_index.setRange(0, len(raman_fovs) - 1)
            self.raman_index.setValue(0)
            self._update_raman_axes()
            self._set_view_controls_enabled(True)

            summary = acquisition.summary()
            cell_count = sum(
                len(
                    acquisition.raman_cells(
                        index,
                        time_index=time_index,
                        z_index=z_index,
                    )
                )
                for index in range(len(raman_fovs))
                for time_index in acquisition.raman_times(index)
                for z_index in acquisition.raman_z_indices(
                    index, time_index
                )
            )
            elapsed = time.perf_counter() - started
            self.summary_label.setText(
                f"{summary['image_files']:,} images; "
                f"{len(raman_fovs):,} Raman FOVs; "
                f"{cell_count:,} cells; "
                f"{len(channels)} channels; indexed in {elapsed:.2f} s"
            )
            self.status_label.setText(
                "Indexed. The spectrum and image update automatically when "
                "Raman t, p/FOV, z, cell, channel, or preview options change. "
                "Exact image/Raman FOV matches are used first; center matching "
                "is the fallback."
            )
        except Exception as error:
            self.acquisition = None
            self.geometry = None
            self.summary_label.setText("Indexing failed")
            self.status_label.setText(f"Indexing failed: {error}")
        finally:
            QApplication.restoreOverrideCursor()
            self.index_btn.setEnabled(True)
            self._auto_refresh_suspended = False
        if self.acquisition is not None and self.geometry is not None:
            self._schedule_auto_refresh()

    @staticmethod
    def _set_combo_values(combo, values):
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for value in values:
            combo.addItem(str(value), int(value))
        if previous in values:
            combo.setCurrentIndex(values.index(previous))
        combo.blockSignals(False)

    def _selected_raman_t(self):
        value = self.raman_t_combo.currentData()
        return None if value is None else int(value)

    def _selected_raman_z(self):
        value = self.raman_z_combo.currentData()
        return None if value is None else int(value)

    def _update_raman_axes(self, *_args):
        if self.acquisition is None:
            self.raman_t_combo.clear()
            self.raman_z_combo.clear()
            self.cell_index.setRange(0, 0)
            return
        times = list(
            self.acquisition.raman_times(self.raman_index.value())
        )
        self._set_combo_values(self.raman_t_combo, times)
        self._update_raman_z_and_cells()

    def _update_raman_z_and_cells(self, *_args):
        if self.acquisition is None:
            self.raman_z_combo.clear()
            self.cell_index.setRange(0, 0)
            return
        time_index = self._selected_raman_t()
        z_indices = (
            []
            if time_index is None
            else list(
                self.acquisition.raman_z_indices(
                    self.raman_index.value(), time_index
                )
            )
        )
        self._set_combo_values(self.raman_z_combo, z_indices)
        self._update_cell_range()

    def _update_cell_range(self, *_args):
        if self.acquisition is None:
            self.cell_index.setRange(0, 0)
            return
        cells = self.acquisition.raman_cells(
            self.raman_index.value(),
            time_index=self._selected_raman_t(),
            z_index=self._selected_raman_z(),
        )
        self.cell_index.setRange(0, max(0, len(cells) - 1))
        if self.cell_index.value() >= len(cells):
            self.cell_index.setValue(max(0, len(cells) - 1))

    def show_selection(self):
        """Load and draw exactly one spectrum plus one tile or mosaic."""
        self._auto_refresh_timer.stop()
        if self.acquisition is None or self.geometry is None:
            self.status_label.setText("Index an acquisition first.")
            return
        self.show_btn.setEnabled(False)
        self.status_label.setText("Loading selection...")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        started = time.perf_counter()
        try:
            raman_index = self.raman_index.value()
            time_index = self._selected_raman_t()
            z_index = self._selected_raman_z()
            cell_index = self.cell_index.value()
            channel = self.channel_combo.currentData()
            spectrum_data = _mean_raman_spectrum(
                self.acquisition,
                raman_index,
                cell_index,
                time_index=time_index,
                z_index=z_index,
            )

            self.spectrum_ax.clear()
            self.image_ax.clear()
            _plot_raman_spectrum(self.spectrum_ax, *spectrum_data)

            if self.stitched_check.isChecked():
                mosaic, marker, metadata, match = raman_stitched_preview(
                    self.acquisition,
                    raman_index,
                    self.geometry,
                    imaging_channel=channel,
                    max_side=int(self.preview_size.currentData()),
                    cell_index=cell_index,
                    time_index=time_index,
                    z_index=z_index,
                )
                if mosaic is not None:
                    self.image_ax.imshow(mosaic, cmap="gray")
                    self.image_ax.scatter(
                        [marker[0]],
                        [marker[1]],
                        s=140,
                        facecolor="none",
                        edgecolor="red",
                        linewidth=2,
                    )
                    label = _image_channel_label(
                        self.acquisition, match.image_key.c
                    )
                    self.image_ax.set_title(
                        f"Stitched {label}; {metadata['tiles']} tiles; "
                        f"stride {metadata['downsample_stride']}"
                    )
            else:
                match = find_image_for_raman_index(
                    self.acquisition,
                    raman_index,
                    self.geometry,
                    imaging_channel=channel,
                    cell_index=cell_index,
                    time_index=time_index,
                    z_index=z_index,
                )
                if match is not None:
                    image = self.acquisition.load_image(match.image_key)
                    self.image_ax.imshow(image, cmap="gray")
                    self.image_ax.scatter(
                        [match.image_pixel_xy[0]],
                        [match.image_pixel_xy[1]],
                        s=140,
                        facecolor="none",
                        edgecolor="red",
                        linewidth=2,
                    )
                    self.image_ax.set_title(
                        f"Containing tile t/p/c/z={tuple(match.image_key)}; "
                        f"{match.center_distance_pixels:.1f} px from center"
                    )

            if match is None:
                self.image_ax.set_title(
                    "No image contains this Raman position"
                )
            self.image_ax.set_axis_off()
            self.figure.tight_layout()
            self.canvas.draw_idle()
            elapsed = time.perf_counter() - started
            self.status_label.setText(
                f"Showing Raman t={time_index}, p/FOV={raman_index}, "
                f"z={z_index}, cell={cell_index} in {elapsed:.2f} s"
            )
        except Exception as error:
            self.status_label.setText(f"Could not show selection: {error}")
        finally:
            QApplication.restoreOverrideCursor()
            self.show_btn.setEnabled(True)
