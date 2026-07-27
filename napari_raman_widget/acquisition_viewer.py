"""Qt viewer for large saved Raman acquisitions.

The viewer indexes filenames once and only loads the selected Raman spectrum
and image tiles. It never connects to microscope hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from qtpy.QtCore import Qt
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

        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        central = QWidget()
        layout = QVBoxLayout(central)

        intro = QLabel(
            "Index imaging and Raman folders separately. Only the requested "
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
            "Raman folder:",
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
        view_row.addWidget(QLabel("Raman index / p:"))
        self.raman_index = QSpinBox()
        self.raman_index.setRange(0, 2_147_483_647)
        view_row.addWidget(self.raman_index)
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
        self.show_btn = QPushButton("Show")
        self.show_btn.clicked.connect(self.show_selection)
        view_row.addWidget(self.show_btn)
        layout.addLayout(view_row)

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

        self._set_view_controls_enabled(False)
        self.model_path.editingFinished.connect(self.refresh_objectives)
        self.refresh_objectives(preferred=objective)
        self._infer_related_paths()

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
            self.raman_index,
            self.channel_combo,
            self.stitched_check,
            self.preview_size,
            self.show_btn,
        ):
            widget.setEnabled(enabled)

    def _infer_related_paths(self):
        imaging_text = self.imaging_path.text().strip()
        if not imaging_text:
            return
        imaging = Path(imaging_text)
        if not self.sequence_path.text().strip():
            sequence = imaging / "useq-sequence.json"
            if sequence.exists():
                self.sequence_path.setText(str(sequence))
        if not self.raman_path.text().strip():
            raman = imaging / "raman"
            if raman.is_dir():
                self.raman_path.setText(str(raman))

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
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Raman folder",
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
            raman_positions = sorted(
                {key.p for key in acquisition.raman_files}
            )
            if not raman_positions:
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
            self.raman_index.setRange(
                int(raman_positions[0]), int(raman_positions[-1])
            )
            self.raman_index.setValue(int(raman_positions[0]))
            self.acquisition = acquisition
            self.geometry = geometry
            self._set_view_controls_enabled(True)

            summary = acquisition.summary()
            elapsed = time.perf_counter() - started
            self.summary_label.setText(
                f"{summary['image_files']:,} images; "
                f"{summary['raman_files']:,} Raman positions; "
                f"{len(channels)} channels; indexed in {elapsed:.2f} s"
            )
            self.status_label.setText(
                "Indexed. Choose a Raman index and channel, then Show. "
                "Containing-tile view is fastest; stitching is optional."
            )
        except Exception as error:
            self.acquisition = None
            self.geometry = None
            self.summary_label.setText("Indexing failed")
            self.status_label.setText(f"Indexing failed: {error}")
        finally:
            QApplication.restoreOverrideCursor()
            self.index_btn.setEnabled(True)

    def show_selection(self):
        """Load and draw exactly one spectrum plus one tile or mosaic."""
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
            channel = self.channel_combo.currentData()
            spectrum_data = _mean_raman_spectrum(
                self.acquisition, raman_index
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
                f"Showing Raman p={raman_index} in {elapsed:.2f} s"
            )
        except Exception as error:
            self.status_label.setText(f"Could not show selection: {error}")
        finally:
            QApplication.restoreOverrideCursor()
            self.show_btn.setEnabled(True)
